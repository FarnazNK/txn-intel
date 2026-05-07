"""Transaction anomaly detection.

Two-stage approach:
  1. IsolationForest on raw transaction features (unsupervised) — produces an
     anomaly score per transaction.
  2. LightGBM head trained on (features + IF_score) against the injected
     is_anomaly label. The head learns to combine IF score with engineered
     features to improve precision.

In production the IF score alone would be the fallback when no labels exist;
the head improves things once any labels accumulate.
"""
from __future__ import annotations

from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sqlalchemy import text

from app.core.logging import get_logger
from app.db.session import engine
from app.ml.registry import ModelMetadata, ModelRegistry

log = get_logger(__name__)
MODEL_NAME = "anomaly"

# Features computed at transaction time
TXN_FEATURES = [
    "amount",
    "amount_log",
    "quantity",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "amount_zscore_customer",
    "amount_zscore_merchant",
    "txn_count_customer_24h",
    "channel_web", "channel_mobile", "channel_in_person", "channel_api",
]


def fetch_training_data(limit: int = 500_000) -> pd.DataFrame:
    """Fetch a sample of transactions with engineered features.

    Computes per-customer and per-merchant amount stats via window functions.
    """
    sql = text("""
        WITH cust_stats AS (
            SELECT customer_id,
                   AVG(amount)::float AS cust_amt_mean,
                   STDDEV_POP(amount)::float AS cust_amt_std
            FROM transactions GROUP BY customer_id
        ),
        merch_stats AS (
            SELECT merchant_id,
                   AVG(amount)::float AS merch_amt_mean,
                   STDDEV_POP(amount)::float AS merch_amt_std
            FROM transactions GROUP BY merchant_id
        ),
        sample AS (
            SELECT * FROM transactions
            ORDER BY random()
            LIMIT :limit
        )
        SELECT
            s.id, s.customer_id, s.merchant_id,
            s.occurred_at, s.amount, s.quantity, s.channel,
            s.is_anomaly,
            cs.cust_amt_mean, cs.cust_amt_std,
            ms.merch_amt_mean, ms.merch_amt_std
        FROM sample s
        LEFT JOIN cust_stats  cs ON cs.customer_id = s.customer_id
        LEFT JOIN merch_stats ms ON ms.merchant_id = s.merchant_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"limit": limit})
    log.info("fetched %d transactions, anomaly rate=%.4f",
             len(df), df["is_anomaly"].mean())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = df["amount"].astype(float)
    df["amount_log"] = np.log1p(df["amount"])
    df["quantity"] = df["quantity"].astype(int)
    df["hour_of_day"] = pd.to_datetime(df["occurred_at"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["occurred_at"]).dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Z-scores; clip stds to avoid div-by-zero
    df["cust_amt_std"] = df["cust_amt_std"].fillna(1.0).clip(lower=1e-3)
    df["merch_amt_std"] = df["merch_amt_std"].fillna(1.0).clip(lower=1e-3)
    df["amount_zscore_customer"] = (df["amount"] - df["cust_amt_mean"]) / df["cust_amt_std"]
    df["amount_zscore_merchant"] = (df["amount"] - df["merch_amt_mean"]) / df["merch_amt_std"]

    # Per-customer 24h count via subquery would be expensive; approximate via
    # a join we precomputed elsewhere. Here, default to 1 (placeholder kept for
    # schema parity; production would compute this in a feature pipeline job).
    df["txn_count_customer_24h"] = 1

    for ch in ("web", "mobile", "in_person", "api"):
        df[f"channel_{ch}"] = (df["channel"] == ch).astype(int)
    return df


def train(sample_limit: int = 500_000) -> str:
    df = fetch_training_data(sample_limit)
    df = engineer_features(df)

    X = df[TXN_FEATURES].astype(float).values
    y = df["is_anomaly"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    log.info("fitting IsolationForest...")
    iso = IsolationForest(
        n_estimators=200, contamination=float(max(0.005, y_tr.mean())),
        random_state=42, n_jobs=-1,
    )
    iso.fit(X_tr)
    # decision_function: higher = more normal. Flip so higher = more anomalous.
    iso_score_tr = -iso.decision_function(X_tr)
    iso_score_te = -iso.decision_function(X_te)

    iso_only_metrics = {
        "roc_auc": float(roc_auc_score(y_te, iso_score_te)),
        "pr_auc": float(average_precision_score(y_te, iso_score_te)),
    }
    log.info("IF-only metrics: %s", iso_only_metrics)

    # Stage 2: LightGBM with IF score as additional feature
    X_tr_aug = np.hstack([X_tr, iso_score_tr.reshape(-1, 1)])
    X_te_aug = np.hstack([X_te, iso_score_te.reshape(-1, 1)])

    log.info("fitting LightGBM head...")
    pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    head = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1,
    )
    head.fit(X_tr_aug, y_tr)
    proba = head.predict_proba(X_te_aug)[:, 1]
    head_metrics = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "iso_only_pr_auc": iso_only_metrics["pr_auc"],
    }
    log.info("head metrics: %s", head_metrics)

    bundle = {
        "isolation_forest": iso,
        "head": head,
        "feature_order": TXN_FEATURES,
    }

    registry = ModelRegistry()
    version = ModelRegistry.new_version()
    meta = ModelMetadata(
        name=MODEL_NAME,
        version=version,
        trained_at=datetime.utcnow().isoformat(),
        feature_schema=TXN_FEATURES,
        metrics=head_metrics,
        training_window={"sampled_n": str(len(df))},
        n_train=int(len(X_tr)),
        n_eval=int(len(X_te)),
        notes="two-stage IF + LightGBM head",
    )
    registry.save(MODEL_NAME, bundle, meta)
    current = registry.champion(MODEL_NAME)
    if current is None:
        registry.promote(MODEL_NAME, version)
    else:
        winner = registry.compare(MODEL_NAME, current, version, "pr_auc", True)
        if winner == version:
            registry.promote(MODEL_NAME, version)
    return version


def score(bundle: dict, features: np.ndarray) -> np.ndarray:
    """Score one or more transactions with a loaded bundle."""
    iso_score = -bundle["isolation_forest"].decision_function(features).reshape(-1, 1)
    augmented = np.hstack([features, iso_score])
    return bundle["head"].predict_proba(augmented)[:, 1]


if __name__ == "__main__":
    from app.core.logging import configure_logging
    configure_logging()
    train()
