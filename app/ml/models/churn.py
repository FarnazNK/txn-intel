"""Churn prediction.

Label definition: a customer is "churned at as_of_date" if they had >=1 txn in
the 90 days BEFORE as_of_date but ZERO txns in the 60 days AFTER as_of_date.
This gives a forward-looking, behaviorally grounded label.

Features: from feat_customer_daily (point-in-time as of label date).
Models: LogisticRegression baseline + GradientBoostingClassifier champion.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from app.core.logging import get_logger
from app.db.session import engine
from app.ml.features.customer import CUSTOMER_FEATURES, materialize_customer_features
from app.ml.registry import ModelMetadata, ModelRegistry

log = get_logger(__name__)
MODEL_NAME = "churn"


def build_labels(label_dates: list[date]) -> pd.DataFrame:
    """Build churn labels by joining transaction activity around each label date.

    For each (customer, label_date), label = 1 if active in [-90, 0) and inactive in [0, +60).
    """
    sql = text("""
        WITH params AS (SELECT :label_date::date AS d),
        active_before AS (
            SELECT DISTINCT customer_id
            FROM transactions, params
            WHERE occurred_at >= params.d - INTERVAL '90 days'
              AND occurred_at <  params.d
              AND status = 'completed'
        ),
        active_after AS (
            SELECT DISTINCT customer_id
            FROM transactions, params
            WHERE occurred_at >= params.d
              AND occurred_at <  params.d + INTERVAL '60 days'
              AND status = 'completed'
        )
        SELECT ab.customer_id,
               (SELECT d FROM params) AS as_of_date,
               CASE WHEN aa.customer_id IS NULL THEN 1 ELSE 0 END AS churned
        FROM active_before ab
        LEFT JOIN active_after aa ON aa.customer_id = ab.customer_id
    """)
    parts = []
    with engine.connect() as conn:
        for d in label_dates:
            df = pd.read_sql(sql, conn, params={"label_date": d})
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fetch_features_for_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Pull features from feat_customer_daily for each (customer, as_of_date)."""
    if labels.empty:
        return labels
    sql = text("""
        SELECT f.* FROM feat_customer_daily f
        JOIN (SELECT UNNEST(:cids) AS cid, UNNEST(:dates::date[]) AS d) k
          ON f.customer_id = k.cid AND f.as_of_date = k.d
    """)
    with engine.connect() as conn:
        feats = pd.read_sql(
            sql, conn,
            params={
                "cids": labels["customer_id"].tolist(),
                "dates": [d.isoformat() if hasattr(d, "isoformat") else d
                          for d in labels["as_of_date"]],
            },
        )
    return labels.merge(feats, on=["customer_id", "as_of_date"], how="inner")


def build_pipelines() -> dict[str, Pipeline]:
    pre = ColumnTransformer([("num", StandardScaler(), CUSTOMER_FEATURES)])
    return {
        "logreg": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=200, class_weight="balanced", C=1.0)),
        ]),
        "gbm": Pipeline([
            ("pre", pre),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ]),
    }


def evaluate(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    # Pick threshold that maximizes F1 on PR curve
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1s = 2 * p * r / np.clip(p + r, 1e-9, None)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(f1s) > 1 else 0
    threshold = float(t[best_idx]) if len(t) else 0.5
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "f1_at_best": float(f1_score(y_true, y_pred)),
        "best_threshold": threshold,
        "positive_rate": float(y_true.mean()),
    }


def train(label_dates: list[date] | None = None,
          backfill_features: bool = True) -> str:
    """Train churn models and register the better one as champion.

    label_dates: list of as-of dates to label on. We need a 60-day forward window
                 after each, so don't include dates within 60 days of "now".
    """
    if label_dates is None:
        today = date.today()
        # Use 4 monthly snapshots ending 75 days ago
        label_dates = [today - timedelta(days=75 + 30 * i) for i in range(4)]

    if backfill_features:
        log.info("materializing features for label dates: %s", label_dates)
        for d in label_dates:
            materialize_customer_features(d)

    log.info("building labels...")
    labels = build_labels(label_dates)
    log.info("got %d labeled rows, positive rate=%.4f",
             len(labels), labels["churned"].mean() if len(labels) else 0)

    df = fetch_features_for_labels(labels)
    if df.empty:
        raise RuntimeError("no joined feature/label rows")

    X = df[CUSTOMER_FEATURES].values
    y = df["churned"].values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Wrap as DataFrame for ColumnTransformer
    X_tr_df = pd.DataFrame(X_tr, columns=CUSTOMER_FEATURES)
    X_te_df = pd.DataFrame(X_te, columns=CUSTOMER_FEATURES)

    pipelines = build_pipelines()
    results: dict[str, tuple[Pipeline, dict[str, float]]] = {}
    for name, pipe in pipelines.items():
        log.info("fitting %s...", name)
        pipe.fit(X_tr_df, y_tr)
        proba = pipe.predict_proba(X_te_df)[:, 1]
        metrics = evaluate(y_te, proba)
        log.info("  %s metrics: %s", name, {k: round(v, 4) for k, v in metrics.items()})
        results[name] = (pipe, metrics)

    # Pick winner by PR-AUC
    winner_name = max(results, key=lambda k: results[k][1]["pr_auc"])
    winner_model, winner_metrics = results[winner_name]
    log.info("winner: %s (pr_auc=%.4f)", winner_name, winner_metrics["pr_auc"])

    registry = ModelRegistry()
    version = ModelRegistry.new_version()
    meta = ModelMetadata(
        name=MODEL_NAME,
        version=version,
        trained_at=pd.Timestamp.utcnow().isoformat(),
        feature_schema=CUSTOMER_FEATURES,
        metrics=winner_metrics,
        training_window={
            "label_dates": ",".join(d.isoformat() for d in label_dates),
        },
        n_train=int(len(X_tr)),
        n_eval=int(len(X_te)),
        notes=f"winner={winner_name}",
        extra={"all_results": {k: v[1] for k, v in results.items()}},
    )
    registry.save(MODEL_NAME, winner_model, meta)

    # Champion/challenger logic: promote if no champion or if better
    current = registry.champion(MODEL_NAME)
    if current is None:
        registry.promote(MODEL_NAME, version)
    else:
        winner_version = registry.compare(
            MODEL_NAME, current, version, "pr_auc", higher_is_better=True,
        )
        if winner_version == version:
            registry.promote(MODEL_NAME, version)
            log.info("new model promoted to champion")
        else:
            log.info("kept existing champion %s", current)
    return version


if __name__ == "__main__":
    from app.core.logging import configure_logging
    configure_logging()
    train()
