"""Transaction anomaly detection — TensorFlow / Keras.

Two-stage deep approach (parallels the old IF + LGBM design):

  Stage 1 — Autoencoder (unsupervised).
      A small symmetric autoencoder trained on the *normal* slice of the data
      only. At score time, the per-row reconstruction MSE becomes an anomaly
      score: high reconstruction error == unfamiliar pattern.

  Stage 2 — Supervised dense head.
      A Keras MLP trained on (features + AE reconstruction error) against the
      injected ``is_anomaly`` label. The head learns to combine reconstruction
      error with engineered features. We use class weights so the rare
      positive class is up-weighted.

Both networks are saved in native Keras format (``.keras``). The supporting
StandardScaler and feature order are saved alongside via joblib.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Keep TF quiet by default — the noise hurts log readability in prod.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core.logging import get_logger  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.ml.registry import ModelMetadata, ModelRegistry  # noqa: E402

log = get_logger(__name__)
MODEL_NAME = "anomaly"

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


# ── Data ─────────────────────────────────────────────────────────────────────
def fetch_training_data(limit: int = 500_000) -> pd.DataFrame:
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

    df["cust_amt_std"] = df["cust_amt_std"].fillna(1.0).clip(lower=1e-3)
    df["merch_amt_std"] = df["merch_amt_std"].fillna(1.0).clip(lower=1e-3)
    df["amount_zscore_customer"] = (df["amount"] - df["cust_amt_mean"]) / df["cust_amt_std"]
    df["amount_zscore_merchant"] = (df["amount"] - df["merch_amt_mean"]) / df["merch_amt_std"]

    df["txn_count_customer_24h"] = 1

    for ch in ("web", "mobile", "in_person", "api"):
        df[f"channel_{ch}"] = (df["channel"] == ch).astype(int)
    return df


# ── Models ───────────────────────────────────────────────────────────────────
def build_autoencoder(in_dim: int, latent: int = 4) -> keras.Model:
    """Symmetric AE: in_dim -> 16 -> 8 -> latent -> 8 -> 16 -> in_dim."""
    inputs = keras.Input(shape=(in_dim,), name="ae_in")
    x = keras.layers.Dense(16, activation="relu")(inputs)
    x = keras.layers.Dense(8, activation="relu")(x)
    z = keras.layers.Dense(latent, activation="relu", name="latent")(x)
    x = keras.layers.Dense(8, activation="relu")(z)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(in_dim, activation="linear", name="ae_out")(x)
    ae = keras.Model(inputs, outputs, name="anomaly_autoencoder")
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae


def build_head(in_dim: int) -> keras.Model:
    """Supervised classifier on (features + ae_recon_error)."""
    inputs = keras.Input(shape=(in_dim,), name="head_in")
    x = keras.layers.Dense(64, activation="relu")(inputs)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="head_out")(x)
    head = keras.Model(inputs, outputs, name="anomaly_head")
    head.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return head


def _recon_error(ae: keras.Model, X: np.ndarray) -> np.ndarray:
    recon = ae.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


# ── Persistence ──────────────────────────────────────────────────────────────
def _save_artifacts(bundle: dict, version_dir: Path) -> None:
    bundle["autoencoder"].save(version_dir / "autoencoder.keras")
    bundle["head"].save(version_dir / "head.keras")
    side = {
        "scaler": bundle["scaler"],
        "feature_order": bundle["feature_order"],
        "ae_latent": bundle["ae_latent"],
    }
    joblib.dump(side, version_dir / "bundle.joblib")


def _load_artifacts(version_dir: Path) -> dict:
    ae = keras.models.load_model(version_dir / "autoencoder.keras", compile=False)
    head = keras.models.load_model(version_dir / "head.keras", compile=False)
    side = joblib.load(version_dir / "bundle.joblib")
    return {
        "autoencoder": ae,
        "head": head,
        "scaler": side["scaler"],
        "feature_order": side["feature_order"],
        "ae_latent": side["ae_latent"],
    }


def load_champion(registry: ModelRegistry | None = None) -> tuple[dict, ModelMetadata]:
    registry = registry or ModelRegistry()
    return registry.load(MODEL_NAME, loader=_load_artifacts)


# ── Public scoring API (used by inference service) ───────────────────────────
def score(bundle: dict, features: np.ndarray) -> np.ndarray:
    """Score one or more transactions. ``features`` is shaped (N, in_dim)
    in the order of ``bundle['feature_order']`` (unscaled).
    """
    X = bundle["scaler"].transform(features)
    err = _recon_error(bundle["autoencoder"], X).reshape(-1, 1)
    X_aug = np.hstack([X, err])
    proba = bundle["head"].predict(X_aug, verbose=0).reshape(-1)
    return proba


# ── Entrypoint ───────────────────────────────────────────────────────────────
def train(sample_limit: int = 500_000) -> str:
    df = fetch_training_data(sample_limit)
    df = engineer_features(df)

    X = df[TXN_FEATURES].astype(float).values
    y = df["is_anomaly"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    in_dim = X_tr_s.shape[1]

    # ── Stage 1: AE on NORMAL training rows only ─────────────────────────────
    normal_mask = y_tr == 0
    X_tr_normal = X_tr_s[normal_mask]
    log.info("training autoencoder on %d normal rows (in_dim=%d)",
             len(X_tr_normal), in_dim)
    ae = build_autoencoder(in_dim=in_dim, latent=4)
    ae.fit(
        X_tr_normal, X_tr_normal,
        epochs=20, batch_size=512, shuffle=True, verbose=0,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True,
        )],
    )

    # AE-only metrics (sanity check vs old IF-only baseline)
    err_te = _recon_error(ae, X_te_s)
    ae_only_metrics = {
        "roc_auc": float(roc_auc_score(y_te, err_te)),
        "pr_auc": float(average_precision_score(y_te, err_te)),
    }
    log.info("AE-only metrics: %s", ae_only_metrics)

    # ── Stage 2: Supervised head ─────────────────────────────────────────────
    err_tr = _recon_error(ae, X_tr_s).reshape(-1, 1)
    err_te = err_te.reshape(-1, 1)
    X_tr_aug = np.hstack([X_tr_s, err_tr])
    X_te_aug = np.hstack([X_te_s, err_te])

    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    class_weight = {0: 1.0, 1: max(neg, 1.0) / max(pos, 1.0)}
    log.info("training head with class_weight=%s", class_weight)

    head = build_head(in_dim=X_tr_aug.shape[1])
    head.fit(
        X_tr_aug, y_tr,
        epochs=30, batch_size=512, shuffle=True, verbose=0,
        validation_split=0.1, class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=5, restore_best_weights=True,
        )],
    )

    proba = head.predict(X_te_aug, verbose=0).reshape(-1)
    head_metrics = {
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
        "ae_only_pr_auc": ae_only_metrics["pr_auc"],
    }
    log.info("head metrics: %s", head_metrics)

    bundle = {
        "autoencoder": ae,
        "head": head,
        "scaler": scaler,
        "feature_order": TXN_FEATURES,
        "ae_latent": 4,
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
        n_train=int(len(X_tr_s)),
        n_eval=int(len(X_te_s)),
        notes="keras autoencoder + dense head",
        extra={"framework": "tensorflow"},
    )
    registry.save(MODEL_NAME, bundle, meta, saver=_save_artifacts)
    current = registry.champion(MODEL_NAME)
    if current is None:
        registry.promote(MODEL_NAME, version)
    else:
        winner = registry.compare(MODEL_NAME, current, version, "pr_auc", True)
        if winner == version:
            registry.promote(MODEL_NAME, version)
    return version


if __name__ == "__main__":
    from app.core.logging import configure_logging
    configure_logging()
    train()
