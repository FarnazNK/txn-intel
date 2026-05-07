"""Drift monitoring.

Two metrics:
  - PSI (Population Stability Index) per feature, comparing live distribution
    against the training reference.
  - KS (Kolmogorov-Smirnov) statistic on model output scores.

Both are stored in drift_metrics. A simple threshold (PSI > 0.2, KS > 0.1)
flags the feature/model for retraining.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)

PSI_BINS = 10
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.1


def psi(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population Stability Index between two samples."""
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    # Build bin edges from reference quantiles
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_hist / max(1, ref_hist.sum()), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(1, cur_hist.sum()), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def fetch_reference_features(as_of_date: datetime) -> pd.DataFrame:
    from sqlalchemy import text
    from app.db.session import engine
    sql = text("SELECT * FROM feat_customer_daily WHERE as_of_date = :d")
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"d": as_of_date.date()})


def fetch_current_predictions(model_name: str, days: int = 7) -> pd.DataFrame:
    from sqlalchemy import text
    from app.db.session import engine
    cutoff = datetime.utcnow() - timedelta(days=days)
    sql = text("""
        SELECT score, features
        FROM prediction_log
        WHERE model_name = :n AND created_at >= :c
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"n": model_name, "c": cutoff})


def write_drift(model_name: str, feature: str, metric: str, value: float) -> None:
    from sqlalchemy import text
    from app.db.session import engine
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO drift_metrics (model_name, feature, metric, value, computed_at)
                VALUES (:n, :f, :m, :v, :t)
            """),
            {"n": model_name, "f": feature, "m": metric, "v": value, "t": datetime.utcnow()},
        )


def run_for_churn(reference_date: datetime) -> dict[str, Any]:
    """Compute PSI per feature and KS on score distribution for churn model."""
    from app.ml.features.customer import CUSTOMER_FEATURES
    reference = fetch_reference_features(reference_date)
    current_pred = fetch_current_predictions("churn", days=7)

    if reference.empty or current_pred.empty:
        log.warning("not enough data: reference=%d current=%d",
                    len(reference), len(current_pred))
        return {"status": "insufficient_data"}

    # Reconstruct current feature distributions from logged features
    feat_records = pd.DataFrame([f for f in current_pred["features"]])

    flags = []
    for feature in CUSTOMER_FEATURES:
        if feature not in reference.columns or feature not in feat_records.columns:
            continue
        ref_vals = reference[feature].astype(float).values
        cur_vals = feat_records[feature].astype(float).values
        score = psi(ref_vals, cur_vals)
        write_drift("churn", feature, "psi", score)
        if score > PSI_THRESHOLD:
            flags.append({"feature": feature, "psi": score})
            log.warning("PSI flag: %s = %.3f", feature, score)

    # KS on output score: compare current scores against a reference batch.
    # In practice we'd score the reference set with the champion model offline
    # and store that as the KS baseline. Here we compare against uniform noise
    # for a placeholder — replace with real reference scores in production.
    rng = np.random.default_rng(0)
    reference_scores = rng.beta(2, 8, size=10_000)  # placeholder: typical churn proba
    ks_stat, _ = stats.ks_2samp(reference_scores, current_pred["score"].values)
    write_drift("churn", "_score", "ks", float(ks_stat))
    if ks_stat > KS_THRESHOLD:
        log.warning("KS flag on churn scores: %.3f", ks_stat)

    return {
        "psi_flags": flags,
        "ks_score_stat": float(ks_stat),
        "needs_retrain": len(flags) > 0 or ks_stat > KS_THRESHOLD,
    }


if __name__ == "__main__":
    from sqlalchemy import text
    from app.db.session import engine
    configure_logging()
    # Use the most recent feature snapshot as the reference for this demo.
    with engine.connect() as conn:
        ref_row = conn.execute(
            text("SELECT MAX(as_of_date) AS d FROM feat_customer_daily")
        ).first()
    if ref_row and ref_row.d:
        report = run_for_churn(datetime.combine(ref_row.d, datetime.min.time()))
        log.info("drift report: %s", report)
    else:
        log.warning("no reference snapshot found")
