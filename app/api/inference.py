"""Inference service: loads models once at startup, serves predictions."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.core.logging import get_logger
from app.db.session import engine
from app.ml.features.customer import (
    CUSTOMER_FEATURES,
    get_online_customer_features,
    vector_for_model,
)
from app.ml.models import anomaly as anomaly_mod
from app.ml.models import recommend as recommend_mod
from app.ml.registry import ModelRegistry

log = get_logger(__name__)

# Risk band thresholds for churn — calibrated from training metrics
RISK_BANDS = [(0.0, 0.20, "low"), (0.20, 0.50, "medium"),
              (0.50, 0.80, "high"), (0.80, 1.01, "critical")]


class InferenceService:
    def __init__(self) -> None:
        self.registry = ModelRegistry()
        self.churn_model: Any = None
        self.churn_meta: Any = None
        self.anomaly_bundle: Any = None
        self.anomaly_meta: Any = None
        self.recommend_bundle: Any = None
        self.recommend_meta: Any = None

    def load(self) -> None:
        for name, attr in [("churn", "churn"), ("anomaly", "anomaly"), ("recommend", "recommend")]:
            try:
                model, meta = self.registry.load(name)
                if name == "churn":
                    self.churn_model, self.churn_meta = model, meta
                elif name == "anomaly":
                    self.anomaly_bundle, self.anomaly_meta = model, meta
                elif name == "recommend":
                    self.recommend_bundle, self.recommend_meta = model, meta
                log.info("loaded %s %s", name, meta.version)
            except FileNotFoundError:
                log.warning("no champion for %s, endpoint will return 503", name)

    def model_versions(self) -> dict[str, str | None]:
        return {
            "churn": self.churn_meta.version if self.churn_meta else None,
            "anomaly": self.anomaly_meta.version if self.anomaly_meta else None,
            "recommend": self.recommend_meta.version if self.recommend_meta else None,
        }

    # ── Churn ────────────────────────────────────────────────────────────────
    def predict_churn(self, customer_id: int) -> dict[str, Any]:
        if self.churn_model is None:
            raise RuntimeError("churn model not loaded")
        features = get_online_customer_features(customer_id)
        if features is None:
            raise ValueError(f"no features available for customer {customer_id}")
        # Prepare model input as DataFrame to match training pipeline
        X = pd.DataFrame([{c: features[c] for c in CUSTOMER_FEATURES}])
        proba = float(self.churn_model.predict_proba(X)[0, 1])
        band = next(name for lo, hi, name in RISK_BANDS if lo <= proba < hi)
        self._log_prediction("churn", self.churn_meta.version, customer_id, proba,
                             {c: features[c] for c in CUSTOMER_FEATURES})
        return {"probability": proba, "risk_band": band, "features": features,
                "version": self.churn_meta.version}

    # ── Anomaly ──────────────────────────────────────────────────────────────
    def predict_anomaly(self, txn_features: dict[str, float]) -> dict[str, Any]:
        if self.anomaly_bundle is None:
            raise RuntimeError("anomaly model not loaded")
        order = self.anomaly_bundle["feature_order"]
        missing = [k for k in order if k not in txn_features]
        if missing:
            raise ValueError(f"missing features: {missing}")
        X = np.array([[float(txn_features[k]) for k in order]])
        score = float(anomaly_mod.score(self.anomaly_bundle, X)[0])
        threshold = 0.5
        self._log_prediction("anomaly", self.anomaly_meta.version, 0, score, txn_features)
        return {"score": score, "is_anomalous": score >= threshold,
                "threshold": threshold, "version": self.anomaly_meta.version}

    # ── Recommendations ──────────────────────────────────────────────────────
    def recommend(self, customer_id: int, n: int = 10) -> dict[str, Any]:
        if self.recommend_bundle is None:
            raise RuntimeError("recommend model not loaded")
        items = recommend_mod.recommend(self.recommend_bundle, customer_id, n)
        return {"items": items, "version": self.recommend_meta.version}

    # ── Logging ──────────────────────────────────────────────────────────────
    def _log_prediction(self, name: str, version: str, entity_id: int,
                        score: float, features: dict[str, Any]) -> None:
        # JSON-safe features
        safe = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in features.items()}
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO prediction_log (model_name, model_version, entity_id, score, features, created_at)
                    VALUES (:n, :v, :e, :s, CAST(:f AS jsonb), :t)
                """), {
                    "n": name, "v": version, "e": int(entity_id),
                    "s": float(score),
                    "f": __import__("json").dumps(safe, default=str),
                    "t": datetime.utcnow(),
                })
        except Exception as e:
            log.error("prediction log write failed: %s", e)


_service: InferenceService | None = None


def get_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
        _service.load()
    return _service
