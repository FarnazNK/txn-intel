"""Churn prediction — PyTorch MLP.

Label definition: a customer is "churned at as_of_date" if they had >=1 txn in
the 90 days BEFORE as_of_date but ZERO txns in the 60 days AFTER as_of_date.
This gives a forward-looking, behaviorally grounded label.

Features: from feat_customer_daily (point-in-time as of label date).
Model: a small MLP trained with class-weighted BCE, AdamW, early stopping on
validation PR-AUC. We keep a sklearn StandardScaler in front for numerical
stability — fit on the training split only.

The artifact bundle persisted to the registry is:
    - model.pt        : torch state_dict
    - bundle.joblib   : dict with {scaler, feature_order, model_config}
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from app.core.logging import get_logger
from app.db.session import engine
from app.ml.features.customer import CUSTOMER_FEATURES, materialize_customer_features
from app.ml.registry import ModelMetadata, ModelRegistry

log = get_logger(__name__)
MODEL_NAME = "churn"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Torch model ──────────────────────────────────────────────────────────────
class ChurnMLP(nn.Module):
    """Small feed-forward classifier producing a single logit.

    Architecture: [in_dim -> 64 -> 32 -> 1] with BatchNorm, ReLU, Dropout.
    Kept deliberately compact: the feature set is ~10-30 numerical features
    and the dataset is tens of thousands of rows, so depth/width past this
    just overfits.
    """

    def __init__(self, in_dim: int, hidden: tuple[int, int] = (64, 32),
                 dropout: float = 0.2) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)


# ── Data ─────────────────────────────────────────────────────────────────────
def build_labels(label_dates: list[date]) -> pd.DataFrame:
    """Build churn labels by joining transaction activity around each label date."""
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


# ── Training loop ────────────────────────────────────────────────────────────
def _train_mlp(X_tr: np.ndarray, y_tr: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray,
               in_dim: int, *,
               epochs: int = 60, batch_size: int = 512,
               lr: float = 1e-3, weight_decay: float = 1e-4,
               patience: int = 8) -> tuple[ChurnMLP, dict]:
    """Train the MLP with early stopping on validation PR-AUC.

    Returns the best model state and a small history dict.
    """
    model = ChurnMLP(in_dim=in_dim).to(DEVICE)

    # Class-weighted BCE: pos_weight = neg/pos so the positive class is
    # up-weighted in proportion to its rarity.
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor(max(neg, 1.0) / max(pos, 1.0), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.from_numpy(X_tr.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
    X_val_t = torch.from_numpy(X_val.astype(np.float32)).to(DEVICE)

    ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_pr_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * xb.size(0)
        train_loss = running / len(ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
        val_proba = 1.0 / (1.0 + np.exp(-val_logits))
        val_pr_auc = float(average_precision_score(y_val, val_proba))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pr_auc": val_pr_auc})

        if val_pr_auc > best_pr_auc + 1e-5:
            best_pr_auc = val_pr_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("early stopping at epoch %d (best val_pr_auc=%.4f)",
                         epoch, best_pr_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_pr_auc": best_pr_auc, "history": history}


def _predict_proba(model: ChurnMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X.astype(np.float32)).to(DEVICE)).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
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


# ── Persistence ──────────────────────────────────────────────────────────────
def _save_artifacts(bundle: dict, version_dir: Path) -> None:
    """Registry-compatible saver."""
    torch.save(bundle["model"].state_dict(), version_dir / "model.pt")
    side = {
        "scaler": bundle["scaler"],
        "feature_order": bundle["feature_order"],
        "model_config": bundle["model_config"],
    }
    joblib.dump(side, version_dir / "bundle.joblib")


def _load_artifacts(version_dir: Path) -> dict:
    """Registry-compatible loader."""
    side = joblib.load(version_dir / "bundle.joblib")
    cfg = side["model_config"]
    model = ChurnMLP(in_dim=cfg["in_dim"], hidden=tuple(cfg["hidden"]),
                     dropout=cfg["dropout"]).to(DEVICE)
    state = torch.load(version_dir / "model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "scaler": side["scaler"],
        "feature_order": side["feature_order"],
        "model_config": cfg,
    }


def load_champion(registry: ModelRegistry | None = None) -> tuple[dict, ModelMetadata]:
    """Convenience used by the inference service."""
    registry = registry or ModelRegistry()
    return registry.load(MODEL_NAME, loader=_load_artifacts)


def predict_proba(bundle: dict, features_df: pd.DataFrame) -> np.ndarray:
    """Score a small DataFrame whose columns are the feature schema."""
    X = features_df[bundle["feature_order"]].astype(float).values
    X = bundle["scaler"].transform(X)
    return _predict_proba(bundle["model"], X)


# ── Entrypoint ───────────────────────────────────────────────────────────────
def train(label_dates: list[date] | None = None,
          backfill_features: bool = True) -> str:
    if label_dates is None:
        today = date.today()
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

    X = df[CUSTOMER_FEATURES].astype(float).values
    y = df["churned"].values.astype(int)

    # 60/20/20 train/val/test
    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr_full, y_tr_full, test_size=0.25, stratify=y_tr_full, random_state=42,
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    in_dim = X_tr_s.shape[1]
    log.info("training MLP (in_dim=%d, n_train=%d, n_val=%d)", in_dim, len(X_tr_s), len(X_val_s))
    model, train_info = _train_mlp(X_tr_s, y_tr, X_val_s, y_val, in_dim=in_dim)

    test_proba = _predict_proba(model, X_te_s)
    metrics = evaluate(y_te, test_proba)
    log.info("test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_order": CUSTOMER_FEATURES,
        "model_config": {"in_dim": in_dim, "hidden": [64, 32], "dropout": 0.2},
    }

    registry = ModelRegistry()
    version = ModelRegistry.new_version()
    meta = ModelMetadata(
        name=MODEL_NAME,
        version=version,
        trained_at=pd.Timestamp.utcnow().isoformat(),
        feature_schema=CUSTOMER_FEATURES,
        metrics=metrics,
        training_window={
            "label_dates": ",".join(d.isoformat() for d in label_dates),
        },
        n_train=int(len(X_tr_s)),
        n_eval=int(len(X_te_s)),
        notes=f"pytorch mlp; best_val_pr_auc={train_info['best_val_pr_auc']:.4f}",
        extra={"framework": "pytorch", "device": str(DEVICE)},
    )
    registry.save(MODEL_NAME, bundle, meta, saver=_save_artifacts)

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
