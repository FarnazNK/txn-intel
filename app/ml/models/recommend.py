"""Product recommendation — PyTorch two-tower retrieve + Keras ranker.

Stage 1 (candidate generation) — PyTorch.
    A two-tower neural collaborative filter: separate user and item embedding
    tables of dim ``LATENT_DIM``. Trained with BPR (Bayesian Personalized
    Ranking) loss: for each (user, positive_item) pair, sample a random
    negative item and minimize ``-log(sigmoid(score_pos - score_neg))``. The
    learned dot product user·item replaces the SVD-on-confidence-weighted-
    matrix score from the old design.

Stage 2 (re-ranker) — TensorFlow / Keras.
    A small dense classifier that takes ``[two_tower_score, price,
    popularity_log]`` and predicts P(click). Trained with binary
    cross-entropy on positive vs sampled-negative pairs. Replaces the
    LightGBM LambdaRanker.

Trained with leave-last-out: hold each customer's most-recent purchase
for NDCG@10 evaluation against 99 random negatives.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from sklearn.metrics import ndcg_score  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.core.logging import get_logger  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.ml.registry import ModelMetadata, ModelRegistry  # noqa: E402

log = get_logger(__name__)
MODEL_NAME = "recommend"

LATENT_DIM = 32
N_NEGATIVES = 5
N_CANDIDATES = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── PyTorch two-tower model ─────────────────────────────────────────────────
class TwoTower(nn.Module):
    """User + item embedding tables; score = u · v.

    We keep it as plain dot-product (no MLP fusion) because the second-stage
    Keras ranker handles non-linear combination of the score with side
    features. This makes Stage 1 fast and the embeddings reusable.
    """

    def __init__(self, n_users: int, n_items: int, dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, u_idx: torch.Tensor, i_idx: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(u_idx) * self.item_emb(i_idx)).sum(dim=-1)

    def forward(self, u_idx: torch.Tensor, pos_idx: torch.Tensor,
                neg_idx: torch.Tensor) -> torch.Tensor:
        return self.score(u_idx, pos_idx) - self.score(u_idx, neg_idx)


# ── Data ─────────────────────────────────────────────────────────────────────
def fetch_interactions(min_date: datetime | None = None) -> pd.DataFrame:
    where_clause = "WHERE status = 'completed'"
    if min_date is not None:
        where_clause += f" AND occurred_at >= '{min_date.isoformat()}'"
    sql = text(f"""
        SELECT customer_id, product_id, COUNT(*)::int AS n,
               SUM(amount)::float AS spend,
               MAX(occurred_at) AS last_at
        FROM transactions
        {where_clause}
        GROUP BY customer_id, product_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    log.info("fetched %d (customer, product) pairs", len(df))
    return df


def fetch_product_features() -> pd.DataFrame:
    sql = text("""
        SELECT p.id AS product_id, p.merchant_id, p.category, p.price::float AS price,
               COALESCE(stat.popularity, 0)::int AS popularity
        FROM products p
        LEFT JOIN (
            SELECT product_id, COUNT(*) AS popularity
            FROM transactions WHERE status = 'completed'
            GROUP BY product_id
        ) stat ON stat.product_id = p.id
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def build_index(interactions: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    cust_ids = sorted(interactions["customer_id"].unique())
    prod_ids = sorted(interactions["product_id"].unique())
    return ({c: i for i, c in enumerate(cust_ids)},
            {p: i for i, p in enumerate(prod_ids)})


# ── Stage 1: two-tower training with BPR ─────────────────────────────────────
def _train_two_tower(train_pos: pd.DataFrame,
                     cust_to_idx: dict[int, int],
                     prod_to_idx: dict[int, int],
                     *, dim: int = LATENT_DIM,
                     epochs: int = 8, batch_size: int = 4096,
                     lr: float = 5e-3) -> TwoTower:
    """BPR training. ``train_pos`` must have customer_id, product_id, n columns."""
    n_users = len(cust_to_idx)
    n_items = len(prod_to_idx)

    # Index arrays + user→seen-items set for negative-sampling rejection
    u_arr = train_pos["customer_id"].map(cust_to_idx).values.astype(np.int64)
    i_arr = train_pos["product_id"].map(prod_to_idx).values.astype(np.int64)

    seen: dict[int, set[int]] = {}
    for u, i in zip(u_arr, i_arr):
        seen.setdefault(int(u), set()).add(int(i))

    model = TwoTower(n_users=n_users, n_items=n_items, dim=dim).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(42)

    n_pos = len(u_arr)
    log.info("BPR: %d positives, %d users, %d items", n_pos, n_users, n_items)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_pos)
        running = 0.0
        for start in range(0, n_pos, batch_size):
            batch_idx = perm[start:start + batch_size]
            u_b = u_arr[batch_idx]
            pos_b = i_arr[batch_idx]
            # Sample a random negative per row; rejection-sample once if seen.
            neg_b = rng.integers(0, n_items, size=len(batch_idx))
            for k, (u, ng) in enumerate(zip(u_b, neg_b)):
                if int(ng) in seen.get(int(u), ()):
                    neg_b[k] = rng.integers(0, n_items)

            u_t = torch.from_numpy(u_b).to(DEVICE)
            p_t = torch.from_numpy(pos_b).to(DEVICE)
            n_t = torch.from_numpy(neg_b.astype(np.int64)).to(DEVICE)

            optim.zero_grad()
            diff = model(u_t, p_t, n_t)
            loss = -torch.log(torch.sigmoid(diff) + 1e-9).mean()
            loss.backward()
            optim.step()
            running += float(loss.item()) * len(batch_idx)
        log.info("  BPR epoch %d: loss=%.4f", epoch, running / max(n_pos, 1))

    model.eval()
    return model


def _extract_embeddings(model: TwoTower) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        u = model.user_emb.weight.detach().cpu().numpy()
        v = model.item_emb.weight.detach().cpu().numpy()
    return u, v


# ── Stage 2: Keras ranker ────────────────────────────────────────────────────
def build_ranker(in_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(in_dim,), name="rank_in")
    x = keras.layers.Dense(32, activation="relu")(inputs)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="rank_out")(x)
    m = keras.Model(inputs, outputs, name="recommend_ranker")
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return m


def make_train_pairs(train_pos: pd.DataFrame, user_emb: np.ndarray, item_emb: np.ndarray,
                     cust_to_idx: dict, prod_to_idx: dict, prod_features: pd.DataFrame,
                     n_negatives: int = N_NEGATIVES) -> pd.DataFrame:
    """Build ranker training set: positive pairs + sampled negatives."""
    rng = np.random.default_rng(42)
    all_prod_idx = np.arange(len(prod_to_idx))
    idx_to_prod = {idx: pid for pid, idx in prod_to_idx.items()}
    prod_feat_idx = prod_features.set_index("product_id")
    rows: list[dict] = []

    def sample_negs(seen: set[int], n: int) -> list[int]:
        out: list[int] = []
        while len(out) < n:
            cand = rng.choice(all_prod_idx, size=n * 2)
            for c in cand:
                if int(c) not in seen:
                    out.append(int(c))
                    if len(out) >= n:
                        break
        return out

    for cust_id, group in train_pos.groupby("customer_id"):
        if cust_id not in cust_to_idx:
            continue
        u_idx = cust_to_idx[cust_id]
        seen_idx = {prod_to_idx[p] for p in group["product_id"] if p in prod_to_idx}
        for _, row in group.iterrows():
            if row["product_id"] not in prod_to_idx:
                continue
            p_idx = prod_to_idx[row["product_id"]]
            rows.append(_make_row(u_idx, p_idx, 1, cust_id, row["product_id"],
                                  user_emb, item_emb, prod_feat_idx))
            for n_idx in sample_negs(seen_idx, n_negatives):
                p_id_neg = idx_to_prod[n_idx]
                rows.append(_make_row(u_idx, n_idx, 0, cust_id, p_id_neg,
                                      user_emb, item_emb, prod_feat_idx))

    return pd.DataFrame(rows)


def _make_row(u_idx: int, p_idx: int, label: int, cust_id: int, prod_id: int,
              user_emb: np.ndarray, item_emb: np.ndarray,
              prod_feat_idx: pd.DataFrame) -> dict:
    score = float(np.dot(user_emb[u_idx], item_emb[p_idx]))
    if prod_id in prod_feat_idx.index:
        pf = prod_feat_idx.loc[prod_id]
        price = float(pf["price"])
        pop = int(pf["popularity"])
    else:
        price, pop = 0.0, 0
    return {
        "customer_id": int(cust_id),
        "product_id": int(prod_id),
        "label": int(label),
        "two_tower_score": score,
        "price": price,
        "popularity_log": float(np.log1p(pop)),
    }


# ── Persistence ──────────────────────────────────────────────────────────────
def _save_artifacts(bundle: dict, version_dir: Path) -> None:
    # Two-tower: save state_dict + dims; embeddings extracted at load time.
    tt_state = {
        "user_emb.weight": bundle["_two_tower"].user_emb.weight.detach().cpu(),
        "item_emb.weight": bundle["_two_tower"].item_emb.weight.detach().cpu(),
    }
    torch.save({
        "state_dict": tt_state,
        "n_users": bundle["_two_tower"].user_emb.num_embeddings,
        "n_items": bundle["_two_tower"].item_emb.num_embeddings,
        "dim": bundle["_two_tower"].user_emb.embedding_dim,
    }, version_dir / "two_tower.pt")
    bundle["ranker"].save(version_dir / "ranker.keras")
    side = {
        "cust_to_idx": bundle["cust_to_idx"],
        "prod_to_idx": bundle["prod_to_idx"],
        "feature_cols": bundle["feature_cols"],
        "prod_features": bundle["prod_features"],
    }
    joblib.dump(side, version_dir / "bundle.joblib")


def _load_artifacts(version_dir: Path) -> dict:
    blob = torch.load(version_dir / "two_tower.pt", map_location=DEVICE)
    model = TwoTower(n_users=blob["n_users"], n_items=blob["n_items"],
                     dim=blob["dim"]).to(DEVICE)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    user_emb, item_emb = _extract_embeddings(model)

    ranker = keras.models.load_model(version_dir / "ranker.keras", compile=False)
    side = joblib.load(version_dir / "bundle.joblib")
    return {
        "user_emb": user_emb,
        "item_emb": item_emb,
        "cust_to_idx": side["cust_to_idx"],
        "prod_to_idx": side["prod_to_idx"],
        "ranker": ranker,
        "feature_cols": side["feature_cols"],
        "prod_features": side["prod_features"],
        "_two_tower": model,
    }


def load_champion(registry: ModelRegistry | None = None) -> tuple[dict, ModelMetadata]:
    registry = registry or ModelRegistry()
    return registry.load(MODEL_NAME, loader=_load_artifacts)


# ── Eval ─────────────────────────────────────────────────────────────────────
def _evaluate_ranker(eval_pos: pd.DataFrame, ranker: keras.Model,
                     user_emb: np.ndarray, item_emb: np.ndarray,
                     cust_to_idx: dict, prod_to_idx: dict,
                     prod_features: pd.DataFrame, k: int = 10) -> dict[str, float]:
    rng = np.random.default_rng(42)
    all_prod_idx = np.arange(len(prod_to_idx))
    idx_to_prod = {idx: pid for pid, idx in prod_to_idx.items()}
    prod_feat_idx = prod_features.set_index("product_id")
    ndcgs: list[float] = []
    sample = eval_pos.sample(min(2000, len(eval_pos)), random_state=42)
    for _, row in sample.iterrows():
        cid = row["customer_id"]
        if cid not in cust_to_idx:
            continue
        true_pid = row["product_id"]
        if true_pid not in prod_to_idx:
            continue
        u_idx = cust_to_idx[cid]
        neg_idx = rng.choice(all_prod_idx, size=99, replace=False)
        cand_idx = np.append(neg_idx, prod_to_idx[true_pid])
        rng.shuffle(cand_idx)
        feats = []
        labels = []
        for c_idx in cand_idx:
            p_id = idx_to_prod[int(c_idx)]
            mf = float(np.dot(user_emb[u_idx], item_emb[c_idx]))
            if p_id in prod_feat_idx.index:
                pf = prod_feat_idx.loc[p_id]
                price = float(pf["price"])
                pop = int(pf["popularity"])
            else:
                price, pop = 0.0, 0
            feats.append([mf, price, np.log1p(pop)])
            labels.append(1 if p_id == true_pid else 0)
        scores = ranker.predict(np.array(feats), verbose=0).reshape(-1)
        ndcgs.append(ndcg_score([labels], [scores], k=k))
    return {"ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0}


# ── Entrypoint ───────────────────────────────────────────────────────────────
def train() -> str:
    interactions = fetch_interactions()
    if len(interactions) < 100:
        raise RuntimeError("not enough interactions to train recommender")

    prod_features = fetch_product_features()
    cust_to_idx, prod_to_idx = build_index(interactions)

    interactions = interactions.copy()
    interactions["last_at"] = pd.to_datetime(interactions["last_at"])
    interactions = interactions.sort_values(["customer_id", "last_at"])
    interactions["rank_within"] = (
        interactions.groupby("customer_id")["last_at"].rank("first", ascending=False)
    )
    eval_pos = interactions[interactions["rank_within"] == 1]
    train_pos = interactions[interactions["rank_within"] > 1]

    log.info("training two-tower on %d positive (cust, prod) pairs", len(train_pos))
    tt_model = _train_two_tower(train_pos, cust_to_idx, prod_to_idx)
    user_emb, item_emb = _extract_embeddings(tt_model)
    log.info("two-tower fit: user_emb=%s, item_emb=%s", user_emb.shape, item_emb.shape)

    train_df = make_train_pairs(
        train_pos, user_emb, item_emb, cust_to_idx, prod_to_idx, prod_features,
    )
    log.info("ranker training rows: %d", len(train_df))

    feature_cols = ["two_tower_score", "price", "popularity_log"]
    ranker = build_ranker(in_dim=len(feature_cols))
    ranker.fit(
        train_df[feature_cols].values.astype(np.float32),
        train_df["label"].values.astype(np.float32),
        epochs=10, batch_size=1024, shuffle=True, verbose=0, validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True,
        )],
    )
    log.info("ranker fit complete")

    eval_metrics = _evaluate_ranker(
        eval_pos, ranker, user_emb, item_emb, cust_to_idx, prod_to_idx, prod_features,
    )
    log.info("eval metrics: %s", eval_metrics)

    bundle = {
        "user_emb": user_emb,
        "item_emb": item_emb,
        "cust_to_idx": cust_to_idx,
        "prod_to_idx": prod_to_idx,
        "ranker": ranker,
        "feature_cols": feature_cols,
        "prod_features": prod_features.set_index("product_id"),
        "_two_tower": tt_model,
    }

    registry = ModelRegistry()
    version = ModelRegistry.new_version()
    meta = ModelMetadata(
        name=MODEL_NAME,
        version=version,
        trained_at=datetime.utcnow().isoformat(),
        feature_schema=feature_cols,
        metrics=eval_metrics,
        training_window={"interactions": str(len(interactions))},
        n_train=int(len(train_df)),
        n_eval=int(len(eval_pos)),
        notes="pytorch two-tower + keras ranker",
        extra={"framework": "pytorch+tensorflow"},
    )
    registry.save(MODEL_NAME, bundle, meta, saver=_save_artifacts)
    current = registry.champion(MODEL_NAME)
    if current is None:
        registry.promote(MODEL_NAME, version)
    else:
        winner = registry.compare(MODEL_NAME, current, version, "ndcg_at_10", True)
        if winner == version:
            registry.promote(MODEL_NAME, version)
    return version


# ── Public recommend API (used by inference service) ─────────────────────────
def recommend(bundle: dict, customer_id: int, n: int = 10) -> list[tuple[int, float]]:
    cust_to_idx = bundle["cust_to_idx"]
    prod_to_idx = bundle["prod_to_idx"]
    if customer_id not in cust_to_idx:
        prod_feat = bundle["prod_features"]
        top = prod_feat.nlargest(n, "popularity")
        return [(int(pid), float(p["popularity"])) for pid, p in top.iterrows()]

    u_idx = cust_to_idx[customer_id]
    user_vec = bundle["user_emb"][u_idx]
    scores = bundle["item_emb"] @ user_vec  # all items

    # Stage 1: top N_CANDIDATES by two-tower score
    cand_idx = np.argpartition(-scores, N_CANDIDATES)[:N_CANDIDATES]
    idx_to_prod = {idx: pid for pid, idx in prod_to_idx.items()}
    prod_feat_idx = bundle["prod_features"]
    feats: list[list[float]] = []
    cand_pids: list[int] = []
    for c_idx in cand_idx:
        pid = idx_to_prod[int(c_idx)]
        if pid in prod_feat_idx.index:
            pf = prod_feat_idx.loc[pid]
            feats.append([
                float(scores[c_idx]),
                float(pf["price"]),
                float(np.log1p(pf["popularity"])),
            ])
            cand_pids.append(pid)

    # Stage 2: rerank with Keras ranker
    rank_scores = bundle["ranker"].predict(
        np.array(feats, dtype=np.float32), verbose=0,
    ).reshape(-1)
    order = np.argsort(-rank_scores)[:n]
    return [(int(cand_pids[i]), float(rank_scores[i])) for i in order]


if __name__ == "__main__":
    from app.core.logging import configure_logging
    configure_logging()
    train()
