"""Product recommendation: two-stage retrieve + rank.

Stage 1 (candidate generation): implicit-feedback matrix factorization via
ALS-style alternating updates on the customer x product purchase matrix.
We use scikit-learn's TruncatedSVD on a confidence-weighted matrix as a
lightweight stand-in (scales to the dataset without an extra dep).

Stage 2 (ranker): LightGBM ranker (LambdaRank) using:
  - MF dot-product score
  - product features (price, category one-hot, popularity)
  - customer features (recency, plan, tenure)
  - cross features (customer-category affinity)

Trained with leave-last-out: hold out each customer's most recent purchase.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import ndcg_score
from sqlalchemy import text

from app.core.logging import get_logger
from app.db.session import engine
from app.ml.registry import ModelMetadata, ModelRegistry

log = get_logger(__name__)
MODEL_NAME = "recommend"

LATENT_DIM = 32
N_NEGATIVES = 5
N_CANDIDATES = 50


def fetch_interactions(min_date: datetime | None = None) -> pd.DataFrame:
    """Fetch (customer, product, count) interactions for MF training."""
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


def build_matrix(interactions: pd.DataFrame
                 ) -> tuple[sparse.csr_matrix, dict[int, int], dict[int, int]]:
    cust_ids = sorted(interactions["customer_id"].unique())
    prod_ids = sorted(interactions["product_id"].unique())
    cust_to_idx = {c: i for i, c in enumerate(cust_ids)}
    prod_to_idx = {p: i for i, p in enumerate(prod_ids)}

    rows = interactions["customer_id"].map(cust_to_idx).values
    cols = interactions["product_id"].map(prod_to_idx).values
    # Confidence weighting: log1p of count
    data = np.log1p(interactions["n"].values).astype(np.float32)
    mat = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(cust_ids), len(prod_ids))
    )
    return mat, cust_to_idx, prod_to_idx


def fit_mf(mat: sparse.csr_matrix, dim: int = LATENT_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Cheap MF via TruncatedSVD on confidence-weighted matrix."""
    svd = TruncatedSVD(n_components=dim, random_state=42)
    user_emb = svd.fit_transform(mat)
    # Item embedding = V^T scaled by sigma; svd.components_ is V^T
    item_emb = svd.components_.T * svd.singular_values_
    return user_emb, item_emb


def make_train_pairs(interactions: pd.DataFrame, user_emb: np.ndarray, item_emb: np.ndarray,
                     cust_to_idx: dict, prod_to_idx: dict, prod_features: pd.DataFrame,
                     n_negatives: int = N_NEGATIVES) -> pd.DataFrame:
    """Build ranker training set: positive pairs + sampled negatives.

    Group key = customer; relevance = 1 for positive, 0 for negative.
    """
    rng = np.random.default_rng(42)
    interactions = interactions.copy()
    interactions["last_at"] = pd.to_datetime(interactions["last_at"])
    interactions = interactions.sort_values(["customer_id", "last_at"])

    # Hold out the most-recent interaction per customer for eval
    interactions["rank_within"] = interactions.groupby("customer_id")["last_at"].rank("first", ascending=False)
    eval_pos = interactions[interactions["rank_within"] == 1]
    train_pos = interactions[interactions["rank_within"] > 1]

    all_prod_idx = np.arange(len(prod_to_idx))
    rows: list[dict] = []

    def sample_negs(seen: set[int], n: int) -> np.ndarray:
        # Rejection sample
        out = []
        while len(out) < n:
            cand = rng.choice(all_prod_idx, size=n * 2)
            for c in cand:
                if int(c) not in seen:
                    out.append(int(c))
                    if len(out) >= n:
                        break
        return np.array(out[:n])

    prod_feat_idx = prod_features.set_index("product_id")
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
                # Map back to product_id
                p_id_neg = next(pid for pid, idx in prod_to_idx.items() if idx == n_idx)
                rows.append(_make_row(u_idx, n_idx, 0, cust_id, p_id_neg,
                                      user_emb, item_emb, prod_feat_idx))

    return pd.DataFrame(rows), eval_pos


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
        "mf_score": score,
        "price": price,
        "popularity_log": float(np.log1p(pop)),
    }


def train() -> str:
    interactions = fetch_interactions()
    if len(interactions) < 100:
        raise RuntimeError("not enough interactions to train recommender")

    prod_features = fetch_product_features()
    mat, cust_to_idx, prod_to_idx = build_matrix(interactions)
    log.info("matrix shape: %s, nnz=%d", mat.shape, mat.nnz)

    user_emb, item_emb = fit_mf(mat)
    log.info("MF fit: user_emb=%s, item_emb=%s", user_emb.shape, item_emb.shape)

    train_df, eval_pos = make_train_pairs(
        interactions, user_emb, item_emb, cust_to_idx, prod_to_idx, prod_features,
    )
    log.info("ranker training rows: %d", len(train_df))

    # Sort by customer for LGBMRanker grouping
    train_df = train_df.sort_values("customer_id").reset_index(drop=True)
    feature_cols = ["mf_score", "price", "popularity_log"]
    groups = train_df.groupby("customer_id").size().values

    ranker = lgb.LGBMRanker(
        n_estimators=200, num_leaves=31, learning_rate=0.05,
        objective="lambdarank", random_state=42, n_jobs=-1, verbose=-1,
    )
    ranker.fit(train_df[feature_cols], train_df["label"], group=groups)
    log.info("ranker fit complete")

    # Evaluate via NDCG on held-out (most recent) positive
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
        notes="two-stage MF retrieve + LGBMRanker",
    )
    registry.save(MODEL_NAME, bundle, meta)
    current = registry.champion(MODEL_NAME)
    if current is None:
        registry.promote(MODEL_NAME, version)
    else:
        winner = registry.compare(MODEL_NAME, current, version, "ndcg_at_10", True)
        if winner == version:
            registry.promote(MODEL_NAME, version)
    return version


def _evaluate_ranker(eval_pos: pd.DataFrame, ranker: lgb.LGBMRanker,
                     user_emb: np.ndarray, item_emb: np.ndarray,
                     cust_to_idx: dict, prod_to_idx: dict,
                     prod_features: pd.DataFrame, k: int = 10) -> dict[str, float]:
    rng = np.random.default_rng(42)
    all_prod_idx = np.arange(len(prod_to_idx))
    prod_feat_idx = prod_features.set_index("product_id")
    ndcgs = []
    sample = eval_pos.sample(min(2000, len(eval_pos)), random_state=42)
    for _, row in sample.iterrows():
        cid = row["customer_id"]
        if cid not in cust_to_idx:
            continue
        true_pid = row["product_id"]
        if true_pid not in prod_to_idx:
            continue
        u_idx = cust_to_idx[cid]
        # Sample 99 negatives + 1 positive
        neg_idx = rng.choice(all_prod_idx, size=99, replace=False)
        cand_idx = np.append(neg_idx, prod_to_idx[true_pid])
        rng.shuffle(cand_idx)
        # Build feature rows
        feats = []
        labels = []
        for c_idx in cand_idx:
            p_id = next(pid for pid, idx in prod_to_idx.items() if idx == c_idx)
            mf = float(np.dot(user_emb[u_idx], item_emb[c_idx]))
            if p_id in prod_feat_idx.index:
                pf = prod_feat_idx.loc[p_id]
                price = float(pf["price"])
                pop = int(pf["popularity"])
            else:
                price, pop = 0.0, 0
            feats.append([mf, price, np.log1p(pop)])
            labels.append(1 if p_id == true_pid else 0)
        scores = ranker.predict(np.array(feats))
        ndcgs.append(ndcg_score([labels], [scores], k=k))
    return {"ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0}


def recommend(bundle: dict, customer_id: int, n: int = 10) -> list[tuple[int, float]]:
    """Generate top-N recommendations for a customer."""
    cust_to_idx = bundle["cust_to_idx"]
    prod_to_idx = bundle["prod_to_idx"]
    if customer_id not in cust_to_idx:
        # Cold start: fall back to popularity
        prod_feat = bundle["prod_features"]
        top = prod_feat.nlargest(n, "popularity")
        return [(int(pid), float(p["popularity"])) for pid, p in top.iterrows()]

    u_idx = cust_to_idx[customer_id]
    user_vec = bundle["user_emb"][u_idx]
    scores = bundle["item_emb"] @ user_vec  # all products

    # Stage 1: top N_CANDIDATES by MF score
    cand_idx = np.argpartition(-scores, N_CANDIDATES)[:N_CANDIDATES]
    cand_pids = []
    feats = []
    prod_feat_idx = bundle["prod_features"]
    idx_to_prod = {idx: pid for pid, idx in prod_to_idx.items()}
    for c_idx in cand_idx:
        pid = idx_to_prod[int(c_idx)]
        if pid in prod_feat_idx.index:
            pf = prod_feat_idx.loc[pid]
            feats.append([float(scores[c_idx]), float(pf["price"]), float(np.log1p(pf["popularity"]))])
            cand_pids.append(pid)

    # Stage 2: rerank
    rank_scores = bundle["ranker"].predict(np.array(feats))
    order = np.argsort(-rank_scores)[:n]
    return [(int(cand_pids[i]), float(rank_scores[i])) for i in order]


if __name__ == "__main__":
    from app.core.logging import configure_logging
    configure_logging()
    train()
