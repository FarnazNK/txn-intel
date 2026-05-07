"""Feature store: offline computation + online retrieval.

Offline: SQL-based point-in-time aggregations materialized into feat_* tables.
Online: Redis cache keyed by (feature_view, entity_id) with TTL.

Point-in-time correctness: when training, we compute features as of a label
date; when serving, we compute as of "now". The same SQL is used in both modes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import redis
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import engine

log = get_logger(__name__)
_settings = get_settings()
_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(_settings.redis_url, decode_responses=True)
    return _redis


CUSTOMER_FEATURES = [
    "txn_count_30d",
    "txn_count_90d",
    "amount_sum_30d",
    "amount_sum_90d",
    "amount_mean_30d",
    "days_since_last_txn",
    "distinct_products_90d",
    "support_tickets_90d",
    "tenure_days",
]


# Single SQL that computes all customer features for a given as-of date.
# Designed so both batch (training labels at varying as_of_date) and online
# (single customer at "now") use the same expression.
_CUSTOMER_FEATURES_SQL = """
WITH params AS (
    SELECT :as_of_date::date AS as_of_date
),
base_customers AS (
    SELECT c.id AS customer_id, c.signup_date
    FROM customers c
    {customer_filter}
),
txn_30 AS (
    SELECT customer_id,
           COUNT(*)                          AS txn_count_30d,
           COALESCE(SUM(amount), 0)::float   AS amount_sum_30d,
           COALESCE(AVG(amount), 0)::float   AS amount_mean_30d
    FROM transactions, params
    WHERE occurred_at < params.as_of_date
      AND occurred_at >= params.as_of_date - INTERVAL '30 days'
      AND status = 'completed'
    GROUP BY customer_id
),
txn_90 AS (
    SELECT customer_id,
           COUNT(*)                          AS txn_count_90d,
           COALESCE(SUM(amount), 0)::float   AS amount_sum_90d,
           COUNT(DISTINCT product_id)        AS distinct_products_90d,
           MAX(occurred_at)                  AS last_txn_at
    FROM transactions, params
    WHERE occurred_at < params.as_of_date
      AND occurred_at >= params.as_of_date - INTERVAL '90 days'
      AND status = 'completed'
    GROUP BY customer_id
),
tickets_90 AS (
    SELECT customer_id, COUNT(*) AS support_tickets_90d
    FROM support_tickets, params
    WHERE created_at < params.as_of_date
      AND created_at >= params.as_of_date - INTERVAL '90 days'
    GROUP BY customer_id
)
SELECT
    bc.customer_id,
    p.as_of_date,
    COALESCE(t30.txn_count_30d, 0)            AS txn_count_30d,
    COALESCE(t90.txn_count_90d, 0)            AS txn_count_90d,
    COALESCE(t30.amount_sum_30d, 0.0)         AS amount_sum_30d,
    COALESCE(t90.amount_sum_90d, 0.0)         AS amount_sum_90d,
    COALESCE(t30.amount_mean_30d, 0.0)        AS amount_mean_30d,
    COALESCE(
        EXTRACT(EPOCH FROM (p.as_of_date - t90.last_txn_at)) / 86400,
        9999
    )::int                                    AS days_since_last_txn,
    COALESCE(t90.distinct_products_90d, 0)    AS distinct_products_90d,
    COALESCE(tk.support_tickets_90d, 0)       AS support_tickets_90d,
    GREATEST(0, (p.as_of_date - bc.signup_date))::int AS tenure_days
FROM base_customers bc
CROSS JOIN params p
LEFT JOIN txn_30  t30 ON t30.customer_id = bc.customer_id
LEFT JOIN txn_90  t90 ON t90.customer_id = bc.customer_id
LEFT JOIN tickets_90 tk ON tk.customer_id = bc.customer_id
"""


@dataclass
class FeatureRequest:
    customer_ids: list[int] | None
    as_of_date: date


def compute_customer_features(req: FeatureRequest, eng: Engine = engine) -> pd.DataFrame:
    """Compute features for a set of customers as of a given date.

    If customer_ids is None, computes for all customers — used for batch backfill.
    """
    if req.customer_ids is not None:
        # Inline-safe: SQL parameter binding for IN clause
        sql = _CUSTOMER_FEATURES_SQL.format(
            customer_filter="WHERE c.id = ANY(:customer_ids)"
        )
        params = {"as_of_date": req.as_of_date, "customer_ids": req.customer_ids}
    else:
        sql = _CUSTOMER_FEATURES_SQL.format(customer_filter="")
        params = {"as_of_date": req.as_of_date}

    with eng.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def materialize_customer_features(as_of_date: date) -> int:
    """Compute features for ALL customers and write to feat_customer_daily.

    Used for offline training feature builds. Idempotent on (customer_id, as_of_date).
    """
    df = compute_customer_features(FeatureRequest(None, as_of_date))
    if df.empty:
        return 0
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM feat_customer_daily WHERE as_of_date = :d"),
            {"d": as_of_date},
        )
    df.to_sql("feat_customer_daily", engine, if_exists="append", index=False, chunksize=10_000)
    log.info("materialized %d customer feature rows for %s", len(df), as_of_date)
    return len(df)


def get_online_customer_features(customer_id: int) -> dict | None:
    """Online retrieval: cache → fall through to live SQL.

    Returns a feature dict suitable for direct model input.
    """
    r = _get_redis()
    key = f"feat:customer:{customer_id}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)

    df = compute_customer_features(
        FeatureRequest(customer_ids=[customer_id], as_of_date=date.today())
    )
    if df.empty:
        return None
    feats = df.iloc[0].to_dict()
    # Make JSON-safe
    feats = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in feats.items()}
    r.setex(key, _settings.feature_cache_ttl_seconds, json.dumps(feats, default=str))
    return feats


def invalidate_customer_cache(customer_ids: Iterable[int]) -> None:
    r = _get_redis()
    pipe = r.pipeline()
    for cid in customer_ids:
        pipe.delete(f"feat:customer:{cid}")
    pipe.execute()


def vector_for_model(features: dict, columns: list[str] = CUSTOMER_FEATURES) -> list[float]:
    """Convert feature dict into ordered vector for model input."""
    return [float(features[c]) for c in columns]
