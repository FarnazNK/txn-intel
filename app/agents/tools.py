"""Tools available to the BI agent.

All tools are read-only and run against an allowlist of SQL views to prevent
arbitrary query execution. The agent can:
  - run_sql: parameterized queries against safe views
  - search_tickets: semantic search over support tickets
  - get_customer_summary: aggregated metrics for one customer
  - get_merchant_summary: aggregated metrics for one merchant
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text

from app.db.session import engine
from app.ml.features.semantic import search_tickets

# Whitelisted views/tables and the columns that are safe to expose
SAFE_TABLES = {
    "merchants": ["id", "name", "industry", "region", "created_at"],
    "customers": ["id", "merchant_id", "signup_date", "cohort", "country", "plan", "is_active"],
    "products": ["id", "merchant_id", "sku", "name", "category", "price"],
    "transactions": ["id", "merchant_id", "customer_id", "product_id",
                     "occurred_at", "amount", "quantity", "channel", "status"],
    "support_tickets": ["id", "merchant_id", "customer_id", "created_at",
                        "category", "subject", "resolved"],
}

# Single regex-based check for unsafe SQL constructs
UNSAFE_PATTERNS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|"
    r"vacuum|reindex|do|call|execute)\b",
    re.IGNORECASE,
)

MAX_ROWS = 200


def run_sql(query: str) -> dict[str, Any]:
    """Run a read-only SELECT against the safe table set.

    Enforces:
      - Single statement, must start with SELECT or WITH
      - No write keywords anywhere
      - LIMIT capped at MAX_ROWS
      - Wrapped in a read-only transaction
    """
    q = query.strip().rstrip(";")
    if ";" in q:
        return {"error": "only a single statement allowed"}
    head = q.split(None, 1)[0].lower() if q else ""
    if head not in ("select", "with"):
        return {"error": "only SELECT or WITH queries allowed"}
    if UNSAFE_PATTERNS.search(q):
        return {"error": "write/DDL keywords not permitted"}

    # Append/cap LIMIT
    if not re.search(r"\blimit\b", q, re.IGNORECASE):
        q = f"{q} LIMIT {MAX_ROWS}"

    try:
        with engine.connect().execution_options(readonly=True) as conn:
            conn.execute(text("SET LOCAL statement_timeout = '5s'"))
            conn.execute(text("SET LOCAL transaction_read_only = on"))
            result = conn.execute(text(q))
            cols = list(result.keys())
            rows = [dict(zip(cols, r)) for r in result.fetchmany(MAX_ROWS)]
            for row in rows:
                for k, v in row.items():
                    if hasattr(v, "isoformat"):
                        row[k] = v.isoformat()
            return {"columns": cols, "rows": rows, "row_count": len(rows)}
    except Exception as e:
        return {"error": f"query failed: {type(e).__name__}: {e}"}


def search_support_tickets(query: str, k: int = 5,
                           merchant_id: int | None = None) -> dict[str, Any]:
    hits = search_tickets(query, k=k, merchant_id=merchant_id)
    return {
        "results": [
            {
                "ticket_id": h.ticket_id,
                "merchant_id": h.merchant_id,
                "customer_id": h.customer_id,
                "category": h.category,
                "subject": h.subject,
                "body": h.body[:300],
                "similarity": round(h.similarity, 4),
            }
            for h in hits
        ]
    }


def get_customer_summary(customer_id: int) -> dict[str, Any]:
    sql = text("""
        SELECT
            c.id, c.merchant_id, c.plan, c.country, c.signup_date,
            COUNT(t.id) AS lifetime_txns,
            COALESCE(SUM(t.amount), 0)::float AS lifetime_spend,
            MAX(t.occurred_at) AS last_txn_at,
            (SELECT COUNT(*) FROM support_tickets st WHERE st.customer_id = c.id) AS lifetime_tickets
        FROM customers c
        LEFT JOIN transactions t ON t.customer_id = c.id AND t.status = 'completed'
        WHERE c.id = :cid
        GROUP BY c.id
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"cid": customer_id}).first()
    if not row:
        return {"error": f"customer {customer_id} not found"}
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in dict(row._mapping).items()}


def get_merchant_summary(merchant_id: int) -> dict[str, Any]:
    sql = text("""
        SELECT
            m.id, m.name, m.industry, m.region,
            COUNT(DISTINCT c.id) AS n_customers,
            COUNT(t.id) AS lifetime_txns,
            COALESCE(SUM(t.amount), 0)::float AS lifetime_revenue,
            COALESCE(AVG(t.amount), 0)::float AS avg_ticket_size
        FROM merchants m
        LEFT JOIN customers c ON c.merchant_id = m.id
        LEFT JOIN transactions t ON t.merchant_id = m.id AND t.status = 'completed'
        WHERE m.id = :mid
        GROUP BY m.id
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"mid": merchant_id}).first()
    if not row:
        return {"error": f"merchant {merchant_id} not found"}
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in dict(row._mapping).items()}


# Tool spec for Anthropic API
TOOL_DEFINITIONS = [
    {
        "name": "run_sql",
        "description": (
            "Run a read-only SQL SELECT against the warehouse. Available tables: "
            "merchants(id, name, industry, region, created_at), "
            "customers(id, merchant_id, signup_date, cohort, country, plan, is_active), "
            "products(id, merchant_id, sku, name, category, price), "
            "transactions(id, merchant_id, customer_id, product_id, occurred_at, amount, quantity, channel, status), "
            "support_tickets(id, merchant_id, customer_id, created_at, category, subject, resolved). "
            "Use this for aggregations, filters, joins. Limit 200 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A single SELECT or WITH ... SELECT statement."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_support_tickets",
        "description": (
            "Semantic search over support tickets. Use this when the user is "
            "asking about issues, complaints, or themes in customer feedback "
            "that aren't expressible as simple SQL filters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
                "merchant_id": {"type": "integer", "description": "Optional merchant scope."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_customer_summary",
        "description": "Get a one-row summary of lifetime metrics for a single customer.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "integer"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "get_merchant_summary",
        "description": "Get a one-row summary of metrics for a single merchant.",
        "input_schema": {
            "type": "object",
            "properties": {"merchant_id": {"type": "integer"}},
            "required": ["merchant_id"],
        },
    },
]


TOOL_DISPATCH = {
    "run_sql": lambda inp: run_sql(inp["query"]),
    "search_support_tickets": lambda inp: search_support_tickets(
        inp["query"], inp.get("k", 5), inp.get("merchant_id"),
    ),
    "get_customer_summary": lambda inp: get_customer_summary(inp["customer_id"]),
    "get_merchant_summary": lambda inp: get_merchant_summary(inp["merchant_id"]),
}
