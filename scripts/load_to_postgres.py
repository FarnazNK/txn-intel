"""Load parquet files from data/raw/ into Postgres.

Creates schema, enables pgvector, bulk-inserts via COPY for transactions.
Run after scripts/generate_data.py.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from app.core.logging import configure_logging, get_logger
from app.db.models import Base
from app.db.session import engine

DATA_DIR = Path("./data/raw")
log = get_logger(__name__)


def setup_extensions() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        log.info("pgvector extension ready")


def create_schema() -> None:
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    log.info("schema created")


def copy_df(df: pd.DataFrame, table: str) -> None:
    """Bulk insert via COPY FROM STDIN."""
    if len(df) == 0:
        return
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
        buf.seek(0)
        cols = ",".join(df.columns)
        cur.copy_expert(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\\\N')", buf)
        raw.commit()
    finally:
        raw.close()


def load() -> None:
    setup_extensions()
    create_schema()

    for name, table in [
        ("merchants", "merchants"),
        ("customers", "customers"),
        ("products", "products"),
        ("transactions", "transactions"),
        ("tickets", "support_tickets"),
    ]:
        path = DATA_DIR / f"{name}.parquet"
        if not path.exists():
            log.warning("missing %s, skipping", path)
            continue
        df = pd.read_parquet(path)
        log.info("loading %s rows into %s", len(df), table)
        copy_df(df, table)

    # Build vector index after data is loaded
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_tickets_embedding "
            "ON support_tickets USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        ))
        log.info("vector index created")


if __name__ == "__main__":
    configure_logging()
    load()
