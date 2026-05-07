"""Embed support tickets and store in pgvector for semantic search.

Run after data is loaded; idempotent (skips already-embedded rows).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from tqdm import tqdm

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import engine

log = get_logger(__name__)
_settings = get_settings()


def get_unembedded_batch(batch_size: int) -> list[tuple[int, str]]:
    sql = text("""
        SELECT id, subject || E'\\n' || body AS text
        FROM support_tickets
        WHERE embedding IS NULL
        ORDER BY id
        LIMIT :n
    """)
    with engine.connect() as conn:
        return [(r.id, r.text) for r in conn.execute(sql, {"n": batch_size})]


def write_embeddings(rows: Iterable[tuple[int, np.ndarray]]) -> None:
    with engine.begin() as conn:
        for tid, vec in rows:
            conn.execute(
                text("UPDATE support_tickets SET embedding = :v WHERE id = :id"),
                {"v": vec.tolist(), "id": tid},
            )


def run(batch_size: int = 256) -> int:
    model = SentenceTransformer(_settings.embedding_model)
    total = 0
    while True:
        batch = get_unembedded_batch(batch_size)
        if not batch:
            break
        texts = [t for _, t in batch]
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        write_embeddings(zip([tid for tid, _ in batch], vecs))
        total += len(batch)
        log.info("embedded %d so far", total)
    log.info("done, total embedded: %d", total)
    return total


if __name__ == "__main__":
    configure_logging()
    run()
