"""Semantic search over support tickets using pgvector cosine similarity."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import engine


@dataclass
class TicketHit:
    ticket_id: int
    merchant_id: int
    customer_id: int
    category: str
    subject: str
    body: str
    similarity: float


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def search_tickets(query: str, k: int = 10,
                   merchant_id: int | None = None) -> list[TicketHit]:
    vec = _embedder().encode([query], normalize_embeddings=True)[0].tolist()
    where = ""
    params: dict = {"q": vec, "k": k}
    if merchant_id is not None:
        where = "WHERE merchant_id = :merchant_id"
        params["merchant_id"] = merchant_id
    sql = text(f"""
        SELECT id, merchant_id, customer_id, category, subject, body,
               1 - (embedding <=> CAST(:q AS vector)) AS similarity
        FROM support_tickets
        {where}
        AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
    """.replace("\n        AND embedding" if where else "WHERE embedding",
                "AND embedding" if where else "WHERE embedding"))
    # Fix WHERE/AND combinator
    base_where = "WHERE embedding IS NOT NULL"
    if merchant_id is not None:
        base_where = "WHERE merchant_id = :merchant_id AND embedding IS NOT NULL"
    sql = text(f"""
        SELECT id, merchant_id, customer_id, category, subject, body,
               1 - (embedding <=> CAST(:q AS vector)) AS similarity
        FROM support_tickets
        {base_where}
        ORDER BY embedding <=> CAST(:q AS vector)
        LIMIT :k
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, params).all()
    return [
        TicketHit(
            ticket_id=r.id, merchant_id=r.merchant_id, customer_id=r.customer_id,
            category=r.category, subject=r.subject, body=r.body,
            similarity=float(r.similarity),
        )
        for r in rows
    ]
