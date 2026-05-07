from datetime import datetime, date

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.session import Base

EMBED_DIM = get_settings().embedding_dim


class Merchant(Base):
    __tablename__ = "merchants"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    industry: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    region: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[date] = mapped_column(Date, nullable=False)


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.id"), index=True)
    signup_date: Mapped[date] = mapped_column(Date, nullable=False)
    cohort: Mapped[str] = mapped_column(String(16), nullable=False)
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    plan: Mapped[str] = mapped_column(String(16), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.id"), index=True)
    sku: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)


class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.id"), index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.id"), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    amount: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    channel: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        Index("ix_txn_merchant_time", "merchant_id", "occurred_at"),
        Index("ix_txn_customer_time", "customer_id", "occurred_at"),
    )


class SupportTicket(Base):
    __tablename__ = "support_tickets"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.id"), index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    subject: Mapped[str] = mapped_column(String(256), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)


# ── Feature store offline tables ───────────────────────────────────────────────


class CustomerFeatures(Base):
    """Point-in-time feature snapshot per (customer_id, as_of_date)."""

    __tablename__ = "feat_customer_daily"
    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    txn_count_30d: Mapped[int] = mapped_column(Integer, nullable=False)
    txn_count_90d: Mapped[int] = mapped_column(Integer, nullable=False)
    amount_sum_30d: Mapped[float] = mapped_column(Float, nullable=False)
    amount_sum_90d: Mapped[float] = mapped_column(Float, nullable=False)
    amount_mean_30d: Mapped[float] = mapped_column(Float, nullable=False)
    days_since_last_txn: Mapped[int] = mapped_column(Integer, nullable=False)
    distinct_products_90d: Mapped[int] = mapped_column(Integer, nullable=False)
    support_tickets_90d: Mapped[int] = mapped_column(Integer, nullable=False)
    tenure_days: Mapped[int] = mapped_column(Integer, nullable=False)


# ── Predictions + monitoring ───────────────────────────────────────────────────


class PredictionLog(Base):
    __tablename__ = "prediction_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DriftMetric(Base):
    __tablename__ = "drift_metrics"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    feature: Mapped[str] = mapped_column(String(64), nullable=False)
    metric: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
