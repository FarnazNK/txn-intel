from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    customer_id: int


class ChurnResponse(BaseModel):
    customer_id: int
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_band: str
    model_version: str
    features: dict[str, Any]


class AnomalyRequest(BaseModel):
    transaction_features: dict[str, float]


class AnomalyResponse(BaseModel):
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomalous: bool
    threshold: float
    model_version: str


class RecommendRequest(BaseModel):
    customer_id: int
    n: int = 10


class RecommendItem(BaseModel):
    product_id: int
    score: float


class RecommendResponse(BaseModel):
    customer_id: int
    items: list[RecommendItem]
    model_version: str


class TicketSearchRequest(BaseModel):
    query: str
    k: int = 10
    merchant_id: int | None = None


class TicketSearchHit(BaseModel):
    ticket_id: int
    merchant_id: int
    customer_id: int
    category: str
    subject: str
    similarity: float


class TicketSearchResponse(BaseModel):
    hits: list[TicketSearchHit]


class AgentRequest(BaseModel):
    question: str


class AgentToolCall(BaseModel):
    tool: str
    input: dict[str, Any]
    result_summary: str


class AgentResponseModel(BaseModel):
    answer: str
    turns: int
    tool_calls: list[AgentToolCall]


class HealthResponse(BaseModel):
    status: str
    models: dict[str, str | None]
    timestamp: datetime
