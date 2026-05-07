"""FastAPI application: prediction, recommendation, search, and agent endpoints."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.agents import bi_agent
from app.api import schemas
from app.api.inference import InferenceService, get_service
from app.core.logging import configure_logging, get_logger
from app.ml.features.semantic import search_tickets

log = get_logger(__name__)

REQUEST_COUNT = Counter(
    "txn_intel_requests_total", "API requests", ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "txn_intel_request_seconds", "Request latency", ["endpoint"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    service = get_service()
    log.info("models loaded: %s", service.model_versions())
    yield


app = FastAPI(title="Transaction Intelligence Platform", lifespan=lifespan)


@app.get("/health", response_model=schemas.HealthResponse)
def health(service: InferenceService = Depends(get_service)):
    return schemas.HealthResponse(
        status="ok",
        models=service.model_versions(),
        timestamp=datetime.utcnow(),
    )


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict/churn", response_model=schemas.ChurnResponse)
def predict_churn(req: schemas.ChurnRequest,
                  service: InferenceService = Depends(get_service)):
    with REQUEST_LATENCY.labels("predict_churn").time():
        try:
            result = service.predict_churn(req.customer_id)
            REQUEST_COUNT.labels("predict_churn", "ok").inc()
            return schemas.ChurnResponse(
                customer_id=req.customer_id,
                churn_probability=result["probability"],
                risk_band=result["risk_band"],
                model_version=result["version"],
                features=result["features"],
            )
        except ValueError as e:
            REQUEST_COUNT.labels("predict_churn", "404").inc()
            raise HTTPException(404, str(e))
        except RuntimeError as e:
            REQUEST_COUNT.labels("predict_churn", "503").inc()
            raise HTTPException(503, str(e))


@app.post("/predict/anomaly", response_model=schemas.AnomalyResponse)
def predict_anomaly(req: schemas.AnomalyRequest,
                    service: InferenceService = Depends(get_service)):
    with REQUEST_LATENCY.labels("predict_anomaly").time():
        try:
            result = service.predict_anomaly(req.transaction_features)
            REQUEST_COUNT.labels("predict_anomaly", "ok").inc()
            return schemas.AnomalyResponse(
                anomaly_score=result["score"],
                is_anomalous=result["is_anomalous"],
                threshold=result["threshold"],
                model_version=result["version"],
            )
        except ValueError as e:
            REQUEST_COUNT.labels("predict_anomaly", "400").inc()
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            REQUEST_COUNT.labels("predict_anomaly", "503").inc()
            raise HTTPException(503, str(e))


@app.post("/recommend", response_model=schemas.RecommendResponse)
def recommend(req: schemas.RecommendRequest,
              service: InferenceService = Depends(get_service)):
    with REQUEST_LATENCY.labels("recommend").time():
        try:
            result = service.recommend(req.customer_id, req.n)
            REQUEST_COUNT.labels("recommend", "ok").inc()
            return schemas.RecommendResponse(
                customer_id=req.customer_id,
                items=[schemas.RecommendItem(product_id=p, score=s)
                       for p, s in result["items"]],
                model_version=result["version"],
            )
        except RuntimeError as e:
            REQUEST_COUNT.labels("recommend", "503").inc()
            raise HTTPException(503, str(e))


@app.post("/search/tickets", response_model=schemas.TicketSearchResponse)
def search(req: schemas.TicketSearchRequest):
    with REQUEST_LATENCY.labels("search_tickets").time():
        hits = search_tickets(req.query, k=req.k, merchant_id=req.merchant_id)
        REQUEST_COUNT.labels("search_tickets", "ok").inc()
        return schemas.TicketSearchResponse(hits=[
            schemas.TicketSearchHit(
                ticket_id=h.ticket_id, merchant_id=h.merchant_id,
                customer_id=h.customer_id, category=h.category,
                subject=h.subject, similarity=h.similarity,
            )
            for h in hits
        ])


@app.post("/agent/query", response_model=schemas.AgentResponseModel)
def agent_query(req: schemas.AgentRequest):
    with REQUEST_LATENCY.labels("agent_query").time():
        result = bi_agent.ask(req.question)
        REQUEST_COUNT.labels("agent_query", "ok").inc()
        return schemas.AgentResponseModel(
            answer=result.answer,
            turns=result.turns,
            tool_calls=[schemas.AgentToolCall(**tc) for tc in result.tool_calls],
        )
