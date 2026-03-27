"""API route handlers."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Query, Request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from recommender.api import metrics as m
from recommender.api.models import Mode, SimilarResponse

router = APIRouter()


@router.get("/similar", response_model=SimilarResponse)
def similar(
    request: Request,
    post_id: int = Query(..., description="Source post ID"),
    limit: int = Query(default=6, ge=1, le=20),
    explain: bool = Query(default=False),
    include_scores: bool = Query(default=True),
    mode: Mode = Query(default="favorites", description="Recommendation mode"),
):
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    t0 = time.perf_counter()
    try:
        result = engine.query(post_id=post_id, limit=limit, explain=explain, include_scores=include_scores, mode=mode)
        m.requests_total.labels(endpoint="/similar", status="200").inc()
        return result
    except Exception:
        m.requests_total.labels(endpoint="/similar", status="500").inc()
        raise
    finally:
        m.request_latency.labels(endpoint="/similar").observe(time.perf_counter() - t0)


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request):
    if not hasattr(request.app.state, "engine") or request.app.state.engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_version": request.app.state.engine._b.version}


@router.get("/metrics")
def metrics_endpoint():
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
