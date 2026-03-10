"""NPU Inference Service — FastAPI application.

Exposes:
  POST /v1/inference          — synchronous single-item inference
  POST /v1/inference/batch    — batch inference
  GET  /v1/models             — list available models
  GET  /v1/health             — health check
  GET  /metrics               — Prometheus metrics
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

app = FastAPI(title="NPU Inference Service", version="1.0.0")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "npu_inference_requests_total",
    "Total inference requests",
    ["model", "status"],
)
REQUEST_LATENCY = Histogram(
    "npu_inference_latency_seconds",
    "Inference request latency",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)
BATCH_SIZE_HIST = Histogram(
    "npu_inference_batch_size",
    "Number of items per batch request",
    buckets=[1, 4, 8, 16, 32, 64, 128, 256],
)

# ---------------------------------------------------------------------------
# Registered models (production: loaded from model registry)
# ---------------------------------------------------------------------------
AVAILABLE_MODELS: dict[str, Any] = {
    "llm-v2": {"description": "LLM distillation model v2", "max_batch_size": 64},
    "llm-v1": {"description": "LLM distillation model v1 (legacy)", "max_batch_size": 32},
}

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class InferenceRequest(BaseModel):
    model: str
    inputs: list[dict[str, Any]]


class InferenceResponse(BaseModel):
    model: str
    outputs: list[dict[str, Any]]
    latency_ms: float


class BatchInferenceRequest(BaseModel):
    model: str
    inputs: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "models": list(AVAILABLE_MODELS.keys()),
        "details": AVAILABLE_MODELS,
    }


@app.post("/v1/inference", response_model=InferenceResponse)
def inference(req: InferenceRequest) -> InferenceResponse:
    return _run_inference(req.model, req.inputs)


@app.post("/v1/inference/batch", response_model=InferenceResponse)
def batch_inference(req: BatchInferenceRequest) -> InferenceResponse:
    BATCH_SIZE_HIST.observe(len(req.inputs))
    return _run_inference(req.model, req.inputs)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------------------------------------------------------------------------
# Core inference logic
# ---------------------------------------------------------------------------


def _run_inference(model: str, inputs: list[dict[str, Any]]) -> InferenceResponse:
    if model not in AVAILABLE_MODELS:
        REQUEST_COUNT.labels(model=model, status="error").inc()
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    model_info = AVAILABLE_MODELS[model]
    max_batch = model_info.get("max_batch_size", 32)
    if len(inputs) > max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(inputs)} exceeds model limit {max_batch}",
        )

    t0 = time.monotonic()
    try:
        outputs = _call_npu_backend(model, inputs)
        status = "success"
    except Exception as exc:
        REQUEST_COUNT.labels(model=model, status="error").inc()
        logger.exception("Inference backend error for model %s", model)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.monotonic() - t0) * 1000
    REQUEST_COUNT.labels(model=model, status=status).inc()
    REQUEST_LATENCY.labels(model=model).observe(latency_ms / 1000)

    return InferenceResponse(model=model, outputs=outputs, latency_ms=round(latency_ms, 2))


def _call_npu_backend(model: str, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Placeholder — replace with real NPU SDK / triton client call."""
    # In production: call Triton Inference Server or vendor NPU SDK here
    return [{"input_id": i, "output": f"[{model}] result for item {i}"} for i, _ in enumerate(inputs)]
