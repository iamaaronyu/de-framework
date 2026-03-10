"""NPU 推理服务 — FastAPI 应用。

对外暴露接口：
  POST /v1/inference          — 同步单条推理
  POST /v1/inference/batch    — 批量推理
  GET  /v1/models             — 查询可用模型列表
  GET  /v1/health             — 健康检查
  GET  /metrics               — Prometheus 指标
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
# Prometheus 指标
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "npu_inference_requests_total",
    "推理请求总数",
    ["model", "status"],
)
REQUEST_LATENCY = Histogram(
    "npu_inference_latency_seconds",
    "推理请求耗时",
    ["model"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)
BATCH_SIZE_HIST = Histogram(
    "npu_inference_batch_size",
    "每次批量推理的请求条数",
    buckets=[1, 4, 8, 16, 32, 64, 128, 256],
)

# ---------------------------------------------------------------------------
# 已注册模型（生产环境从模型注册中心动态加载）
# ---------------------------------------------------------------------------
AVAILABLE_MODELS: dict[str, Any] = {
    "llm-v2": {"description": "LLM 蒸馏模型 v2", "max_batch_size": 64},
    "llm-v1": {"description": "LLM 蒸馏模型 v1（历史版本）", "max_batch_size": 32},
}

# ---------------------------------------------------------------------------
# 请求 / 响应数据模型
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
# 路由
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
# 核心推理逻辑
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
        logger.exception("模型 %s 推理后端异常", model)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = (time.monotonic() - t0) * 1000
    REQUEST_COUNT.labels(model=model, status=status).inc()
    REQUEST_LATENCY.labels(model=model).observe(latency_ms / 1000)

    return InferenceResponse(model=model, outputs=outputs, latency_ms=round(latency_ms, 2))


def _call_npu_backend(model: str, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """占位实现 — 生产环境替换为真实 NPU SDK / Triton 客户端调用。"""
    # 生产环境：在此调用 Triton Inference Server 或厂商 NPU SDK
    return [{"input_id": i, "output": f"[{model}] result for item {i}"} for i, _ in enumerate(inputs)]
