"""Tests for the NPU inference FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from npu_service.server import app, AVAILABLE_MODELS

client = TestClient(app)


def test_health():
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_models():
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert set(data["models"]) == set(AVAILABLE_MODELS.keys())


def test_inference_success():
    resp = client.post(
        "/v1/inference",
        json={"model": "llm-v2", "inputs": [{"text": "hello world"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "llm-v2"
    assert len(data["outputs"]) == 1
    assert "latency_ms" in data


def test_inference_unknown_model():
    resp = client.post(
        "/v1/inference",
        json={"model": "nonexistent-model", "inputs": [{"text": "hi"}]},
    )
    assert resp.status_code == 404


def test_batch_inference_exceeds_limit():
    model = "llm-v2"
    max_batch = AVAILABLE_MODELS[model]["max_batch_size"]
    oversized = [{"text": f"item {i}"} for i in range(max_batch + 1)]
    resp = client.post(
        "/v1/inference/batch",
        json={"model": model, "inputs": oversized},
    )
    assert resp.status_code == 400


def test_batch_inference_success():
    inputs = [{"text": f"sample {i}"} for i in range(5)]
    resp = client.post(
        "/v1/inference/batch",
        json={"model": "llm-v2", "inputs": inputs},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["outputs"]) == 5


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"npu_inference_requests_total" in resp.content
