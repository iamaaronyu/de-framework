"""Tests for NPU invoker client."""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import MagicMock, patch

from framework.npu_invoker import NPUInvoker, NPUInvokerError


def _make_invoker(base_url: str = "http://npu-test:8080") -> NPUInvoker:
    return NPUInvoker(base_url, timeout_seconds=5.0, max_retries=2)


class TestNPUInvokerHealth:
    def test_health_returns_true_on_200(self):
        invoker = _make_invoker()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(invoker._client, "get", return_value=mock_resp):
            assert invoker.health() is True

    def test_health_returns_false_on_connection_error(self):
        invoker = _make_invoker()
        with patch.object(invoker._client, "get", side_effect=httpx.ConnectError("refused")):
            assert invoker.health() is False


class TestNPUInvokerBatch:
    def test_infer_batch_splits_correctly(self):
        invoker = _make_invoker()
        inputs = [{"text": f"item {i}"} for i in range(10)]
        expected_outputs = [{"output": f"result {i}"} for i in range(10)]

        call_args = []

        def fake_post(path: str, json: dict) -> MagicMock:
            call_args.append(json)
            batch_size = len(json["inputs"])
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "outputs": [{"output": f"result {i}"} for i in range(batch_size)]
            }
            return resp

        with patch.object(invoker._client, "post", side_effect=fake_post):
            outputs = invoker.infer_batch("llm-v2", inputs, batch_size=4)

        assert len(outputs) == 10
        # 10 items with batch_size=4 → 3 calls (4+4+2)
        assert len(call_args) == 3

    def test_infer_raises_on_missing_outputs_key(self):
        invoker = _make_invoker()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"unexpected": "format"}

        with patch.object(invoker._client, "post", return_value=mock_resp):
            with pytest.raises(NPUInvokerError, match="Unexpected response format"):
                invoker.infer("llm-v2", [{"text": "hello"}])


class TestNPUInvokerListModels:
    def test_list_models(self):
        invoker = _make_invoker()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"models": ["llm-v1", "llm-v2"]}

        with patch.object(invoker._client, "get", return_value=mock_resp):
            models = invoker.list_models()

        assert models == ["llm-v1", "llm-v2"]
