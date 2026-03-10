"""NPU 推理集群客户端。

封装对 NPU 推理服务的 REST API 调用，支持重试、mini-batch 拆分与超时控制。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class NPUInvokerError(Exception):
    pass


class NPUInvoker:
    """NPU 推理集群 REST API 的 HTTP 客户端。"""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def infer(self, model: str, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """单次同步推理调用。"""
        return self._infer_with_retry(model=model, inputs=inputs)

    def infer_batch(
        self, model: str, inputs: list[dict[str, Any]], batch_size: int = 32
    ) -> list[dict[str, Any]]:
        """将 *inputs* 拆分为 mini-batch 分批调用，聚合后返回完整结果。"""
        results: list[dict[str, Any]] = []
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i : i + batch_size]
            logger.debug(
                "向 NPU 发送 batch %d-%d（共 %d 条）",
                i,
                i + len(chunk),
                len(chunk),
            )
            results.extend(self._infer_with_retry(model=model, inputs=chunk))
        return results

    def list_models(self) -> list[str]:
        resp = self._client.get("/v1/models")
        resp.raise_for_status()
        return resp.json().get("models", [])

    def health(self) -> bool:
        try:
            resp = self._client.get("/v1/health", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> NPUInvoker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _infer_with_retry(
        self, model: str, inputs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        @retry(
            retry=retry_if_exception_type((httpx.HTTPError, NPUInvokerError)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _call() -> list[dict[str, Any]]:
            payload = {"model": model, "inputs": inputs}
            try:
                resp = self._client.post("/v1/inference/batch", json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "NPU API 返回 %d：%s", exc.response.status_code, exc.response.text
                )
                raise NPUInvokerError(str(exc)) from exc
            data = resp.json()
            if "outputs" not in data:
                raise NPUInvokerError(f"响应格式异常：{data}")
            return data["outputs"]

        return _call()
