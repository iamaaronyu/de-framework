"""NPU inference cluster client.

Wraps REST API calls to the NPU inference service with retry, batching,
and timeout support.
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
    """HTTP client for the NPU inference cluster REST API."""

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
    # Public API
    # ------------------------------------------------------------------

    def infer(self, model: str, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Single synchronous inference call."""
        return self._infer_with_retry(model=model, inputs=inputs)

    def infer_batch(
        self, model: str, inputs: list[dict[str, Any]], batch_size: int = 32
    ) -> list[dict[str, Any]]:
        """Split *inputs* into mini-batches and aggregate results."""
        results: list[dict[str, Any]] = []
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i : i + batch_size]
            logger.debug(
                "Sending batch %d-%d (%d items) to NPU",
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
    # Internal
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
                    "NPU API returned %d: %s", exc.response.status_code, exc.response.text
                )
                raise NPUInvokerError(str(exc)) from exc
            data = resp.json()
            if "outputs" not in data:
                raise NPUInvokerError(f"Unexpected response format: {data}")
            return data["outputs"]

        return _call()
