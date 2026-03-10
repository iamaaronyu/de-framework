"""NPU inference stage for the LLM distillation pipeline."""

from __future__ import annotations

import logging
from typing import Any, Iterator

from framework.models import JobContext
from framework.npu_invoker import NPUInvoker, NPUInvokerError
from framework.stage_runner import BaseStage

logger = logging.getLogger(__name__)


class InferenceStage(BaseStage):
    """Calls the NPU inference API in mini-batches and attaches results."""

    def __init__(self, config: dict[str, Any], npu_invoker: NPUInvoker) -> None:
        super().__init__(config)
        self._npu = npu_invoker

    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]:
        model: str = self.config["model"]
        batch_size: int = self.config.get("batch_size", 32)

        batch: list[dict[str, Any]] = []

        for record in inputs:
            batch.append(record)
            if len(batch) >= batch_size:
                yield from self._infer_batch(model, batch)
                batch = []

        if batch:
            yield from self._infer_batch(model, batch)

    def _infer_batch(
        self, model: str, batch: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        npu_inputs = [{"text": r.get("text", ""), "id": r.get("id")} for r in batch]
        try:
            outputs = self._npu.infer_batch(model, npu_inputs, batch_size=len(npu_inputs))
        except NPUInvokerError as exc:
            # Mark records as failed rather than dropping them
            logger.error("NPU inference failed for batch of %d: %s", len(batch), exc)
            return [
                {**r, "npu_output": None, "npu_error": str(exc), "inference_ok": False}
                for r in batch
            ]

        enriched = []
        for record, output in zip(batch, outputs, strict=False):
            enriched.append({**record, "npu_output": output, "inference_ok": True})
        logger.debug("Inferred batch of %d records with model '%s'", len(batch), model)
        return enriched
