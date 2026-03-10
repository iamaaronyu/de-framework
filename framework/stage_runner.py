"""Stage execution with retry and error isolation."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator

from framework.models import JobContext, StageResult, StageStatus

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """All pipeline stages must subclass this."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]:
        """Consume *inputs* and yield output records."""

    def validate_config(self) -> None:
        """Override to validate self.config at startup."""


class StageRunner:
    """Executes a single stage, collecting metrics and handling errors."""

    def __init__(self, max_retries: int = 1) -> None:
        self._max_retries = max_retries

    def run(
        self,
        stage: BaseStage,
        stage_id: str,
        upstream_data: list[dict[str, Any]],
        ctx: JobContext,
    ) -> tuple[StageResult, list[dict[str, Any]]]:
        """Run *stage*, return (result, output_records)."""
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._max_retries:
            if attempt > 0:
                wait = 2**attempt
                logger.info(
                    "Retrying stage '%s' (attempt %d/%d) after %ds",
                    stage_id,
                    attempt,
                    self._max_retries,
                    wait,
                )
                time.sleep(wait)

            try:
                return self._execute(stage, stage_id, upstream_data, ctx)
            except Exception as exc:
                last_error = exc
                logger.warning("Stage '%s' attempt %d failed: %s", stage_id, attempt, exc)
                attempt += 1

        result = StageResult(
            stage_id=stage_id,
            status=StageStatus.FAILED,
            records_in=len(upstream_data),
            error=str(last_error),
        )
        return result, []

    def _execute(
        self,
        stage: BaseStage,
        stage_id: str,
        upstream_data: list[dict[str, Any]],
        ctx: JobContext,
    ) -> tuple[StageResult, list[dict[str, Any]]]:
        t0 = time.monotonic()
        output = list(stage.process(iter(upstream_data), ctx))
        duration = time.monotonic() - t0

        result = StageResult(
            stage_id=stage_id,
            status=StageStatus.SUCCESS,
            records_in=len(upstream_data),
            records_out=len(output),
            records_filtered=len(upstream_data) - len(output),
            duration_seconds=round(duration, 3),
        )
        logger.info(
            "Stage '%s' succeeded in %.2fs: in=%d out=%d filtered=%d",
            stage_id,
            duration,
            result.records_in,
            result.records_out,
            result.records_filtered,
        )
        return result, output
