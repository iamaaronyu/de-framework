"""Stage 执行引擎，支持重试与错误隔离。"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Iterator

from framework.models import JobContext, StageResult, StageStatus

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """所有流水线 Stage 必须继承此基类。"""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]:
        """消费上游数据 *inputs*，yield 处理后的记录。"""

    def validate_config(self) -> None:
        """子类可重写此方法，在启动时校验 self.config。"""


class StageRunner:
    """执行单个 Stage，采集指标并处理异常。"""

    def __init__(self, max_retries: int = 1) -> None:
        self._max_retries = max_retries

    def run(
        self,
        stage: BaseStage,
        stage_id: str,
        upstream_data: list[dict[str, Any]],
        ctx: JobContext,
    ) -> tuple[StageResult, list[dict[str, Any]]]:
        """执行 *stage*，返回 (StageResult, 输出记录列表)。"""
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._max_retries:
            if attempt > 0:
                wait = 2**attempt
                logger.info(
                    "重试 Stage '%s'（第 %d/%d 次）等待 %ds",
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
                logger.warning("Stage '%s' 第 %d 次执行失败：%s", stage_id, attempt, exc)
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
            "Stage '%s' 执行成功，耗时 %.2fs：输入=%d 输出=%d 过滤=%d",
            stage_id,
            duration,
            result.records_in,
            result.records_out,
            result.records_filtered,
        )
        return result, output
