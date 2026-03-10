"""向云道平台实时上报任务进度。"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx

from framework.models import JobStatus, StageResult, StageStatus

logger = logging.getLogger(__name__)


class ProgressReporter:
    """向云道任务管理平台上报任务与 Stage 状态。"""

    def __init__(self, yundao_url: str, job_id: str) -> None:
        self._url = yundao_url.rstrip("/")
        self._job_id = job_id
        self._lock = threading.Lock()
        self._client = httpx.Client(base_url=self._url, timeout=10.0)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def report_job_start(self) -> None:
        self._post(
            f"/api/jobs/{self._job_id}/status",
            {"status": JobStatus.RUNNING, "message": "Job started"},
        )

    def report_stage_start(self, stage_id: str) -> None:
        self._post(
            f"/api/jobs/{self._job_id}/stages/{stage_id}",
            {"status": StageStatus.RUNNING},
        )

    def report_stage_result(self, result: StageResult) -> None:
        self._post(
            f"/api/jobs/{self._job_id}/stages/{result.stage_id}",
            {
                "status": result.status,
                "records_in": result.records_in,
                "records_out": result.records_out,
                "records_filtered": result.records_filtered,
                "duration_seconds": result.duration_seconds,
                "error": result.error,
            },
        )

    def report_job_complete(self, status: JobStatus, error: str | None = None) -> None:
        self._post(
            f"/api/jobs/{self._job_id}/status",
            {"status": status, "error": error},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ProgressReporter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> None:
        with self._lock:
            try:
                resp = self._client.post(path, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # 进度上报失败不能中断流水线主流程
                logger.warning("任务 %s 进度上报失败：%s", self._job_id, exc)
