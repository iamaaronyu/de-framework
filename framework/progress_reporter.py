"""Real-time progress reporting to the Yundao platform."""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx

from framework.models import JobStatus, StageResult, StageStatus

logger = logging.getLogger(__name__)


class ProgressReporter:
    """Reports job and stage status to the Yundao task management platform."""

    def __init__(self, yundao_url: str, job_id: str) -> None:
        self._url = yundao_url.rstrip("/")
        self._job_id = job_id
        self._lock = threading.Lock()
        self._client = httpx.Client(base_url=self._url, timeout=10.0)

    # ------------------------------------------------------------------
    # Public
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
    # Internal
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> None:
        with self._lock:
            try:
                resp = self._client.post(path, json=payload)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                # Progress reporting failure must never abort the pipeline
                logger.warning("Progress report failed for job %s: %s", self._job_id, exc)
