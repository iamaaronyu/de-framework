"""执行框架共享数据模型。"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class StageStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class StageConfig:
    id: str
    type: str
    module: str
    class_name: str
    depends_on: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    name: str
    version: str
    description: str
    stages: list[StageConfig]


@dataclass
class JobContext:
    job_id: str
    pipeline_name: str
    pipeline_version: str
    input_path: str
    output_path: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    stage_id: str
    status: StageStatus
    records_in: int = 0
    records_out: int = 0
    records_filtered: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    stage_results: list[StageResult] = field(default_factory=list)
    total_records_in: int = 0
    total_records_out: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
