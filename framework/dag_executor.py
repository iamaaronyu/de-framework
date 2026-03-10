"""DAG 解析与拓扑执行引擎。"""

from __future__ import annotations

import importlib
import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import yaml

from framework.models import PipelineConfig, StageConfig

if TYPE_CHECKING:
    from framework.stage_runner import BaseStage

logger = logging.getLogger(__name__)


class DAGValidationError(Exception):
    pass


def load_pipeline_config(path: str) -> PipelineConfig:
    """解析 pipeline.yaml 文件，返回 PipelineConfig 对象。"""
    with open(path) as f:
        raw = yaml.safe_load(f)

    stages = []
    for s in raw.get("stages", []):
        stages.append(
            StageConfig(
                id=s["id"],
                type=s["type"],
                module=s["module"],
                class_name=s["class"],
                depends_on=s.get("depends_on", []),
                config=s.get("config", {}),
            )
        )

    return PipelineConfig(
        name=raw["name"],
        version=raw["version"],
        description=raw.get("description", ""),
        stages=stages,
    )


def build_execution_order(stages: list[StageConfig]) -> list[StageConfig]:
    """使用 Kahn 算法对 Stage 做拓扑排序，返回有序执行列表。

    若存在环或未知依赖，抛出 DAGValidationError。
    """
    ids = {s.id for s in stages}
    stage_map = {s.id: s for s in stages}

    for s in stages:
        for dep in s.depends_on:
            if dep not in ids:
                raise DAGValidationError(
                    f"Stage '{s.id}' 依赖了不存在的 Stage '{dep}'"
                )

    in_degree: dict[str, int] = {s.id: 0 for s in stages}
    dependents: dict[str, list[str]] = defaultdict(list)

    for s in stages:
        for dep in s.depends_on:
            in_degree[s.id] += 1
            dependents[dep].append(s.id)

    queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
    ordered: list[StageConfig] = []

    while queue:
        sid = queue.popleft()
        ordered.append(stage_map[sid])
        for dependent in dependents[sid]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(stages):
        raise DAGValidationError("流水线 DAG 中存在环形依赖")

    return ordered


def load_stage_instance(stage_cfg: StageConfig, extra_kwargs: dict[str, Any]) -> "BaseStage":
    """动态 import 并实例化指定的 Stage 类。"""
    module = importlib.import_module(stage_cfg.module)
    cls = getattr(module, stage_cfg.class_name)
    return cls(config=stage_cfg.config, **extra_kwargs)
