"""DAG parsing and topological execution engine."""

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
    """Parse a pipeline.yaml file into a PipelineConfig."""
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
    """Return stages in topological order (Kahn's algorithm).

    Raises DAGValidationError if the graph has a cycle or unknown dependency.
    """
    ids = {s.id for s in stages}
    stage_map = {s.id: s for s in stages}

    for s in stages:
        for dep in s.depends_on:
            if dep not in ids:
                raise DAGValidationError(
                    f"Stage '{s.id}' depends on unknown stage '{dep}'"
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
        raise DAGValidationError("Cycle detected in pipeline DAG")

    return ordered


def load_stage_instance(stage_cfg: StageConfig, extra_kwargs: dict[str, Any]) -> "BaseStage":
    """Dynamically import and instantiate a Stage class."""
    module = importlib.import_module(stage_cfg.module)
    cls = getattr(module, stage_cfg.class_name)
    return cls(config=stage_cfg.config, **extra_kwargs)
