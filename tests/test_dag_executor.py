"""Tests for DAG topological execution."""

import pytest

from framework.dag_executor import DAGValidationError, build_execution_order
from framework.models import StageConfig


def _stage(sid: str, depends_on: list[str] | None = None) -> StageConfig:
    return StageConfig(
        id=sid,
        type="cpu_transform",
        module="fake.module",
        class_name="FakeStage",
        depends_on=depends_on or [],
    )


def test_linear_dag_order():
    stages = [_stage("c", ["b"]), _stage("b", ["a"]), _stage("a")]
    ordered = build_execution_order(stages)
    ids = [s.id for s in ordered]
    assert ids.index("a") < ids.index("b") < ids.index("c")


def test_no_dependencies():
    stages = [_stage("x"), _stage("y"), _stage("z")]
    ordered = build_execution_order(stages)
    assert {s.id for s in ordered} == {"x", "y", "z"}


def test_diamond_dag():
    # a -> b, a -> c, b -> d, c -> d
    stages = [
        _stage("a"),
        _stage("b", ["a"]),
        _stage("c", ["a"]),
        _stage("d", ["b", "c"]),
    ]
    ordered = build_execution_order(stages)
    ids = [s.id for s in ordered]
    assert ids.index("a") < ids.index("b")
    assert ids.index("a") < ids.index("c")
    assert ids.index("b") < ids.index("d")
    assert ids.index("c") < ids.index("d")


def test_cycle_raises():
    stages = [_stage("a", ["b"]), _stage("b", ["a"])]
    with pytest.raises(DAGValidationError, match="Cycle"):
        build_execution_order(stages)


def test_unknown_dependency_raises():
    stages = [_stage("a", ["nonexistent"])]
    with pytest.raises(DAGValidationError, match="unknown stage"):
        build_execution_order(stages)
