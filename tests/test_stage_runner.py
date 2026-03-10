"""Tests for stage execution and retry logic."""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from framework.models import JobContext, StageStatus
from framework.stage_runner import BaseStage, StageRunner


def _ctx() -> JobContext:
    return JobContext(
        job_id="test-job",
        pipeline_name="test-pipe",
        pipeline_version="v0.0.1",
        input_path="/tmp/in",
        output_path="/tmp/out",
    )


class PassthroughStage(BaseStage):
    def process(self, inputs: Iterator[dict], ctx: JobContext) -> Iterator[dict]:
        yield from inputs


class FilterStage(BaseStage):
    def process(self, inputs: Iterator[dict], ctx: JobContext) -> Iterator[dict]:
        for r in inputs:
            if r.get("keep"):
                yield r


class AlwaysFailStage(BaseStage):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.call_count = 0

    def process(self, inputs: Iterator[dict], ctx: JobContext) -> Iterator[dict]:
        self.call_count += 1
        raise RuntimeError("deliberate failure")
        yield  # make it a generator


class FailThenSucceedStage(BaseStage):
    def __init__(self, config: dict[str, Any], fail_times: int = 1) -> None:
        super().__init__(config)
        self._fail_times = fail_times
        self._calls = 0

    def process(self, inputs: Iterator[dict], ctx: JobContext) -> Iterator[dict]:
        self._calls += 1
        if self._calls <= self._fail_times:
            raise RuntimeError("transient error")
        yield from inputs


def test_passthrough_stage():
    data = [{"id": i} for i in range(5)]
    runner = StageRunner(max_retries=0)
    result, output = runner.run(PassthroughStage({}), "pass", data, _ctx())
    assert result.status == StageStatus.SUCCESS
    assert result.records_in == 5
    assert result.records_out == 5
    assert result.records_filtered == 0
    assert output == data


def test_filter_stage():
    data = [{"id": 0, "keep": True}, {"id": 1, "keep": False}, {"id": 2, "keep": True}]
    runner = StageRunner(max_retries=0)
    result, output = runner.run(FilterStage({}), "filter", data, _ctx())
    assert result.status == StageStatus.SUCCESS
    assert result.records_out == 2
    assert result.records_filtered == 1


def test_always_fail_returns_failed_status():
    stage = AlwaysFailStage({})
    runner = StageRunner(max_retries=0)
    result, output = runner.run(stage, "fail", [{"id": 0}], _ctx())
    assert result.status == StageStatus.FAILED
    assert output == []
    assert "deliberate failure" in result.error


def test_retry_succeeds_on_second_attempt():
    stage = FailThenSucceedStage({}, fail_times=1)
    # StageRunner max_retries=1 means up to 2 attempts total
    runner = StageRunner(max_retries=1)
    # Patch sleep to avoid real delay in tests
    import framework.stage_runner as sr_mod
    original_sleep = __import__("time").sleep

    import time
    time.sleep = lambda _: None
    try:
        result, output = runner.run(stage, "retry-stage", [{"id": 0}], _ctx())
    finally:
        time.sleep = original_sleep

    assert result.status == StageStatus.SUCCESS
    assert stage._calls == 2
