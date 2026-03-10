"""Tests for pipeline.yaml loading and stage config parsing."""

from __future__ import annotations

import textwrap
import tempfile
from pathlib import Path

import pytest

from framework.dag_executor import load_pipeline_config
from framework.models import PipelineConfig


SAMPLE_YAML = textwrap.dedent("""\
    name: "test-pipe"
    version: "v1.0.0"
    description: "test pipeline"
    stages:
      - id: load
        type: data_loader
        module: "test.module"
        class: "LoadStage"
        config:
          batch_size: 100
      - id: process
        type: cpu_transform
        module: "test.module"
        class: "ProcessStage"
        depends_on: [load]
        config:
          threshold: 0.5
""")


def test_load_pipeline_config_parses_correctly():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(SAMPLE_YAML)
        path = f.name

    try:
        cfg = load_pipeline_config(path)
        assert isinstance(cfg, PipelineConfig)
        assert cfg.name == "test-pipe"
        assert cfg.version == "v1.0.0"
        assert len(cfg.stages) == 2

        load_stage = cfg.stages[0]
        assert load_stage.id == "load"
        assert load_stage.type == "data_loader"
        assert load_stage.config["batch_size"] == 100
        assert load_stage.depends_on == []

        process_stage = cfg.stages[1]
        assert process_stage.depends_on == ["load"]
        assert process_stage.config["threshold"] == 0.5
    finally:
        Path(path).unlink()
