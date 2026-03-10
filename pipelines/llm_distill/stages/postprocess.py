"""Postprocessing and output writing stage for the LLM distillation pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from framework.models import JobContext
from framework.stage_runner import BaseStage

logger = logging.getLogger(__name__)


class PostprocessStage(BaseStage):
    """Filters low-quality outputs and writes distilled records to output_path."""

    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]:
        quality_threshold: float = self.config.get("quality_threshold", 0.8)
        min_tokens: int = self.config.get("min_output_tokens", 10)
        output_format: str = self.config.get("output_format", "jsonl")

        output_path = Path(ctx.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / f"distilled_{ctx.job_id}.{output_format}"

        written = 0
        filtered = 0

        with open(out_file, "w", encoding="utf-8") as f:
            for record in inputs:
                if not record.get("inference_ok", False):
                    filtered += 1
                    continue

                npu_output = record.get("npu_output") or {}
                quality_score = float(npu_output.get("quality_score", 1.0))
                output_text = str(npu_output.get("output", ""))

                if quality_score < quality_threshold:
                    filtered += 1
                    continue

                token_count = len(output_text.split())
                if token_count < min_tokens:
                    filtered += 1
                    continue

                distilled = {
                    "id": record.get("id"),
                    "input": record.get("text", ""),
                    "output": output_text,
                    "quality_score": quality_score,
                    "pipeline": ctx.pipeline_name,
                    "pipeline_version": ctx.pipeline_version,
                    "job_id": ctx.job_id,
                }
                f.write(json.dumps(distilled, ensure_ascii=False) + "\n")
                yield distilled
                written += 1

        logger.info(
            "PostprocessStage complete: written=%d filtered=%d to %s",
            written,
            filtered,
            out_file,
        )
