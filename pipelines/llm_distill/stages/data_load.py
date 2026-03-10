"""Data loading stage for the LLM distillation pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from framework.models import JobContext
from framework.stage_runner import BaseStage

logger = logging.getLogger(__name__)


class DataLoadStage(BaseStage):
    """Reads JSONL records from input_path and yields them as dicts."""

    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]:
        input_path = Path(ctx.input_path)
        batch_size: int = self.config.get("batch_size", 1000)
        file_format: str = self.config.get("file_format", "jsonl")

        files = sorted(input_path.glob(f"*.{file_format}")) if input_path.is_dir() else [input_path]
        if not files:
            logger.warning("No %s files found at %s", file_format, input_path)
            return

        count = 0
        for filepath in files:
            logger.info("Loading file: %s", filepath)
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping malformed line in %s: %s", filepath, exc)
                        continue
                    yield record
                    count += 1
                    if count % batch_size == 0:
                        logger.debug("Loaded %d records so far", count)

        logger.info("DataLoadStage complete: %d records loaded from %s", count, input_path)
