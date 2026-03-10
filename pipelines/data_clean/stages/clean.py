"""数据清洗流水线各 Stage 实现。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from framework.models import JobContext
from framework.stage_runner import BaseStage

logger = logging.getLogger(__name__)


class DataLoadStage(BaseStage):
    """从 input_path 加载 JSONL 记录。"""

    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]:
        input_path = Path(ctx.input_path)
        fmt = self.config.get("file_format", "jsonl")
        files = sorted(input_path.glob(f"*.{fmt}")) if input_path.is_dir() else [input_path]

        for filepath in files:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            pass


class CleanStage(BaseStage):
    """去重、长度过滤与语种白名单过滤。"""

    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]:
        dedup: bool = self.config.get("dedup", True)
        min_len: int = self.config.get("min_length", 20)
        max_len: int = self.config.get("max_length", 8192)
        lang_filter: list[str] = self.config.get("lang_filter", [])

        seen: set[int] = set()

        for record in inputs:
            text = record.get("text", "")

            if len(text) < min_len or len(text) > max_len:
                continue

            if lang_filter and record.get("lang") not in lang_filter:
                continue

            if dedup:
                h = hash(text)
                if h in seen:
                    continue
                seen.add(h)

            yield record
