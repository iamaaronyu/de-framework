"""LLM 蒸馏流水线 — 数据加载 Stage。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from framework.models import JobContext
from framework.stage_runner import BaseStage

logger = logging.getLogger(__name__)


class DataLoadStage(BaseStage):
    """从 input_path 读取 JSONL 记录，逐条 yield 为字典。"""

    def process(
        self, inputs: Iterator[dict[str, Any]], ctx: JobContext
    ) -> Iterator[dict[str, Any]]:
        input_path = Path(ctx.input_path)
        batch_size: int = self.config.get("batch_size", 1000)
        file_format: str = self.config.get("file_format", "jsonl")

        files = sorted(input_path.glob(f"*.{file_format}")) if input_path.is_dir() else [input_path]
        if not files:
            logger.warning("在 %s 未找到 %s 文件", input_path, file_format)
            return

        count = 0
        for filepath in files:
            logger.info("正在加载文件：%s", filepath)
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("跳过 %s 中格式错误的行：%s", filepath, exc)
                        continue
                    yield record
                    count += 1
                    if count % batch_size == 0:
                        logger.debug("已加载 %d 条记录", count)

        logger.info("DataLoadStage 完成：从 %s 共加载 %d 条记录", input_path, count)
