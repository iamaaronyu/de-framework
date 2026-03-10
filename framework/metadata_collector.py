"""流水线执行元数据采集与上报至 DataHub。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LineageEdge:
    upstream_urn: str
    downstream_urn: str
    transformation: str = "distillation"


class MetadataCollector:
    """采集流水线执行元数据并写入 DataHub。"""

    def __init__(self, datahub_server: str, pipeline_name: str, pipeline_version: str) -> None:
        self._server = datahub_server
        self._pipeline_name = pipeline_name
        self._pipeline_version = pipeline_version
        self._metrics: dict[str, Any] = {}
        self._lineage_edges: list[LineageEdge] = []

    # ------------------------------------------------------------------
    # 指标累积
    # ------------------------------------------------------------------

    def record_metric(self, key: str, value: Any) -> None:
        self._metrics[key] = value

    def add_lineage(self, upstream_path: str, downstream_path: str) -> None:
        self._lineage_edges.append(
            LineageEdge(
                upstream_urn=self._path_to_urn(upstream_path),
                downstream_urn=self._path_to_urn(downstream_path),
            )
        )

    # ------------------------------------------------------------------
    # 写入 DataHub
    # ------------------------------------------------------------------

    def flush(self, job_id: str) -> None:
        """将所有已采集的元数据批量写入 DataHub。"""
        try:
            self._write_run_event(job_id)
            self._write_lineage()
        except Exception as exc:
            # 元数据上报失败不能导致任务失败
            logger.warning("任务 %s 元数据写入失败：%s", job_id, exc)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _write_run_event(self, job_id: str) -> None:
        try:
            from datahub.emitter.rest_emitter import DatahubRestEmitter
            from datahub.metadata.schema_classes import (
                DataProcessInstanceRunEventClass,
                DataProcessInstanceRunResultClass,
            )

            emitter = DatahubRestEmitter(gms_server=self._server)
            run_event = DataProcessInstanceRunEventClass(
                timestampMillis=int(time.time() * 1000),
                status=DataProcessInstanceRunResultClass.SUCCESS,
                result=DataProcessInstanceRunResultClass(
                    type="SUCCESS",
                    nativeResultType="pipeline_run",
                ),
            )
            emitter.emit(run_event)
            logger.info("DataHub 运行事件已上报，任务 %s", job_id)
        except ImportError:
            logger.warning(
                "datahub-ingestion 未安装，跳过元数据上报。"
                "请执行：pip install datahub-ingestion"
            )
        except Exception as exc:
            logger.warning("DataHub 运行事件上报失败：%s", exc)

    def _write_lineage(self) -> None:
        if not self._lineage_edges:
            return
        try:
            from datahub.emitter.rest_emitter import DatahubRestEmitter
            from datahub.metadata.schema_classes import UpstreamLineageClass, UpstreamClass

            emitter = DatahubRestEmitter(gms_server=self._server)
            for edge in self._lineage_edges:
                upstream = UpstreamClass(
                    dataset=edge.upstream_urn,
                    type=edge.transformation.upper(),
                )
                lineage = UpstreamLineageClass(upstreams=[upstream])
                emitter.emit_mce(
                    {
                        "entityType": "dataset",
                        "entityUrn": edge.downstream_urn,
                        "aspectName": "upstreamLineage",
                        "aspect": lineage,
                    }
                )
            logger.info("DataHub 血缘关系已上报：%d 条边", len(self._lineage_edges))
        except ImportError:
            logger.warning("datahub-ingestion 未安装，跳过血缘上报。")
        except Exception as exc:
            logger.warning("DataHub 血缘上报失败：%s", exc)

    @staticmethod
    def _path_to_urn(path: str) -> str:
        """将 OBS/S3 路径转换为 DataHub dataset URN 格式。"""
        clean = path.replace("obs://", "").replace("s3://", "")
        return f"urn:li:dataset:(urn:li:dataPlatform:obs,{clean},PROD)"
