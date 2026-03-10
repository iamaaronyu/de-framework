"""Metadata collection and reporting to DataHub."""

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
    """Collects pipeline execution metadata and writes to DataHub."""

    def __init__(self, datahub_server: str, pipeline_name: str, pipeline_version: str) -> None:
        self._server = datahub_server
        self._pipeline_name = pipeline_name
        self._pipeline_version = pipeline_version
        self._metrics: dict[str, Any] = {}
        self._lineage_edges: list[LineageEdge] = []

    # ------------------------------------------------------------------
    # Metric accumulation
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
    # Flush to DataHub
    # ------------------------------------------------------------------

    def flush(self, job_id: str) -> None:
        """Write all collected metadata to DataHub."""
        try:
            self._write_run_event(job_id)
            self._write_lineage()
        except Exception as exc:
            # Metadata reporting failure must not fail the job
            logger.warning("Metadata flush failed for job %s: %s", job_id, exc)

    # ------------------------------------------------------------------
    # Internal
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
            logger.info("DataHub run event emitted for job %s", job_id)
        except ImportError:
            logger.warning(
                "datahub-ingestion not installed; skipping metadata emission. "
                "Install with: pip install datahub-ingestion"
            )
        except Exception as exc:
            logger.warning("Failed to emit DataHub run event: %s", exc)

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
            logger.info("DataHub lineage emitted: %d edges", len(self._lineage_edges))
        except ImportError:
            logger.warning(
                "datahub-ingestion not installed; skipping lineage emission."
            )
        except Exception as exc:
            logger.warning("Failed to emit DataHub lineage: %s", exc)

    @staticmethod
    def _path_to_urn(path: str) -> str:
        """Convert an OBS/S3 path to a DataHub dataset URN."""
        clean = path.replace("obs://", "").replace("s3://", "")
        return f"urn:li:dataset:(urn:li:dataPlatform:obs,{clean},PROD)"
