"""Task Bootstrap — entry point for the execution framework.

Execution flow:
  1. Parse job parameters
  2. Download versioned pipeline code via code-download service
  3. Load pipeline.yaml from downloaded code
  4. Build DAG execution order
  5. Execute each stage sequentially, reporting progress
  6. Flush metadata to DataHub
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from framework.dag_executor import build_execution_order, load_pipeline_config, load_stage_instance
from framework.metadata_collector import MetadataCollector
from framework.models import JobContext, JobResult, JobStatus, StageStatus
from framework.npu_invoker import NPUInvoker
from framework.progress_reporter import ProgressReporter
from framework.stage_runner import StageRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variable keys (injected by Yundao / K8s)
ENV_YUNDAO_URL = "YUNDAO_URL"
ENV_NPU_URL = "NPU_SERVICE_URL"
ENV_CODE_DOWNLOAD_URL = "CODE_DOWNLOAD_URL"
ENV_DATAHUB_URL = "DATAHUB_GMS_URL"


def _download_pipeline_code(
    download_url: str,
    pipeline_name: str,
    version: str,
    work_dir: Path,
) -> Path:
    """Download and extract pipeline code; return extraction path."""
    import httpx

    dest = work_dir / pipeline_name / version
    if dest.exists():
        logger.info("Cache hit: %s %s already at %s", pipeline_name, version, dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    url = (
        f"{download_url.rstrip('/')}/code/download"
        f"?pipeline={pipeline_name}&version={version}"
    )
    logger.info("Downloading pipeline code from %s", url)
    with httpx.stream("GET", url, timeout=300) as resp:
        resp.raise_for_status()
        archive = dest / "code.tar.gz"
        with open(archive, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=65536):
                f.write(chunk)

    import tarfile

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest)
    archive.unlink()
    logger.info("Pipeline code extracted to %s", dest)
    return dest


def run_job(
    job_id: str,
    pipeline_name: str,
    pipeline_version: str,
    input_path: str,
    output_path: str,
    dev_mode: bool = False,
) -> JobResult:
    yundao_url = os.environ.get(ENV_YUNDAO_URL, "http://yundao-platform/")
    npu_url = os.environ.get(ENV_NPU_URL, "http://npu-service:8080/")
    download_url = os.environ.get(ENV_CODE_DOWNLOAD_URL, "http://code-download-service:8081/")
    datahub_url = os.environ.get(ENV_DATAHUB_URL, "http://datahub-gms:8080/")

    work_dir = Path(os.environ.get("PIPELINE_WORK_DIR", "/tmp/de-pipelines"))

    ctx = JobContext(
        job_id=job_id,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        input_path=input_path,
        output_path=output_path,
    )

    with ProgressReporter(yundao_url, job_id) as reporter:
        reporter.report_job_start()
        metadata = MetadataCollector(datahub_url, pipeline_name, pipeline_version)
        metadata.add_lineage(input_path, output_path)

        t_start = time.monotonic()

        try:
            # Step 1: download code (skip in dev_mode — use local source tree)
            if dev_mode:
                code_dir = Path(__file__).parent.parent
                logger.info("Dev mode: using local source at %s", code_dir)
            else:
                code_dir = _download_pipeline_code(
                    download_url, pipeline_name, pipeline_version, work_dir
                )
                sys.path.insert(0, str(code_dir))

            # Step 2: load and validate pipeline config
            pipeline_yaml = code_dir / "pipeline.yaml"
            pipeline_cfg = load_pipeline_config(str(pipeline_yaml))

            # Step 3: topological sort
            ordered_stages = build_execution_order(pipeline_cfg.stages)

            # Step 4: execute stages
            runner = StageRunner(max_retries=2)
            npu = NPUInvoker(npu_url)
            stage_results = []
            current_data: list[dict] = []

            for stage_cfg in ordered_stages:
                reporter.report_stage_start(stage_cfg.id)

                instance = load_stage_instance(
                    stage_cfg,
                    extra_kwargs={"npu_invoker": npu} if stage_cfg.type == "npu_inference" else {},
                )
                result, current_data = runner.run(instance, stage_cfg.id, current_data, ctx)
                stage_results.append(result)
                reporter.report_stage_result(result)

                if result.status == StageStatus.FAILED:
                    logger.error("Stage '%s' failed; aborting job", stage_cfg.id)
                    break

            npu.close()

            overall_status = (
                JobStatus.SUCCESS
                if all(r.status == StageStatus.SUCCESS for r in stage_results)
                else JobStatus.FAILED
            )

        except Exception as exc:
            logger.exception("Unhandled error in job %s", job_id)
            overall_status = JobStatus.FAILED
            stage_results = []
            error_msg = str(exc)
            reporter.report_job_complete(overall_status, error=error_msg)
            return JobResult(
                job_id=job_id,
                status=overall_status,
                error=error_msg,
                duration_seconds=round(time.monotonic() - t_start, 3),
            )

        duration = round(time.monotonic() - t_start, 3)
        reporter.report_job_complete(overall_status)

        metadata.record_metric("total_duration_seconds", duration)
        metadata.record_metric(
            "total_records_out", sum(r.records_out for r in stage_results)
        )
        metadata.flush(job_id)

        return JobResult(
            job_id=job_id,
            status=overall_status,
            stage_results=stage_results,
            total_records_in=stage_results[0].records_in if stage_results else 0,
            total_records_out=stage_results[-1].records_out if stage_results else 0,
            duration_seconds=duration,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DE Framework Task Bootstrap")
    parser.add_argument("--job-id", default="dev-local-001")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--dev-mode", action="store_true")
    args = parser.parse_args()

    result = run_job(
        job_id=args.job_id,
        pipeline_name=args.pipeline,
        pipeline_version=args.version,
        input_path=args.input_path,
        output_path=args.output_path,
        dev_mode=args.dev_mode,
    )

    if result.status == JobStatus.SUCCESS:
        logger.info(
            "Job %s completed successfully in %.2fs. Records out: %d",
            result.job_id,
            result.duration_seconds,
            result.total_records_out,
        )
        sys.exit(0)
    else:
        logger.error("Job %s failed: %s", result.job_id, result.error)
        sys.exit(1)


if __name__ == "__main__":
    main()
