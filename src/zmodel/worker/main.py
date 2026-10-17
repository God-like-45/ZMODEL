"""Container entrypoint for Kubernetes and standalone worker jobs."""

import argparse
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from zmodel.core.execution.schemas import (
    TimingBreakdown,
    WorkerJobResult,
    WorkerJobSpec,
    WorkerJobState,
)
from zmodel.worker.runner import execute_worker_task

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("zmodel-worker")

_TERMINATION_REQUESTED = False


def _signal_handler(signum: int, frame: Any) -> None:
    global _TERMINATION_REQUESTED
    logger.warning(f"Termination signal {signum} received. Cleaning up worker...")
    _TERMINATION_REQUESTED = True


def parse_job_spec(args: argparse.Namespace) -> WorkerJobSpec:
    """Resolve and validate WorkerJobSpec from CLI flag, file, or environment."""
    raw_json: Optional[str] = None

    if args.spec_json:
        raw_json = args.spec_json
    elif args.spec_file and os.path.exists(args.spec_file):
        with open(args.spec_file, "r", encoding="utf-8") as f:
            raw_json = f.read()
    elif os.environ.get("ZMODEL_JOB_SPEC"):
        raw_json = os.environ["ZMODEL_JOB_SPEC"]
    elif os.path.exists("/etc/zmodel/job-spec.json"):
        with open("/etc/zmodel/job-spec.json", "r", encoding="utf-8") as f:
            raw_json = f.read()

    if not raw_json:
        raise ValueError("No worker job specification provided via --spec-json, --spec-file, or ZMODEL_JOB_SPEC.")

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as ex:
        raise ValueError(f"Malformed JSON in worker job specification: {ex}")

    return WorkerJobSpec.from_dict(data)


def write_job_result(result: WorkerJobResult, output_path: Optional[str] = None) -> None:
    """Serialize and output WorkerJobResult."""
    res_dict = result.to_dict()
    res_json = json.dumps(res_dict, indent=2)

    # 1. Output file if specified
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(res_json)
        logger.info(f"Result written to {output_path}")

    # 2. Default pod result path if mounted
    pod_result_path = "/etc/zmodel/job-result.json"
    if os.path.exists(os.path.dirname(pod_result_path)):
        try:
            with open(pod_result_path, "w", encoding="utf-8") as f:
                f.write(res_json)
        except Exception as ex:
            logger.debug(f"Could not write to {pod_result_path}: {ex}")

    # 3. Always print structured summary to stdout
    print(f"--- WORKER_JOB_RESULT_START ---")
    print(res_json)
    print(f"--- WORKER_JOB_RESULT_END ---")


def main() -> int:
    """Worker entrypoint."""
    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(description="Z-ModelPilot Containerized Worker")
    parser.add_argument("--spec-json", type=str, help="Inline JSON WorkerJobSpec string")
    parser.add_argument("--spec-file", type=str, help="Path to JSON file containing WorkerJobSpec")
    parser.add_argument("--output-file", type=str, help="Optional output path to write WorkerJobResult JSON")
    args = parser.parse_args()

    started_at = datetime.now(timezone.utc).isoformat()
    spec: Optional[WorkerJobSpec] = None

    try:
        spec = parse_job_spec(args)
        logger.info(f"Worker initiated for job {spec.job_id} (type={spec.job_type.value}, model={spec.model_fingerprint[:12]})")

        # Execute
        result_data, out_artifacts, timing = execute_worker_task(spec)

        if _TERMINATION_REQUESTED:
            logger.warning("Execution was interrupted by termination signal.")
            res = WorkerJobResult(
                job_id=spec.job_id,
                status=WorkerJobState.CANCELLED,
                model_fingerprint=spec.model_fingerprint,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                timing=timing,
                error_category="CANCELLED",
                error_message="Worker terminated by external signal (SIGTERM/SIGINT)",
            )
            write_job_result(res, args.output_file)
            return 130

        res = WorkerJobResult(
            job_id=spec.job_id,
            status=WorkerJobState.SUCCEEDED,
            model_fingerprint=spec.model_fingerprint,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            timing=timing,
            worker_image=os.environ.get("WORKER_IMAGE", "zmodel-worker:latest"),
            worker_node=os.environ.get("NODE_NAME", "k8s-worker"),
            pod_name=os.environ.get("POD_NAME", None),
            execution_environment={
                "cpu_count": os.cpu_count(),
                "k8s_node": os.environ.get("NODE_NAME"),
            },
            result_data=result_data,
            output_artifact_ids=out_artifacts,
        )
        write_job_result(res, args.output_file)
        logger.info(f"Job {spec.job_id} successfully completed in {timing.total_duration_seconds:.2f}s")
        return 0

    except Exception as ex:
        logger.exception(f"Worker execution failed: {ex}")
        res = WorkerJobResult(
            job_id=spec.job_id if spec else "unknown",
            status=WorkerJobState.FAILED,
            model_fingerprint=spec.model_fingerprint if spec else "unknown",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            timing=TimingBreakdown(),
            error_category="EXECUTION_ERROR",
            error_message=str(ex),
        )
        write_job_result(res, args.output_file)
        return 1


if __name__ == "__main__":
    sys.exit(main())
