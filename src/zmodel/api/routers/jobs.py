"""API routes for inspecting, monitoring, and cancelling worker jobs across execution backends."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from zmodel.api.schemas import (
    BackendHealthResponse,
    WorkerJobCancelResponse,
    WorkerJobListResponse,
    WorkerJobResponse,
)
from zmodel.core.execution import (
    BackendUnavailableError,
    get_execution_backend,
)
from zmodel.db.repositories import WorkerJobRepository
from zmodel.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["Worker Jobs"])


@router.get("/health", response_model=BackendHealthResponse, summary="Inspect execution backend health")
def check_backend_health(backend: Optional[str] = Query(None, description="Backend to inspect ('local' or 'kubernetes')")):
    """Inspect availability and connectivity of the execution backend."""
    try:
        backend_inst = get_execution_backend(name=backend)
        health = backend_inst.check_health()
        return BackendHealthResponse(
            backend_name=health.backend_name,
            is_available=health.is_available,
            details=health.details,
            error_message=health.error_message,
        )
    except BackendUnavailableError as ex:
        return BackendHealthResponse(
            backend_name=backend or "kubernetes",
            is_available=False,
            error_message=str(ex),
        )
    except Exception as ex:
        return BackendHealthResponse(
            backend_name=backend or "unknown",
            is_available=False,
            error_message=str(ex),
        )


@router.get("", response_model=WorkerJobListResponse, summary="List worker jobs")
def list_worker_jobs(
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    backend: Optional[str] = Query(None, description="Filter by execution backend"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by job state"),
    limit: int = Query(50, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db=Depends(get_db),
):
    """List historical and active worker jobs with pagination and status filters."""
    repo = WorkerJobRepository(session=db)
    records, total = repo.list_jobs(
        job_type=job_type,
        backend=backend,
        status=status_filter,
        limit=limit,
        offset=offset,
    )
    items = []
    for r in records:
        d = r.to_dict()
        items.append(
            WorkerJobResponse(
                id=d["id"],
                job_type=d["job_type"],
                execution_backend=d["execution_backend"],
                model_fingerprint=d["model_fingerprint"],
                experiment_id=d["experiment_id"],
                run_id=d["run_id"],
                status=d["status"],
                resource_profile=d["resource_profile"],
                k8s_namespace=d["k8s_namespace"],
                k8s_job_name=d["k8s_job_name"],
                k8s_pod_name=d["k8s_pod_name"],
                retry_count=d["retry_count"],
                max_retries=d["max_retries"],
                worker_image=d["worker_image"],
                queue_delay_seconds=d["queue_delay_seconds"],
                startup_overhead_seconds=d["startup_overhead_seconds"],
                warmup_overhead_seconds=d["warmup_overhead_seconds"],
                inference_duration_seconds=d["inference_duration_seconds"],
                total_duration_seconds=d["total_duration_seconds"],
                result=d["result_json"],
                output_artifact_ids=d["output_artifact_ids"],
                error_category=d["error_category"],
                error_message=d["error_message"],
                created_at=d["created_at"],
                started_at=d["started_at"],
                completed_at=d["completed_at"],
            )
        )
    return WorkerJobListResponse(jobs=items, total=total, limit=limit, offset=offset)


@router.get("/{job_id}", response_model=WorkerJobResponse, summary="Retrieve worker job status and result")
def get_worker_job(job_id: str, db=Depends(get_db)):
    """Retrieve full lifecycle status, timing breakdown, and metrics for a specific worker job."""
    repo = WorkerJobRepository(session=db)
    record = repo.get_job(job_id)

    # If persisted in DB
    if record:
        d = record.to_dict()
        # If currently running or submitted on a backend, query backend for live status
        if d["status"] in ("QUEUED", "SUBMITTED", "RUNNING"):
            try:
                backend_inst = get_execution_backend(name=d["execution_backend"])
                live_state = backend_inst.get_job_status(job_id)
                if live_state.value != d["status"]:
                    d["status"] = live_state.value
                    repo.update_job_status(job_id, status=live_state.value)
            except Exception:
                pass

        return WorkerJobResponse(
            id=d["id"],
            job_type=d["job_type"],
            execution_backend=d["execution_backend"],
            model_fingerprint=d["model_fingerprint"],
            experiment_id=d["experiment_id"],
            run_id=d["run_id"],
            status=d["status"],
            resource_profile=d["resource_profile"],
            k8s_namespace=d["k8s_namespace"],
            k8s_job_name=d["k8s_job_name"],
            k8s_pod_name=d["k8s_pod_name"],
            retry_count=d["retry_count"],
            max_retries=d["max_retries"],
            worker_image=d["worker_image"],
            queue_delay_seconds=d["queue_delay_seconds"],
            startup_overhead_seconds=d["startup_overhead_seconds"],
            warmup_overhead_seconds=d["warmup_overhead_seconds"],
            inference_duration_seconds=d["inference_duration_seconds"],
            total_duration_seconds=d["total_duration_seconds"],
            result=d["result_json"],
            output_artifact_ids=d["output_artifact_ids"],
            error_category=d["error_category"],
            error_message=d["error_message"],
            created_at=d["created_at"],
            started_at=d["started_at"],
            completed_at=d["completed_at"],
        )

    # Check active backend if not yet in DB or ephemeral
    try:
        backend_inst = get_execution_backend()
        job_result = backend_inst.get_job_result(job_id)
        if job_result:
            return WorkerJobResponse(
                id=job_result.job_id,
                job_type="benchmark",
                execution_backend=backend_inst.name,
                model_fingerprint=job_result.model_fingerprint,
                status=job_result.status.value,
                resource_profile="standard_cpu",
                k8s_namespace=None,
                k8s_job_name=None,
                k8s_pod_name=job_result.pod_name,
                retry_count=0,
                max_retries=1,
                worker_image=job_result.worker_image,
                queue_delay_seconds=job_result.timing.queue_delay_seconds,
                startup_overhead_seconds=job_result.timing.startup_overhead_seconds,
                warmup_overhead_seconds=job_result.timing.warmup_overhead_seconds,
                inference_duration_seconds=job_result.timing.inference_duration_seconds,
                total_duration_seconds=job_result.timing.total_duration_seconds,
                result=job_result.result_data,
                output_artifact_ids=job_result.output_artifact_ids,
                error_category=job_result.error_category,
                error_message=job_result.error_message,
                created_at=job_result.started_at,
                started_at=job_result.started_at,
                completed_at=job_result.completed_at,
            )
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Worker job '{job_id}' not found.",
    )


@router.post("/{job_id}/cancel", response_model=WorkerJobCancelResponse, summary="Cancel an active worker job")
def cancel_worker_job(job_id: str, db=Depends(get_db)):
    """Request cancellation and cleanup of an active or queued worker job."""
    repo = WorkerJobRepository(session=db)
    record = repo.get_job(job_id)
    if not record:
        # Check active backend if not yet in database
        try:
            backend_inst = get_execution_backend()
            backend_cancelled = backend_inst.cancel_job(job_id)
            if backend_cancelled:
                return WorkerJobCancelResponse(
                    job_id=job_id,
                    cancelled=True,
                    message="Job cancelled in execution backend.",
                )
        except Exception:
            pass
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker job '{job_id}' not found.")

    backend_name = record.execution_backend if record else None
    backend_success = False

    try:
        backend_inst = get_execution_backend(name=backend_name)
        backend_success = backend_inst.cancel_job(job_id)
    except Exception as ex:
        logger.warning(f"Backend cancellation warning for {job_id}: {ex}")

    repo.update_job_status(job_id, status="CANCELLED", error_category="CANCELLED", error_message="Cancelled by user request")
    return WorkerJobCancelResponse(
        job_id=job_id,
        cancelled=True,
        message="Cancellation signal transmitted to execution backend." if backend_success else "Job marked cancelled in database.",
    )
