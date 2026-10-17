import hashlib
import os
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from zmodel.api.config import get_api_settings
from zmodel.api.jobs import JobQueueFullError, JobStatus, get_job_manager
from zmodel.api.model_store import resolve_model_bytes
from zmodel.api.schemas import (
    BenchmarkConfigSchema,
    BenchmarkJobResponse,
    BenchmarkListResponse,
    BenchmarkRunItem,
    BenchmarkStatusResponse,
    BenchmarkSubmitRequest,
)
from zmodel.core.benchmark import BenchmarkConfig, BenchmarkEngine
from zmodel.core.execution import (
    BackendUnavailableError,
    WorkerJobSpec,
    WorkerJobType,
    get_execution_backend,
)
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.db.repositories import BenchmarkRepository, WorkerJobRepository
from zmodel.db.session import session_scope

router = APIRouter(prefix="/api/v1/benchmarks", tags=["Reproducible Benchmarks"])


def _execute_benchmark_task(
    raw_bytes: bytes,
    filename: str,
    cfg_schema: BenchmarkConfigSchema,
    progress_callback=None,
):
    """Worker task executed in background thread pool."""
    if progress_callback:
        progress_callback(15, "Inspecting model signature and verifying tensors")

    summary = load_and_inspect_model(raw_bytes, filename=filename)

    b_config = BenchmarkConfig(
        execution_provider=cfg_schema.execution_provider,
        warmup_iterations=cfg_schema.warmup_iterations,
        measured_iterations=cfg_schema.measured_iterations,
        batch_size=cfg_schema.batch_size,
        repeated_rounds=cfg_schema.repeated_rounds,
        random_seed=cfg_schema.random_seed,
        dynamic_dimensions=dict(cfg_schema.dynamic_dimensions or {}),
        intra_op_num_threads=cfg_schema.intra_op_num_threads,
        inter_op_num_threads=cfg_schema.inter_op_num_threads,
    )

    if progress_callback:
        progress_callback(35, "Creating ONNX Runtime session and verifying correctness")

    report = BenchmarkEngine.run_benchmark(raw_bytes, summary, b_config)

    if progress_callback:
        progress_callback(85, "Persisting benchmark measurements to repository")

    try:
        repo = BenchmarkRepository()
        repo.save_benchmark_report(report)
    except Exception:
        pass  # Graceful tolerance if database is offline

    return report.to_dict()


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=BenchmarkJobResponse,
    summary="Submit Bounded Benchmark Job",
    description=(
        "Queue an empirical, bounded benchmark execution for a model. Measures genuine latency, "
        "throughput, memory delta, and correctness on the host runtime. Returns HTTP 202 with a job tracking ID."
    ),
)
def submit_benchmark(request: BenchmarkSubmitRequest) -> BenchmarkJobResponse:
    settings = get_api_settings()

    # Enforce configured execution bounds
    if request.config.measured_iterations > settings.max_benchmark_iterations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"measured_iterations ({request.config.measured_iterations}) exceeds configured system ceiling of {settings.max_benchmark_iterations}.",
        )
    if request.config.warmup_iterations > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="warmup_iterations cannot exceed 100.",
        )
    if request.config.batch_size > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="batch_size cannot exceed 128.",
        )

    # Safely resolve model bytes
    raw_bytes, filename = resolve_model_bytes(
        model_fingerprint=request.model_fingerprint,
        sample_model_name=request.sample_model_name,
    )

    # Validate configuration via dataclass logic
    try:
        b_config = BenchmarkConfig(
            execution_provider=request.config.execution_provider,
            warmup_iterations=request.config.warmup_iterations,
            measured_iterations=request.config.measured_iterations,
            batch_size=request.config.batch_size,
            repeated_rounds=request.config.repeated_rounds,
            random_seed=request.config.random_seed,
            dynamic_dimensions=dict(request.config.dynamic_dimensions or {}),
            intra_op_num_threads=request.config.intra_op_num_threads,
            inter_op_num_threads=request.config.inter_op_num_threads,
        )
        b_config.validate()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid benchmark configuration: {str(e)}",
        ) from e

    target_backend = (request.execution_backend or os.environ.get("EXECUTION_BACKEND", "local")).lower().strip()
    model_hash = hashlib.sha256(raw_bytes).hexdigest()

    if target_backend == "kubernetes":
        try:
            k8s_backend = get_execution_backend("kubernetes")
            job_id = f"job-k8s-{uuid.uuid4().hex[:12]}"
            spec = WorkerJobSpec(
                job_id=job_id,
                job_type=WorkerJobType.BENCHMARK,
                model_fingerprint=model_hash,
                model_filename=filename,
                resource_profile=request.resource_profile or "standard_cpu",
                parameters=request.config.model_dump(),
            )
            try:
                from zmodel.core.artifact_store import get_artifact_store
                store = get_artifact_store()
                store.store_artifact(raw_bytes, parent_fingerprint=model_hash, transformation_name="k8s_benchmark", original_filename=filename)
            except Exception:
                pass

            try:
                with session_scope() as s:
                    repo = WorkerJobRepository(session=s)
                    repo.create_job(spec, backend="kubernetes")
            except Exception:
                pass

            k8s_backend.submit_job(spec)
            return BenchmarkJobResponse(
                job_id=job_id,
                status="SUBMITTED",
                message="Benchmark execution job submitted to Kubernetes cluster.",
            )
        except BackendUnavailableError as bue:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Kubernetes backend unavailable: {bue}",
            )

    job_mgr = get_job_manager()
    try:
        job_id = job_mgr.submit_job(
            job_type="benchmark",
            task_fn=_execute_benchmark_task,
            raw_bytes=raw_bytes,
            filename=filename,
            cfg_schema=request.config,
        )
        try:
            with session_scope() as s:
                repo = WorkerJobRepository(session=s)
                spec = WorkerJobSpec(
                    job_id=job_id,
                    job_type=WorkerJobType.BENCHMARK,
                    model_fingerprint=model_hash,
                    model_filename=filename,
                    resource_profile=request.resource_profile or "standard_cpu",
                    parameters=request.config.model_dump(),
                )
                repo.create_job(spec, backend="local")
        except Exception:
            pass
    except JobQueueFullError as qfe:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(qfe),
        ) from qfe

    return BenchmarkJobResponse(
        job_id=job_id,
        status="QUEUED",
        message="Benchmark execution job queued in bounded worker pool.",
    )


@router.get(
    "/{benchmark_id}/status",
    response_model=BenchmarkStatusResponse,
    summary="Get Benchmark Job Status",
    description="Check the execution lifecycle state, progress, or failure message of a submitted benchmark job.",
)
def get_benchmark_status(benchmark_id: str) -> BenchmarkStatusResponse:
    job_mgr = get_job_manager()
    job = job_mgr.get_job(benchmark_id)

    if job:
        return BenchmarkStatusResponse(
            job_id=job.job_id,
            benchmark_id=job.resource_id,
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            progress_pct=job.progress_pct,
            progress_message=job.progress_message,
            error=job.error,
            result=job.result if job.status == JobStatus.COMPLETED else None,
        )

    # Check worker jobs repository (supports kubernetes backend and completed jobs)
    try:
        with session_scope() as s:
            w_repo = WorkerJobRepository(session=s)
            w_job = w_repo.get_job(benchmark_id)
            if w_job:
                d = w_job.to_dict()
                return BenchmarkStatusResponse(
                    job_id=d["id"],
                    benchmark_id=d["run_id"],
                    status=d["status"],
                    created_at=d["created_at"] or "",
                    started_at=d["started_at"],
                    completed_at=d["completed_at"],
                    progress_pct=100 if d["status"] == "SUCCEEDED" else 50,
                    progress_message=f"Job {d['status']} on {d['execution_backend']} backend",
                    error=d["error_message"],
                    result=d["result_json"],
                )
    except Exception:
        pass

    # Check if benchmark exists in persistent database
    try:
        repo = BenchmarkRepository()
        run = repo.get_benchmark_run(benchmark_id)
        if run:
            return BenchmarkStatusResponse(
                job_id=benchmark_id,
                benchmark_id=run.id,
                status="COMPLETED",
                created_at=run.created_at.isoformat() if run.created_at else "",
                started_at=run.created_at.isoformat() if run.created_at else None,
                completed_at=run.created_at.isoformat() if run.created_at else None,
                progress_pct=100,
                progress_message="Execution completed and persisted in database",
                error=None,
                result=run.to_dict(),
            )
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Benchmark job or run ID '{benchmark_id}' not found in active queue or database.",
    )


@router.get(
    "/{benchmark_id}",
    summary="Get Completed Benchmark Report",
    description="Retrieve the complete reproducible benchmark report including latency percentiles, throughput, and memory stats.",
)
def get_benchmark_detail(benchmark_id: str):
    # Check database first
    try:
        repo = BenchmarkRepository()
        run = repo.get_benchmark_run(benchmark_id)
        if run:
            return run.to_dict()
    except Exception:
        pass

    # Check active in-process job queue
    job_mgr = get_job_manager()
    job = job_mgr.get_job(benchmark_id)
    if job:
        if job.status == JobStatus.COMPLETED and job.result:
            return job.result
        elif job.status == JobStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Benchmark job failed: {job.error}",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Benchmark job is currently {job.status.value} ({job.progress_pct}% - {job.progress_message}). Query /{benchmark_id}/status for updates.",
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Benchmark '{benchmark_id}' not found.",
    )


@router.get(
    "",
    response_model=BenchmarkListResponse,
    summary="List Persisted Benchmark Runs",
    description="Retrieve paginated list of empirical benchmark records from PostgreSQL with filtering.",
)
def list_benchmarks(
    fingerprint: Optional[str] = Query(None, description="Filter by model SHA-256 fingerprint"),
    status: Optional[str] = Query(None, description="Filter by status (e.g. SUCCESS, FAILED)"),
    provider: Optional[str] = Query(None, description="Filter by execution provider"),
    limit: int = Query(20, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
) -> BenchmarkListResponse:
    try:
        repo = BenchmarkRepository()
        runs, total = repo.list_benchmark_runs(
            fingerprint=fingerprint,
            status=status,
            provider=provider,
            limit=limit,
            offset=offset,
        )
    except Exception as ex:
        return BenchmarkListResponse(runs=[], total=0, limit=limit, offset=offset)

    items = []
    for r in runs:
        cfg = r.configuration or {}
        items.append(
            BenchmarkRunItem(
                id=r.id,
                model_fingerprint=r.model_fingerprint,
                execution_provider=r.execution_provider,
                batch_size=cfg.get("batch_size", 1),
                intra_op_threads=cfg.get("intra_op_num_threads"),
                status=r.status,
                latency_median_ms=r.median_latency_ms,
                throughput_samples_sec=r.samples_per_second,
                peak_rss_mb=r.peak_rss_mb,
                correctness_status=r.correctness_status,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
        )

    return BenchmarkListResponse(
        runs=items,
        total=total,
        limit=limit,
        offset=offset,
    )
