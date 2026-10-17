import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from zmodel.api.jobs import JobManager, JobQueueFullError, JobStatus, sanitize_error_message
from zmodel.api.model_store import resolve_model_bytes
from zmodel.api.schemas import (
    GraphOptimizationExperimentResponse,
    GraphOptimizationStatusResponse,
    GraphOptimizationSubmitResponse,
    GraphTransformationCandidateResponse,
)
from zmodel.core.artifact_store import ArtifactSecurityError, get_artifact_store
from zmodel.core.execution import (
    BackendUnavailableError,
    WorkerJobSpec,
    WorkerJobType,
    get_execution_backend,
)
from zmodel.core.graph_optimizer import (
    GraphOptimizationEngine,
    GraphOptimizationReport,
    RecommendationStatus,
)
from zmodel.core.graph_transforms import TransformationType
from zmodel.db.repositories import (
    GraphOptimizationRepository,
    ModelRepository,
    WorkerJobRepository,
)
from zmodel.db.session import session_scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/graph-optimizations", tags=["Graph Optimization"])

# Global job manager reference initialized in app.py
_JOB_MANAGER: Optional[JobManager] = None


def set_job_manager(jm: JobManager) -> None:
    global _JOB_MANAGER
    _JOB_MANAGER = jm


def get_job_manager() -> JobManager:
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager(concurrency=2, max_queued=20)
    return _JOB_MANAGER


def _map_candidate_orm_to_response(
    experiment_id: str, cand: Any
) -> GraphTransformationCandidateResponse:
    """Format ORM candidate into API response schema."""
    artifact_id = getattr(cand, "artifact_id", None)
    rec_status = getattr(cand, "recommendation_status", "")
    download_url = None
    if artifact_id and rec_status != RecommendationStatus.REJECTED.value:
        download_url = f"/api/v1/graph-optimizations/{experiment_id}/candidates/{cand.candidate_id}/artifact"

    return GraphTransformationCandidateResponse(
        candidate_id=cand.candidate_id,
        transform_type=cand.transform_type,
        transform_name=cand.transform_name,
        tool_name=cand.tool_name,
        tool_version=cand.tool_version,
        state=cand.state,
        original_hash=cand.original_hash,
        transformed_hash=cand.transformed_hash,
        artifact_id=artifact_id,
        graph_changed=bool(cand.graph_changed),
        model_size_bytes_before=cand.model_size_bytes_before,
        model_size_bytes_after=cand.model_size_bytes_after,
        size_reduction_pct=cand.size_reduction_pct,
        diff_summary=cand.diff_summary,
        validation_report=cand.validation_report,
        compatibility_report=cand.compatibility_report,
        compatibility_delta=cand.compatibility_delta,
        benchmark_report=cand.benchmark_report,
        speedup_factor=cand.speedup_factor,
        latency_reduction_pct=cand.latency_reduction_pct,
        throughput_improvement_pct=cand.throughput_improvement_pct,
        recommendation_status=rec_status,
        recommendation_rationale=cand.recommendation_rationale,
        rejection_reason=cand.rejection_reason,
        warnings=cand.warnings or [],
        duration_ms=cand.duration_ms,
        artifact_download_url=download_url,
    )


def _map_experiment_orm_to_response(exp: Any) -> GraphOptimizationExperimentResponse:
    """Format ORM experiment into API response schema."""
    candidates = [
        _map_candidate_orm_to_response(exp.id, c) for c in (exp.candidates or [])
    ]
    return GraphOptimizationExperimentResponse(
        id=exp.id,
        model_name=exp.model_name,
        model_fingerprint=exp.model_fingerprint,
        target_profile_id=exp.target_profile_id,
        target_profile_name=exp.target_profile_name,
        validation_depth=exp.validation_depth,
        benchmark_enabled=bool(exp.benchmark_enabled),
        status=exp.status,
        best_candidate_id=exp.best_candidate_id,
        summary_counts=exp.summary_counts or {},
        overall_recommendation=exp.overall_recommendation,
        limitations=exp.limitations or [],
        error_message=exp.error_message,
        created_at=exp.created_at.isoformat() if hasattr(exp.created_at, "isoformat") else str(exp.created_at),
        completed_at=exp.completed_at.isoformat() if hasattr(exp.completed_at, "isoformat") and exp.completed_at else None,
        candidates=candidates,
    )


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GraphOptimizationSubmitResponse,
    summary="Submit ONNX Graph Optimization Experiment",
    description=(
        "Dispatches an asynchronous, bounded graph optimization experiment. "
        "Applies constant folding, operator fusion, and shape inference, evaluates correctness "
        "and compatibility, benchmarks outputs, and persists results."
    ),
)
async def submit_graph_optimization(
    file: Optional[UploadFile] = File(None, description="Direct ONNX model binary upload"),
    model_fingerprint: Optional[str] = Form(None, description="SHA-256 fingerprint of preloaded model"),
    sample_model_name: Optional[str] = Form(None, description="Sample model filename (e.g. linear_classifier.onnx)"),
    selected_transforms: Optional[str] = Form(
        None, description="JSON array of transform names: ['shape_inference', 'ort_basic', 'ort_extended', 'ort_all']"
    ),
    target_profile_id: Optional[str] = Form(None, description="Target architecture capability profile ID"),
    validation_depth: str = Form("FULL", description="Validation depth: FULL or STRUCTURAL_ONLY"),
    enable_benchmarking: bool = Form(True, description="Whether to benchmark eligible candidates"),
    atol: float = Form(1e-4, description="Absolute numerical tolerance threshold"),
    rtol: float = Form(1e-3, description="Relative numerical tolerance threshold"),
    test_seed: int = Form(42, description="Random seed for deterministic inputs"),
    execution_backend: Optional[str] = Form(None, description="Execution backend: 'local' or 'kubernetes'"),
    resource_profile: Optional[str] = Form(None, description="Compute resource profile: 'graph_opt', 'standard_cpu'"),
) -> GraphOptimizationSubmitResponse:
    # 1. Resolve model bytes safely
    model_bytes, filename = resolve_model_bytes(
        model_fingerprint=model_fingerprint,
        sample_model_name=sample_model_name,
        upload_file=file,
    )
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    target_backend = (execution_backend or os.environ.get("EXECUTION_BACKEND", "local")).lower().strip()

    # 2. Parse selected transforms
    transforms_list = [
        TransformationType.SHAPE_INFERENCE,
        TransformationType.ORT_BASIC,
        TransformationType.ORT_EXTENDED,
        TransformationType.ORT_ALL,
    ]
    if selected_transforms:
        try:
            parsed = json.loads(selected_transforms)
            if isinstance(parsed, list):
                transforms_list = [TransformationType(t) for t in parsed if t in [m.value for m in TransformationType]]
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid selected_transforms parameter. Expected JSON array of strings.",
            )

    experiment_id = f"gexp_{os.urandom(6).hex()}"

    if target_backend == "kubernetes":
        try:
            k8s_backend = get_execution_backend("kubernetes")
            job_id = f"job-k8s-{uuid.uuid4().hex[:12]}"
            spec = WorkerJobSpec(
                job_id=job_id,
                job_type=WorkerJobType.GRAPH_OPTIMIZATION,
                model_fingerprint=model_hash,
                model_filename=filename,
                experiment_id=experiment_id,
                resource_profile=resource_profile or "graph_opt",
                parameters={
                    "transformations": [t.value for t in transforms_list],
                    "target_profile_id": target_profile_id,
                    "validation_depth": validation_depth,
                    "enable_benchmarking": enable_benchmarking,
                    "atol": atol,
                    "rtol": rtol,
                    "test_seed": test_seed,
                },
            )
            try:
                store = get_artifact_store()
                store.store_artifact(model_bytes, parent_fingerprint=model_hash, transformation_name="k8s_graph_opt", original_filename=filename)
            except Exception:
                pass

            try:
                with session_scope() as s:
                    repo = WorkerJobRepository(session=s)
                    repo.create_job(spec, backend="kubernetes")
            except Exception:
                pass

            k8s_backend.submit_job(spec)
            return GraphOptimizationSubmitResponse(
                job_id=job_id,
                experiment_id=experiment_id,
                status="SUBMITTED",
                message="Graph optimization experiment submitted to Kubernetes cluster.",
                poll_url=f"/api/v1/graph-optimizations/{experiment_id}/status",
                results_url=f"/api/v1/graph-optimizations/{experiment_id}",
            )
        except BackendUnavailableError as bue:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Kubernetes backend unavailable: {bue}",
            )

    def _execute_task(progress_cb) -> Dict[str, Any]:
        engine = GraphOptimizationEngine(default_atol=atol, default_rtol=rtol, default_seed=test_seed)
        report = engine.run_optimization_experiment(
            model_bytes=model_bytes,
            original_filename=filename,
            selected_transforms=transforms_list,
            target_profile_id=target_profile_id,
            validation_depth=validation_depth,
            enable_benchmarking=enable_benchmarking,
            progress_callback=progress_cb,
        )
        # Force experiment ID
        report.experiment_id = experiment_id

        # Persist to database
        repo = GraphOptimizationRepository()
        repo.create_experiment(report)
        return report.to_dict()

    jm = get_job_manager()
    try:
        job_id = jm.submit_job(
            job_type="graph_optimization",
            task_fn=_execute_task,
            resource_id=experiment_id,
        )
        try:
            with session_scope() as s:
                repo = WorkerJobRepository(session=s)
                spec = WorkerJobSpec(
                    job_id=job_id,
                    job_type=WorkerJobType.GRAPH_OPTIMIZATION,
                    model_fingerprint=model_hash,
                    model_filename=filename,
                    experiment_id=experiment_id,
                    resource_profile=resource_profile or "graph_opt",
                    parameters={
                        "transformations": [t.value for t in transforms_list],
                        "target_profile_id": target_profile_id,
                        "validation_depth": validation_depth,
                        "enable_benchmarking": enable_benchmarking,
                        "atol": atol,
                        "rtol": rtol,
                        "test_seed": test_seed,
                    },
                )
                repo.create_job(spec, backend="local")
        except Exception:
            pass
    except JobQueueFullError as ex:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(ex),
        )

    return GraphOptimizationSubmitResponse(
        job_id=job_id,
        experiment_id=experiment_id,
        status="QUEUED",
        message="Graph optimization experiment accepted and scheduled for background execution.",
        poll_url=f"/api/v1/graph-optimizations/{experiment_id}/status",
        results_url=f"/api/v1/graph-optimizations/{experiment_id}",
    )


@router.get(
    "/{experiment_id}/status",
    response_model=GraphOptimizationStatusResponse,
    summary="Get Graph Optimization Experiment Status",
)
def get_graph_optimization_status(experiment_id: str) -> GraphOptimizationStatusResponse:
    jm = get_job_manager()
    job = jm.get_job_by_resource_id(experiment_id)

    if job:
        return GraphOptimizationStatusResponse(
            job_id=job.job_id,
            experiment_id=experiment_id,
            status=job.status.value,
            progress_pct=job.progress_pct,
            progress_message=job.progress_message,
            error=job.error,
        )

    # Check WorkerJobRepository (supports kubernetes backend and completed jobs)
    try:
        with session_scope() as s:
            w_repo = WorkerJobRepository(session=s)
            w_job = w_repo.get_job(experiment_id)
            if not w_job:
                # Search by experiment_id in records
                records, _ = w_repo.list_jobs(limit=10)
                for r in records:
                    if r.experiment_id == experiment_id:
                        w_job = r
                        break
            if w_job:
                d = w_job.to_dict()
                return GraphOptimizationStatusResponse(
                    job_id=d["id"],
                    experiment_id=experiment_id,
                    status=d["status"],
                    progress_pct=100 if d["status"] == "SUCCEEDED" else 50,
                    progress_message=f"Job {d['status']} on {d['execution_backend']} backend",
                    error=d["error_message"],
                )
    except Exception:
        pass

    # Check database repository if job is completed or restarted
    repo = GraphOptimizationRepository()
    exp = repo.get_experiment(experiment_id)
    if exp:
        return GraphOptimizationStatusResponse(
            job_id=None,
            experiment_id=experiment_id,
            status=exp.status,
            progress_pct=100 if exp.status in ("SUCCESS", "FAILED") else 50,
            progress_message=f"Experiment {exp.status.lower()}",
            error=exp.error_message,
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Graph optimization experiment '{experiment_id}' not found.",
    )


@router.get(
    "/{experiment_id}",
    response_model=GraphOptimizationExperimentResponse,
    summary="Get Complete Graph Optimization Experiment Report",
)
def get_graph_optimization_experiment(experiment_id: str) -> GraphOptimizationExperimentResponse:
    repo = GraphOptimizationRepository()
    exp = repo.get_experiment(experiment_id)

    if not exp:
        # Check if still running in job manager
        jm = get_job_manager()
        job = jm.get_job_by_resource_id(experiment_id)
        if job and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail=f"Experiment '{experiment_id}' is currently executing. Status: {job.status.value}.",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Graph optimization experiment '{experiment_id}' not found.",
        )

    return _map_experiment_orm_to_response(exp)


@router.get(
    "/{experiment_id}/candidates",
    response_model=List[GraphTransformationCandidateResponse],
    summary="List Graph Transformation Candidates",
)
def list_graph_transformation_candidates(
    experiment_id: str,
) -> List[GraphTransformationCandidateResponse]:
    repo = GraphOptimizationRepository()
    exp = repo.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found.",
        )
    return [_map_candidate_orm_to_response(exp.id, c) for c in (exp.candidates or [])]


@router.get(
    "/{experiment_id}/candidates/{candidate_id}",
    response_model=GraphTransformationCandidateResponse,
    summary="Get Specific Graph Transformation Candidate Detail",
)
def get_graph_transformation_candidate(
    experiment_id: str, candidate_id: str
) -> GraphTransformationCandidateResponse:
    repo = GraphOptimizationRepository()
    cand = repo.get_candidate(experiment_id, candidate_id)
    if not cand:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate '{candidate_id}' not found in experiment '{experiment_id}'.",
        )
    return _map_candidate_orm_to_response(experiment_id, cand)


@router.get(
    "/{experiment_id}/candidates/{candidate_id}/artifact",
    summary="Controlled Transformed ONNX Model Artifact Download",
    description=(
        "Safely streams the validated transformed ONNX model binary. "
        "Enforces strict path-traversal prevention, SHA-256 fingerprint integrity check, "
        "and rejects candidates that failed structural or numerical validation."
    ),
)
def download_transformed_artifact(
    experiment_id: str,
    candidate_id: str,
) -> Response:
    repo = GraphOptimizationRepository()
    cand = repo.get_candidate(experiment_id, candidate_id)

    if not cand:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate '{candidate_id}' not found in experiment '{experiment_id}'.",
        )

    # Validate eligibility for download
    if cand.recommendation_status == RecommendationStatus.REJECTED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Candidate '{candidate_id}' was REJECTED during validation "
                f"({cand.rejection_reason or 'Validation failed'}). Rejected artifacts are not available for deployment."
            ),
        )

    if not cand.artifact_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No artifact stored for candidate '{candidate_id}'.",
        )

    store = get_artifact_store()
    try:
        raw_bytes, info = store.read_artifact_bytes(cand.artifact_id, expected_hash=cand.transformed_hash)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact file for candidate '{candidate_id}' has expired or is missing.",
        )
    except ArtifactSecurityError as ex:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Artifact access denied: {sanitize_error_message(ex)}",
        )
    except ValueError as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Artifact integrity error: {str(ex)}",
        )

    # Sanitize download filename
    filename = re.sub(r"[^A-Za-z0-9_\-\.]", "_", info.suggested_filename)

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": "application/octet-stream",
        "X-Artifact-ID": cand.artifact_id,
        "X-Artifact-Fingerprint": info.sha256_hash,
        "X-Validation-Status": cand.state,
    }

    return Response(
        content=raw_bytes,
        media_type="application/octet-stream",
        headers=headers,
    )
