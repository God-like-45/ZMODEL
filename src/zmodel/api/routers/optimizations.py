import hashlib
import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from zmodel.api.config import get_api_settings
from zmodel.api.jobs import JobQueueFullError, JobStatus, get_job_manager
from zmodel.api.model_store import resolve_model_bytes
from zmodel.api.schemas import (
    AdvancedOptimizationSubmitRequest,
    CandidateEvaluationItem,
    EnterpriseReportResponse,
    OptimizationJobResponse,
    OptimizationStatusResponse,
    OptimizationSubmitRequest,
    ParetoCandidateItem,
    ParetoFrontierResponse,
    RecommendationResponse,
    SensitivityAnalysisResponse,
)
from zmodel.core.advanced_optimizer import (
    AdvancedOptimizationEngine,
    EnterpriseReportGenerator,
    RecommendationGoal,
    SearchStrategy,
)
from zmodel.core.execution import (
    BackendUnavailableError,
    WorkerJobSpec,
    WorkerJobType,
    get_execution_backend,
)
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.optimizer import (
    HardConstraint,
    OptimizationEngine,
    OptimizationObjective,
    OptimizationRequest,
)
from zmodel.core.profiles import load_builtin_profiles
from zmodel.db.repositories import OptimizationRepository, WorkerJobRepository
from zmodel.db.session import session_scope

router = APIRouter(prefix="/api/v1/optimizations", tags=["Evidence-Based Optimization"])


def _execute_optimization_task(
    raw_bytes: bytes,
    filename: str,
    profile_id: str,
    req_schema: OptimizationSubmitRequest,
    progress_callback=None,
):
    """Worker task executed in background thread pool."""
    profiles = load_builtin_profiles()
    profile = profiles[profile_id]

    if progress_callback:
        progress_callback(10, "Inspecting model structure for candidate generation")

    summary = load_and_inspect_model(raw_bytes, filename=filename)

    # Convert hard constraints
    constraints = [
        HardConstraint(
            name=c.name,
            metric_field=c.metric_field,
            operator=c.operator,
            threshold=c.threshold,
            unit=c.unit,
        )
        for c in req_schema.hard_constraints
    ]

    opt_req = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        primary_objective=OptimizationObjective(req_schema.primary_objective),
        hard_constraints=constraints,
        max_candidates=req_schema.max_candidates,
        batch_sizes=req_schema.batch_sizes,
        intra_op_threads=req_schema.intra_op_threads,
        inter_op_threads=req_schema.inter_op_threads,
        warmup_iterations=req_schema.warmup_iterations,
        measured_iterations=req_schema.measured_iterations,
        repeated_rounds=req_schema.repeated_rounds,
        execution_providers=req_schema.execution_providers,
    )

    def engine_progress(stage: str, current: int, total: int):
        if progress_callback:
            pct = 10 + int((current / max(1, total)) * 80)
            progress_callback(pct, f"{stage}: {current}/{total}")

    report = OptimizationEngine.run_optimization(
        raw_bytes,
        summary,
        profile,
        opt_req,
        progress_callback=engine_progress,
    )

    if progress_callback:
        progress_callback(95, "Persisting optimization experiment and candidate evaluations")

    try:
        repo = OptimizationRepository()
        repo.save_optimization_report(report)
    except Exception:
        pass  # Graceful tolerance if database is offline

    return report.to_dict()


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=OptimizationJobResponse,
    summary="Submit Optimization Experiment Job",
    description=(
        "Queue an evidence-based multi-candidate optimization experiment. Generates runtime configurations, "
        "filters them against capability profiles, evaluates hard constraints, benchmarks candidates, "
        "and computes Pareto frontiers. Returns HTTP 202 with an experiment tracking ID."
    ),
)
def submit_optimization(request: OptimizationSubmitRequest) -> OptimizationJobResponse:
    settings = get_api_settings()

    # Enforce configured candidate limit
    if request.max_candidates > settings.max_optimization_candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"max_candidates ({request.max_candidates}) exceeds system maximum of {settings.max_optimization_candidates}.",
        )

    # Validate objective
    try:
        OptimizationObjective(request.primary_objective)
    except ValueError:
        valid_objs = [o.value for o in OptimizationObjective]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid primary_objective '{request.primary_objective}'. Allowed objectives: {valid_objs}",
        )

    # Validate profile
    profiles = load_builtin_profiles()
    if request.profile_id not in profiles:
        available = list(profiles.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown profile ID '{request.profile_id}'. Available profiles: {available}",
        )

    # Safely resolve model bytes
    raw_bytes, filename = resolve_model_bytes(
        model_fingerprint=request.model_fingerprint,
        sample_model_name=request.sample_model_name,
    )
    model_hash = hashlib.sha256(raw_bytes).hexdigest()
    target_backend = (request.execution_backend or os.environ.get("EXECUTION_BACKEND", "local")).lower().strip()

    if target_backend == "kubernetes":
        try:
            k8s_backend = get_execution_backend("kubernetes")
            job_id = f"job-k8s-{uuid.uuid4().hex[:12]}"
            spec = WorkerJobSpec(
                job_id=job_id,
                job_type=WorkerJobType.OPTIMIZATION,
                model_fingerprint=model_hash,
                model_filename=filename,
                resource_profile=request.resource_profile or "large_opt",
                parameters=request.model_dump(),
            )
            try:
                from zmodel.core.artifact_store import get_artifact_store
                store = get_artifact_store()
                store.store_artifact(raw_bytes, parent_fingerprint=model_hash, transformation_name="k8s_opt", original_filename=filename)
            except Exception:
                pass

            try:
                with session_scope() as s:
                    repo = WorkerJobRepository(session=s)
                    repo.create_job(spec, backend="kubernetes")
            except Exception:
                pass

            k8s_backend.submit_job(spec)
            return OptimizationJobResponse(
                job_id=job_id,
                experiment_id=job_id,
                status="SUBMITTED",
                message="Optimization search job submitted to Kubernetes cluster.",
            )
        except BackendUnavailableError as bue:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Kubernetes backend unavailable: {bue}",
            )

    job_mgr = get_job_manager()
    try:
        job_id = job_mgr.submit_job(
            job_type="optimization",
            task_fn=_execute_optimization_task,
            raw_bytes=raw_bytes,
            filename=filename,
            profile_id=request.profile_id,
            req_schema=request,
        )
        try:
            with session_scope() as s:
                repo = WorkerJobRepository(session=s)
                spec = WorkerJobSpec(
                    job_id=job_id,
                    job_type=WorkerJobType.OPTIMIZATION,
                    model_fingerprint=model_hash,
                    model_filename=filename,
                    resource_profile=request.resource_profile or "large_opt",
                    parameters=request.model_dump(),
                )
                repo.create_job(spec, backend="local")
        except Exception:
            pass
    except JobQueueFullError as qfe:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(qfe),
        ) from qfe

    return OptimizationJobResponse(
        job_id=job_id,
        experiment_id=job_id,
        status="QUEUED",
        message="Optimization search job queued in bounded worker pool.",
    )


@router.get(
    "/{experiment_id}/status",
    response_model=OptimizationStatusResponse,
    summary="Get Optimization Experiment Status",
    description="Check the execution lifecycle state, progress, or failure message of an optimization search job.",
)
def get_optimization_status(experiment_id: str) -> OptimizationStatusResponse:
    job_mgr = get_job_manager()
    job = job_mgr.get_job(experiment_id)

    if job:
        return OptimizationStatusResponse(
            job_id=job.job_id,
            experiment_id=job.resource_id or job.job_id,
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
            w_job = w_repo.get_job(experiment_id)
            if w_job:
                d = w_job.to_dict()
                return OptimizationStatusResponse(
                    job_id=d["id"],
                    experiment_id=d["experiment_id"] or d["id"],
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

    # Check database for completed experiment
    try:
        repo = OptimizationRepository()
        exp = repo.get_optimization_experiment(experiment_id)
        if exp:
            return OptimizationStatusResponse(
                job_id=experiment_id,
                experiment_id=exp.id,
                status="COMPLETED",
                created_at=exp.created_at.isoformat() if exp.created_at else "",
                started_at=exp.created_at.isoformat() if exp.created_at else None,
                completed_at=exp.completed_at.isoformat() if exp.completed_at else None,
                progress_pct=100,
                progress_message="Optimization experiment completed and persisted in database",
                error=None,
                result=exp.to_dict(),
            )
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Optimization experiment '{experiment_id}' not found in active queue or database.",
    )


@router.get(
    "/{experiment_id}",
    summary="Get Completed Optimization Experiment Report",
    description="Retrieve full optimization report including baseline vs recommended candidate, Pareto frontier, and rationale.",
)
def get_optimization_detail(experiment_id: str):
    try:
        repo = OptimizationRepository()
        exp = repo.get_optimization_experiment(experiment_id)
        if exp:
            return exp.to_dict()
    except Exception:
        pass

    job_mgr = get_job_manager()
    job = job_mgr.get_job(experiment_id)
    if job:
        if job.status == JobStatus.COMPLETED and job.result:
            return job.result
        elif job.status == JobStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Optimization experiment failed: {job.error}",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Optimization experiment is currently {job.status.value} ({job.progress_pct}% - {job.progress_message}). Query /{experiment_id}/status for progress.",
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Optimization experiment '{experiment_id}' not found.",
    )


@router.get(
    "/{experiment_id}/candidates",
    response_model=List[CandidateEvaluationItem],
    summary="List Candidates for Optimization Experiment",
    description="Retrieve itemized candidate evaluations, feasibility results, speedups, and Pareto classifications.",
)
def list_experiment_candidates(experiment_id: str) -> List[CandidateEvaluationItem]:
    try:
        repo = OptimizationRepository()
        candidates = repo.get_candidates_for_experiment(experiment_id)
        if candidates:
            return [
                CandidateEvaluationItem(
                    candidate_id=c.candidate_id,
                    benchmark_run_id=c.benchmark_run_id,
                    status=c.status,
                    rank=c.rank,
                    is_feasible=c.is_feasible,
                    is_pareto_optimal=c.is_pareto_optimal,
                    is_baseline=c.is_baseline,
                    speedup_vs_baseline=c.speedup_vs_baseline,
                    latency_reduction_pct=c.latency_reduction_pct,
                    measured_median_latency_ms=c.measured_median_latency_ms,
                    measured_throughput_ips=c.measured_throughput_ips,
                    measured_memory_mb=c.measured_memory_mb,
                    configuration=c.configuration,
                    rejection_reason=c.rejection_reason,
                )
                for c in candidates
            ]
    except Exception:
        pass

    # Check job result in memory if not yet persisted or if db is offline
    job_mgr = get_job_manager()
    job = job_mgr.get_job(experiment_id)
    if job and job.status == JobStatus.COMPLETED and job.result:
        raw_cands = job.result.get("candidates", [])
        return [
            CandidateEvaluationItem(
                candidate_id=c.get("candidate_id"),
                benchmark_run_id=c.get("benchmark_run_id"),
                status=c.get("status"),
                rank=c.get("rank"),
                is_feasible=c.get("is_feasible", False),
                is_pareto_optimal=c.get("is_pareto_optimal", False),
                is_baseline=c.get("is_baseline", False),
                speedup_vs_baseline=c.get("speedup_vs_baseline"),
                latency_reduction_pct=c.get("latency_reduction_pct"),
                measured_median_latency_ms=c.get("measured_median_latency_ms"),
                measured_throughput_ips=c.get("measured_throughput_ips"),
                measured_memory_mb=c.get("measured_memory_mb"),
                configuration=c.get("config", {}),
                rejection_reason=c.get("rejection_reason"),
            )
            for c in raw_cands
        ]

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Candidates for experiment '{experiment_id}' not found.",
    )


def _execute_advanced_optimization_task(
    raw_bytes: bytes,
    filename: str,
    profile_id: str,
    req_schema: AdvancedOptimizationSubmitRequest,
    progress_callback=None,
):
    """Background task for executing Phase 12 Advanced Optimization search."""
    profiles = load_builtin_profiles()
    profile = profiles[profile_id]

    if progress_callback:
        progress_callback(5, "Inspecting model structure for advanced search")

    summary = load_and_inspect_model(raw_bytes, filename=filename)

    constraints = [
        HardConstraint(
            name=c.name,
            metric_field=c.metric_field,
            operator=c.operator,
            threshold=c.threshold,
            unit=c.unit,
        )
        for c in req_schema.hard_constraints
    ]

    opt_req = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        primary_objective=OptimizationObjective(req_schema.primary_objective),
        hard_constraints=constraints,
        max_candidates=req_schema.max_candidates,
        batch_sizes=req_schema.batch_sizes,
        intra_op_threads=req_schema.intra_op_threads,
        inter_op_threads=req_schema.inter_op_threads,
        warmup_iterations=req_schema.warmup_iterations,
        measured_iterations=req_schema.measured_iterations,
        repeated_rounds=req_schema.repeated_rounds,
        random_seed=req_schema.random_seed,
        execution_providers=req_schema.execution_providers,
    )

    strategy = SearchStrategy(req_schema.search_strategy)
    goal = RecommendationGoal(req_schema.recommendation_goal)

    def engine_progress(msg: str, current: int, total: int):
        if progress_callback:
            pct = 10 + int((current / max(1, total)) * 80)
            progress_callback(pct, msg)

    report_dict = AdvancedOptimizationEngine.execute_advanced_optimization(
        model_path_or_bytes=raw_bytes,
        model_summary=summary,
        profile=profile,
        request=opt_req,
        strategy=strategy,
        recommendation_goal=goal,
        graph_optimization_levels=req_schema.graph_optimization_levels,
        execution_modes=req_schema.execution_modes,
        progress_callback=engine_progress,
    )

    if progress_callback:
        progress_callback(95, "Persisting advanced optimization report and candidates")

    try:
        repo = OptimizationRepository()
        repo.save_advanced_optimization_report(report_dict)
    except Exception:
        pass

    return report_dict


@router.post(
    "/search",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=OptimizationJobResponse,
    summary="Submit Advanced Optimization Search Job",
    description=(
        "Queue a Phase 12 bounded configuration search job (Grid, Random, or Successive Halving) "
        "across execution providers, thread parameters, batch sizes, and graph optimization levels. "
        "Returns HTTP 202 with an experiment job tracking ID."
    ),
)
def submit_advanced_optimization(request: AdvancedOptimizationSubmitRequest) -> OptimizationJobResponse:
    settings = get_api_settings()

    if request.max_candidates > settings.max_optimization_candidates:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"max_candidates ({request.max_candidates}) exceeds system maximum of {settings.max_optimization_candidates}.",
        )

    try:
        SearchStrategy(request.search_strategy)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid search_strategy '{request.search_strategy}'. Allowed: {[s.value for s in SearchStrategy]}",
        )

    try:
        RecommendationGoal(request.recommendation_goal)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid recommendation_goal '{request.recommendation_goal}'. Allowed: {[g.value for g in RecommendationGoal]}",
        )

    profiles = load_builtin_profiles()
    if request.profile_id not in profiles:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown profile ID '{request.profile_id}'. Available profiles: {list(profiles.keys())}",
        )

    raw_bytes, filename = resolve_model_bytes(
        model_fingerprint=request.model_fingerprint,
        sample_model_name=request.sample_model_name,
    )

    job_mgr = get_job_manager()
    try:
        job_id = job_mgr.submit_job(
            job_type="advanced_optimization",
            task_fn=_execute_advanced_optimization_task,
            raw_bytes=raw_bytes,
            filename=filename,
            profile_id=request.profile_id,
            req_schema=request,
        )
    except JobQueueFullError as qfe:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(qfe),
        ) from qfe

    return OptimizationJobResponse(
        job_id=job_id,
        experiment_id=job_id,
        status="QUEUED",
        message=f"Advanced {request.search_strategy} optimization search job queued successfully.",
    )


@router.get(
    "/{experiment_id}/pareto",
    response_model=ParetoFrontierResponse,
    summary="Get Multi-Objective Pareto Frontier",
    description="Retrieve the non-dominated Pareto frontier candidates comparing latency, throughput, and memory.",
)
def get_experiment_pareto_frontier(experiment_id: str) -> ParetoFrontierResponse:
    job_mgr = get_job_manager()
    job = job_mgr.get_job(experiment_id)
    report_dict = None
    if job and job.status == JobStatus.COMPLETED and job.result:
        report_dict = job.result
    else:
        try:
            repo = OptimizationRepository()
            exp = repo.get_optimization_experiment(experiment_id)
            if exp:
                report_dict = exp.to_dict()
        except Exception:
            pass

    if not report_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization experiment '{experiment_id}' not found.",
        )

    pareto_ids = report_dict.get("pareto_frontier_ids", [])
    candidates = report_dict.get("candidates", [])
    frontier_items = []

    for c in candidates:
        cid = c.get("candidate_id")
        cfg = c.get("configuration") or c.get("config") or {}
        m = c.get("metrics") or {}
        is_pareto = cid in pareto_ids or c.get("is_pareto_optimal", False)

        if is_pareto:
            frontier_items.append(
                ParetoCandidateItem(
                    candidate_id=cid,
                    is_pareto_optimal=True,
                    median_latency_ms=m.get("median_latency_ms") or c.get("measured_median_latency_ms"),
                    p95_latency_ms=m.get("p95_latency_ms"),
                    throughput_inf_sec=m.get("throughput_inf_sec") or c.get("measured_throughput_ips"),
                    peak_memory_mb=m.get("peak_memory_mb") or c.get("measured_memory_mb"),
                    configuration=cfg,
                )
            )

    return ParetoFrontierResponse(
        experiment_id=experiment_id,
        pareto_frontier_ids=pareto_ids,
        frontier_candidates=frontier_items,
        total_evaluated=len(candidates),
        pareto_count=len(frontier_items),
        tradeoff_explanation=(
            f"The Pareto frontier contains {len(frontier_items)} non-dominated candidate configurations. "
            "No candidate on this frontier can be improved in one objective (e.g. latency) without sacrificing another (e.g. throughput or memory)."
        ),
    )


@router.get(
    "/{experiment_id}/sensitivity",
    response_model=SensitivityAnalysisResponse,
    summary="Get Model Parameter Sensitivity Analysis",
    description="Retrieve latency elasticity and throughput scaling response across batch size, thread counts, and runtime settings.",
)
def get_experiment_sensitivity(experiment_id: str) -> SensitivityAnalysisResponse:
    job_mgr = get_job_manager()
    job = job_mgr.get_job(experiment_id)
    report_dict = None
    if job and job.status == JobStatus.COMPLETED and job.result:
        report_dict = job.result
    else:
        try:
            repo = OptimizationRepository()
            exp = repo.get_optimization_experiment(experiment_id)
            if exp:
                report_dict = exp.to_dict()
                if "sensitivity" in exp.request_payload:
                    report_dict["sensitivity"] = exp.request_payload["sensitivity"]
        except Exception:
            pass

    if not report_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization experiment '{experiment_id}' not found.",
        )

    sens_dict = report_dict.get("sensitivity") or {}
    return SensitivityAnalysisResponse(
        experiment_id=experiment_id,
        parameters=sens_dict.get("parameters", {}),
        primary_influencing_parameter=sens_dict.get("primary_influencing_parameter"),
        summary_insight=sens_dict.get("summary_insight", "Sensitivity analysis metrics computed from empirical benchmark points."),
    )


@router.get(
    "/{experiment_id}/recommendation",
    response_model=RecommendationResponse,
    summary="Get Evidence-Backed Deployment Recommendation",
    description="Retrieve the optimal candidate configuration recommendation for a selected goal (latency, throughput, memory, or balanced).",
)
def get_experiment_recommendation(experiment_id: str) -> RecommendationResponse:
    job_mgr = get_job_manager()
    job = job_mgr.get_job(experiment_id)
    report_dict = None
    if job and job.status == JobStatus.COMPLETED and job.result:
        report_dict = job.result
    else:
        try:
            repo = OptimizationRepository()
            exp = repo.get_optimization_experiment(experiment_id)
            if exp:
                report_dict = exp.to_dict()
                if "recommendation" in exp.request_payload:
                    report_dict["recommendation"] = exp.request_payload["recommendation"]
        except Exception:
            pass

    if not report_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization experiment '{experiment_id}' not found.",
        )

    rec = report_dict.get("recommendation") or {}
    candidates = report_dict.get("candidates", [])
    best_id = rec.get("recommended_candidate_id") or report_dict.get("best_candidate_id")
    best_cand = next((c for c in candidates if c.get("candidate_id") == best_id), {})

    m = best_cand.get("metrics") or {}
    cfg = best_cand.get("configuration") or best_cand.get("config") or rec.get("recommended_config") or {}

    is_z = report_dict.get("environment", {}).get("is_s390x", False) and not report_dict.get("is_illustrative_profile", False)
    provenance = "TARGET_VERIFIED (IBM Z s390x)" if is_z else "LOCALLY_VERIFIED (x86_64 Host · IBM Z Pending)"

    return RecommendationResponse(
        experiment_id=experiment_id,
        goal=rec.get("goal", "BALANCED_SCORE"),
        recommended_candidate_id=best_id,
        recommended_config=cfg,
        is_target_verified=is_z,
        provenance_badge=provenance,
        median_latency_ms=rec.get("median_latency_ms") or m.get("median_latency_ms"),
        throughput_inf_sec=rec.get("throughput_inf_sec") or m.get("throughput_inf_sec"),
        speedup_vs_baseline=best_cand.get("speedup_vs_baseline"),
        latency_reduction_pct=best_cand.get("latency_reduction_pct"),
        reliability=rec.get("reliability"),
        rationale=rec.get("rationale") or report_dict.get("recommendation_rationale") or "Evidence-based recommendation.",
        unverified_requirements=rec.get("unverified_requirements", [
            "IBM Z target validation remains pending; only the listed local checks have been verified."
        ]),
    )


@router.get(
    "/{experiment_id}/report",
    response_model=EnterpriseReportResponse,
    summary="Get 18-Section Enterprise Audit Report",
    description="Retrieve the full reproducible 18-section deployment optimization report in Markdown or JSON format with cryptographic integrity checksum.",
)
def get_enterprise_report(experiment_id: str, format: str = "markdown") -> EnterpriseReportResponse:
    job_mgr = get_job_manager()
    job = job_mgr.get_job(experiment_id)
    report_dict = None
    if job and job.status == JobStatus.COMPLETED and job.result:
        report_dict = job.result
    else:
        try:
            repo = OptimizationRepository()
            exp = repo.get_optimization_experiment(experiment_id)
            if exp:
                report_dict = exp.to_dict()
                if "markdown_report" in exp.request_payload:
                    report_dict["markdown_report"] = exp.request_payload["markdown_report"]
                if "json_report" in exp.request_payload:
                    report_dict["json_report"] = exp.request_payload["json_report"]
        except Exception:
            pass

    if not report_dict:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Optimization experiment '{experiment_id}' not found.",
        )

    fmt = format.lower().strip()
    if fmt == "json":
        content = report_dict.get("json_report") or EnterpriseReportGenerator.generate_json(report_dict)
    else:
        content = report_dict.get("markdown_report") or EnterpriseReportGenerator.generate_markdown(report_dict)

    sha256_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return EnterpriseReportResponse(
        experiment_id=experiment_id,
        format=fmt,
        report_content=content,
        sha256_checksum=sha256_hash,
    )

