"""Persistent experiment history and comparison REST endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from zmodel.api.schemas import (
    BenchmarkComparisonRequest,
    BenchmarkComparisonResponse,
    CandidateEvaluationItem,
    ExperimentListItem,
    ExperimentListResponse,
)
from zmodel.db.comparator import compare_benchmark_runs
from zmodel.db.repositories import BenchmarkRepository, OptimizationRepository

router = APIRouter(prefix="/api/v1/experiments", tags=["Experiment History & Comparisons"])


@router.get(
    "",
    response_model=ExperimentListResponse,
    summary="List Persisted Optimization Experiments",
    description="Retrieve paginated historical optimization experiments from PostgreSQL with optional filtering.",
)
def list_experiments(
    fingerprint: Optional[str] = Query(None, description="Filter by model SHA-256 fingerprint"),
    status: Optional[str] = Query(None, description="Filter by status (e.g. COMPLETED, FAILED)"),
    objective: Optional[str] = Query(None, description="Filter by primary objective"),
    limit: int = Query(20, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
) -> ExperimentListResponse:
    try:
        repo = OptimizationRepository()
        experiments, total = repo.list_optimization_experiments(
            fingerprint=fingerprint,
            status=status,
            objective=objective,
            limit=limit,
            offset=offset,
        )
    except Exception:
        return ExperimentListResponse(experiments=[], total=0, limit=limit, offset=offset)

    items = [
        ExperimentListItem(
            id=exp.id,
            model_name=exp.model_name,
            model_fingerprint=exp.model_fingerprint,
            profile_id=exp.profile_id,
            primary_objective=exp.primary_objective,
            status=exp.status,
            best_candidate_id=exp.best_candidate_id,
            summary_counts=exp.summary_counts or {},
            created_at=exp.created_at.isoformat() if exp.created_at else "",
        )
        for exp in experiments
    ]

    return ExperimentListResponse(
        experiments=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{experiment_id}",
    summary="Get Experiment Details",
    description="Retrieve full persisted record of an optimization experiment including rationale and summary counts.",
)
def get_experiment_detail(experiment_id: str):
    try:
        repo = OptimizationRepository()
        exp = repo.get_optimization_experiment(experiment_id)
        if exp:
            return exp.to_dict()
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Experiment '{experiment_id}' not found in database.",
    )


@router.get(
    "/{experiment_id}/candidates",
    response_model=List[CandidateEvaluationItem],
    summary="List Evaluated Candidates for Experiment",
    description="Retrieve persisted candidate evaluations, rankings, speedup factors, and Pareto classifications.",
)
def get_experiment_candidates(experiment_id: str) -> List[CandidateEvaluationItem]:
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

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Candidates for experiment '{experiment_id}' not found in database.",
    )


@router.post(
    "/compare",
    response_model=BenchmarkComparisonResponse,
    summary="Compare Two Empirical Benchmark Runs",
    description=(
        "Compare two persisted benchmark executions (Run A = baseline, Run B = candidate). "
        "Strictly verifies scientific comparability: rejects comparisons between different models, "
        "failed correctness checks, or differing workload batch sizes before computing relative metrics."
    ),
)
def compare_runs(request: BenchmarkComparisonRequest) -> BenchmarkComparisonResponse:
    try:
        bench_repo = BenchmarkRepository()
        run_a = bench_repo.get_benchmark_run(request.run_id_a)
        run_b = bench_repo.get_benchmark_run(request.run_id_b)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query error during run retrieval: {str(ex)}",
        ) from ex

    if not run_a:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Baseline benchmark Run A '{request.run_id_a}' not found in database.",
        )
    if not run_b:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate benchmark Run B '{request.run_id_b}' not found in database.",
        )

    res = compare_benchmark_runs(run_a, run_b)

    delta_lat = None
    speedup = None
    lat_red = None
    delta_tp = None

    if res.is_valid:
        lat_comp = res.metric_comparisons.get("median_latency")
        if lat_comp:
            delta_lat = lat_comp.absolute_delta
            speedup = lat_comp.speedup_factor
            lat_red = lat_comp.percentage_improvement

        tp_comp = res.metric_comparisons.get("samples_throughput") or res.metric_comparisons.get("inference_throughput")
        if tp_comp:
            delta_tp = tp_comp.absolute_delta

        summary_narrative = (
            f"Valid comparison between Run A and Run B. "
            f"Latency delta: {delta_lat:+.3f} ms (Speedup: {speedup or 1.0:.2f}x). "
            f"Throughput delta: {delta_tp:+.1f} samples/sec."
            if delta_lat is not None and delta_tp is not None
            else "Valid comparison with completed metrics."
        )
    else:
        summary_narrative = (
            f"Comparison disqualified ({res.validity.value}): "
            + "; ".join(res.disqualification_reasons)
        )

    return BenchmarkComparisonResponse(
        validity=res.validity.value,
        disqualification_reasons=res.disqualification_reasons,
        run_a_summary={
            "run_id": run_a.id,
            "model_name": run_a.model_name,
            "model_fingerprint": run_a.model_fingerprint,
            "execution_provider": run_a.execution_provider,
            "batch_size": run_a.batch_size,
            "median_latency_ms": run_a.latency_median_ms,
            "throughput_samples_sec": run_a.throughput_samples_sec,
        },
        run_b_summary={
            "run_id": run_b.id,
            "model_name": run_b.model_name,
            "model_fingerprint": run_b.model_fingerprint,
            "execution_provider": run_b.execution_provider,
            "batch_size": run_b.batch_size,
            "median_latency_ms": run_b.latency_median_ms,
            "throughput_samples_sec": run_b.throughput_samples_sec,
        },
        delta_latency_ms=delta_lat,
        speedup_factor=speedup,
        latency_reduction_pct=lat_red,
        delta_throughput_samples_sec=delta_tp,
        comparison_summary=summary_narrative,
    )
