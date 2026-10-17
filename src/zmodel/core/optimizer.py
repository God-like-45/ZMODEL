"""
Phase 5: Evidence-Based Optimization Engine.

Identifies, filters, benchmarks, and ranks model execution configurations using:
1. Compatibility findings from Phase 3.
2. Real ONNX Runtime benchmarks from Phase 4.
3. User-defined latency, throughput, and memory constraints.
4. Reproducible candidate configuration metadata.
5. Strict correctness validation and Pareto frontier analysis.

Never invents performance measurements, fabricates speedups, or claims an optimization
improved performance before measuring it.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import io
import json
import platform
import psutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from zmodel.core.benchmark import (
    BenchmarkConfig,
    BenchmarkEngine,
    BenchmarkError,
    BenchmarkReport,
    CorrectnessValidationError,
    discover_execution_providers,
)
from zmodel.core.compatibility import (
    CompatibilityAnalyzer,
    CompatibilityFinding,
    CompatibilityReport,
)
from zmodel.core.model_loader import ModelSummary
from zmodel.core.profiles import CapabilityProfile
from zmodel.core.system_info import get_system_environment


class OptimizationObjective(str, Enum):
    """Primary objective guiding candidate ranking."""
    MINIMIZE_MEDIAN_LATENCY = "MINIMIZE_MEDIAN_LATENCY"
    MINIMIZE_P95_LATENCY = "MINIMIZE_P95_LATENCY"
    MAXIMIZE_THROUGHPUT = "MAXIMIZE_THROUGHPUT"
    MINIMIZE_MEMORY = "MINIMIZE_MEMORY"
    EXPLORE_TRADEOFFS = "EXPLORE_TRADEOFFS"  # Multi-objective Pareto focus


class CandidateStatus(str, Enum):
    """Status lifecycle of an execution candidate."""
    GENERATED = "GENERATED"
    REJECTED = "REJECTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    BENCHMARKED = "BENCHMARKED"
    FAILED = "FAILED"
    INFEASIBLE = "INFEASIBLE"


class ExperimentStatus(str, Enum):
    """Overall status lifecycle of an optimization experiment."""
    CREATED = "CREATED"
    GENERATING_CANDIDATES = "GENERATING_CANDIDATES"
    FILTERING_CANDIDATES = "FILTERING_CANDIDATES"
    BENCHMARKING = "BENCHMARKING"
    RANKING = "RANKING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"


class OptimizationError(Exception):
    """Base exception for optimization engine failures."""
    pass


class InvalidOptimizationRequestError(OptimizationError):
    """Raised when optimization request parameters violate validity constraints."""
    pass


@dataclass
class HardConstraint:
    """A user-defined non-negotiable operational boundary."""
    name: str
    metric_field: str  # e.g. "median_latency_ms", "p95_latency_ms", "throughput_inf_sec", "peak_memory_mb"
    operator: str      # "<=", ">=", "=="
    threshold: float
    unit: str
    is_mandatory: bool = True

    def evaluate(self, candidate_metrics: Dict[str, Any]) -> ConstraintEvaluation:
        """Evaluate this constraint against observed benchmark metrics."""
        observed = candidate_metrics.get(self.metric_field)
        if observed is None:
            return ConstraintEvaluation(
                constraint_name=self.name,
                metric_field=self.metric_field,
                threshold=self.threshold,
                observed_value=None,
                unit=self.unit,
                passed=False,
                is_available=False,
                explanation=f"Metric '{self.metric_field}' was not available from benchmark measurements.",
            )

        passed = False
        if self.operator == "<=":
            passed = observed <= self.threshold
        elif self.operator == ">=":
            passed = observed >= self.threshold
        elif self.operator == "==":
            passed = abs(observed - self.threshold) < 1e-6

        status_text = "Satisfied" if passed else "Violated"
        explanation = (
            f"{self.name}: observed {observed:.3f} {self.unit} {self.operator} "
            f"threshold {self.threshold:.3f} {self.unit} -> {status_text}"
        )
        return ConstraintEvaluation(
            constraint_name=self.name,
            metric_field=self.metric_field,
            threshold=self.threshold,
            observed_value=float(observed),
            unit=self.unit,
            passed=passed,
            is_available=True,
            explanation=explanation,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstraintEvaluation:
    """Outcome of evaluating one hard constraint on a candidate."""
    constraint_name: str
    metric_field: str
    threshold: float
    observed_value: Optional[float]
    unit: str
    passed: bool
    is_available: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionCandidate:
    """A discrete execution configuration undergoing optimization evaluation."""
    candidate_id: str
    config: BenchmarkConfig
    generation_reason: str
    status: CandidateStatus = CandidateStatus.GENERATED
    rejection_reason: Optional[str] = None
    compatibility_findings: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_report: Optional[BenchmarkReport] = None
    constraint_evaluations: List[ConstraintEvaluation] = field(default_factory=list)
    is_feasible: bool = False
    is_pareto_optimal: bool = False
    is_baseline: bool = False
    speedup_vs_baseline: Optional[float] = None
    latency_reduction_pct: Optional[float] = None
    throughput_improvement_pct: Optional[float] = None
    rank: Optional[int] = None

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Extract standardized metric values from the attached benchmark report."""
        if not self.benchmark_report or self.benchmark_report.status != "SUCCESS":
            return {}
        rep = self.benchmark_report
        lat = rep.latency or {}
        tp = rep.throughput or {}
        mem = rep.memory or {}
        corr = rep.correctness or {}
        correctness_passed = corr.get("passed", False) if isinstance(corr, dict) else getattr(corr, "passed", getattr(corr, "valid", False))
        return {
            "median_latency_ms": lat.get("median_ms"),
            "mean_latency_ms": lat.get("mean_ms"),
            "p95_latency_ms": lat.get("p95_ms"),
            "p90_latency_ms": lat.get("p90_ms"),
            "p99_latency_ms": lat.get("p99_ms"),
            "min_latency_ms": lat.get("min_ms"),
            "max_latency_ms": lat.get("max_ms"),
            "throughput_inf_sec": tp.get("inferences_per_second"),
            "throughput_samples_sec": tp.get("samples_per_second"),
            "peak_memory_mb": mem.get("peak_rss_mb"),
            "memory_delta_mb": mem.get("session_delta_mb", mem.get("session_memory_delta_mb")),
            "correctness_valid": bool(correctness_passed),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to JSON-serializable dictionary."""
        config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else asdict(self.config)
        return {
            "candidate_id": self.candidate_id,
            "config": config_dict,
            "generation_reason": self.generation_reason,
            "status": self.status.value,
            "rejection_reason": self.rejection_reason,
            "compatibility_findings": self.compatibility_findings,
            "benchmark_report": self.benchmark_report.to_dict() if self.benchmark_report else None,
            "constraint_evaluations": [ce.to_dict() for ce in self.constraint_evaluations],
            "is_feasible": self.is_feasible,
            "is_pareto_optimal": self.is_pareto_optimal,
            "is_baseline": self.is_baseline,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "latency_reduction_pct": self.latency_reduction_pct,
            "throughput_improvement_pct": self.throughput_improvement_pct,
            "rank": self.rank,
        }


@dataclass
class OptimizationRequest:
    """Structured request defining an optimization experiment."""
    model_fingerprint: str
    profile_id: str
    profile_version: str
    primary_objective: OptimizationObjective = OptimizationObjective.MINIMIZE_MEDIAN_LATENCY
    hard_constraints: List[HardConstraint] = field(default_factory=list)
    max_candidates: int = 8
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2])
    intra_op_threads: List[int] = field(default_factory=lambda: [1, 2, 4])
    inter_op_threads: List[int] = field(default_factory=lambda: [1])
    warmup_iterations: int = 5
    measured_iterations: int = 25
    repeated_rounds: int = 2
    random_seed: int = 42
    dynamic_dimensions: Dict[str, int] = field(default_factory=dict)
    execution_providers: Optional[List[str]] = None

    def validate(self) -> None:
        """Validate request parameters against safety and resource bounds."""
        if not self.model_fingerprint:
            raise InvalidOptimizationRequestError("Model SHA-256 fingerprint must be specified.")
        if not self.profile_id:
            raise InvalidOptimizationRequestError("Target capability profile ID must be specified.")
        if self.max_candidates < 1 or self.max_candidates > 50:
            raise InvalidOptimizationRequestError(
                f"max_candidates must be between 1 and 50 (received {self.max_candidates})."
            )
        if self.warmup_iterations < 0 or self.warmup_iterations > 1000:
            raise InvalidOptimizationRequestError(
                f"warmup_iterations must be between 0 and 1,000 (received {self.warmup_iterations})."
            )
        if self.measured_iterations < 1 or self.measured_iterations > 5000:
            raise InvalidOptimizationRequestError(
                f"measured_iterations must be between 1 and 5,000 (received {self.measured_iterations})."
            )
        if self.repeated_rounds < 1 or self.repeated_rounds > 10:
            raise InvalidOptimizationRequestError(
                f"repeated_rounds must be between 1 and 10 (received {self.repeated_rounds})."
            )
        for b in self.batch_sizes:
            if b < 1 or b > 256:
                raise InvalidOptimizationRequestError(f"Batch size {b} must be between 1 and 256.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_fingerprint": self.model_fingerprint,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "primary_objective": self.primary_objective.value,
            "hard_constraints": [hc.to_dict() for hc in self.hard_constraints],
            "max_candidates": self.max_candidates,
            "batch_sizes": self.batch_sizes,
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "repeated_rounds": self.repeated_rounds,
            "random_seed": self.random_seed,
            "dynamic_dimensions": self.dynamic_dimensions,
            "execution_providers": self.execution_providers,
        }


@dataclass
class OptimizationReport:
    """Complete, reproducible audit artifact of an optimization experiment."""
    experiment_id: str
    timestamp_utc: str
    status: ExperimentStatus
    model_name: str
    model_fingerprint: str
    profile_id: str
    profile_name: str
    profile_version: str
    primary_objective: str
    request: OptimizationRequest
    baseline_candidate_id: Optional[str]
    best_candidate_id: Optional[str]
    candidates: List[ExecutionCandidate]
    pareto_frontier_ids: List[str]
    summary_counts: Dict[str, int]
    recommendation_rationale: str
    environment: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp_utc": self.timestamp_utc,
            "status": self.status.value,
            "model_name": self.model_name,
            "model_fingerprint": self.model_fingerprint,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "primary_objective": self.primary_objective,
            "request": self.request.to_dict(),
            "baseline_candidate_id": self.baseline_candidate_id,
            "best_candidate_id": self.best_candidate_id,
            "summary_counts": self.summary_counts,
            "recommendation_rationale": self.recommendation_rationale,
            "pareto_frontier_ids": self.pareto_frontier_ids,
            "candidates": [c.to_dict() for c in self.candidates],
            "environment": self.environment,
            "warnings": self.warnings,
            "limitations": self.limitations,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export full report as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Export tabular candidate-level metrics and ranking to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Rank",
            "Candidate ID",
            "Status",
            "Is Feasible",
            "Is Pareto Optimal",
            "Provider",
            "Batch Size",
            "Intra-op Threads",
            "Inter-op Threads",
            "Median Latency (ms)",
            "Mean Latency (ms)",
            "P95 Latency (ms)",
            "Throughput (inf/s)",
            "Memory Delta (MB)",
            "Peak RSS (MB)",
            "Speedup vs Baseline",
            "Latency Reduction (%)",
            "Throughput Improvement (%)",
            "Rejection / Infeasibility Reason",
        ])

        for c in self.candidates:
            m = c.get_metrics_dict()
            cfg = c.config
            rejection = c.rejection_reason or ""
            if not rejection and not c.is_feasible and c.constraint_evaluations:
                failed_evals = [e.explanation for e in c.constraint_evaluations if not e.passed]
                rejection = "; ".join(failed_evals)

            writer.writerow([
                c.rank if c.rank is not None else "N/A",
                c.candidate_id,
                c.status.value,
                "YES" if c.is_feasible else "NO",
                "YES" if c.is_pareto_optimal else "NO",
                cfg.execution_provider,
                cfg.batch_size,
                cfg.intra_op_num_threads if cfg.intra_op_num_threads is not None else "Default",
                cfg.inter_op_num_threads if cfg.inter_op_num_threads is not None else "Default",
                f"{m.get('median_latency_ms', 0.0):.3f}" if m.get("median_latency_ms") is not None else "N/A",
                f"{m.get('mean_latency_ms', 0.0):.3f}" if m.get("mean_latency_ms") is not None else "N/A",
                f"{m.get('p95_latency_ms', 0.0):.3f}" if m.get("p95_latency_ms") is not None else "N/A",
                f"{m.get('throughput_inf_sec', 0.0):.1f}" if m.get("throughput_inf_sec") is not None else "N/A",
                f"{m.get('memory_delta_mb', 0.0):.2f}" if m.get("memory_delta_mb") is not None else "N/A",
                f"{m.get('peak_memory_mb', 0.0):.2f}" if m.get("peak_memory_mb") is not None else "N/A",
                f"{c.speedup_vs_baseline:.2f}x" if c.speedup_vs_baseline is not None else "N/A",
                f"{c.latency_reduction_pct:+.1f}%" if c.latency_reduction_pct is not None else "N/A",
                f"{c.throughput_improvement_pct:+.1f}%" if c.throughput_improvement_pct is not None else "N/A",
                rejection,
            ])
        return output.getvalue()


class OptimizationEngine:
    """Core optimization service coordinating candidate generation, filtering, benchmarking, and ranking."""

    @classmethod
    def generate_candidates(
        cls,
        model_summary: ModelSummary,
        request: OptimizationRequest,
        available_providers: List[str],
    ) -> List[ExecutionCandidate]:
        """
        Deterministically generate execution candidates within explicit resource bounds.
        Validates static dimensions, eliminates duplicates, and records generation rationale.
        """
        candidates: List[ExecutionCandidate] = []
        seen_keys = set()
        cand_counter = 0

        # Determine target providers to explore
        target_providers = request.execution_providers or ["CPUExecutionProvider"]
        providers_to_use = [p for p in target_providers if p in available_providers]
        if not providers_to_use:
            providers_to_use = [available_providers[0]] if available_providers else ["CPUExecutionProvider"]

        # Check model static input dimension constraints (especially batch dimension at index 0)
        fixed_batch_dim: Optional[int] = None
        for inp in model_summary.inputs:
            if inp.shape and len(inp.shape) > 0:
                d0 = inp.shape[0]
                if isinstance(d0, int) and d0 > 0:
                    fixed_batch_dim = d0
                    break

        # 1. Always create the baseline candidate: provider default, intra=None, inter=None, batch=1 (or fixed)
        cand_counter += 1
        base_batch = fixed_batch_dim if fixed_batch_dim is not None else 1
        baseline_cfg = BenchmarkConfig(
            execution_provider=providers_to_use[0],
            batch_size=base_batch,
            warmup_iterations=request.warmup_iterations,
            measured_iterations=request.measured_iterations,
            repeated_rounds=request.repeated_rounds,
            random_seed=request.random_seed,
            dynamic_dimensions=dict(request.dynamic_dimensions),
            intra_op_num_threads=None,
            inter_op_num_threads=None,
        )
        base_cand = ExecutionCandidate(
            candidate_id=f"CAND-{cand_counter:02d}-BASELINE",
            config=baseline_cfg,
            generation_reason="Standard default baseline configuration",
            status=CandidateStatus.GENERATED,
            is_baseline=True,
        )
        candidates.append(base_cand)
        key_tuple = (baseline_cfg.execution_provider, baseline_cfg.batch_size, None, None)
        seen_keys.add(key_tuple)

        # 2. Thread scaling variations
        for provider in providers_to_use:
            for intra in request.intra_op_threads:
                for inter in request.inter_op_threads:
                    if len(candidates) >= request.max_candidates:
                        break
                    key = (provider, base_batch, intra, inter)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    cand_counter += 1

                    cfg = BenchmarkConfig(
                        execution_provider=provider,
                        batch_size=base_batch,
                        warmup_iterations=request.warmup_iterations,
                        measured_iterations=request.measured_iterations,
                        repeated_rounds=request.repeated_rounds,
                        random_seed=request.random_seed,
                        dynamic_dimensions=dict(request.dynamic_dimensions),
                        intra_op_num_threads=intra,
                        inter_op_num_threads=inter,
                    )
                    candidates.append(
                        ExecutionCandidate(
                            candidate_id=f"CAND-{cand_counter:02d}",
                            config=cfg,
                            generation_reason=f"Runtime thread tuning: intra={intra}, inter={inter}",
                            status=CandidateStatus.GENERATED,
                        )
                    )

        # 3. Batch scaling variations (if permitted by model input shapes)
        for b in request.batch_sizes:
            if len(candidates) >= request.max_candidates:
                break
            if b == base_batch:
                continue

            cand_counter += 1
            key = (providers_to_use[0], b, None, None)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            cfg = BenchmarkConfig(
                execution_provider=providers_to_use[0],
                batch_size=b,
                warmup_iterations=request.warmup_iterations,
                measured_iterations=request.measured_iterations,
                repeated_rounds=request.repeated_rounds,
                random_seed=request.random_seed,
                dynamic_dimensions=dict(request.dynamic_dimensions),
                intra_op_num_threads=None,
                inter_op_num_threads=None,
            )

            # Check whether model allows this batch size
            if fixed_batch_dim is not None and b != fixed_batch_dim:
                c = ExecutionCandidate(
                    candidate_id=f"CAND-{cand_counter:02d}",
                    config=cfg,
                    generation_reason=f"Batch scaling evaluation (batch_size={b})",
                    status=CandidateStatus.REJECTED,
                    rejection_reason=(
                        f"Model input signature has fixed dimension [0]={fixed_batch_dim}; "
                        f"batch size {b} violates declared static shape constraint."
                    ),
                )
                candidates.append(c)
            else:
                c = ExecutionCandidate(
                    candidate_id=f"CAND-{cand_counter:02d}",
                    config=cfg,
                    generation_reason=f"Batch throughput scaling (batch_size={b})",
                    status=CandidateStatus.GENERATED,
                )
                candidates.append(c)

        # 4. Multi-provider exploration if multiple providers available
        if len(providers_to_use) > 1:
            for alt_provider in providers_to_use[1:]:
                if len(candidates) >= request.max_candidates:
                    break
                key = (alt_provider, base_batch, None, None)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                cand_counter += 1
                cfg = BenchmarkConfig(
                    execution_provider=alt_provider,
                    batch_size=base_batch,
                    warmup_iterations=request.warmup_iterations,
                    measured_iterations=request.measured_iterations,
                    repeated_rounds=request.repeated_rounds,
                    random_seed=request.random_seed,
                    dynamic_dimensions=dict(request.dynamic_dimensions),
                    intra_op_num_threads=None,
                    inter_op_num_threads=None,
                )
                candidates.append(
                    ExecutionCandidate(
                        candidate_id=f"CAND-{cand_counter:02d}",
                        config=cfg,
                        generation_reason=f"Alternative execution provider ({alt_provider})",
                        status=CandidateStatus.GENERATED,
                    )
                )

        return candidates

    @classmethod
    def filter_candidates_by_compatibility(
        cls,
        candidates: List[ExecutionCandidate],
        model_summary: ModelSummary,
        profile: CapabilityProfile,
    ) -> List[ExecutionCandidate]:
        """
        Evaluate candidates against the target capability profile using Phase 3 engine.
        Rejects candidates that violate compatibility rules or target unsupported providers.
        """
        compat_report: CompatibilityReport = CompatibilityAnalyzer.analyze(model_summary, profile)
        has_fail = any(f.status == "FAIL" for f in compat_report.findings)

        for cand in candidates:
            if cand.status == CandidateStatus.REJECTED:
                continue

            cand.compatibility_findings = [f.to_dict() for f in compat_report.findings]

            if has_fail:
                fail_ops = [f.op_type for f in compat_report.findings if f.status == "FAIL" and f.op_type]
                profile_title = getattr(profile, "display_name", getattr(profile, "name", "Profile"))
                cand.status = CandidateStatus.REJECTED
                cand.rejection_reason = (
                    f"Profile '{profile_title}' has FAIL findings for operators: {fail_ops}. "
                    f"Candidate cannot execute on target profile."
                )
                continue

            req_provider = cand.config.execution_provider
            supp_providers = getattr(profile, "execution_providers", None)
            if supp_providers and req_provider not in supp_providers:
                cand.status = CandidateStatus.REJECTED
                cand.rejection_reason = (
                    f"Execution provider '{req_provider}' is not listed in profile's "
                    f"supported providers ({supp_providers})."
                )
                continue

            cand.status = CandidateStatus.QUEUED

        return candidates

    @classmethod
    def evaluate_constraints(
        cls,
        candidate: ExecutionCandidate,
        hard_constraints: List[HardConstraint],
    ) -> Tuple[bool, List[ConstraintEvaluation]]:
        """
        Evaluate all configured hard constraints against the candidate's observed metrics.
        Returns (is_feasible, evaluations_list).
        """
        evaluations: List[ConstraintEvaluation] = []
        if not candidate.benchmark_report or candidate.benchmark_report.status != "SUCCESS":
            candidate.is_feasible = False
            return False, evaluations

        metrics = candidate.get_metrics_dict()
        all_passed = True

        if not metrics.get("correctness_valid", False):
            all_passed = False
            evaluations.append(
                ConstraintEvaluation(
                    constraint_name="Inference Correctness",
                    metric_field="correctness_valid",
                    threshold=1.0,
                    observed_value=0.0,
                    unit="bool",
                    passed=False,
                    is_available=True,
                    explanation="Inference correctness check failed (non-finite output or runtime error).",
                )
            )

        for hc in hard_constraints:
            ev = hc.evaluate(metrics)
            evaluations.append(ev)
            if not ev.passed:
                all_passed = False

        candidate.constraint_evaluations = evaluations
        candidate.is_feasible = all_passed
        if not all_passed and candidate.status == CandidateStatus.BENCHMARKED:
            candidate.status = CandidateStatus.INFEASIBLE

        return all_passed, evaluations

    @classmethod
    def calculate_pareto_frontier(cls, candidates: List[ExecutionCandidate]) -> List[str]:
        """
        Identify the Pareto-optimal subset of feasible candidates.
        Considers Latency (minimize), Throughput (maximize), and Peak Memory (minimize, if available).
        """
        feasible = [c for c in candidates if c.is_feasible and c.benchmark_report]
        if not feasible:
            return []

        frontier_ids: List[str] = []

        for i, cand_a in enumerate(feasible):
            metrics_a = cand_a.get_metrics_dict()
            lat_a = metrics_a.get("median_latency_ms")
            tp_a = metrics_a.get("throughput_inf_sec")
            mem_a = metrics_a.get("peak_memory_mb")

            if lat_a is None or tp_a is None:
                continue

            dominated = False
            for j, cand_b in enumerate(feasible):
                if i == j:
                    continue
                metrics_b = cand_b.get_metrics_dict()
                lat_b = metrics_b.get("median_latency_ms")
                tp_b = metrics_b.get("throughput_inf_sec")
                mem_b = metrics_b.get("peak_memory_mb")

                if lat_b is None or tp_b is None:
                    continue

                b_as_good_lat = lat_b <= (lat_a + 1e-9)
                b_as_good_tp = tp_b >= (tp_a - 1e-9)
                b_as_good_mem = True
                b_better_mem = False
                if mem_a is not None and mem_b is not None:
                    b_as_good_mem = mem_b <= (mem_a + 1e-9)
                    b_better_mem = mem_b < (mem_a - 1e-9)

                b_strictly_better = (
                    (lat_b < (lat_a - 1e-9)) or
                    (tp_b > (tp_a + 1e-9)) or
                    b_better_mem
                )

                if b_as_good_lat and b_as_good_tp and b_as_good_mem and b_strictly_better:
                    dominated = True
                    break

            if not dominated:
                cand_a.is_pareto_optimal = True
                frontier_ids.append(cand_a.candidate_id)
            else:
                cand_a.is_pareto_optimal = False

        return frontier_ids

    @classmethod
    def calculate_relative_improvements(
        cls,
        candidate: ExecutionCandidate,
        baseline_candidate: Optional[ExecutionCandidate],
    ) -> None:
        """Calculate speedup and latency reduction vs baseline configuration."""
        if not baseline_candidate or not baseline_candidate.benchmark_report:
            return
        if not candidate.benchmark_report or (candidate.status != CandidateStatus.BENCHMARKED and not candidate.is_feasible):
            return

        base_metrics = baseline_candidate.get_metrics_dict()
        cand_metrics = candidate.get_metrics_dict()

        base_lat = base_metrics.get("median_latency_ms")
        cand_lat = cand_metrics.get("median_latency_ms")

        base_tp = base_metrics.get("throughput_inf_sec")
        cand_tp = cand_metrics.get("throughput_inf_sec")

        if base_lat is not None and cand_lat is not None and base_lat > 0 and cand_lat > 0:
            candidate.speedup_vs_baseline = float(base_lat / cand_lat)
            candidate.latency_reduction_pct = float(((base_lat - cand_lat) / base_lat) * 100.0)

        if base_tp is not None and cand_tp is not None and base_tp > 0:
            candidate.throughput_improvement_pct = float(((cand_tp - base_tp) / base_tp) * 100.0)

    @classmethod
    def rank_candidates(
        cls,
        candidates: List[ExecutionCandidate],
        objective: OptimizationObjective,
    ) -> List[ExecutionCandidate]:
        """Rank feasible candidates based on primary objective with deterministic tie-breaking."""
        feasible = [c for c in candidates if c.is_feasible]
        infeasible = [c for c in candidates if not c.is_feasible]

        def sort_key(c: ExecutionCandidate) -> Tuple[Any, ...]:
            m = c.get_metrics_dict()
            lat = m.get("median_latency_ms", float("inf"))
            p95 = m.get("p95_latency_ms", float("inf"))
            tp = m.get("throughput_inf_sec", 0.0)
            mem = m.get("peak_memory_mb", float("inf"))

            if objective == OptimizationObjective.MINIMIZE_MEDIAN_LATENCY:
                return (lat, -tp, c.candidate_id)
            elif objective == OptimizationObjective.MINIMIZE_P95_LATENCY:
                return (p95, lat, -tp, c.candidate_id)
            elif objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
                return (-tp, lat, c.candidate_id)
            elif objective == OptimizationObjective.MINIMIZE_MEMORY:
                return (mem, lat, -tp, c.candidate_id)
            else:
                return (0 if c.is_pareto_optimal else 1, lat, -tp, c.candidate_id)

        feasible_sorted = sorted(feasible, key=sort_key)
        for rank_idx, c in enumerate(feasible_sorted, start=1):
            c.rank = rank_idx

        for c in infeasible:
            c.rank = None

        return feasible_sorted + infeasible

    @classmethod
    def generate_recommendation_rationale(
        cls,
        best_candidate: Optional[ExecutionCandidate],
        baseline_candidate: Optional[ExecutionCandidate],
        objective: OptimizationObjective,
        profile: CapabilityProfile,
        feasible_count: int,
        total_benchmarked: int,
    ) -> str:
        """Generate human-readable, evidence-based explanation for the recommended configuration."""
        if not best_candidate:
            prof_name = getattr(profile, "display_name", getattr(profile, "name", "Profile"))
            return (
                f"No feasible candidate satisfied all configured operational constraints and correctness checks. "
                f"Evaluated {total_benchmarked} benchmarked candidates against {prof_name}. "
                f"Review the constraint violations table below to identify bottlenecks or adjust limits."
            )

        m = best_candidate.get_metrics_dict()
        cfg = best_candidate.config
        obj_name = objective.value.replace("_", " ").title()

        speedup_str = ""
        if best_candidate.speedup_vs_baseline is not None:
            if best_candidate.speedup_vs_baseline >= 1.0:
                speedup_str = (
                    f"providing a {best_candidate.speedup_vs_baseline:.2f}x speedup "
                    f"({best_candidate.latency_reduction_pct:+.1f}% latency reduction) "
                    f"over the standard baseline"
                )
            else:
                speedup_str = (
                    f"representing a regression ({best_candidate.speedup_vs_baseline:.2f}x relative to baseline) "
                    f"while satisfying operational constraints"
                )

        rationale = (
            f"Candidate '{best_candidate.candidate_id}' is recommended for objective '{obj_name}' {speedup_str}. "
            f"Measured execution metrics: Median Latency = {m.get('median_latency_ms', 0.0):.3f} ms, "
            f"P95 Latency = {m.get('p95_latency_ms', 0.0):.3f} ms, "
            f"Throughput = {m.get('throughput_inf_sec', 0.0):.1f} inf/sec. "
            f"Configuration: Provider = {cfg.execution_provider}, Batch Size = {cfg.batch_size}, "
            f"Intra-op Threads = {cfg.intra_op_num_threads or 'Default'}, "
            f"Inter-op Threads = {cfg.inter_op_num_threads or 'Default'}. "
            f"All {len(best_candidate.constraint_evaluations)} operational constraints and correctness validations were satisfied."
        )

        if profile.is_illustrative:
            prof_title = getattr(profile, "display_name", getattr(profile, "name", "Profile"))
            rationale += (
                f" Note: Profile '{prof_title}' is illustrative draft; "
                f"local execution was performed on host {cfg.execution_provider} and does not establish IBM Z performance."
            )

        return rationale

    @classmethod
    def run_optimization(
        cls,
        model_path_or_bytes: Union[str, bytes],
        model_summary: ModelSummary,
        profile: CapabilityProfile,
        request: OptimizationRequest,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> OptimizationReport:
        """Execute full end-to-end optimization experiment."""
        request.validate()
        experiment_id = f"OPT-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 10000:04d}"
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        env_info = discover_execution_providers()
        available_providers = env_info.available_providers

        # 1. Candidate Generation
        if progress_callback:
            progress_callback("Generating valid execution candidates...", 0, 100)
        candidates = cls.generate_candidates(model_summary, request, available_providers)

        # 2. Compatibility Filtering
        if progress_callback:
            progress_callback("Evaluating candidate compatibility against target profile...", 15, 100)
        candidates = cls.filter_candidates_by_compatibility(candidates, model_summary, profile)

        # 3. Real Benchmarking
        queued_candidates = [c for c in candidates if c.status == CandidateStatus.QUEUED]
        total_to_bench = len(queued_candidates)

        baseline_candidate: Optional[ExecutionCandidate] = None

        for idx, cand in enumerate(queued_candidates, start=1):
            if progress_callback:
                pct = 20 + int((idx / max(total_to_bench, 1)) * 60)
                progress_callback(
                    f"Benchmarking candidate {idx}/{total_to_bench}: {cand.candidate_id} "
                    f"({cand.config.execution_provider}, threads={cand.config.intra_op_num_threads}, batch={cand.config.batch_size})...",
                    pct,
                    100,
                )

            cand.status = CandidateStatus.RUNNING
            try:
                report = BenchmarkEngine.run_benchmark(
                    model_path_or_bytes=model_path_or_bytes,
                    model_summary=model_summary,
                    config=cand.config,
                )
                cand.benchmark_report = report
                cand.status = CandidateStatus.BENCHMARKED
            except CorrectnessValidationError as cve:
                cand.status = CandidateStatus.FAILED
                cand.rejection_reason = f"Correctness validation failed: {str(cve)}"
            except BenchmarkError as be:
                cand.status = CandidateStatus.FAILED
                cand.rejection_reason = f"Runtime benchmark error: {str(be)}"
            except Exception as e:
                cand.status = CandidateStatus.FAILED
                cand.rejection_reason = f"Unexpected execution error: {str(e)}"

            if cand.is_baseline and cand.status == CandidateStatus.BENCHMARKED:
                baseline_candidate = cand

        if baseline_candidate is None:
            benchmarked = [c for c in candidates if c.status == CandidateStatus.BENCHMARKED]
            if benchmarked:
                baseline_candidate = benchmarked[0]
                baseline_candidate.is_baseline = True

        # 4. Evaluate Hard Constraints
        if progress_callback:
            progress_callback("Evaluating operational boundaries and constraints...", 82, 100)

        for cand in candidates:
            if cand.status == CandidateStatus.BENCHMARKED:
                cls.evaluate_constraints(cand, request.hard_constraints)

        # 5. Pareto Frontier Calculation
        if progress_callback:
            progress_callback("Calculating Pareto multi-objective trade-off frontier...", 88, 100)
        pareto_frontier_ids = cls.calculate_pareto_frontier(candidates)

        # 6. Relative Improvements vs Baseline
        for cand in candidates:
            cls.calculate_relative_improvements(cand, baseline_candidate)

        # 7. Candidate Ranking
        if progress_callback:
            progress_callback("Ranking feasible candidates according to primary objective...", 94, 100)
        ranked_candidates = cls.rank_candidates(candidates, request.primary_objective)

        feasible_candidates = [c for c in ranked_candidates if c.is_feasible]
        best_candidate = feasible_candidates[0] if feasible_candidates else None

        # 8. Summary Counts
        summary_counts = {
            "total_generated": len(candidates),
            "rejected": sum(1 for c in candidates if c.status == CandidateStatus.REJECTED),
            "benchmarked": sum(1 for c in candidates if c.benchmark_report is not None),
            "feasible": len(feasible_candidates),
            "infeasible": sum(1 for c in candidates if c.status == CandidateStatus.INFEASIBLE),
            "failed": sum(1 for c in candidates if c.status == CandidateStatus.FAILED),
            "pareto_count": len(pareto_frontier_ids),
        }

        # 9. Recommendation Rationale
        rationale = cls.generate_recommendation_rationale(
            best_candidate=best_candidate,
            baseline_candidate=baseline_candidate,
            objective=request.primary_objective,
            profile=profile,
            feasible_count=len(feasible_candidates),
            total_benchmarked=summary_counts["benchmarked"],
        )

        if summary_counts["feasible"] > 0:
            status = ExperimentStatus.COMPLETED
            if profile.is_illustrative or any(c.compatibility_findings for c in candidates):
                status = ExperimentStatus.COMPLETED_WITH_WARNINGS
        elif summary_counts["benchmarked"] > 0:
            status = ExperimentStatus.COMPLETED_WITH_WARNINGS
        else:
            status = ExperimentStatus.FAILED

        warnings: List[str] = []
        limitations: List[str] = [
            "Local benchmarks executed on host CPUExecutionProvider reflect host CPU architecture.",
            "Local CPU benchmarking does NOT prove or represent execution on IBM Z (s390x) or NNPA hardware.",
            "Memory metrics represent process RSS variations and include Python/Streamlit overhead.",
            "Inference timing measures strictly runtime session.run() boundary; input generation is excluded.",
        ]

        host_os = platform.system()
        if profile.is_illustrative:
            prof_title = getattr(profile, "display_name", getattr(profile, "name", "Profile"))
            warnings.append(
                f"Selected profile '{prof_title}' is an illustrative draft. "
                f"Measurements were conducted on local {host_os} host."
            )
        if not feasible_candidates:
            warnings.append("No feasible candidates satisfied all mandatory operational constraints.")

        report = OptimizationReport(
            experiment_id=experiment_id,
            timestamp_utc=timestamp_utc,
            status=status,
            model_name=model_summary.filename,
            model_fingerprint=model_summary.sha256_hash,
            profile_id=getattr(profile, "profile_id", getattr(profile, "id", "")),
            profile_name=getattr(profile, "display_name", getattr(profile, "name", "")),
            profile_version=profile.version,
            primary_objective=request.primary_objective.value,
            request=request,
            baseline_candidate_id=baseline_candidate.candidate_id if baseline_candidate else None,
            best_candidate_id=best_candidate.candidate_id if best_candidate else None,
            candidates=ranked_candidates,
            pareto_frontier_ids=pareto_frontier_ids,
            summary_counts=summary_counts,
            recommendation_rationale=rationale,
            environment={
                "os_name": host_os,
                "os_release": platform.release(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "onnxruntime_version": env_info.installed_ort_version,
                "available_providers": env_info.available_providers,
                "cpu_count_logical": psutil.cpu_count(logical=True) if hasattr(psutil, "cpu_count") else "unknown",
            },
            warnings=warnings,
            limitations=limitations,
        )

        if progress_callback:
            progress_callback("Optimization experiment completed successfully.", 100, 100)

        return report


# Re-export Phase 12 Advanced Optimization components for unified access
try:
    from zmodel.core.advanced_optimizer import (
        AdvancedOptimizationEngine,
        AdvancedParetoAnalyzer,
        EnterpriseReportGenerator,
        EvidenceRecommendation,
        MeasurementReliability,
        ParameterSensitivity,
        RecommendationEngine,
        RecommendationGoal,
        ReliabilityAnalyzer,
        ReliabilityRating,
        SearchStrategy,
        SensitivityAnalyzer,
        SensitivityPoint,
        SensitivityReport,
    )
except ImportError:
    pass

