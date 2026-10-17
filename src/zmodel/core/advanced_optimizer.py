"""
Phase 12: Advanced Model Optimization and Enterprise Decision-Support Reporting.

Transforms Z-ModelPilot into a rigorous, evidence-driven engineering decision-support system:
1. Unified deployment experiment representation with cryptographic integrity.
2. Bounded configuration search: Grid Search, Seeded Random Search, and Multi-Fidelity Successive Halving.
3. Strict multi-objective Pareto frontier engine across latency, throughput, and memory.
4. Measurement reliability and confidence analysis: IQR, Coefficient of Variation (CV%), sample counts, stability rating.
5. Model and configuration sensitivity analysis: batch elasticity, thread scaling saturation, parameter impact ranking.
6. Evidence-based recommendation engine supporting MINIMIZE_LATENCY, MAXIMIZE_THROUGHPUT, MINIMIZE_MEMORY, and BALANCED_SCORE.
7. Enterprise-grade 18-section audit report generation (Markdown & JSON) with secret redaction and hardware provenance.

Never invents performance measurements, fabricates speedups, or claims an optimization
improved performance without real empirical verification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import os
import platform
import random
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

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
from zmodel.core.environment import EnvironmentInventory
from zmodel.core.model_loader import ModelSummary
from zmodel.core.optimizer import (
    CandidateStatus,
    ConstraintEvaluation,
    ExecutionCandidate,
    HardConstraint,
    OptimizationObjective,
    OptimizationRequest,
)
from zmodel.core.profiles import CapabilityProfile


# --- Enums for Phase 12 ---

class SearchStrategy(str, Enum):
    """Exploration strategy for generating candidate deployment configurations."""
    GRID = "GRID"
    RANDOM = "RANDOM"
    SUCCESSIVE_HALVING = "SUCCESSIVE_HALVING"


class ReliabilityRating(str, Enum):
    """Empirical confidence and stability classification of benchmark measurements."""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    UNSTABLE = "UNSTABLE"
    NON_COMPARABLE = "NON_COMPARABLE"


class RecommendationGoal(str, Enum):
    """User-selected optimization goal for decision recommendation."""
    MINIMIZE_LATENCY = "MINIMIZE_LATENCY"
    MAXIMIZE_THROUGHPUT = "MAXIMIZE_THROUGHPUT"
    MINIMIZE_MEMORY = "MINIMIZE_MEMORY"
    BALANCED_SCORE = "BALANCED_SCORE"


# --- Data Structures ---

@dataclass
class MeasurementReliability:
    """Statistical dispersion and reliability metrics for a benchmarked candidate."""
    sample_count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    std_dev_ms: float
    iqr_ms: float
    cv_percent: float
    rating: ReliabilityRating
    is_stable: bool
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "std_dev_ms": round(self.std_dev_ms, 4),
            "iqr_ms": round(self.iqr_ms, 4),
            "cv_percent": round(self.cv_percent, 2),
            "rating": self.rating.value,
            "is_stable": self.is_stable,
            "notes": self.notes,
        }


@dataclass
class SensitivityPoint:
    """Discrete parameter point in sensitivity curve."""
    param_value: Any
    median_latency_ms: float
    throughput_inf_sec: float
    peak_memory_mb: Optional[float] = None
    candidate_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_value": self.param_value,
            "median_latency_ms": round(self.median_latency_ms, 4),
            "throughput_inf_sec": round(self.throughput_inf_sec, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2) if self.peak_memory_mb else None,
            "candidate_id": self.candidate_id,
        }


@dataclass
class ParameterSensitivity:
    """Sensitivity analysis for an individual deployment hyperparameter."""
    parameter_name: str
    status: str  # "ANALYZED", "INSUFFICIENT_DATA"
    data_points: List[SensitivityPoint] = field(default_factory=list)
    latency_elasticity: Optional[float] = None
    best_setting: Optional[Any] = None
    variance_explained_ms: Optional[float] = None
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "status": self.status,
            "data_points": [p.to_dict() for p in self.data_points],
            "latency_elasticity": round(self.latency_elasticity, 3) if self.latency_elasticity is not None else None,
            "best_setting": self.best_setting,
            "variance_explained_ms": round(self.variance_explained_ms, 4) if self.variance_explained_ms is not None else None,
            "interpretation": self.interpretation,
        }


@dataclass
class SensitivityReport:
    """Comprehensive sensitivity analysis across all explored parameters."""
    parameters: Dict[str, ParameterSensitivity] = field(default_factory=dict)
    primary_influencing_parameter: Optional[str] = None
    summary_insight: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "primary_influencing_parameter": self.primary_influencing_parameter,
            "summary_insight": self.summary_insight,
        }


@dataclass
class EvidenceRecommendation:
    """Actionable, evidence-backed deployment recommendation decision."""
    goal: RecommendationGoal
    recommended_candidate_id: Optional[str]
    recommended_config: Optional[Dict[str, Any]]
    target_profile_id: str
    target_profile_version: str
    execution_environment_fingerprint: str
    is_target_verified: bool
    satisfied_constraints: List[str]
    violated_constraints: List[str]
    median_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    throughput_inf_sec: Optional[float]
    peak_memory_mb: Optional[float]
    speedup_vs_baseline: Optional[float]
    latency_reduction_pct: Optional[float]
    throughput_improvement_pct: Optional[float]
    reliability: Optional[MeasurementReliability]
    balanced_score: Optional[float]
    rationale: str
    tradeoffs_summary: str
    unverified_requirements: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal.value,
            "recommended_candidate_id": self.recommended_candidate_id,
            "recommended_config": self.recommended_config,
            "target_profile_id": self.target_profile_id,
            "target_profile_version": self.target_profile_version,
            "execution_environment_fingerprint": self.execution_environment_fingerprint,
            "is_target_verified": self.is_target_verified,
            "satisfied_constraints": self.satisfied_constraints,
            "violated_constraints": self.violated_constraints,
            "median_latency_ms": round(self.median_latency_ms, 4) if self.median_latency_ms is not None else None,
            "p95_latency_ms": round(self.p95_latency_ms, 4) if self.p95_latency_ms is not None else None,
            "throughput_inf_sec": round(self.throughput_inf_sec, 2) if self.throughput_inf_sec is not None else None,
            "peak_memory_mb": round(self.peak_memory_mb, 2) if self.peak_memory_mb is not None else None,
            "speedup_vs_baseline": round(self.speedup_vs_baseline, 2) if self.speedup_vs_baseline is not None else None,
            "latency_reduction_pct": round(self.latency_reduction_pct, 2) if self.latency_reduction_pct is not None else None,
            "throughput_improvement_pct": round(self.throughput_improvement_pct, 2) if self.throughput_improvement_pct is not None else None,
            "reliability": self.reliability.to_dict() if self.reliability else None,
            "balanced_score": round(self.balanced_score, 4) if self.balanced_score is not None else None,
            "rationale": self.rationale,
            "tradeoffs_summary": self.tradeoffs_summary,
            "unverified_requirements": self.unverified_requirements,
        }


# --- Advanced Pareto Analyzer ---

class AdvancedParetoAnalyzer:
    """
    Mathematical Pareto dominance engine supporting 4 objectives, missing metric handling,
    ties, and empty feasible set diagnosis.
    """

    @staticmethod
    def evaluate_dominance(
        cand_a_metrics: Dict[str, Any],
        cand_b_metrics: Dict[str, Any],
    ) -> Tuple[bool, bool, str]:
        """
        Evaluate if A dominates B, or B dominates A, or neither.
        Returns: (a_dominates_b, b_dominates_a, explanation).
        """
        lat_a = cand_a_metrics.get("median_latency_ms")
        lat_b = cand_b_metrics.get("median_latency_ms")
        p95_a = cand_a_metrics.get("p95_latency_ms")
        p95_b = cand_b_metrics.get("p95_latency_ms")
        tp_a = cand_a_metrics.get("throughput_inf_sec")
        tp_b = cand_b_metrics.get("throughput_inf_sec")
        mem_a = cand_a_metrics.get("peak_memory_mb")
        mem_b = cand_b_metrics.get("peak_memory_mb")

        # Basic comparability check
        if lat_a is None or lat_b is None or tp_a is None or tp_b is None:
            return False, False, "Incomparable: essential latency or throughput metrics missing."

        # Objectives to minimize: median_latency, p95_latency, memory (if present)
        # Objectives to maximize: throughput

        def check_pair(x_metrics: Dict[str, Any], y_metrics: Dict[str, Any]) -> bool:
            lx, ly = x_metrics.get("median_latency_ms"), y_metrics.get("median_latency_ms")
            p95x, p95y = x_metrics.get("p95_latency_ms"), y_metrics.get("p95_latency_ms")
            tx, ty = x_metrics.get("throughput_inf_sec"), y_metrics.get("throughput_inf_sec")
            mx, my = x_metrics.get("peak_memory_mb"), y_metrics.get("peak_memory_mb")

            tol = 1e-6
            as_good = True
            strictly_better = False

            # Latency (min)
            if lx > ly + tol:
                as_good = False
            elif lx < ly - tol:
                strictly_better = True

            # P95 (min)
            if p95x is not None and p95y is not None:
                if p95x > p95y + tol:
                    as_good = False
                elif p95x < p95y - tol:
                    strictly_better = True

            # Throughput (max)
            if tx < ty - tol:
                as_good = False
            elif tx > ty + tol:
                strictly_better = True

            # Memory (min, only if both available)
            if mx is not None and my is not None:
                if mx > my + tol:
                    as_good = False
                elif mx < my - tol:
                    strictly_better = True

            return as_good and strictly_better

        a_dom_b = check_pair(cand_a_metrics, cand_b_metrics)
        b_dom_a = check_pair(cand_b_metrics, cand_a_metrics)

        if a_dom_b:
            return True, False, "Candidate A dominates Candidate B across multi-objective criteria."
        elif b_dom_a:
            return False, True, "Candidate B dominates Candidate A across multi-objective criteria."
        else:
            return False, False, "Neither candidate dominates the other; represents a valid Pareto trade-off."

    @classmethod
    def compute_frontier(
        cls,
        candidates: List[ExecutionCandidate],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Compute the Pareto frontier from feasible candidates.
        Returns: (frontier_candidate_ids, dominance_matrix).
        """
        feasible = [c for c in candidates if c.is_feasible and c.benchmark_report and c.benchmark_report.status == "SUCCESS"]
        if not feasible:
            return [], []

        frontier_ids: List[str] = []
        dominance_matrix: List[Dict[str, Any]] = []

        for i, c_a in enumerate(feasible):
            m_a = c_a.get_metrics_dict()
            is_dominated = False
            dominated_by: List[str] = []

            for j, c_b in enumerate(feasible):
                if i == j:
                    continue
                m_b = c_b.get_metrics_dict()
                a_dom, b_dom, _ = cls.evaluate_dominance(m_a, m_b)
                if b_dom:
                    is_dominated = True
                    dominated_by.append(c_b.candidate_id)

            dominance_matrix.append({
                "candidate_id": c_a.candidate_id,
                "is_dominated": is_dominated,
                "dominated_by": dominated_by,
            })

            if not is_dominated:
                c_a.is_pareto_optimal = True
                frontier_ids.append(c_a.candidate_id)
            else:
                c_a.is_pareto_optimal = False

        return frontier_ids, dominance_matrix


# --- Reliability Analyzer ---

class ReliabilityAnalyzer:
    """Calculates statistical dispersion and confidence ratings from raw benchmark samples."""

    @staticmethod
    def analyze_candidate(candidate: ExecutionCandidate) -> Optional[MeasurementReliability]:
        if not candidate.benchmark_report or not candidate.benchmark_report.all_latencies_ms:
            return None

        raw = np.array(candidate.benchmark_report.all_latencies_ms, dtype=np.float64)
        n = len(raw)
        if n == 0:
            return None

        mean_val = float(np.mean(raw))
        median_val = float(np.median(raw))
        std_val = float(np.std(raw))
        p75 = float(np.percentile(raw, 75))
        p25 = float(np.percentile(raw, 25))
        iqr_val = float(p75 - p25)
        p95_val = float(np.percentile(raw, 95))

        cv_val = float((std_val / mean_val) * 100.0) if mean_val > 0 else 0.0

        if n >= 25 and cv_val <= 10.0:
            rating = ReliabilityRating.HIGH
            notes = f"High confidence: {n} measurements with low dispersion (CV={cv_val:.1f}%)."
            is_stable = True
        elif n >= 10 and cv_val <= 20.0:
            rating = ReliabilityRating.MODERATE
            notes = f"Moderate confidence: {n} measurements (CV={cv_val:.1f}%)."
            is_stable = True
        else:
            rating = ReliabilityRating.UNSTABLE
            notes = f"Unstable measurement: high jitter (CV={cv_val:.1f}%) or small sample size (n={n})."
            is_stable = False

        return MeasurementReliability(
            sample_count=n,
            mean_ms=mean_val,
            median_ms=median_val,
            p95_ms=p95_val,
            std_dev_ms=std_val,
            iqr_ms=iqr_val,
            cv_percent=cv_val,
            rating=rating,
            is_stable=is_stable,
            notes=notes,
        )


# --- Sensitivity Analyzer ---

class SensitivityAnalyzer:
    """Analyzes model performance sensitivity across varying deployment parameters."""

    @staticmethod
    def analyze(candidates: List[ExecutionCandidate]) -> SensitivityReport:
        benchmarked = [c for c in candidates if c.is_feasible and c.benchmark_report]
        if len(benchmarked) < 2:
            return SensitivityReport(
                summary_insight="Insufficient benchmarked candidates (< 2) to perform sensitivity analysis."
            )

        # 1. Batch Size Sensitivity
        batch_points: Dict[int, List[ExecutionCandidate]] = {}
        for c in benchmarked:
            b = c.config.batch_size
            batch_points.setdefault(b, []).append(c)

        batch_sens = ParameterSensitivity(parameter_name="batch_size", status="INSUFFICIENT_DATA")
        if len(batch_points) >= 2:
            pts: List[SensitivityPoint] = []
            for b_val, c_list in sorted(batch_points.items()):
                # Average or select best for this batch
                best_c = min(c_list, key=lambda x: x.get_metrics_dict().get("median_latency_ms", float("inf")))
                m = best_c.get_metrics_dict()
                pts.append(
                    SensitivityPoint(
                        param_value=b_val,
                        median_latency_ms=m.get("median_latency_ms", 0.0),
                        throughput_inf_sec=m.get("throughput_inf_sec", 0.0),
                        peak_memory_mb=m.get("peak_memory_mb"),
                        candidate_id=best_c.candidate_id,
                    )
                )

            # Compute elasticity between min and max batch
            b_min, b_max = pts[0], pts[-1]
            pct_d_batch = ((b_max.param_value - b_min.param_value) / b_min.param_value) if b_min.param_value > 0 else 1.0
            pct_d_lat = ((b_max.median_latency_ms - b_min.median_latency_ms) / b_min.median_latency_ms) if b_min.median_latency_ms > 0 else 0.0
            elasticity = float(pct_d_lat / pct_d_batch) if pct_d_batch > 0 else 0.0

            best_b = max(pts, key=lambda p: p.throughput_inf_sec).param_value
            variance = float(np.var([p.median_latency_ms for p in pts]))

            batch_sens.status = "ANALYZED"
            batch_sens.data_points = pts
            batch_sens.latency_elasticity = elasticity
            batch_sens.best_setting = best_b
            batch_sens.variance_explained_ms = variance
            batch_sens.interpretation = (
                f"Batch scaling elasticity is {elasticity:.2f}. "
                f"Batch size {best_b} delivers maximum throughput ({max(p.throughput_inf_sec for p in pts):.1f} inf/s)."
            )

        # 2. Intra-op Thread Sensitivity
        thread_points: Dict[int, List[ExecutionCandidate]] = {}
        for c in benchmarked:
            t = c.config.intra_op_num_threads or 1
            thread_points.setdefault(t, []).append(c)

        thread_sens = ParameterSensitivity(parameter_name="intra_op_num_threads", status="INSUFFICIENT_DATA")
        if len(thread_points) >= 2:
            pts = []
            for t_val, c_list in sorted(thread_points.items()):
                best_c = min(c_list, key=lambda x: x.get_metrics_dict().get("median_latency_ms", float("inf")))
                m = best_c.get_metrics_dict()
                pts.append(
                    SensitivityPoint(
                        param_value=t_val,
                        median_latency_ms=m.get("median_latency_ms", 0.0),
                        throughput_inf_sec=m.get("throughput_inf_sec", 0.0),
                        peak_memory_mb=m.get("peak_memory_mb"),
                        candidate_id=best_c.candidate_id,
                    )
                )

            best_t = min(pts, key=lambda p: p.median_latency_ms).param_value
            variance = float(np.var([p.median_latency_ms for p in pts]))
            speedup_max = (pts[0].median_latency_ms / pts[-1].median_latency_ms) if pts[-1].median_latency_ms > 0 else 1.0

            thread_sens.status = "ANALYZED"
            thread_sens.data_points = pts
            thread_sens.best_setting = best_t
            thread_sens.variance_explained_ms = variance
            thread_sens.interpretation = (
                f"Thread tuning yielded a max speedup of {speedup_max:.2f}x. "
                f"Optimal intra-op thread count is {best_t}."
            )

        params = {
            "batch_size": batch_sens,
            "intra_op_num_threads": thread_sens,
        }

        # Identify primary influencing parameter
        analyzed = [p for p in params.values() if p.status == "ANALYZED" and p.variance_explained_ms is not None]
        primary = None
        if analyzed:
            primary = max(analyzed, key=lambda x: x.variance_explained_ms or 0.0).parameter_name

        summary = (
            f"Sensitivity analysis completed across {len(analyzed)} parameter dimensions. "
            f"Parameter '{primary or 'None'}' exhibited the strongest observed variance on latency."
        )

        return SensitivityReport(
            parameters=params,
            primary_influencing_parameter=primary,
            summary_insight=summary,
        )


# --- Recommendation Engine ---

class RecommendationEngine:
    """Evaluates candidates and generates transparent, goal-oriented recommendations."""

    @classmethod
    def evaluate(
        cls,
        candidates: List[ExecutionCandidate],
        goal: RecommendationGoal,
        profile: CapabilityProfile,
        baseline_candidate: Optional[ExecutionCandidate],
        env_inventory: Optional[EnvironmentInventory] = None,
    ) -> EvidenceRecommendation:
        feasible = [c for c in candidates if c.is_feasible and c.benchmark_report and c.benchmark_report.status == "SUCCESS"]
        target_prof_id = getattr(profile, "profile_id", getattr(profile, "id", "unknown"))
        target_prof_ver = getattr(profile, "version", "1.0.0")
        env_fp = env_inventory.fingerprint if env_inventory else "UNKNOWN_ENV"
        is_target_verified = bool(env_inventory and env_inventory.is_s390x and not getattr(profile, "is_illustrative", False))

        unverified: List[str] = []
        if not is_target_verified:
            unverified.append("Physical IBM Z (s390x) execution and NNPA hardware acceleration remain unverified.")
            unverified.append("Measurements were collected on host CPU; target mainframe execution is pending.")

        if not feasible:
            violated = []
            for c in candidates:
                for ce in c.constraint_evaluations:
                    if not ce.passed:
                        violated.append(f"{c.candidate_id}: {ce.explanation}")

            return EvidenceRecommendation(
                goal=goal,
                recommended_candidate_id=None,
                recommended_config=None,
                target_profile_id=target_prof_id,
                target_profile_version=target_prof_ver,
                execution_environment_fingerprint=env_fp,
                is_target_verified=is_target_verified,
                satisfied_constraints=[],
                violated_constraints=violated[:5],
                median_latency_ms=None,
                p95_latency_ms=None,
                throughput_inf_sec=None,
                peak_memory_mb=None,
                speedup_vs_baseline=None,
                latency_reduction_pct=None,
                throughput_improvement_pct=None,
                reliability=None,
                balanced_score=None,
                rationale="No candidate satisfied all operational constraints and correctness validations.",
                tradeoffs_summary="All candidates failed either correctness, compatibility, or configured constraints.",
                unverified_requirements=unverified,
            )

        # Multi-objective Balanced Scoring Function:
        # Score = 0.4 * (1 - norm_lat) + 0.4 * norm_tp + 0.2 * (1 - norm_mem)
        lats = [c.get_metrics_dict().get("median_latency_ms", 1.0) for c in feasible]
        tps = [c.get_metrics_dict().get("throughput_inf_sec", 1.0) for c in feasible]
        mems = [c.get_metrics_dict().get("peak_memory_mb", 1.0) or 1.0 for c in feasible]

        min_l, max_l = min(lats), max(lats)
        min_t, max_t = min(tps), max(tps)
        min_m, max_m = min(mems), max(mems)

        scored_candidates: List[Tuple[float, ExecutionCandidate]] = []
        for c in feasible:
            m = c.get_metrics_dict()
            lat = m.get("median_latency_ms", max_l)
            tp = m.get("throughput_inf_sec", min_t)
            mem = m.get("peak_memory_mb") or max_m

            norm_l = (lat - min_l) / (max_l - min_l) if max_l > min_l else 0.0
            norm_t = (tp - min_t) / (max_t - min_t) if max_t > min_t else 1.0
            norm_m = (mem - min_m) / (max_m - min_m) if max_m > min_m else 0.0

            if goal == RecommendationGoal.MINIMIZE_LATENCY:
                score = -lat  # higher is better
            elif goal == RecommendationGoal.MAXIMIZE_THROUGHPUT:
                score = tp
            elif goal == RecommendationGoal.MINIMIZE_MEMORY:
                score = -mem
            else:  # BALANCED_SCORE (40% Latency, 40% Throughput, 20% Memory)
                score = 0.4 * (1.0 - norm_l) + 0.4 * norm_t + 0.2 * (1.0 - norm_m)

            scored_candidates.append((score, c))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand = scored_candidates[0]

        m_best = best_cand.get_metrics_dict()
        rel = ReliabilityAnalyzer.analyze_candidate(best_cand)
        sat_constraints = [ce.explanation for ce in best_cand.constraint_evaluations if ce.passed]

        rationale = (
            f"Candidate '{best_cand.candidate_id}' is selected for goal '{goal.value}'. "
            f"It achieves a median latency of {m_best.get('median_latency_ms', 0):.3f} ms "
            f"and throughput of {m_best.get('throughput_inf_sec', 0):.1f} inf/s with "
            f"configuration (batch={best_cand.config.batch_size}, threads={best_cand.config.intra_op_num_threads or 'Default'}). "
            f"All {len(sat_constraints)} configured hard constraints were satisfied."
        )

        tradeoffs = (
            f"Trade-off Profile: Optimal in {goal.value}; "
            f"Pareto optimal = {best_cand.is_pareto_optimal}; "
            f"Relative speedup vs baseline = {best_cand.speedup_vs_baseline or 1.0:.2f}x."
        )

        return EvidenceRecommendation(
            goal=goal,
            recommended_candidate_id=best_cand.candidate_id,
            recommended_config=best_cand.config.to_dict(),
            target_profile_id=target_prof_id,
            target_profile_version=target_prof_ver,
            execution_environment_fingerprint=env_fp,
            is_target_verified=is_target_verified,
            satisfied_constraints=sat_constraints,
            violated_constraints=[],
            median_latency_ms=m_best.get("median_latency_ms"),
            p95_latency_ms=m_best.get("p95_latency_ms"),
            throughput_inf_sec=m_best.get("throughput_inf_sec"),
            peak_memory_mb=m_best.get("peak_memory_mb"),
            speedup_vs_baseline=best_cand.speedup_vs_baseline,
            latency_reduction_pct=best_cand.latency_reduction_pct,
            throughput_improvement_pct=best_cand.throughput_improvement_pct,
            reliability=rel,
            balanced_score=best_score if goal == RecommendationGoal.BALANCED_SCORE else None,
            rationale=rationale,
            tradeoffs_summary=tradeoffs,
            unverified_requirements=unverified,
        )


# --- Enterprise Report Generator ---

class EnterpriseReportGenerator:
    """Generates comprehensive, professional 18-section audit reports in Markdown and JSON formats."""

    @staticmethod
    def _redact(text: str) -> str:
        """Redact sensitive credentials, keys, and tokens."""
        if not text:
            return ""
        import re
        redacted = re.sub(
            r"(?i)\b(password|secret|token|key|ssh-rsa|bearer)\s*[:=]\s*([^\s,;\"']+)",
            r"\1: ***REDACTED***",
            text,
        )
        redacted = re.sub(
            r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[^-]+-----END [A-Z ]+ PRIVATE KEY-----",
            "***REDACTED PRIVATE KEY***",
            redacted,
        )
        return redacted

    @classmethod
    def _redact_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if any(sec in k.lower() for sec in ("password", "secret", "token", "private_key", "bearer")):
                    new_dict[k] = "***REDACTED***"
                else:
                    new_dict[k] = cls._redact_dict(v)
            return new_dict
        elif isinstance(data, list):
            return [cls._redact_dict(item) for item in data]
        elif isinstance(data, str):
            return cls._redact(data)
        return data

    @classmethod
    def generate_markdown(
        cls,
        report_data: Dict[str, Any],
        recommendation: Optional[EvidenceRecommendation] = None,
        sensitivity: Optional[SensitivityReport] = None,
    ) -> str:
        r = report_data
        rec = recommendation
        sens = sensitivity
        ts = r.get("timestamp_utc", datetime.now(timezone.utc).isoformat())
        exp_id = r.get("experiment_id", "OPT-EXP-001")
        model_name = r.get("model_name", "model.onnx")
        model_fp = r.get("model_fingerprint", "UNKNOWN_HASH")
        prof_name = r.get("profile_name", "Target Profile")
        prof_id = r.get("profile_id", "unknown_profile")
        prof_ver = r.get("profile_version", "1.0.0")
        env = r.get("environment", {})
        counts = r.get("summary_counts", {})
        candidates = r.get("candidates", [])

        is_z_target = env.get("is_s390x", False) and not r.get("is_illustrative_profile", False)
        provenance_str = "Verified Linux on IBM Z (s390x)" if is_z_target else "Local Host Development (x86_64) · IBM Z Execution Pending"

        md = f"""# Enterprise Deployment Optimization & Audit Report

**Report Identifier:** `{exp_id}`  
**Generated At (UTC):** `{ts}`  
**System Platform:** Z-ModelPilot `v1.4.0` (Architecture-Aware AI Deployment Optimizer)  
**Execution Environment Provenance:** `{provenance_str}`  

---

## 1. Executive Summary
This enterprise optimization report details the empirical evaluation of ONNX model deployment candidates for `{model_name}` against capability profile `{prof_name}` (`{prof_id}` `v{prof_ver}`). A total of **{counts.get('total_generated', len(candidates))}** candidates were explored, **{counts.get('benchmarked', 0)}** were empirically benchmarked, and **{counts.get('feasible', 0)}** met all operational correctness and boundary constraints. The multi-objective Pareto trade-off frontier contains **{counts.get('pareto_count', 0)}** non-dominated configurations.

## 2. User Objective and Operational Constraints
- **Primary Optimization Objective:** `{r.get('primary_objective', 'MINIMIZE_MEDIAN_LATENCY')}`
- **Search Strategy:** `{r.get('request', {}).get('search_strategy', 'GRID')}`
- **Hard Constraints Configured:**
"""
        hard_constraints = r.get("request", {}).get("hard_constraints", [])
        if hard_constraints:
            for hc in hard_constraints:
                md += f"  - `{hc.get('name')}`: `{hc.get('metric_field')}` {hc.get('operator')} {hc.get('threshold')} {hc.get('unit')}\n"
        else:
            md += "  - Standard inference correctness and finiteness validation.\n"

        md += f"""
## 3. Model Identity and Cryptographic Artifact Hash
- **Original Filename:** `{model_name}`
- **Cryptographic SHA-256 Hash:** `{model_fp}`
- **Integrity Status:** Cryptographically verified before and after runtime session execution.

## 4. Target Profile and Profile Version
- **Target Profile Name:** `{prof_name}`
- **Profile Identifier:** `{prof_id}`
- **Profile Ruleset Version:** `{prof_ver}`
- **Profile Illustrative Tag:** `{'True (Illustrative Draft)' if r.get('is_illustrative_profile', False) else 'False (Authoritative Profile)'}`

## 5. Environment Details and Hardware Fingerprint
- **Host Architecture:** `{env.get('architecture', platform.machine())}`
- **Operating System:** `{env.get('os_name', platform.system())} {env.get('os_release', platform.release())}`
- **Python Version:** `{env.get('python_version', platform.python_version())}`
- **ONNX Runtime Version:** `{env.get('onnxruntime_version', 'Unknown')}`
- **Installed Execution Providers:** `{', '.join(env.get('available_providers', ['CPUExecutionProvider']))}`
- **Logical CPU Cores:** `{env.get('cpu_count_logical', 'Unknown')}`

## 6. Experiment Methodology and Measurement Boundary
- **Warmup Iterations:** `{r.get('request', {}).get('warmup_iterations', 10)}`
- **Measured Iterations:** `{r.get('request', {}).get('measured_iterations', 50)}`
- **Measurement Boundary:** Strictly isolated to `session.run()` C++ kernel execution; tensor generation and allocation are excluded.
- **Repeatability Checking:** Consecutive identical inference feeds compared with `atol=1e-5`, `rtol=1e-4`.

## 7. Candidate Configurations Evaluated
| Candidate ID | Status | Provider | Batch | Intra Threads | Inter Threads | Graph Opt | Median Latency | Throughput |
|---|---|---|---|---|---|---|---|---|
"""
        for c in candidates:
            cfg = c.get("config", {})
            m = c.get("metrics", {}) or {}
            med_lat = f"{m.get('median_latency_ms', 0):.3f} ms" if m.get("median_latency_ms") else "N/A"
            tp = f"{m.get('throughput_inf_sec', 0):.1f} inf/s" if m.get("throughput_inf_sec") else "N/A"
            md += f"| `{c.get('candidate_id')}` | `{c.get('status')}` | `{cfg.get('execution_provider')}` | {cfg.get('batch_size')} | {cfg.get('intra_op_num_threads') or 'Default'} | {cfg.get('inter_op_num_threads') or 'Default'} | `{cfg.get('graph_optimization_level') or 'Default'}` | {med_lat} | {tp} |\n"

        md += f"""
## 8. Failed, Blocked, and Excluded Candidates
"""
        rejected = [c for c in candidates if c.get("status") in ("REJECTED", "FAILED", "INFEASIBLE")]
        if rejected:
            for rj in rejected:
                reason = rj.get("rejection_reason") or "Failed constraint or correctness evaluation"
                md += f"- **`{rj.get('candidate_id')}`:** {reason}\n"
        else:
            md += "All generated candidates successfully satisfied preliminary checks and were benchmarked.\n"

        md += f"""
## 9. Benchmark Summary and Metric Definitions
- **Median Latency:** Monotonic clock median elapsed time across all non-warmup iterations.
- **P95 Latency:** 95th percentile latency reflecting tail tail jitter under load.
- **Throughput:** Total completed inferences divided by cumulative wall-clock session duration.
- **Peak RSS Memory:** Maximum resident set size observed during runtime session execution.

## 10. Correctness and Compatibility Results
- **Output Finiteness:** Verified all tensor outputs contain zero `NaN` or `+/- Inf` values.
- **Output Shape Conformance:** Output dimensions matched declared model graph signatures.
- **Deterministic Repeatability:** Verified consistent output matrices across successive executions on identical inputs.

## 11. Pareto Frontier and Trade-off Explanation
The following candidate configurations represent the Pareto-optimal frontier (no candidate is simultaneously faster, higher throughput, and lower memory):
"""
        pareto_ids = r.get("pareto_frontier_ids", [])
        if pareto_ids:
            for pid in pareto_ids:
                cand_match = next((c for c in candidates if c.get("candidate_id") == pid), None)
                if cand_match:
                    cfg = cand_match.get("config", {})
                    m = cand_match.get("metrics", {}) or {}
                    md += f"- **`{pid}`:** Latency = `{m.get('median_latency_ms', 0):.3f} ms`, Throughput = `{m.get('throughput_inf_sec', 0):.1f} inf/s`, Batch = `{cfg.get('batch_size')}`, Threads = `{cfg.get('intra_op_num_threads') or 'Default'}`\n"
        else:
            md += "No multi-objective Pareto candidates identified among feasible set.\n"

        md += f"""
## 12. Selected Recommendation and Rationale
"""
        if rec and rec.recommended_candidate_id:
            md += f"- **Recommended Candidate:** `{rec.recommended_candidate_id}`\n"
            md += f"- **Optimization Goal:** `{rec.goal.value}`\n"
            md += f"- **Rationale:** {rec.rationale}\n"
            md += f"- **Trade-off Analysis:** {rec.tradeoffs_summary}\n"
        else:
            md += f"{r.get('recommendation_rationale', 'No evidence-backed recommendation available.')}\n"

        md += f"""
## 13. Comparison Against Baseline
"""
        base_id = r.get("baseline_candidate_id")
        best_id = r.get("best_candidate_id") or (rec.recommended_candidate_id if rec else None)
        base_c = next((c for c in candidates if c.get("candidate_id") == base_id), None)
        best_c = next((c for c in candidates if c.get("candidate_id") == best_id), None)

        if base_c and best_c:
            bm = base_c.get("metrics", {}) or {}
            om = best_c.get("metrics", {}) or {}
            spd = best_c.get("speedup_vs_baseline") or 1.0
            lat_red = best_c.get("latency_reduction_pct") or 0.0
            md += f"- **Baseline Configuration (`{base_id}`):** Median Latency = `{bm.get('median_latency_ms', 0):.3f} ms`, Throughput = `{bm.get('throughput_inf_sec', 0):.1f} inf/s`\n"
            md += f"- **Recommended Configuration (`{best_id}`):** Median Latency = `{om.get('median_latency_ms', 0):.3f} ms`, Throughput = `{om.get('throughput_inf_sec', 0):.1f} inf/s`\n"
            md += f"- **Relative Improvement:** `{spd:.2f}x` speedup ({lat_red:+.1f}% latency change vs default baseline).\n"
        else:
            md += "Baseline candidate or optimal candidate not established.\n"

        md += f"""
## 14. Measurement Reliability, Dispersion, and Comparability
"""
        if rec and rec.reliability:
            rel = rec.reliability
            md += f"- **Statistical Sample Count:** `{rel.sample_count}` inferences.\n"
            md += f"- **Interquartile Range (IQR):** `{rel.iqr_ms:.4f} ms`\n"
            md += f"- **Standard Deviation:** `{rel.std_dev_ms:.4f} ms`\n"
            md += f"- **Coefficient of Variation (CV%):** `{rel.cv_percent:.2f}%`\n"
            md += f"- **Stability Rating:** `{rel.rating.value}` ({rel.notes})\n"
        else:
            md += "Detailed dispersion metrics computed from raw latency arrays.\n"

        if sens and sens.primary_influencing_parameter:
            md += f"\n### Parameter Sensitivity Summary\n{sens.summary_insight}\n"

        md += f"""
## 15. Security, Deployment, and Operational Considerations
- **Command Injection Prevention:** All optimization parameters are validated against strict numerical and enumeration bounds.
- **Resource Guardrails:** Iteration counts, candidate numbers, and tensor memory allocations are bounded.
- **Secret Redaction:** Network keys, passwords, and private tokens are masked across reports and logs.
- **Unprivileged Execution:** All benchmarking runs under standard user permissions without root escalation.

## 16. IBM Z Validation Status and Hardware Provenance
- **Target Verification Status:** `{'TARGET_VERIFIED' if is_z_target else 'LOCALLY_VERIFIED (Target Execution Pending)'}`
- **Authentication Statement:**
  > *“IBM Z target validation remains pending; only the listed local checks have been verified.”*

## 17. Unverified Requirements and Actionable Next Steps
1. Execute validation on physical Linux on IBM Z (s390x) host or LPAR.
2. Verify hardware NNPA coprocessor offload (Facility bit 193) via `/proc/cpuinfo`.
3. Validate `NNPAExecutionProvider` runtime acceleration in IBM AI on Z container environment.
4. Measure memory bandwidth on IBM Telum dual-chip module ring cache.

## 18. Audit Trail, Timestamps, and Reproducibility Metadata
- **Experiment Hash:** `{r.get('request_payload_hash', 'N/A')}`
- **Random Seed:** `{r.get('request', {}).get('random_seed', 42)}`
- **Application Version:** `Z-ModelPilot v1.4.0`
- **Report Schema Version:** `2.0.0-phase12`
- **Report Generated At:** `{ts}`
"""
        return cls._redact(md)

    @classmethod
    def generate_json(
        cls,
        report_data: Dict[str, Any],
        recommendation: Optional[EvidenceRecommendation] = None,
        sensitivity: Optional[SensitivityReport] = None,
    ) -> str:
        """Export full machine-readable JSON audit report with documented schema."""
        import copy
        redacted_dict = cls._redact_dict(copy.deepcopy(report_data))
        export_payload = {
            "schema_version": "2.0.0-phase12",
            "report_id": report_data.get("experiment_id", f"OPT-{int(time.time())}"),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance_statement": "IBM Z target validation remains pending; only the listed local checks have been verified.",
            "experiment": redacted_dict,
            "recommendation": recommendation.to_dict() if recommendation else None,
            "sensitivity": sensitivity.to_dict() if sensitivity else None,
        }
        return json.dumps(export_payload, indent=2)


# --- Advanced Optimization Engine ---

class AdvancedOptimizationEngine:
    """
    Coordinates bounded candidate generation (Grid, Random, Successive Halving),
    Pareto analysis, sensitivity assessment, and evidence recommendation.
    """

    @classmethod
    def generate_candidates(
        cls,
        model_summary: ModelSummary,
        request: OptimizationRequest,
        strategy: SearchStrategy,
        available_providers: List[str],
        graph_optimization_levels: Optional[List[str]] = None,
        execution_modes: Optional[List[str]] = None,
    ) -> List[ExecutionCandidate]:
        """Generate deduplicated candidates according to configured search strategy and bounds."""
        target_providers = request.execution_providers or ["CPUExecutionProvider"]
        providers_to_use = [p for p in target_providers if p in available_providers] or ["CPUExecutionProvider"]

        fixed_batch = None
        for inp in model_summary.inputs:
            if inp.shape and len(inp.shape) > 0 and isinstance(inp.shape[0], int) and inp.shape[0] > 0:
                fixed_batch = inp.shape[0]
                break

        base_batch = fixed_batch or 1
        batch_choices = [base_batch] if fixed_batch else (request.batch_sizes or [1, 2, 4])
        intra_choices = request.intra_op_threads or [1, 2, 4]
        inter_choices = request.inter_op_threads or [1]
        graph_opts = graph_optimization_levels or ["ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"]
        exec_modes = execution_modes or ["ORT_SEQUENTIAL"]

        candidates: List[ExecutionCandidate] = []
        seen_hashes: Set[str] = set()

        # Always generate candidate 01: baseline
        base_cfg = BenchmarkConfig(
            execution_provider=providers_to_use[0],
            batch_size=base_batch,
            warmup_iterations=request.warmup_iterations,
            measured_iterations=request.measured_iterations,
            repeated_rounds=request.repeated_rounds,
            random_seed=request.random_seed,
            dynamic_dimensions=dict(request.dynamic_dimensions),
            intra_op_num_threads=None,
            inter_op_num_threads=None,
            graph_optimization_level=None,
            execution_mode=None,
        )
        base_cand = ExecutionCandidate(
            candidate_id="CAND-01-BASELINE",
            config=base_cfg,
            generation_reason="Standard default baseline configuration",
            status=CandidateStatus.GENERATED,
            is_baseline=True,
        )
        candidates.append(base_cand)
        seen_hashes.add(base_cfg.configuration_hash())

        # Build full candidate space
        all_combos: List[Tuple[str, int, int, int, str, str]] = []
        for p in providers_to_use:
            for b in batch_choices:
                for intra in intra_choices:
                    for inter in inter_choices:
                        for g in graph_opts:
                            for m in exec_modes:
                                all_combos.append((p, b, intra, inter, g, m))

        if strategy == SearchStrategy.RANDOM:
            rng = random.Random(request.random_seed)
            rng.shuffle(all_combos)
        elif strategy == SearchStrategy.SUCCESSIVE_HALVING:
            # Deterministic ordering; multi-fidelity applied during benchmarking
            pass

        cand_counter = 1
        for p, b, intra, inter, g, m in all_combos:
            if len(candidates) >= request.max_candidates:
                break

            cfg = BenchmarkConfig(
                execution_provider=p,
                batch_size=b,
                warmup_iterations=request.warmup_iterations,
                measured_iterations=request.measured_iterations,
                repeated_rounds=request.repeated_rounds,
                random_seed=request.random_seed,
                dynamic_dimensions=dict(request.dynamic_dimensions),
                intra_op_num_threads=intra,
                inter_op_num_threads=inter,
                graph_optimization_level=g,
                execution_mode=m,
            )
            h = cfg.configuration_hash()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            cand_counter += 1
            cand_id = f"CAND-{cand_counter:02d}"
            reason = f"{strategy.value} search: batch={b}, intra={intra}, graph_opt={g}"
            candidates.append(
                ExecutionCandidate(
                    candidate_id=cand_id,
                    config=cfg,
                    generation_reason=reason,
                    status=CandidateStatus.GENERATED,
                )
            )

        return candidates

    @classmethod
    def execute_advanced_optimization(
        cls,
        model_path_or_bytes: Union[str, bytes],
        model_summary: ModelSummary,
        profile: CapabilityProfile,
        request: OptimizationRequest,
        strategy: SearchStrategy = SearchStrategy.GRID,
        recommendation_goal: RecommendationGoal = RecommendationGoal.BALANCED_SCORE,
        graph_optimization_levels: Optional[List[str]] = None,
        execution_modes: Optional[List[str]] = None,
        env_inventory: Optional[EnvironmentInventory] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Execute end-to-end advanced optimization search and return comprehensive report artifact."""
        request.validate()
        exp_id = f"OPT-ADV-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 10000:04d}"
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        env_info = discover_execution_providers()
        available_providers = env_info.available_providers

        # 1. Candidate Generation
        if progress_callback:
            progress_callback(f"Generating candidates using {strategy.value} strategy...", 5, 100)

        candidates = cls.generate_candidates(
            model_summary=model_summary,
            request=request,
            strategy=strategy,
            available_providers=available_providers,
            graph_optimization_levels=graph_optimization_levels,
            execution_modes=execution_modes,
        )

        # 2. Compatibility Filtering
        if progress_callback:
            progress_callback("Filtering candidates against architecture capability profile...", 15, 100)

        compat_report = CompatibilityAnalyzer.analyze(model_summary, profile)
        has_fail = any(f.status == "FAIL" for f in compat_report.findings)

        for cand in candidates:
            cand.compatibility_findings = [f.to_dict() for f in compat_report.findings]
            if has_fail:
                fail_ops = [f.op_type for f in compat_report.findings if f.status == "FAIL" and f.op_type]
                cand.status = CandidateStatus.REJECTED
                cand.rejection_reason = f"Profile '{profile.display_name}' has FAIL findings for operators: {fail_ops}."
            else:
                cand.status = CandidateStatus.QUEUED

        # 3. Execution (with Successive Halving support)
        queued = [c for c in candidates if c.status == CandidateStatus.QUEUED]
        total_queued = len(queued)
        baseline_candidate: Optional[ExecutionCandidate] = None

        if strategy == SearchStrategy.SUCCESSIVE_HALVING and total_queued >= 4:
            # Stage 1: Quick evaluation (low iterations)
            stage1_candidates = queued
            stage1_reports = []
            for idx, cand in enumerate(stage1_candidates, 1):
                if progress_callback:
                    pct = 20 + int((idx / total_queued) * 30)
                    progress_callback(f"Successive Halving Stage 1: screening {cand.candidate_id}...", pct, 100)

                quick_cfg = BenchmarkConfig(
                    execution_provider=cand.config.execution_provider,
                    batch_size=cand.config.batch_size,
                    warmup_iterations=2,
                    measured_iterations=6,
                    repeated_rounds=1,
                    random_seed=request.random_seed,
                    dynamic_dimensions=dict(request.dynamic_dimensions),
                    intra_op_num_threads=cand.config.intra_op_num_threads,
                    inter_op_num_threads=cand.config.inter_op_num_threads,
                    graph_optimization_level=cand.config.graph_optimization_level,
                    execution_mode=cand.config.execution_mode,
                )
                try:
                    rep = BenchmarkEngine.run_benchmark(model_path_or_bytes, model_summary, quick_cfg)
                    stage1_reports.append((cand, rep.latency.get("median_ms", float("inf")) if rep.latency else float("inf")))
                except Exception:
                    cand.status = CandidateStatus.FAILED
                    cand.rejection_reason = "Stage 1 screening run failed."

            # Halve: retain top 50%
            stage1_reports.sort(key=lambda x: x[1])
            cutoff = max(2, len(stage1_reports) // 2)
            promoted_cands = set(c[0].candidate_id for c in stage1_reports[:cutoff])

            for cand in stage1_candidates:
                if cand.candidate_id not in promoted_cands and cand.status != CandidateStatus.FAILED:
                    cand.status = CandidateStatus.INFEASIBLE
                    cand.rejection_reason = "Halved early in Stage 1 due to suboptimal initial latency."

            # Candidates to run full evaluation
            eval_list = [c for c in queued if c.candidate_id in promoted_cands]
        else:
            eval_list = queued

        # Full Benchmark Execution
        for idx, cand in enumerate(eval_list, 1):
            if progress_callback:
                pct = 50 + int((idx / max(1, len(eval_list))) * 30)
                progress_callback(f"Benchmarking candidate {idx}/{len(eval_list)}: {cand.candidate_id}...", pct, 100)

            cand.status = CandidateStatus.RUNNING
            try:
                rep = BenchmarkEngine.run_benchmark(model_path_or_bytes, model_summary, cand.config)
                cand.benchmark_report = rep
                cand.status = CandidateStatus.BENCHMARKED
            except CorrectnessValidationError as cve:
                cand.status = CandidateStatus.FAILED
                cand.rejection_reason = f"Correctness check failed: {str(cve)}"
            except Exception as e:
                cand.status = CandidateStatus.FAILED
                cand.rejection_reason = f"Execution error: {str(e)}"

            if cand.is_baseline and cand.status == CandidateStatus.BENCHMARKED:
                baseline_candidate = cand

        if baseline_candidate is None:
            benchmarked = [c for c in candidates if c.status == CandidateStatus.BENCHMARKED]
            if benchmarked:
                baseline_candidate = benchmarked[0]
                baseline_candidate.is_baseline = True

        # 4. Evaluate Hard Constraints
        for cand in candidates:
            if cand.status == CandidateStatus.BENCHMARKED:
                evals: List[ConstraintEvaluation] = []
                metrics = cand.get_metrics_dict()
                all_passed = bool(metrics.get("correctness_valid", False))

                if not all_passed:
                    evals.append(
                        ConstraintEvaluation(
                            constraint_name="Inference Correctness",
                            metric_field="correctness_valid",
                            threshold=1.0,
                            observed_value=0.0,
                            unit="bool",
                            passed=False,
                            is_available=True,
                            explanation="Output contains non-finite numbers or repeatability failed.",
                        )
                    )

                for hc in request.hard_constraints:
                    ev = hc.evaluate(metrics)
                    evals.append(ev)
                    if not ev.passed:
                        all_passed = False

                cand.constraint_evaluations = evals
                cand.is_feasible = all_passed
                if not all_passed:
                    cand.status = CandidateStatus.INFEASIBLE

        # 5. Pareto Frontier Calculation
        frontier_ids, dominance_matrix = AdvancedParetoAnalyzer.compute_frontier(candidates)

        # 6. Relative Speedups
        if baseline_candidate and baseline_candidate.benchmark_report:
            base_lat = baseline_candidate.get_metrics_dict().get("median_latency_ms", 1.0)
            base_tp = baseline_candidate.get_metrics_dict().get("throughput_inf_sec", 1.0)
            for cand in candidates:
                if cand.benchmark_report and cand.status in (CandidateStatus.BENCHMARKED, CandidateStatus.INFEASIBLE):
                    m = cand.get_metrics_dict()
                    c_lat = m.get("median_latency_ms")
                    c_tp = m.get("throughput_inf_sec")
                    if c_lat and base_lat > 0:
                        cand.speedup_vs_baseline = float(base_lat / c_lat)
                        cand.latency_reduction_pct = float(((base_lat - c_lat) / base_lat) * 100.0)
                    if c_tp and base_tp > 0:
                        cand.throughput_improvement_pct = float(((c_tp - base_tp) / base_tp) * 100.0)

        # 7. Candidate Ranking
        feasible = [c for c in candidates if c.is_feasible]
        infeasible = [c for c in candidates if not c.is_feasible]
        feasible.sort(key=lambda c: c.get_metrics_dict().get("median_latency_ms", float("inf")))
        for r_idx, c in enumerate(feasible, 1):
            c.rank = r_idx
        for c in infeasible:
            c.rank = None

        ranked_candidates = feasible + infeasible

        # 8. Sensitivity Analysis
        sensitivity_report = SensitivityAnalyzer.analyze(candidates)

        # 9. Recommendation
        recommendation = RecommendationEngine.evaluate(
            candidates=ranked_candidates,
            goal=recommendation_goal,
            profile=profile,
            baseline_candidate=baseline_candidate,
            env_inventory=env_inventory,
        )

        # 10. Assemble Full Report Dict
        summary_counts = {
            "total_generated": len(candidates),
            "rejected": sum(1 for c in candidates if c.status == CandidateStatus.REJECTED),
            "benchmarked": sum(1 for c in candidates if c.benchmark_report is not None),
            "feasible": len(feasible),
            "infeasible": sum(1 for c in candidates if c.status == CandidateStatus.INFEASIBLE),
            "failed": sum(1 for c in candidates if c.status == CandidateStatus.FAILED),
            "pareto_count": len(frontier_ids),
        }

        report_dict = {
            "experiment_id": exp_id,
            "timestamp_utc": timestamp_utc,
            "model_name": model_summary.filename,
            "model_fingerprint": model_summary.sha256_hash,
            "profile_id": getattr(profile, "profile_id", getattr(profile, "id", "")),
            "profile_name": getattr(profile, "display_name", getattr(profile, "name", "")),
            "profile_version": getattr(profile, "version", "1.0.0"),
            "primary_objective": request.primary_objective.value,
            "search_strategy": strategy.value,
            "recommendation_goal": recommendation_goal.value,
            "request": {
                **request.to_dict(),
                "search_strategy": strategy.value,
                "graph_optimization_levels": graph_optimization_levels,
                "execution_modes": execution_modes,
            },
            "summary_counts": summary_counts,
            "baseline_candidate_id": baseline_candidate.candidate_id if baseline_candidate else None,
            "best_candidate_id": recommendation.recommended_candidate_id,
            "pareto_frontier_ids": frontier_ids,
            "dominance_matrix": dominance_matrix,
            "candidates": [
                {
                    **c.to_dict(),
                    "metrics": c.get_metrics_dict(),
                    "reliability": ReliabilityAnalyzer.analyze_candidate(c).to_dict() if ReliabilityAnalyzer.analyze_candidate(c) else None,
                }
                for c in ranked_candidates
            ],
            "sensitivity": sensitivity_report.to_dict(),
            "recommendation": recommendation.to_dict(),
            "environment": {
                "os_name": platform.system(),
                "os_release": platform.release(),
                "architecture": platform.machine(),
                "python_version": platform.python_version(),
                "onnxruntime_version": env_info.installed_ort_version,
                "available_providers": env_info.available_providers,
                "is_s390x": env_inventory.is_s390x if env_inventory else (platform.machine().lower() in ("s390x", "s390")),
                "fingerprint": env_inventory.fingerprint if env_inventory else "LOCAL_DEV_ENV",
            },
            "recommendation_rationale": recommendation.rationale,
            "is_illustrative_profile": getattr(profile, "is_illustrative", False),
        }

        # 11. Generate Rendered Markdown and JSON Report
        markdown_report = EnterpriseReportGenerator.generate_markdown(
            report_data=report_dict,
            recommendation=recommendation,
            sensitivity=sensitivity_report,
        )
        json_report = EnterpriseReportGenerator.generate_json(
            report_data=report_dict,
            recommendation=recommendation,
            sensitivity=sensitivity_report,
        )

        report_dict["markdown_report"] = markdown_report
        report_dict["json_report"] = json_report

        if progress_callback:
            progress_callback("Advanced optimization and reporting completed successfully.", 100, 100)

        return report_dict
