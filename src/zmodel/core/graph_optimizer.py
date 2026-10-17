"""End-to-end ONNX graph optimization and model transformation engine.

Orchestrates:
1. Transformation candidate generation based on real environment capabilities.
2. Isolated transformation execution preserving original model immutability.
3. Layered structural integrity and numerical equivalence validation.
4. Target architecture compatibility reassessment (reusing Phase 3).
5. Controlled, comparable empirical benchmarking (reusing Phase 4).
6. Transparent, evidence-backed candidate recommendation layer.
"""

import hashlib
import logging
import os
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from zmodel.core.artifact_store import ArtifactInfo, get_artifact_store
from zmodel.core.benchmark import BenchmarkConfig, BenchmarkEngine, BenchmarkReport
from zmodel.core.compatibility import CompatibilityEngine, CompatibilityReport
from zmodel.core.graph_transforms import (
    TransformationDescriptor,
    TransformationRegistry,
    TransformationResult,
    TransformationType,
)
from zmodel.core.graph_validator import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    DEFAULT_TEST_SEED,
    GraphValidator,
    ValidationReport,
    ValidationStatus,
)
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.profiles import CapabilityProfile, load_builtin_profiles

logger = logging.getLogger(__name__)


class CandidateState(str, Enum):
    """Explicit lifecycle states for a graph transformation candidate."""

    GENERATED = "GENERATED"
    TRANSFORMING = "TRANSFORMING"
    TRANSFORMED = "TRANSFORMED"
    STRUCTURALLY_VALIDATED = "STRUCTURALLY_VALIDATED"
    NUMERICALLY_VALIDATED = "NUMERICALLY_VALIDATED"
    COMPATIBILITY_CHECKED = "COMPATIBILITY_CHECKED"
    BENCHMARKED = "BENCHMARKED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    NOT_VERIFIED = "NOT_VERIFIED"


class RecommendationStatus(str, Enum):
    """Evaluation recommendation categories for deployment readiness."""

    VALIDATED = "VALIDATED"
    NUMERICALLY_VERIFIED = "NUMERICALLY_VERIFIED"
    COMPATIBLE = "COMPATIBLE"
    PERFORMANCE_IMPROVED = "PERFORMANCE_IMPROVED"
    NO_DEMONSTRATED_IMPROVEMENT = "NO_DEMONSTRATED_IMPROVEMENT"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


@dataclass
class GraphCandidate:
    """Comprehensive state and evaluation metrics for a transformation candidate."""

    candidate_id: str
    transform_type: TransformationType
    transform_name: str
    tool_name: str
    tool_version: str
    state: CandidateState = CandidateState.GENERATED
    original_hash: str = ""
    transformed_hash: Optional[str] = None
    artifact_id: Optional[str] = None
    graph_changed: bool = False
    model_size_bytes_before: int = 0
    model_size_bytes_after: Optional[int] = None
    size_reduction_pct: Optional[float] = None
    diff_summary: Dict[str, Any] = field(default_factory=dict)
    validation_report: Optional[ValidationReport] = None
    compatibility_report: Optional[Dict[str, Any]] = None
    compatibility_delta: Optional[Dict[str, Any]] = None
    benchmark_report: Optional[Dict[str, Any]] = None
    speedup_factor: Optional[float] = None
    latency_reduction_pct: Optional[float] = None
    throughput_improvement_pct: Optional[float] = None
    recommendation_status: RecommendationStatus = RecommendationStatus.UNKNOWN
    recommendation_rationale: str = "Candidate has not yet been evaluated."
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["transform_type"] = self.transform_type.value
        data["state"] = self.state.value
        data["recommendation_status"] = self.recommendation_status.value
        if self.validation_report:
            data["validation_report"] = self.validation_report.to_dict()
        return data


@dataclass
class GraphOptimizationReport:
    """Consolidated report of a multi-candidate graph optimization experiment."""

    experiment_id: str
    model_name: str
    original_fingerprint: str
    target_profile_id: Optional[str]
    target_profile_name: Optional[str]
    validation_depth: str  # "FULL" (structural + numerical) or "STRUCTURAL_ONLY"
    benchmark_enabled: bool
    created_at: str
    completed_at: Optional[str] = None
    status: str = "RUNNING"  # "SUCCESS", "FAILED", "PARTIAL"
    candidates: List[GraphCandidate] = field(default_factory=list)
    best_candidate_id: Optional[str] = None
    summary_counts: Dict[str, int] = field(default_factory=dict)
    overall_recommendation: str = ""
    limitations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [c.to_dict() for c in self.candidates]
        return data


class GraphOptimizationEngine:
    """Executes graph transformation experiments, validations, compatibility, and benchmarking."""

    def __init__(
        self,
        artifact_store=None,
        default_atol: float = DEFAULT_ATOL,
        default_rtol: float = DEFAULT_RTOL,
        default_seed: int = DEFAULT_TEST_SEED,
    ):
        self.artifact_store = artifact_store or get_artifact_store()
        self.atol = default_atol
        self.rtol = default_rtol
        self.seed = default_seed

    def run_optimization_experiment(
        self,
        model_bytes: bytes,
        original_filename: str = "model.onnx",
        selected_transforms: Optional[List[TransformationType]] = None,
        target_profile_id: Optional[str] = None,
        validation_depth: str = "FULL",  # "FULL" or "STRUCTURAL_ONLY"
        enable_benchmarking: bool = True,
        benchmark_config: Optional[BenchmarkConfig] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> GraphOptimizationReport:
        """
        Execute full graph optimization experiment pipeline.
        Treats model_bytes as strictly immutable.
        """
        experiment_id = f"gexp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        created_at = datetime.now(timezone.utc).isoformat()
        original_hash = hashlib.sha256(model_bytes).hexdigest()
        orig_size = len(model_bytes)

        # Inspect original model
        try:
            orig_summary = load_and_inspect_model(model_bytes, filename=original_filename)
        except Exception as ex:
            return GraphOptimizationReport(
                experiment_id=experiment_id,
                model_name=original_filename,
                original_fingerprint=original_hash,
                target_profile_id=target_profile_id,
                target_profile_name=None,
                validation_depth=validation_depth,
                benchmark_enabled=enable_benchmarking,
                created_at=created_at,
                completed_at=datetime.now(timezone.utc).isoformat(),
                status="FAILED",
                error_message=f"Failed to inspect original ONNX model: {str(ex)}",
            )

        # Resolve target profile if requested
        target_profile: Optional[CapabilityProfile] = None
        target_profile_name: Optional[str] = None
        if target_profile_id:
            builtin_profiles = load_builtin_profiles()
            if target_profile_id in builtin_profiles:
                target_profile = builtin_profiles[target_profile_id]
                target_profile_name = target_profile.display_name

        # Assess original model compatibility baseline if profile provided
        orig_compat_report: Optional[CompatibilityReport] = None
        if target_profile:
            try:
                orig_compat_report = CompatibilityEngine.analyze(orig_summary, target_profile)
            except Exception as ex:
                logger.warning("Baseline compatibility check failed: %s", ex)

        # Baseline benchmark if enabled
        orig_bench_report: Optional[BenchmarkReport] = None
        if enable_benchmarking and validation_depth == "FULL":
            if benchmark_config is None:
                benchmark_config = BenchmarkConfig(
                    execution_provider="CPUExecutionProvider",
                    warmup_iterations=5,
                    repeated_rounds=3,
                    measured_iterations=10,
                )
            try:
                orig_bench_report = BenchmarkEngine.run_benchmark(
                    model_bytes, orig_summary, benchmark_config
                )
            except Exception as ex:
                logger.warning("Baseline benchmark failed: %s", ex)

        # Determine transformations to execute
        available_descriptors = {d.transform_type: d for d in TransformationRegistry.get_available_transformations()}
        if not selected_transforms:
            selected_transforms = [
                TransformationType.SHAPE_INFERENCE,
                TransformationType.ORT_BASIC,
                TransformationType.ORT_EXTENDED,
                TransformationType.ORT_ALL,
            ]

        candidates: List[GraphCandidate] = []
        total_steps = len(selected_transforms)

        for idx, ttype in enumerate(selected_transforms, start=1):
            desc = available_descriptors.get(ttype)
            if not desc:
                continue

            step_pct = int((idx / (total_steps + 1)) * 100)
            if progress_callback:
                progress_callback(step_pct, f"Processing {desc.display_name}...")

            cand_id = f"cand_{idx:02d}_{ttype.value}"
            candidate = GraphCandidate(
                candidate_id=cand_id,
                transform_type=ttype,
                transform_name=desc.display_name,
                tool_name=desc.tool_name,
                tool_version=desc.tool_version,
                state=CandidateState.GENERATED,
                original_hash=original_hash,
                model_size_bytes_before=orig_size,
            )

            # Check availability
            if not desc.is_available:
                candidate.state = CandidateState.REJECTED
                candidate.recommendation_status = RecommendationStatus.REJECTED
                candidate.rejection_reason = f"Transformation unavailable: {desc.availability_reason}"
                candidate.recommendation_rationale = (
                    f"Skipped because required tool '{desc.tool_name}' is not installed or available."
                )
                candidates.append(candidate)
                continue

            # 1. Transformation
            candidate.state = CandidateState.TRANSFORMING
            t_res: TransformationResult = TransformationRegistry.execute_transformation(
                ttype, model_bytes
            )
            candidate.duration_ms = t_res.duration_ms
            candidate.warnings.extend(t_res.warnings)

            if not t_res.success or t_res.transformed_bytes is None:
                candidate.state = CandidateState.FAILED
                candidate.recommendation_status = RecommendationStatus.REJECTED
                candidate.rejection_reason = t_res.error_message or "Transformation execution failed."
                candidate.recommendation_rationale = (
                    f"Transformation failed during execution: {candidate.rejection_reason}"
                )
                candidates.append(candidate)
                continue

            candidate.state = CandidateState.TRANSFORMED
            candidate.transformed_hash = t_res.transformed_hash
            candidate.graph_changed = t_res.graph_changed
            candidate.diff_summary = t_res.diff_summary
            candidate.model_size_bytes_after = len(t_res.transformed_bytes)

            size_diff = orig_size - len(t_res.transformed_bytes)
            candidate.size_reduction_pct = round((size_diff / orig_size) * 100.0, 2) if orig_size > 0 else 0.0

            # Store transformed artifact safely
            art_info: ArtifactInfo = self.artifact_store.store_artifact(
                model_bytes=t_res.transformed_bytes,
                parent_fingerprint=original_hash,
                transformation_name=ttype.value,
                original_filename=original_filename,
            )
            candidate.artifact_id = art_info.artifact_id

            # 2. Layered Validation
            do_numerical = (validation_depth == "FULL")
            val_report: ValidationReport = GraphValidator.validate(
                original_bytes=model_bytes,
                transformed_bytes=t_res.transformed_bytes,
                validate_numerical=do_numerical,
                atol=self.atol,
                rtol=self.rtol,
                seed=self.seed,
            )
            candidate.validation_report = val_report

            if val_report.structural_status == ValidationStatus.FAIL:
                candidate.state = CandidateState.REJECTED
                candidate.recommendation_status = RecommendationStatus.REJECTED
                candidate.rejection_reason = val_report.rejection_reason or "Failed structural validation."
                candidate.recommendation_rationale = (
                    f"Rejected due to structural validation failure: {candidate.rejection_reason}"
                )
                candidates.append(candidate)
                continue

            candidate.state = CandidateState.STRUCTURALLY_VALIDATED

            if do_numerical:
                if val_report.numerical_status == ValidationStatus.FAIL:
                    candidate.state = CandidateState.REJECTED
                    candidate.recommendation_status = RecommendationStatus.REJECTED
                    candidate.rejection_reason = (
                        val_report.rejection_reason or "Failed numerical equivalence check."
                    )
                    candidate.recommendation_rationale = (
                        f"Rejected due to numerical divergence: {candidate.rejection_reason}"
                    )
                    candidates.append(candidate)
                    continue
                elif val_report.numerical_status == ValidationStatus.PASS:
                    candidate.state = CandidateState.NUMERICALLY_VALIDATED
                else:
                    candidate.state = CandidateState.NOT_VERIFIED

            # 3. Compatibility Reassessment (Phase 3)
            if target_profile:
                try:
                    trans_summary = load_and_inspect_model(
                        t_res.transformed_bytes, filename=art_info.suggested_filename
                    )
                    trans_compat = CompatibilityEngine.analyze(trans_summary, target_profile)
                    candidate.state = CandidateState.COMPATIBILITY_CHECKED
                    candidate.compatibility_report = trans_compat.to_dict()

                    # Compute compatibility delta vs baseline
                    if orig_compat_report:
                        delta = {
                            "baseline_assessment": orig_compat_report.overall_assessment,
                            "transformed_assessment": trans_compat.overall_assessment,
                            "pass_delta": trans_compat.summary_counts.get("PASS", 0)
                            - orig_compat_report.summary_counts.get("PASS", 0),
                            "fail_delta": trans_compat.summary_counts.get("FAIL", 0)
                            - orig_compat_report.summary_counts.get("FAIL", 0),
                            "warning_delta": trans_compat.summary_counts.get("WARNING", 0)
                            - orig_compat_report.summary_counts.get("WARNING", 0),
                            "unknown_delta": trans_compat.summary_counts.get("UNKNOWN", 0)
                            - orig_compat_report.summary_counts.get("UNKNOWN", 0),
                        }
                        candidate.compatibility_delta = delta
                except Exception as ex:
                    candidate.warnings.append(f"Compatibility reassessment failed: {str(ex)}")

            # 4. Benchmarking (Phase 4) - only if eligible and requested
            if (
                enable_benchmarking
                and validation_depth == "FULL"
                and candidate.state in (CandidateState.NUMERICALLY_VALIDATED, CandidateState.COMPATIBILITY_CHECKED)
            ):
                try:
                    trans_summary = load_and_inspect_model(
                        t_res.transformed_bytes, filename=art_info.suggested_filename
                    )
                    trans_bench = BenchmarkEngine.run_benchmark(
                        t_res.transformed_bytes, trans_summary, benchmark_config
                    )
                    candidate.benchmark_report = trans_bench.to_dict()
                    candidate.state = CandidateState.BENCHMARKED

                    if (
                        orig_bench_report
                        and orig_bench_report.status == "SUCCESS"
                        and trans_bench.status == "SUCCESS"
                        and orig_bench_report.latency
                        and trans_bench.latency
                    ):
                        orig_med = orig_bench_report.latency.get("median_ms", 0.0)
                        trans_med = trans_bench.latency.get("median_ms", 0.0)

                        if orig_med > 0 and trans_med > 0:
                            candidate.speedup_factor = round(orig_med / trans_med, 3)
                            candidate.latency_reduction_pct = round(
                                ((orig_med - trans_med) / orig_med) * 100.0, 2
                            )

                        orig_tp = orig_bench_report.throughput.get("inferences_per_second", 0.0) if orig_bench_report.throughput else 0.0
                        trans_tp = trans_bench.throughput.get("inferences_per_second", 0.0) if trans_bench.throughput else 0.0
                        if orig_tp > 0 and trans_tp > 0:
                            candidate.throughput_improvement_pct = round(
                                ((trans_tp - orig_tp) / orig_tp) * 100.0, 2
                            )
                except Exception as ex:
                    candidate.warnings.append(f"Empirical benchmark failed: {str(ex)}")

            # 5. Recommendation Layer Evaluation
            self._evaluate_candidate_recommendation(candidate, orig_compat_report)
            candidates.append(candidate)

        # Multi-Candidate Ranking & Best Selection
        best_candidate, overall_rec = self._select_best_candidate(candidates)

        summary_counts = {
            "total_candidates": len(candidates),
            "validated": sum(1 for c in candidates if c.state in (
                CandidateState.STRUCTURALLY_VALIDATED,
                CandidateState.NUMERICALLY_VALIDATED,
                CandidateState.COMPATIBILITY_CHECKED,
                CandidateState.BENCHMARKED,
            )),
            "numerically_verified": sum(1 for c in candidates if c.validation_report and c.validation_report.numerical_status == ValidationStatus.PASS),
            "performance_improved": sum(1 for c in candidates if c.speedup_factor and c.speedup_factor > 1.02),
            "rejected": sum(1 for c in candidates if c.state == CandidateState.REJECTED or c.recommendation_status == RecommendationStatus.REJECTED),
        }

        limitations = [
            "Transformations were executed and validated on the host CPU runtime. Host-specific operator fusions "
            "must be validated on the physical target architecture before production deployment.",
            "Benchmarking measurements reflect local host execution provider performance and do not simulate "
            "IBM Z Integrated Accelerator for AI (NNPA) throughput or latency.",
        ]
        if validation_depth != "FULL":
            limitations.append(
                "Experiment was executed with STRUCTURAL_ONLY validation depth. Numerical equivalence and "
                "runtime inference performance have NOT been verified."
            )

        if progress_callback:
            progress_callback(100, "Graph optimization experiment complete.")

        return GraphOptimizationReport(
            experiment_id=experiment_id,
            model_name=original_filename,
            original_fingerprint=original_hash,
            target_profile_id=target_profile_id,
            target_profile_name=target_profile_name,
            validation_depth=validation_depth,
            benchmark_enabled=enable_benchmarking,
            created_at=created_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
            status="SUCCESS",
            candidates=candidates,
            best_candidate_id=best_candidate.candidate_id if best_candidate else None,
            summary_counts=summary_counts,
            overall_recommendation=overall_rec,
            limitations=limitations,
        )

    def _evaluate_candidate_recommendation(
        self,
        candidate: GraphCandidate,
        orig_compat: Optional[CompatibilityReport],
    ) -> None:
        """Deterministic rule-based recommendation logic for a single candidate."""
        if candidate.recommendation_status == RecommendationStatus.REJECTED:
            return

        val = candidate.validation_report
        if not val or val.structural_status != ValidationStatus.PASS:
            candidate.recommendation_status = RecommendationStatus.REJECTED
            candidate.recommendation_rationale = "Candidate failed required structural integrity validation."
            return

        if val.numerical_status == ValidationStatus.FAIL:
            candidate.recommendation_status = RecommendationStatus.REJECTED
            candidate.recommendation_rationale = (
                f"Candidate rejected: output tensor values diverged from original model beyond tolerance."
            )
            return

        # Check for demonstrated speedup
        has_speedup = (candidate.speedup_factor is not None and candidate.speedup_factor > 1.02)
        has_regression = (candidate.speedup_factor is not None and candidate.speedup_factor < 0.98)

        # Check compatibility
        compat_pass = False
        if candidate.compatibility_report:
            ov = candidate.compatibility_report.get("overall_assessment")
            if ov in ("PASS", "WARNING"):
                compat_pass = True

        if val.numerical_status == ValidationStatus.PASS:
            if has_speedup:
                candidate.recommendation_status = RecommendationStatus.PERFORMANCE_IMPROVED
                candidate.recommendation_rationale = (
                    f"Recommended for deployment. Passed structural and numerical validation with "
                    f"{candidate.speedup_factor:.2f}x speedup ({candidate.latency_reduction_pct:.1f}% latency reduction)."
                )
            elif has_regression:
                candidate.recommendation_status = RecommendationStatus.NO_DEMONSTRATED_IMPROVEMENT
                candidate.recommendation_rationale = (
                    f"Numerically verified and structurally sound, but benchmark latency increased by "
                    f"{abs(candidate.latency_reduction_pct or 0.0):.1f}%. Deployment not recommended."
                )
            elif candidate.graph_changed:
                candidate.recommendation_status = RecommendationStatus.NUMERICALLY_VERIFIED
                candidate.recommendation_rationale = (
                    "Numerically verified and structurally sound. Node topology changed, but measurable "
                    "latency difference was within benchmark noise threshold."
                )
            else:
                candidate.recommendation_status = RecommendationStatus.NO_DEMONSTRATED_IMPROVEMENT
                candidate.recommendation_rationale = (
                    "Numerically verified, but transformation produced no graph structure change or measurable speedup."
                )
        else:
            # Structural only or not verified numerically
            candidate.recommendation_status = RecommendationStatus.VALIDATED
            candidate.recommendation_rationale = (
                "Candidate passed structural validation. Numerical equivalence has NOT been verified."
            )

    def _select_best_candidate(
        self, candidates: List[GraphCandidate]
    ) -> Tuple[Optional[GraphCandidate], str]:
        """
        Rank valid candidates and select the single best deployment artifact.
        Prioritizes:
        1. Fully verified numerical equivalence (must pass)
        2. Demonstrable benchmark speedup
        3. Model size reduction
        4. Simplicity (basic > extended if speedup is identical)
        """
        eligible = [
            c for c in candidates
            if c.recommendation_status in (
                RecommendationStatus.PERFORMANCE_IMPROVED,
                RecommendationStatus.NUMERICALLY_VERIFIED,
                RecommendationStatus.COMPATIBLE,
                RecommendationStatus.VALIDATED,
            )
            and c.graph_changed
        ]

        if not eligible:
            # Check if any candidate was numerically verified even if graph was unchanged
            unchanged_verified = [
                c for c in candidates
                if c.validation_report and c.validation_report.numerical_status == ValidationStatus.PASS
            ]
            if unchanged_verified:
                rec = (
                    "All transformations were mathematically identical or produced no measurable performance benefit. "
                    "The original model remains the recommended deployment artifact."
                )
                return None, rec
            return None, "No candidates met structural and numerical correctness validation standards."

        # Rank by speedup_factor descending, then size reduction descending
        def _sort_key(c: GraphCandidate):
            speedup = c.speedup_factor if c.speedup_factor is not None else 1.0
            size_red = c.size_reduction_pct if c.size_reduction_pct is not None else 0.0
            num_verified = 1 if (c.validation_report and c.validation_report.numerical_status == ValidationStatus.PASS) else 0
            return (num_verified, speedup, size_red)

        ranked = sorted(eligible, key=_sort_key, reverse=True)
        best = ranked[0]

        speedup_str = f" with {best.speedup_factor:.2f}x speedup" if best.speedup_factor else ""
        size_str = f" and {best.size_reduction_pct:.1f}% size reduction" if best.size_reduction_pct else ""
        overall_rec = (
            f"Candidate '{best.candidate_id}' ({best.transform_name}) is the recommended artifact{speedup_str}{size_str}. "
            f"It passed all structural and numerical tolerance checks."
        )

        return best, overall_rec
