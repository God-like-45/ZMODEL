"""Benchmark comparison service enforcing strict model fingerprint, workload, and measurement comparability."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from zmodel.db.models import BenchmarkRunORM


class ComparisonValidity(str, Enum):
    VALID = "VALID"
    INCOMPATIBLE_MODELS = "INCOMPATIBLE_MODELS"
    INCOMPATIBLE_WORKLOAD = "INCOMPATIBLE_WORKLOAD"
    INVALID_CORRECTNESS = "INVALID_CORRECTNESS"


@dataclass
class MetricComparison:
    """Individual metric delta comparison between baseline (A) and candidate (B)."""
    metric_name: str
    unit: str
    value_a: Optional[float]
    value_b: Optional[float]
    absolute_delta: Optional[float]
    percentage_improvement: Optional[float]
    speedup_factor: Optional[float] = None
    interpretation: str = ""


@dataclass
class BenchmarkComparisonResult:
    """Rigorous, auditable side-by-side benchmark comparison result."""
    is_valid: bool
    validity: ComparisonValidity
    disqualification_reasons: List[str]
    run_a_id: str
    run_b_id: str
    model_name_a: str
    model_name_b: str
    model_fingerprint: str
    metric_comparisons: Dict[str, MetricComparison] = field(default_factory=dict)
    configuration_differences: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "validity": self.validity.value,
            "disqualification_reasons": self.disqualification_reasons,
            "run_a_id": self.run_a_id,
            "run_b_id": self.run_b_id,
            "model_fingerprint": self.model_fingerprint,
            "configuration_differences": self.configuration_differences,
            "metric_comparisons": {
                k: {
                    "metric_name": v.metric_name,
                    "unit": v.unit,
                    "value_a": v.value_a,
                    "value_b": v.value_b,
                    "absolute_delta": v.absolute_delta,
                    "percentage_improvement": v.percentage_improvement,
                    "speedup_factor": v.speedup_factor,
                    "interpretation": v.interpretation,
                }
                for k, v in self.metric_comparisons.items()
            },
        }


def compare_benchmark_runs(run_a: BenchmarkRunORM, run_b: BenchmarkRunORM) -> BenchmarkComparisonResult:
    """
    Compare two empirical benchmark runs, verifying comparability before calculating relative metrics.
    Run A is treated as the reference/baseline; Run B is the comparison candidate.
    """
    disqualifications: List[str] = []

    # 1. Model Fingerprint Verification
    if run_a.model_fingerprint != run_b.model_fingerprint:
        disqualifications.append(
            f"Model fingerprints differ (Run A: {run_a.model_fingerprint[:12]} vs Run B: {run_b.model_fingerprint[:12]}). "
            "Cross-model performance comparisons are scientifically invalid."
        )

    # 2. Correctness Status Verification
    if run_a.correctness_status != "PASS":
        disqualifications.append(f"Run A failed correctness verification (status: {run_a.correctness_status}).")
    if run_b.correctness_status != "PASS":
        disqualifications.append(f"Run B failed correctness verification (status: {run_b.correctness_status}).")

    # 3. Workload Semantics: Batch Size & Dynamic Dimensions Check
    cfg_a = run_a.configuration or {}
    cfg_b = run_b.configuration or {}
    batch_a = cfg_a.get("batch_size")
    batch_b = cfg_b.get("batch_size")

    if batch_a is not None and batch_b is not None and batch_a != batch_b:
        disqualifications.append(
            f"Workload batch sizes differ (Run A: {batch_a} vs Run B: {batch_b}). "
            "Changing batch size alters compute semantics; per-inference latency speedup cannot be directly claimed."
        )

    # 4. Input Shapes Check
    input_spec_a = run_a.input_specifications or {}
    input_spec_b = run_b.input_specifications or {}
    shapes_a = input_spec_a.get("resolved_shapes", {})
    shapes_b = input_spec_b.get("resolved_shapes", {})
    if shapes_a and shapes_b and shapes_a != shapes_b:
        disqualifications.append(
            f"Resolved input tensor shapes differ (Run A: {shapes_a} vs Run B: {shapes_b}). "
            "Workloads with differing input geometries cannot be directly compared."
        )

    # Find configuration differences
    config_diffs: Dict[str, Tuple[Any, Any]] = {}
    all_keys = set(cfg_a.keys()).union(set(cfg_b.keys()))
    for k in sorted(all_keys):
        val_a = cfg_a.get(k)
        val_b = cfg_b.get(k)
        if val_a != val_b:
            config_diffs[k] = (val_a, val_b)

    if run_a.execution_provider != run_b.execution_provider:
        config_diffs["execution_provider"] = (run_a.execution_provider, run_b.execution_provider)

    # Determine validity
    if disqualifications:
        validity = ComparisonValidity.INCOMPATIBLE_MODELS if run_a.model_fingerprint != run_b.model_fingerprint else ComparisonValidity.INCOMPATIBLE_WORKLOAD
        return BenchmarkComparisonResult(
            is_valid=False,
            validity=validity,
            disqualification_reasons=disqualifications,
            run_a_id=run_a.id,
            run_b_id=run_b.id,
            model_name_a=run_a.model_name,
            model_name_b=run_b.model_name,
            model_fingerprint=run_a.model_fingerprint,
            configuration_differences=config_diffs,
        )

    # If valid, calculate metric comparisons
    metrics: Dict[str, MetricComparison] = {}

    # Median Latency (Lower is better)
    if run_a.median_latency_ms is not None and run_b.median_latency_ms is not None:
        val_a = run_a.median_latency_ms
        val_b = run_b.median_latency_ms
        delta = val_b - val_a
        reduction_pct = ((val_a - val_b) / val_a * 100.0) if val_a > 0 else None
        speedup = (val_a / val_b) if val_b > 0 else None
        interp = "Faster" if delta < 0 else ("Slower (Regression)" if delta > 0 else "Equal")
        metrics["median_latency"] = MetricComparison(
            metric_name="Median Latency",
            unit="ms",
            value_a=val_a,
            value_b=val_b,
            absolute_delta=round(delta, 4),
            percentage_improvement=round(reduction_pct, 2) if reduction_pct is not None else None,
            speedup_factor=round(speedup, 2) if speedup is not None else None,
            interpretation=interp,
        )

    # P95 Latency (Lower is better)
    if run_a.p95_latency_ms is not None and run_b.p95_latency_ms is not None:
        val_a = run_a.p95_latency_ms
        val_b = run_b.p95_latency_ms
        delta = val_b - val_a
        reduction_pct = ((val_a - val_b) / val_a * 100.0) if val_a > 0 else None
        speedup = (val_a / val_b) if val_b > 0 else None
        interp = "Faster (Tail)" if delta < 0 else ("Slower (Tail Regression)" if delta > 0 else "Equal")
        metrics["p95_latency"] = MetricComparison(
            metric_name="P95 Latency",
            unit="ms",
            value_a=val_a,
            value_b=val_b,
            absolute_delta=round(delta, 4),
            percentage_improvement=round(reduction_pct, 2) if reduction_pct is not None else None,
            speedup_factor=round(speedup, 2) if speedup is not None else None,
            interpretation=interp,
        )

    # Throughput (Higher is better)
    if run_a.inferences_per_second is not None and run_b.inferences_per_second is not None:
        val_a = run_a.inferences_per_second
        val_b = run_b.inferences_per_second
        delta = val_b - val_a
        imp_pct = ((val_b - val_a) / val_a * 100.0) if val_a > 0 else None
        interp = "Higher Throughput" if delta > 0 else ("Lower Throughput" if delta < 0 else "Equal")
        metrics["throughput"] = MetricComparison(
            metric_name="Throughput",
            unit="inf/sec",
            value_a=val_a,
            value_b=val_b,
            absolute_delta=round(delta, 2),
            percentage_improvement=round(imp_pct, 2) if imp_pct is not None else None,
            speedup_factor=None,
            interpretation=interp,
        )

    # Memory Session Delta (Lower is better when comparable)
    if run_a.session_delta_mb is not None and run_b.session_delta_mb is not None:
        val_a = run_a.session_delta_mb
        val_b = run_b.session_delta_mb
        delta = val_b - val_a
        interp = "Less Memory" if delta < 0 else ("More Memory" if delta > 0 else "Equal")
        metrics["session_delta_memory"] = MetricComparison(
            metric_name="Session Memory Delta",
            unit="MB",
            value_a=val_a,
            value_b=val_b,
            absolute_delta=round(delta, 2),
            percentage_improvement=None,
            speedup_factor=None,
            interpretation=interp,
        )

    # Peak RSS (Lower is better when comparable)
    if run_a.peak_rss_mb is not None and run_b.peak_rss_mb is not None:
        val_a = run_a.peak_rss_mb
        val_b = run_b.peak_rss_mb
        delta = val_b - val_a
        interp = "Lower Peak RSS" if delta < 0 else ("Higher Peak RSS" if delta > 0 else "Equal")
        metrics["peak_rss"] = MetricComparison(
            metric_name="Peak Process RSS",
            unit="MB",
            value_a=val_a,
            value_b=val_b,
            absolute_delta=round(delta, 2),
            percentage_improvement=None,
            speedup_factor=None,
            interpretation=interp,
        )

    return BenchmarkComparisonResult(
        is_valid=True,
        validity=ComparisonValidity.VALID,
        disqualification_reasons=[],
        run_a_id=run_a.id,
        run_b_id=run_b.id,
        model_name_a=run_a.model_name,
        model_name_b=run_b.model_name,
        model_fingerprint=run_a.model_fingerprint,
        metric_comparisons=metrics,
        configuration_differences=config_diffs,
    )
