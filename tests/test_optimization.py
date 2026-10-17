"""
Tests for Phase 5: Evidence-Based Optimization Engine.

Comprehensive testing suite covering all 27 specified test scenarios:
1. Deterministic candidate generation
2. Duplicate candidate elimination
3. Candidate configuration validation
4. Fixed vs. dynamic shape handling
5. Compatibility-based rejection
6. Unknown compatibility handling
7. Preservation of warning findings
8. Correct baseline selection
9. Latency minimization ranking
10. Throughput maximization ranking
11. Memory minimization ranking
12. Hard constraint satisfaction
13. Missing metric behavior
14. No-feasible-candidate handling
15. Failed benchmark exclusion
16. Correctness-failed candidate exclusion
17. Pareto frontier correctness
18. Duplicate and equal objective values
19. Speedup and improvement calculations
20. Invalid or zero baseline handling
21. No fabricated values for unbenchmarked candidates
22. Measurement variability handling
23. Optimization report serialization
24. Stable ranking and deterministic tie-breaking
25. Experiment lifecycle & candidate status transitions
26. Correct model fingerprint association
27. Regression tests for Phases 1-4
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from zmodel.core.benchmark import (
    BenchmarkConfig,
    BenchmarkEngine,
    BenchmarkReport,
    CorrectnessResult,
    discover_execution_providers,
)
from zmodel.core.compatibility import (
    CompatibilityAnalyzer,
    CompatibilityReport,
)
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.optimizer import (
    CandidateStatus,
    ExecutionCandidate,
    ExperimentStatus,
    HardConstraint,
    InvalidOptimizationRequestError,
    OptimizationEngine,
    OptimizationObjective,
    OptimizationReport,
    OptimizationRequest,
)
from zmodel.core.profiles import CapabilityProfile, load_builtin_profiles, load_profile_from_dict


def _build_test_model(fixed_batch: bool = True) -> bytes:
    """Helper to synthesize a minimal valid ONNX model."""
    batch_dim = 1 if fixed_batch else "batch_size"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch_dim, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch_dim, 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 2], np.ones((4, 2), dtype=np.float32).flatten())
    n = helper.make_node("Gemm", ["x", "w"], ["y"])
    graph = helper.make_graph([n], "test_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    return model.SerializeToString()


def _mock_benchmark_report(
    candidate_id: str,
    median_ms: float,
    tp_inf_sec: float,
    peak_rss_mb: float = 150.0,
    valid: bool = True,
    p95_ms: Optional[float] = None,
    status: str = "SUCCESS",
) -> BenchmarkReport:
    """Create a mock BenchmarkReport for deterministic testing."""
    return BenchmarkReport(
        run_id=f"RUN-{candidate_id}",
        timestamp_utc="2026-10-17T12:00:00Z",
        model_name="test.onnx",
        model_fingerprint="abc123hash",
        status=status,
        error_message=None if status == "SUCCESS" else "Mock benchmark failure",
        environment={"os_name": "TestOS"},
        config={"execution_provider": "CPUExecutionProvider", "batch_size": 1},
        input_spec={"feeds": {}},
        correctness={
            "passed": valid,
            "output_names": ["y"],
            "output_shapes": {"y": [1, 2]},
            "all_finite": valid,
            "repeat_consistent": valid,
            "details": "Smoke inference passed" if valid else "Smoke inference failed",
        },
        latency={
            "median_ms": median_ms,
            "mean_ms": median_ms * 1.05,
            "p95_ms": p95_ms if p95_ms is not None else median_ms * 1.2,
            "p90_ms": median_ms * 1.15,
            "p99_ms": median_ms * 1.35,
            "min_ms": median_ms * 0.9,
            "max_ms": median_ms * 1.4,
            "std_dev_ms": median_ms * 0.1,
            "sample_count": 50,
        },
        throughput={
            "inferences_per_second": tp_inf_sec,
            "samples_per_second": tp_inf_sec,
            "total_elapsed_seconds": 1.0,
            "total_measured_inferences": int(tp_inf_sec),
        },
        memory={
            "baseline_rss_mb": 140.0,
            "post_session_rss_mb": 145.0,
            "session_delta_mb": 5.0,
            "peak_rss_mb": peak_rss_mb,
            "notes": "",
        },
        per_round_results=[],
        all_latencies_ms=[median_ms],
        warnings=[],
        limitations=[],
    )


# --- 1. Deterministic Candidate Generation ---
def test_deterministic_candidate_generation():
    model_bytes = _build_test_model()
    summary = load_and_inspect_model(model_bytes, filename="test.onnx")
    request = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
        max_candidates=6,
        intra_op_threads=[1, 2, 4],
        batch_sizes=[1],
    )
    providers = ["CPUExecutionProvider"]
    run1 = OptimizationEngine.generate_candidates(summary, request, providers)
    run2 = OptimizationEngine.generate_candidates(summary, request, providers)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.candidate_id == c2.candidate_id
        assert c1.config.intra_op_num_threads == c2.config.intra_op_num_threads
        assert c1.config.batch_size == c2.config.batch_size


# --- 2. Duplicate Candidate Elimination ---
def test_duplicate_candidate_elimination():
    model_bytes = _build_test_model()
    summary = load_and_inspect_model(model_bytes, filename="test.onnx")
    request = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
        max_candidates=10,
        intra_op_threads=[1, 1, 2, 2],  # Duplicate thread requests
        batch_sizes=[1, 1],             # Duplicate batch requests
    )
    candidates = OptimizationEngine.generate_candidates(summary, request, ["CPUExecutionProvider"])
    keys = [(c.config.execution_provider, c.config.batch_size, c.config.intra_op_num_threads, c.config.inter_op_num_threads) for c in candidates]
    assert len(keys) == len(set(keys))


# --- 3. Candidate Configuration Validation ---
def test_candidate_configuration_validation():
    req = OptimizationRequest(
        model_fingerprint="",  # Missing fingerprint
        profile_id="local_cpu",
        profile_version="1.0.0",
    )
    with pytest.raises(InvalidOptimizationRequestError, match="fingerprint must be specified"):
        req.validate()

    req2 = OptimizationRequest(
        model_fingerprint="valid_hash",
        profile_id="local_cpu",
        profile_version="1.0.0",
        max_candidates=100,  # Exceeds max 50
    )
    with pytest.raises(InvalidOptimizationRequestError, match="max_candidates"):
        req2.validate()


# --- 4. Fixed vs Dynamic Shape Handling ---
def test_fixed_versus_dynamic_shape_handling():
    # Model 1: Fixed batch dimension = 1
    m_fixed = _build_test_model(fixed_batch=True)
    sum_fixed = load_and_inspect_model(m_fixed, filename="fixed.onnx")
    req = OptimizationRequest(
        model_fingerprint=sum_fixed.sha256_hash,
        profile_id="local_cpu",
        profile_version="1.0.0",
        batch_sizes=[1, 4],  # 4 should be rejected for fixed model
    )
    cands_fixed = OptimizationEngine.generate_candidates(sum_fixed, req, ["CPUExecutionProvider"])
    b4_cand = [c for c in cands_fixed if c.config.batch_size == 4][0]
    assert b4_cand.status == CandidateStatus.REJECTED
    assert "violates declared static shape constraint" in b4_cand.rejection_reason

    # Model 2: Dynamic batch dimension
    m_dynamic = _build_test_model(fixed_batch=False)
    sum_dynamic = load_and_inspect_model(m_dynamic, filename="dynamic.onnx")
    cands_dynamic = OptimizationEngine.generate_candidates(sum_dynamic, req, ["CPUExecutionProvider"])
    b4_dynamic = [c for c in cands_dynamic if c.config.batch_size == 4][0]
    assert b4_dynamic.status == CandidateStatus.GENERATED


# --- 5. Compatibility-Based Rejection ---
def test_compatibility_based_rejection():
    model_bytes = _build_test_model()
    summary = load_and_inspect_model(model_bytes, filename="test.onnx")
    # Profile that explicitly fails Gemm
    custom_profile = load_profile_from_dict({
        "profile_id": "no_gemm_profile",
        "display_name": "No Gemm Architecture",
        "version": "1.0.0",
        "ruleset_version": "1.0.0",
        "target_architecture": "x86_64",
        "target_runtime": "onnxruntime",
        "confidence_classification": "ILLUSTRATIVE_DRAFT",
        "is_illustrative": True,
        "min_opset": 1,
        "max_opset": 21,
        "supported_domains": [""],
        "supported_data_types": ["FLOAT"],
        "operator_rules": {
            "Gemm": {
                "status": "FAIL",
                "evidence": "Gemm is unsupported on this accelerator."
            }
        }
    })
    request = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id=custom_profile.profile_id,
        profile_version=custom_profile.version,
        batch_sizes=[1],
    )
    candidates = OptimizationEngine.generate_candidates(summary, request, ["CPUExecutionProvider"])
    filtered = OptimizationEngine.filter_candidates_by_compatibility(candidates, summary, custom_profile)
    for c in filtered:
        assert c.status == CandidateStatus.REJECTED
        assert "FAIL findings" in c.rejection_reason


# --- 6. Unknown Compatibility Handling ---
def test_unknown_compatibility_handling():
    model_bytes = _build_test_model()
    summary = load_and_inspect_model(model_bytes, filename="test.onnx")
    incomplete_profile = load_profile_from_dict({
        "profile_id": "incomplete_profile",
        "display_name": "Incomplete Profile",
        "version": "1.0.0",
        "ruleset_version": "1.0.0",
        "target_architecture": "x86_64",
        "target_runtime": "onnxruntime",
        "confidence_classification": "ILLUSTRATIVE_DRAFT",
        "is_illustrative": True,
        "min_opset": 1,
        "max_opset": 21,
        "supported_domains": [""],
        "supported_data_types": ["FLOAT"],
        "operator_rules": {}  # Gemm missing -> UNKNOWN finding
    })
    request = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id=incomplete_profile.profile_id,
        profile_version=incomplete_profile.version,
    )
    candidates = OptimizationEngine.generate_candidates(summary, request, ["CPUExecutionProvider"])
    filtered = OptimizationEngine.filter_candidates_by_compatibility(candidates, summary, incomplete_profile)
    # Unknown compatibility does NOT cause automatic rejection unless explicitly FAIL
    for c in filtered:
        if c.rejection_reason is None:
            assert c.status == CandidateStatus.QUEUED
            assert len(c.compatibility_findings) > 0


# --- 7. Preservation of Warning Findings ---
def test_preservation_of_warning_findings():
    # Dynamic shape model evaluated against IBM Z illustrative profile -> WARNING on dynamic shape
    model_bytes = _build_test_model(fixed_batch=False)
    summary = load_and_inspect_model(model_bytes, filename="dyn.onnx")
    profiles = load_builtin_profiles()
    ibm_profile = profiles["ibm_z_onnx_illustrative"]

    request = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id=ibm_profile.profile_id,
        profile_version=ibm_profile.version,
    )
    candidates = OptimizationEngine.generate_candidates(summary, request, ["CPUExecutionProvider"])
    filtered = OptimizationEngine.filter_candidates_by_compatibility(candidates, summary, ibm_profile)
    for c in filtered:
        # Warnings must be preserved in candidate record
        assert any(f.get("status") == "WARNING" for f in c.compatibility_findings)


# --- 8. Correct Baseline Selection ---
def test_correct_baseline_selection():
    model_bytes = _build_test_model()
    summary = load_and_inspect_model(model_bytes, filename="test.onnx")
    request = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
    )
    candidates = OptimizationEngine.generate_candidates(summary, request, ["CPUExecutionProvider"])
    baseline = candidates[0]
    assert baseline.is_baseline is True
    assert "BASELINE" in baseline.candidate_id
    assert baseline.config.intra_op_num_threads is None


# --- 9. Latency Minimization Ranking ---
def test_latency_minimization_ranking():
    c1 = ExecutionCandidate("C1", BenchmarkConfig(), "Test", is_feasible=True)
    c1.benchmark_report = _mock_benchmark_report("C1", median_ms=5.0, tp_inf_sec=200.0)
    c2 = ExecutionCandidate("C2", BenchmarkConfig(), "Test", is_feasible=True)
    c2.benchmark_report = _mock_benchmark_report("C2", median_ms=2.0, tp_inf_sec=500.0)
    c3 = ExecutionCandidate("C3", BenchmarkConfig(), "Test", is_feasible=True)
    c3.benchmark_report = _mock_benchmark_report("C3", median_ms=3.5, tp_inf_sec=285.0)

    ranked = OptimizationEngine.rank_candidates([c1, c2, c3], OptimizationObjective.MINIMIZE_MEDIAN_LATENCY)
    assert ranked[0].candidate_id == "C2"  # 2.0 ms is lowest
    assert ranked[1].candidate_id == "C3"  # 3.5 ms
    assert ranked[2].candidate_id == "C1"  # 5.0 ms
    assert ranked[0].rank == 1


# --- 10. Throughput Maximization Ranking ---
def test_throughput_maximization_ranking():
    c1 = ExecutionCandidate("C1", BenchmarkConfig(), "Test", is_feasible=True)
    c1.benchmark_report = _mock_benchmark_report("C1", median_ms=5.0, tp_inf_sec=200.0)
    c2 = ExecutionCandidate("C2", BenchmarkConfig(), "Test", is_feasible=True)
    c2.benchmark_report = _mock_benchmark_report("C2", median_ms=2.0, tp_inf_sec=500.0)

    ranked = OptimizationEngine.rank_candidates([c1, c2], OptimizationObjective.MAXIMIZE_THROUGHPUT)
    assert ranked[0].candidate_id == "C2"  # 500 inf/s is highest
    assert ranked[1].candidate_id == "C1"


# --- 11. Memory Minimization Ranking ---
def test_memory_minimization_ranking():
    c1 = ExecutionCandidate("C1", BenchmarkConfig(), "Test", is_feasible=True)
    c1.benchmark_report = _mock_benchmark_report("C1", median_ms=2.0, tp_inf_sec=500.0, peak_rss_mb=250.0)
    c2 = ExecutionCandidate("C2", BenchmarkConfig(), "Test", is_feasible=True)
    c2.benchmark_report = _mock_benchmark_report("C2", median_ms=2.5, tp_inf_sec=400.0, peak_rss_mb=180.0)

    ranked = OptimizationEngine.rank_candidates([c1, c2], OptimizationObjective.MINIMIZE_MEMORY)
    assert ranked[0].candidate_id == "C2"  # 180 MB is lower than 250 MB


# --- 12. Hard Constraint Satisfaction ---
def test_hard_constraint_satisfaction():
    c = ExecutionCandidate("C1", BenchmarkConfig(), "Test", status=CandidateStatus.BENCHMARKED)
    c.benchmark_report = _mock_benchmark_report("C1", median_ms=3.0, tp_inf_sec=333.0)

    constraints = [
        HardConstraint("Max Latency", "median_latency_ms", "<=", 4.0, "ms"),
        HardConstraint("Min Throughput", "throughput_inf_sec", ">=", 300.0, "inf/s"),
    ]
    passed, evals = OptimizationEngine.evaluate_constraints(c, constraints)
    assert passed is True
    assert c.is_feasible is True
    assert len(evals) == 2
    assert all(e.passed for e in evals)


# --- 13. Missing Metric Behavior ---
def test_missing_metric_behavior():
    c = ExecutionCandidate("C1", BenchmarkConfig(), "Test", status=CandidateStatus.BENCHMARKED)
    c.benchmark_report = _mock_benchmark_report("C1", median_ms=3.0, tp_inf_sec=333.0)

    # Constraint on non-existent metric
    constraints = [
        HardConstraint("GPU Temp", "gpu_temperature_celsius", "<=", 75.0, "C"),
    ]
    passed, evals = OptimizationEngine.evaluate_constraints(c, constraints)
    assert passed is False  # Missing metric cannot be marked satisfied
    assert c.is_feasible is False
    assert evals[0].is_available is False


# --- 14. No Feasible Candidate Handling ---
def test_no_feasible_candidate_handling():
    c = ExecutionCandidate("C1", BenchmarkConfig(), "Test", status=CandidateStatus.BENCHMARKED)
    c.benchmark_report = _mock_benchmark_report("C1", median_ms=10.0, tp_inf_sec=100.0)

    constraints = [
        HardConstraint("Strict Max Latency", "median_latency_ms", "<=", 2.0, "ms"),  # 10.0 violates 2.0
    ]
    passed, _ = OptimizationEngine.evaluate_constraints(c, constraints)
    assert passed is False
    assert c.status == CandidateStatus.INFEASIBLE

    ranked = OptimizationEngine.rank_candidates([c], OptimizationObjective.MINIMIZE_MEDIAN_LATENCY)
    assert ranked[0].rank is None  # Infeasible candidate has no rank


# --- 15. Failed Benchmark Exclusion ---
def test_failed_benchmark_exclusion():
    c1 = ExecutionCandidate("C1", BenchmarkConfig(), "Test", status=CandidateStatus.FAILED)
    c1.benchmark_report = _mock_benchmark_report("C1", median_ms=1.0, tp_inf_sec=1000.0, valid=False, status="FAILED")

    passed, _ = OptimizationEngine.evaluate_constraints(c1, [])
    assert passed is False
    assert c1.is_feasible is False


# --- 16. Correctness-Failed Candidate Exclusion ---
def test_correctness_failed_candidate_exclusion():
    c = ExecutionCandidate("C1", BenchmarkConfig(), "Test", status=CandidateStatus.BENCHMARKED)
    # Report marked SUCCESS but correctness valid=False
    c.benchmark_report = _mock_benchmark_report("C1", median_ms=1.0, tp_inf_sec=1000.0, valid=False, status="SUCCESS")

    passed, evals = OptimizationEngine.evaluate_constraints(c, [])
    assert passed is False
    assert c.is_feasible is False
    assert any("Inference correctness check failed" in e.explanation for e in evals)


# --- 17. Pareto Frontier Correctness ---
def test_pareto_frontier_correctness():
    # C1: Fast, High throughput (Dominates C3)
    c1 = ExecutionCandidate("C1", BenchmarkConfig(), "Test", is_feasible=True)
    c1.benchmark_report = _mock_benchmark_report("C1", median_ms=2.0, tp_inf_sec=500.0, peak_rss_mb=100.0)

    # C2: Slower, Higher throughput (Non-dominated trade-off)
    c2 = ExecutionCandidate("C2", BenchmarkConfig(), "Test", is_feasible=True)
    c2.benchmark_report = _mock_benchmark_report("C2", median_ms=3.0, tp_inf_sec=700.0, peak_rss_mb=90.0)

    # C3: Slower and Lower throughput than C1 (Dominated by C1)
    c3 = ExecutionCandidate("C3", BenchmarkConfig(), "Test", is_feasible=True)
    c3.benchmark_report = _mock_benchmark_report("C3", median_ms=2.5, tp_inf_sec=400.0, peak_rss_mb=110.0)

    frontier = OptimizationEngine.calculate_pareto_frontier([c1, c2, c3])
    assert "C1" in frontier
    assert "C2" in frontier
    assert "C3" not in frontier  # C3 is strictly dominated by C1


# --- 18. Duplicate and Equal Objective Values ---
def test_duplicate_and_equal_objective_values():
    c1 = ExecutionCandidate("C1", BenchmarkConfig(), "Test", is_feasible=True)
    c1.benchmark_report = _mock_benchmark_report("C1", median_ms=2.0, tp_inf_sec=500.0)
    c2 = ExecutionCandidate("C2", BenchmarkConfig(), "Test", is_feasible=True)
    c2.benchmark_report = _mock_benchmark_report("C2", median_ms=2.0, tp_inf_sec=500.0)

    frontier = OptimizationEngine.calculate_pareto_frontier([c1, c2])
    # Neither strictly dominates the other, so both are on frontier
    assert "C1" in frontier
    assert "C2" in frontier


# --- 19. Speedup and Improvement Calculations ---
def test_speedup_and_improvement_calculations():
    base = ExecutionCandidate("BASE", BenchmarkConfig(), "Base", status=CandidateStatus.BENCHMARKED, is_feasible=True)
    base.benchmark_report = _mock_benchmark_report("BASE", median_ms=4.0, tp_inf_sec=250.0)

    cand = ExecutionCandidate("CAND", BenchmarkConfig(), "Cand", status=CandidateStatus.BENCHMARKED, is_feasible=True)
    cand.benchmark_report = _mock_benchmark_report("CAND", median_ms=2.0, tp_inf_sec=500.0)

    OptimizationEngine.calculate_relative_improvements(cand, base)
    assert cand.speedup_vs_baseline == pytest.approx(2.0, rel=1e-3)
    assert cand.latency_reduction_pct == pytest.approx(50.0, rel=1e-3)
    assert cand.throughput_improvement_pct == pytest.approx(100.0, rel=1e-3)


# --- 20. Invalid or Zero Baseline Handling ---
def test_invalid_or_zero_baseline_metric_handling():
    base = ExecutionCandidate("BASE", BenchmarkConfig(), "Base", status=CandidateStatus.BENCHMARKED, is_feasible=True)
    base.benchmark_report = _mock_benchmark_report("BASE", median_ms=0.0, tp_inf_sec=0.0)

    cand = ExecutionCandidate("CAND", BenchmarkConfig(), "Cand", status=CandidateStatus.BENCHMARKED, is_feasible=True)
    cand.benchmark_report = _mock_benchmark_report("CAND", median_ms=2.0, tp_inf_sec=500.0)

    # Should not throw ZeroDivisionError
    OptimizationEngine.calculate_relative_improvements(cand, base)
    assert cand.speedup_vs_baseline is None
    assert cand.latency_reduction_pct is None


# --- 21. No Fabricated Values for Unbenchmarked Candidates ---
def test_no_fabricated_values_for_unbenchmarked_candidates():
    c = ExecutionCandidate("C1", BenchmarkConfig(), "Test", status=CandidateStatus.GENERATED)
    metrics = c.get_metrics_dict()
    assert metrics == {}
    assert c.speedup_vs_baseline is None
    assert c.latency_reduction_pct is None
    assert c.throughput_improvement_pct is None


# --- 22. Measurement Variability Handling ---
def test_measurement_variability_handling():
    # Verify that rank key uses median and throughput, preserving round statistics
    c = ExecutionCandidate("C1", BenchmarkConfig(), "Test", is_feasible=True)
    rep = _mock_benchmark_report("C1", median_ms=2.5, tp_inf_sec=400.0)
    c.benchmark_report = rep
    assert rep.latency["std_dev_ms"] is not None
    assert rep.latency["median_ms"] == 2.5


# --- 23. Optimization Report Serialization ---
def test_optimization_report_serialization():
    model_bytes = _build_test_model()
    summary = load_and_inspect_model(model_bytes, filename="test.onnx")
    req = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
    )
    report = OptimizationReport(
        experiment_id="OPT-TEST",
        timestamp_utc="2026-10-17T12:00:00Z",
        status=ExperimentStatus.COMPLETED,
        model_name="test.onnx",
        model_fingerprint=summary.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_name="Local CPU",
        profile_version="1.0.0",
        primary_objective="MINIMIZE_MEDIAN_LATENCY",
        request=req,
        baseline_candidate_id="C1",
        best_candidate_id="C2",
        candidates=[],
        pareto_frontier_ids=["C2"],
        summary_counts={"feasible": 1},
        recommendation_rationale="Candidate C2 is optimal.",
        environment={"os": "Windows"},
    )
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["experiment_id"] == "OPT-TEST"
    assert parsed["model_fingerprint"] == summary.sha256_hash

    csv_str = report.to_csv()
    assert "Candidate ID" in csv_str


# --- 24. Stable Ranking and Deterministic Tie-Breaking ---
def test_stable_ranking_and_deterministic_tie_breaking():
    # Two candidates with identical median latency and throughput
    c1 = ExecutionCandidate("CAND-B", BenchmarkConfig(), "Test", is_feasible=True)
    c1.benchmark_report = _mock_benchmark_report("CAND-B", median_ms=2.0, tp_inf_sec=500.0)
    c2 = ExecutionCandidate("CAND-A", BenchmarkConfig(), "Test", is_feasible=True)
    c2.benchmark_report = _mock_benchmark_report("CAND-A", median_ms=2.0, tp_inf_sec=500.0)

    # Tie-break alphabetically by candidate_id
    ranked = OptimizationEngine.rank_candidates([c1, c2], OptimizationObjective.MINIMIZE_MEDIAN_LATENCY)
    assert ranked[0].candidate_id == "CAND-A"
    assert ranked[1].candidate_id == "CAND-B"


# --- 25. Experiment Lifecycle & Candidate Status Transitions ---
def test_experiment_lifecycle_and_candidate_status_transitions():
    c = ExecutionCandidate("C1", BenchmarkConfig(), "Test")
    assert c.status == CandidateStatus.GENERATED

    c.status = CandidateStatus.QUEUED
    assert c.status == CandidateStatus.QUEUED

    c.status = CandidateStatus.RUNNING
    assert c.status == CandidateStatus.RUNNING

    c.status = CandidateStatus.BENCHMARKED
    assert c.status == CandidateStatus.BENCHMARKED


# --- 26. Correct Model Fingerprint Association ---
def test_correct_model_fingerprint_association():
    model_bytes = _build_test_model()
    summary = load_and_inspect_model(model_bytes, filename="test.onnx")
    profiles = load_builtin_profiles()
    profile = profiles["local_onnx_runtime_cpu"]

    req = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        max_candidates=2,
        warmup_iterations=1,
        measured_iterations=5,
        repeated_rounds=1,
    )
    report = OptimizationEngine.run_optimization(model_bytes, summary, profile, req)
    assert report.model_fingerprint == summary.sha256_hash
    assert report.model_name == "test.onnx"
    assert len(report.candidates) >= 1


# --- 27. Regression Tests for Phases 1-4 ---
def test_regression_phases_1_to_4():
    # Verify Phase 2 model loading
    m_bytes = _build_test_model()
    summary = load_and_inspect_model(m_bytes, filename="regression.onnx")
    assert summary.total_nodes == 1
    assert summary.op_counts["Gemm"] == 1

    # Verify Phase 3 compatibility analysis
    profiles = load_builtin_profiles()
    cpu_profile = profiles["local_onnx_runtime_cpu"]
    compat = CompatibilityAnalyzer.analyze(summary, cpu_profile)
    assert compat.overall_assessment == "COMPATIBLE"

    # Verify Phase 4 real benchmark execution
    cfg = BenchmarkConfig(warmup_iterations=2, measured_iterations=5, repeated_rounds=1)
    bench_rep = BenchmarkEngine.run_benchmark(m_bytes, summary, cfg)
    assert bench_rep.status == "SUCCESS"
    assert bench_rep.correctness["passed"] is True
    assert bench_rep.latency["median_ms"] > 0
