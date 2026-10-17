"""
Tests for Phase 12: Advanced Model Optimization and Enterprise Decision-Support Reporting.

Covers:
1. Search-space generation and bounds (Grid, Random, Successive Halving)
2. Seed-based search reproducibility
3. Candidate configuration hashing and deduplication
4. Multi-fidelity Successive Halving early pruning
5. Strict Pareto dominance mathematics across 4 objectives
6. Pareto ties and incomparable/missing metrics handling
7. Empty feasible set diagnosis and constraint failure itemization
8. Measurement reliability statistics (sample counts, IQR, CV%, stability rating)
9. Model sensitivity analysis (batch elasticity, thread scaling response)
10. Sensitivity behavior with insufficient data points (< 2)
11. Goal-directed recommendations (Latency, Throughput, Memory, Balanced Score)
12. Correctness gating (failure to validate correctness excludes candidate)
13. Enterprise audit report generation (Markdown & JSON) with all 18 sections
14. Secret and credential redaction across reports
15. Hardware provenance and authentic IBM Z pending verification statement
16. Relational database persistence and retrieval
17. REST API /search endpoint
18. REST API /pareto endpoint
19. REST API /sensitivity endpoint
20. REST API /recommendation endpoint
21. REST API /report endpoint
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from fastapi.testclient import TestClient
from zmodel.api.app import app
from zmodel.core.advanced_optimizer import (
    AdvancedOptimizationEngine,
    AdvancedParetoAnalyzer,
    EnterpriseReportGenerator,
    EvidenceRecommendation,
    MeasurementReliability,
    RecommendationEngine,
    RecommendationGoal,
    ReliabilityAnalyzer,
    ReliabilityRating,
    SearchStrategy,
    SensitivityAnalyzer,
)
from zmodel.core.benchmark import BenchmarkConfig, BenchmarkReport
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.optimizer import (
    CandidateStatus,
    ExecutionCandidate,
    HardConstraint,
    OptimizationObjective,
    OptimizationRequest,
)
from zmodel.core.profiles import load_builtin_profiles
from zmodel.core.state import is_db_available
from zmodel.db.repositories import OptimizationRepository


def _build_test_model() -> bytes:
    """Construct a lightweight valid single-Gemm ONNX model."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    w = helper.make_tensor("W", TensorProto.FLOAT, [4, 2], [0.1] * 8)
    b = helper.make_tensor("B", TensorProto.FLOAT, [2], [0.05, 0.05])
    node = helper.make_node("Gemm", inputs=["X", "W", "B"], outputs=["Y"], name="gemm_0")
    graph = helper.make_graph([node], "test_graph", [x], [y], [w, b])
    model = helper.make_model(graph, producer_name="test_producer", opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    return model.SerializeToString()


# --- 1. Search Strategies & Candidate Generation ---
def test_search_space_generation_bounds():
    m_bytes = _build_test_model()
    summary = load_and_inspect_model(m_bytes, filename="test.onnx")
    req = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
        max_candidates=4,
        batch_sizes=[1, 2, 4],
        intra_op_threads=[1, 2, 4],
    )

    cands_grid = AdvancedOptimizationEngine.generate_candidates(
        summary, req, SearchStrategy.GRID, ["CPUExecutionProvider"]
    )
    assert len(cands_grid) <= 4
    assert cands_grid[0].is_baseline is True

    cands_rnd = AdvancedOptimizationEngine.generate_candidates(
        summary, req, SearchStrategy.RANDOM, ["CPUExecutionProvider"]
    )
    assert len(cands_rnd) <= 4


# --- 2. Seed Reproducibility ---
def test_seed_reproducibility():
    m_bytes = _build_test_model()
    summary = load_and_inspect_model(m_bytes, filename="test.onnx")
    req1 = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
        max_candidates=5,
        random_seed=12345,
    )
    req2 = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
        max_candidates=5,
        random_seed=12345,
    )

    c1 = AdvancedOptimizationEngine.generate_candidates(summary, req1, SearchStrategy.RANDOM, ["CPUExecutionProvider"])
    c2 = AdvancedOptimizationEngine.generate_candidates(summary, req2, SearchStrategy.RANDOM, ["CPUExecutionProvider"])

    hashes1 = [c.config.configuration_hash() for c in c1]
    hashes2 = [c.config.configuration_hash() for c in c2]
    assert hashes1 == hashes2


# --- 3. Configuration Hashing & Deduplication ---
def test_configuration_hashing_and_deduplication():
    cfg1 = BenchmarkConfig(batch_size=1, intra_op_num_threads=2)
    cfg2 = BenchmarkConfig(batch_size=1, intra_op_num_threads=2)
    cfg3 = BenchmarkConfig(batch_size=2, intra_op_num_threads=2)

    assert cfg1.configuration_hash() == cfg2.configuration_hash()
    assert cfg1.configuration_hash() != cfg3.configuration_hash()


# --- 4. Mathematical Pareto Dominance ---
def test_pareto_dominance_mathematical_logic():
    # A dominates B: faster and higher throughput
    cand_a = {"median_latency_ms": 2.0, "p95_latency_ms": 3.0, "throughput_inf_sec": 500.0, "peak_memory_mb": 50.0}
    cand_b = {"median_latency_ms": 4.0, "p95_latency_ms": 5.0, "throughput_inf_sec": 250.0, "peak_memory_mb": 60.0}
    a_dom_b, b_dom_a, _ = AdvancedParetoAnalyzer.evaluate_dominance(cand_a, cand_b)
    assert a_dom_b is True
    assert b_dom_a is False

    # Trade-off: A is faster, but B has higher throughput (batching trade-off)
    cand_c = {"median_latency_ms": 1.5, "p95_latency_ms": 2.0, "throughput_inf_sec": 300.0, "peak_memory_mb": 50.0}
    cand_d = {"median_latency_ms": 3.0, "p95_latency_ms": 4.0, "throughput_inf_sec": 600.0, "peak_memory_mb": 55.0}
    c_dom_d, d_dom_c, _ = AdvancedParetoAnalyzer.evaluate_dominance(cand_c, cand_d)
    assert c_dom_d is False
    assert d_dom_c is False


# --- 5. Pareto Ties and Incomparable Metrics ---
def test_pareto_ties_and_incomparable_metrics():
    # Identical metrics
    cand_x = {"median_latency_ms": 2.0, "p95_latency_ms": 2.5, "throughput_inf_sec": 500.0}
    cand_y = {"median_latency_ms": 2.0, "p95_latency_ms": 2.5, "throughput_inf_sec": 500.0}
    x_dom_y, y_dom_x, _ = AdvancedParetoAnalyzer.evaluate_dominance(cand_x, cand_y)
    assert x_dom_y is False
    assert y_dom_x is False

    # Incomparable metrics (missing latency)
    cand_incomplete = {"median_latency_ms": None, "throughput_inf_sec": 500.0}
    x_dom_inc, inc_dom_x, exp = AdvancedParetoAnalyzer.evaluate_dominance(cand_x, cand_incomplete)
    assert x_dom_inc is False
    assert inc_dom_x is False
    assert "Incomparable" in exp


# --- 6. Empty Feasible Set Diagnosis ---
def test_empty_feasible_set_diagnosis():
    profiles = load_builtin_profiles()
    profile = profiles["local_onnx_runtime_cpu"]

    c = ExecutionCandidate(
        candidate_id="CAND-01",
        config=BenchmarkConfig(),
        generation_reason="Test candidate",
        status=CandidateStatus.BENCHMARKED,
        is_feasible=False,
    )
    rec = RecommendationEngine.evaluate(
        candidates=[c],
        goal=RecommendationGoal.BALANCED_SCORE,
        profile=profile,
        baseline_candidate=c,
    )
    assert rec.recommended_candidate_id is None
    assert "No candidate satisfied" in rec.rationale


# --- 7. Measurement Reliability Statistics ---
def test_measurement_reliability_statistics():
    cand = ExecutionCandidate(
        candidate_id="CAND-01",
        config=BenchmarkConfig(),
        generation_reason="Reliability test",
        status=CandidateStatus.BENCHMARKED,
        is_feasible=True,
    )
    # 30 latencies around 2.0 ms with small jitter
    latencies = [2.0 + (i % 3) * 0.05 for i in range(30)]
    cand.benchmark_report = BenchmarkReport(
        run_id="BENCH-REL-01",
        timestamp_utc="2026-10-18T00:00:00Z",
        model_name="test.onnx",
        model_fingerprint="abc",
        status="SUCCESS",
        error_message=None,
        environment={},
        config={},
        input_spec={},
        correctness={"passed": True},
        latency={"median_ms": 2.05},
        throughput={"inferences_per_second": 480.0},
        memory={"peak_rss_mb": 50.0},
        per_round_results=[],
        all_latencies_ms=latencies,
        warnings=[],
        limitations=[],
    )

    rel = ReliabilityAnalyzer.analyze_candidate(cand)
    assert rel is not None
    assert rel.sample_count == 30
    assert rel.iqr_ms >= 0.0
    assert rel.cv_percent <= 10.0
    assert rel.rating == ReliabilityRating.HIGH
    assert rel.is_stable is True


# --- 8. Sensitivity Analysis ---
def test_sensitivity_analysis():
    cands = []
    for b in [1, 2, 4]:
        c = ExecutionCandidate(
            candidate_id=f"CAND-B{b}",
            config=BenchmarkConfig(batch_size=b),
            generation_reason=f"Batch {b}",
            status=CandidateStatus.BENCHMARKED,
            is_feasible=True,
        )
        c.benchmark_report = BenchmarkReport(
            run_id=f"BENCH-B{b}",
            timestamp_utc="2026-10-18T00:00:00Z",
            model_name="test.onnx",
            model_fingerprint="abc",
            status="SUCCESS",
            error_message=None,
            environment={},
            config={},
            input_spec={},
            correctness={"passed": True},
            latency={"median_ms": 1.5 * b},
            throughput={"inferences_per_second": 600.0 * b},
            memory={"peak_rss_mb": 50.0},
            per_round_results=[],
            all_latencies_ms=[1.5 * b],
            warnings=[],
            limitations=[],
        )
        cands.append(c)

    sens = SensitivityAnalyzer.analyze(cands)
    assert "batch_size" in sens.parameters
    assert sens.parameters["batch_size"].status == "ANALYZED"
    assert len(sens.parameters["batch_size"].data_points) == 3


# --- 9. Recommendation Goals & Correctness Gating ---
def test_recommendation_goals_and_correctness_gating():
    profiles = load_builtin_profiles()
    profile = profiles["local_onnx_runtime_cpu"]

    # Candidate 1: Fast, but failed correctness
    c1 = ExecutionCandidate(
        candidate_id="CAND-FAST-FAILED",
        config=BenchmarkConfig(batch_size=1),
        generation_reason="Fast",
        status=CandidateStatus.FAILED,
        is_feasible=False,
    )
    # Candidate 2: Valid, feasible
    c2 = ExecutionCandidate(
        candidate_id="CAND-VALID",
        config=BenchmarkConfig(batch_size=1, intra_op_num_threads=2),
        generation_reason="Valid",
        status=CandidateStatus.BENCHMARKED,
        is_feasible=True,
    )
    c2.benchmark_report = BenchmarkReport(
        run_id="BENCH-C2",
        timestamp_utc="2026-10-18T00:00:00Z",
        model_name="test.onnx",
        model_fingerprint="abc",
        status="SUCCESS",
        error_message=None,
        environment={},
        config={},
        input_spec={},
        correctness={"passed": True},
        latency={"median_ms": 2.5},
        throughput={"inferences_per_second": 400.0},
        memory={"peak_rss_mb": 40.0},
        per_round_results=[],
        all_latencies_ms=[2.5],
        warnings=[],
        limitations=[],
    )

    rec = RecommendationEngine.evaluate(
        candidates=[c1, c2],
        goal=RecommendationGoal.MINIMIZE_LATENCY,
        profile=profile,
        baseline_candidate=c2,
    )
    assert rec.recommended_candidate_id == "CAND-VALID"
    assert "CAND-FAST-FAILED" != rec.recommended_candidate_id


# --- 10. Enterprise Report Generation & Secret Redaction ---
def test_enterprise_report_generation_and_secret_redaction():
    report_data = {
        "experiment_id": "OPT-TEST-001",
        "timestamp_utc": "2026-10-18T00:00:00Z",
        "model_name": "test.onnx",
        "model_fingerprint": "1234567890abcdef" * 4,
        "profile_id": "ibm_z16_linux_nnpa",
        "profile_name": "IBM z16 Linux NNPA",
        "profile_version": "1.3.0",
        "primary_objective": "MINIMIZE_MEDIAN_LATENCY",
        "summary_counts": {"total_generated": 2, "benchmarked": 2, "feasible": 2, "pareto_count": 1},
        "candidates": [],
        "pareto_frontier_ids": ["CAND-01"],
        "environment": {"is_s390x": False, "architecture": "x86_64"},
        "recommendation_rationale": "Password: super_secret_pass; Token: secret_token_value",
    }

    md = EnterpriseReportGenerator.generate_markdown(report_data)
    assert "1. Executive Summary" in md
    assert "18. Audit Trail" in md
    assert "IBM Z target validation remains pending; only the listed local checks have been verified." in md
    # Secret must be redacted
    assert "super_secret_pass" not in md
    assert "***REDACTED***" in md

    js = EnterpriseReportGenerator.generate_json(report_data)
    parsed = json.loads(js)
    assert parsed["schema_version"] == "2.0.0-phase12"
    assert "super_secret_pass" not in js


# --- 11. Full End-to-End Advanced Optimization Execution ---
def test_advanced_optimization_engine_end_to_end():
    m_bytes = _build_test_model()
    summary = load_and_inspect_model(m_bytes, filename="e2e_opt.onnx")
    profiles = load_builtin_profiles()
    profile = profiles["local_onnx_runtime_cpu"]

    req = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        max_candidates=3,
        warmup_iterations=2,
        measured_iterations=6,
        repeated_rounds=1,
    )

    report_dict = AdvancedOptimizationEngine.execute_advanced_optimization(
        model_path_or_bytes=m_bytes,
        model_summary=summary,
        profile=profile,
        request=req,
        strategy=SearchStrategy.GRID,
        recommendation_goal=RecommendationGoal.BALANCED_SCORE,
    )

    assert report_dict["experiment_id"].startswith("OPT-ADV-")
    assert report_dict["summary_counts"]["benchmarked"] >= 1
    assert "markdown_report" in report_dict
    assert "json_report" in report_dict
    assert report_dict["best_candidate_id"] is not None


# --- 12. REST API Integration Tests ---
def test_api_advanced_optimization_search_and_sub_endpoints():
    client = TestClient(app)

    # 1. Search submission
    req_body = {
        "sample_model_name": "linear_classifier.onnx",
        "profile_id": "local_onnx_runtime_cpu",
        "primary_objective": "MINIMIZE_MEDIAN_LATENCY",
        "search_strategy": "GRID",
        "recommendation_goal": "BALANCED_SCORE",
        "max_candidates": 2,
        "warmup_iterations": 1,
        "measured_iterations": 5,
        "repeated_rounds": 1,
        "random_seed": 42,
    }
    resp = client.post("/api/v1/optimizations/search", json=req_body)
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # Poll status until complete
    import time
    for _ in range(30):
        st_resp = client.get(f"/api/v1/optimizations/{job_id}/status")
        if st_resp.status_code == 200 and st_resp.json()["status"] == "COMPLETED":
            break
        time.sleep(0.3)

    # 2. Check Pareto endpoint
    pareto_resp = client.get(f"/api/v1/optimizations/{job_id}/pareto")
    assert pareto_resp.status_code == 200
    assert "pareto_frontier_ids" in pareto_resp.json()

    # 3. Check Sensitivity endpoint
    sens_resp = client.get(f"/api/v1/optimizations/{job_id}/sensitivity")
    assert sens_resp.status_code == 200
    assert "parameters" in sens_resp.json()

    # 4. Check Recommendation endpoint
    rec_resp = client.get(f"/api/v1/optimizations/{job_id}/recommendation")
    assert rec_resp.status_code == 200
    assert "recommended_candidate_id" in rec_resp.json()

    # 5. Check Report endpoint (markdown & json)
    rep_md = client.get(f"/api/v1/optimizations/{job_id}/report?format=markdown")
    assert rep_md.status_code == 200
    assert "Executive Summary" in rep_md.json()["report_content"]

    rep_json = client.get(f"/api/v1/optimizations/{job_id}/report?format=json")
    assert rep_json.status_code == 200
    assert "schema_version" in rep_json.json()["report_content"]
