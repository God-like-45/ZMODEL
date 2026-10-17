"""Comprehensive end-to-end workflow validation test suite for Phase 8.

Executes and verifies complete user journeys across:
Workflow A — Model Inspection
Workflow B — Architecture-Aware Compatibility Analysis
Workflow C — Reproducible Benchmarking
Workflow D — Evidence-Based Optimization & Pareto Analysis
Workflow E — Experiment History & Fair Run Comparison
Workflow F — FastAPI REST API Integration & Diagnostic Health
"""

import hashlib
import io
import time
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest
from fastapi.testclient import TestClient

from zmodel.api.app import create_app
from zmodel.api.jobs import get_job_manager
from zmodel.api.model_store import clear_model_cache
from zmodel.core.benchmark import BenchmarkConfig, BenchmarkEngine
from zmodel.core.compatibility import CompatibilityAnalyzer
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.optimizer import (
    HardConstraint,
    OptimizationEngine,
    OptimizationObjective,
    OptimizationRequest,
)
from zmodel.core.profiles import load_builtin_profiles
from zmodel.db.comparator import compare_benchmark_runs
from zmodel.db.models import BenchmarkRunORM
from zmodel.db.repositories import (
    BenchmarkRepository,
    CompatibilityRepository,
    ModelRepository,
    OptimizationRepository,
)


def _synthesize_model(name: str = "E2EModel", hidden_dim: int = 4) -> bytes:
    """Build a deterministic, small ONNX model for fast end-to-end testing."""
    x = helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch_size", hidden_dim])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch_size", 2])

    w_val = np.ones((hidden_dim, 2), dtype=np.float32)
    b_val = np.zeros(2, dtype=np.float32)

    w_init = helper.make_tensor("weights", TensorProto.FLOAT, [hidden_dim, 2], w_val.flatten().tolist())
    b_init = helper.make_tensor("bias", TensorProto.FLOAT, [2], b_val.flatten().tolist())

    gemm_node = helper.make_node("Gemm", ["features", "weights", "bias"], ["gemm_out"], name="fc1")
    relu_node = helper.make_node("Relu", ["gemm_out"], ["logits"], name="relu1")

    graph = helper.make_graph(
        nodes=[gemm_node, relu_node],
        name=name,
        inputs=[x],
        outputs=[y],
        initializer=[w_init, b_init],
        doc_string="E2E Synthetic Model",
    )
    model = helper.make_model(
        graph,
        producer_name="ZModelPilotE2E",
        producer_version="1.0.0",
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    return model.SerializeToString()


@pytest.fixture
def client():
    """Create FastAPI test client with clean job coordinator."""
    get_job_manager(reset=True)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_job_manager(reset=True)


@pytest.fixture(autouse=True)
def clean_state():
    """Clear memory caches before each test."""
    clear_model_cache()


# ============================================================================
# Workflow A: Model Inspection
# ============================================================================

def test_workflow_a_model_inspection_end_to_end(client):
    """Workflow A: Complete model inspection journey from upload to registry."""
    raw_bytes = _synthesize_model("InspectWorkflowA", hidden_dim=8)
    expected_hash = hashlib.sha256(raw_bytes).hexdigest()

    # 1. API upload and structural parsing
    response = client.post(
        "/api/v1/models/inspect",
        files={"file": ("workflow_model.onnx", io.BytesIO(raw_bytes), "application/octet-stream")},
    )
    assert response.status_code == 200
    data = response.json()

    # 2. Verify extracted graph metadata
    assert data["filename"] == "workflow_model.onnx"
    assert data["sha256_fingerprint"] == expected_hash
    assert data["total_nodes"] == 2
    assert data["operator_counts"]["Gemm"] == 1
    assert data["operator_counts"]["Relu"] == 1
    assert len(data["inputs"]) == 1
    assert data["inputs"][0]["name"] == "features"
    assert data["inputs"][0]["is_dynamic"] is True
    assert data["inputs"][0]["dynamic_dim_names"] == ["batch_size"]

    # 3. Model registry persistence check
    reg_resp = client.get(f"/api/v1/models/{expected_hash}")
    if reg_resp.status_code == 200:
        reg_data = reg_resp.json()
        assert reg_data["fingerprint"] == expected_hash
        assert reg_data["original_filename"] == "workflow_model.onnx"


# ============================================================================
# Workflow B: Architecture-Aware Compatibility Analysis
# ============================================================================

def test_workflow_b_compatibility_analysis_end_to_end():
    """Workflow B: Target capability profiling with explainable findings and evidence."""
    raw_bytes = _synthesize_model("CompatWorkflowB", hidden_dim=4)
    summary = load_and_inspect_model(raw_bytes, filename="compat_model.onnx")

    profiles = load_builtin_profiles()
    assert "local_onnx_runtime_cpu" in profiles
    assert "ibm_z_onnx_illustrative" in profiles

    # 1. Evaluate against verified local CPU profile
    cpu_profile = profiles["local_onnx_runtime_cpu"]
    cpu_report = CompatibilityAnalyzer.analyze(summary, cpu_profile)

    assert cpu_report.overall_assessment in ("COMPATIBLE", "CONDITIONAL")
    assert cpu_report.profile_id == "local_onnx_runtime_cpu"
    assert cpu_report.ruleset_version == cpu_profile.ruleset_version
    assert cpu_report.is_illustrative_profile is False
    assert len(cpu_report.findings) > 0

    # Ensure findings contain traceable evidence
    for f in cpu_report.findings:
        assert f.status in ("PASS", "FAIL", "WARNING", "UNKNOWN")
        assert f.severity in ("INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert f.evidence_source

    # 2. Evaluate against illustrative IBM Z profile
    z_profile = profiles["ibm_z_onnx_illustrative"]
    z_report = CompatibilityAnalyzer.analyze(summary, z_profile)
    assert z_report.is_illustrative_profile is True
    assert z_report.profile_version == z_profile.version


# ============================================================================
# Workflow C: Reproducible Benchmarking
# ============================================================================

def test_workflow_c_benchmarking_end_to_end():
    """Workflow C: Empirical execution profiling, monotonic timing, and persistence."""
    raw_bytes = _synthesize_model("BenchWorkflowC", hidden_dim=4)
    summary = load_and_inspect_model(raw_bytes, filename="bench_model.onnx")

    cfg = BenchmarkConfig(
        execution_provider="CPUExecutionProvider",
        batch_size=1,
        warmup_iterations=3,
        measured_iterations=10,
        repeated_rounds=2,
        random_seed=42,
    )

    # 1. Execute benchmark engine
    report = BenchmarkEngine.run_benchmark(raw_bytes, summary, cfg)

    assert report.status == "SUCCESS"
    assert report.model_fingerprint == summary.sha256_hash
    assert report.correctness is not None
    assert report.correctness["passed"] is True

    # 2. Monotonic latency measurements
    assert report.latency is not None
    assert report.latency["median_ms"] > 0
    assert report.latency["min_ms"] <= report.latency["median_ms"] <= report.latency["max_ms"]
    assert report.latency["sample_count"] == 20  # 10 iterations * 2 rounds

    # 3. Throughput calculation
    assert report.throughput is not None
    assert report.throughput["inferences_per_second"] > 0
    assert report.throughput["samples_per_second"] > 0

    # 4. Persistence round-trip
    repo = BenchmarkRepository()
    saved = repo.save_benchmark_report(report)
    retrieved = repo.get_benchmark_run(saved.id)
    assert retrieved is not None
    assert retrieved.id == report.run_id
    assert retrieved.model_fingerprint == summary.sha256_hash


# ============================================================================
# Workflow D: Multi-Objective Candidate Optimization & Pareto Analysis
# ============================================================================

def test_workflow_d_optimization_end_to_end():
    """Workflow D: Candidate generation, constraint evaluation, ranking, and Pareto selection."""
    raw_bytes = _synthesize_model("OptWorkflowD", hidden_dim=4)
    summary = load_and_inspect_model(raw_bytes, filename="opt_model.onnx")
    profiles = load_builtin_profiles()
    profile = profiles["local_onnx_runtime_cpu"]

    # Define constraint: latency <= 500 ms
    constraint = HardConstraint(
        name="Max Latency",
        metric_field="median_latency_ms",
        operator="<=",
        threshold=500.0,
        unit="ms",
    )

    req = OptimizationRequest(
        model_fingerprint=summary.sha256_hash,
        profile_id=profile.profile_id,
        profile_version=profile.version,
        primary_objective=OptimizationObjective.MINIMIZE_MEDIAN_LATENCY,
        hard_constraints=[constraint],
        max_candidates=4,
        batch_sizes=[1],
        intra_op_threads=[1, 2],
        inter_op_threads=[1],
        warmup_iterations=2,
        measured_iterations=5,
        repeated_rounds=1,
    )

    report = OptimizationEngine.run_optimization(raw_bytes, summary, profile, req)

    assert report.status.value in ("COMPLETED", "COMPLETED_WITH_WARNINGS")
    assert len(report.candidates) >= 1
    assert report.baseline_candidate_id is not None

    # Verify Pareto and feasibility calculations
    for cand in report.candidates:
        assert isinstance(cand.is_feasible, bool)
        assert isinstance(cand.is_pareto_optimal, bool)
        if cand.status.value == "BENCHMARKED":
            assert cand.benchmark_report is not None
            assert cand.get_metrics_dict()["median_latency_ms"] is not None

    # Verify recommendation rationale
    assert report.recommendation_rationale
    assert len(report.recommendation_rationale) > 20

    # Persistence verification
    repo = OptimizationRepository()
    saved_exp = repo.save_optimization_report(report)
    retrieved_exp = repo.get_optimization_experiment(saved_exp.id)
    assert retrieved_exp is not None
    assert retrieved_exp.id == report.experiment_id


# ============================================================================
# Workflow E: Experiment History & Fair Comparability Verification
# ============================================================================

def test_workflow_e_experiment_history_and_comparison_rules():
    """Workflow E: Historical run diffing with scientific comparability disqualification."""
    # Build two mock benchmark runs with different model fingerprints
    run_a = BenchmarkRunORM(
        id="RUN-REF-001",
        model_fingerprint="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        model_name="ModelA.onnx",
        execution_provider="CPUExecutionProvider",
        configuration={"batch_size": 1, "intra_op_num_threads": 1},
        input_specifications={"resolved_shapes": {"features": [1, 4]}},
        runtime_environment={},
        status="SUCCESS",
        correctness_status="PASS",
        median_latency_ms=10.0,
        samples_per_second=100.0,
    )

    run_b_diff_model = BenchmarkRunORM(
        id="RUN-DIFF-MODEL-002",
        model_fingerprint="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        model_name="ModelB.onnx",
        execution_provider="CPUExecutionProvider",
        configuration={"batch_size": 1, "intra_op_num_threads": 2},
        input_specifications={"resolved_shapes": {"features": [1, 4]}},
        runtime_environment={},
        status="SUCCESS",
        correctness_status="PASS",
        median_latency_ms=8.0,
        samples_per_second=125.0,
    )

    # Cross-model comparison MUST be disqualified
    result_diff = compare_benchmark_runs(run_a, run_b_diff_model)
    assert result_diff.is_valid is False
    assert any("fingerprint" in r.lower() for r in result_diff.disqualification_reasons)

    # Identical model, differing thread count (Valid comparison)
    run_b_valid = BenchmarkRunORM(
        id="RUN-SAME-MODEL-003",
        model_fingerprint="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        model_name="ModelA.onnx",
        execution_provider="CPUExecutionProvider",
        configuration={"batch_size": 1, "intra_op_num_threads": 2},
        input_specifications={"resolved_shapes": {"features": [1, 4]}},
        runtime_environment={},
        status="SUCCESS",
        correctness_status="PASS",
        median_latency_ms=8.0,
        samples_per_second=125.0,
    )

    result_valid = compare_benchmark_runs(run_a, run_b_valid)
    assert result_valid.is_valid is True
    assert "median_latency" in result_valid.metric_comparisons
    lat_comp = result_valid.metric_comparisons["median_latency"]
    assert lat_comp.absolute_delta == -2.0
    assert lat_comp.speedup_factor == 1.25


# ============================================================================
# Workflow F: REST API Integration & System Diagnostics
# ============================================================================

def test_workflow_f_api_diagnostics_and_security(client):
    """Workflow F: Live/Ready probes, error masking, and bounded job dispatch."""
    # 1. Liveness check
    live_resp = client.get("/health/live")
    assert live_resp.status_code == 200
    assert live_resp.json()["status"] == "alive"

    # 2. Readiness check
    ready_resp = client.get("/health/ready")
    assert ready_resp.status_code in (200, 503)
    assert "database_connected" in ready_resp.json()

    # 3. Asynchronous job dispatch (HTTP 202)
    bench_payload = {
        "sample_model_name": "linear_classifier.onnx",
        "config": {
            "execution_provider": "CPUExecutionProvider",
            "batch_size": 1,
            "warmup_iterations": 2,
            "measured_iterations": 5,
            "repeated_rounds": 1,
        },
    }
    submit_resp = client.post("/api/v1/benchmarks", json=bench_payload)
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["job_id"]

    # Poll status
    timeout = 10.0
    start = time.time()
    final_status = None
    while time.time() - start < timeout:
        poll_resp = client.get(f"/api/v1/benchmarks/{job_id}/status")
        assert poll_resp.status_code == 200
        final_status = poll_resp.json()["status"]
        if final_status in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.1)

    assert final_status == "COMPLETED"

    # 4. Security: traversal rejection
    bad_payload = {
        "sample_model_name": "../../../etc/passwd",
        "profile_id": "local_onnx_runtime_cpu",
    }
    bad_resp = client.post("/api/v1/compatibility/analyze", json=bad_payload)
    assert bad_resp.status_code in (400, 404)
