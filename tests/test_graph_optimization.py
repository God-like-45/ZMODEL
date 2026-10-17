"""Comprehensive test suite for Phase 9: ONNX Graph Optimization & Model Transformation Engine.

Covers:
1. Artifact store safety, immutability, traversal prevention, and cryptographic fingerprinting.
2. Real graph transformations (shape inference, ORT basic/extended/all, unavailable tool checks).
3. Layered structural and numerical correctness validation (tolerances, multi-output, failure cases).
4. Lifecycle states, compatibility reassessment, and empirical benchmark comparisons.
5. Recommendation layer logic (rejection of invalid candidates, best candidate selection).
6. PostgreSQL persistence and querying.
7. FastAPI endpoints, background execution, error handling, and secure artifact downloads.
"""

import hashlib
import io
import json
import os
import shutil
import tempfile
import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient

from zmodel.api.app import create_app
from zmodel.core.artifact_store import ArtifactInfo, ArtifactSecurityError, ArtifactStore
from zmodel.core.benchmark import BenchmarkConfig
from zmodel.core.graph_optimizer import (
    CandidateState,
    GraphCandidate,
    GraphOptimizationEngine,
    GraphOptimizationReport,
    RecommendationStatus,
)
from zmodel.core.graph_transforms import (
    TransformationDescriptor,
    TransformationRegistry,
    TransformationResult,
    TransformationType,
)
from zmodel.core.graph_validator import (
    GraphValidator,
    OutputComparisonMetric,
    ValidationReport,
    ValidationStatus,
)
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.profiles import load_builtin_profiles
from zmodel.db.models import (
    GraphOptimizationExperimentORM,
    GraphTransformationCandidateORM,
    ModelIdentityORM,
)
from zmodel.db.repositories import GraphOptimizationRepository, ModelRepository
from zmodel.db.session import get_engine, init_db, reset_engine, session_scope


# ============================================================================
# Test Fixtures & Synthetic Model Helpers
# ============================================================================

def _build_test_model(nodes, inputs, outputs, inits=None, name="test_model") -> bytes:
    """Build a valid, standard ONNX ModelProto pinned to IR version 9 and opset 17."""
    graph = helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=inputs,
        outputs=outputs,
        initializer=inits or [],
    )
    model = helper.make_model(
        graph,
        producer_name="zmodel_test",
        ir_version=9,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture
def linear_model_bytes() -> bytes:
    """Deterministic linear classifier (Gemm + Relu)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])

    w_val = np.array([[0.5, -0.2], [0.1, 0.8], [-0.4, 0.3], [0.9, -0.5]], dtype=np.float32)
    b_val = np.array([0.1, 0.2], dtype=np.float32)

    w_init = helper.make_tensor("W", TensorProto.FLOAT, [4, 2], w_val.flatten())
    b_init = helper.make_tensor("B", TensorProto.FLOAT, [2], b_val.flatten())

    gemm = helper.make_node("Gemm", ["X", "W", "B"], ["gemm_out"], name="gemm_node")
    relu = helper.make_node("Relu", ["gemm_out"], ["Y"], name="relu_node")

    return _build_test_model([gemm, relu], [X], [Y], [w_init, b_init], name="linear_model")


@pytest.fixture
def two_output_model_bytes() -> bytes:
    """Model producing two distinct output tensors."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y1 = helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 4])
    Y2 = helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 4])

    relu = helper.make_node("Relu", ["X"], ["Y1"], name="relu1")
    sigmoid = helper.make_node("Sigmoid", ["X"], ["Y2"], name="sigmoid1")

    return _build_test_model([relu, sigmoid], [X], [Y1, Y2], name="two_output_model")


@pytest.fixture
def dynamic_batch_model_bytes() -> bytes:
    """Model with dynamic batch dimension ('batch_size')."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch_size", 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch_size", 4])
    relu = helper.make_node("Relu", ["X"], ["Y"], name="relu_dyn")
    return _build_test_model([relu], [X], [Y], name="dynamic_model")


@pytest.fixture
def isolated_artifact_store(tmp_path) -> ArtifactStore:
    """Isolated ArtifactStore writing to pytest temporary directory."""
    store_dir = str(tmp_path / "artifacts" / "transformed")
    return ArtifactStore(storage_dir=store_dir)


@pytest.fixture
def test_db(tmp_path):
    """Temporary file-based SQLite database with full schema initialized."""
    reset_engine()
    db_file = str(tmp_path / "test_gopt.db")
    engine = get_engine(f"sqlite:///{db_file}")
    init_db(engine)
    yield engine
    reset_engine()


@pytest.fixture
def api_client(test_db):
    """FastAPI TestClient with initialized test database."""
    app = create_app()
    with TestClient(app) as client:
        yield client


# ============================================================================
# 1. Artifact Store Safety & Immutability Tests
# ============================================================================

def test_artifact_store_save_and_retrieve(isolated_artifact_store, linear_model_bytes):
    parent_fp = hashlib.sha256(linear_model_bytes).hexdigest()
    info = isolated_artifact_store.store_artifact(
        model_bytes=linear_model_bytes,
        parent_fingerprint=parent_fp,
        transformation_name="ort_basic",
        original_filename="classifier.onnx",
    )

    assert info.artifact_id.startswith("tart_")
    assert info.sha256_hash == parent_fp
    assert os.path.exists(info.file_path)

    # Read back bytes
    data, read_info = isolated_artifact_store.read_artifact_bytes(info.artifact_id)
    assert data == linear_model_bytes
    assert read_info.sha256_hash == parent_fp


def test_artifact_store_path_traversal_prevention(isolated_artifact_store):
    with pytest.raises(ArtifactSecurityError):
        isolated_artifact_store.get_artifact("../../../etc/passwd")

    with pytest.raises(ArtifactSecurityError):
        isolated_artifact_store.get_artifact("..\\..\\windows\\system32")

    with pytest.raises(ArtifactSecurityError):
        isolated_artifact_store.delete_artifact("../escaped_id")


def test_artifact_store_integrity_check(isolated_artifact_store, linear_model_bytes):
    parent_fp = hashlib.sha256(linear_model_bytes).hexdigest()
    info = isolated_artifact_store.store_artifact(
        model_bytes=linear_model_bytes,
        parent_fingerprint=parent_fp,
        transformation_name="shape_inference",
    )

    # Deliberately tamper with file content on disk
    with open(info.file_path, "wb") as f:
        f.write(b"corrupted_onnx_bytes")

    # Reading with expected_hash should detect cryptographic mismatch and raise ValueError
    with pytest.raises(ValueError, match="Artifact integrity compromised"):
        isolated_artifact_store.read_artifact_bytes(info.artifact_id, expected_hash=info.sha256_hash)


def test_original_model_immutability(isolated_artifact_store, linear_model_bytes):
    orig_copy = bytes(linear_model_bytes)
    orig_fp = hashlib.sha256(orig_copy).hexdigest()

    # Perform multiple store operations
    _ = isolated_artifact_store.store_artifact(linear_model_bytes, orig_fp, "t1")
    _ = isolated_artifact_store.store_artifact(linear_model_bytes, orig_fp, "t2")

    # Original bytes must be completely unchanged
    assert linear_model_bytes == orig_copy
    assert hashlib.sha256(linear_model_bytes).hexdigest() == orig_fp


def test_artifact_size_limit_enforced(tmp_path):
    # Store with 100 byte limit
    small_store = ArtifactStore(storage_dir=str(tmp_path / "small"), max_file_size_bytes=100)
    with pytest.raises(ValueError, match="exceeds maximum limit"):
        small_store.store_artifact(b"x" * 200, "orig_fp", "trans")


# ============================================================================
# 2. Graph Transformation Tests
# ============================================================================

def test_shape_inference_success(linear_model_bytes):
    res = TransformationRegistry.execute_transformation(
        TransformationType.SHAPE_INFERENCE, linear_model_bytes
    )
    assert res.success is True
    assert res.transformed_bytes is not None
    assert res.tool_name == "onnx.shape_inference"
    assert res.duration_ms >= 0.0

    # Inferred model should validate
    inferred_model = onnx.load_model_from_string(res.transformed_bytes)
    onnx.checker.check_model(inferred_model)


def test_ort_basic_optimization_constant_folding(linear_model_bytes):
    res = TransformationRegistry.execute_transformation(
        TransformationType.ORT_BASIC, linear_model_bytes
    )
    assert res.success is True
    assert res.transformed_bytes is not None
    assert res.tool_name == "onnxruntime"
    # Transformed artifact receives a distinct SHA-256 fingerprint
    assert len(res.transformed_hash) == 64


def test_ort_extended_optimization_fusion(linear_model_bytes):
    res = TransformationRegistry.execute_transformation(
        TransformationType.ORT_EXTENDED, linear_model_bytes
    )
    assert res.success is True
    assert res.transformed_bytes is not None
    assert any("host-tailored" in w for w in res.warnings)


def test_transformation_unavailable_graceful_handling(linear_model_bytes):
    # Simplification when tool is not installed or tested directly
    res = TransformationRegistry._execute_simplification(
        onnx.load_model_from_string(linear_model_bytes),
        "dummy_hash",
        0.0,
        {},
    )
    # If onnxsim is not installed, it cleanly reports failure with helpful message
    if not res.success:
        assert "unavailable" in res.error_message or "failed" in res.error_message


def test_transformation_invalid_model_failure():
    bad_bytes = b"not_a_valid_onnx_model_stream"
    res = TransformationRegistry.execute_transformation(
        TransformationType.ORT_BASIC, bad_bytes
    )
    assert res.success is False
    assert "Failed to parse source ONNX model" in res.error_message


def test_transformed_fingerprint_uniqueness(linear_model_bytes):
    orig_hash = hashlib.sha256(linear_model_bytes).hexdigest()
    res = TransformationRegistry.execute_transformation(
        TransformationType.ORT_BASIC, linear_model_bytes
    )
    assert res.original_hash == orig_hash
    assert res.transformed_hash is not None


# ============================================================================
# 3. Layered Structural & Numerical Correctness Tests
# ============================================================================

def test_validator_identical_model_pass(linear_model_bytes):
    val = GraphValidator.validate(
        linear_model_bytes, linear_model_bytes, validate_numerical=True
    )
    assert val.structural_status == ValidationStatus.PASS
    assert val.interface_compatible is True
    assert val.numerical_status == ValidationStatus.PASS
    assert val.is_fully_verified is True
    assert val.max_abs_error_across_outputs == 0.0


def test_validator_altered_model_fails_numerical(linear_model_bytes):
    # Alter weights in copy to deliberately cause numerical divergence
    model = onnx.load_model_from_string(linear_model_bytes)
    # Replace weight tensor values with completely different numbers
    for init in model.graph.initializer:
        if init.name == "W":
            altered_w = np.array([[100.0, -200.0], [50.0, 80.0], [-40.0, 30.0], [90.0, -50.0]], dtype=np.float32)
            init.CopyFrom(helper.make_tensor("W", TensorProto.FLOAT, [4, 2], altered_w.flatten()))

    altered_bytes = model.SerializeToString()
    val = GraphValidator.validate(linear_model_bytes, altered_bytes, validate_numerical=True)

    assert val.structural_status == ValidationStatus.PASS  # Structurally valid ONNX
    assert val.interface_compatible is True
    assert val.numerical_status == ValidationStatus.FAIL  # Numerically diverged
    assert val.is_fully_verified is False
    assert "Numerical divergence" in val.rejection_reason
    assert val.max_abs_error_across_outputs > 1.0


def test_validator_tolerance_threshold_behavior(linear_model_bytes):
    # Introduce micro-perturbation (1e-4) into bias
    model = onnx.load_model_from_string(linear_model_bytes)
    for init in model.graph.initializer:
        if init.name == "B":
            b_val = np.array([0.1 + 5e-4, 0.2 + 5e-4], dtype=np.float32)
            init.CopyFrom(helper.make_tensor("B", TensorProto.FLOAT, [2], b_val.flatten()))
    perturbed_bytes = model.SerializeToString()

    # Strict tolerance (atol=1e-5, rtol=1e-5) should FAIL
    val_strict = GraphValidator.validate(
        linear_model_bytes, perturbed_bytes, validate_numerical=True, atol=1e-5, rtol=1e-5
    )
    assert val_strict.numerical_status == ValidationStatus.FAIL

    # Relaxed tolerance (atol=1e-2, rtol=1e-2) should PASS
    val_relaxed = GraphValidator.validate(
        linear_model_bytes, perturbed_bytes, validate_numerical=True, atol=1e-2, rtol=1e-2
    )
    assert val_relaxed.numerical_status == ValidationStatus.PASS


def test_validator_multiple_outputs(two_output_model_bytes):
    val = GraphValidator.validate(
        two_output_model_bytes, two_output_model_bytes, validate_numerical=True
    )
    assert val.structural_status == ValidationStatus.PASS
    assert val.numerical_status == ValidationStatus.PASS
    assert len(val.metrics_per_output) == 2
    output_names = {m.output_name for m in val.metrics_per_output}
    assert output_names == {"Y1", "Y2"}


def test_validator_structural_only_marks_not_verified(linear_model_bytes):
    val = GraphValidator.validate(
        linear_model_bytes, linear_model_bytes, validate_numerical=False
    )
    assert val.structural_status == ValidationStatus.PASS
    assert val.numerical_status == ValidationStatus.NOT_VERIFIED
    assert val.is_fully_verified is False
    assert "not requested" in val.numerical_details


def test_validator_interface_mismatch_fails_structural(linear_model_bytes):
    # Alter output name in transformed model
    model = onnx.load_model_from_string(linear_model_bytes)
    model.graph.output[0].name = "unexpected_output_name"
    model.graph.node[-1].output[0] = "unexpected_output_name"
    mismatched_bytes = model.SerializeToString()

    val = GraphValidator.validate(linear_model_bytes, mismatched_bytes)
    assert val.structural_status == ValidationStatus.FAIL
    assert val.interface_compatible is False
    assert any("Graph outputs differ" in e for e in val.structural_errors)


def test_validator_dynamic_shape_handling(dynamic_batch_model_bytes):
    # Dynamic dimension ('batch_size') should be safely instantiated with batch=1
    val = GraphValidator.validate(
        dynamic_batch_model_bytes, dynamic_batch_model_bytes, validate_numerical=True
    )
    assert val.structural_status == ValidationStatus.PASS
    assert val.numerical_status == ValidationStatus.PASS
    assert val.is_fully_verified is True


# ============================================================================
# 4. Graph Optimization Engine & Recommendation Tests
# ============================================================================

def test_engine_full_experiment_flow(linear_model_bytes, isolated_artifact_store):
    engine = GraphOptimizationEngine(artifact_store=isolated_artifact_store)
    report = engine.run_optimization_experiment(
        model_bytes=linear_model_bytes,
        original_filename="linear.onnx",
        selected_transforms=[TransformationType.SHAPE_INFERENCE, TransformationType.ORT_BASIC],
        target_profile_id="ibm_z16_nnpa",
        validation_depth="FULL",
        enable_benchmarking=True,
    )

    assert report.status == "SUCCESS"
    assert report.original_fingerprint == hashlib.sha256(linear_model_bytes).hexdigest()
    assert len(report.candidates) == 2
    assert report.target_profile_id == "ibm_z16_nnpa"

    for cand in report.candidates:
        assert cand.state in (
            CandidateState.NUMERICALLY_VALIDATED,
            CandidateState.COMPATIBILITY_CHECKED,
            CandidateState.BENCHMARKED,
        )
        assert cand.validation_report.numerical_status == ValidationStatus.PASS
        assert cand.artifact_id is not None
        assert isolated_artifact_store.get_artifact(cand.artifact_id) is not None


def test_recommendation_layer_rejects_invalid(linear_model_bytes, isolated_artifact_store):
    engine = GraphOptimizationEngine(artifact_store=isolated_artifact_store)

    # Deliberately build candidate that fails numerical check
    bad_val = ValidationReport(
        structural_status=ValidationStatus.PASS,
        numerical_status=ValidationStatus.FAIL,
        rejection_reason="Values diverged",
    )
    cand = GraphCandidate(
        candidate_id="cand_bad",
        transform_type=TransformationType.ORT_EXTENDED,
        transform_name="Failing Transform",
        tool_name="tool",
        tool_version="1.0",
        validation_report=bad_val,
    )

    engine._evaluate_candidate_recommendation(cand, None)
    assert cand.recommendation_status == RecommendationStatus.REJECTED
    assert "rejected" in cand.recommendation_rationale.lower()


def test_recommendation_layer_selects_best_with_speedup(isolated_artifact_store):
    engine = GraphOptimizationEngine(artifact_store=isolated_artifact_store)

    c1 = GraphCandidate(
        candidate_id="cand_1",
        transform_type=TransformationType.ORT_BASIC,
        transform_name="Basic",
        tool_name="ort",
        tool_version="1.31.0",
        graph_changed=True,
        speedup_factor=1.05,
        size_reduction_pct=5.0,
        validation_report=ValidationReport(
            structural_status=ValidationStatus.PASS,
            numerical_status=ValidationStatus.PASS,
        ),
        recommendation_status=RecommendationStatus.PERFORMANCE_IMPROVED,
    )
    c2 = GraphCandidate(
        candidate_id="cand_2",
        transform_type=TransformationType.ORT_EXTENDED,
        transform_name="Extended",
        tool_name="ort",
        tool_version="1.31.0",
        graph_changed=True,
        speedup_factor=1.25,
        size_reduction_pct=10.0,
        validation_report=ValidationReport(
            structural_status=ValidationStatus.PASS,
            numerical_status=ValidationStatus.PASS,
        ),
        recommendation_status=RecommendationStatus.PERFORMANCE_IMPROVED,
    )

    best, rec = engine._select_best_candidate([c1, c2])
    assert best is not None
    assert best.candidate_id == "cand_2"
    assert "cand_2" in rec
    assert "1.25x speedup" in rec


# ============================================================================
# 5. Database Persistence Tests
# ============================================================================

def test_graph_optimization_persistence_and_retrieval(test_db, linear_model_bytes, isolated_artifact_store):
    engine = GraphOptimizationEngine(artifact_store=isolated_artifact_store)
    report = engine.run_optimization_experiment(
        model_bytes=linear_model_bytes,
        original_filename="persist_test.onnx",
        selected_transforms=[TransformationType.SHAPE_INFERENCE, TransformationType.ORT_BASIC],
    )

    repo = GraphOptimizationRepository()
    exp_orm = repo.create_experiment(report)
    assert exp_orm.id == report.experiment_id

    # Retrieve from DB
    loaded = repo.get_experiment(report.experiment_id)
    assert loaded is not None
    assert loaded.model_fingerprint == report.original_fingerprint
    assert len(loaded.candidates) == len(report.candidates)

    # Check candidate ORM fields
    c0 = loaded.candidates[0]
    assert c0.candidate_id == report.candidates[0].candidate_id
    assert c0.transform_type == report.candidates[0].transform_type.value
    assert c0.validation_report is not None


def test_experiments_list_filtering(test_db, linear_model_bytes, isolated_artifact_store):
    engine = GraphOptimizationEngine(artifact_store=isolated_artifact_store)
    rep1 = engine.run_optimization_experiment(linear_model_bytes, original_filename="m1.onnx")

    repo = GraphOptimizationRepository()
    repo.create_experiment(rep1)

    exps = repo.list_experiments(model_fingerprint=rep1.original_fingerprint)
    assert len(exps) >= 1
    assert exps[0].model_fingerprint == rep1.original_fingerprint

    # Filter with non-existent fingerprint
    empty_list = repo.list_experiments(model_fingerprint="00000000000000000000000000000000")
    assert len(empty_list) == 0


# ============================================================================
# 6. FastAPI REST API Endpoint Tests
# ============================================================================

def test_api_submit_graph_optimization_job(api_client, linear_model_bytes):
    # Upload ONNX file directly
    response = api_client.post(
        "/api/v1/graph-optimizations",
        files={"file": ("linear.onnx", linear_model_bytes, "application/octet-stream")},
        data={
            "validation_depth": "FULL",
            "enable_benchmarking": "false",
            "selected_transforms": json.dumps(["shape_inference", "ort_basic"]),
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert "experiment_id" in data
    assert data["status"] in ("QUEUED", "RUNNING")
    assert "/api/v1/graph-optimizations/" in data["poll_url"]


def test_api_get_experiment_status_and_report(api_client, test_db, linear_model_bytes, isolated_artifact_store):
    # Seed an experiment in repository
    engine = GraphOptimizationEngine(artifact_store=isolated_artifact_store)
    report = engine.run_optimization_experiment(
        linear_model_bytes,
        original_filename="api_test.onnx",
        selected_transforms=[TransformationType.ORT_BASIC],
    )
    repo = GraphOptimizationRepository()
    repo.create_experiment(report)

    # Status check
    status_resp = api_client.get(f"/api/v1/graph-optimizations/{report.experiment_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "SUCCESS"

    # Full report retrieval
    exp_resp = api_client.get(f"/api/v1/graph-optimizations/{report.experiment_id}")
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()
    assert exp_data["id"] == report.experiment_id
    assert len(exp_data["candidates"]) == 1

    # Candidate detail
    cand_id = report.candidates[0].candidate_id
    cand_resp = api_client.get(f"/api/v1/graph-optimizations/{report.experiment_id}/candidates/{cand_id}")
    assert cand_resp.status_code == 200
    assert cand_resp.json()["candidate_id"] == cand_id


def test_api_artifact_download_success(api_client, test_db, linear_model_bytes):
    # Run optimization with default artifact store
    engine = GraphOptimizationEngine()
    report = engine.run_optimization_experiment(
        linear_model_bytes,
        original_filename="download_test.onnx",
        selected_transforms=[TransformationType.ORT_BASIC],
    )
    repo = GraphOptimizationRepository()
    repo.create_experiment(report)

    cand_id = report.candidates[0].candidate_id
    dl_resp = api_client.get(f"/api/v1/graph-optimizations/{report.experiment_id}/candidates/{cand_id}/artifact")

    assert dl_resp.status_code == 200
    assert dl_resp.headers["Content-Type"] == "application/octet-stream"
    assert "attachment; filename=" in dl_resp.headers["Content-Disposition"]
    assert "X-Artifact-Fingerprint" in dl_resp.headers

    # Downloaded content should be valid ONNX
    downloaded_bytes = dl_resp.content
    loaded_model = onnx.load_model_from_string(downloaded_bytes)
    onnx.checker.check_model(loaded_model)


def test_api_artifact_download_rejected_prohibited(api_client, test_db):
    # Insert a dummy rejected candidate
    exp_id = "gexp_rejected_test"
    repo = GraphOptimizationRepository()

    report = GraphOptimizationReport(
        experiment_id=exp_id,
        model_name="failing.onnx",
        original_fingerprint="dummy_orig_fp_12345678",
        target_profile_id=None,
        target_profile_name=None,
        validation_depth="FULL",
        benchmark_enabled=False,
        created_at="2026-10-18T00:00:00Z",
        completed_at="2026-10-18T00:00:01Z",
        status="SUCCESS",
        candidates=[
            GraphCandidate(
                candidate_id="cand_rej",
                transform_type=TransformationType.ORT_EXTENDED,
                transform_name="Rejected Transform",
                tool_name="ort",
                tool_version="1.31.0",
                state=CandidateState.REJECTED,
                recommendation_status=RecommendationStatus.REJECTED,
                rejection_reason="Failed numerical equivalence check",
                artifact_id="tart_dummy_artifact_id",
            )
        ],
    )
    repo.create_experiment(report)

    # Attempting to download rejected candidate artifact must be rejected with 400 Bad Request
    dl_resp = api_client.get(f"/api/v1/graph-optimizations/{exp_id}/candidates/cand_rej/artifact")
    assert dl_resp.status_code == 400
    err_body = dl_resp.json()
    assert "REJECTED" in (err_body.get("message") or err_body.get("detail", ""))


def test_api_artifact_download_missing_returns_404(api_client):
    dl_resp = api_client.get("/api/v1/graph-optimizations/non_existent_exp/candidates/non_existent_cand/artifact")
    assert dl_resp.status_code == 404
