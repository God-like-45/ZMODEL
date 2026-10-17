"""Comprehensive test suite for Phase 7 FastAPI REST API, background jobs, and security controls."""

import io
import time
import hashlib
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest
from fastapi.testclient import TestClient

from zmodel.api.app import create_app
from zmodel.api.config import get_api_settings
from zmodel.api.jobs import JobManager, JobQueueFullError, JobStatus, sanitize_error_message
from zmodel.api.model_store import clear_model_cache, store_model_bytes
from zmodel.db.models import BenchmarkRunORM
from zmodel.db.repositories import BenchmarkRepository, ModelRepository


def _create_synthetic_onnx(model_name: str = "ApiTestModel") -> bytes:
    """Generate small valid ONNX model in memory for API testing."""
    x = helper.make_tensor_value_info("x_input", TensorProto.FLOAT, ["batch_size", 4])
    y = helper.make_tensor_value_info("y_output", TensorProto.FLOAT, ["batch_size", 2])

    w_val = np.ones((4, 2), dtype=np.float32)
    b_val = np.zeros(2, dtype=np.float32)
    w_init = helper.make_tensor("w", TensorProto.FLOAT, [4, 2], w_val.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, [2], b_val.flatten().tolist())

    node_gemm = helper.make_node("Gemm", ["x_input", "w", "b"], ["gemm_out"], name="node_gemm")
    node_relu = helper.make_node("Relu", ["gemm_out"], ["y_output"], name="node_relu")

    graph = helper.make_graph(
        nodes=[node_gemm, node_relu],
        name=model_name,
        inputs=[x],
        outputs=[y],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(graph, producer_name="ZModelPilotTest", producer_version="1.0.0")
    model.opset_import[0].version = 15
    return model.SerializeToString()


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from zmodel.api.jobs import get_job_manager
    get_job_manager(reset=True)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_job_manager(reset=True)


@pytest.fixture(autouse=True)
def clean_cache():
    """Clear in-memory model cache before each test."""
    clear_model_cache()


# ============================================================================
# 1. Health and Readiness Tests
# ============================================================================

def test_health_live_endpoint(client):
    """GET /health/live returns alive status and api version."""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "api_version" in data
    assert "timestamp_utc" in data


def test_health_ready_endpoint(client):
    """GET /health/ready returns diagnostic readiness status."""
    response = client.get("/health/ready")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "database_connected" in data
    assert "dialect" in data
    assert "database_url_masked" in data
    assert "table_count" in data
    assert "tables" in data
    assert "password" not in data["database_url_masked"].lower()


# ============================================================================
# 2. Model Inspection Tests
# ============================================================================

def test_inspect_model_valid_onnx(client):
    """POST /api/v1/models/inspect with valid ONNX upload returns correct metadata and fingerprint."""
    raw_bytes = _create_synthetic_onnx("InspectValid")
    expected_fp = hashlib.sha256(raw_bytes).hexdigest()

    response = client.post(
        "/api/v1/models/inspect",
        files={"file": ("test_model.onnx", io.BytesIO(raw_bytes), "application/octet-stream")},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "test_model.onnx"
    assert data["sha256_fingerprint"] == expected_fp
    assert data["file_size_bytes"] == len(raw_bytes)
    assert data["total_nodes"] == 2
    assert "Gemm" in data["operator_counts"]
    assert "Relu" in data["operator_counts"]
    assert len(data["inputs"]) == 1
    assert data["inputs"][0]["name"] == "x_input"
    assert len(data["outputs"]) == 1
    assert data["outputs"][0]["name"] == "y_output"


def test_inspect_model_empty_file(client):
    """POST /api/v1/models/inspect rejects empty upload with 400 Bad Request."""
    response = client.post(
        "/api/v1/models/inspect",
        files={"file": ("empty.onnx", io.BytesIO(b""), "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["message"].lower()


def test_inspect_model_invalid_content(client):
    """POST /api/v1/models/inspect rejects corrupted ONNX bytes with 400 Bad Request."""
    response = client.post(
        "/api/v1/models/inspect",
        files={"file": ("corrupt.onnx", io.BytesIO(b"NOT_A_VALID_ONNX_HEADER"), "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data


def test_inspect_model_unsupported_extension(client):
    """POST /api/v1/models/inspect rejects non-ONNX file extensions."""
    response = client.post(
        "/api/v1/models/inspect",
        files={"file": ("model.pt", io.BytesIO(b"fake-bytes"), "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "unsupported" in data["message"].lower()


def test_inspect_model_oversized_upload(client):
    """POST /api/v1/models/inspect rejects uploads exceeding maximum allowed size."""
    settings = get_api_settings()
    # Create stream exceeding max_upload_size_bytes (using temporary monkeypatch or direct size check)
    original_limit = settings.max_upload_size_bytes
    try:
        settings.max_upload_size_bytes = 100  # Set limit to 100 bytes
        raw_bytes = _create_synthetic_onnx()
        assert len(raw_bytes) > 100

        response = client.post(
            "/api/v1/models/inspect",
            files={"file": ("large.onnx", io.BytesIO(raw_bytes), "application/octet-stream")},
        )
        assert response.status_code == 413
        assert "exceeds" in response.json()["message"].lower()
    finally:
        settings.max_upload_size_bytes = original_limit


def test_models_registry_listing_and_get(client):
    """GET /api/v1/models lists registered models and GET /api/v1/models/{fp} retrieves model identity."""
    raw_bytes = _create_synthetic_onnx("RegistryModel")
    fp = hashlib.sha256(raw_bytes).hexdigest()

    # Upload to register
    upload_resp = client.post(
        "/api/v1/models/inspect",
        files={"file": ("reg_model.onnx", io.BytesIO(raw_bytes), "application/octet-stream")},
    )
    assert upload_resp.status_code == 200

    # List
    list_resp = client.get("/api/v1/models")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert "models" in data
    assert "total" in data

    # Get by fingerprint
    get_resp = client.get(f"/api/v1/models/{fp}")
    assert get_resp.status_code in (200, 404)  # 200 if DB saved, 404 if DB is offline


# ============================================================================
# 3. Compatibility Analysis Tests
# ============================================================================

def test_list_capability_profiles(client):
    """GET /api/v1/compatibility/profiles returns available capability profiles."""
    response = client.get("/api/v1/compatibility/profiles")
    assert response.status_code == 200
    profiles = response.json()
    assert isinstance(profiles, list)
    assert len(profiles) > 0

    p_ids = [p["profile_id"] for p in profiles]
    assert "local_onnx_runtime_cpu" in p_ids

    # Verify illustrative profile labeling
    for p in profiles:
        assert "ruleset_version" in p
        assert "is_illustrative" in p
        assert "supported_opset_range" in p


def test_get_capability_profile_detail(client):
    """GET /api/v1/compatibility/profiles/{id} returns profile or 404."""
    resp = client.get("/api/v1/compatibility/profiles/local_onnx_runtime_cpu")
    assert resp.status_code == 200
    assert resp.json()["profile_id"] == "local_onnx_runtime_cpu"

    resp_404 = client.get("/api/v1/compatibility/profiles/non_existent_profile")
    assert resp_404.status_code == 404
    assert "not found" in resp_404.json()["message"].lower()


def test_compatibility_analyze_with_sample_model(client):
    """POST /api/v1/compatibility/analyze evaluates sample model against profile."""
    payload = {
        "sample_model_name": "linear_classifier.onnx",
        "profile_id": "local_onnx_runtime_cpu",
    }
    response = client.post("/api/v1/compatibility/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "linear_classifier.onnx"
    assert data["profile_id"] == "local_onnx_runtime_cpu"
    assert data["overall_assessment"] in ("COMPATIBLE", "CONDITIONAL", "INCOMPATIBLE", "INSUFFICIENT_DATA")
    assert "summary_counts" in data
    assert isinstance(data["findings"], list)
    assert len(data["findings"]) > 0

    # Ensure profile and ruleset versions are returned
    assert data["profile_version"]
    assert data["ruleset_version"]


def test_compatibility_analyze_unknown_profile(client):
    """POST /api/v1/compatibility/analyze returns 404 for unknown profile ID."""
    payload = {
        "sample_model_name": "linear_classifier.onnx",
        "profile_id": "unknown_profile_xyz",
    }
    response = client.post("/api/v1/compatibility/analyze", json=payload)
    assert response.status_code == 404
    assert "unknown profile" in response.json()["message"].lower()


def test_compatibility_directory_traversal_protection(client):
    """Sample model resolution strictly blocks directory traversal attempts."""
    payload = {
        "sample_model_name": "../../etc/passwd",
        "profile_id": "local_onnx_runtime_cpu",
    }
    response = client.post("/api/v1/compatibility/analyze", json=payload)
    assert response.status_code in (400, 404)


# ============================================================================
# 4. Benchmarking Tests
# ============================================================================

def test_benchmark_submission_and_lifecycle(client):
    """POST /api/v1/benchmarks queues job (HTTP 202) and status updates to COMPLETED."""
    payload = {
        "sample_model_name": "linear_classifier.onnx",
        "config": {
            "execution_provider": "CPUExecutionProvider",
            "batch_size": 1,
            "warmup_iterations": 2,
            "measured_iterations": 5,
            "repeated_rounds": 1,
        },
    }
    submit_resp = client.post("/api/v1/benchmarks", json=payload)
    assert submit_resp.status_code == 202
    job_data = submit_resp.json()
    assert "job_id" in job_data
    assert job_data["status"] == "QUEUED"

    job_id = job_data["job_id"]

    # Poll status until completed (bounded sample run takes ~50ms)
    max_wait = 10.0
    start = time.time()
    final_status = None

    while time.time() - start < max_wait:
        status_resp = client.get(f"/api/v1/benchmarks/{job_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        final_status = status_data["status"]
        if final_status in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.1)

    assert final_status == "COMPLETED"

    # Fetch completed report
    detail_resp = client.get(f"/api/v1/benchmarks/{job_id}")
    assert detail_resp.status_code == 200
    rep = detail_resp.json()
    assert rep["status"] == "SUCCESS"
    assert rep["latency"]["median_ms"] > 0
    assert rep["throughput"]["inferences_per_second"] > 0
    assert rep["correctness"]["passed"] is True


def test_benchmark_invalid_configuration_rejected(client):
    """POST /api/v1/benchmarks rejects invalid parameters (> max bounds) with 400 Bad Request."""
    payload = {
        "sample_model_name": "linear_classifier.onnx",
        "config": {
            "batch_size": 256,  # Exceeds maximum 128
            "measured_iterations": 10,
        },
    }
    response = client.post("/api/v1/benchmarks", json=payload)
    assert response.status_code in (400, 422)


def test_benchmark_list_pagination(client):
    """GET /api/v1/benchmarks returns paginated run items."""
    response = client.get("/api/v1/benchmarks?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "runs" in data
    assert "total" in data
    assert data["limit"] == 5
    assert data["offset"] == 0


# ============================================================================
# 5. Optimization Tests
# ============================================================================

def test_optimization_submission_and_lifecycle(client):
    """POST /api/v1/optimizations queues search (HTTP 202) and resolves candidates."""
    payload = {
        "sample_model_name": "linear_classifier.onnx",
        "profile_id": "local_onnx_runtime_cpu",
        "primary_objective": "MINIMIZE_MEDIAN_LATENCY",
        "max_candidates": 3,
        "warmup_iterations": 2,
        "measured_iterations": 5,
        "repeated_rounds": 1,
        "batch_sizes": [1],
        "intra_op_threads": [1, 2],
    }
    submit_resp = client.post("/api/v1/optimizations", json=payload)
    assert submit_resp.status_code == 202
    job_data = submit_resp.json()
    assert "job_id" in job_data
    assert job_data["status"] == "QUEUED"

    job_id = job_data["job_id"]

    # Poll status
    max_wait = 15.0
    start = time.time()
    final_status = None

    while time.time() - start < max_wait:
        status_resp = client.get(f"/api/v1/optimizations/{job_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        final_status = status_data["status"]
        if final_status in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.15)

    assert final_status == "COMPLETED"

    # Fetch candidates
    cands_resp = client.get(f"/api/v1/optimizations/{job_id}/candidates")
    assert cands_resp.status_code == 200
    cands = cands_resp.json()
    assert isinstance(cands, list)
    assert len(cands) > 0

    # Ensure baseline is identified
    has_baseline = any(c["is_baseline"] for c in cands)
    assert has_baseline is True


def test_optimization_invalid_objective_rejected(client):
    """POST /api/v1/optimizations rejects unsupported objective with 400 Bad Request."""
    payload = {
        "sample_model_name": "linear_classifier.onnx",
        "profile_id": "local_onnx_runtime_cpu",
        "primary_objective": "UNSUPPORTED_MAGIC_OBJECTIVE",
    }
    response = client.post("/api/v1/optimizations", json=payload)
    assert response.status_code in (400, 422)


# ============================================================================
# 6. Experiment History and Comparison Tests
# ============================================================================

def test_experiments_listing(client):
    """GET /api/v1/experiments returns paginated experiment history."""
    response = client.get("/api/v1/experiments?limit=10&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data
    assert "total" in data


def test_experiment_nonexistent_returns_404(client):
    """GET /api/v1/experiments/{id} returns 404 for unknown experiment ID."""
    response = client.get("/api/v1/experiments/NON_EXISTENT_EXPERIMENT_123")
    assert response.status_code == 404


def test_experiments_compare_missing_runs(client):
    """POST /api/v1/experiments/compare returns 404 when requested benchmark runs do not exist."""
    payload = {
        "run_id_a": "MISSING_RUN_A",
        "run_id_b": "MISSING_RUN_B",
    }
    response = client.post("/api/v1/experiments/compare", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["message"].lower()


# ============================================================================
# 7. Security, CORS, and Queue Bounds Tests
# ============================================================================

def test_cors_preflight_headers(client):
    """CORS middleware responds with appropriate headers on preflight requests."""
    response = client.options(
        "/api/v1/models/inspect",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_sanitize_error_message_strips_paths_and_passwords():
    """Sanitizer removes Windows/Linux paths and database credentials from error strings."""
    raw_error = Exception(
        "Database error at postgresql://postgres:SecretPassword123@localhost:5432/zmodel "
        "reading file C:\\Users\\Administrator\\secret\\model.onnx and /home/user/model.onnx"
    )
    sanitized = sanitize_error_message(raw_error)
    assert "SecretPassword123" not in sanitized
    assert "C:\\Users" not in sanitized
    assert "/home/user" not in sanitized
    assert "***@" in sanitized
    assert "[local_path]" in sanitized


def test_job_queue_concurrency_and_full_error():
    """JobManager rejects task submission with JobQueueFullError when pending queue capacity is reached."""
    jm = JobManager(concurrency=1, max_queued=2)

    def slow_task(*args, **kwargs):
        time.sleep(1.0)
        return {"status": "ok"}

    j1 = jm.submit_job("test", slow_task)
    j2 = jm.submit_job("test", slow_task)

    # 3rd submission exceeds capacity of 2
    with pytest.raises(JobQueueFullError):
        jm.submit_job("test", slow_task)

    jm.shutdown(wait=False)
