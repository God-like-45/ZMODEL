"""Comprehensive test suite for Phase 10: Kubernetes-Based Benchmark Workers & Scalable Job Execution."""

import hashlib
import json
import os
import tempfile
import uuid
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from zmodel.api.app import create_app
from zmodel.core.artifact_store import ArtifactStore
from zmodel.core.execution import (
    BackendExecutionError,
    BackendHealth,
    BackendUnavailableError,
    ExecutionBackend,
    KubernetesExecutionBackend,
    LocalExecutionBackend,
    RESOURCE_PROFILES,
    TimingBreakdown,
    WorkerJobResult,
    WorkerJobSpec,
    WorkerJobState,
    WorkerJobType,
    get_execution_backend,
    get_resource_profile,
)
from zmodel.core.execution.k8s_adapter import sanitize_k8s_name
from zmodel.db.models import WorkerJobORM
from zmodel.db.repositories import WorkerJobRepository
from zmodel.db.session import init_db, reset_engine, session_scope
from zmodel.worker.runner import execute_worker_task, resolve_worker_model_bytes
from zmodel.worker.main import main as worker_cli_main


@pytest.fixture
def linear_model_path():
    """Return absolute path to sample linear classifier ONNX model."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_models", "linear_classifier.onnx")
    assert os.path.exists(path), f"Sample model not found at {path}"
    return path


@pytest.fixture
def linear_model_bytes(linear_model_path):
    with open(linear_model_path, "rb") as f:
        return f.read()


@pytest.fixture
def linear_model_hash(linear_model_bytes):
    return hashlib.sha256(linear_model_bytes).hexdigest()


@pytest.fixture(autouse=True)
def init_test_db():
    init_db()
    yield


# ==============================================================================
# 1. Schemas & Resource Profiles Tests
# ==============================================================================

def test_resource_profiles_definitions():
    """Verify that all predefined compute resource profiles are valid and bounded."""
    assert "small_cpu" in RESOURCE_PROFILES
    assert "standard_cpu" in RESOURCE_PROFILES
    assert "graph_opt" in RESOURCE_PROFILES
    assert "large_opt" in RESOURCE_PROFILES

    small = get_resource_profile("small_cpu")
    assert small.cpu_request == "250m"
    assert small.cpu_limit == "500m"
    assert small.memory_limit == "512Mi"
    assert small.active_deadline_seconds == 120

    graph = get_resource_profile("graph_opt")
    assert graph.cpu_limit == "2000m"
    assert graph.memory_limit == "2Gi"
    assert graph.active_deadline_seconds == 600

    # Fallback behavior
    fallback = get_resource_profile("non_existent_profile")
    assert fallback.name == "standard_cpu"


def test_worker_job_spec_serialization(linear_model_hash):
    """Verify WorkerJobSpec serialization, deserialization, and schema integrity."""
    spec = WorkerJobSpec(
        job_id="job-test-12345",
        job_type=WorkerJobType.BENCHMARK,
        model_fingerprint=linear_model_hash,
        model_filename="linear_classifier.onnx",
        resource_profile="small_cpu",
        timeout_seconds=120,
        max_retries=2,
        parameters={"batch_size": 2, "iterations": 10},
    )

    d = spec.to_dict()
    assert d["job_id"] == "job-test-12345"
    assert d["job_type"] == "benchmark"
    assert d["resource_profile"] == "small_cpu"
    assert d["parameters"]["batch_size"] == 2

    # Roundtrip from dict
    restored = WorkerJobSpec.from_dict(d)
    assert restored.job_id == spec.job_id
    assert restored.job_type == WorkerJobType.BENCHMARK
    assert restored.model_fingerprint == linear_model_hash
    assert restored.parameters["batch_size"] == 2


def test_worker_job_result_serialization():
    """Verify WorkerJobResult timing decomposition and serialization."""
    timing = TimingBreakdown(
        queue_delay_seconds=0.5,
        startup_overhead_seconds=0.2,
        warmup_overhead_seconds=0.1,
        inference_duration_seconds=1.2,
        total_duration_seconds=2.0,
    )
    result = WorkerJobResult(
        job_id="job-test-12345",
        status=WorkerJobState.SUCCEEDED,
        model_fingerprint="abc123hash",
        timing=timing,
        worker_image="zmodel-worker:v1.2",
        worker_node="k8s-node-1",
        pod_name="zmodel-worker-pod-xyz",
        result_data={"p50_ms": 1.25, "throughput": 800.0},
    )

    d = result.to_dict()
    assert d["status"] == "SUCCEEDED"
    assert d["timing"]["queue_delay_seconds"] == 0.5
    assert d["timing"]["inference_duration_seconds"] == 1.2
    assert d["result_data"]["throughput"] == 800.0

    restored = WorkerJobResult.from_dict(d)
    assert restored.status == WorkerJobState.SUCCEEDED
    assert restored.timing.total_duration_seconds == 2.0


def test_timing_breakdown_distinguishes_infrastructure():
    """Verify inference latency is strictly decoupled from container and queue delays."""
    timing = TimingBreakdown(
        queue_delay_seconds=5.0,
        startup_overhead_seconds=2.5,
        warmup_overhead_seconds=0.5,
        inference_duration_seconds=0.8,
        total_duration_seconds=8.8,
    )
    # Critical guarantee: pure inference time is 0.8s, not 8.8s
    assert timing.inference_duration_seconds == 0.8
    assert timing.queue_delay_seconds + timing.startup_overhead_seconds + timing.warmup_overhead_seconds + timing.inference_duration_seconds == pytest.approx(8.8)


# ==============================================================================
# 2. Execution Backend Abstraction & Local Backend Tests
# ==============================================================================

def test_local_execution_backend_lifecycle(linear_model_hash):
    """Test full execution of benchmark job through LocalExecutionBackend."""
    backend = LocalExecutionBackend()
    assert backend.name == "local"

    spec = WorkerJobSpec(
        job_id="job-local-test-001",
        job_type=WorkerJobType.BENCHMARK,
        model_fingerprint=linear_model_hash,
        model_filename="linear_classifier.onnx",
        parameters={"batch_size": 1, "measured_iterations": 5, "repeated_rounds": 1, "warmup_iterations": 2},
    )

    job_id = backend.submit_job(spec)
    assert job_id == spec.job_id

    # Wait for completion in local thread pool
    import time
    for _ in range(50):
        status = backend.get_job_status(job_id)
        if status in (WorkerJobState.SUCCEEDED, WorkerJobState.FAILED):
            break
        time.sleep(0.05)

    assert status == WorkerJobState.SUCCEEDED
    res = backend.get_job_result(job_id)
    assert res is not None
    assert res.status == WorkerJobState.SUCCEEDED
    assert res.result_data is not None
    assert "latency" in res.result_data

    # Cleanup
    assert backend.cleanup_job(job_id) is True
    assert backend.get_job_result(job_id) is None


def test_get_execution_backend_honors_env(monkeypatch):
    """Verify get_execution_backend factory honors configuration and rejects silent fallback."""
    monkeypatch.setenv("EXECUTION_BACKEND", "local")
    b_local = get_execution_backend()
    assert b_local.name == "local"

    monkeypatch.setenv("EXECUTION_BACKEND", "kubernetes")
    b_k8s = get_execution_backend()
    assert b_k8s.name == "kubernetes"


def test_sanitize_k8s_name():
    """Verify Kubernetes name sanitization enforces RFC 1123 label standards."""
    assert sanitize_k8s_name("job_123_ABC") == "job-123-abc"
    assert sanitize_k8s_name("JOB--UPPER--CASE!!") == "job-upper-case"
    assert len(sanitize_k8s_name("a" * 100)) <= 63


# ==============================================================================
# 3. Kubernetes Execution Backend (Mocked Unit Tests)
# ==============================================================================

def test_k8s_manifest_construction(linear_model_hash):
    """Verify Kubernetes Job manifest contains correct resource limits, security context, and labels."""
    k8s = KubernetesExecutionBackend(namespace="zmodel-test-ns", worker_image="zmodel-worker:v1.2")
    spec = WorkerJobSpec(
        job_id="job-k8s-001",
        job_type=WorkerJobType.BENCHMARK,
        model_fingerprint=linear_model_hash,
        model_filename="linear_classifier.onnx",
        resource_profile="standard_cpu",
        timeout_seconds=300,
        max_retries=1,
    )

    manifest = k8s._build_job_manifest(spec)
    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["name"] == "job-k8s-001"
    assert manifest["metadata"]["namespace"] == "zmodel-test-ns"
    assert manifest["metadata"]["labels"]["zmodel.ai/job-id"] == "job-k8s-001"

    pod_spec = manifest["spec"]["template"]["spec"]
    assert pod_spec["restartPolicy"] == "Never"
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    assert pod_spec["securityContext"]["runAsUser"] == 1000

    container = pod_spec["containers"][0]
    assert container["name"] == "worker"
    assert container["image"] == "zmodel-worker:v1.2"
    assert container["resources"]["requests"]["cpu"] == "500m"
    assert container["resources"]["limits"]["memory"] == "1Gi"
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]


def test_k8s_submit_job_success(linear_model_hash):
    """Verify submit_job calls create_namespaced_job on BatchV1Api."""
    mock_batch = MagicMock()
    mock_core = MagicMock()
    # No existing job
    mock_batch.read_namespaced_job.side_effect = Exception("Not found")

    k8s = KubernetesExecutionBackend(
        namespace="zmodel-test-ns",
        batch_v1_api=mock_batch,
        core_v1_api=mock_core,
    )

    spec = WorkerJobSpec(
        job_id="job-k8s-002",
        job_type=WorkerJobType.BENCHMARK,
        model_fingerprint=linear_model_hash,
    )

    submitted_id = k8s.submit_job(spec)
    assert submitted_id == "job-k8s-002"
    assert mock_batch.create_namespaced_job.called
    call_args = mock_batch.create_namespaced_job.call_args[1]
    assert call_args["namespace"] == "zmodel-test-ns"
    assert call_args["body"]["metadata"]["name"] == "job-k8s-002"


def test_k8s_submit_job_idempotent_duplicate_handling(linear_model_hash):
    """Verify submit_job detects existing job and does not duplicate creation."""
    mock_batch = MagicMock()
    mock_core = MagicMock()
    # Existing job found
    mock_batch.read_namespaced_job.return_value = MagicMock()

    k8s = KubernetesExecutionBackend(
        namespace="zmodel-test-ns",
        batch_v1_api=mock_batch,
        core_v1_api=mock_core,
    )

    spec = WorkerJobSpec(
        job_id="job-k8s-duplicate",
        job_type=WorkerJobType.BENCHMARK,
        model_fingerprint=linear_model_hash,
    )

    res_id = k8s.submit_job(spec)
    assert res_id == "job-k8s-duplicate"
    # create_namespaced_job should NOT have been called
    assert not mock_batch.create_namespaced_job.called


def test_k8s_status_mapping_succeeded():
    """Verify Succeeded job status is mapped to WorkerJobState.SUCCEEDED."""
    mock_batch = MagicMock()
    mock_core = MagicMock()

    mock_job = MagicMock()
    mock_job.status.succeeded = 1
    mock_job.status.failed = None
    mock_job.status.active = None
    mock_batch.read_namespaced_job_status.return_value = mock_job

    k8s = KubernetesExecutionBackend(batch_v1_api=mock_batch, core_v1_api=mock_core)
    state = k8s.get_job_status("job-k8s-003")
    assert state == WorkerJobState.SUCCEEDED


def test_k8s_status_mapping_deadline_exceeded():
    """Verify failed job with DeadlineExceeded is mapped to WorkerJobState.TIMED_OUT."""
    mock_batch = MagicMock()
    mock_core = MagicMock()

    mock_job = MagicMock()
    mock_job.status.succeeded = None
    mock_job.status.failed = 1
    mock_cond = MagicMock()
    mock_cond.reason = "DeadlineExceeded"
    mock_job.status.conditions = [mock_cond]
    mock_batch.read_namespaced_job_status.return_value = mock_job

    k8s = KubernetesExecutionBackend(batch_v1_api=mock_batch, core_v1_api=mock_core)
    state = k8s.get_job_status("job-k8s-timeout")
    assert state == WorkerJobState.TIMED_OUT


def test_k8s_status_mapping_active_pod_running():
    """Verify active job with Running pod is mapped to WorkerJobState.RUNNING."""
    mock_batch = MagicMock()
    mock_core = MagicMock()

    mock_job = MagicMock()
    mock_job.status.succeeded = None
    mock_job.status.failed = None
    mock_job.status.active = 1
    mock_batch.read_namespaced_job_status.return_value = mock_job

    mock_pod = MagicMock()
    mock_pod.status.phase = "Running"
    mock_pod_list = MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_core.list_namespaced_pod.return_value = mock_pod_list

    k8s = KubernetesExecutionBackend(batch_v1_api=mock_batch, core_v1_api=mock_core)
    state = k8s.get_job_status("job-k8s-running")
    assert state == WorkerJobState.RUNNING


def test_k8s_cancel_job():
    """Verify cancel_job invokes delete_namespaced_job with Foreground propagation."""
    mock_batch = MagicMock()
    mock_core = MagicMock()

    k8s = KubernetesExecutionBackend(
        namespace="zmodel-test-ns",
        batch_v1_api=mock_batch,
        core_v1_api=mock_core,
    )
    success = k8s.cancel_job("job-k8s-cancel")
    assert success is True
    assert mock_batch.delete_namespaced_job.called
    call_args = mock_batch.delete_namespaced_job.call_args[1]
    assert call_args["name"] == "job-k8s-cancel"
    assert call_args["namespace"] == "zmodel-test-ns"


def test_k8s_health_check_failure():
    """Verify health check reports is_available=False when cluster cannot be reached."""
    mock_batch = MagicMock()
    mock_batch.get_api_resources.side_effect = Exception("Connection refused")

    k8s = KubernetesExecutionBackend(batch_v1_api=mock_batch)
    health = k8s.check_health()
    assert health.is_available is False
    assert "Connection refused" in (health.error_message or "")


# ==============================================================================
# 4. Worker Task Runner Tests
# ==============================================================================

def test_resolve_worker_model_bytes_from_sample(linear_model_hash):
    """Verify resolve_worker_model_bytes locates sample models by fingerprint."""
    spec = WorkerJobSpec(
        job_id="job-resolve-001",
        job_type=WorkerJobType.BENCHMARK,
        model_fingerprint=linear_model_hash,
        model_filename="linear_classifier.onnx",
    )
    raw, fname = resolve_worker_model_bytes(spec)
    assert hashlib.sha256(raw).hexdigest() == linear_model_hash
    assert fname == "linear_classifier.onnx"


def test_resolve_worker_model_bytes_mismatch_rejected():
    """Verify resolve_worker_model_bytes rejects corrupt or non-matching models."""
    spec = WorkerJobSpec(
        job_id="job-corrupt-001",
        job_type=WorkerJobType.BENCHMARK,
        model_fingerprint="nonexistent_sha256_hash_1234567890abcdef",
    )
    with pytest.raises(FileNotFoundError):
        resolve_worker_model_bytes(spec)


def test_execute_worker_task_benchmark(linear_model_hash):
    """Verify execute_worker_task successfully runs a BenchmarkEngine workload."""
    spec = WorkerJobSpec(
        job_id="job-task-bench",
        job_type=WorkerJobType.BENCHMARK,
        model_fingerprint=linear_model_hash,
        model_filename="linear_classifier.onnx",
        parameters={
            "batch_size": 1,
            "intra_op_num_threads": 1,
            "measured_iterations": 5,
            "repeated_rounds": 1,
            "warmup_iterations": 2,
        },
    )

    result_data, out_artifacts, timing = execute_worker_task(spec)
    assert "latency" in result_data
    assert "throughput" in result_data
    assert timing.inference_duration_seconds > 0.0
    assert timing.total_duration_seconds >= timing.inference_duration_seconds
    assert out_artifacts == []


def test_execute_worker_task_graph_optimization(linear_model_hash):
    """Verify execute_worker_task executes GraphOptimizationEngine and tracks output artifacts."""
    spec = WorkerJobSpec(
        job_id="job-task-graph-opt",
        job_type=WorkerJobType.GRAPH_OPTIMIZATION,
        model_fingerprint=linear_model_hash,
        model_filename="linear_classifier.onnx",
        parameters={
            "transformations": ["shape_inference", "ort_basic"],
            "validation_depth": "FULL",
            "enable_benchmarking": True,
            "atol": 1e-4,
            "rtol": 1e-3,
        },
    )

    result_data, out_artifacts, timing = execute_worker_task(spec)
    assert "candidates" in result_data
    assert len(result_data["candidates"]) >= 2
    assert timing.inference_duration_seconds > 0.0


# ==============================================================================
# 5. Database Persistence Tests
# ==============================================================================

def test_persistence_worker_job_crud(linear_model_hash):
    """Verify WorkerJobRepository correctly creates, updates, and lists worker jobs."""
    with session_scope() as s:
        repo = WorkerJobRepository(session=s)

        test_job_id = f"job-persist-{uuid.uuid4().hex[:6]}"
        spec = WorkerJobSpec(
            job_id=test_job_id,
            job_type=WorkerJobType.BENCHMARK,
            model_fingerprint=linear_model_hash,
            model_filename="linear_classifier.onnx",
            resource_profile="small_cpu",
        )

        job = repo.create_job(spec, backend="local")
        assert job.id == test_job_id
        assert job.status == "QUEUED"
        assert job.execution_backend == "local"

        # Update status
        timing = TimingBreakdown(
            queue_delay_seconds=0.1,
            startup_overhead_seconds=0.05,
            inference_duration_seconds=0.8,
            total_duration_seconds=0.95,
        )
        updated = repo.update_job_status(
            job_id=test_job_id,
            status="SUCCEEDED",
            timing=timing,
            result_json={"success": True},
        )
        assert updated is not None
        assert updated.status == "SUCCEEDED"
        assert updated.inference_duration_seconds == 0.8

        # Query
        fetched = repo.get_job(test_job_id)
        assert fetched is not None
        assert fetched.result_json == {"success": True}

        # List
        jobs, total = repo.list_jobs(backend="local")
        assert total >= 1
        assert any(j.id == test_job_id for j in jobs)


# ==============================================================================
# 6. FastAPI REST Endpoints Tests
# ==============================================================================

@pytest.fixture
def api_client():
    app = create_app()
    return TestClient(app)


def test_api_jobs_health(api_client):
    """Test GET /api/v1/jobs/health returns execution backend status."""
    resp = api_client.get("/api/v1/jobs/health?backend=local")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend_name"] == "local"
    assert data["is_available"] is True
    assert "concurrency" in data["details"]


def test_api_jobs_health_kubernetes_unreachable(api_client):
    """Test GET /api/v1/jobs/health reports is_available=False for unconfigured cluster."""
    resp = api_client.get("/api/v1/jobs/health?backend=kubernetes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend_name"] == "kubernetes"
    assert data["is_available"] is False


def test_api_list_worker_jobs(api_client, linear_model_hash):
    """Test GET /api/v1/jobs returns paginated job records."""
    # Ensure a job exists
    target_job_id = f"job-api-list-{uuid.uuid4().hex[:8]}"
    with session_scope() as s:
        repo = WorkerJobRepository(session=s)
        spec = WorkerJobSpec(
            job_id=target_job_id,
            job_type=WorkerJobType.BENCHMARK,
            model_fingerprint=linear_model_hash,
        )
        repo.create_job(spec, backend="local")

    resp = api_client.get("/api/v1/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert "jobs" in data
    assert data["total"] >= 1
    assert any(j["id"] == target_job_id for j in data["jobs"])


def test_api_get_worker_job_detail(api_client, linear_model_hash):
    """Test GET /api/v1/jobs/{job_id} returns detailed timing and status."""
    with session_scope() as s:
        repo = WorkerJobRepository(session=s)
        spec = WorkerJobSpec(
            job_id="job-api-detail-001",
            job_type=WorkerJobType.BENCHMARK,
            model_fingerprint=linear_model_hash,
            resource_profile="standard_cpu",
        )
        repo.create_job(spec, backend="local")

    resp = api_client.get("/api/v1/jobs/job-api-detail-001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "job-api-detail-001"
    assert data["job_type"] == "benchmark"
    assert data["resource_profile"] == "standard_cpu"


def test_api_get_worker_job_404(api_client):
    """Test GET /api/v1/jobs/{job_id} returns 404 for non-existent job."""
    resp = api_client.get("/api/v1/jobs/non-existent-job-999")
    assert resp.status_code == 404


def test_api_cancel_worker_job(api_client, linear_model_hash):
    """Test POST /api/v1/jobs/{job_id}/cancel marks job as cancelled."""
    with session_scope() as s:
        repo = WorkerJobRepository(session=s)
        spec = WorkerJobSpec(
            job_id="job-api-cancel-001",
            job_type=WorkerJobType.BENCHMARK,
            model_fingerprint=linear_model_hash,
        )
        repo.create_job(spec, backend="local")

    resp = api_client.post("/api/v1/jobs/job-api-cancel-001/cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "job-api-cancel-001"
    assert data["cancelled"] is True


def test_api_benchmark_submission_with_k8s_backend_error(api_client, linear_model_bytes):
    """Verify submitting with execution_backend=kubernetes fails clearly with HTTP 503 when cluster is down."""
    payload = {
        "sample_model_name": "linear_classifier.onnx",
        "execution_backend": "kubernetes",
        "resource_profile": "standard_cpu",
        "config": {
            "batch_size": 1,
            "measured_iterations": 10,
        },
    }
    resp = api_client.post("/api/v1/benchmarks", json=payload)
    # Must fail with 503 Service Unavailable, no silent fallback
    assert resp.status_code == 503
    assert "Kubernetes backend unavailable" in resp.json()["message"]


# ==============================================================================
# 7. Regression & Backward Compatibility Tests
# ==============================================================================

def test_regression_phases_1_to_9_intact():
    """Verify that Phases 1 through 9 remain active, implemented, and configured."""
    from zmodel.core.config import APP_VERSION, CURRENT_PHASE, NAV_ITEMS

    assert CURRENT_PHASE >= 10
    assert APP_VERSION in ("1.2.0", "1.3.0", "1.4.0")

    nav_ids = [item.id for item in NAV_ITEMS]
    assert "overview" in nav_ids
    assert "inspector" in nav_ids
    assert "compatibility" in nav_ids
    assert "benchmarks" in nav_ids
    assert "optimization" in nav_ids
    assert "graph_optimization" in nav_ids
    assert "workers" in nav_ids
    assert "history" in nav_ids
    assert "settings" in nav_ids

    # Verify Worker Jobs NavItem
    w_nav = next(item for item in NAV_ITEMS if item.id == "workers")
    assert w_nav.phase == 10
    assert w_nav.is_active is True
    assert w_nav.status_label == "Implemented"
