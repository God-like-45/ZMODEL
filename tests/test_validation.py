"""Comprehensive automated test suite for Phase 11: IBM Z / Linux on IBM Z Validation & Deployment Readiness."""

import json
import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from zmodel.core.environment import EnvironmentInventory, inspect_environment
from zmodel.core.target_profiles import (
    TargetValidationProfile,
    get_target_profile,
    load_target_profiles,
)
from zmodel.core.validation_engine import (
    CheckStatus,
    OverallValidationStatus,
    ValidationCheck,
    ValidationEngine,
    ValidationReport,
)
from zmodel.core.target_runner import (
    LocalValidationRunner,
    MockIBMZTargetRunner,
    RemoteSSHTargetRunner,
    TargetNotConfiguredError,
    TargetRunner,
    get_target_runner,
)
from zmodel.core.benchmark import (
    BenchmarkConfig,
    BenchmarkEngine,
)
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.db.models import Base, ValidationRunORM
from zmodel.db.repositories import ValidationRepository
from zmodel.db.session import get_engine, session_scope
from zmodel.api.app import create_app
from fastapi.testclient import TestClient


SAMPLE_LINEAR_MODEL = os.path.join("sample_models", "linear_classifier.onnx")


@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure database schema is created and available for persistence and API tests."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    yield


# =========================================================================
# 1. Environment Detection & Inventory Tests
# =========================================================================

def test_inspect_environment_local():
    """Verify local environment inspection gathers non-empty facts with canonical fingerprint."""
    env = inspect_environment(collection_method="LOCAL_INSPECTION")
    assert env.architecture in ("x86_64", "AMD64", "arm64", "aarch64", "s390x")
    assert env.python_version != "UNKNOWN"
    assert len(env.canonical_fingerprint) == 64  # SHA-256
    assert isinstance(env.available_providers, list)
    assert env.collection_method == "LOCAL_INSPECTION"


def test_environment_unknown_architecture_fallback():
    """Verify missing/unknown architecture data is safely represented as UNKNOWN without crashing."""
    with patch("platform.machine", return_value=""):
        env = inspect_environment()
        assert env.architecture == "UNKNOWN"
        assert not env.is_s390x


def test_environment_s390x_flag():
    """Verify s390x architecture flag detection."""
    env_s390x = EnvironmentInventory(
        architecture="s390x",
        os_name="Linux",
        os_release="5.14.0",
        kernel_version="5.14.0-s390x",
        python_version="3.11.5",
        cpu_model="IBM z16 (3931)",
        onnxruntime_version="1.17.0",
        available_providers=["CPUExecutionProvider", "NNPAExecutionProvider"],
        has_nnpa_facility=True,
        has_nnpa_provider=True,
    )
    assert env_s390x.is_s390x is True
    assert env_s390x.has_nnpa_provider is True


# =========================================================================
# 2. Target Profiles Tests
# =========================================================================

def test_load_target_profiles():
    """Verify target profiles load with correct versions and constraints."""
    profiles = load_target_profiles()
    assert "ibm_z16_linux_nnpa" in profiles
    assert "ibm_z_linux_cpu" in profiles
    assert "local_host_development" in profiles

    z16 = profiles["ibm_z16_linux_nnpa"]
    assert z16.architecture == "s390x"
    assert z16.requires_accelerator is True
    assert any("NNPA" in acc for acc in z16.required_accelerators)
    assert "NNPAExecutionProvider" in z16.required_providers


def test_get_target_profile_fallback():
    """Verify fallback to local profile on unknown profile ID when requested, else raises TargetProfileError."""
    from zmodel.core.target_profiles import TargetProfileError
    with pytest.raises(TargetProfileError):
        get_target_profile("non_existent_profile_xyz")

    p = get_target_profile("non_existent_profile_xyz", fallback_to_default=True)
    assert p.profile_id == "local_host_development"


# =========================================================================
# 3. 12-Check Validation Engine Tests
# =========================================================================

def test_validation_engine_on_local_host_evaluating_s390x():
    """When evaluated on an x86_64 local host for an s390x profile:
    - Target hardware checks must return NOT_RUN.
    - Local loadability/numerical checks pass.
    - Overall status is LOCALLY_VERIFIED (never claimed as TARGET_VERIFIED).
    """
    profile = get_target_profile("ibm_z16_linux_nnpa")
    env = inspect_environment()

    # If the test is running on an x86_64 host
    if not env.is_s390x:
        engine = ValidationEngine(profile=profile, environment=env)
        report = engine.evaluate(model_path=SAMPLE_LINEAR_MODEL)

        assert report.overall_status == OverallValidationStatus.LOCALLY_VERIFIED
        assert "Locally Verified" in report.provenance_badge
        assert report.summary_counts["NOT_RUN"] >= 3  # arch, os, providers, accelerator
        assert report.summary_counts["PASS"] >= 6    # load, io, operators, numerical, etc.

        # Check specific checks
        check_map = {c.check_id: c for c in report.checks}
        assert check_map["check_01_arch"].status == CheckStatus.NOT_RUN
        assert check_map["check_06_loadability"].status == CheckStatus.PASS
        assert check_map["check_10_numerical"].status == CheckStatus.PASS
        assert check_map["check_12_accelerator"].status == CheckStatus.NOT_RUN


def test_validation_engine_on_simulated_ibm_z():
    """Verify that when evaluated with authentic s390x + NNPA environment, status is TARGET_VERIFIED."""
    profile = get_target_profile("ibm_z16_linux_nnpa")
    runner = MockIBMZTargetRunner(profile=profile)
    report = runner.run_validation(model_path=SAMPLE_LINEAR_MODEL)

    assert report.overall_status == OverallValidationStatus.TARGET_VERIFIED
    assert report.provenance_badge == "Verified on IBM Z (s390x)"
    assert report.summary_counts["NOT_RUN"] == 0
    assert report.summary_counts["FAIL"] == 0


def test_validation_engine_missing_execution_provider_fallback():
    """Verify that if a required provider is missing or falls back, it fails or warns, never silent pass."""
    profile = get_target_profile("ibm_z16_linux_nnpa")
    # Environment missing NNPAExecutionProvider
    env = EnvironmentInventory(
        architecture="s390x",
        os_name="Linux",
        os_release="5.14.0",
        kernel_version="5.14.0-s390x",
        python_version="3.11.5",
        cpu_model="IBM z16 (3931)",
        onnxruntime_version="1.17.0",
        available_providers=["CPUExecutionProvider"],  # NNPA missing!
        has_nnpa_facility=True,
        has_nnpa_provider=False,
    )
    engine = ValidationEngine(profile=profile, environment=env)
    report = engine.evaluate(model_path=SAMPLE_LINEAR_MODEL)

    check_map = {c.check_id: c for c in report.checks}
    assert check_map["check_05_providers"].status == CheckStatus.FAIL
    assert "NNPAExecutionProvider" in check_map["check_05_providers"].evidence
    assert report.overall_status in (OverallValidationStatus.FAILED_VALIDATION, OverallValidationStatus.PARTIALLY_VERIFIED)


def test_validation_engine_model_loading_failure():
    """Verify non-existent or invalid model produces FAIL on loadability check."""
    profile = get_target_profile("local_host_development")
    env = inspect_environment()
    engine = ValidationEngine(profile=profile, environment=env)
    report = engine.evaluate(model_path="non_existent_model_xyz.onnx")

    check_map = {c.check_id: c for c in report.checks}
    assert check_map["check_06_loadability"].status == CheckStatus.FAIL
    assert report.overall_status == OverallValidationStatus.FAILED_VALIDATION


def test_validation_numerical_correctness_tolerance():
    """Verify numerical correctness evaluates max_abs_error and tolerances."""
    profile = get_target_profile("local_host_development")
    env = inspect_environment()
    engine = ValidationEngine(profile=profile, environment=env)
    report = engine.evaluate(model_path=SAMPLE_LINEAR_MODEL)

    check_map = {c.check_id: c for c in report.checks}
    assert check_map["check_10_numerical"].status == CheckStatus.PASS
    assert "max_abs_error" in report.numerical_correctness
    assert report.numerical_correctness["passed"] is True


# =========================================================================
# 4. Target Runner & Security Tests
# =========================================================================

def test_remote_ssh_target_runner_unconfigured_error():
    """Verify RemoteSSHTargetRunner raises TargetNotConfiguredError when environment variables are missing."""
    with patch.dict(os.environ, {}, clear=True):
        runner = RemoteSSHTargetRunner()
        with pytest.raises(TargetNotConfiguredError) as exc_info:
            runner.check_connection()
        assert "ZMODEL_TARGET_HOST" in str(exc_info.value)


def test_remote_ssh_target_runner_secret_redaction():
    """Verify private keys, tokens, or credentials are never leaked in logs or summaries."""
    with patch.dict(os.environ, {
        "ZMODEL_TARGET_HOST": "zlinux.enterprise.internal",
        "ZMODEL_TARGET_USER": "aiuser",
        "ZMODEL_TARGET_KEY_PATH": "/secrets/super_secret_ssh_key.pem",
    }):
        runner = RemoteSSHTargetRunner()
        summary = runner.get_configuration_summary()
        assert "super_secret" not in json.dumps(summary)
        assert summary["configured"] is True
        assert summary["key_configured"] is True


def test_get_target_runner_factory():
    """Verify get_target_runner returns appropriate runner implementations."""
    local_runner = get_target_runner(mode="local")
    assert isinstance(local_runner, LocalValidationRunner)

    mock_runner = get_target_runner(mode="mock_ibm_z")
    assert isinstance(mock_runner, MockIBMZTargetRunner)


# =========================================================================
# 5. Benchmark Provenance & Architecture Integrity Tests
# =========================================================================

def test_benchmark_provenance_recording():
    """Verify that BenchmarkEngine records benchmark_provenance and host architecture accurately."""
    with open(SAMPLE_LINEAR_MODEL, "rb") as f:
        model_bytes = f.read()

    summary = load_and_inspect_model(model_bytes, filename="linear_classifier.onnx")
    config = BenchmarkConfig(
        execution_provider="CPUExecutionProvider",
        warmup_iterations=2,
        measured_iterations=5,
        batch_size=1,
        repeated_rounds=1,
    )
    report = BenchmarkEngine.run_benchmark(model_bytes, summary, config)

    assert report.status == "SUCCESS"
    assert report.environment_meta is not None
    assert "benchmark_provenance" in report.environment_meta
    assert report.environment_meta["benchmark_provenance"] in ("LOCAL_HOST", "IBM_Z_NATIVE")
    assert "provenance_label" in report.environment_meta
    # Must never claim s390x on x86_64 host
    if report.environment_meta["host_architecture"] != "s390x":
        assert report.environment_meta["target_architecture_verified"] is False
        assert "Local Host" in report.environment_meta["provenance_label"]


# =========================================================================
# 6. Database Persistence & Repository Tests
# =========================================================================

def test_validation_repository_crud():
    """Verify ValidationRepository saves, retrieves, and lists validation runs."""
    profile = get_target_profile("ibm_z16_linux_nnpa")
    runner = MockIBMZTargetRunner(profile=profile)
    report = runner.run_validation(model_path=SAMPLE_LINEAR_MODEL)

    with session_scope() as session:
        repo = ValidationRepository(session)
        orm_obj = repo.save_validation_report(report)
        assert orm_obj.id == report.run_id

        # Retrieve by ID
        fetched = repo.get_validation_run(report.run_id)
        assert fetched is not None
        assert fetched.id == report.run_id
        assert fetched.overall_status == "TARGET_VERIFIED"
        assert fetched.target_profile_id == "ibm_z16_linux_nnpa"
        assert len(fetched.checks_json) == len(report.checks)

        # List runs
        runs, total = repo.list_validation_runs(limit=10)
        assert total >= 1
        assert any(r.id == report.run_id for r in runs)


# =========================================================================
# 7. FastAPI REST API Validation Endpoints Tests
# =========================================================================

def test_api_validation_environment_endpoint():
    """Verify GET /api/v1/validation/environment returns detected environment facts."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/api/v1/validation/environment")
    assert resp.status_code == 200
    data = resp.json()
    assert "architecture" in data
    assert "environment_fingerprint" in data
    assert "available_providers" in data


def test_api_validation_profiles_endpoints():
    """Verify GET /api/v1/validation/profiles and GET /profiles/{id}."""
    app = create_app()
    client = TestClient(app)

    resp = client.get("/api/v1/validation/profiles")
    assert resp.status_code == 200
    profiles = resp.json()
    assert len(profiles) >= 3
    assert any(p["profile_id"] == "ibm_z16_linux_nnpa" for p in profiles)

    resp_single = client.get("/api/v1/validation/profiles/ibm_z16_linux_nnpa")
    assert resp_single.status_code == 200
    p_data = resp_single.json()
    assert p_data["target_architecture"] == "s390x"


def test_api_validation_evaluate_endpoint():
    """Verify POST /api/v1/validation/evaluate executes 12-point checks and returns structured report."""
    app = create_app()
    client = TestClient(app)

    payload = {
        "sample_model_name": "linear_classifier.onnx",
        "target_profile_id": "ibm_z16_linux_nnpa",
        "execution_mode": "LOCAL",
    }
    resp = client.post("/api/v1/validation/evaluate", json=payload)
    assert resp.status_code == 200
    report_data = resp.json()

    assert "run_id" in report_data
    assert "overall_status" in report_data
    assert "checks" in report_data
    assert len(report_data["checks"]) == 12
    assert "provenance_badge" in report_data


def test_api_validation_history_endpoints():
    """Verify GET /api/v1/validation/runs lists persisted validation runs."""
    app = create_app()
    client = TestClient(app)

    # First evaluate to ensure at least one record
    client.post("/api/v1/validation/evaluate", json={
        "sample_model_name": "linear_classifier.onnx",
        "target_profile_id": "local_host_development",
        "execution_mode": "LOCAL",
    })

    resp = client.get("/api/v1/validation/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert "runs" in data
    assert len(data["runs"]) >= 1
    run_id = data["runs"][0]["run_id"]

    resp_single = client.get(f"/api/v1/validation/runs/{run_id}")
    assert resp_single.status_code == 200
    assert resp_single.json()["run_id"] == run_id
