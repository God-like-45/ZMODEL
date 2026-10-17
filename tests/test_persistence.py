"""Comprehensive unit and integration tests for Phase 6: PostgreSQL Persistence & Experiment History."""

import os
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from zmodel.core.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    LatencySummary,
    MemorySummary,
    ThroughputSummary,
)
from zmodel.core.compatibility import (
    CompatibilityFinding,
    CompatibilityReport,
)
from zmodel.core.model_loader import (
    ModelMetadata,
    ModelSummary,
    OpsetInfo,
    TensorInfo,
)
from zmodel.core.optimizer import (
    CandidateStatus,
    ConstraintEvaluation,
    ExecutionCandidate,
    ExperimentStatus,
    HardConstraint,
    OptimizationObjective,
    OptimizationReport,
    OptimizationRequest,
)
from zmodel.db.comparator import (
    ComparisonValidity,
    compare_benchmark_runs,
)
from zmodel.db.config import (
    check_database_connection,
    get_database_url,
    mask_database_url,
)
from zmodel.db.models import (
    Base,
    BenchmarkRunORM,
    CandidateEvaluationORM,
    CompatibilityAssessmentORM,
    ModelIdentityORM,
    OptimizationExperimentORM,
)
from zmodel.db.repositories import (
    BenchmarkRepository,
    CompatibilityRepository,
    HistoryOverviewRepository,
    ModelRepository,
    OptimizationRepository,
)
from zmodel.db.session import get_session_factory, session_scope


@pytest.fixture
def test_db():
    """Create an isolated, in-memory SQLite test database with all tables initialized."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = get_session_factory(engine)
    yield engine, session_factory
    engine.dispose()


@pytest.fixture
def sample_model_summary() -> ModelSummary:
    """Generate a clean synthetic ModelSummary for persistence testing."""
    return ModelSummary(
        filename="test_linear.onnx",
        file_size_bytes=4096,
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        metadata=ModelMetadata(
            producer_name="pytorch",
            producer_version="2.3.0",
            model_version=1,
            ir_version=9,
            doc_string="Linear test classifier",
            domain="ai.onnx",
            custom_metadata={},
        ),
        opsets=[OpsetInfo(domain="", version=17)],
        inputs=[TensorInfo("X", "float32", [1, 10], 2, False, [], "batch 1, 10 features")],
        outputs=[TensorInfo("Y", "float32", [1, 2], 2, False, [], "batch 1, 2 classes")],
        nodes=[],
        weights=[],
        op_counts={"Gemm": 1, "Relu": 1},
        distinct_operator_count=2,
        total_nodes=2,
        total_parameters=256,
        total_weight_bytes=1024,
        has_external_data=False,
        external_data_tensors=[],
        warnings=[],
        is_valid=True,
        parsing_status="VALID",
        inference_status="INFERRED",
        compatibility_status="COMPATIBLE",
    )


# 1. Database Configuration & Credential Masking Tests
def test_database_configuration_and_masking():
    assert mask_database_url("postgresql+psycopg://user:secret123@localhost:5432/testdb") == (
        "postgresql+psycopg://user:***@localhost:5432/testdb"
    )
    assert mask_database_url("sqlite:///local.db") == "sqlite:///local.db"
    assert mask_database_url("") == "Not Configured"
    assert mask_database_url(None) != ""


# 2. Database Connectivity Health Check
def test_database_connectivity_health_check(test_db):
    engine, _ = test_db
    status = check_database_connection(engine)
    assert status["connected"] is True
    assert status["dialect"] == "sqlite"
    assert status["table_count"] in (5, 7, 8, 9)
    assert "models" in status["tables"]
    assert "benchmark_runs" in status["tables"]
    assert "optimization_experiments" in status["tables"]
    assert "graph_optimization_experiments" in status["tables"]
    assert "worker_jobs" in status["tables"]
    assert "validation_runs" in status["tables"]
    assert status["error"] is None


# 3. Model Identity Registration
def test_model_identity_registration(test_db, sample_model_summary):
    _, session_factory = test_db
    with session_factory() as s:
        repo = ModelRepository(session=s)
        model = repo.register_or_update(sample_model_summary)
        s.commit()

        assert model.fingerprint == sample_model_summary.sha256_hash
        assert model.original_filename == "test_linear.onnx"
        assert model.file_size_bytes == 4096
        assert model.ir_version == 9
        assert model.first_seen_at is not None
        assert model.last_seen_at is not None


# 4. Same Filename with Different Fingerprints (Distinct Records)
def test_same_filename_different_fingerprints(test_db, sample_model_summary):
    _, session_factory = test_db
    summary_v1 = sample_model_summary
    summary_v2 = ModelSummary(
        filename="test_linear.onnx",  # Identical filename
        file_size_bytes=8192,
        sha256_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # Distinct fingerprint
        metadata=summary_v1.metadata,
        opsets=summary_v1.opsets,
        inputs=summary_v1.inputs,
        outputs=summary_v1.outputs,
        nodes=[],
        weights=[],
        op_counts={},
        distinct_operator_count=0,
        total_nodes=0,
        total_parameters=512,
        total_weight_bytes=2048,
        has_external_data=False,
        external_data_tensors=[],
        warnings=[],
        is_valid=True,
        parsing_status="VALID",
        inference_status="INFERRED",
        compatibility_status="COMPATIBLE",
    )

    with session_factory() as s:
        repo = ModelRepository(session=s)
        repo.register_or_update(summary_v1)
        repo.register_or_update(summary_v2)
        s.commit()

        models = repo.list_models()
        assert len(models) == 2
        fps = {m.fingerprint for m in models}
        assert summary_v1.sha256_hash in fps
        assert summary_v2.sha256_hash in fps


# 5. Repeated Registration of Same Fingerprint (Updates last_seen)
def test_repeated_registration_updates_last_seen(test_db, sample_model_summary):
    _, session_factory = test_db
    with session_factory() as s:
        repo = ModelRepository(session=s)
        m1 = repo.register_or_update(sample_model_summary)
        s.commit()
        t1 = m1.last_seen_at

        # Re-register
        m2 = repo.register_or_update(sample_model_summary)
        s.commit()
        t2 = m2.last_seen_at

        assert m1.fingerprint == m2.fingerprint
        assert repo.count_models() == 1
        assert t2 >= t1


# 6. Saving and Retrieving Compatibility Assessments
def test_save_and_retrieve_compatibility_assessment(test_db):
    _, session_factory = test_db
    report = CompatibilityReport(
        model_name="test_linear.onnx",
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        profile_id="local_onnx_runtime_cpu",
        profile_name="Local CPU",
        profile_version="1.0.0",
        ruleset_version="2026.1",
        target_architecture="x86_64",
        target_runtime="ONNX Runtime CPU",
        confidence_classification="VERIFIED_DOCUMENTED",
        analysis_timestamp="2026-10-17T12:00:00Z",
        overall_assessment="COMPATIBLE",
        summary_counts={"PASS": 2, "FAIL": 0, "WARNING": 0, "UNKNOWN": 0},
        findings=[
            CompatibilityFinding(
                finding_id="F-01",
                category="operator",
                status="PASS",
                severity="CRITICAL",
                feature="Gemm",
                observed_value="opset 17",
                profile_rule="Supported opset 7-21",
                explanation="Gemm is supported",
                evidence_source="Official ONNX spec",
                recommended_action="Safe for deployment",
            )
        ],
        unresolved_questions=[],
        assumptions=["Standard CPU execution"],
        recommended_next_steps=["Run empirical benchmark"],
        is_illustrative_profile=False,
    )

    with session_factory() as s:
        repo = CompatibilityRepository(session=s)
        saved = repo.save_assessment(report)
        s.commit()

        assert saved.overall_assessment == "COMPATIBLE"
        assert saved.profile_version == "1.0.0"
        assert saved.ruleset_version == "2026.1"
        assert saved.model_fingerprint == report.model_fingerprint

        retrieved = repo.get_assessment(saved.id)
        assert retrieved is not None
        assert retrieved.id == saved.id
        assert retrieved.summary_counts["PASS"] == 2


# 7. Preserving Profile and Ruleset Versions Historically
def test_preserving_profile_and_ruleset_versions(test_db):
    _, session_factory = test_db
    report_v1 = CompatibilityReport(
        model_name="test.onnx",
        model_fingerprint="fp1234567890abcdef",
        profile_id="ibm_z_onnx_illustrative",
        profile_name="IBM Z",
        profile_version="1.0.0",
        ruleset_version="2026.1",
        target_architecture="s390x",
        target_runtime="ONNX Runtime NNPA",
        confidence_classification="ILLUSTRATIVE_DRAFT",
        analysis_timestamp="2026-01-01T00:00:00Z",
        overall_assessment="CONDITIONAL",
        summary_counts={"PASS": 1, "WARNING": 1},
        findings=[],
        unresolved_questions=[],
        assumptions=[],
        recommended_next_steps=[],
        is_illustrative_profile=True,
    )

    with session_factory() as s:
        repo = CompatibilityRepository(session=s)
        rec = repo.save_assessment(report_v1)
        s.commit()

        assert rec.profile_version == "1.0.0"
        assert rec.ruleset_version == "2026.1"
        assert rec.is_illustrative_profile is True


# 8. Saving Successful Benchmark Runs
def test_saving_successful_benchmark_run(test_db):
    _, session_factory = test_db
    bench_rep = BenchmarkReport(
        run_id="run-succ-001",
        timestamp_utc="2026-10-17T14:00:00Z",
        model_name="test_linear.onnx",
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status="SUCCESS",
        error_message=None,
        environment={"os": "Windows", "python": "3.11"},
        config={"batch_size": 1, "intra_op_num_threads": 2, "execution_provider": "CPUExecutionProvider"},
        input_spec={"X": {"shape": [1, 10], "dtype": "float32"}},
        correctness={"passed": True, "details": "Smoke test passed"},
        latency={"median_ms": 0.0152, "mean_ms": 0.0155, "p95_ms": 0.0180, "p99_ms": 0.0210, "min_ms": 0.0140, "max_ms": 0.0250, "std_dev_ms": 0.0012, "sample_count": 50},
        throughput={"inferences_per_second": 65780.0, "samples_per_second": 65780.0},
        memory={"session_delta_mb": 1.2, "peak_rss_mb": 84.5},
        per_round_results=[],
        all_latencies_ms=[0.0152],
        warnings=[],
        limitations=["CPU only"],
    )

    with session_factory() as s:
        repo = BenchmarkRepository(session=s)
        saved = repo.save_benchmark_report(bench_rep)
        s.commit()

        assert saved.id == "run-succ-001"
        assert saved.status == "SUCCESS"
        assert saved.correctness_status == "PASS"
        assert saved.median_latency_ms == 0.0152
        assert saved.inferences_per_second == 65780.0
        assert saved.session_delta_mb == 1.2
        assert saved.peak_rss_mb == 84.5
        assert saved.schema_version == "1.0.0"


# 9. Persisting Failed Benchmark Runs with Correct Status
def test_persisting_failed_benchmark_run(test_db):
    _, session_factory = test_db
    failed_rep = BenchmarkReport(
        run_id="run-fail-002",
        timestamp_utc="2026-10-17T14:05:00Z",
        model_name="test_linear.onnx",
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status="FAILED",
        error_message="Non-finite output detected (NaN/Inf in logits)",
        environment={},
        config={"batch_size": 1, "intra_op_num_threads": 1, "execution_provider": "CPUExecutionProvider"},
        input_spec={},
        correctness={"passed": False, "details": "NaN detected"},
        latency=None,
        throughput=None,
        memory=None,
        per_round_results=[],
        all_latencies_ms=[],
        warnings=["Non-finite outputs"],
        limitations=[],
    )

    with session_factory() as s:
        repo = BenchmarkRepository(session=s)
        saved = repo.save_benchmark_report(failed_rep)
        s.commit()

        assert saved.id == "run-fail-002"
        assert saved.status == "FAILED"
        assert saved.correctness_status == "FAIL"
        assert "Non-finite output detected" in saved.error_message
        assert saved.median_latency_ms is None
        assert saved.inferences_per_second is None


# 10. Missing Memory and Latency Metrics Explicitly Stored as None (Not 0.0)
def test_missing_metrics_explicitly_stored_as_none(test_db):
    _, session_factory = test_db
    rep = BenchmarkReport(
        run_id="run-none-003",
        timestamp_utc="2026-10-17T14:10:00Z",
        model_name="test_linear.onnx",
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status="SUCCESS",
        error_message=None,
        environment={},
        config={"batch_size": 1, "intra_op_num_threads": 1, "execution_provider": "CPUExecutionProvider"},
        input_spec={},
        correctness={"passed": True, "details": "ok"},
        latency={"median_ms": 0.02, "mean_ms": 0.02, "p95_ms": 0.025, "p99_ms": 0.03, "min_ms": 0.015, "max_ms": 0.035, "std_dev_ms": 0.002, "sample_count": 10},
        throughput=None,  # Missing throughput
        memory=None,      # Missing memory
        per_round_results=[],
        all_latencies_ms=[],
        warnings=[],
        limitations=[],
    )

    with session_factory() as s:
        repo = BenchmarkRepository(session=s)
        saved = repo.save_benchmark_report(rep)
        s.commit()

        assert saved.inferences_per_second is None
        assert saved.session_delta_mb is None
        assert saved.peak_rss_mb is None


# 11. Saving Optimization Experiments
def test_saving_optimization_experiment(test_db):
    _, session_factory = test_db
    req = OptimizationRequest(
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
        primary_objective=OptimizationObjective.MINIMIZE_MEDIAN_LATENCY,
    )
    opt_rep = OptimizationReport(
        experiment_id="exp-001",
        timestamp_utc="2026-10-17T15:00:00Z",
        status=ExperimentStatus.COMPLETED,
        model_name="test_linear.onnx",
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        profile_id="local_onnx_runtime_cpu",
        profile_name="Local CPU",
        profile_version="1.0.0",
        primary_objective="MINIMIZE_MEDIAN_LATENCY",
        request=req,
        baseline_candidate_id="CAND-01",
        best_candidate_id="CAND-02",
        candidates=[
            ExecutionCandidate(
                candidate_id="CAND-01",
                config=BenchmarkConfig(batch_size=1, intra_op_num_threads=1),
                generation_reason="Baseline configuration",
                status=CandidateStatus.BENCHMARKED,
                is_feasible=True,
                is_baseline=True,
                rank=2,
            ),
            ExecutionCandidate(
                candidate_id="CAND-02",
                config=BenchmarkConfig(batch_size=1, intra_op_num_threads=2),
                generation_reason="Parallel threads",
                status=CandidateStatus.BENCHMARKED,
                is_feasible=True,
                is_pareto_optimal=True,
                speedup_vs_baseline=1.5,
                latency_reduction_pct=33.3,
                rank=1,
            ),
        ],
        pareto_frontier_ids=["CAND-02"],
        summary_counts={"candidates_total": 2, "candidates_feasible": 2, "candidates_rejected": 0},
        recommendation_rationale="CAND-02 yielded 1.5x speedup.",
        environment={},
    )

    with session_factory() as s:
        repo = OptimizationRepository(session=s)
        saved = repo.save_optimization_report(opt_rep)
        s.commit()

        assert saved.id == "exp-001"
        assert saved.best_candidate_id == "CAND-02"
        assert saved.baseline_candidate_id == "CAND-01"
        assert len(saved.candidates) == 2
        assert saved.candidates[0].candidate_id == "CAND-02"  # Ordered by rank (rank 1 first)


# 12. Candidate-to-Experiment Relationships and Cascade Delete
def test_candidate_to_experiment_cascade_delete(test_db):
    _, session_factory = test_db
    req = OptimizationRequest(
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
    )
    opt_rep = OptimizationReport(
        experiment_id="exp-cascade-001",
        timestamp_utc="2026-10-17T15:00:00Z",
        status=ExperimentStatus.COMPLETED,
        model_name="test.onnx",
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        profile_id="local_onnx_runtime_cpu",
        profile_name="Local CPU",
        profile_version="1.0.0",
        primary_objective="MINIMIZE_MEDIAN_LATENCY",
        request=req,
        baseline_candidate_id=None,
        best_candidate_id=None,
        candidates=[
            ExecutionCandidate(
                candidate_id="CAND-01",
                config=BenchmarkConfig(batch_size=1),
                generation_reason="Test candidate",
                status=CandidateStatus.GENERATED,
            )
        ],
        pareto_frontier_ids=[],
        summary_counts={},
        recommendation_rationale="",
        environment={},
    )

    with session_factory() as s:
        repo = OptimizationRepository(session=s)
        repo.save_optimization_report(opt_rep)
        s.commit()

        # Check candidate exists
        cands = s.scalars(select(CandidateEvaluationORM).where(CandidateEvaluationORM.experiment_id == "exp-cascade-001")).all()
        assert len(list(cands)) == 1

        # Delete experiment
        exp = s.get(OptimizationExperimentORM, "exp-cascade-001")
        s.delete(exp)
        s.commit()

        # Verify candidate was cascade-deleted
        cands_after = s.scalars(select(CandidateEvaluationORM).where(CandidateEvaluationORM.experiment_id == "exp-cascade-001")).all()
        assert len(list(cands_after)) == 0


# 13. Preserving Rejected and Unbenchmarked Candidates with Rejection Reasons
def test_preserving_rejected_candidates(test_db):
    _, session_factory = test_db
    req = OptimizationRequest(
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        profile_id="local_onnx_runtime_cpu",
        profile_version="1.0.0",
    )
    opt_rep = OptimizationReport(
        experiment_id="exp-rej-001",
        timestamp_utc="2026-10-17T15:00:00Z",
        status=ExperimentStatus.COMPLETED_WITH_WARNINGS,
        model_name="test.onnx",
        model_fingerprint="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        profile_id="local_onnx_runtime_cpu",
        profile_name="Local CPU",
        profile_version="1.0.0",
        primary_objective="MINIMIZE_MEDIAN_LATENCY",
        request=req,
        baseline_candidate_id=None,
        best_candidate_id=None,
        candidates=[
            ExecutionCandidate(
                candidate_id="CAND-04",
                config=BenchmarkConfig(batch_size=4),
                generation_reason="Batch size 4",
                status=CandidateStatus.REJECTED,
                rejection_reason="Model has fixed batch dimension = 1; requested batch size 4 is incompatible",
                is_feasible=False,
            )
        ],
        pareto_frontier_ids=[],
        summary_counts={"candidates_rejected": 1},
        recommendation_rationale="",
        environment={},
    )

    with session_factory() as s:
        repo = OptimizationRepository(session=s)
        saved = repo.save_optimization_report(opt_rep)
        s.commit()

        cand = saved.candidates[0]
        assert cand.status == "REJECTED"
        assert cand.is_feasible is False
        assert "fixed batch dimension = 1" in cand.rejection_reason
        assert cand.measured_median_latency_ms is None


# 14. Referential Integrity Enforcement
def test_referential_integrity_foreign_key(test_db):
    engine, session_factory = test_db
    # In SQLite, enable foreign keys explicitly
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = ON;")

    with session_factory() as s:
        s.connection().exec_driver_sql("PRAGMA foreign_keys = ON;")
        # Attempt to insert candidate pointing to non-existent experiment
        orphan_cand = CandidateEvaluationORM(
            id="orphan-01",
            experiment_id="non_existent_exp_id",
            candidate_id="CAND-01",
            configuration={},
            generation_reason="test",
            status="GENERATED",
        )
        s.add(orphan_cand)
        with pytest.raises(IntegrityError):
            s.commit()


# 15. Transaction Rollback after Persistence Failure
def test_transaction_rollback_after_failure(test_db):
    _, session_factory = test_db
    with pytest.raises(ValueError):
        with session_scope(test_db[0]) as s:
            m = ModelIdentityORM(
                fingerprint="fail_fp_12345",
                original_filename="fail.onnx",
                file_size_bytes=100,
            )
            s.add(m)
            s.flush()
            # Force an unexpected failure
            raise ValueError("Simulated unexpected crash during transaction")

    with session_factory() as s:
        # Verify the record was rolled back cleanly
        res = s.get(ModelIdentityORM, "fail_fp_12345")
        assert res is None


# 16. Duplicate-Submit and Idempotency Behavior
def test_idempotent_duplicate_submission(test_db):
    _, session_factory = test_db
    bench_rep = BenchmarkReport(
        run_id="run-idem-001",
        timestamp_utc="2026-10-17T14:00:00Z",
        model_name="test.onnx",
        model_fingerprint="fp_idem_123",
        status="SUCCESS",
        error_message=None,
        environment={},
        config={"batch_size": 1, "intra_op_num_threads": 1, "execution_provider": "CPUExecutionProvider"},
        input_spec={},
        correctness={"passed": True, "details": "ok"},
        latency={"median_ms": 0.01},
        throughput={},
        memory={},
        per_round_results=[],
        all_latencies_ms=[],
        warnings=[],
        limitations=[],
    )

    with session_factory() as s:
        repo = BenchmarkRepository(session=s)
        r1 = repo.save_benchmark_report(bench_rep)
        s.commit()

        # Re-save identical run
        r2 = repo.save_benchmark_report(bench_rep)
        s.commit()

        assert r1.id == r2.id
        assert repo.count_benchmark_runs() == 1


# 17. Filtering and Pagination
def test_repository_filtering_and_pagination(test_db):
    _, session_factory = test_db
    with session_factory() as s:
        repo = BenchmarkRepository(session=s)
        # Create 5 benchmark runs across 2 models
        for i in range(5):
            fp = "model_A_fp" if i < 3 else "model_B_fp"
            stat = "SUCCESS" if i % 2 == 0 else "FAILED"
            rep = BenchmarkReport(
                run_id=f"run-pag-{i}",
                timestamp_utc=f"2026-10-17T14:0{i}:00Z",
                model_name=f"model_{i}.onnx",
                model_fingerprint=fp,
                status=stat,
                error_message=None if stat == "SUCCESS" else "Error",
                environment={},
                config={"batch_size": 1, "intra_op_num_threads": 1, "execution_provider": "CPUExecutionProvider"},
                input_spec={},
                correctness={"passed": stat == "SUCCESS", "details": ""},
                latency={"median_ms": 0.01 * (i + 1)},
                throughput={},
                memory={},
                per_round_results=[],
                all_latencies_ms=[],
                warnings=[],
                limitations=[],
            )
            repo.save_benchmark_report(rep)
        s.commit()

        # Test model filter
        runs_a, total_a = repo.list_benchmark_runs(fingerprint="model_A_fp")
        assert total_a == 3
        assert len(runs_a) == 3

        # Test status filter
        runs_failed, total_failed = repo.list_benchmark_runs(status="FAILED")
        assert total_failed == 2

        # Test pagination (limit 2, offset 0)
        runs_p1, _ = repo.list_benchmark_runs(limit=2, offset=0)
        assert len(runs_p1) == 2


# 18. Benchmark Comparison Compatibility Checks
def test_benchmark_comparison_disqualifies_incompatible_runs():
    run_a = BenchmarkRunORM(
        id="run-A",
        model_fingerprint="fp_model_one",
        model_name="model_one.onnx",
        execution_provider="CPUExecutionProvider",
        configuration={"batch_size": 1, "intra_op_num_threads": 1},
        input_specifications={"resolved_shapes": {"X": [1, 10]}},
        runtime_environment={},
        status="SUCCESS",
        correctness_status="PASS",
        median_latency_ms=0.02,
        inferences_per_second=50000.0,
    )

    # Incompatible Model Fingerprint
    run_b_diff_model = BenchmarkRunORM(
        id="run-B",
        model_fingerprint="fp_model_two",  # Different fingerprint!
        model_name="model_two.onnx",
        execution_provider="CPUExecutionProvider",
        configuration={"batch_size": 1, "intra_op_num_threads": 2},
        input_specifications={"resolved_shapes": {"X": [1, 10]}},
        runtime_environment={},
        status="SUCCESS",
        correctness_status="PASS",
        median_latency_ms=0.01,
        inferences_per_second=100000.0,
    )

    cmp_diff_model = compare_benchmark_runs(run_a, run_b_diff_model)
    assert cmp_diff_model.is_valid is False
    assert cmp_diff_model.validity == ComparisonValidity.INCOMPATIBLE_MODELS
    assert any("fingerprints differ" in r for r in cmp_diff_model.disqualification_reasons)

    # Incompatible Batch Size
    run_c_diff_batch = BenchmarkRunORM(
        id="run-C",
        model_fingerprint="fp_model_one",
        model_name="model_one.onnx",
        execution_provider="CPUExecutionProvider",
        configuration={"batch_size": 4, "intra_op_num_threads": 1},  # Different batch size!
        input_specifications={"resolved_shapes": {"X": [4, 10]}},
        runtime_environment={},
        status="SUCCESS",
        correctness_status="PASS",
        median_latency_ms=0.04,
        inferences_per_second=25000.0,
    )

    cmp_diff_batch = compare_benchmark_runs(run_a, run_c_diff_batch)
    assert cmp_diff_batch.is_valid is False
    assert cmp_diff_batch.validity == ComparisonValidity.INCOMPATIBLE_WORKLOAD
    assert any("batch sizes differ" in r for r in cmp_diff_batch.disqualification_reasons)

    # Valid Comparison
    run_d_compatible = BenchmarkRunORM(
        id="run-D",
        model_fingerprint="fp_model_one",
        model_name="model_one.onnx",
        execution_provider="CPUExecutionProvider",
        configuration={"batch_size": 1, "intra_op_num_threads": 4},
        input_specifications={"resolved_shapes": {"X": [1, 10]}},
        runtime_environment={},
        status="SUCCESS",
        correctness_status="PASS",
        median_latency_ms=0.01,
        inferences_per_second=100000.0,
    )

    cmp_valid = compare_benchmark_runs(run_a, run_d_compatible)
    assert cmp_valid.is_valid is True
    assert cmp_valid.validity == ComparisonValidity.VALID
    assert len(cmp_valid.disqualification_reasons) == 0
    assert "median_latency" in cmp_valid.metric_comparisons
    assert cmp_valid.metric_comparisons["median_latency"].speedup_factor == 2.0
    assert cmp_valid.metric_comparisons["median_latency"].percentage_improvement == 50.0


# 19. Historical Retrieval after Simulated Application Restart
def test_historical_retrieval_after_simulated_restart(tmp_path, sample_model_summary):
    db_file = tmp_path / "restart_test.db"
    db_url = f"sqlite:///{db_file}"

    # First session: insert data and close engine
    engine1 = create_engine(db_url)
    Base.metadata.create_all(bind=engine1)
    factory1 = get_session_factory(engine1)
    with factory1() as s:
        repo = ModelRepository(session=s)
        repo.register_or_update(sample_model_summary)
        s.commit()
    engine1.dispose()

    # Second session: simulate application restart with completely fresh engine
    engine2 = create_engine(db_url)
    factory2 = get_session_factory(engine2)
    with factory2() as s:
        repo = ModelRepository(session=s)
        retrieved = repo.get_by_fingerprint(sample_model_summary.sha256_hash)
        assert retrieved is not None
        assert retrieved.original_filename == "test_linear.onnx"
        assert retrieved.file_size_bytes == 4096
    engine2.dispose()


# 20. Schema and Report Version Preservation
def test_schema_and_report_version_preservation(test_db):
    _, session_factory = test_db
    with session_factory() as s:
        b_repo = BenchmarkRepository(session=s)
        rep = BenchmarkReport(
            run_id="run-ver-001",
            timestamp_utc="2026-10-17T14:00:00Z",
            model_name="test.onnx",
            model_fingerprint="fp_ver_123",
            status="SUCCESS",
            error_message=None,
            environment={},
            config={"batch_size": 1, "intra_op_num_threads": 1, "execution_provider": "CPUExecutionProvider"},
            input_spec={},
            correctness={"passed": True, "details": "ok"},
            latency={"median_ms": 0.01},
            throughput={},
            memory={},
            per_round_results=[],
            all_latencies_ms=[],
            warnings=[],
            limitations=[],
        )
        saved = b_repo.save_benchmark_report(rep)
        s.commit()

        assert saved.schema_version == "1.0.0"


# 21. Empty Database State Handling
def test_empty_database_state_handling(test_db):
    _, session_factory = test_db
    with session_factory() as s:
        overview_repo = HistoryOverviewRepository(session=s)
        overview = overview_repo.get_overview_metrics()

        assert overview["total_models"] == 0
        assert overview["total_experiments"] == 0
        assert overview["total_benchmarks"] == 0
        assert overview["recent_experiments"] == []
        assert overview["recent_benchmarks"] == []


# 22. Regression Tests for Phases 1–5 Integration
def test_regression_phases_1_to_5_integration(test_db, sample_model_summary):
    from zmodel.core.compatibility import CompatibilityAnalyzer
    from zmodel.core.profiles import load_builtin_profiles

    # Check profiles load cleanly
    profiles = load_builtin_profiles()
    assert "local_onnx_runtime_cpu" in profiles

    # Run compatibility analysis
    compat_report = CompatibilityAnalyzer.analyze(sample_model_summary, profiles["local_onnx_runtime_cpu"])
    assert compat_report.overall_assessment in ("COMPATIBLE", "CONDITIONAL")

    # Persist compatibility report
    _, session_factory = test_db
    with session_factory() as s:
        compat_repo = CompatibilityRepository(session=s)
        saved_compat = compat_repo.save_assessment(compat_report)
        s.commit()
        assert saved_compat.id is not None
