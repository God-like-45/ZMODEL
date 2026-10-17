"""SQLAlchemy 2.x ORM models for normalized PostgreSQL experiment history."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class ModelIdentityORM(Base):
    """
    Durable registry of unique ONNX model identities.
    Keyed strictly by SHA-256 fingerprint; models with identical filenames but
    differing fingerprints remain distinct records.
    Raw model bytes are never stored in the database.
    """
    __tablename__ = "models"

    fingerprint = Column(String(64), primary_key=True, doc="SHA-256 hex string")
    original_filename = Column(String(255), nullable=False, index=True)
    file_size_bytes = Column(BigInteger, nullable=False)
    ir_version = Column(Integer, nullable=True)
    producer_name = Column(String(128), nullable=True)
    producer_version = Column(String(64), nullable=True)
    opset_imports = Column(JSON, nullable=True, doc="Dictionary or list of opset imports")
    graph_metadata = Column(JSON, nullable=True, doc="Node counts, parameter counts, tensor signatures")
    first_seen_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    last_seen_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)

    # Relationships
    compatibility_assessments = relationship(
        "CompatibilityAssessmentORM", back_populates="model", cascade="all, delete-orphan"
    )
    benchmark_runs = relationship(
        "BenchmarkRunORM", back_populates="model", cascade="all, delete-orphan"
    )
    optimization_experiments = relationship(
        "OptimizationExperimentORM", back_populates="model", cascade="all, delete-orphan"
    )
    graph_optimization_experiments = relationship(
        "GraphOptimizationExperimentORM", back_populates="model", cascade="all, delete-orphan"
    )
    worker_jobs = relationship(
        "WorkerJobORM", back_populates="model", cascade="all, delete-orphan"
    )
    validation_runs = relationship(
        "ValidationRunORM", back_populates="model", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "original_filename": self.original_filename,
            "file_size_bytes": self.file_size_bytes,
            "ir_version": self.ir_version,
            "producer_name": self.producer_name,
            "producer_version": self.producer_version,
            "opset_imports": self.opset_imports,
            "graph_metadata": self.graph_metadata,
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
        }


class CompatibilityAssessmentORM(Base):
    """Historical record of an evidence-based compatibility assessment against a versioned profile."""
    __tablename__ = "compatibility_assessments"

    id = Column(String(64), primary_key=True)
    model_fingerprint = Column(String(64), ForeignKey("models.fingerprint"), nullable=False, index=True)
    profile_id = Column(String(64), nullable=False, index=True)
    profile_name = Column(String(128), nullable=False)
    profile_version = Column(String(32), nullable=False)
    ruleset_version = Column(String(32), nullable=False)
    overall_assessment = Column(String(32), nullable=False, index=True)
    summary_counts = Column(JSON, nullable=False)
    findings = Column(JSON, nullable=False)
    assumptions = Column(JSON, nullable=True)
    unresolved_questions = Column(JSON, nullable=True)
    recommended_next_steps = Column(JSON, nullable=True)
    is_illustrative_profile = Column(Boolean, default=False, nullable=False)
    schema_version = Column(String(16), default="1.0.0", nullable=False)
    assessed_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)

    # Relationship
    model = relationship("ModelIdentityORM", back_populates="compatibility_assessments")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_fingerprint": self.model_fingerprint,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "ruleset_version": self.ruleset_version,
            "overall_assessment": self.overall_assessment,
            "summary_counts": self.summary_counts,
            "findings": self.findings,
            "assumptions": self.assumptions,
            "unresolved_questions": self.unresolved_questions,
            "recommended_next_steps": self.recommended_next_steps,
            "is_illustrative_profile": self.is_illustrative_profile,
            "schema_version": self.schema_version,
            "assessed_at": self.assessed_at.isoformat() if self.assessed_at else None,
        }


class BenchmarkRunORM(Base):
    """Empirical benchmark execution record capturing actual ONNX Runtime performance."""
    __tablename__ = "benchmark_runs"

    id = Column(String(64), primary_key=True)
    model_fingerprint = Column(String(64), ForeignKey("models.fingerprint"), nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    execution_provider = Column(String(64), nullable=False, index=True)
    configuration = Column(JSON, nullable=False)
    input_specifications = Column(JSON, nullable=False)
    runtime_environment = Column(JSON, nullable=False)
    status = Column(String(32), nullable=False, index=True)  # SUCCESS, FAILED, CANCELLED
    error_message = Column(Text, nullable=True)
    correctness_status = Column(String(32), nullable=False)  # PASS, FAIL, UNKNOWN
    correctness_details = Column(Text, nullable=True)

    # Core Latency Metrics (milliseconds)
    median_latency_ms = Column(Float, nullable=True)
    mean_latency_ms = Column(Float, nullable=True)
    min_latency_ms = Column(Float, nullable=True)
    max_latency_ms = Column(Float, nullable=True)
    p90_latency_ms = Column(Float, nullable=True)
    p95_latency_ms = Column(Float, nullable=True)
    p99_latency_ms = Column(Float, nullable=True)
    std_dev_latency_ms = Column(Float, nullable=True)
    sample_count = Column(Integer, nullable=True)

    # Throughput Metrics
    inferences_per_second = Column(Float, nullable=True)
    samples_per_second = Column(Float, nullable=True)

    # Memory Metrics (Megabytes)
    session_delta_mb = Column(Float, nullable=True)
    peak_rss_mb = Column(Float, nullable=True)

    # Granular round data & limitations
    per_round_results = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)
    limitations = Column(JSON, nullable=True)
    schema_version = Column(String(16), default="1.0.0", nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship
    model = relationship("ModelIdentityORM", back_populates="benchmark_runs")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_fingerprint": self.model_fingerprint,
            "model_name": self.model_name,
            "execution_provider": self.execution_provider,
            "configuration": self.configuration,
            "input_specifications": self.input_specifications,
            "runtime_environment": self.runtime_environment,
            "status": self.status,
            "error_message": self.error_message,
            "correctness_status": self.correctness_status,
            "correctness_details": self.correctness_details,
            "median_latency_ms": self.median_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p90_latency_ms": self.p90_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "std_dev_latency_ms": self.std_dev_latency_ms,
            "sample_count": self.sample_count,
            "inferences_per_second": self.inferences_per_second,
            "samples_per_second": self.samples_per_second,
            "session_delta_mb": self.session_delta_mb,
            "peak_rss_mb": self.peak_rss_mb,
            "per_round_results": self.per_round_results,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class OptimizationExperimentORM(Base):
    """Durable record of an optimization search across multiple execution candidates."""
    __tablename__ = "optimization_experiments"

    id = Column(String(64), primary_key=True)
    model_fingerprint = Column(String(64), ForeignKey("models.fingerprint"), nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    profile_id = Column(String(64), nullable=False, index=True)
    profile_name = Column(String(128), nullable=False)
    profile_version = Column(String(32), nullable=False)
    primary_objective = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, index=True)
    request_payload = Column(JSON, nullable=False)
    baseline_candidate_id = Column(String(64), nullable=True)
    best_candidate_id = Column(String(64), nullable=True)
    summary_counts = Column(JSON, nullable=False)
    recommendation_rationale = Column(Text, nullable=True)
    pareto_frontier_ids = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)
    limitations = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    environment = Column(JSON, nullable=True)
    schema_version = Column(String(16), default="1.0.0", nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    model = relationship("ModelIdentityORM", back_populates="optimization_experiments")
    candidates = relationship(
        "CandidateEvaluationORM",
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="CandidateEvaluationORM.rank",
        lazy="selectin",
    )

    def to_dict(self, candidates: Optional[List[Any]] = None) -> Dict[str, Any]:
        cands = candidates
        if cands is None:
            try:
                cands = self.candidates
            except Exception:
                cands = []
        return {
            "id": self.id,
            "model_fingerprint": self.model_fingerprint,
            "model_name": self.model_name,
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "primary_objective": self.primary_objective,
            "status": self.status,
            "request_payload": self.request_payload,
            "baseline_candidate_id": self.baseline_candidate_id,
            "best_candidate_id": self.best_candidate_id,
            "summary_counts": self.summary_counts,
            "recommendation_rationale": self.recommendation_rationale,
            "pareto_frontier_ids": self.pareto_frontier_ids,
            "warnings": self.warnings,
            "limitations": self.limitations,
            "error_message": self.error_message,
            "environment": self.environment,
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "candidates": [c.to_dict() for c in cands] if cands else [],
        }


class CandidateEvaluationORM(Base):
    """Granular execution candidate evaluation belonging to an optimization experiment."""
    __tablename__ = "candidate_evaluations"

    id = Column(String(64), primary_key=True)
    experiment_id = Column(
        String(64),
        ForeignKey("optimization_experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id = Column(String(64), nullable=False)
    benchmark_run_id = Column(String(64), ForeignKey("benchmark_runs.id"), nullable=True, index=True)
    configuration = Column(JSON, nullable=False)
    generation_reason = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, index=True)
    rejection_reason = Column(Text, nullable=True)
    compatibility_findings = Column(JSON, nullable=True)
    constraint_evaluations = Column(JSON, nullable=True)
    is_feasible = Column(Boolean, default=False, nullable=False)
    is_pareto_optimal = Column(Boolean, default=False, nullable=False)
    is_baseline = Column(Boolean, default=False, nullable=False)
    rank = Column(Integer, nullable=True)

    # Measured Performance Snapshots
    measured_median_latency_ms = Column(Float, nullable=True)
    measured_throughput_ips = Column(Float, nullable=True)
    measured_memory_mb = Column(Float, nullable=True)
    speedup_vs_baseline = Column(Float, nullable=True)
    latency_reduction_pct = Column(Float, nullable=True)
    throughput_improvement_pct = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    # Relationships
    experiment = relationship("OptimizationExperimentORM", back_populates="candidates")
    benchmark_run = relationship("BenchmarkRunORM")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "benchmark_run_id": self.benchmark_run_id,
            "configuration": self.configuration,
            "generation_reason": self.generation_reason,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "compatibility_findings": self.compatibility_findings,
            "constraint_evaluations": self.constraint_evaluations,
            "is_feasible": self.is_feasible,
            "is_pareto_optimal": self.is_pareto_optimal,
            "is_baseline": self.is_baseline,
            "rank": self.rank,
            "measured_median_latency_ms": self.measured_median_latency_ms,
            "measured_throughput_ips": self.measured_throughput_ips,
            "measured_memory_mb": self.measured_memory_mb,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "latency_reduction_pct": self.latency_reduction_pct,
            "throughput_improvement_pct": self.throughput_improvement_pct,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class GraphOptimizationExperimentORM(Base):
    """
    Durable record of an ONNX graph transformation experiment.
    Tracks transformation candidates, validation results, compatibility findings,
    and empirical benchmark comparisons without storing model binaries.
    """
    __tablename__ = "graph_optimization_experiments"

    id = Column(String(64), primary_key=True)
    model_fingerprint = Column(String(64), ForeignKey("models.fingerprint"), nullable=False, index=True)
    model_name = Column(String(255), nullable=False)
    target_profile_id = Column(String(64), nullable=True, index=True)
    target_profile_name = Column(String(128), nullable=True)
    validation_depth = Column(String(32), default="FULL", nullable=False)
    benchmark_enabled = Column(Boolean, default=True, nullable=False)
    status = Column(String(32), nullable=False, index=True)  # SUCCESS, FAILED, RUNNING
    best_candidate_id = Column(String(64), nullable=True)
    summary_counts = Column(JSON, nullable=False)
    overall_recommendation = Column(Text, nullable=True)
    limitations = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    schema_version = Column(String(16), default="1.0.0", nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    model = relationship("ModelIdentityORM", back_populates="graph_optimization_experiments")
    candidates = relationship(
        "GraphTransformationCandidateORM",
        back_populates="experiment",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_dict(self, candidates: Optional[List[Any]] = None) -> Dict[str, Any]:
        cands = candidates
        if cands is None:
            try:
                cands = self.candidates
            except Exception:
                cands = []
        return {
            "id": self.id,
            "model_fingerprint": self.model_fingerprint,
            "model_name": self.model_name,
            "target_profile_id": self.target_profile_id,
            "target_profile_name": self.target_profile_name,
            "validation_depth": self.validation_depth,
            "benchmark_enabled": self.benchmark_enabled,
            "status": self.status,
            "best_candidate_id": self.best_candidate_id,
            "summary_counts": self.summary_counts,
            "overall_recommendation": self.overall_recommendation,
            "limitations": self.limitations,
            "error_message": self.error_message,
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "candidates": [c.to_dict() for c in cands] if cands else [],
        }


class GraphTransformationCandidateORM(Base):
    """
    Evaluation record for an individual graph transformation candidate.
    Links to stored artifact metadata via artifact_id and transformed_hash.
    """
    __tablename__ = "graph_transformation_candidates"

    id = Column(String(64), primary_key=True)
    experiment_id = Column(
        String(64),
        ForeignKey("graph_optimization_experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    candidate_id = Column(String(64), nullable=False)
    transform_type = Column(String(64), nullable=False)
    transform_name = Column(String(128), nullable=False)
    tool_name = Column(String(64), nullable=False)
    tool_version = Column(String(32), nullable=False)
    state = Column(String(32), nullable=False, index=True)
    original_hash = Column(String(64), nullable=False)
    transformed_hash = Column(String(64), nullable=True, index=True)
    artifact_id = Column(String(128), nullable=True)
    graph_changed = Column(Boolean, default=False, nullable=False)
    model_size_bytes_before = Column(BigInteger, nullable=False)
    model_size_bytes_after = Column(BigInteger, nullable=True)
    size_reduction_pct = Column(Float, nullable=True)
    diff_summary = Column(JSON, nullable=True)
    validation_report = Column(JSON, nullable=True)
    compatibility_report = Column(JSON, nullable=True)
    compatibility_delta = Column(JSON, nullable=True)
    benchmark_report = Column(JSON, nullable=True)
    speedup_factor = Column(Float, nullable=True)
    latency_reduction_pct = Column(Float, nullable=True)
    throughput_improvement_pct = Column(Float, nullable=True)
    recommendation_status = Column(String(32), nullable=False, index=True)
    recommendation_rationale = Column(Text, nullable=True)
    rejection_reason = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)
    duration_ms = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)

    # Relationship
    experiment = relationship("GraphOptimizationExperimentORM", back_populates="candidates")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "candidate_id": self.candidate_id,
            "transform_type": self.transform_type,
            "transform_name": self.transform_name,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "state": self.state,
            "original_hash": self.original_hash,
            "transformed_hash": self.transformed_hash,
            "artifact_id": self.artifact_id,
            "graph_changed": self.graph_changed,
            "model_size_bytes_before": self.model_size_bytes_before,
            "model_size_bytes_after": self.model_size_bytes_after,
            "size_reduction_pct": self.size_reduction_pct,
            "diff_summary": self.diff_summary,
            "validation_report": self.validation_report,
            "compatibility_report": self.compatibility_report,
            "compatibility_delta": self.compatibility_delta,
            "benchmark_report": self.benchmark_report,
            "speedup_factor": self.speedup_factor,
            "latency_reduction_pct": self.latency_reduction_pct,
            "throughput_improvement_pct": self.throughput_improvement_pct,
            "recommendation_status": self.recommendation_status,
            "recommendation_rationale": self.recommendation_rationale,
            "rejection_reason": self.rejection_reason,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class WorkerJobORM(Base):
    """
    Durable record tracking the full lifecycle of an asynchronous worker job.
    Captures execution backend ('local' or 'kubernetes'), resource allocation profile,
    Kubernetes metadata, timing breakdowns, and final execution metrics.
    """
    __tablename__ = "worker_jobs"

    id = Column(String(64), primary_key=True)
    job_type = Column(String(32), nullable=False, index=True)
    execution_backend = Column(String(32), nullable=False, default="local", index=True)
    model_fingerprint = Column(
        String(64),
        ForeignKey("models.fingerprint", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    experiment_id = Column(String(64), nullable=True, index=True)
    run_id = Column(String(64), nullable=True, index=True)
    status = Column(String(32), nullable=False, default="QUEUED", index=True)
    resource_profile = Column(String(32), nullable=False, default="standard_cpu")
    k8s_namespace = Column(String(64), nullable=True)
    k8s_job_name = Column(String(64), nullable=True)
    k8s_pod_name = Column(String(64), nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=1, nullable=False)
    worker_image = Column(String(128), nullable=True)
    queue_delay_seconds = Column(Float, default=0.0, nullable=False)
    startup_overhead_seconds = Column(Float, default=0.0, nullable=False)
    warmup_overhead_seconds = Column(Float, default=0.0, nullable=False)
    inference_duration_seconds = Column(Float, default=0.0, nullable=False)
    total_duration_seconds = Column(Float, default=0.0, nullable=False)
    spec_json = Column(JSON, nullable=True)
    result_json = Column(JSON, nullable=True)
    output_artifact_ids = Column(JSON, nullable=True)
    error_category = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationship
    model = relationship("ModelIdentityORM", back_populates="worker_jobs")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "execution_backend": self.execution_backend,
            "model_fingerprint": self.model_fingerprint,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "resource_profile": self.resource_profile,
            "k8s_namespace": self.k8s_namespace,
            "k8s_job_name": self.k8s_job_name,
            "k8s_pod_name": self.k8s_pod_name,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "worker_image": self.worker_image,
            "queue_delay_seconds": self.queue_delay_seconds,
            "startup_overhead_seconds": self.startup_overhead_seconds,
            "warmup_overhead_seconds": self.warmup_overhead_seconds,
            "inference_duration_seconds": self.inference_duration_seconds,
            "total_duration_seconds": self.total_duration_seconds,
            "spec_json": self.spec_json,
            "result_json": self.result_json,
            "output_artifact_ids": self.output_artifact_ids,
            "error_category": self.error_category,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ValidationRunORM(Base):
    """
    Durable record of an IBM Z / Linux on IBM Z validation run.
    Persists evaluated checks, detected environment facts, numerical precision, and audit evidence.
    """
    __tablename__ = "validation_runs"

    id = Column(String(128), primary_key=True, doc="Unique run identifier, e.g. VAL-20261018-0001")
    model_fingerprint = Column(
        String(64),
        ForeignKey("models.fingerprint", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_profile_id = Column(String(64), nullable=False, index=True)
    target_profile_version = Column(String(32), nullable=False)
    environment_fingerprint = Column(String(64), nullable=False, index=True)
    execution_mode = Column(String(32), nullable=False, default="LOCAL")
    overall_status = Column(String(32), nullable=False, index=True)
    status_rationale = Column(Text, nullable=True)
    provenance_badge = Column(String(128), nullable=True)
    benchmark_eligibility = Column(Boolean, nullable=False, default=False)
    summary_counts = Column(JSON, nullable=False)
    checks_json = Column(JSON, nullable=False)
    numerical_correctness = Column(JSON, nullable=True)
    environment_info = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)

    # Relationship
    model = relationship("ModelIdentityORM", back_populates="validation_runs")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_fingerprint": self.model_fingerprint,
            "target_profile_id": self.target_profile_id,
            "target_profile_version": self.target_profile_version,
            "environment_fingerprint": self.environment_fingerprint,
            "execution_mode": self.execution_mode,
            "overall_status": self.overall_status,
            "status_rationale": self.status_rationale,
            "provenance_badge": self.provenance_badge,
            "benchmark_eligibility": self.benchmark_eligibility,
            "summary_counts": self.summary_counts,
            "checks": self.checks_json,
            "numerical_correctness": self.numerical_correctness,
            "environment": self.environment_info,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }



