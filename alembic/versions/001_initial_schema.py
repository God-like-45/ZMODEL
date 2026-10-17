"""Initial Phase 6 schema for models, assessments, benchmarks, and optimization experiments.

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-17 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Models Table
    op.create_table(
        "models",
        sa.Column("fingerprint", sa.String(64), primary_key=True),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("ir_version", sa.Integer(), nullable=True),
        sa.Column("producer_name", sa.String(128), nullable=True),
        sa.Column("producer_version", sa.String(64), nullable=True),
        sa.Column("opset_imports", sa.JSON(), nullable=True),
        sa.Column("graph_metadata", sa.JSON(), nullable=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_models_original_filename", "models", ["original_filename"])

    # 2. Compatibility Assessments Table
    op.create_table(
        "compatibility_assessments",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("model_fingerprint", sa.String(64), sa.ForeignKey("models.fingerprint"), nullable=False),
        sa.Column("profile_id", sa.String(64), nullable=False),
        sa.Column("profile_name", sa.String(128), nullable=False),
        sa.Column("profile_version", sa.String(32), nullable=False),
        sa.Column("ruleset_version", sa.String(32), nullable=False),
        sa.Column("overall_assessment", sa.String(32), nullable=False),
        sa.Column("summary_counts", sa.JSON(), nullable=False),
        sa.Column("findings", sa.JSON(), nullable=False),
        sa.Column("assumptions", sa.JSON(), nullable=True),
        sa.Column("unresolved_questions", sa.JSON(), nullable=True),
        sa.Column("recommended_next_steps", sa.JSON(), nullable=True),
        sa.Column("is_illustrative_profile", sa.Boolean(), default=False, nullable=False),
        sa.Column("schema_version", sa.String(16), default="1.0.0", nullable=False),
        sa.Column("assessed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_compat_model_fingerprint", "compatibility_assessments", ["model_fingerprint"])
    op.create_index("ix_compat_profile_id", "compatibility_assessments", ["profile_id"])
    op.create_index("ix_compat_overall_assessment", "compatibility_assessments", ["overall_assessment"])
    op.create_index("ix_compat_assessed_at", "compatibility_assessments", ["assessed_at"])

    # 3. Benchmark Runs Table
    op.create_table(
        "benchmark_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("model_fingerprint", sa.String(64), sa.ForeignKey("models.fingerprint"), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("execution_provider", sa.String(64), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("input_specifications", sa.JSON(), nullable=False),
        sa.Column("runtime_environment", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("correctness_status", sa.String(32), nullable=False),
        sa.Column("correctness_details", sa.Text(), nullable=True),
        sa.Column("median_latency_ms", sa.Float(), nullable=True),
        sa.Column("mean_latency_ms", sa.Float(), nullable=True),
        sa.Column("min_latency_ms", sa.Float(), nullable=True),
        sa.Column("max_latency_ms", sa.Float(), nullable=True),
        sa.Column("p90_latency_ms", sa.Float(), nullable=True),
        sa.Column("p95_latency_ms", sa.Float(), nullable=True),
        sa.Column("p99_latency_ms", sa.Float(), nullable=True),
        sa.Column("std_dev_latency_ms", sa.Float(), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=True),
        sa.Column("inferences_per_second", sa.Float(), nullable=True),
        sa.Column("samples_per_second", sa.Float(), nullable=True),
        sa.Column("session_delta_mb", sa.Float(), nullable=True),
        sa.Column("peak_rss_mb", sa.Float(), nullable=True),
        sa.Column("per_round_results", sa.JSON(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=True),
        sa.Column("limitations", sa.JSON(), nullable=True),
        sa.Column("schema_version", sa.String(16), default="1.0.0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_benchmark_runs_model_fingerprint", "benchmark_runs", ["model_fingerprint"])
    op.create_index("ix_benchmark_runs_provider", "benchmark_runs", ["execution_provider"])
    op.create_index("ix_benchmark_runs_status", "benchmark_runs", ["status"])
    op.create_index("ix_benchmark_runs_created_at", "benchmark_runs", ["created_at"])

    # 4. Optimization Experiments Table
    op.create_table(
        "optimization_experiments",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("model_fingerprint", sa.String(64), sa.ForeignKey("models.fingerprint"), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("profile_id", sa.String(64), nullable=False),
        sa.Column("profile_name", sa.String(128), nullable=False),
        sa.Column("profile_version", sa.String(32), nullable=False),
        sa.Column("primary_objective", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("request_payload", sa.JSON(), nullable=False),
        sa.Column("baseline_candidate_id", sa.String(64), nullable=True),
        sa.Column("best_candidate_id", sa.String(64), nullable=True),
        sa.Column("summary_counts", sa.JSON(), nullable=False),
        sa.Column("recommendation_rationale", sa.Text(), nullable=True),
        sa.Column("pareto_frontier_ids", sa.JSON(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=True),
        sa.Column("limitations", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("environment", sa.JSON(), nullable=True),
        sa.Column("schema_version", sa.String(16), default="1.0.0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_opt_experiments_model_fingerprint", "optimization_experiments", ["model_fingerprint"])
    op.create_index("ix_opt_experiments_profile_id", "optimization_experiments", ["profile_id"])
    op.create_index("ix_opt_experiments_objective", "optimization_experiments", ["primary_objective"])
    op.create_index("ix_opt_experiments_status", "optimization_experiments", ["status"])
    op.create_index("ix_opt_experiments_created_at", "optimization_experiments", ["created_at"])

    # 5. Candidate Evaluations Table
    op.create_table(
        "candidate_evaluations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.String(64),
            sa.ForeignKey("optimization_experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("candidate_id", sa.String(64), nullable=False),
        sa.Column("benchmark_run_id", sa.String(64), sa.ForeignKey("benchmark_runs.id"), nullable=True),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("generation_reason", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("compatibility_findings", sa.JSON(), nullable=True),
        sa.Column("constraint_evaluations", sa.JSON(), nullable=True),
        sa.Column("is_feasible", sa.Boolean(), default=False, nullable=False),
        sa.Column("is_pareto_optimal", sa.Boolean(), default=False, nullable=False),
        sa.Column("is_baseline", sa.Boolean(), default=False, nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("measured_median_latency_ms", sa.Float(), nullable=True),
        sa.Column("measured_throughput_ips", sa.Float(), nullable=True),
        sa.Column("measured_memory_mb", sa.Float(), nullable=True),
        sa.Column("speedup_vs_baseline", sa.Float(), nullable=True),
        sa.Column("latency_reduction_pct", sa.Float(), nullable=True),
        sa.Column("throughput_improvement_pct", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_candidates_experiment_id", "candidate_evaluations", ["experiment_id"])
    op.create_index("ix_candidates_benchmark_run_id", "candidate_evaluations", ["benchmark_run_id"])
    op.create_index("ix_candidates_status", "candidate_evaluations", ["status"])


def downgrade() -> None:
    op.drop_table("candidate_evaluations")
    op.drop_table("optimization_experiments")
    op.drop_table("benchmark_runs")
    op.drop_table("compatibility_assessments")
    op.drop_table("models")
