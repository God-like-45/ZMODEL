"""Phase 9 schema for ONNX graph optimization experiments and transformation candidates.

Revision ID: 002_graph_optimization_schema
Revises: 001_initial_schema
Create Date: 2026-10-18 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "002_graph_optimization_schema"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Graph Optimization Experiments Table
    op.create_table(
        "graph_optimization_experiments",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("model_fingerprint", sa.String(64), sa.ForeignKey("models.fingerprint"), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("target_profile_id", sa.String(64), nullable=True),
        sa.Column("target_profile_name", sa.String(128), nullable=True),
        sa.Column("validation_depth", sa.String(32), default="FULL", nullable=False),
        sa.Column("benchmark_enabled", sa.Boolean(), default=True, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("best_candidate_id", sa.String(64), nullable=True),
        sa.Column("summary_counts", sa.JSON(), nullable=False),
        sa.Column("overall_recommendation", sa.Text(), nullable=True),
        sa.Column("limitations", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("schema_version", sa.String(16), default="1.0.0", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_gopt_exp_model_fingerprint",
        "graph_optimization_experiments",
        ["model_fingerprint"],
    )
    op.create_index(
        "ix_gopt_exp_target_profile_id",
        "graph_optimization_experiments",
        ["target_profile_id"],
    )
    op.create_index(
        "ix_gopt_exp_status",
        "graph_optimization_experiments",
        ["status"],
    )
    op.create_index(
        "ix_gopt_exp_created_at",
        "graph_optimization_experiments",
        ["created_at"],
    )

    # 2. Graph Transformation Candidates Table
    op.create_table(
        "graph_transformation_candidates",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "experiment_id",
            sa.String(64),
            sa.ForeignKey("graph_optimization_experiments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("candidate_id", sa.String(64), nullable=False),
        sa.Column("transform_type", sa.String(64), nullable=False),
        sa.Column("transform_name", sa.String(128), nullable=False),
        sa.Column("tool_name", sa.String(64), nullable=False),
        sa.Column("tool_version", sa.String(32), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("original_hash", sa.String(64), nullable=False),
        sa.Column("transformed_hash", sa.String(64), nullable=True),
        sa.Column("artifact_id", sa.String(128), nullable=True),
        sa.Column("graph_changed", sa.Boolean(), default=False, nullable=False),
        sa.Column("model_size_bytes_before", sa.BigInteger(), nullable=False),
        sa.Column("model_size_bytes_after", sa.BigInteger(), nullable=True),
        sa.Column("size_reduction_pct", sa.Float(), nullable=True),
        sa.Column("diff_summary", sa.JSON(), nullable=True),
        sa.Column("validation_report", sa.JSON(), nullable=True),
        sa.Column("compatibility_report", sa.JSON(), nullable=True),
        sa.Column("compatibility_delta", sa.JSON(), nullable=True),
        sa.Column("benchmark_report", sa.JSON(), nullable=True),
        sa.Column("speedup_factor", sa.Float(), nullable=True),
        sa.Column("latency_reduction_pct", sa.Float(), nullable=True),
        sa.Column("throughput_improvement_pct", sa.Float(), nullable=True),
        sa.Column("recommendation_status", sa.String(32), nullable=False),
        sa.Column("recommendation_rationale", sa.Text(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("warnings", sa.JSON(), nullable=True),
        sa.Column("duration_ms", sa.Float(), default=0.0, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_gopt_cand_experiment_id",
        "graph_transformation_candidates",
        ["experiment_id"],
    )
    op.create_index(
        "ix_gopt_cand_state",
        "graph_transformation_candidates",
        ["state"],
    )
    op.create_index(
        "ix_gopt_cand_transformed_hash",
        "graph_transformation_candidates",
        ["transformed_hash"],
    )
    op.create_index(
        "ix_gopt_cand_recommendation_status",
        "graph_transformation_candidates",
        ["recommendation_status"],
    )


def downgrade() -> None:
    op.drop_table("graph_transformation_candidates")
    op.drop_table("graph_optimization_experiments")
