"""Phase 11 schema for IBM Z / Linux on IBM Z validation runs.

Revision ID: 004_validation_runs_schema
Revises: 003_worker_jobs_schema
Create Date: 2026-10-18 02:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "004_validation_runs_schema"
down_revision: Union[str, None] = "003_worker_jobs_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "validation_runs",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column(
            "model_fingerprint",
            sa.String(64),
            sa.ForeignKey("models.fingerprint", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("target_profile_id", sa.String(64), nullable=False),
        sa.Column("target_profile_version", sa.String(32), nullable=False),
        sa.Column("environment_fingerprint", sa.String(64), nullable=False),
        sa.Column("execution_mode", sa.String(32), server_default="LOCAL", nullable=False),
        sa.Column("overall_status", sa.String(32), nullable=False),
        sa.Column("status_rationale", sa.Text(), nullable=True),
        sa.Column("provenance_badge", sa.String(128), nullable=True),
        sa.Column("benchmark_eligibility", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("summary_counts", sa.JSON(), nullable=False),
        sa.Column("checks_json", sa.JSON(), nullable=False),
        sa.Column("numerical_correctness", sa.JSON(), nullable=True),
        sa.Column("environment_info", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_index("ix_validation_runs_model_fingerprint", "validation_runs", ["model_fingerprint"])
    op.create_index("ix_validation_runs_target_profile_id", "validation_runs", ["target_profile_id"])
    op.create_index("ix_validation_runs_overall_status", "validation_runs", ["overall_status"])
    op.create_index("ix_validation_runs_created_at", "validation_runs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_validation_runs_created_at", table_name="validation_runs")
    op.drop_index("ix_validation_runs_overall_status", table_name="validation_runs")
    op.drop_index("ix_validation_runs_target_profile_id", table_name="validation_runs")
    op.drop_index("ix_validation_runs_model_fingerprint", table_name="validation_runs")
    op.drop_table("validation_runs")
