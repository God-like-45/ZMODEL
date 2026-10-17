"""Phase 10 schema for Kubernetes and local worker jobs.

Revision ID: 003_worker_jobs_schema
Revises: 002_graph_optimization_schema
Create Date: 2026-10-18 01:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "003_worker_jobs_schema"
down_revision: Union[str, None] = "002_graph_optimization_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "worker_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("job_type", sa.String(32), nullable=False),
        sa.Column("execution_backend", sa.String(32), server_default="local", nullable=False),
        sa.Column(
            "model_fingerprint",
            sa.String(64),
            sa.ForeignKey("models.fingerprint", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("experiment_id", sa.String(64), nullable=True),
        sa.Column("run_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), server_default="QUEUED", nullable=False),
        sa.Column("resource_profile", sa.String(32), server_default="standard_cpu", nullable=False),
        sa.Column("k8s_namespace", sa.String(64), nullable=True),
        sa.Column("k8s_job_name", sa.String(64), nullable=True),
        sa.Column("k8s_pod_name", sa.String(64), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_retries", sa.Integer(), server_default="1", nullable=False),
        sa.Column("worker_image", sa.String(128), nullable=True),
        sa.Column("queue_delay_seconds", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("startup_overhead_seconds", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("warmup_overhead_seconds", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("inference_duration_seconds", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("total_duration_seconds", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("spec_json", sa.JSON(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("output_artifact_ids", sa.JSON(), nullable=True),
        sa.Column("error_category", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index("ix_worker_jobs_job_type", "worker_jobs", ["job_type"])
    op.create_index("ix_worker_jobs_backend", "worker_jobs", ["execution_backend"])
    op.create_index("ix_worker_jobs_model_fingerprint", "worker_jobs", ["model_fingerprint"])
    op.create_index("ix_worker_jobs_experiment_id", "worker_jobs", ["experiment_id"])
    op.create_index("ix_worker_jobs_run_id", "worker_jobs", ["run_id"])
    op.create_index("ix_worker_jobs_status", "worker_jobs", ["status"])
    op.create_index("ix_worker_jobs_created_at", "worker_jobs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_worker_jobs_created_at", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_status", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_run_id", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_experiment_id", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_model_fingerprint", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_backend", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_job_type", table_name="worker_jobs")
    op.drop_table("worker_jobs")
