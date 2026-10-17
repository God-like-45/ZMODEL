"""Streamlit view for Kubernetes and local worker job monitoring and lifecycle management."""

import json
import logging
from typing import Any, Dict, List, Optional
import streamlit as st

from zmodel.core.execution import (
    RESOURCE_PROFILES,
    WorkerJobState,
    WorkerJobType,
    get_execution_backend,
)
from zmodel.db.repositories import WorkerJobRepository
from zmodel.db.session import session_scope

logger = logging.getLogger(__name__)


def render_workers_view() -> None:
    """Render the Phase 10 Worker Jobs control plane and lifecycle monitoring dashboard."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span class="zm-badge zm-badge-info">Phase 10: Scalable Execution</span>
                <span class="zm-badge zm-badge-neutral">Kubernetes & Control Plane</span>
            </div>
            <h2 style="margin: 0; color: #0F172A; font-weight: 700; font-size: 1.6rem; letter-spacing: -0.02em;">
                Worker Job Orchestration & Scalable Execution
            </h2>
            <p style="margin: 0.35rem 0 0 0; color: #475569; font-size: 0.88rem;">
                Decoupled control plane managing containerized worker jobs, Kubernetes resource quotas, execution deadlines, and timing breakdowns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Backend Connectivity & Health Check
    col_b1, col_b2, col_b3 = st.columns([1.5, 1.5, 1])

    backend_choice = col_b1.selectbox(
        "Active Execution Backend",
        options=["local", "kubernetes"],
        format_func=lambda b: "Local (In-Process Threads)" if b == "local" else "Kubernetes (Cluster Jobs)",
        index=0,
        help="Select whether benchmark and optimization tasks are executed in local threads or dispatched to Kubernetes cluster worker jobs.",
    )

    with col_b2:
        try:
            backend_inst = get_execution_backend(name=backend_choice)
            health = backend_inst.check_health()
            if health.is_available:
                st.markdown(
                    f"""
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 0.5rem 0.75rem; margin-top: 1.5rem;">
                        <div style="font-size: 0.72rem; color: #166534; font-weight: 600;">BACKEND STATUS</div>
                        <div style="font-size: 0.88rem; font-weight: 700; color: #15803D;">● Connected & Healthy</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 6px; padding: 0.5rem 0.75rem; margin-top: 1.5rem;">
                        <div style="font-size: 0.72rem; color: #991B1B; font-weight: 600;">BACKEND STATUS</div>
                        <div style="font-size: 0.84rem; font-weight: 600; color: #DC2626;">▲ Cluster Unreachable</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as ex:
            st.error(f"Backend probe failed: {ex}")

    with col_b3:
        if st.button("Refresh Jobs", use_container_width=True, key="btn_refresh_jobs"):
            st.rerun()

    # 2. Resource Profiles Card
    with st.expander("Container Resource Allocation Profiles & Timing Breakdown", expanded=False):
        st.markdown(
            """
            <p style="font-size: 0.82rem; color: #475569; margin-bottom: 0.75rem;">
                Worker pods execute with strict Kubernetes resource requests and limits to guarantee benchmark reproducibility and prevent node exhaustion.
            </p>
            """,
            unsafe_allow_html=True,
        )
        p_cols = st.columns(len(RESOURCE_PROFILES))
        for idx, (p_name, prof) in enumerate(RESOURCE_PROFILES.items()):
            with p_cols[idx]:
                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.65rem;">
                        <div style="font-weight: 700; color: #0F172A; font-size: 0.85rem;">{prof.name}</div>
                        <div style="font-size: 0.72rem; color: #64748B; margin-top: 0.25rem;">CPU Req / Lim: <b>{prof.cpu_request} / {prof.cpu_limit}</b></div>
                        <div style="font-size: 0.72rem; color: #64748B;">Mem Req / Lim: <b>{prof.memory_request} / {prof.memory_limit}</b></div>
                        <div style="font-size: 0.72rem; color: #64748B;">Deadline: <b>{prof.active_deadline_seconds}s</b></div>
                        <div style="font-size: 0.70rem; color: #475569; margin-top: 0.35rem; line-height: 1.3;">{prof.description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 0.75rem; margin-top: 0.75rem; font-size: 0.78rem; color: #1E40AF; line-height: 1.45;">
                <b>Critical Benchmark Timing Boundary:</b> Infrastructure startup overhead (Kubernetes scheduling, container image pulling, model deserialization) is tracked separately as <code>queue_delay_seconds</code> and <code>startup_overhead_seconds</code>. <b>Inference duration latency measurements never include scheduling or container initiation delays.</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Query Jobs from Database
    try:
        with session_scope() as s:
            repo = WorkerJobRepository(session=s)
            records, total = repo.list_jobs(limit=50)
            jobs_data = [r.to_dict() for r in records]
    except Exception as ex:
        st.warning(f"Could not connect to database to list historical jobs: {ex}")
        jobs_data = []
        total = 0

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem; margin-bottom: 0.5rem;">
            <div style="font-weight: 700; font-size: 1.05rem; color: #0F172A;">Worker Job History & Active Tasks ({total})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not jobs_data:
        st.info("No worker jobs have been submitted yet. Trigger a benchmark, configuration optimization, or graph optimization experiment via UI or API.")
        return

    # Table of Jobs
    cols_header = st.columns([2, 1.2, 1, 1.2, 1.2, 1.2, 1.2])
    cols_header[0].markdown("**Job ID / Correlation**")
    cols_header[1].markdown("**Type**")
    cols_header[2].markdown("**Backend**")
    cols_header[3].markdown("**Status**")
    cols_header[4].markdown("**Profile**")
    cols_header[5].markdown("**Inference Time**")
    cols_header[6].markdown("**Action**")

    st.divider()

    for job in jobs_data:
        c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 1.2, 1, 1.2, 1.2, 1.2, 1.2])

        c1.markdown(f"`{job['id']}`<br><span style='font-size:0.7rem; color:#64748B;'>{job.get('experiment_id') or job.get('run_id') or '-'}</span>", unsafe_allow_html=True)
        c2.markdown(f"**{job['job_type']}**")
        c3.markdown(f"`{job['execution_backend']}`")

        # Status badge
        st_val = job["status"]
        if st_val == "SUCCEEDED":
            badge_html = "<span class='zm-badge zm-badge-pass'>SUCCEEDED</span>"
        elif st_val in ("RUNNING", "SUBMITTED"):
            badge_html = "<span class='zm-badge zm-badge-info'>RUNNING</span>"
        elif st_val == "QUEUED":
            badge_html = "<span class='zm-badge zm-badge-neutral'>QUEUED</span>"
        elif st_val in ("CANCEL_REQUESTED", "CANCELLED"):
            badge_html = "<span class='zm-badge zm-badge-warning'>CANCELLED</span>"
        else:
            badge_html = "<span class='zm-badge zm-badge-fail'>FAILED</span>"

        c4.markdown(badge_html, unsafe_allow_html=True)
        c5.markdown(f"`{job['resource_profile']}`")

        inf_time = job.get("inference_duration_seconds", 0.0)
        c6.markdown(f"{inf_time:.3f}s" if inf_time > 0 else "-")

        with c7:
            if st_val in ("RUNNING", "QUEUED", "SUBMITTED"):
                if st.button("Cancel", key=f"btn_cancel_{job['id']}", type="secondary"):
                    try:
                        backend_inst = get_execution_backend(name=job["execution_backend"])
                        backend_inst.cancel_job(job["id"])
                        with session_scope() as s:
                            repo = WorkerJobRepository(session=s)
                            repo.update_job_status(job["id"], status="CANCELLED", error_category="CANCELLED", error_message="Cancelled by user from Streamlit UI")
                        st.success(f"Cancelled {job['id']}")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Cancel failed: {ex}")

        # Inspection Expander for Timing Breakdown
        with st.expander(f"Inspect Details for {job['id']}", expanded=False):
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown("**Timing Decomposition:**")
                st.write(f"- Queue / Scheduling Delay: `{job.get('queue_delay_seconds', 0.0):.4f}s`")
                st.write(f"- Container Startup Overhead: `{job.get('startup_overhead_seconds', 0.0):.4f}s`")
                st.write(f"- Warmup Duration: `{job.get('warmup_overhead_seconds', 0.0):.4f}s`")
                st.write(f"- Pure Inference Benchmark: `{job.get('inference_duration_seconds', 0.0):.4f}s`")
                st.write(f"- Total Job Wallclock Duration: `{job.get('total_duration_seconds', 0.0):.4f}s`")

            with t_col2:
                st.markdown("**Cluster & Environment Metadata:**")
                st.write(f"- Namespace: `{job.get('k8s_namespace') or 'N/A (Local)'}`")
                st.write(f"- Kubernetes Job Name: `{job.get('k8s_job_name') or 'N/A'}`")
                st.write(f"- Kubernetes Pod Name: `{job.get('k8s_pod_name') or 'N/A'}`")
                st.write(f"- Worker Image: `{job.get('worker_image') or 'local-process'}`")
                st.write(f"- Target Model SHA-256: `{job.get('model_fingerprint')}`")

            if job.get("error_message"):
                st.error(f"Error ({job.get('error_category') or 'Error'}): {job['error_message']}")

            if job.get("result_json"):
                st.markdown("**Structured Result Payload:**")
                st.json(job["result_json"])
