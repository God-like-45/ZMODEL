"""Streamlit view for Phase 11: IBM Z / Linux on IBM Z Target Validation & Deployment Readiness."""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import streamlit as st

from zmodel.core.environment import inspect_environment
from zmodel.core.state import get_active_model
from zmodel.core.target_profiles import TargetValidationProfile, load_target_profiles
from zmodel.core.validation_engine import (
    CheckStatus,
    OverallValidationStatus as ValidationStatus,
    ValidationCheck,
    ValidationEngine,
    ValidationReport,
)
from zmodel.db.repositories import ValidationRepository
from zmodel.db.session import session_scope

logger = logging.getLogger(__name__)


def _render_check_badge(status: CheckStatus) -> str:
    """Return HTML badge for a check status."""
    if status == CheckStatus.PASS:
        return '<span style="background: #DCFCE7; color: #166534; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; border: 1px solid #BBF7D0;">PASS</span>'
    elif status == CheckStatus.FAIL:
        return '<span style="background: #FEE2E2; color: #991B1B; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; border: 1px solid #FECACA;">FAIL</span>'
    elif status == CheckStatus.WARNING:
        return '<span style="background: #FEF3C7; color: #92400E; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; border: 1px solid #FDE68A;">WARNING</span>'
    elif status == CheckStatus.NOT_RUN:
        return '<span style="background: #F1F5F9; color: #475569; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; border: 1px solid #CBD5E1;">NOT_RUN</span>'
    else:
        return '<span style="background: #F3F4F6; color: #374151; font-weight: 700; font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; border: 1px solid #E5E7EB;">UNKNOWN</span>'


def _render_overall_badge(status: ValidationStatus) -> str:
    """Return HTML badge for overall validation status."""
    if status == ValidationStatus.TARGET_VERIFIED:
        return '<span style="background: #DCFCE7; color: #166534; font-weight: 700; font-size: 0.85rem; padding: 4px 12px; border-radius: 6px; border: 1px solid #86EFAC;">● IBM Z TARGET VERIFIED</span>'
    elif status == ValidationStatus.LOCALLY_VERIFIED:
        return '<span style="background: #EFF6FF; color: #1E40AF; font-weight: 700; font-size: 0.85rem; padding: 4px 12px; border-radius: 6px; border: 1px solid #BFDBFE;">◈ LOCALLY VERIFIED (IBM Z PENDING)</span>'
    elif status == ValidationStatus.PARTIALLY_VERIFIED:
        return '<span style="background: #FEF3C7; color: #92400E; font-weight: 700; font-size: 0.85rem; padding: 4px 12px; border-radius: 6px; border: 1px solid #FDE68A;">▲ PARTIALLY VERIFIED</span>'
    elif status == ValidationStatus.FAILED_VALIDATION:
        return '<span style="background: #FEE2E2; color: #991B1B; font-weight: 700; font-size: 0.85rem; padding: 4px 12px; border-radius: 6px; border: 1px solid #FECACA;">✕ FAILED VALIDATION</span>'
    else:
        return '<span style="background: #F1F5F9; color: #475569; font-weight: 700; font-size: 0.85rem; padding: 4px 12px; border-radius: 6px; border: 1px solid #CBD5E1;">○ NOT VERIFIED</span>'


def render_validation_view() -> None:
    """Render the Phase 11 IBM Z Target Validation and Deployment Readiness dashboard."""
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                <span class="zm-badge zm-badge-info">Phase 11: IBM Z Validation</span>
                <span class="zm-badge zm-badge-neutral">Deployment Readiness</span>
            </div>
            <h2 style="margin: 0; color: #0F172A; font-weight: 700; font-size: 1.6rem; letter-spacing: -0.02em;">
                IBM Z & Linux on IBM Z Validation Engine
            </h2>
            <p style="margin: 0.35rem 0 0 0; color: #475569; font-size: 0.88rem;">
                Rigorous separation of local development facts from authentic IBM Z target verification (<code style="background: #F1F5F9; padding: 2px 5px; border-radius: 3px;">s390x</code>). Evaluates 12 empirical compatibility criteria, provider fallback detection, and numerical precision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Environment Detection (Safe & Read-Only)
    env = inspect_environment(collection_method="LOCAL_INSPECTION")
    profiles = load_target_profiles()

    # Hardware Banner
    if not env.is_s390x:
        st.markdown(
            f"""
            <div style="background: #FFFBEB; border: 1px solid #FCD34D; border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.1rem; color: #D97706;">⚠️</span>
                    <strong style="color: #92400E; font-size: 0.88rem;">Host Architecture Notice: {env.architecture} (Non-IBM Z Host)</strong>
                </div>
                <div style="font-size: 0.82rem; color: #78350F; margin-top: 0.35rem; line-height: 1.5;">
                    The current host is <strong>{env.os_name} {env.architecture}</strong>. Target architecture checks (<code style="background: #FEF3C7; padding: 1px 4px; border-radius: 3px;">s390x</code>) and accelerator verification (NNPA facility 193) will not run locally and cannot be claimed as passed. Evaluated models will receive <strong style="color: #1E40AF;">Locally Verified</strong> readiness status until validated on authentic IBM Z hardware.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: #F0FDF4; border: 1px solid #86EFAC; border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 1.1rem; color: #16A34A;">✓</span>
                    <strong style="color: #166534; font-size: 0.88rem;">Target Architecture Detected: s390x (IBM Z / LinuxONE)</strong>
                </div>
                <div style="font-size: 0.82rem; color: #14532D; margin-top: 0.35rem;">
                    CPU Model: {env.cpu_model} | NNPA Facility 193: {'Available' if env.has_nnpa_facility else 'Not Present'} | Providers: {', '.join(env.available_providers)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Controls: Model & Profile Selection
    col_c1, col_c2, col_c3 = st.columns([1.5, 1.5, 1])

    profile_keys = list(profiles.keys())
    selected_profile_id = col_c1.selectbox(
        "Target Validation Profile",
        options=profile_keys,
        format_func=lambda k: f"{profiles[k].profile_name} (v{profiles[k].profile_version})",
        index=0,
        help="Select the hardware profile and architecture rules to validate against.",
    )
    profile = profiles[selected_profile_id]

    active_model = get_active_model()
    model_name = active_model.filename if active_model else "No model loaded"
    model_path = active_model.model_path if active_model else ""

    col_c2.text_input(
        "Active Model",
        value=model_name,
        disabled=True,
        help="Current model loaded in workspace.",
    )

    with col_c3:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        btn_validate = st.button(
            "Run Validation",
            type="primary",
            use_container_width=True,
            disabled=(active_model is None or not os.path.exists(model_path)),
            help="Execute the 12-point validation engine against the selected target profile.",
        )

    if not active_model or not os.path.exists(model_path):
        st.info("Load an ONNX model in the Overview or Inspector tab to run target validation.")

    # Execute Validation
    if btn_validate and active_model and os.path.exists(model_path):
        with st.spinner("Executing 12-point target validation checks..."):
            engine = ValidationEngine(profile=profile, environment=env)
            report = engine.evaluate(model_path=model_path)
            st.session_state["latest_validation_report"] = report

            # Persist to database if available
            try:
                with session_scope() as session:
                    repo = ValidationRepository(session)
                    repo.save_validation_report(report)
            except Exception as ex:
                logger.warning(f"Could not persist validation report to database: {ex}")

    report: Optional[ValidationReport] = st.session_state.get("latest_validation_report")

    # If no run yet, we can show a baseline comparison
    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # 3. Environment Inventory & Comparison
    st.markdown("### 1. Detected Host Facts vs. Target Requirements")
    st.markdown(
        "<p style='color: #64748B; font-size: 0.84rem; margin-top: -0.5rem;'>Truthful comparison between local system inspection and configured target profile expectations.</p>",
        unsafe_allow_html=True,
    )

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.04em; margin-bottom: 0.5rem;">
                    Detected Host Facts (Fingerprint: <code style="font-size: 0.7rem;">{env.canonical_fingerprint[:8]}...</code>)
                </div>
                <table style="width: 100%; font-size: 0.82rem; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Architecture</td><td style="padding: 4px 0; font-weight: 600; color: #0F172A;">{env.architecture}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">OS & Kernel</td><td style="padding: 4px 0; color: #0F172A;">{env.os_name} {env.os_release} ({env.kernel_version[:18]}...)</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Python Version</td><td style="padding: 4px 0; color: #0F172A;">{env.python_version}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">ONNX Runtime</td><td style="padding: 4px 0; color: #0F172A;">{env.onnxruntime_version or 'Not installed'}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Installed Providers</td><td style="padding: 4px 0; color: #2563EB;">{', '.join(env.available_providers)}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">NNPA Accelerator</td><td style="padding: 4px 0; color: #0F172A;">{'Available (Facility 193)' if env.has_nnpa_facility else 'Not Present'}</td></tr>
                    <tr><td style="padding: 4px 0; color: #64748B;">Collection Method</td><td style="padding: 4px 0; color: #0F172A;">{env.collection_method}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with comp_col2:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.04em; margin-bottom: 0.5rem;">
                    Target Profile Requirements ({profile.profile_name})
                </div>
                <table style="width: 100%; font-size: 0.82rem; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Target Architecture</td><td style="padding: 4px 0; font-weight: 600; color: #0F172A;">{profile.architecture}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Supported OS</td><td style="padding: 4px 0; color: #0F172A;">{', '.join(profile.supported_os)}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Python Constraint</td><td style="padding: 4px 0; color: #0F172A;">{profile.min_python_version} – {profile.max_python_version}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Min ONNX Runtime</td><td style="padding: 4px 0; color: #0F172A;">{profile.min_ort_version}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Required Providers</td><td style="padding: 4px 0; color: #2563EB;">{', '.join(profile.required_providers)}</td></tr>
                    <tr style="border-bottom: 1px solid #EDF2F7;"><td style="padding: 4px 0; color: #64748B;">Accelerator Required</td><td style="padding: 4px 0; color: #0F172A;">{'Yes (' + ', '.join(profile.required_accelerators) + ')' if profile.requires_accelerator else 'No (CPU baseline)'}</td></tr>
                    <tr><td style="padding: 4px 0; color: #64748B;">Verification Evidence</td><td style="padding: 4px 0; color: #0F172A;">{profile.evidence_source}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 4. Active Validation Report Results
    if report:
        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("### 2. Validation Run Evaluation & Provenance")

        # Top Metric Ribbon
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"**Overall Status**<br>{_render_overall_badge(report.overall_status)}", unsafe_allow_html=True)
        m2.metric("Checks Passed", f"{report.summary_counts.get('PASS', 0)} / {len(report.checks)}")
        m3.metric("Deferred / Not Run", report.summary_counts.get("NOT_RUN", 0))
        m4.metric("Failures / Warnings", f"{report.summary_counts.get('FAIL', 0)} Fail / {report.summary_counts.get('WARNING', 0)} Warn")

        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1rem; margin-top: 0.75rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: #64748B; font-weight: 700; letter-spacing: 0.04em;">Evaluation Rationale & Provenance</div>
                <div style="font-size: 0.86rem; color: #1E293B; margin-top: 0.25rem;">{report.status_rationale}</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.35rem;">
                    Provenance Badge: <strong style="color: #2563EB;">{report.provenance_badge}</strong> |
                    Run ID: <code style="font-size: 0.75rem;">{report.run_id}</code> |
                    Artifact SHA-256: <code style="font-size: 0.75rem;">{report.model_fingerprint[:16]}...</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 5. Checklist Table
        st.markdown("### 3. Comprehensive 12-Point Checklist")
        st.markdown(
            "<p style='color: #64748B; font-size: 0.84rem; margin-top: -0.5rem;'>Detailed evaluation criteria distinguishing local verification from requirements that require real IBM Z execution.</p>",
            unsafe_allow_html=True,
        )

        for check in report.checks:
            badge_html = _render_check_badge(check.status)
            target_badge = '<span style="background: #F3E8FF; color: #6B21A8; font-size: 0.7rem; font-weight: 600; padding: 1px 6px; border-radius: 3px;">Requires IBM Z Target</span>' if check.requires_target_hardware else '<span style="background: #F1F5F9; color: #475569; font-size: 0.7rem; padding: 1px 6px; border-radius: 3px;">Evaluated Locally</span>'
            
            with st.expander(f"{check.check_id.upper()} — {check.name} ({check.status.value})", expanded=(check.status in (CheckStatus.FAIL, CheckStatus.WARNING))):
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                        <div>{badge_html} &nbsp; {target_badge}</div>
                    </div>
                    <table style="width: 100%; font-size: 0.82rem; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #F1F5F9;"><td style="width: 25%; color: #64748B; padding: 4px 0;">Requirement</td><td style="padding: 4px 0; font-weight: 500;">{check.requirement}</td></tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;"><td style="color: #64748B; padding: 4px 0;">Expected Value</td><td style="padding: 4px 0;"><code>{check.expected_value}</code></td></tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;"><td style="color: #64748B; padding: 4px 0;">Detected Value</td><td style="padding: 4px 0;"><code>{check.detected_value}</code></td></tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;"><td style="color: #64748B; padding: 4px 0;">Evidence & Diagnostics</td><td style="padding: 4px 0; color: #334155;">{check.evidence}</td></tr>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )
                if check.remediation:
                    st.info(f"**Remediation Guidance:** {check.remediation}")

        # 6. Benchmark Provenance & Guardrails
        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("### 4. Benchmark Integrity & Hardware Provenance")
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1rem;">
                <div style="font-weight: 600; color: #0F172A; font-size: 0.88rem; margin-bottom: 0.35rem;">
                    Strict Benchmark Comparison Guardrails
                </div>
                <div style="font-size: 0.82rem; color: #475569; line-height: 1.5;">
                    • <strong>Cross-Architecture Equivalence Prohibition:</strong> Local x86_64 latencies must <em>never</em> be compared directly against IBM Z latencies as equivalent. Memory hierarchies, SIMD vectors, cache structures, and hardware accelerators (NNPA) differ substantially.<br>
                    • <strong>Execution Provider Fallback Protection:</strong> If a model requests a specialized provider (e.g. NNPA execution provider) that is missing on the target, fallback to CPUExecutionProvider is flagged as a warning/failure, never reported as hardware-accelerated execution.<br>
                    • <strong>Provenance Tagging:</strong> All benchmark metrics stored by Z-ModelPilot explicitly record whether they were obtained via <code>LOCAL_HOST</code> or <code>IBM_Z_NATIVE</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 7. Actionable Next-Steps Checklist (When Blocked or Local)
        if report.overall_status != ValidationStatus.TARGET_VERIFIED:
            st.markdown("### 5. Deployment Readiness Next-Steps Checklist")
            st.markdown(
                """
                To elevate this model from **Locally Verified** to **IBM Z Target Verified**, complete the following operational steps:
                1. **Target Access Provisioning:** Establish authorized SSH or container runner access to an identified Linux on IBM Z (`s390x`) LPAR or z/VM guest.
                2. **Environment Variable Configuration:** Set `ZMODEL_TARGET_HOST`, `ZMODEL_TARGET_USER`, and `ZMODEL_TARGET_KEY_PATH` in your secure environment.
                3. **Remote Integrity Check:** Transmit the model artifact and verify SHA-256 integrity on the target node.
                4. **Target Execution Provider Probe:** Verify that `onnxruntime` is installed with the required execution provider (e.g., CPU or NNPA).
                5. **Target Benchmark Execution:** Run isolated benchmark iterations on the target hardware to capture authentic IBM Z latency and throughput.
                """
            )

    # 8. Historical Validation Runs
    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("### 6. Validation Run Audit History")

    try:
        with session_scope() as session:
            repo = ValidationRepository(session)
            history = repo.list_validation_runs(limit=10)
            if history:
                history_rows = []
                for run in history:
                    history_rows.append({
                        "Run ID": run.run_id[:8] + "...",
                        "Profile": run.profile_id,
                        "Target Arch": run.target_architecture,
                        "Host Arch": run.host_architecture,
                        "Status": run.overall_status,
                        "Passed": f"{run.checks_passed}/{len(run.checks_data or [])}",
                        "Timestamp": run.created_at.strftime("%Y-%m-%d %H:%M") if run.created_at else "N/A",
                    })
                st.dataframe(history_rows, use_container_width=True)
            else:
                st.caption("No historical validation runs found in database.")
    except Exception as ex:
        st.caption(f"Historical persistence query unavailable: {ex}")
