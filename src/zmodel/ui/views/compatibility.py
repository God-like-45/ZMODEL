"""Compatibility Analysis view: target-runtime profiling, operator rule evaluation, and explainable reporting."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st

from zmodel.core.compatibility import (
    CompatibilityAnalyzer,
    CompatibilityFinding,
    CompatibilityReport,
)
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.profiles import CapabilityProfile, load_builtin_profiles
from zmodel.core.state import (
    get_active_model,
    get_compatibility_report,
    get_selected_profile_id,
    navigate_to,
    set_active_model,
    set_compatibility_report,
    set_selected_profile_id,
)
from zmodel.ui.components import (
    format_bytes,
    get_status_badge_html,
    render_assessment_banner,
    render_metric_box,
    render_workspace_header,
)


def render_compatibility_view() -> None:
    """Render the Compatibility Analysis page."""
    render_workspace_header(
        title="Compatibility Analysis",
        description="Verify model operators, opsets, and tensor data types against target runtime capability profiles with explainable evidence.",
        environment_badge="Phase 3 · Active",
    )

    model: Optional[ModelSummary] = get_active_model()
    builtin_profiles: Dict[str, CapabilityProfile] = load_builtin_profiles()

    # --- 1. No Model State ---
    if model is None:
        _render_empty_model_state()
        return

    # --- 2. Active Model Banner ---
    _render_model_context_bar(model)

    # --- 3. Profile Selection Area ---
    selected_profile = _render_profile_selector(builtin_profiles)
    if selected_profile is None:
        st.warning("No capability profiles found in profiles_data directory.")
        return

    # Check if we have an existing report matching active model and selected profile
    existing_report: Optional[CompatibilityReport] = get_compatibility_report()
    needs_reanalysis = (
        existing_report is None
        or existing_report.model_fingerprint != model.sha256_hash
        or existing_report.profile_id != selected_profile.profile_id
    )

    col_btn, col_btn_right = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("Run Compatibility Analysis", type="primary", use_container_width=True)

    if run_clicked or needs_reanalysis:
        with st.spinner(f"Evaluating compatibility against {selected_profile.display_name}..."):
            report = CompatibilityAnalyzer.analyze(model, selected_profile)
            set_compatibility_report(report)
            existing_report = report

    if existing_report is not None:
        _render_analysis_results(existing_report, selected_profile)


def _render_empty_model_state() -> None:
    """Render helpful callout when no model has been uploaded or selected."""
    st.markdown(
        """
        <div class="surface-card" style="padding: 2rem 1.5rem; text-align: center; margin: 1.5rem 0;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;">
                No Inspected Model Loaded
            </div>
            <div style="font-size: 0.86rem; color: #475569; max-width: 580px; margin: 0 auto 1.25rem auto; line-height: 1.5;">
                To run compatibility analysis, first upload or select an ONNX model.
                Static inspection extracts the exact operator topology, opset imports, and tensor signatures required to evaluate target capability rules.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Load Linear Classifier Sample", use_container_width=True):
            sample_path = Path("sample_models/linear_classifier.onnx")
            if sample_path.exists():
                summary = load_and_inspect_model(sample_path.read_bytes(), filename=sample_path.name)
                set_active_model(summary, source_description="Sample: Linear Classifier")
                st.rerun()
    with col2:
        if st.button("Load ConvNet Sample", use_container_width=True):
            sample_path = Path("sample_models/conv_net.onnx")
            if sample_path.exists():
                summary = load_and_inspect_model(sample_path.read_bytes(), filename=sample_path.name)
                set_active_model(summary, source_description="Sample: ConvNet")
                st.rerun()
    with col3:
        if st.button("Go to Model Inspector", use_container_width=True):
            navigate_to("inspector")
            st.rerun()


def _render_model_context_bar(model: ModelSummary) -> None:
    """Render compact strip showing the currently loaded model under analysis."""
    op_summary = ", ".join(f"{op} ({cnt})" for op, cnt in list(model.op_counts.items())[:4])
    if len(model.op_counts) > 4:
        op_summary += f", +{len(model.op_counts) - 4} more"

    html = (
        f'<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">'
        f'<div>'
        f'<div style="font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B;">Active Model Under Analysis</div>'
        f'<div style="font-size: 0.92rem; font-weight: 700; color: #0F172A; font-family: monospace;">{model.filename}</div>'
        f'</div>'
        f'<div style="font-size: 0.78rem; color: #475569;">'
        f'<span>Nodes: <strong>{model.total_nodes}</strong></span> · '
        f'<span>Params: <strong>{model.total_parameters:,}</strong></span> · '
        f'<span>Ops: <strong>{op_summary}</strong></span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_profile_selector(profiles: Dict[str, CapabilityProfile]) -> Optional[CapabilityProfile]:
    """Render profile selection interface with transparent confidence declarations."""
    current_pid = get_selected_profile_id()
    profile_keys = list(profiles.keys())
    if not profile_keys:
        return None

    # Find index of current profile
    default_idx = profile_keys.index(current_pid) if current_pid in profile_keys else 0

    st.markdown(
        """
        <div style="font-weight: 600; font-size: 0.88rem; color: #0F172A; margin-bottom: 0.35rem;">
            Select Target Runtime Capability Profile
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_pid = st.selectbox(
        "Target Profile",
        options=profile_keys,
        index=default_idx,
        format_func=lambda pid: f"{profiles[pid].display_name} (v{profiles[pid].version})",
        label_visibility="collapsed",
        key="compat_profile_selector",
    )
    set_selected_profile_id(selected_pid)
    profile = profiles[selected_pid]

    # Profile metadata panel
    conf_style = (
        "background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0;"
        if profile.confidence_classification == "VERIFIED_DOCUMENTED"
        else "background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A;"
    )

    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem 1.15rem; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div>
                    <span style="font-weight: 700; font-size: 0.95rem; color: #0F172A;">{profile.display_name}</span>
                    <span style="font-family: monospace; font-size: 0.75rem; color: #64748B; margin-left: 0.5rem;">Profile v{profile.version} · Ruleset {profile.ruleset_version}</span>
                </div>
                <span style="font-family: monospace; font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.45rem; border-radius: 4px; {conf_style}">
                    {profile.confidence_classification}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; font-size: 0.8rem; color: #334155; margin-bottom: 0.5rem;">
                <div><span style="color: #64748B;">Architecture:</span> <strong>{profile.target_architecture}</strong></div>
                <div><span style="color: #64748B;">Runtime:</span> <strong>{profile.target_runtime}</strong></div>
                <div><span style="color: #64748B;">Opset Range:</span> <strong>[{profile.min_opset}, {profile.max_opset}]</strong></div>
                <div><span style="color: #64748B;">Verified Types:</span> <strong>{', '.join(profile.supported_data_types[:4])}</strong></div>
            </div>
            <div style="font-size: 0.76rem; color: #64748B; border-top: 1px solid #F1F5F9; padding-top: 0.4rem;">
                <span>Evidence Source: <strong>{profile.evidence_source}</strong></span> · 
                <span>Verification Date: <strong>{profile.verification_date}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return profile


def _render_analysis_results(report: CompatibilityReport, profile: CapabilityProfile) -> None:
    """Render complete compatibility assessment, findings breakdown, and export."""
    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.25rem 0;' />", unsafe_allow_html=True)

    # 1. Assessment Banner
    render_assessment_banner(
        assessment=report.overall_assessment,
        summary_counts=report.summary_counts,
        is_illustrative=report.is_illustrative_profile,
    )

    # 2. Metric Boxes
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        render_metric_box("Total Checks", str(report.summary_counts.get("TOTAL", 0)), "Grounded assertions")
    with m_col2:
        render_metric_box("PASS", str(report.summary_counts.get("PASS", 0)), "Supported features")
    with m_col3:
        render_metric_box("FAIL", str(report.summary_counts.get("FAIL", 0)), "Explicit violations")
    with m_col4:
        render_metric_box("WARNING", str(report.summary_counts.get("WARNING", 0)), "Conditional / Fallback")
    with m_col5:
        render_metric_box("UNKNOWN", str(report.summary_counts.get("UNKNOWN", 0)), "Unverified in profile")

    # 3. Findings Filtering & Breakdown
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 1.25rem 0 0.5rem 0;">
            Detailed Compatibility Findings & Evidence
        </div>
        """,
        unsafe_allow_html=True,
    )

    f_col1, f_col2 = st.columns([1, 1])
    with f_col1:
        status_filter = st.selectbox(
            "Filter by Status",
            options=["ALL", "FAIL", "WARNING", "UNKNOWN", "PASS"],
            index=0,
            key="compat_status_filter",
        )
    with f_col2:
        category_filter = st.selectbox(
            "Filter by Category",
            options=["ALL", "operator", "opset", "data_type", "graph_feature", "profile_completeness"],
            index=0,
            key="compat_cat_filter",
        )

    filtered_findings = [
        f for f in report.findings
        if (status_filter == "ALL" or f.status == status_filter)
        and (category_filter == "ALL" or f.category == category_filter)
    ]

    st.caption(f"Showing **{len(filtered_findings)}** of **{len(report.findings)}** findings.")

    if not filtered_findings:
        st.info("No findings match the selected status and category filters.")
    else:
        for finding in filtered_findings:
            _render_finding_card(finding)

    # 4. Assumptions & Next Steps Expander
    with st.expander("Assumptions, Unresolved Questions & Recommended Actions", expanded=True):
        col_assump, col_unres = st.columns(2)
        with col_assump:
            st.markdown("**Deterministic Assumptions:**")
            for item in report.assumptions:
                st.markdown(f"- <span style='font-size: 0.82rem; color: #334155;'>{item}</span>", unsafe_allow_html=True)

        with col_unres:
            st.markdown("**Unresolved Questions:**")
            for item in report.unresolved_questions:
                st.markdown(f"- <span style='font-size: 0.82rem; color: #B45309;'>{item}</span>", unsafe_allow_html=True)

        st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 0.75rem 0;' />", unsafe_allow_html=True)
        st.markdown("**Recommended Next Steps:**")
        for step in report.recommended_next_steps:
            st.markdown(f"1. <span style='font-size: 0.82rem; color: #1E293B;'>{step}</span>", unsafe_allow_html=True)

    # 5. Export Report
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 1.25rem 0 0.5rem 0;">
            Export Compatibility Report
        </div>
        """,
        unsafe_allow_html=True,
    )

    report_json = report.to_json(indent=2)
    export_filename = f"compatibility_report_{report.model_fingerprint[:8]}_{report.profile_id}.json"

    col_exp1, col_exp2 = st.columns([1, 2])
    with col_exp1:
        st.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name=export_filename,
            mime="application/json",
            use_container_width=True,
        )

    with col_exp2:
        st.caption(
            "Export contains sanitized model fingerprint, operator findings, and target profile rules. "
            "Internal paths and binary model bytes are excluded."
        )

    with st.expander("Inspect Raw JSON Report"):
        st.code(report_json, language="json")


def _render_finding_card(finding: CompatibilityFinding) -> None:
    """Render individual finding with full explainability details."""
    badge_html = get_status_badge_html(finding.status)

    sev_styles = {
        "CRITICAL": "color: #991B1B; font-weight: 600;",
        "HIGH": "color: #B91C1C; font-weight: 600;",
        "MEDIUM": "color: #B45309; font-weight: 500;",
        "LOW": "color: #475569;",
        "INFO": "color: #059669;",
    }
    sev_style = sev_styles.get(finding.severity, "color: #475569;")

    card_border = {
        "PASS": "#10B981",
        "FAIL": "#EF4444",
        "WARNING": "#F59E0B",
        "UNKNOWN": "#94A3B8",
    }.get(finding.status, "#94A3B8")

    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid {card_border}; border-radius: 6px; padding: 0.85rem 1.15rem; margin-bottom: 0.65rem;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">
                <div style="display: flex; align-items: baseline; gap: 0.5rem;">
                    <span style="font-weight: 700; font-size: 0.88rem; color: #0F172A;">{finding.feature}</span>
                    <span style="font-size: 0.72rem; color: #64748B; font-family: monospace;">[{finding.finding_id}]</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-size: 0.72rem; {sev_style}">{finding.severity}</span>
                    {badge_html}
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #334155; line-height: 1.45; margin-bottom: 0.4rem;">
                {finding.explanation}
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.5rem; font-size: 0.76rem; background: #F8FAFC; border: 1px solid #F1F5F9; border-radius: 4px; padding: 0.45rem 0.65rem; margin-bottom: 0.35rem;">
                <div><span style="color: #64748B;">Observed Model:</span> <code style="font-size: 0.72rem;">{finding.observed_value}</code></div>
                <div><span style="color: #64748B;">Profile Rule:</span> <span style="color: #0F172A;">{finding.profile_rule or 'No explicit profile rule'}</span></div>
            </div>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; font-size: 0.74rem; color: #64748B;">
                <div><span>Recommendation:</span> <strong style="color: #334155;">{finding.recommended_action}</strong></div>
                <div><span>Source:</span> <em>{finding.evidence_source}</em></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
