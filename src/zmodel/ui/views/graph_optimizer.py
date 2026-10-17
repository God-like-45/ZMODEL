"""Phase 9 ONNX Graph Optimization & Model Transformation Engine UI View.

Provides an interactive workspace for:
1. Selecting an active ONNX model and inspecting its baseline topology.
2. Configuring graph-level transformations (constant folding, operator fusion, shape inference).
3. Choosing validation depth (full structural + numerical equivalence vs structural-only).
4. Running isolated transformations and viewing live candidate lifecycles.
5. Inspecting structural integrity and numerical divergence metrics (max/mean absolute & relative error).
6. Reviewing target architecture compatibility shifts (Phase 3 integration).
7. Comparing empirical benchmark latencies and throughputs (Phase 4 integration).
8. Safely downloading validated, cryptographically fingerprinted deployment artifacts.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from zmodel.core.artifact_store import ArtifactInfo, get_artifact_store
from zmodel.core.benchmark import BenchmarkConfig
from zmodel.core.graph_optimizer import (
    CandidateState,
    GraphCandidate,
    GraphOptimizationEngine,
    GraphOptimizationReport,
    RecommendationStatus,
)
from zmodel.core.graph_transforms import (
    TransformationDescriptor,
    TransformationRegistry,
    TransformationType,
)
from zmodel.core.graph_validator import ValidationStatus
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.profiles import CapabilityProfile, load_builtin_profiles
from zmodel.core.state import (
    get_active_model,
    get_active_model_bytes,
    get_selected_profile_id,
    navigate_to,
    set_active_model,
)
from zmodel.db.repositories import GraphOptimizationRepository
from zmodel.ui.components import (
    format_bytes,
    render_metric_box,
    render_workspace_header,
)


def _render_empty_model_state() -> None:
    """Render empty state when no ONNX model is loaded."""
    st.markdown(
        """
        <div style="background: #FFFFFF; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 2.5rem 1.5rem; text-align: center; margin: 1.5rem 0;">
            <div style="font-size: 1.15rem; font-weight: 600; color: #1E293B; margin-bottom: 0.5rem;">No Active Model Loaded</div>
            <div style="font-size: 0.85rem; color: #64748B; max-width: 520px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                Load an ONNX model from the Overview workspace, or select one of the built-in sample models below to explore graph-level transformations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Linear Classifier Sample", use_container_width=True):
            sample_path = os.path.join("sample_models", "linear_classifier.onnx")
            if os.path.exists(sample_path):
                summary = load_and_inspect_model(sample_path, filename="linear_classifier.onnx")
                set_active_model(summary, source_description="Sample: linear_classifier.onnx")
                st.rerun()
    with col2:
        if st.button("Load ConvNet Sample", use_container_width=True):
            sample_path = os.path.join("sample_models", "conv_net.onnx")
            if os.path.exists(sample_path):
                summary = load_and_inspect_model(sample_path, filename="conv_net.onnx")
                set_active_model(summary, source_description="Sample: conv_net.onnx")
                st.rerun()


def render_graph_optimization_view() -> None:
    """Render the main Graph Optimization workspace view."""
    render_workspace_header(
        title="ONNX Graph Optimization & Model Transformation Engine",
        description="Graph-level operator fusion, constant folding, shape inference, and deterministic numerical equivalence verification.",
        environment_badge="Phase 9 · Active",
    )

    # Clarity Banner: Distinguish Graph Optimization from Runtime Configuration Tuning
    st.markdown(
        """
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6; border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 1.25rem;">
            <div style="font-size: 0.82rem; font-weight: 600; color: #1E293B; margin-bottom: 0.2rem;">
                Graph-Level Model Optimization vs Runtime Configuration Tuning
            </div>
            <div style="font-size: 0.78rem; color: #475569; line-height: 1.45;">
                This workspace applies <b>graph-level transformations</b> (mutating node topology, folding constant subgraphs, fusing operations, and inferring static shapes).
                This is strictly distinct from Phase 5's <b>runtime configuration optimization</b>, which tunes thread counts, batch sizes, and execution providers without modifying the model binary.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model: Optional[ModelSummary] = get_active_model()
    if model is None:
        _render_empty_model_state()
        return

    model_bytes = get_active_model_bytes()
    if model_bytes is None:
        st.warning("Active model binary is not available in memory. Please reload the model from the Overview page.")
        return

    # --- Baseline Model Summary ---
    st.markdown("#### Baseline Model Topology")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        render_metric_box("Model Name", model.filename or "model.onnx")
    with mcol2:
        render_metric_box("Model Size", format_bytes(model.total_weight_bytes or len(model_bytes)))
    with mcol3:
        render_metric_box("Total Nodes", str(model.total_nodes))
    with mcol4:
        render_metric_box("SHA-256", f"{model.sha256_hash[:12]}...")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # --- Transformation Configuration Form ---
    with st.expander("Transformation Parameters & Validation Controls", expanded=True):
        all_descriptors = TransformationRegistry.get_available_transformations()

        st.markdown("**1. Select Graph Transformations to Evaluate**")
        selected_types: List[TransformationType] = []

        cols = st.columns(len(all_descriptors))
        for idx, desc in enumerate(all_descriptors):
            with cols[idx % len(cols)]:
                status_icon = "✓" if desc.is_available else "✕"
                label = f"{desc.display_name.split('(')[0].strip()}"
                checked = desc.is_available and desc.transform_type != TransformationType.SIMPLIFICATION
                is_sel = st.checkbox(
                    label,
                    value=checked,
                    disabled=not desc.is_available,
                    key=f"chk_transform_{desc.transform_type.value}",
                    help=f"{desc.description}\n\nTool: {desc.tool_name} (v{desc.tool_version})\nAvailability: {desc.availability_reason}",
                )
                if is_sel and desc.is_available:
                    selected_types.append(desc.transform_type)
                if not desc.is_available:
                    st.caption(f"<span style='color: #94A3B8; font-size: 0.7rem;'>Not Installed ({desc.tool_name})</span>", unsafe_allow_html=True)

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("**2. Validation Depth & Numerical Tolerances**")
        vcol1, vcol2 = st.columns([2, 2])
        with vcol1:
            val_depth = st.radio(
                "Validation Depth",
                options=["FULL", "STRUCTURAL_ONLY"],
                format_func=lambda x: (
                    "Full Validation (Structural Check + Deterministic Numerical Equivalence)"
                    if x == "FULL"
                    else "Structural Only (Skip inference; marks numerical equivalence as NOT_VERIFIED)"
                ),
                key="gopt_val_depth",
            )
        with vcol2:
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                atol_val = st.number_input(
                    "Absolute Tolerance (atol)",
                    value=1e-4,
                    format="%.1e",
                    step=1e-5,
                    key="gopt_atol",
                    help="Maximum acceptable absolute difference between original and transformed model outputs.",
                )
            with subcol2:
                rtol_val = st.number_input(
                    "Relative Tolerance (rtol)",
                    value=1e-3,
                    format="%.1e",
                    step=1e-4,
                    key="gopt_rtol",
                    help="Maximum acceptable relative difference near non-zero values.",
                )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("**3. Target Capability Profile & Benchmarking**")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            profiles = load_builtin_profiles()
            profile_keys = list(profiles.keys())
            current_prof_id = get_selected_profile_id() or "ibm_z16_nnpa"
            def_idx = profile_keys.index(current_prof_id) if current_prof_id in profile_keys else 0

            target_prof_id = st.selectbox(
                "Target Compatibility Profile",
                options=profile_keys,
                index=def_idx,
                format_func=lambda k: f"{profiles[k].display_name} ({profiles[k].version})",
                key="gopt_target_profile",
            )
        with bcol2:
            enable_bench = st.checkbox(
                "Benchmark eligible transformed candidates",
                value=True,
                key="gopt_enable_bench",
                help="Executes Phase 4 empirical benchmark under identical runtime parameters on candidates that pass validation.",
            )

    # --- Run Button & Execution ---
    run_col1, run_col2 = st.columns([1, 4])
    with run_col1:
        run_clicked = st.button("Run Transformations", type="primary", use_container_width=True)

    if run_clicked:
        if not selected_types:
            st.error("Please select at least one available graph transformation method.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        def _update_progress(pct: int, msg: str):
            progress_bar.progress(pct)
            status_text.info(f"**Progress ({pct}%):** {msg}")

        engine = GraphOptimizationEngine(
            default_atol=float(atol_val),
            default_rtol=float(rtol_val),
            default_seed=42,
        )

        bconfig = BenchmarkConfig(
            execution_provider="CPUExecutionProvider",
            warmup_iterations=5,
            repeated_rounds=3,
            measured_iterations=10,
        )

        report = engine.run_optimization_experiment(
            model_bytes=model_bytes,
            original_filename=model.filename or "model.onnx",
            selected_transforms=selected_types,
            target_profile_id=target_prof_id,
            validation_depth=val_depth,
            enable_benchmarking=enable_bench,
            benchmark_config=bconfig,
            progress_callback=_update_progress,
        )

        # Persist experiment to database
        try:
            repo = GraphOptimizationRepository()
            repo.create_experiment(report)
        except Exception as ex:
            st.warning(f"Experiment metadata could not be written to PostgreSQL: {ex}")

        # Store in session state for rendering
        st.session_state["active_graph_opt_report"] = report
        progress_bar.empty()
        status_text.empty()
        st.success(f"Graph optimization experiment completed successfully ({len(report.candidates)} candidates evaluated).")

    # --- Render Results ---
    active_report: Optional[GraphOptimizationReport] = st.session_state.get("active_graph_opt_report")
    if active_report is None:
        # Check database for most recent experiment for this model
        try:
            repo = GraphOptimizationRepository()
            exps = repo.list_experiments(model_fingerprint=model.sha256_hash, limit=1)
            if exps:
                st.info(f"Loaded previous graph optimization experiment from history: `{exps[0].id}`.")
        except Exception:
            pass
        return

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Optimization Experiment Results")

    # --- Executive Recommendation Banner ---
    rec_color = "#2563EB" if active_report.best_candidate_id else "#64748B"
    st.markdown(
        f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid {rec_color}; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #64748B; letter-spacing: 0.05em;">Executive Recommendation</div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-top: 0.3rem;">
                {active_report.overall_recommendation}
            </div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 0.4rem;">
                Experiment ID: <code>{active_report.experiment_id}</code> · Validation Depth: <b>{active_report.validation_depth}</b> · Target Profile: <b>{active_report.target_profile_name or 'None'}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Candidates Summary Table ---
    st.markdown("#### Candidate Evaluation Summary")
    table_rows = []
    for c in active_report.candidates:
        size_str = f"{c.size_reduction_pct:+.1f}%" if c.size_reduction_pct is not None else "N/A"
        speedup_str = f"{c.speedup_factor:.2f}x" if c.speedup_factor is not None else "N/A"

        num_status = "N/A"
        max_err_str = "N/A"
        if c.validation_report:
            num_status = c.validation_report.numerical_status.value
            if c.validation_report.max_abs_error_across_outputs is not None:
                max_err_str = f"{c.validation_report.max_abs_error_across_outputs:.2e}"

        table_rows.append({
            "Candidate ID": c.candidate_id,
            "Transformation": c.transform_name,
            "State": c.state.value,
            "Graph Changed": "Yes" if c.graph_changed else "No",
            "Size Δ": size_str,
            "Numerical Check": num_status,
            "Max Abs Error": max_err_str,
            "Speedup": speedup_str,
            "Recommendation": c.recommendation_status.value,
        })

    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Detailed Investigation Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Correctness & Numerical Verification",
        "Graph Topology Diffs",
        "Target Compatibility Shifts",
        "Empirical Benchmark Comparison",
        "Artifact Download & Management",
    ])

    # Tab 1: Correctness & Numerical Verification
    with tab1:
        st.markdown("##### Layered Correctness Validation Details")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 1rem;">
                Structural validation verifies ONNX operator and graph constraints via <code>onnx.checker</code>.
                Numerical equivalence feeds identical deterministic seeded inputs to both models and measures output tensor divergence.
            </div>
            """,
            unsafe_allow_html=True,
        )

        for c in active_report.candidates:
            if not c.validation_report:
                continue

            vr = c.validation_report
            pill_color = "#10B981" if vr.structural_status == ValidationStatus.PASS else "#EF4444"
            st.markdown(
                f"""
                <div style="border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.85rem; margin-bottom: 0.85rem; background: #FFFFFF;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; color: #1E293B; font-size: 0.85rem;">{c.candidate_id} — {c.transform_name}</span>
                        <span style="font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; background: #F1F5F9; color: {pill_color}; font-weight: 600;">
                            Structural: {vr.structural_status.value} · Numerical: {vr.numerical_status.value}
                        </span>
                    </div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 0.4rem;">
                        {vr.numerical_details}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if vr.metrics_per_output:
                out_rows = []
                for m in vr.metrics_per_output:
                    out_rows.append({
                        "Output Tensor": m.output_name,
                        "Shape": str(m.shape),
                        "Dtype": m.dtype,
                        "Max Absolute Error": f"{m.max_absolute_error:.6e}",
                        "Mean Absolute Error": f"{m.mean_absolute_error:.6e}",
                        "Max Relative Error": f"{m.max_relative_error:.6e}",
                        "Within Tolerance": "✓ PASS" if m.passed else "✕ FAIL",
                    })
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

    # Tab 2: Graph Topology Diffs
    with tab2:
        st.markdown("##### Structural Graph Transformations")
        for c in active_report.candidates:
            if not c.diff_summary:
                continue
            d = c.diff_summary
            st.markdown(f"**{c.candidate_id} ({c.transform_name})**")
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            with dcol1:
                render_metric_box("Nodes Before / After", f"{d.get('nodes_before', 0)} → {d.get('nodes_after', 0)}")
            with dcol2:
                render_metric_box("Node Delta", f"{d.get('node_delta', 0):+d}")
            with dcol3:
                render_metric_box("Initializers", f"{d.get('initializers_before', 0)} → {d.get('initializers_after', 0)}")
            with dcol4:
                render_metric_box("Value Infos", f"{d.get('value_infos_before', 0)} → {d.get('value_infos_after', 0)}")

            if d.get("added_operators") or d.get("removed_operators"):
                st.caption(
                    f"Operator changes: Added: {d.get('added_operators', {})} | Removed: {d.get('removed_operators', {})}"
                )
            st.markdown("<hr style='margin: 0.75rem 0; border: none; border-top: 1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # Tab 3: Target Compatibility Shifts
    with tab3:
        st.markdown("##### Target Architecture Compatibility Reassessment (Phase 3)")
        if not active_report.target_profile_name:
            st.info("No target capability profile was selected for compatibility reassessment.")
        else:
            compat_rows = []
            for c in active_report.candidates:
                if c.compatibility_report:
                    sc = c.compatibility_report.get("summary_counts", {})
                    overall = c.compatibility_report.get("overall_assessment", "UNKNOWN")
                    delta = c.compatibility_delta or {}
                    compat_rows.append({
                        "Candidate ID": c.candidate_id,
                        "Transformation": c.transform_name,
                        "Overall Assessment": overall,
                        "PASS Findings": sc.get("PASS", 0),
                        "FAIL Findings": sc.get("FAIL", 0),
                        "WARNING Findings": sc.get("WARNING", 0),
                        "UNKNOWN Findings": sc.get("UNKNOWN", 0),
                        "Assessment Shift": f"{delta.get('baseline_assessment', '')} → {delta.get('transformed_assessment', '')}" if delta else "None",
                    })
            if compat_rows:
                st.dataframe(pd.DataFrame(compat_rows), use_container_width=True, hide_index=True)

    # Tab 4: Empirical Benchmark Comparison
    with tab4:
        st.markdown("##### Empirical ONNX Runtime Benchmarking (Phase 4)")
        if not active_report.benchmark_enabled:
            st.info("Benchmarking was disabled for this experiment.")
        else:
            bench_rows = []
            for c in active_report.candidates:
                if c.benchmark_report and c.benchmark_report.get("latency"):
                    lat = c.benchmark_report["latency"]
                    tp = c.benchmark_report.get("throughput") or {}
                    bench_rows.append({
                        "Candidate ID": c.candidate_id,
                        "Transformation": c.transform_name,
                        "Median Latency (ms)": f"{lat.get('median_ms', 0.0):.3f}",
                        "P95 Latency (ms)": f"{lat.get('p95_ms', 0.0):.3f}",
                        "Throughput (inf/sec)": f"{tp.get('inferences_per_second', 0.0):.1f}",
                        "Speedup vs Baseline": f"{c.speedup_factor:.2f}x" if c.speedup_factor else "1.00x",
                        "Latency Reduction": f"{c.latency_reduction_pct:+.1f}%" if c.latency_reduction_pct else "0.0%",
                    })
            if bench_rows:
                st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No candidates produced comparable benchmark runs.")

    # Tab 5: Artifact Download & Management
    with tab5:
        st.markdown("##### Controlled Artifact Storage & Download")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #64748B; margin-bottom: 1rem;">
                Download cryptographically verified transformed ONNX models. Original models are strictly preserved.
                Artifacts that failed validation are rejected and prevented from download.
            </div>
            """,
            unsafe_allow_html=True,
        )

        store = get_artifact_store()
        for c in active_report.candidates:
            if not c.artifact_id:
                continue

            can_download = (c.recommendation_status != RecommendationStatus.REJECTED)
            badge_text = "VERIFIED" if can_download else "REJECTED"
            badge_color = "#10B981" if can_download else "#EF4444"

            acol1, acol2 = st.columns([3, 1])
            with acol1:
                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; font-weight: 600; color: #1E293B;">
                        {c.candidate_id} ({c.transform_name})
                        <span style="font-size: 0.7rem; background: #F1F5F9; color: {badge_color}; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">
                            {badge_text}
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748B;">
                        Artifact ID: <code>{c.artifact_id}</code> · Hash: <code>{c.transformed_hash or 'N/A'}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with acol2:
                if can_download:
                    try:
                        raw_bytes, info = store.read_artifact_bytes(c.artifact_id)
                        st.download_button(
                            label=f"Download ONNX",
                            data=raw_bytes,
                            file_name=info.suggested_filename,
                            mime="application/octet-stream",
                            key=f"dl_btn_{c.candidate_id}",
                            use_container_width=True,
                        )
                    except Exception as ex:
                        st.caption(f"Error reading artifact: {ex}")
                else:
                    st.caption("<span style='color: #EF4444; font-size: 0.75rem;'>Download Prohibited (Validation Failed)</span>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #F1F5F9;'>", unsafe_allow_html=True)
