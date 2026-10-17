"""
Phase 12: Advanced Optimization & Enterprise Reporting UI View.

Provides an interactive decision-support workspace to:
1. Configure bounded optimization searches: Grid Search, Seeded Random Search, and Multi-Fidelity Successive Halving.
2. Filter candidates based on architecture compatibility profiles.
3. Benchmark candidates using genuine ONNX Runtime inference.
4. Explore multi-objective Pareto frontiers, constraint evaluations, and sensitivity curves.
5. Review evidence-backed deployment recommendations (Latency, Throughput, Memory, or Balanced Score).
6. Export reproducible 18-section enterprise audit reports (Markdown & JSON) and CSV candidate rankings.
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import altair as alt
import pandas as pd
import streamlit as st

from zmodel.core.advanced_optimizer import (
    AdvancedOptimizationEngine,
    EnterpriseReportGenerator,
    RecommendationGoal,
    SearchStrategy,
)
from zmodel.core.benchmark import discover_execution_providers
from zmodel.core.compatibility import CompatibilityAnalyzer, CompatibilityReport
from zmodel.core.environment import EnvironmentInventory
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.optimizer import (
    CandidateStatus,
    ExecutionCandidate,
    HardConstraint,
    OptimizationObjective,
    OptimizationReport,
    OptimizationRequest,
)
from zmodel.core.profiles import CapabilityProfile, load_builtin_profiles
from zmodel.core.state import (
    get_active_model,
    get_active_model_bytes,
    get_optimization_report,
    get_selected_profile_id,
    navigate_to,
    set_active_model,
    set_optimization_report,
    set_selected_profile_id,
)
from zmodel.ui.components import (
    format_bytes,
    render_metric_box,
    render_optimization_banner,
    render_workspace_header,
)


def render_optimization_view() -> None:
    """Render the Phase 12 Advanced Optimization & Enterprise Decision-Support view."""
    render_workspace_header(
        title="Optimization & Decision Support",
        description="Multi-objective Pareto frontiers, bounded configuration search, sensitivity analysis, and enterprise decision reports.",
        environment_badge="Phase 12 · Active",
    )

    model: Optional[ModelSummary] = get_active_model()
    profiles = load_builtin_profiles()
    env_info = discover_execution_providers()

    # --- 1. Empty State ---
    if model is None:
        _render_empty_model_state()
        return

    # --- 2. Active Model & Profile Context Strip ---
    selected_profile = _render_context_strip(model, profiles)

    # --- 3. Configuration & Optimization Controls ---
    setup_config = _render_optimization_setup(model, selected_profile, env_info)
    request = setup_config["request"]
    strategy = setup_config["strategy"]
    goal = setup_config["goal"]
    graph_opts = setup_config["graph_opts"]
    exec_modes = setup_config["exec_modes"]

    col_btn, col_btn_spc = st.columns([1.2, 2.8])
    with col_btn:
        run_clicked = st.button("Run Optimization Search", type="primary", use_container_width=True)

    # --- 4. Execution Workflow ---
    if run_clicked:
        model_bytes = get_active_model_bytes()
        if not model_bytes:
            st.error("Model binary content is unavailable in session state. Please re-upload or re-select the model.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(msg: str, current: int, total: int) -> None:
            pct = min(max(int((current / max(total, 1)) * 100), 0), 100)
            progress_bar.progress(pct)
            status_text.caption(f"**Optimization in progress ({pct}%):** {msg}")

        try:
            env_inv = EnvironmentInventory.collect()
            report_dict = AdvancedOptimizationEngine.execute_advanced_optimization(
                model_path_or_bytes=model_bytes,
                model_summary=model,
                profile=selected_profile,
                request=request,
                strategy=strategy,
                recommendation_goal=goal,
                graph_optimization_levels=graph_opts,
                execution_modes=exec_modes,
                env_inventory=env_inv,
                progress_callback=on_progress,
            )
            set_optimization_report(report_dict)
            progress_bar.empty()
            status_text.empty()
            st.success("Advanced optimization search and audit report completed successfully.")
            st.rerun()
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Optimization execution failed: {str(e)}")

    # --- 5. Results Display ---
    raw_rep = get_optimization_report()
    if raw_rep is not None:
        model_fp = raw_rep.get("model_fingerprint") if isinstance(raw_rep, dict) else raw_rep.model_fingerprint
        if model_fp == model.sha256_hash:
            _render_optimization_results(raw_rep)
        else:
            st.info("Previous optimization report was invalidated because a new model was loaded.")


def _render_empty_model_state() -> None:
    """Render empty state when no model is active."""
    st.markdown(
        """
        <div class="surface-card" style="padding: 2.25rem 1.5rem; text-align: center; margin: 1.5rem 0;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;">
                No Model Loaded for Optimization
            </div>
            <div style="font-size: 0.86rem; color: #475569; max-width: 600px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                Select or upload an ONNX model to explore candidate runtime configurations, evaluate multi-objective
                trade-offs against target capability profiles, and identify the optimal Pareto frontier.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Load Linear Classifier Sample", use_container_width=True):
            p = Path("sample_models/linear_classifier.onnx")
            if p.exists():
                data = p.read_bytes()
                summary = load_and_inspect_model(data, filename=p.name)
                set_active_model(summary, source_description="Sample: Linear Classifier", model_bytes=data)
                st.rerun()
    with col2:
        if st.button("Load ConvNet Sample", use_container_width=True):
            p = Path("sample_models/conv_net.onnx")
            if p.exists():
                data = p.read_bytes()
                summary = load_and_inspect_model(data, filename=p.name)
                set_active_model(summary, source_description="Sample: ConvNet", model_bytes=data)
                st.rerun()
    with col3:
        if st.button("Go to Model Inspector", use_container_width=True):
            navigate_to("inspector")
            st.rerun()


def _render_context_strip(
    model: ModelSummary,
    profiles: Dict[str, CapabilityProfile],
) -> CapabilityProfile:
    """Render target model details, profile selector, and compatibility summary."""
    curr_profile_id = get_selected_profile_id()
    if curr_profile_id not in profiles:
        curr_profile_id = next(iter(profiles.keys()))

    col_ctx1, col_ctx2 = st.columns([1.5, 1])
    with col_ctx1:
        input_shapes = ", ".join(f"{t.name}: {t.shape}" for t in model.inputs)
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;">
                <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em;">Target Model</div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #0F172A; font-family: monospace;">{model.filename}</div>
                <div style="font-size: 0.78rem; color: #475569; margin-top: 0.2rem;">
                    Nodes: <strong>{model.total_nodes}</strong> · Params: <strong>{model.total_parameters:,}</strong> · Inputs: <code>{input_shapes}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ctx2:
        profile_options = list(profiles.keys())
        selected_id = st.selectbox(
            "Target Capability Profile:",
            options=profile_options,
            index=profile_options.index(curr_profile_id) if curr_profile_id in profile_options else 0,
            format_func=lambda pid: f"{profiles[pid].display_name} ({profiles[pid].target_architecture})",
            key="opt_profile_select",
        )
        if selected_id != curr_profile_id:
            set_selected_profile_id(selected_id)

    selected_profile = profiles[selected_id]

    compat_report = CompatibilityAnalyzer.analyze(model, selected_profile)
    p_cnt = sum(1 for f in compat_report.findings if f.status == "PASS")
    f_cnt = sum(1 for f in compat_report.findings if f.status == "FAIL")
    w_cnt = sum(1 for f in compat_report.findings if f.status == "WARNING")
    u_cnt = sum(1 for f in compat_report.findings if f.status == "UNKNOWN")

    tag_color = "#059669" if compat_report.overall_assessment == "COMPATIBLE" else "#D97706"
    if f_cnt > 0:
        tag_color = "#DC2626"

    illustrative_warning = ""
    if selected_profile.is_illustrative:
        illustrative_warning = (
            '<div style="font-size: 0.76rem; color: #92400E; margin-top: 0.35rem; font-style: italic;">'
            '⚠️ Profile is an illustrative draft. Local host benchmarks do not prove or establish IBM Z compatibility.'
            '</div>'
        )

    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.65rem 1rem; margin-bottom: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                <div style="font-size: 0.8rem; font-weight: 600; color: #1E293B;">
                    Static Compatibility Status: <span style="color: {tag_color}; font-weight: 700;">{compat_report.overall_assessment}</span>
                </div>
                <div style="display: flex; gap: 0.4rem; font-size: 0.72rem; font-family: monospace;">
                    <span style="background: #ECFDF5; color: #065F46; padding: 0.1rem 0.35rem; border-radius: 3px; border: 1px solid #A7F3D0;">PASS: {p_cnt}</span>
                    <span style="background: #FEF2F2; color: #991B1B; padding: 0.1rem 0.35rem; border-radius: 3px; border: 1px solid #FECACA;">FAIL: {f_cnt}</span>
                    <span style="background: #FFFBEB; color: #92400E; padding: 0.1rem 0.35rem; border-radius: 3px; border: 1px solid #FDE68A;">WARN: {w_cnt}</span>
                    <span style="background: #F1F5F9; color: #334155; padding: 0.1rem 0.35rem; border-radius: 3px; border: 1px solid #CBD5E1;">UNKNOWN: {u_cnt}</span>
                </div>
            </div>
            {illustrative_warning}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return selected_profile


def _render_optimization_setup(
    model: ModelSummary,
    profile: CapabilityProfile,
    env_info: Any,
) -> Dict[str, Any]:
    """Render Phase 12 search strategy, recommendation goal, parameters, and constraints."""
    with st.expander("Optimization Search Strategy & Operational Constraints", expanded=True):
        col_strat, col_goal, col_budget = st.columns([1, 1, 1])

        with col_strat:
            st.markdown("**1. Search Strategy**")
            strat_map = {
                "Grid Search (Bounded Cartesian)": SearchStrategy.GRID,
                "Random Search (Reproducible Seed)": SearchStrategy.RANDOM,
                "Successive Halving (Multi-Fidelity)": SearchStrategy.SUCCESSIVE_HALVING,
            }
            strat_label = st.selectbox(
                "Exploration Strategy:",
                options=list(strat_map.keys()),
                index=0,
                help="Grid: exhaustive over bounds. Random: seeded stochastic sampling. Successive Halving: early pruning of slow candidates.",
            )
            selected_strategy = strat_map[strat_label]

        with col_goal:
            st.markdown("**2. Decision Recommendation Goal**")
            goal_map = {
                "Balanced Multi-Objective Utility": RecommendationGoal.BALANCED_SCORE,
                "Minimize Median Latency": RecommendationGoal.MINIMIZE_LATENCY,
                "Maximize Throughput": RecommendationGoal.MAXIMIZE_THROUGHPUT,
                "Minimize Memory RSS": RecommendationGoal.MINIMIZE_MEMORY,
            }
            goal_label = st.selectbox(
                "Recommendation Goal:",
                options=list(goal_map.keys()),
                index=0,
                help="Governs the final configuration recommendation and ranking policy.",
            )
            selected_goal = goal_map[goal_label]

        with col_budget:
            st.markdown("**3. Exploration Budget**")
            max_cands = st.slider(
                "Max Candidate Configurations:",
                min_value=2,
                max_value=16,
                value=6,
                help="Maximum number of candidates generated and evaluated.",
            )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        col_dims, col_constraints = st.columns([1.2, 1])

        with col_dims:
            st.markdown("**4. Search Parameter Dimensions**")
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                batch_choices = st.multiselect(
                    "Batch Sizes:",
                    options=[1, 2, 4, 8],
                    default=[1, 2],
                    help="Batch sizes to evaluate.",
                )
                thread_choices = st.multiselect(
                    "Intra-op Threads:",
                    options=[1, 2, 4, 8],
                    default=[1, 2, 4],
                    help="Parallel threads used within operator execution.",
                )
            with c_d2:
                graph_opts = st.multiselect(
                    "ORT Graph Optimization Levels:",
                    options=["ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"],
                    default=["ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"],
                    help="ONNX Runtime graph optimization pass levels.",
                )
                exec_modes = st.multiselect(
                    "Execution Modes:",
                    options=["ORT_SEQUENTIAL", "ORT_PARALLEL"],
                    default=["ORT_SEQUENTIAL"],
                    help="Operator dispatch mode.",
                )

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                warmup_cnt = st.number_input("Warm-up Runs:", min_value=1, max_value=100, value=5, step=5)
            with col_b2:
                measured_cnt = st.number_input("Measured Iterations:", min_value=5, max_value=500, value=25, step=10)
            with col_b3:
                seed_val = st.number_input("Random Seed:", min_value=0, max_value=99999, value=42, step=1)

        with col_constraints:
            st.markdown("**5. Hard Operational Constraints (Non-Negotiable)**")
            st.caption("Candidates violating mandatory constraints will be marked INFEASIBLE.")

            hard_constraints: List[HardConstraint] = []

            enable_max_lat = st.checkbox("Maximum Median Latency (ms)", value=False)
            if enable_max_lat:
                max_lat_val = st.number_input("Threshold Latency (ms):", min_value=0.001, max_value=5000.0, value=5.0, step=0.1)
                hard_constraints.append(
                    HardConstraint(
                        name="Maximum Median Latency",
                        metric_field="median_latency_ms",
                        operator="<=",
                        threshold=float(max_lat_val),
                        unit="ms",
                    )
                )

            enable_min_tp = st.checkbox("Minimum Throughput (inf/sec)", value=False)
            if enable_min_tp:
                min_tp_val = st.number_input("Threshold Throughput (inf/s):", min_value=1.0, max_value=1000000.0, value=500.0, step=50.0)
                hard_constraints.append(
                    HardConstraint(
                        name="Minimum Throughput",
                        metric_field="throughput_inf_sec",
                        operator=">=",
                        threshold=float(min_tp_val),
                        unit="inf/sec",
                    )
                )

            enable_max_p95 = st.checkbox("Maximum P95 Latency (ms)", value=False)
            if enable_max_p95:
                max_p95_val = st.number_input("Threshold P95 (ms):", min_value=0.001, max_value=10000.0, value=10.0, step=0.5)
                hard_constraints.append(
                    HardConstraint(
                        name="Maximum P95 Latency",
                        metric_field="p95_latency_ms",
                        operator="<=",
                        threshold=float(max_p95_val),
                        unit="ms",
                    )
                )

    opt_req = OptimizationRequest(
        model_fingerprint=model.sha256_hash,
        profile_id=getattr(profile, "profile_id", getattr(profile, "id", "")),
        profile_version=profile.version,
        primary_objective=OptimizationObjective.MINIMIZE_MEDIAN_LATENCY,
        hard_constraints=hard_constraints,
        max_candidates=int(max_cands),
        batch_sizes=batch_choices if batch_choices else [1],
        intra_op_threads=thread_choices if thread_choices else [1],
        inter_op_threads=[1],
        warmup_iterations=int(warmup_cnt),
        measured_iterations=int(measured_cnt),
        repeated_rounds=2,
        random_seed=int(seed_val),
    )

    return {
        "request": opt_req,
        "strategy": selected_strategy,
        "goal": selected_goal,
        "graph_opts": graph_opts or ["ORT_ENABLE_EXTENDED"],
        "exec_modes": exec_modes or ["ORT_SEQUENTIAL"],
    }


def _render_optimization_results(raw_report: Union[OptimizationReport, Dict[str, Any]]) -> None:
    """Render comprehensive Phase 12 results organized across 5 analytical tabs."""
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    if isinstance(raw_report, dict):
        report_dict = raw_report
        candidates = report_dict.get("candidates", [])
        best_id = report_dict.get("best_candidate_id")
        base_id = report_dict.get("baseline_candidate_id")
        counts = report_dict.get("summary_counts", {})
        status_val = "COMPLETED" if counts.get("feasible", 0) > 0 else "COMPLETED_WITH_WARNINGS"
        best_cand = next((c for c in candidates if c.get("candidate_id") == best_id), None)
        base_cand = next((c for c in candidates if c.get("candidate_id") == base_id), None)
        spd = best_cand.get("speedup_vs_baseline") if best_cand else 1.0
        obj_label = report_dict.get("recommendation_goal", "BALANCED_SCORE").replace("_", " ").title()
    else:
        # Fallback for Phase 5 dataclass
        report_dict = raw_report.to_dict()
        candidates = report_dict.get("candidates", [])
        best_id = raw_report.best_candidate_id
        base_id = raw_report.baseline_candidate_id
        counts = raw_report.summary_counts
        status_val = raw_report.status.value
        best_cand = next((c for c in raw_report.candidates if c.candidate_id == best_id), None)
        base_cand = next((c for c in raw_report.candidates if c.candidate_id == base_id), None)
        spd = best_cand.speedup_vs_baseline if best_cand else 1.0
        obj_label = raw_report.primary_objective.replace("_", " ").title()

    # 1. Executive Banner
    render_optimization_banner(
        status=status_val,
        best_candidate_id=best_id,
        speedup=spd,
        feasible_count=counts.get("feasible", 0),
        total_benchmarked=counts.get("benchmarked", 0),
        objective_label=obj_label,
    )

    # 2. Executive Metrics Cards
    if best_cand and base_cand:
        bm = base_cand.get("metrics", {}) if isinstance(base_cand, dict) else base_cand.get_metrics_dict()
        om = best_cand.get("metrics", {}) if isinstance(best_cand, dict) else best_cand.get_metrics_dict()
        lat_red = best_cand.get("latency_reduction_pct", 0.0) if isinstance(best_cand, dict) else (best_cand.latency_reduction_pct or 0.0)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_metric_box(label="Relative Speedup", value=f"{spd:.2f}x", subtext="Over default baseline")
        with col2:
            render_metric_box(label="Latency Reduction", value=f"{lat_red:+.1f}%", subtext=f"Base: {bm.get('median_latency_ms', 0):.2f} ms")
        with col3:
            render_metric_box(label="Optimal Latency", value=f"{om.get('median_latency_ms', 0):.3f} ms", subtext=f"P95: {om.get('p95_latency_ms', 0):.3f} ms")
        with col4:
            render_metric_box(label="Optimal Throughput", value=f"{om.get('throughput_inf_sec', 0):.1f} inf/s", subtext=f"Base: {bm.get('throughput_inf_sec', 0):.1f} inf/s")
        with col5:
            render_metric_box(label="Memory Delta (RSS)", value=f"{om.get('memory_delta_mb', 0):+.2f} MB", subtext=f"Peak: {om.get('peak_memory_mb', 0):.1f} MB")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 3. Five-Tab Enterprise Decision Support Workspace
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Multi-Objective Pareto Frontier",
        "Evaluated Candidates",
        "Sensitivity Analysis",
        "Evidence-Backed Recommendation",
        "Enterprise Audit Report",
    ])

    # --- TAB 1: Pareto Frontier & Trade-offs ---
    with tab1:
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 0.5rem 0 0.5rem 0;">
                Multi-Objective Trade-Off Frontier (Median Latency vs. Throughput)
            </div>
            """,
            unsafe_allow_html=True,
        )

        plot_data = []
        for c in candidates:
            m = c.get("metrics", {}) if isinstance(c, dict) else c.get_metrics_dict()
            cfg = c.get("config", {}) if isinstance(c, dict) else c.config.to_dict()
            lat = m.get("median_latency_ms")
            tp = m.get("throughput_inf_sec")
            is_pareto = c.get("is_pareto_optimal", False) if isinstance(c, dict) else c.is_pareto_optimal
            is_feas = c.get("is_feasible", False) if isinstance(c, dict) else c.is_feasible

            if lat is not None and tp is not None:
                p_cat = "Pareto Frontier" if is_pareto else ("Feasible (Dominated)" if is_feas else "Infeasible")
                plot_data.append(
                    {
                        "Candidate ID": c.get("candidate_id") if isinstance(c, dict) else c.candidate_id,
                        "Median Latency (ms)": round(lat, 4),
                        "Throughput (inf/s)": round(tp, 1),
                        "Batch Size": cfg.get("batch_size"),
                        "Intra Threads": cfg.get("intra_op_num_threads") or "Default",
                        "Graph Opt": cfg.get("graph_optimization_level") or "Default",
                        "Category": p_cat,
                    }
                )

        if plot_data:
            df_plot = pd.DataFrame(plot_data)
            chart = (
                alt.Chart(df_plot)
                .mark_circle(size=140)
                .encode(
                    x=alt.X("Median Latency (ms):Q", scale=alt.Scale(zero=False), title="Median Latency (ms) [Lower is Better]"),
                    y=alt.Y("Throughput (inf/s):Q", scale=alt.Scale(zero=False), title="Throughput (inf/sec) [Higher is Better]"),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(
                            domain=["Pareto Frontier", "Feasible (Dominated)", "Infeasible"],
                            range=["#2563EB", "#94A3B8", "#EF4444"],
                        ),
                        legend=alt.Legend(title="Frontier Status"),
                    ),
                    tooltip=[
                        "Candidate ID",
                        "Category",
                        "Median Latency (ms)",
                        "Throughput (inf/s)",
                        "Batch Size",
                        "Intra Threads",
                        "Graph Opt",
                    ],
                )
                .properties(height=360)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        pareto_ids = report_dict.get("pareto_frontier_ids", [])
        st.markdown(f"**Identified Pareto-Optimal Candidates ({len(pareto_ids)}):**")
        for pid in pareto_ids:
            st.markdown(f"- <code>{pid}</code>: Represents an optimal non-dominated trade-off point.", unsafe_allow_html=True)

    # --- TAB 2: Evaluated Candidates & Rankings ---
    with tab2:
        table_rows = []
        for c in candidates:
            m = c.get("metrics", {}) if isinstance(c, dict) else c.get_metrics_dict()
            cfg = c.get("config", {}) if isinstance(c, dict) else c.config.to_dict()
            is_pareto = c.get("is_pareto_optimal", False) if isinstance(c, dict) else c.is_pareto_optimal
            is_feas = c.get("is_feasible", False) if isinstance(c, dict) else c.is_feasible
            status_str = "PARETO OPTIMAL" if is_pareto else ("FEASIBLE" if is_feas else "INFEASIBLE")

            rel = c.get("reliability")
            rel_str = rel.get("rating", "N/A") if rel else "N/A"

            spd_val = c.get("speedup_vs_baseline") if isinstance(c, dict) else c.speedup_vs_baseline
            rank_val = c.get("rank") if isinstance(c, dict) else c.rank

            table_rows.append(
                {
                    "Rank": f"#{rank_val}" if rank_val else "—",
                    "Candidate ID": c.get("candidate_id") if isinstance(c, dict) else c.candidate_id,
                    "Status": status_str,
                    "Provider": cfg.get("execution_provider"),
                    "Batch": cfg.get("batch_size"),
                    "Intra Threads": cfg.get("intra_op_num_threads") or "Default",
                    "Graph Opt": cfg.get("graph_optimization_level") or "Default",
                    "Median Latency": f"{m.get('median_latency_ms', 0):.3f} ms" if m.get("median_latency_ms") else "N/A",
                    "P95 Latency": f"{m.get('p95_latency_ms', 0):.3f} ms" if m.get("p95_latency_ms") else "N/A",
                    "Throughput": f"{m.get('throughput_inf_sec', 0):.1f} inf/s" if m.get("throughput_inf_sec") else "N/A",
                    "Speedup": f"{spd_val:.2f}x" if spd_val else "—",
                    "Stability": rel_str,
                }
            )

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # --- TAB 3: Sensitivity Analysis ---
    with tab3:
        sens_data = report_dict.get("sensitivity", {})
        params = sens_data.get("parameters", {})

        st.markdown(f"**Sensitivity Summary:** {sens_data.get('summary_insight', 'Parameter elasticity analysis.')}")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Batch Size Elasticity")
            b_info = params.get("batch_size", {})
            if b_info.get("status") == "ANALYZED":
                st.info(b_info.get("interpretation", ""))
                pts = b_info.get("data_points", [])
                if pts:
                    df_b = pd.DataFrame(pts)
                    st.dataframe(df_b[["param_value", "median_latency_ms", "throughput_inf_sec"]], hide_index=True)
            else:
                st.caption("Insufficient batch size data points (< 2) tested in this experiment.")

        with col_s2:
            st.markdown("#### Intra-Op Thread Scaling Response")
            t_info = params.get("intra_op_num_threads", {})
            if t_info.get("status") == "ANALYZED":
                st.info(t_info.get("interpretation", ""))
                pts = t_info.get("data_points", [])
                if pts:
                    df_t = pd.DataFrame(pts)
                    st.dataframe(df_t[["param_value", "median_latency_ms", "throughput_inf_sec"]], hide_index=True)
            else:
                st.caption("Insufficient thread count data points (< 2) tested in this experiment.")

    # --- TAB 4: Evidence-Based Recommendation ---
    with tab4:
        rec = report_dict.get("recommendation", {})
        if rec and rec.get("recommended_candidate_id"):
            st.markdown(
                f"""
                <div style="background: #F8FAFC; border: 1px solid #3B82F6; border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.76rem; font-weight: 700; color: #2563EB; text-transform: uppercase;">
                        Recommended Deployment Configuration · {rec.get('goal', 'BALANCED_SCORE')}
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 0.25rem 0;">
                        Candidate <code>{rec.get('recommended_candidate_id')}</code>
                    </div>
                    <div style="font-size: 0.88rem; color: #334155; line-height: 1.5; margin: 0.5rem 0;">
                        {rec.get('rationale')}
                    </div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.5rem;">
                        {rec.get('tradeoffs_summary')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            rel_data = rec.get("reliability")
            if rel_data:
                st.markdown(
                    f"**Measurement Confidence Rating:** `{rel_data.get('rating')}` · "
                    f"Sample Count: `{rel_data.get('sample_count')}` · "
                    f"Dispersion: IQR = `{rel_data.get('iqr_ms')} ms`, CV = `{rel_data.get('cv_percent')}%`"
                )

            st.warning("Hardware Verification Notice: “IBM Z target validation remains pending; only the listed local checks have been verified.”")
        else:
            st.warning(report_dict.get("recommendation_rationale", "No feasible candidate met all operational constraints."))

    # --- TAB 5: Enterprise Audit Report & Export ---
    with tab5:
        st.markdown("**18-Section Enterprise Deployment Optimization Report:**")

        md_content = report_dict.get("markdown_report") or EnterpriseReportGenerator.generate_markdown(report_dict)
        json_content = report_dict.get("json_report") or EnterpriseReportGenerator.generate_json(report_dict)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="Download Markdown Report (.md)",
                data=md_content,
                file_name=f"{report_dict.get('experiment_id', 'report')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_d2:
            st.download_button(
                label="Download Audit Schema (.json)",
                data=json_content,
                file_name=f"{report_dict.get('experiment_id', 'report')}.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("Preview Full Markdown Audit Report", expanded=True):
            st.markdown(md_content)
