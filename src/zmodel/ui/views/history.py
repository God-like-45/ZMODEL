"""Experiment History Streamlit View — PostgreSQL durable search, audit, and run comparison."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
import altair as alt
import pandas as pd
import streamlit as st
from zmodel.db.comparator import compare_benchmark_runs
from zmodel.db.config import check_database_connection, mask_database_url
from zmodel.db.models import (
    BenchmarkRunORM,
    CandidateEvaluationORM,
    ModelIdentityORM,
    OptimizationExperimentORM,
)
from zmodel.db.repositories import (
    BenchmarkRepository,
    HistoryOverviewRepository,
    ModelRepository,
    OptimizationRepository,
)
from zmodel.ui.components import (
    render_metric_box,
    render_workspace_header,
)


def render_history_view() -> None:
    """Render the complete Experiment History workspace."""
    render_workspace_header(
        title="Experiment History & Benchmark Registry",
        description="Durable PostgreSQL experiment tracking, reproducible run diffing, and model registry.",
        environment_badge="Phase 6 · Active",
    )

    # 1. Database Health & Diagnostics
    db_status = check_database_connection()
    _render_db_status_strip(db_status)

    if not db_status.get("connected"):
        _render_db_disconnected_banner(db_status)
        return

    # 2. History Overview Metric Cards
    overview_repo = HistoryOverviewRepository()
    try:
        overview = overview_repo.get_overview_metrics()
    except Exception as ex:
        st.error(f"Failed to query database metrics: {ex}")
        return

    _render_overview_kpis(overview)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # 3. Main Workspace Navigation Tabs
    tab_exps, tab_benchs, tab_models, tab_diag = st.tabs([
        "🔬 Optimization Experiments",
        "⚡ Benchmark Runs & Comparison",
        "📦 Model Registry",
        "⚙️ Database & Migrations",
    ])

    with tab_exps:
        _render_optimization_experiments_tab()

    with tab_benchs:
        _render_benchmark_runs_tab()

    with tab_models:
        _render_model_registry_tab()

    with tab_diag:
        _render_db_diagnostics_tab(db_status)


def _render_db_status_strip(db_status: Dict[str, Any]) -> None:
    """Render a subtle horizontal status strip indicating PostgreSQL connection state."""
    is_connected = db_status.get("connected", False)
    dialect = db_status.get("dialect", "unknown").upper()
    masked_url = db_status.get("database_url_masked", "Not Configured")

    pill_color = "#059669" if is_connected else "#DC2626"
    pill_bg = "#ECFDF5" if is_connected else "#FEF2F2"
    pill_border = "#A7F3D0" if is_connected else "#FECACA"
    status_label = f"Connected ({dialect})" if is_connected else "Disconnected"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.5rem 0.85rem; margin-bottom: 1.25rem; font-size: 0.78rem;">
            <div style="display: flex; align-items: center; gap: 0.65rem;">
                <span style="display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.2rem 0.55rem; background: {pill_bg}; color: {pill_color}; border: 1px solid {pill_border}; border-radius: 9999px; font-weight: 600; font-size: 0.72rem;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {pill_color};"></span>
                    {status_label}
                </span>
                <span style="color: #64748B;">Target Database:</span>
                <span style="font-family: var(--zm-font-mono); color: #0F172A; font-weight: 500;">{masked_url}</span>
            </div>
            <div style="color: #64748B; font-size: 0.75rem;">
                Schema Version: <span style="font-weight: 600; color: #0F172A;">1.0.0</span> · Alembic Revision: <span style="font-weight: 600; color: #0F172A;">001_initial_schema</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_db_disconnected_banner(db_status: Dict[str, Any]) -> None:
    """Render a helpful guidance card when PostgreSQL is not reachable."""
    err_msg = db_status.get("error", "Unknown connection error")
    st.markdown(
        f"""
        <div style="background: #FFFBEB; border: 1px solid #FCD34D; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem; color: #92400E; font-weight: 700; font-size: 0.95rem;">
                ⚠️ PostgreSQL Database Unreachable
            </div>
            <div style="color: #78350F; font-size: 0.84rem; margin-top: 0.5rem; line-height: 1.5;">
                Z-ModelPilot requires a running PostgreSQL instance for durable experiment history.
                The application attempted to connect to:
                <code style="background: #FEF3C7; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.8rem;">{db_status.get('database_url_masked')}</code>
            </div>
            <div style="background: #FFFFFF; border: 1px solid #FDE68A; border-radius: 6px; padding: 0.75rem; margin-top: 0.75rem; font-family: var(--zm-font-mono); font-size: 0.78rem; color: #B45309; word-break: break-all;">
                {err_msg}
            </div>
            <div style="margin-top: 0.85rem; font-size: 0.82rem; color: #92400E;">
                <strong>Quick Fix Options:</strong>
                <ol style="margin-top: 0.35rem; padding-left: 1.2rem; line-height: 1.5;">
                    <li>Start a local PostgreSQL container: <code>docker run -d --name zmodel-postgres -e POSTGRES_USER=zmodel_user -e POSTGRES_PASSWORD=zmodel_password -e POSTGRES_DB=zmodel_db -p 5432:5432 postgres:16-alpine</code></li>
                    <li>Or set <code>DATABASE_URL</code> in your environment / <code>.env</code> file.</li>
                    <li>For offline unit testing, set <code>DB_ALLOW_SQLITE_FALLBACK=true</code> in your environment.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview_kpis(overview: Dict[str, Any]) -> None:
    """Render top metric cards for total experiments, benchmarks, and models."""
    col1, col2, col3, col4 = st.columns(4)

    total_exps = overview.get("total_experiments", 0)
    exp_counts = overview.get("experiment_status_counts", {})
    completed_exps = exp_counts.get("COMPLETED", 0) + exp_counts.get("COMPLETED_WITH_WARNINGS", 0)
    failed_exps = exp_counts.get("FAILED", 0)

    total_benchs = overview.get("total_benchmarks", 0)
    bench_counts = overview.get("benchmark_status_counts", {})
    success_benchs = bench_counts.get("SUCCESS", 0)

    total_models = overview.get("total_models", 0)

    with col1:
        render_metric_box(
            label="Total Experiments",
            value=f"{total_exps}",
            subtext=f"{completed_exps} completed · {failed_exps} failed",
        )
    with col2:
        render_metric_box(
            label="Total Benchmark Runs",
            value=f"{total_benchs}",
            subtext=f"{success_benchs} successful executions",
        )
    with col3:
        render_metric_box(
            label="Registered Models",
            value=f"{total_models}",
            subtext="Tracked by SHA-256 hash",
        )
    with col4:
        render_metric_box(
            label="Durable Storage",
            value="Active",
            subtext="PostgreSQL ACID transactions",
        )


def _render_optimization_experiments_tab() -> None:
    """Tab 1: Search, filter, and inspect persisted optimization experiments."""
    opt_repo = OptimizationRepository()
    model_repo = ModelRepository()

    models = model_repo.list_models(limit=100)
    model_options = ["All Models"] + [f"{m.original_filename} ({m.fingerprint[:8]})" for m in models]
    fingerprint_map = {f"{m.original_filename} ({m.fingerprint[:8]})": m.fingerprint for m in models}

    # Filters row
    col_m, col_s, col_o = st.columns([3, 2, 2])
    with col_m:
        sel_model_str = st.selectbox("Filter by Model", options=model_options, key="hist_exp_model_filter")
    with col_s:
        sel_status = st.selectbox(
            "Filter by Status",
            options=["All", "COMPLETED", "COMPLETED_WITH_WARNINGS", "FAILED"],
            key="hist_exp_status_filter",
        )
    with col_o:
        sel_objective = st.selectbox(
            "Filter by Objective",
            options=["All", "MINIMIZE_MEDIAN_LATENCY", "MINIMIZE_P95_LATENCY", "MAXIMIZE_THROUGHPUT", "MINIMIZE_MEMORY", "EXPLORE_TRADEOFFS"],
            key="hist_exp_obj_filter",
        )

    target_fingerprint = fingerprint_map.get(sel_model_str) if sel_model_str != "All Models" else None

    # Query experiments
    experiments, total_count = opt_repo.list_optimization_experiments(
        fingerprint=target_fingerprint,
        status=sel_status,
        objective=sel_objective,
        limit=50,
    )

    if not experiments:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 2.5rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 1.5rem; color: #94A3B8;">🔬</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #334155; margin-top: 0.5rem;">No Optimization Experiments Found</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.25rem;">
                    Run an optimization search from the <strong>Optimization</strong> tab to persist reproducible experiments here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"<div style='font-size: 0.82rem; color: #64748B; margin-top: 0.5rem; margin-bottom: 0.75rem;'>Showing <strong>{len(experiments)}</strong> of <strong>{total_count}</strong> saved experiments</div>", unsafe_allow_html=True)

    # Experiments Table
    exp_rows = []
    for exp in experiments:
        dt_str = exp.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if exp.created_at else "N/A"
        summary = exp.summary_counts or {}
        total_c = summary.get("total_generated", summary.get("candidates_total", 0))
        feas_c = summary.get("feasible", summary.get("candidates_feasible", 0))
        best_cand = exp.best_candidate_id or "None"
        exp_rows.append({
            "Experiment ID": exp.id,
            "Date": dt_str,
            "Model": exp.model_name,
            "Fingerprint": exp.model_fingerprint[:12] + "...",
            "Profile": exp.profile_id,
            "Objective": exp.primary_objective,
            "Status": exp.status,
            "Candidates": f"{total_c} (Feasible: {feas_c})",
            "Best Candidate": best_cand,
        })

    df_exps = pd.DataFrame(exp_rows)
    st.dataframe(df_exps, use_container_width=True, hide_index=True)

    # Experiment Inspector / Drill-down
    st.markdown("---")
    st.markdown("#### Inspect Saved Experiment")

    exp_ids = [e.id for e in experiments]
    selected_exp_id = st.selectbox("Select Experiment ID to inspect", options=exp_ids, key="hist_inspect_exp_id")

    if selected_exp_id:
        target_exp = opt_repo.get_optimization_experiment(selected_exp_id)
        if target_exp:
            candidates = opt_repo.get_candidates_for_experiment(selected_exp_id)
            _render_experiment_detail(target_exp, candidates)


def _render_experiment_detail(
    exp: OptimizationExperimentORM,
    candidates: Optional[List[CandidateEvaluationORM]] = None,
) -> None:
    """Render comprehensive drilldown details for a saved optimization experiment."""
    cands = candidates if candidates is not None else []
    st.markdown(
        f"""
        <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.25rem; margin-top: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="font-size: 1rem; font-weight: 700; color: #0F172A; font-family: var(--zm-font-mono);">{exp.id}</div>
                    <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.25rem;">
                        Model: <span style="color: #0F172A; font-weight: 600;">{exp.model_name}</span> · Fingerprint: <code>{exp.model_fingerprint}</code>
                    </div>
                    <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.2rem;">
                        Profile: <span style="color: #0F172A; font-weight: 600;">{exp.profile_name}</span> (v{exp.profile_version}) · Objective: <span style="color: #1D4ED8; font-weight: 600;">{exp.primary_objective}</span>
                    </div>
                </div>
                <div>
                    <span style="font-size: 0.78rem; font-weight: 600; padding: 0.25rem 0.65rem; border-radius: 4px; background: {'#ECFDF5' if 'COMPLETED' in exp.status else '#FEF2F2'}; color: {'#059669' if 'COMPLETED' in exp.status else '#DC2626'}; border: 1px solid {'#A7F3D0' if 'COMPLETED' in exp.status else '#FECACA'};">
                        {exp.status}
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Candidate Highlight
    if exp.best_candidate_id:
        best_c = next((c for c in cands if c.candidate_id == exp.best_candidate_id), None)
        if best_c:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_metric_box(
                    label="Winning Candidate",
                    value=best_c.candidate_id,
                    subtext=f"Batch: {best_c.configuration.get('batch_size')} · Intra: {best_c.configuration.get('intra_op_num_threads')}",
                )
            with col2:
                sp_str = f"{best_c.speedup_vs_baseline:.2f}x" if best_c.speedup_vs_baseline else "N/A"
                render_metric_box(
                    label="Measured Speedup",
                    value=sp_str,
                    subtext="vs baseline run",
                )
            with col3:
                lat_str = f"{best_c.measured_median_latency_ms:.4f} ms" if best_c.measured_median_latency_ms is not None else "N/A"
                render_metric_box(
                    label="Median Latency",
                    value=lat_str,
                    subtext=f"Reduction: {best_c.latency_reduction_pct:.1f}%" if best_c.latency_reduction_pct is not None else "",
                )
            with col4:
                tp_str = f"{best_c.measured_throughput_ips:,.0f} inf/s" if best_c.measured_throughput_ips is not None else "N/A"
                render_metric_box(
                    label="Throughput",
                    value=tp_str,
                    subtext="Empirically measured",
                )

    # Recommendation Rationale
    if exp.recommendation_rationale:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border-left: 4px solid #3B82F6; padding: 0.85rem 1rem; border-radius: 0 6px 6px 0; margin-top: 1rem; font-size: 0.84rem; color: #1E293B; line-height: 1.5;">
                <strong>Recommendation Rationale:</strong><br>
                {exp.recommendation_rationale}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Candidates Evaluation Table
    st.markdown("<div style='height: 0.85rem;'></div>", unsafe_allow_html=True)
    st.markdown("##### Candidate Evaluations & Pareto Membership")

    cand_rows = []
    for c in cands:
        cfg = c.configuration or {}
        speedup_str = f"{c.speedup_vs_baseline:.2f}x" if c.speedup_vs_baseline else ("1.00x (Baseline)" if c.is_baseline else "—")
        lat_str = f"{c.measured_median_latency_ms:.4f}" if c.measured_median_latency_ms is not None else "—"
        tp_str = f"{c.measured_throughput_ips:,.0f}" if c.measured_throughput_ips is not None else "—"
        mem_str = f"{c.measured_memory_mb:.1f}" if c.measured_memory_mb is not None else "—"

        cand_rows.append({
            "Candidate ID": c.candidate_id,
            "Batch": cfg.get("batch_size"),
            "Intra Threads": cfg.get("intra_op_num_threads"),
            "Inter Threads": cfg.get("inter_op_num_threads"),
            "Status": c.status,
            "Feasible": "YES" if c.is_feasible else ("NO (Rejected)" if c.status == "REJECTED" else "NO"),
            "Pareto": "★ OPTIMAL" if c.is_pareto_optimal else ("DOMINATED" if c.is_feasible else "—"),
            "Rank": c.rank or "—",
            "Median Latency (ms)": lat_str,
            "Throughput (inf/s)": tp_str,
            "Peak RSS (MB)": mem_str,
            "Speedup": speedup_str,
            "Rejection Reason": c.rejection_reason or "—",
        })

    df_cands = pd.DataFrame(cand_rows)
    st.dataframe(df_cands, use_container_width=True, hide_index=True)

    # Interactive Altair Pareto Scatter Plot (if 2+ measured candidates exist)
    measured_cands = [c for c in cands if c.measured_median_latency_ms is not None and c.measured_throughput_ips is not None]
    if len(measured_cands) >= 2:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### Historical Pareto Frontier Plot")
        plot_data = pd.DataFrame([
            {
                "Candidate": c.candidate_id,
                "Latency (ms)": c.measured_median_latency_ms,
                "Throughput (inf/s)": c.measured_throughput_ips,
                "Classification": "Pareto Optimal" if c.is_pareto_optimal else ("Feasible (Dominated)" if c.is_feasible else "Infeasible"),
            }
            for c in measured_cands
        ])

        chart = (
            alt.Chart(plot_data)
            .mark_circle(size=140)
            .encode(
                x=alt.X("Latency (ms):Q", title="Median Latency (ms) — Lower is Better"),
                y=alt.Y("Throughput (inf/s):Q", title="Throughput (inf/s) — Higher is Better"),
                color=alt.Color(
                    "Classification:N",
                    scale=alt.Scale(
                        domain=["Pareto Optimal", "Feasible (Dominated)", "Infeasible"],
                        range=["#10B981", "#3B82F6", "#EF4444"],
                    ),
                ),
                tooltip=["Candidate", "Latency (ms)", "Throughput (inf/s)", "Classification"],
            )
            .properties(height=280)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    # Export Buttons
    col_j, col_c = st.columns([1, 1])
    with col_j:
        json_str = json.dumps(exp.to_dict(candidates=cands), indent=2)
        st.download_button(
            label="Download Experiment JSON",
            data=json_str,
            file_name=f"experiment_{exp.id}.json",
            mime="application/json",
            key=f"dl_json_{exp.id}",
        )
    with col_c:
        csv_str = df_cands.to_csv(index=False)
        st.download_button(
            label="Download Candidates CSV",
            data=csv_str,
            file_name=f"candidates_{exp.id}.csv",
            mime="text/csv",
            key=f"dl_csv_{exp.id}",
        )


def _render_benchmark_runs_tab() -> None:
    """Tab 2: Benchmark Runs List and Side-by-Side Comparability Verification."""
    bench_repo = BenchmarkRepository()
    model_repo = ModelRepository()

    models = model_repo.list_models(limit=100)
    model_options = ["All Models"] + [f"{m.original_filename} ({m.fingerprint[:8]})" for m in models]
    fingerprint_map = {f"{m.original_filename} ({m.fingerprint[:8]})": m.fingerprint for m in models}

    col_m, col_p, col_s = st.columns([3, 2, 2])
    with col_m:
        sel_model = st.selectbox("Filter Benchmark Runs by Model", options=model_options, key="hist_bench_model_filter")
    with col_p:
        sel_provider = st.selectbox("Provider", options=["All", "CPUExecutionProvider", "AzureExecutionProvider"], key="hist_bench_prov_filter")
    with col_s:
        sel_status = st.selectbox("Status", options=["All", "SUCCESS", "FAILED"], key="hist_bench_status_filter")

    target_fp = fingerprint_map.get(sel_model) if sel_model != "All Models" else None

    runs, total_count = bench_repo.list_benchmark_runs(
        fingerprint=target_fp,
        status=sel_status,
        provider=sel_provider,
        limit=50,
    )

    if not runs:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 8px; padding: 2.5rem; text-align: center; margin-top: 1rem;">
                <div style="font-size: 1.5rem; color: #94A3B8;">⚡</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #334155; margin-top: 0.5rem;">No Benchmark Runs Found</div>
                <div style="font-size: 0.8rem; color: #64748B; margin-top: 0.25rem;">
                    Run benchmarks from the <strong>Benchmarks</strong> or <strong>Optimization</strong> tabs to persist empirical runs here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(f"<div style='font-size: 0.82rem; color: #64748B; margin-top: 0.5rem; margin-bottom: 0.75rem;'>Showing <strong>{len(runs)}</strong> of <strong>{total_count}</strong> saved benchmark executions</div>", unsafe_allow_html=True)

    # Benchmark Runs Table
    run_rows = []
    for r in runs:
        cfg = r.configuration or {}
        dt_str = r.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if r.created_at else "N/A"
        lat_str = f"{r.median_latency_ms:.4f}" if r.median_latency_ms is not None else "—"
        tp_str = f"{r.inferences_per_second:,.0f}" if r.inferences_per_second is not None else "—"
        mem_str = f"{r.session_delta_mb:.1f}" if r.session_delta_mb is not None else "—"

        run_rows.append({
            "Run ID": r.id,
            "Date": dt_str,
            "Model": r.model_name,
            "Fingerprint": r.model_fingerprint[:12] + "...",
            "Provider": r.execution_provider,
            "Batch": cfg.get("batch_size", 1),
            "Intra Threads": cfg.get("intra_op_num_threads", 1),
            "Correctness": r.correctness_status,
            "Median Latency (ms)": lat_str,
            "Throughput (inf/s)": tp_str,
            "Session Δ (MB)": mem_str,
            "Status": r.status,
        })

    df_runs = pd.DataFrame(run_rows)
    st.dataframe(df_runs, use_container_width=True, hide_index=True)

    # 4. Side-by-Side Comparator Section
    st.markdown("---")
    st.markdown("#### Side-by-Side Run Comparison & Comparability Check")
    st.markdown(
        "<div style='font-size: 0.82rem; color: #64748B; margin-bottom: 0.75rem;'>"
        "Select two benchmark runs to compare relative performance. The comparator verifies that model fingerprints, "
        "batch sizes, and input shapes match before computing valid relative speedup."
        "</div>",
        unsafe_allow_html=True,
    )

    run_id_map = {r.id: r for r in runs}
    run_labels = [f"{r.id} | {r.model_name} (Batch: {r.configuration.get('batch_size')}, Threads: {r.configuration.get('intra_op_num_threads')}, Med: {r.median_latency_ms:.3f}ms)" if r.median_latency_ms else f"{r.id} | {r.model_name} ({r.status})" for r in runs]
    label_to_id = {lbl: r.id for lbl, r in zip(run_labels, runs)}

    col_a, col_b = st.columns(2)
    with col_a:
        sel_label_a = st.selectbox("Baseline Run (Reference A)", options=run_labels, index=0, key="hist_cmp_a")
    with col_b:
        default_b_idx = 1 if len(run_labels) > 1 else 0
        sel_label_b = st.selectbox("Comparison Candidate (Run B)", options=run_labels, index=default_b_idx, key="hist_cmp_b")

    id_a = label_to_id.get(sel_label_a)
    id_b = label_to_id.get(sel_label_b)

    if id_a and id_b:
        run_a = run_id_map.get(id_a)
        run_b = run_id_map.get(id_b)

        if run_a and run_b:
            comparison = compare_benchmark_runs(run_a, run_b)
            _render_benchmark_comparison_result(comparison, run_a, run_b)


def _render_benchmark_comparison_result(
    cmp: Any,
    run_a: BenchmarkRunORM,
    run_b: BenchmarkRunORM,
) -> None:
    """Render comparison metrics or honest disqualification warnings."""
    if not cmp.is_valid:
        st.markdown(
            f"""
            <div style="background: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 1.15rem; margin-top: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; color: #991B1B; font-weight: 700; font-size: 0.92rem;">
                    ⛔ Incompatible Benchmark Comparison Disqualified
                </div>
                <div style="font-size: 0.82rem; color: #7F1D1D; margin-top: 0.4rem; line-height: 1.5;">
                    The selected benchmark runs cannot be directly compared for speedup or relative improvement because of the following fundamental differences:
                </div>
                <ul style="margin-top: 0.5rem; margin-bottom: 0.25rem; padding-left: 1.25rem; font-size: 0.82rem; color: #991B1B;">
                    {"".join(f"<li>{r}</li>" for r in cmp.disqualification_reasons)}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Valid Comparison
    st.markdown(
        """
        <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.85rem; font-size: 0.8rem; color: #065F46; font-weight: 500;">
            ✓ Comparability Verified: Both runs use identical model fingerprint, validated batch size, and matching input shapes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Comparison Table
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### Metric Deltas & Speedup")
        cmp_rows = []
        for key, m in cmp.metric_comparisons.items():
            val_a_str = f"{m.value_a:.4f} {m.unit}" if m.value_a is not None else "N/A"
            val_b_str = f"{m.value_b:.4f} {m.unit}" if m.value_b is not None else "N/A"
            sp_str = f"{m.speedup_factor:.2f}x" if m.speedup_factor is not None else "—"
            imp_str = f"{m.percentage_improvement:+.1f}%" if m.percentage_improvement is not None else "—"

            cmp_rows.append({
                "Metric": m.metric_name,
                "Run A (Baseline)": val_a_str,
                "Run B (Candidate)": val_b_str,
                "Absolute Delta": f"{m.absolute_delta:+.4f} {m.unit}" if m.absolute_delta is not None else "—",
                "Improvement": imp_str,
                "Speedup": sp_str,
                "Interpretation": m.interpretation,
            })

        df_cmp = pd.DataFrame(cmp_rows)
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("##### Configuration Differences")
        if cmp.configuration_differences:
            diff_rows = []
            for k, (v_a, v_b) in cmp.configuration_differences.items():
                diff_rows.append({
                    "Parameter": k,
                    "Run A": str(v_a),
                    "Run B": str(v_b),
                })
            df_diff = pd.DataFrame(diff_rows)
            st.dataframe(df_diff, use_container_width=True, hide_index=True)
        else:
            st.info("Both runs share identical runtime configuration settings.")


def _render_model_registry_tab() -> None:
    """Tab 3: Model Registry listing all unique models tracked by SHA-256 fingerprint."""
    model_repo = ModelRepository()
    models = model_repo.list_models(limit=100)

    if not models:
        st.info("No models registered in the database yet. Inspect or benchmark an ONNX model to register it.")
        return

    st.markdown(f"<div style='font-size: 0.82rem; color: #64748B; margin-bottom: 0.75rem;'>Registered <strong>{len(models)}</strong> distinct ONNX models</div>", unsafe_allow_html=True)

    rows = []
    for m in models:
        meta = m.graph_metadata or {}
        opsets = m.opset_imports or {}
        opset_str = ", ".join(f"{k}:{v}" for k, v in opsets.items()) if opsets else "N/A"
        size_kb = f"{m.file_size_bytes / 1024:.1f} KB" if m.file_size_bytes else "Unknown"

        rows.append({
            "Original Filename": m.original_filename,
            "SHA-256 Fingerprint": m.fingerprint,
            "File Size": size_kb,
            "IR Version": m.ir_version or "N/A",
            "Nodes": meta.get("total_nodes", "N/A"),
            "Parameters": f"{meta.get('total_parameters', 0):,}" if meta.get("total_parameters") else "N/A",
            "Opsets": opset_str,
            "First Seen": m.first_seen_at.strftime("%Y-%m-%d %H:%M:%S UTC") if m.first_seen_at else "N/A",
            "Last Seen": m.last_seen_at.strftime("%Y-%m-%d %H:%M:%S UTC") if m.last_seen_at else "N/A",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_db_diagnostics_tab(db_status: Dict[str, Any]) -> None:
    """Tab 4: Database connection diagnostics, table schema, and Alembic migration instructions."""
    st.markdown("#### Database Configuration & Schema Diagnostics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 1rem; font-size: 0.82rem;">
                <div style="font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;">Connection Parameters</div>
                <div style="margin-bottom: 0.35rem;">Status: <strong style="color: {'#059669' if db_status.get('connected') else '#DC2626'};">{'CONNECTED' if db_status.get('connected') else 'DISCONNECTED'}</strong></div>
                <div style="margin-bottom: 0.35rem;">Dialect: <code>{db_status.get('dialect')}</code></div>
                <div style="margin-bottom: 0.35rem;">Masked URL: <code>{db_status.get('database_url_masked')}</code></div>
                <div style="margin-bottom: 0.35rem;">Detected Tables: <strong>{db_status.get('table_count', 0)}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 1rem; font-size: 0.82rem;">
                <div style="font-weight: 600; color: #0F172A; margin-bottom: 0.5rem;">Alembic Migration Commands</div>
                <div style="font-family: var(--zm-font-mono); font-size: 0.78rem; background: #FFFFFF; border: 1px solid #CBD5E1; padding: 0.5rem; border-radius: 4px; line-height: 1.5;">
                    # Apply latest schema migrations<br>
                    alembic upgrade head<br><br>
                    # Check current revision<br>
                    alembic current<br><br>
                    # Rollback one migration<br>
                    alembic downgrade -1
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("##### Normalized Tables in Schema")
    tables = [
        {"Table Name": "models", "Primary Key": "fingerprint (VARCHAR 64)", "Description": "Model identity registry keyed by SHA-256 fingerprint. No raw model bytes stored."},
        {"Table Name": "compatibility_assessments", "Primary Key": "id (VARCHAR 64)", "Description": "Historical capability profile evaluations, operator findings, and overall assessments."},
        {"Table Name": "benchmark_runs", "Primary Key": "id (VARCHAR 64)", "Description": "Actual empirical ONNX Runtime benchmarks, latency percentiles, throughput, and memory RSS."},
        {"Table Name": "optimization_experiments", "Primary Key": "id (VARCHAR 64)", "Description": "Multi-candidate optimization searches, objectives, constraints, and baseline references."},
        {"Table Name": "candidate_evaluations", "Primary Key": "id (VARCHAR 64)", "Description": "Granular candidate records, statuses, Pareto membership, and benchmark run references."},
    ]
    st.dataframe(pd.DataFrame(tables), use_container_width=True, hide_index=True)
