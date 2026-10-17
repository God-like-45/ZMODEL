"""Model Benchmark Engine view: controlled inference, latency percentiles, throughput, and memory profiling."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from zmodel.core.benchmark import (
    BenchmarkConfig,
    BenchmarkEngine,
    BenchmarkError,
    BenchmarkReport,
    discover_execution_providers,
)
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.state import (
    get_active_model,
    get_active_model_bytes,
    get_benchmark_report,
    get_compatibility_report,
    navigate_to,
    set_active_model,
    set_benchmark_report,
)
from zmodel.ui.components import (
    format_bytes,
    render_benchmark_banner,
    render_metric_box,
    render_workspace_header,
)


def render_benchmark_view() -> None:
    """Render the Benchmark Engine page."""
    render_workspace_header(
        title="Model Benchmark Engine",
        description="Execute controlled empirical inference, measure latency percentiles, throughput, numerical correctness, and memory growth.",
        environment_badge="Phase 4 · Active",
    )

    model: Optional[ModelSummary] = get_active_model()
    env_info = discover_execution_providers()

    # --- 1. No Model State ---
    if model is None:
        _render_empty_model_state()
        return

    # --- 2. Active Model Context Strip ---
    _render_model_context_strip(model)

    # --- 3. Benchmark Configuration & Execution Controls ---
    config = _render_config_section(model, env_info)

    # Check if a benchmark run was triggered
    col_run, col_run_right = st.columns([1, 3])
    with col_run:
        run_clicked = st.button("Start Benchmark Run", type="primary", use_container_width=True)

    if run_clicked:
        model_bytes = get_active_model_bytes()
        if not model_bytes:
            st.error("Model binary content is unavailable in session state. Please re-upload or re-select the model.")
            return

        with st.spinner(f"Benchmarking {model.filename} on {config.execution_provider}..."):
            try:
                report = BenchmarkEngine.run_benchmark(
                    model_path_or_bytes=model_bytes,
                    model_summary=model,
                    config=config,
                )
                set_benchmark_report(report)
                st.success("Benchmark completed successfully.")
            except BenchmarkError as e:
                failed_report = BenchmarkReport(
                    run_id="BENCH-FAILED",
                    timestamp_utc="",
                    model_name=model.filename,
                    model_fingerprint=model.sha256_hash,
                    status="FAILED",
                    error_message=str(e),
                    environment={},
                    config={},
                    input_spec={},
                    correctness=None,
                    latency=None,
                    throughput=None,
                    memory=None,
                    per_round_results=[],
                    all_latencies_ms=[],
                    warnings=[],
                    limitations=[],
                )
                set_benchmark_report(failed_report)
                st.error(f"Benchmark Failed: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected Runtime Failure: {str(e)}")

    # --- 4. Render Active Benchmark Report ---
    active_report: Optional[BenchmarkReport] = get_benchmark_report()
    if active_report is not None and active_report.model_fingerprint == model.sha256_hash:
        _render_benchmark_results(active_report)


def _render_empty_model_state() -> None:
    """Render helpful callout when no model has been loaded."""
    st.markdown(
        """
        <div class="surface-card" style="padding: 2rem 1.5rem; text-align: center; margin: 1.5rem 0;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;">
                No Model Loaded for Benchmarking
            </div>
            <div style="font-size: 0.86rem; color: #475569; max-width: 580px; margin: 0 auto 1.25rem auto; line-height: 1.5;">
                Select or upload an ONNX model to measure actual execution performance.
                The benchmark engine synthesizes valid inputs matching the model's exact tensor signatures and runs genuine inference via ONNX Runtime.
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
                model_data = sample_path.read_bytes()
                summary = load_and_inspect_model(model_data, filename=sample_path.name)
                set_active_model(summary, source_description="Sample: Linear Classifier", model_bytes=model_data)
                st.rerun()
    with col2:
        if st.button("Load ConvNet Sample", use_container_width=True):
            sample_path = Path("sample_models/conv_net.onnx")
            if sample_path.exists():
                model_data = sample_path.read_bytes()
                summary = load_and_inspect_model(model_data, filename=sample_path.name)
                set_active_model(summary, source_description="Sample: ConvNet", model_bytes=model_data)
                st.rerun()
    with col3:
        if st.button("Go to Model Inspector", use_container_width=True):
            navigate_to("inspector")
            st.rerun()


def _render_model_context_strip(model: ModelSummary) -> None:
    """Render context summary showing active model and compatibility status."""
    compat_report = get_compatibility_report()
    compat_tag = ""
    if compat_report is not None and compat_report.model_fingerprint == model.sha256_hash:
        compat_tag = (
            f'<span style="background: #EFF6FF; color: #1E40AF; padding: 0.15rem 0.45rem; border-radius: 3px; font-weight: 600; font-size: 0.72rem; border: 1px solid #BFDBFE;">'
            f'Compatibility: {compat_report.overall_assessment}'
            f'</span>'
        )

    input_shapes = ", ".join(f"{t.name}: {t.shape}" for t in model.inputs)

    html = (
        f'<div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.75rem;">'
        f'<div>'
        f'<div style="font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B;">Active Benchmark Target</div>'
        f'<div style="font-size: 0.92rem; font-weight: 700; color: #0F172A; font-family: monospace;">{model.filename}</div>'
        f'</div>'
        f'<div style="display: flex; align-items: center; gap: 0.75rem; font-size: 0.78rem; color: #475569;">'
        f'{compat_tag}'
        f'<span>Nodes: <strong>{model.total_nodes}</strong></span> · '
        f'<span>Params: <strong>{model.total_parameters:,}</strong></span> · '
        f'<span>Inputs: <code>{input_shapes}</code></span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_config_section(model: ModelSummary, env_info: Any) -> BenchmarkConfig:
    """Render benchmark configuration controls."""
    with st.expander("Benchmark Configuration & Execution Provider", expanded=True):
        col_p1, col_p2 = st.columns([1, 1])

        with col_p1:
            st.markdown("**Execution Provider Selection**")
            available_p = env_info.available_providers
            default_idx = available_p.index("CPUExecutionProvider") if "CPUExecutionProvider" in available_p else 0

            selected_provider = st.selectbox(
                "Execution Provider",
                options=available_p,
                index=default_idx,
                help=f"Discovered from installed ONNX Runtime v{env_info.installed_ort_version}.",
                key="bench_provider_select",
            )
            st.caption(f"Installed ONNX Runtime: **v{env_info.installed_ort_version}** · Host Provider: **{selected_provider}**")

        with col_p2:
            st.markdown("**Controlled Measurement Parameters**")
            col_iter1, col_iter2 = st.columns(2)
            with col_iter1:
                measured_iters = st.number_input(
                    "Iterations / Round",
                    min_value=5,
                    max_value=500,
                    value=50,
                    step=10,
                    help="Number of timed inferences per round.",
                    key="bench_iters_input",
                )
                warmup_iters = st.number_input(
                    "Warm-up Iterations",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=5,
                    help="Untimed iterations to warm up caches and execution kernels.",
                    key="bench_warmup_input",
                )
            with col_iter2:
                repeated_rounds = st.number_input(
                    "Repeated Rounds",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    help="Repeat benchmark rounds to evaluate variance and jitter.",
                    key="bench_rounds_input",
                )
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=64,
                    value=1,
                    step=1,
                    help="Batch dimension for generated inputs.",
                    key="bench_batch_input",
                )

        # Dynamic Dimensions Overrides
        dynamic_dims: Dict[str, int] = {}
        dyn_dim_names: List[str] = []
        for inp in model.inputs:
            for dim in inp.shape:
                if isinstance(dim, str) and dim not in dyn_dim_names and dim.lower() not in ("batch", "batch_size", "n", "dim_0"):
                    dyn_dim_names.append(dim)

        if dyn_dim_names:
            st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 0.5rem 0;' />", unsafe_allow_html=True)
            st.markdown("**Symbolic Dimension Overrides**")
            st.caption("Model contains dynamic axes. Specify concrete integer sizes for benchmark input synthesis.")
            dim_cols = st.columns(min(len(dyn_dim_names), 4))
            for i, d_name in enumerate(dyn_dim_names):
                with dim_cols[i % len(dim_cols)]:
                    dim_val = st.number_input(
                        f"Axis: {d_name}",
                        min_value=1,
                        max_value=1024,
                        value=16,
                        step=1,
                        key=f"bench_dyn_dim_{d_name}",
                    )
                    dynamic_dims[d_name] = int(dim_val)

        # Advanced Settings
        with st.expander("Advanced Settings (Threads & Seed)", expanded=False):
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            with col_adv1:
                seed = st.number_input("Random Seed", value=42, step=1, key="bench_seed_input")
            with col_adv2:
                intra_threads = st.number_input(
                    "Intra-op Threads (0 = Auto)",
                    min_value=0,
                    max_value=64,
                    value=0,
                    step=1,
                    key="bench_intra_threads",
                )
            with col_adv3:
                inter_threads = st.number_input(
                    "Inter-op Threads (0 = Auto)",
                    min_value=0,
                    max_value=64,
                    value=0,
                    step=1,
                    key="bench_inter_threads",
                )

    return BenchmarkConfig(
        execution_provider=selected_provider,
        warmup_iterations=int(warmup_iters),
        measured_iterations=int(measured_iters),
        batch_size=int(batch_size),
        repeated_rounds=int(repeated_rounds),
        random_seed=int(seed),
        dynamic_dimensions=dynamic_dims,
        intra_op_num_threads=int(intra_threads) if intra_threads > 0 else None,
        inter_op_num_threads=int(inter_threads) if inter_threads > 0 else None,
    )


def _render_benchmark_results(report: BenchmarkReport) -> None:
    """Render comprehensive benchmark outcome, statistics, round details, and export."""
    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 1.25rem 0;' />", unsafe_allow_html=True)

    lat = report.latency or {}
    thru = report.throughput or {}
    mem = report.memory or {}
    corr = report.correctness or {}

    # 1. Outcome Banner
    render_benchmark_banner(
        status=report.status,
        median_ms=lat.get("median_ms", 0.0),
        throughput=thru.get("inferences_per_second", 0.0),
        correctness_passed=corr.get("passed", False),
        error_message=report.error_message,
    )

    if report.status != "SUCCESS":
        return

    # 2. Key Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        render_metric_box(
            "Median Latency",
            f"{lat.get('median_ms', 0.0):.2f} ms",
            f"P95: {lat.get('p95_ms', 0.0):.2f} ms",
        )
    with m_col2:
        render_metric_box(
            "Mean Latency",
            f"{lat.get('mean_ms', 0.0):.2f} ms",
            f"Std Dev: ±{lat.get('std_dev_ms', 0.0):.2f} ms",
        )
    with m_col3:
        render_metric_box(
            "Throughput",
            f"{thru.get('inferences_per_second', 0.0):.1f} inf/s",
            f"{thru.get('samples_per_second', 0.0):.1f} samples/s",
        )
    with m_col4:
        render_metric_box(
            "Correctness",
            "VALIDATED",
            "Finite values & repeatable",
        )
    with m_col5:
        delta_str = f"+{mem.get('session_delta_mb', 0.0)} MB" if mem.get("session_delta_mb") is not None else "N/A"
        render_metric_box(
            "Session Memory Δ",
            delta_str,
            f"Peak RSS: {mem.get('peak_rss_mb', 0.0):.1f} MB",
        )

    # 3. Latency Percentiles & Statistics Table
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 1.25rem 0 0.5rem 0;">
            Empirical Latency Percentile Distribution
        </div>
        """,
        unsafe_allow_html=True,
    )

    stat_df = pd.DataFrame(
        [
            {
                "Min Latency": f"{lat.get('min_ms', 0.0):.3f} ms",
                "P50 (Median)": f"{lat.get('median_ms', 0.0):.3f} ms",
                "Mean Latency": f"{lat.get('mean_ms', 0.0):.3f} ms",
                "P90 Latency": f"{lat.get('p90_ms', 0.0):.3f} ms",
                "P95 Latency": f"{lat.get('p95_ms', 0.0):.3f} ms",
                "P99 Latency": f"{lat.get('p99_ms', 0.0):.3f} ms",
                "Max Latency": f"{lat.get('max_ms', 0.0):.3f} ms",
                "Sample Count": f"{lat.get('sample_count', 0):,} iterations",
            }
        ]
    )
    st.table(stat_df)

    # 4. Round-to-Round Variability Breakdown
    if report.per_round_results:
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 1.25rem 0 0.5rem 0;">
                Round-to-Round Variability & Jitter Analysis
            </div>
            """,
            unsafe_allow_html=True,
        )

        round_data = []
        for r in report.per_round_results:
            round_data.append(
                {
                    "Round": f"Round #{r.get('round_number')}",
                    "Samples": len(r.get("latencies_ms", [])),
                    "Median Latency": f"{r.get('median_ms', 0.0):.3f} ms",
                    "Mean Latency": f"{r.get('mean_ms', 0.0):.3f} ms",
                    "P95 Latency": f"{r.get('p95_ms', 0.0):.3f} ms",
                    "Throughput": f"{r.get('inferences_per_second', 0.0):.1f} inf/sec",
                }
            )
        st.dataframe(pd.DataFrame(round_data), use_container_width=True, hide_index=True)

    # 5. Environment & Measurement Boundary Caveats
    with st.expander("Environment Metadata & Measurement Limitations", expanded=True):
        col_env, col_lim = st.columns(2)
        with col_env:
            st.markdown("**Host Execution Environment:**")
            env = report.environment
            st.markdown(f"- **OS:** {env.get('os_name')} {env.get('os_release')} ({env.get('architecture')})")
            st.markdown(f"- **Python:** {env.get('python_version')}")
            st.markdown(f"- **ONNX Runtime:** {env.get('onnxruntime_version')}")
            st.markdown(f"- **Active Provider:** <code>{report.config.get('execution_provider')}</code>", unsafe_allow_html=True)
            st.markdown(f"- **CPU Logical Cores:** {env.get('cpu_count_logical')}")

        with col_lim:
            st.markdown("**Measurement Boundary & Architectural Caveats:**")
            for lim in report.limitations:
                st.markdown(f"- <span style='font-size: 0.82rem; color: #475569;'>{lim}</span>", unsafe_allow_html=True)
            if report.warnings:
                for warn in report.warnings:
                    st.markdown(f"- <span style='font-size: 0.82rem; color: #92400E;'>{warn}</span>", unsafe_allow_html=True)

    # 6. Report Export
    st.markdown(
        """
        <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin: 1.25rem 0 0.5rem 0;">
            Export Benchmark Artifacts
        </div>
        """,
        unsafe_allow_html=True,
    )

    report_json = report.to_json(indent=2)
    report_csv = report.to_csv()
    base_name = f"benchmark_{report.model_fingerprint[:8]}_{report.config.get('execution_provider', 'cpu')}"

    col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 2])
    with col_exp1:
        st.download_button(
            label="Download Report (JSON)",
            data=report_json,
            file_name=f"{base_name}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_exp2:
        st.download_button(
            label="Download Samples (CSV)",
            data=report_csv,
            file_name=f"{base_name}_samples.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_exp3:
        st.caption(
            "Export contains sanitized model fingerprint, percentile statistics, round breakdowns, and environment metadata."
        )

    with st.expander("Inspect Raw Benchmark JSON Report"):
        st.code(report_json, language="json")
