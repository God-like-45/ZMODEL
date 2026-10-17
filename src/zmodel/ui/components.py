"""Reusable UI components for the refined light design system."""

from typing import Any, Optional
import streamlit as st
from zmodel.core.config import APP_NAME, ENVIRONMENT_TYPE


def format_bytes(num_bytes: int) -> str:
    """Format bytes into clean units (B, KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{num_bytes / (1024 ** 3):.2f} GB"


def render_workspace_header(
    title: str = "Model deployment workspace",
    description: str = "Analyze ONNX model architectures, verify graph structural constraints, and prepare deployment configurations.",
    environment_badge: Optional[str] = ENVIRONMENT_TYPE,
) -> None:
    """Render top workspace header with clean wordmark, title, description, and environment pill."""
    env_html = ""
    if environment_badge:
        env_html = f'<span class="status-pill pill-local">{environment_badge}</span>'

    header_html = (
        '<div class="workspace-header">'
        '<div class="workspace-header-top">'
        '<div style="display: flex; align-items: baseline; gap: 0.75rem;">'
        f'<span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #2563EB;">{APP_NAME}</span>'
        '<span style="color: #CBD5E1;">/</span>'
        f'<span class="workspace-title">{title}</span>'
        '</div>'
        f'{env_html}'
        '</div>'
        f'<div class="workspace-desc">{description}</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_metric_box(label: str, value: str, subtext: Optional[str] = None) -> None:
    """Render a quiet, restrained metric container."""
    sub_html = f'<div class="metric-box-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f'<div class="metric-box"><div class="metric-box-label">{label}</div><div class="metric-box-val">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def render_warning_item(warning: Any) -> None:
    """Render a disciplined inspection warning or advisory callout."""
    badge_style = {
        "warning": "background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D;",
        "advisory": "background: #EFF6FF; color: #1E40AF; border: 1px solid #BFDBFE;",
        "info": "background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1;",
    }.get(warning.level, "background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1;")

    border_color = {
        "warning": "#F59E0B",
        "advisory": "#3B82F6",
        "info": "#94A3B8",
    }.get(warning.level, "#94A3B8")

    card_html = (
        f'<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 3px solid {border_color}; border-radius: 6px; padding: 0.85rem 1.15rem; margin-bottom: 0.65rem;">'
        f'<div style="display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; margin-bottom: 0.25rem;">'
        f'<span style="font-weight: 600; font-size: 0.88rem; color: #0F172A;">{warning.title}</span>'
        f'<span style="font-family: var(--zm-font-mono); font-size: 0.68rem; font-weight: 600; padding: 0.15rem 0.4rem; border-radius: 3px; {badge_style}">{warning.code}</span>'
        f'</div>'
        f'<div style="font-size: 0.82rem; color: #334155; line-height: 1.45;">{warning.message}</div>'
        f'<div style="font-size: 0.76rem; color: #64748B; margin-top: 0.35rem;"><span style="font-weight: 500; color: #475569;">Deployment Impact:</span> {warning.impact}</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_export_guide() -> None:
    """Render an expandable panel with exact ONNX export instructions for major frameworks."""
    with st.expander("How to export models to ONNX (PyTorch, TensorFlow, Transformers)"):
        st.markdown(
            """
**PyTorch (`torch.onnx.export`)**
```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)
```

**Hugging Face Transformers / Optimum**
```bash
optimum-cli export onnx --model meta-llama/Llama-3-8B --task text-generation ./onnx_output/
```

**TensorFlow / Keras (`tf2onnx`)**
```bash
python -m tf2onnx.convert --saved-model tensorflow_model_dir --output model.onnx --opset 17
```
            """
        )


def get_status_badge_html(status: str) -> str:
    """Return accessible status badge HTML pairing color with clear text."""
    st_upper = status.upper()
    styles = {
        "PASS": "background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0;",
        "FAIL": "background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA;",
        "WARNING": "background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A;",
        "UNKNOWN": "background: #F1F5F9; color: #334155; border: 1px solid #CBD5E1;",
    }
    style = styles.get(st_upper, styles["UNKNOWN"])
    return f'<span style="font-family: var(--zm-font-mono, monospace); font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.45rem; border-radius: 4px; letter-spacing: 0.04em; {style}">{st_upper}</span>'


def render_assessment_banner(
    assessment: str,
    summary_counts: dict,
    is_illustrative: bool = False,
) -> None:
    """Render the top overall compatibility assessment banner with deterministic indicators."""
    configs = {
        "COMPATIBLE": {
            "border": "#059669",
            "bg": "#F0FDF4",
            "text_color": "#065F46",
            "title": "COMPATIBLE — Verified Static Capability Match",
            "desc": "All operators, opsets, and boundary tensor signatures are explicitly supported by the selected capability profile.",
        },
        "CONDITIONAL": {
            "border": "#D97706",
            "bg": "#FFFBEB",
            "text_color": "#92400E",
            "title": "CONDITIONAL — Operational Constraints or Warnings Present",
            "desc": "No fatal incompatibilities detected, but conditional support, dynamic shape alignment, or packaging constraints require attention.",
        },
        "INSUFFICIENT_DATA": {
            "border": "#64748B",
            "bg": "#F8FAFC",
            "text_color": "#1E293B",
            "title": "INSUFFICIENT DATA — Unverified Operators or Profile Limits",
            "desc": "The target profile lacks verified rules for one or more model features. Incomplete rules are never assumed as supported.",
        },
        "INCOMPATIBLE": {
            "border": "#DC2626",
            "bg": "#FEF2F2",
            "text_color": "#991B1B",
            "title": "INCOMPATIBLE — Explicit Support Failures Detected",
            "desc": "Model contains operators, opset versions, or data types explicitly unsupported or restricted by the target profile.",
        },
    }

    cfg = configs.get(assessment, configs["INSUFFICIENT_DATA"])

    illustrative_note = ""
    if is_illustrative:
        illustrative_note = (
            '<div style="margin-top: 0.5rem; padding: 0.35rem 0.6rem; background: #FEF3C7; border: 1px solid #FCD34D; '
            'border-radius: 4px; font-size: 0.74rem; color: #78350F; font-weight: 500;">'
            '⚠️ <strong>Illustrative Profile Note:</strong> This profile is an exploratory ruleset. '
            'Static compatibility does not certify real IBM zSystems hardware execution.'
            '</div>'
        )

    p_cnt = summary_counts.get("PASS", 0)
    f_cnt = summary_counts.get("FAIL", 0)
    w_cnt = summary_counts.get("WARNING", 0)
    u_cnt = summary_counts.get("UNKNOWN", 0)

    html = (
        f'<div style="background: {cfg["bg"]}; border: 1px solid #E2E8F0; border-left: 4px solid {cfg["border"]}; '
        f'border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">'
        f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">'
        f'<div style="font-size: 1rem; font-weight: 700; color: {cfg["text_color"]};">{cfg["title"]}</div>'
        f'<div style="display: flex; gap: 0.4rem; font-size: 0.75rem; font-family: monospace;">'
        f'<span style="background: #ECFDF5; color: #065F46; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #A7F3D0; font-weight: 600;">PASS: {p_cnt}</span>'
        f'<span style="background: #FEF2F2; color: #991B1B; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #FECACA; font-weight: 600;">FAIL: {f_cnt}</span>'
        f'<span style="background: #FFFBEB; color: #92400E; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #FDE68A; font-weight: 600;">WARN: {w_cnt}</span>'
        f'<span style="background: #F1F5F9; color: #334155; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #CBD5E1; font-weight: 600;">UNKNOWN: {u_cnt}</span>'
        f'</div>'
        f'</div>'
        f'<div style="font-size: 0.82rem; color: #334155; margin-top: 0.35rem; line-height: 1.45;">{cfg["desc"]}</div>'
        f'{illustrative_note}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_benchmark_banner(
    status: str,
    median_ms: float,
    throughput: float,
    correctness_passed: bool,
    error_message: Optional[str] = None,
) -> None:
    """Render top benchmark execution outcome banner."""
    if status == "SUCCESS":
        correct_pill = (
            '<span style="background: #ECFDF5; color: #065F46; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #A7F3D0; font-weight: 600;">Correctness: PASSED</span>'
            if correctness_passed
            else '<span style="background: #FEF2F2; color: #991B1B; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #FECACA; font-weight: 600;">Correctness: UNVERIFIED</span>'
        )
        html = (
            f'<div style="background: #F0FDF4; border: 1px solid #E2E8F0; border-left: 4px solid #059669; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">'
            f'<div style="font-size: 1rem; font-weight: 700; color: #065F46;">BENCHMARK COMPLETED — Empirical Measurements Verified</div>'
            f'<div style="display: flex; gap: 0.4rem; font-size: 0.75rem; font-family: monospace;">'
            f'{correct_pill}'
            f'<span style="background: #EFF6FF; color: #1E40AF; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #BFDBFE; font-weight: 600;">Median: {median_ms:.2f} ms</span>'
            f'<span style="background: #F8FAFC; color: #334155; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #CBD5E1; font-weight: 600;">{throughput:.1f} inf/sec</span>'
            f'</div>'
            f'</div>'
            f'<div style="font-size: 0.82rem; color: #334155; margin-top: 0.35rem; line-height: 1.45;">'
            f'Measured actual execution on the selected provider. All latency percentiles and throughput figures are derived from high-resolution monotonic timer samples.'
            f'</div>'
            f'</div>'
        )
    else:
        err_text = error_message or "Benchmark execution encountered an unrecoverable runtime error."
        html = (
            f'<div style="background: #FEF2F2; border: 1px solid #E2E8F0; border-left: 4px solid #DC2626; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">'
            f'<div style="font-size: 1rem; font-weight: 700; color: #991B1B;">BENCHMARK EXECUTION FAILED</div>'
            f'<div style="font-size: 0.82rem; color: #7F1D1D; margin-top: 0.35rem; line-height: 1.45;">'
            f'<strong>Error:</strong> {err_text}'
            f'</div>'
            f'<div style="font-size: 0.76rem; color: #991B1B; margin-top: 0.4rem;">'
            f'No performance or throughput numbers are recorded for failed runs.'
            f'</div>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


def render_optimization_banner(
    status: str,
    best_candidate_id: Optional[str],
    speedup: Optional[float],
    feasible_count: int,
    total_benchmarked: int,
    objective_label: str,
    error_message: Optional[str] = None,
) -> None:
    """Render top optimization experiment outcome banner."""
    if status in ("COMPLETED", "COMPLETED_WITH_WARNINGS") and best_candidate_id:
        speedup_pill = ""
        if speedup is not None:
            spd_text = f"Speedup: {speedup:.2f}x" if speedup >= 1.0 else f"Relative: {speedup:.2f}x"
            speedup_pill = (
                f'<span style="background: #EFF6FF; color: #1E40AF; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #BFDBFE; font-weight: 600;">'
                f'{spd_text}'
                f'</span>'
            )

        html = (
            f'<div style="background: #F0FDF4; border: 1px solid #E2E8F0; border-left: 4px solid #059669; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">'
            f'<div style="font-size: 1rem; font-weight: 700; color: #065F46;">OPTIMIZATION COMPLETED — Optimal Feasible Configuration Identified</div>'
            f'<div style="display: flex; gap: 0.4rem; font-size: 0.75rem; font-family: monospace;">'
            f'<span style="background: #ECFDF5; color: #065F46; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #A7F3D0; font-weight: 600;">Best: {best_candidate_id}</span>'
            f'{speedup_pill}'
            f'<span style="background: #F8FAFC; color: #334155; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #CBD5E1; font-weight: 600;">Feasible: {feasible_count}/{total_benchmarked}</span>'
            f'</div>'
            f'</div>'
            f'<div style="font-size: 0.82rem; color: #334155; margin-top: 0.35rem; line-height: 1.45;">'
            f'Recommended candidate strictly satisfies all operational boundaries and correctness validations for objective <strong>{objective_label}</strong>.'
            f'</div>'
            f'</div>'
        )
    elif status in ("COMPLETED", "COMPLETED_WITH_WARNINGS") and not best_candidate_id:
        html = (
            f'<div style="background: #FFFBEB; border: 1px solid #E2E8F0; border-left: 4px solid #D97706; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">'
            f'<div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 0.5rem;">'
            f'<div style="font-size: 1rem; font-weight: 700; color: #92400E;">NO FEASIBLE CONFIGURATION FOUND</div>'
            f'<div style="display: flex; gap: 0.4rem; font-size: 0.75rem; font-family: monospace;">'
            f'<span style="background: #FEF2F2; color: #991B1B; padding: 0.15rem 0.45rem; border-radius: 3px; border: 1px solid #FECACA; font-weight: 600;">Feasible: 0/{total_benchmarked}</span>'
            f'</div>'
            f'</div>'
            f'<div style="font-size: 0.82rem; color: #92400E; margin-top: 0.35rem; line-height: 1.45;">'
            f'All {total_benchmarked} evaluated candidates violated one or more hard operational constraints or failed correctness validation. Review the Constraint Violations breakdown below.'
            f'</div>'
            f'</div>'
        )
    else:
        err_text = error_message or "Optimization search encountered an unrecoverable failure."
        html = (
            f'<div style="background: #FEF2F2; border: 1px solid #E2E8F0; border-left: 4px solid #DC2626; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.25rem;">'
            f'<div style="font-size: 1rem; font-weight: 700; color: #991B1B;">OPTIMIZATION SEARCH FAILED</div>'
            f'<div style="font-size: 0.82rem; color: #7F1D1D; margin-top: 0.35rem; line-height: 1.45;">'
            f'<strong>Error:</strong> {err_text}'
            f'</div>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)

