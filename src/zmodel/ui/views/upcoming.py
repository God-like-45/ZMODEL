"""Upcoming phase view renderer for phases 3 through 7.

Presents honest, unpretentious roadmap specifications without fabricating results.
"""

from typing import Dict
import streamlit as st
from zmodel.core.state import get_active_model
from zmodel.ui.components import render_workspace_header


PHASE_SPECS: Dict[str, Dict[str, str]] = {
    "compatibility": {
        "title": "Compatibility Analysis",
        "phase": 3,
        "description": "Target hardware capability profiles and explainable operator-by-operator compatibility reports.",
        "prereqs": "Phase 2 validated ONNX model graph, tensor signatures, and operator distribution.",
        "planned_deliverables": """
        - Target hardware capability database (x86_64 AVX-512, ARM Neon, NVIDIA TensorRT, OpenVINO).
        - Operator coverage checker identifying unsupported or fallback operations.
        - Precision feasibility check (FP32, FP16, INT8 QDQ quantization support).
        - Explainable diagnostic report with remediation recommendations for unsupported ops.
        """,
    },
    "benchmarks": {
        "title": "Empirical Benchmarking",
        "phase": 4,
        "description": "Controlled empirical execution profiling measuring real latency percentiles (P50, P90, P99), query throughput (QPS), and peak memory usage.",
        "prereqs": "Phase 3 compatibility verification and supported Execution Provider configurations.",
        "planned_deliverables": """
        - Controlled warmup and statistical repeat harness using ONNX Runtime.
        - Synthetic input tensor generator matching model input shapes and data types.
        - Multi-batch and concurrent query throughput benchmarking.
        - Memory consumption profiler (resident set size and peak allocations).
        - Latency distribution visualizer (P50/P90/P99 comparison across execution providers).
        """,
    },
    "optimization": {
        "title": "Pareto Optimization & Selection",
        "phase": 5,
        "description": "Multi-objective Pareto frontier analysis and constrained candidate selection balancing latency, throughput, and memory budgets.",
        "prereqs": "Phase 4 empirical benchmark execution logs.",
        "planned_deliverables": """
        - User-defined deployment constraints (e.g. latency <= 15ms, memory <= 500MB).
        - Multi-objective Pareto frontier visualization (latency vs. throughput vs. footprint).
        - Evidence-based candidate ranking and deployment recommendation scorecard.
        - Optimization config exporter (ORT session options, thread pools, provider options).
        """,
    },
    "history": {
        "title": "Experiment History",
        "phase": 6,
        "description": "Auditable, reproducible experiment history tracking across models, target hardware, and benchmark runs.",
        "prereqs": "Phase 4/5 benchmark and optimization runs.",
        "planned_deliverables": """
        - Structured experiment ledger with model SHA-256 fingerprinting.
        - Side-by-side run comparison and regression diffing.
        - Exportable audit reports (JSON / Markdown / PDF).
        - Reproducibility bundles with exact runtime parameters.
        """,
    },
    "settings": {
        "title": "Settings & Environment",
        "phase": 7,
        "description": "Configurable runtime paths, telemetry preferences, and container execution environment settings.",
        "prereqs": "Active workspace configuration.",
        "planned_deliverables": """
        - Execution Provider configuration (intra/inter-op thread pools, memory arenas).
        - Model cache directory and artifact storage paths.
        - Docker runtime integration settings.
        - Export format preferences.
        """,
    },
}


def render_upcoming_view(page_id: str) -> None:
    """Render an honest upcoming phase preview."""
    spec = PHASE_SPECS.get(page_id, {
        "title": page_id.capitalize(),
        "phase": 3,
        "description": "This feature is scheduled for a future development phase.",
        "prereqs": "Prior phase completion.",
        "planned_deliverables": "Detailed specifications in progress.",
    })

    render_workspace_header(
        title=spec["title"],
        description=f"Scheduled for implementation in Phase {spec['phase']}. Future functionality is cleanly distinguished from active features.",
        environment_badge=f"Phase {spec['phase']} · Upcoming",
    )

    st.markdown(
        f"""
        <div class="roadmap-box">
            <div style="font-weight: 600; font-size: 0.95rem; color: #0F172A; margin-bottom: 0.35rem;">
                Phase {spec['phase']} Roadmap: {spec['title']}
            </div>
            <div style="font-size: 0.85rem; color: #475569; line-height: 1.5; margin-bottom: 0.75rem;">
                {spec['description']}
            </div>
            <div style="font-size: 0.76rem; color: #64748B; font-family: var(--zm-font-mono);">
                Required Prerequisite: {spec['prereqs']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_model = get_active_model()
    if active_model:
        st.info(
            f"Active Model Registered: '{active_model.filename}' ({active_model.total_nodes} nodes, {active_model.total_parameters:,} parameters) "
            f"is loaded from Phase 2 and will be the target graph when Phase {spec['phase']} is implemented."
        )

    st.markdown("##### Planned Phase Deliverables")
    st.markdown(spec["planned_deliverables"])
