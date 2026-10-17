"""Z-ModelPilot: Main Streamlit Application Entrypoint."""

import sys
import os

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import streamlit as st
from zmodel.core.config import (
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    ENVIRONMENT_TYPE,
    NAV_ITEMS,
)
from zmodel.core.state import (
    get_active_model,
    init_session_state,
    navigate_to,
)
from zmodel.core.system_info import get_system_environment
from zmodel.ui.theme import apply_theme
from zmodel.ui.views.benchmark import render_benchmark_view
from zmodel.ui.views.compatibility import render_compatibility_view
from zmodel.ui.views.history import render_history_view
from zmodel.ui.views.inspector import render_inspector_view
from zmodel.ui.views.optimization import render_optimization_view
from zmodel.ui.views.graph_optimizer import render_graph_optimization_view
from zmodel.ui.views.workers import render_workers_view
from zmodel.ui.views.validation import render_validation_view
from zmodel.ui.views.overview import render_overview_view
from zmodel.ui.views.settings import render_settings_view
from zmodel.ui.views.upcoming import render_upcoming_view



def setup_page_config() -> None:
    """Configure Streamlit page layout and title."""
    st.set_page_config(
        page_title="Z-ModelPilot — Model Deployment Workspace",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_sidebar() -> None:
    """Render the compact engineering sidebar navigation."""
    with st.sidebar:
        # Product Identity Wordmark
        st.markdown(
            f"""
            <div class="sidebar-brand-box">
                <div class="sidebar-wordmark">
                    <span>{APP_NAME}</span>
                    <span class="status-pill pill-local" style="font-size: 0.68rem;">v{APP_VERSION}</span>
                </div>
                <div class="sidebar-tagline">{APP_SUBTITLE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Active Model State in Sidebar
        active_model = get_active_model()
        if active_model:
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 1.25rem;">
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748B; font-weight: 600; letter-spacing: 0.04em;">Selected Model</div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #0F172A; font-family: var(--zm-font-mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 0.15rem;">
                        {active_model.filename}
                    </div>
                    <div style="font-size: 0.72rem; color: #059669; font-weight: 500; margin-top: 0.2rem;">
                        ● Verified ({active_model.total_nodes} nodes)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.55rem 0.75rem; margin-bottom: 1.25rem;">
                    <div style="font-size: 0.68rem; text-transform: uppercase; color: #64748B; font-weight: 600; letter-spacing: 0.04em;">Selected Model</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.15rem;">No model loaded</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Navigation Menu
        st.markdown(
            "<div class='sidebar-section-title'>Navigation</div>",
            unsafe_allow_html=True,
        )

        current_page = st.session_state.get("current_page", "overview")

        for item in NAV_ITEMS:
            # Display phase indicator for upcoming phases
            suffix = "" if item.is_active else f" ({item.status_label})"
            button_label = f"{item.label}{suffix}"
            
            is_current = (item.id == current_page)
            btn_type = "primary" if is_current else "secondary"

            if st.button(
                button_label,
                key=f"nav_{item.id}",
                type=btn_type,
                use_container_width=True,
            ):
                navigate_to(item.id)
                st.rerun()

        # Quiet Environment Diagnostics Footer
        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        env = get_system_environment()
        provider_name = env.host_providers[0] if env.host_providers else "None"
        st.markdown(
            f"""
            <div style="border-top: 1px solid #E2E8F0; padding-top: 0.85rem; font-size: 0.72rem; color: #64748B; line-height: 1.5;">
                <div style="font-weight: 600; color: #475569; margin-bottom: 0.25rem;">Workspace Context</div>
                <div>Host: <span style="color: #0F172A;">{env.os_name} {env.architecture}</span></div>
                <div>Python: <span style="color: #0F172A;">{env.python_version}</span></div>
                <div>Local Provider: <span style="color: #2563EB;">{provider_name}</span></div>
                <div>Target Architecture: <span style="color: #64748B;">Not verified</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Main application loop."""
    setup_page_config()
    init_session_state()

    # Support query parameter deep-linking (e.g. ?page=inspector&sample=linear_classifier.onnx)
    if "page" in st.query_params:
        target_page = st.query_params["page"]
        if any(item.id == target_page for item in NAV_ITEMS):
            st.session_state["current_page"] = target_page

    if "sample" in st.query_params and st.session_state.get("active_model") is None:
        sample_file = st.query_params["sample"]
        sample_path = os.path.join("sample_models", sample_file)
        if os.path.exists(sample_path):
            from zmodel.core.model_loader import load_and_inspect_model
            from zmodel.core.state import set_active_model
            summary = load_and_inspect_model(sample_path, filename=sample_file)
            set_active_model(summary, source_description=f"Sample: {sample_file}")

    if "profile" in st.query_params:
        target_profile = st.query_params["profile"]
        from zmodel.core.profiles import load_builtin_profiles
        if target_profile in load_builtin_profiles():
            st.session_state["selected_profile_id"] = target_profile

    apply_theme()
    render_sidebar()

    current_page = st.session_state.get("current_page", "overview")

    if current_page == "overview":
        render_overview_view()
    elif current_page == "inspector":
        render_inspector_view()
    elif current_page == "compatibility":
        render_compatibility_view()
    elif current_page == "benchmarks":
        render_benchmark_view()
    elif current_page == "optimization":
        render_optimization_view()
    elif current_page == "graph_optimization":
        render_graph_optimization_view()
    elif current_page == "workers":
        render_workers_view()
    elif current_page == "validation":
        render_validation_view()
    elif current_page == "history":
        render_history_view()
    elif current_page == "settings":
        render_settings_view()
    else:
        render_overview_view()



if __name__ == "__main__":
    main()
