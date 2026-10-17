"""Overview screen: Model deployment workspace, upload area, and environment status."""

import os
import streamlit as st
from zmodel.core.config import (
    APP_NAME,
    ENVIRONMENT_TYPE,
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_FILE_EXTENSIONS,
)
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.state import (
    clear_active_model,
    get_active_model,
    navigate_to,
    set_active_model,
)
from zmodel.core.system_info import get_system_environment
from zmodel.ui.components import (
    format_bytes,
    render_export_guide,
    render_metric_box,
    render_workspace_header,
)


def render_overview_view() -> None:
    """Render the primary Phase 1 Overview screen."""
    # 1. Header Section
    render_workspace_header(
        title="Model deployment workspace",
        description="Inspect ONNX computational graphs, evaluate model architectures, and prepare deployment configurations based on measured evidence.",
        environment_badge=ENVIRONMENT_TYPE,
    )

    env = get_system_environment()
    active_model = get_active_model()

    # 2. Primary Workspace: Model Upload & Selection Area
    st.markdown(
        """
        <div class="surface-card">
            <div class="surface-card-title">Select or upload an ONNX model</div>
            <div class="surface-card-desc">
                Upload a neural network in ONNX format to parse its computational graph, inspect input/output tensor signatures, and audit operator coverage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_tab_upload, source_tab_sample = st.tabs(["Upload Model File", "Use Verified Sample Model"])

    with source_tab_upload:
        col_up, col_info = st.columns([1.8, 1.2])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload ONNX model file",
                type=["onnx"],
                help=f"Supported format: {', '.join(SUPPORTED_FILE_EXTENSIONS)}. Max size: {MAX_UPLOAD_SIZE_MB} MB.",
                label_visibility="collapsed",
            )
            st.caption(f"Supported formats: **.onnx** · Maximum file size: **{MAX_UPLOAD_SIZE_MB} MB**")

            if uploaded_file is not None:
                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-family: var(--zm-font-mono); font-size: 0.82rem; font-weight: 600; color: #0F172A;">{uploaded_file.name}</span>
                        <span style="font-size: 0.76rem; color: #64748B;">{format_bytes(uploaded_file.size)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
                if st.button("Begin Model Inspection", type="primary", use_container_width=True):
                    with st.spinner("Validating ONNX graph integrity..."):
                        try:
                            file_bytes = uploaded_file.getvalue()
                            summary = load_and_inspect_model(file_bytes, filename=uploaded_file.name)
                            set_active_model(summary, source_description=f"Uploaded: {uploaded_file.name}")
                            st.success(f"Validated {uploaded_file.name}. Navigating to Model Inspector...")
                            navigate_to("inspector")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Validation Error: {str(e)}")

        with col_info:
            st.markdown(
                """
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.9rem 1rem; font-size: 0.8rem; line-height: 1.5; color: #475569;">
                    <div style="font-weight: 600; color: #0F172A; margin-bottom: 0.35rem;">What happens next:</div>
                    <ul style="padding-left: 1.15rem; margin-bottom: 0.5rem;">
                        <li>Graph integrity is validated using <code>onnx.checker</code>.</li>
                        <li>Tensor signatures (shapes, dtypes, dynamic axes) are parsed.</li>
                        <li>Operator distribution and parameter counts are computed.</li>
                    </ul>
                    <div style="font-size: 0.74rem; color: #64748B;">
                        Note: Target hardware compilation and benchmarks are not executed during upload.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with source_tab_sample:
        st.markdown(
            "<div style='font-size: 0.85rem; color: #475569; margin-bottom: 0.75rem;'>Load a pre-verified ONNX model for immediate inspection without uploading your own file:</div>",
            unsafe_allow_html=True,
        )
        c_s1, c_s2 = st.columns([2, 1])
        with c_s1:
            sample_selection = st.selectbox(
                "Choose sample model:",
                options=[
                    "linear_classifier.onnx (2-layer MLP Classifier, 4 nodes, 643 params)",
                    "conv_net.onnx (CNN Vision Backbone, 5 nodes, 13,610 params)",
                ],
                label_visibility="collapsed",
            )
        with c_s2:
            if st.button("Load & Inspect Sample", use_container_width=True):
                fname = "linear_classifier.onnx" if "linear" in sample_selection else "conv_net.onnx"
                fpath = os.path.join("sample_models", fname)
                if os.path.exists(fpath):
                    with st.spinner(f"Loading {fname}..."):
                        try:
                            summary = load_and_inspect_model(fpath, filename=fname)
                            set_active_model(summary, source_description=f"Sample: {fname}")
                            st.success(f"Loaded {fname}. Navigating to Model Inspector...")
                            navigate_to("inspector")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.error(f"Sample file not found at {fpath}")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 3. First-Use Empty State OR Active Model Status
    if active_model is None:
        st.markdown(
            """
            <div class="empty-state-container">
                <div class="empty-state-heading">No model inspected yet</div>
                <div class="empty-state-copy">
                    Upload an ONNX file above or load a sample model to begin structural analysis, tensor signature inspection, and operator coverage auditing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_export_guide()
    else:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 1px solid #BFDBFE; border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px 0 rgba(37, 99, 235, 0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem;">
                    <div>
                        <span class="status-pill pill-active">Active Inspected Model</span>
                        <div style="font-size: 1.15rem; font-weight: 600; font-family: var(--zm-font-mono); color: #0F172A; margin-top: 0.35rem;">
                            {active_model.filename}
                        </div>
                    </div>
                    <div style="font-family: var(--zm-font-mono); font-size: 0.78rem; color: #64748B;">
                        SHA-256: <span style="color: #2563EB;">{active_model.sha256_hash[:16]}...{active_model.sha256_hash[-8:]}</span>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; border-top: 1px solid #F1F5F9; padding-top: 0.75rem;">
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B;">Graph Nodes</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #0F172A; font-family: var(--zm-font-mono);">{active_model.total_nodes}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B;">Parameters</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #0F172A; font-family: var(--zm-font-mono);">{active_model.total_parameters:,}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B;">Weights Size</div>
                        <div style="font-size: 1rem; font-weight: 600; color: #0F172A; font-family: var(--zm-font-mono);">{format_bytes(active_model.total_weight_bytes)}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B;">Validation</div>
                        <div style="font-size: 0.85rem; font-weight: 600; color: #059669;">Verified Graph</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; text-transform: uppercase; color: #64748B;">Deployment Compatibility</div>
                        <div style="font-size: 0.85rem; font-weight: 500; color: #64748B;">Not verified (Phase 3)</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_btn_a, col_btn_b, _ = st.columns([1.5, 1.2, 3])
        with col_btn_a:
            if st.button("Open Deep Graph Inspector ->", type="primary", use_container_width=True):
                navigate_to("inspector")
                st.rerun()
        with col_btn_b:
            if st.button("Unload Model", use_container_width=True):
                clear_active_model()
                st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 4. Workspace Information
    st.markdown("##### Workspace & Environment Information")
    st.caption("Environment attributes detected directly from the local host without simulation.")

    w1, w2, w3, w4 = st.columns(4)
    with w1:
        render_metric_box(
            label="Host Environment",
            value=f"{env.os_name} {env.architecture}",
            subtext=f"Python {env.python_version}",
        )
    with w2:
        render_metric_box(
            label="ONNX Parser Version",
            value=f"v{env.onnx_version or 'N/A'}",
            subtext=f"IR Specification v{env.onnx_ir_version or 'N/A'}",
        )
    with w3:
        host_ort = env.host_providers[0] if env.host_providers else "None"
        render_metric_box(
            label="Host Runtime Provider",
            value=host_ort,
            subtext="Local inference provider",
        )
    with w4:
        # Explicitly marked as Not verified, as requested
        render_metric_box(
            label="Target Architecture",
            value="Not verified",
            subtext="Profiling scheduled for Phase 2",
        )
