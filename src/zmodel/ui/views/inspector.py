"""Model Inspector view: comprehensive ONNX validation, graph analysis, and tensor inspection."""

import os
import altair as alt
import pandas as pd
import streamlit as st
from zmodel.core.config import MAX_UPLOAD_SIZE_MB, SUPPORTED_FILE_EXTENSIONS
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.state import get_active_model, set_active_model
from zmodel.ui.components import (
    format_bytes,
    render_metric_box,
    render_warning_item,
    render_workspace_header,
)


def render_inspector_view() -> None:
    """Render the Model Inspector page."""
    render_workspace_header(
        title="Model Inspector",
        description="Comprehensive ONNX computational graph analysis, tensor signatures, operator distributions, and structural connection tracing.",
        environment_badge="Phase 2 · Active",
    )

    current_summary: ModelSummary = get_active_model()

    # --- 1. Model Source Selection Area ---
    st.markdown(
        """
        <div class="surface-card" style="padding: 1rem 1.25rem; margin-bottom: 1rem;">
            <div class="surface-card-title">Select Model for Inspection</div>
            <div class="surface-card-desc" style="margin-bottom: 0.5rem;">
                Upload an ONNX file or switch between verified sample models. Static analysis parses graph schemas without running execution or inference.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    src_tab_upload, src_tab_sample = st.tabs(["Upload .onnx File", "Switch to Sample Model"])

    with src_tab_upload:
        col_u1, col_u2 = st.columns([2, 1])
        with col_u1:
            uploaded_file = st.file_uploader(
                "Upload ONNX model",
                type=["onnx"],
                help=f"Supported format: {', '.join(SUPPORTED_FILE_EXTENSIONS)}. Limit: {MAX_UPLOAD_SIZE_MB} MB.",
                key="inspector_model_uploader",
                label_visibility="collapsed",
            )
            st.caption(f"Accepted format: **.onnx** · Max upload limit: **{MAX_UPLOAD_SIZE_MB} MB**")
            if uploaded_file is not None:
                # Compare against active model to avoid reparsing identical file
                if current_summary is None or current_summary.filename != uploaded_file.name or current_summary.file_size_bytes != uploaded_file.size:
                    with st.spinner(f"Parsing and inspecting {uploaded_file.name}..."):
                        try:
                            file_bytes = uploaded_file.getvalue()
                            summary = load_and_inspect_model(file_bytes, filename=uploaded_file.name)
                            set_active_model(summary, source_description=f"Uploaded: {uploaded_file.name}", model_bytes=file_bytes)
                            st.success(f"Successfully loaded and validated {uploaded_file.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Upload Error: {str(e)}")

        with col_u2:
            st.markdown(
                """
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.75rem 0.9rem; font-size: 0.76rem; color: #475569; line-height: 1.45;">
                    <div style="font-weight: 600; color: #0F172A; margin-bottom: 0.2rem;">Static Inspection Boundary</div>
                    Model graphs are parsed strictly in-memory. Execution runtimes and hardware compilation are never invoked during static inspection.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with src_tab_sample:
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            sample_choice = st.selectbox(
                "Select a verified sample model:",
                options=[
                    "linear_classifier.onnx (2-layer MLP Classifier, 4 nodes, 643 params)",
                    "conv_net.onnx (CNN Vision Backbone, 5 nodes, 13,610 params)",
                ],
                key="inspector_sample_dropdown",
                label_visibility="collapsed",
            )
        with col_s2:
            if st.button("Load Sample Model", key="inspector_load_sample_btn", use_container_width=True):
                fname = "linear_classifier.onnx" if "linear" in sample_choice else "conv_net.onnx"
                fpath = os.path.join("sample_models", fname)
                if os.path.exists(fpath):
                    with st.spinner(f"Loading {fname}..."):
                        try:
                            m_bytes = Path(fpath).read_bytes()
                            summary = load_and_inspect_model(m_bytes, filename=fname)
                            set_active_model(summary, source_description=f"Sample: {fname}", model_bytes=m_bytes)
                            st.success(f"Loaded {fname}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading sample: {str(e)}")
                else:
                    st.error(f"Sample file not found at {fpath}")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # --- 2. Empty State Handling ---
    if current_summary is None:
        st.markdown(
            """
            <div class="empty-state-container">
                <div class="empty-state-heading">No model loaded for inspection</div>
                <div class="empty-state-copy">
                    Upload an ONNX file above or select a verified sample model to inspect its computational graph topology, tensor signatures, and operator distribution.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # --- 3. Model Overview & Identification Banner ---
    ext_pill = '<span class="status-pill pill-warning" style="margin-left: 0.5rem;">External Data References</span>' if current_summary.has_external_data else ""
    banner_html = (
        '<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1.15rem 1.35rem; margin-bottom: 1.25rem; box-shadow: 0 1px 2px 0 rgba(0,0,0,0.03);">'
        '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.5rem;">'
        '<div style="display: flex; align-items: center; gap: 0.6rem;">'
        '<span class="status-pill pill-success">Validated ONNX</span>'
        f'<span style="font-weight: 600; font-size: 1.05rem; font-family: var(--zm-font-mono); color: #0F172A;">{current_summary.filename}</span>'
        f'{ext_pill}'
        '</div>'
        f'<div style="font-family: var(--zm-font-mono); font-size: 0.74rem; color: #64748B;">SHA-256: <span style="color: #2563EB;">{current_summary.sha256_hash}</span></div>'
        '</div>'
        f'<div style="font-size: 0.8rem; color: #475569;">'
        f'Validation: <b>{current_summary.parsing_status}</b> · Inference: <span style="color: #64748B;">{current_summary.inference_status}</span> · Hardware Target: <span style="color: #64748B;">{current_summary.compatibility_status}</span>'
        '</div>'
        '</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    # Overview Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        render_metric_box("Total Nodes", str(current_summary.total_nodes))
    with m2:
        render_metric_box("Unique Operators", str(current_summary.distinct_operator_count))
    with m3:
        render_metric_box("Parameters", f"{current_summary.total_parameters:,}", subtext="Exact initializer elements")
    with m4:
        render_metric_box("Weights Footprint", format_bytes(current_summary.total_weight_bytes), subtext="Raw payload size")
    with m5:
        render_metric_box("I/O Tensors", f"{len(current_summary.inputs)} in / {len(current_summary.outputs)} out")
    with m6:
        opset_str = f"v{current_summary.opsets[0].version}" if current_summary.opsets else "N/A"
        render_metric_box("Primary Opset", opset_str, subtext=f"IR v{current_summary.metadata.ir_version}")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # --- 4. Detailed Inspection Tabs ---
    tab_io, tab_ops, tab_graph, tab_weights, tab_meta, tab_warnings = st.tabs([
        "Tensor Signatures (I/O)",
        "Operator Analysis",
        "Graph & Connections",
        "Weights & Initializers",
        "Model Metadata",
        f"Inspection Warnings ({len(current_summary.warnings)})",
    ])

    # Tab A: Tensor Signatures (Inputs & Outputs)
    with tab_io:
        st.markdown("###### Model Inputs")
        st.caption("Inputs defined in the ONNX computational graph. Dynamic dimensions (e.g. batch size) are determined at runtime.")
        if current_summary.inputs:
            inputs_records = [
                {
                    "Input Name": inp.name,
                    "Data Type": inp.data_type,
                    "Shape": str(inp.shape),
                    "Rank": f"Rank {inp.rank}" if inp.rank is not None else "Unknown",
                    "Dynamic Axes": "Yes (symbolic)" if inp.is_dynamic else "Fixed",
                    "Description": inp.dimension_description,
                }
                for inp in current_summary.inputs
            ]
            st.dataframe(pd.DataFrame(inputs_records), use_container_width=True, hide_index=True)
        else:
            st.info("No external inputs declared.")

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        st.markdown("###### Model Outputs")
        st.caption("Outputs emitted by the computational graph.")
        if current_summary.outputs:
            outputs_records = [
                {
                    "Output Name": out.name,
                    "Data Type": out.data_type,
                    "Shape": str(out.shape),
                    "Rank": f"Rank {out.rank}" if out.rank is not None else "Unknown",
                    "Dynamic Axes": "Yes (symbolic)" if out.is_dynamic else "Fixed",
                    "Description": out.dimension_description,
                }
                for out in current_summary.outputs
            ]
            st.dataframe(pd.DataFrame(outputs_records), use_container_width=True, hide_index=True)
        else:
            st.info("No outputs declared.")

    # Tab B: Operator Analysis
    with tab_ops:
        st.markdown("###### Operator Frequency Distribution")
        st.caption("Frequencies of operator types occurring across the computational graph. Operator count establishes graph composition only; execution speed depends on hardware kernels and tensor dimensions.")
        if current_summary.op_counts:
            col_chart, col_tbl = st.columns([1.3, 1])
            
            # Prepare operator dataframe
            op_items = [
                {
                    "Operator": op,
                    "Node Count": count,
                    "Share": f"{(count / current_summary.total_nodes) * 100:.1f}%",
                    "Count": count,
                }
                for op, count in current_summary.op_counts.items()
            ]
            df_ops = pd.DataFrame(op_items)

            with col_chart:
                # Restrained horizontal bar chart using Altair
                chart = (
                    alt.Chart(df_ops)
                    .mark_bar(color="#2563EB", cornerRadiusEnd=3)
                    .encode(
                        x=alt.X("Count:Q", title="Node Count", axis=alt.Axis(tickMinStep=1)),
                        y=alt.Y("Operator:N", sort="-x", title="Operator Type"),
                        tooltip=["Operator", "Count", "Share"],
                    )
                    .properties(height=min(400, max(180, len(df_ops) * 32)))
                    .configure_axis(
                        labelFontSize=11,
                        titleFontSize=12,
                        gridColor="#F1F5F9",
                    )
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, use_container_width=True)

            with col_tbl:
                st.dataframe(
                    df_ops[["Operator", "Node Count", "Share"]],
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )
        else:
            st.info("No computational operators in graph.")

    # Tab C: Graph Structure & Connection Inspector
    with tab_graph:
        st.markdown("###### Graph Topology & Connection Inspector")
        st.caption("Inspect graph connectivity, trace tensor dataflow, and analyze predecessor/successor connections between nodes.")

        # Interactive Connection Tracer
        node_names = [n.name for n in current_summary.nodes]
        if node_names:
            c_sel1, c_sel2 = st.columns([2, 1])
            with c_sel1:
                selected_node_name = st.selectbox(
                    "Select a node to inspect connections:",
                    options=node_names,
                    key="connection_node_selector",
                )
            with c_sel2:
                st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
                st.caption(f"Graph contains {len(node_names)} nodes.")

            # Find selected node
            selected_node = next((n for n in current_summary.nodes if n.name == selected_node_name), None)
            if selected_node:
                st.markdown(
                    f"""
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.9rem 1.15rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem;">
                            <span style="font-weight: 600; font-family: var(--zm-font-mono); color: #0F172A; font-size: 0.95rem;">{selected_node.name}</span>
                            <span class="status-pill pill-local">Op: {selected_node.op_type} · Domain: {selected_node.domain}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_in_trace, col_out_trace = st.columns(2)
                with col_in_trace:
                    st.markdown("<b>Incoming Tensors & Predecessor Nodes:</b>", unsafe_allow_html=True)
                    if selected_node.inputs:
                        for inp_t in selected_node.inputs:
                            # Find producer
                            prod = "Graph Input" if any(i.name == inp_t for i in current_summary.inputs) else (
                                "Weight Initializer" if any(w.name == inp_t for w in current_summary.weights) else (
                                    next((n.name for n in current_summary.nodes if inp_t in n.outputs), "Intermediate Tensor")
                                )
                            )
                            st.markdown(f"- Tensor: `{inp_t}` <span style='font-size: 0.78rem; color: #64748B;'>(Produced by: <b>{prod}</b>)</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: #64748B;'>No incoming inputs</span>", unsafe_allow_html=True)

                with col_out_trace:
                    st.markdown("<b>Outgoing Tensors & Consumer Nodes:</b>", unsafe_allow_html=True)
                    if selected_node.outputs:
                        for out_t in selected_node.outputs:
                            consumers = [n.name for n in current_summary.nodes if out_t in n.inputs]
                            is_graph_out = any(o.name == out_t for o in current_summary.outputs)
                            cons_label = ", ".join(consumers) if consumers else ("Graph Output" if is_graph_out else "Terminal Node")
                            st.markdown(f"- Tensor: `{out_t}` <span style='font-size: 0.78rem; color: #64748B;'>(Consumed by: <b>{cons_label}</b>)</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: #64748B;'>No outgoing outputs</span>", unsafe_allow_html=True)

                if selected_node.attributes:
                    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                    st.markdown("<b>Node Attributes:</b>", unsafe_allow_html=True)
                    attrs_df = pd.DataFrame([
                        {"Attribute Name": k, "Configured Value": str(v)}
                        for k, v in selected_node.attributes.items()
                    ])
                    st.dataframe(attrs_df, use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # Full Graph Searchable Table
        st.markdown("###### Complete Graph Nodes Table")
        cf1, cf2 = st.columns([2, 1])
        with cf1:
            search_query = st.text_input("Filter nodes by name or tensor:", placeholder="e.g. Gemm, Conv, input_features", key="table_search_input")
        with cf2:
            filter_op = st.selectbox("Filter by Op Type:", ["All Operators"] + list(current_summary.op_counts.keys()), key="table_filter_op")

        filtered_nodes = current_summary.nodes
        if filter_op != "All Operators":
            filtered_nodes = [n for n in filtered_nodes if n.op_type == filter_op]
        if search_query:
            sq = search_query.lower()
            filtered_nodes = [
                n for n in filtered_nodes
                if sq in n.name.lower() or sq in n.op_type.lower() or any(sq in inp.lower() for inp in n.inputs) or any(sq in out.lower() for out in n.outputs)
            ]

        nodes_table_data = [
            {
                "Index": n.index,
                "Node Name": n.name,
                "Op Type": n.op_type,
                "Domain": n.domain,
                "Inputs": ", ".join(n.inputs),
                "Outputs": ", ".join(n.outputs),
                "Predecessors": len(n.predecessors),
                "Successors": len(n.successors),
            }
            for n in filtered_nodes
        ]
        st.dataframe(pd.DataFrame(nodes_table_data), use_container_width=True, hide_index=True, height=350)

    # Tab D: Weights & Initializers
    with tab_weights:
        st.markdown("###### Initializer Tensors (Weights & Biases)")
        st.caption("Model parameters stored as initializers in the graph. Exact byte footprint calculated without fabrication.")
        if current_summary.weights:
            weights_data = [
                {
                    "Tensor Name": w.name,
                    "Data Type": w.data_type,
                    "Shape": str(w.shape),
                    "Elements": f"{w.num_elements:,}",
                    "Size": format_bytes(w.size_bytes),
                    "Storage": "External Reference" if w.has_external_data else "Embedded Payload",
                }
                for w in current_summary.weights
            ]
            st.dataframe(pd.DataFrame(weights_data), use_container_width=True, hide_index=True)
        else:
            st.info("No weight initializers present in graph (e.g. architecture topology without bound weights).")

    # Tab E: Model Metadata
    with tab_meta:
        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown("###### Model Provenance")
            meta_records = [
                {"Property": "Producer Name", "Value": current_summary.metadata.producer_name},
                {"Property": "Producer Version", "Value": current_summary.metadata.producer_version},
                {"Property": "Model Version", "Value": str(current_summary.metadata.model_version)},
                {"Property": "ONNX IR Version", "Value": str(current_summary.metadata.ir_version)},
                {"Property": "Domain", "Value": current_summary.metadata.domain},
                {"Property": "Docstring", "Value": current_summary.metadata.doc_string},
                {"Property": "SHA-256 Fingerprint", "Value": current_summary.sha256_hash},
                {"Property": "File Size", "Value": f"{format_bytes(current_summary.file_size_bytes)} ({current_summary.file_size_bytes} bytes)"},
            ]
            st.dataframe(pd.DataFrame(meta_records), use_container_width=True, hide_index=True)

        with cm2:
            st.markdown("###### Opset Imports")
            opsets_data = [
                {"Domain": op.domain, "Opset Version": f"v{op.version}"}
                for op in current_summary.opsets
            ]
            st.dataframe(pd.DataFrame(opsets_data), use_container_width=True, hide_index=True)

            if current_summary.metadata.custom_metadata:
                st.markdown("###### Custom Metadata Key-Values")
                custom_df = pd.DataFrame([
                    {"Key": k, "Value": v}
                    for k, v in current_summary.metadata.custom_metadata.items()
                ])
                st.dataframe(custom_df, use_container_width=True, hide_index=True)

    # Tab F: Inspection Warnings
    with tab_warnings:
        st.markdown("###### Grounded Inspection Warnings & Advisories")
        st.caption("Warnings generated strictly from graph properties, metadata presence, and tensor signatures.")
        if current_summary.warnings:
            for w in current_summary.warnings:
                render_warning_item(w)
        else:
            st.markdown(
                """
                <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 6px; padding: 1rem 1.25rem; color: #065F46;">
                    <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem;">No Structural Anomalies Detected</div>
                    <div style="font-size: 0.82rem;">The computational graph declares complete metadata, standard operator domains, embedded initializers, and static tensor bounds.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
