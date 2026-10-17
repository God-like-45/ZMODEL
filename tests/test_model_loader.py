"""Comprehensive unit tests for ONNX model loader, schema inspector, and validation engine.

Covers all Phase 2 acceptance criteria using synthetic models without external downloads:
1. Small valid ONNX model with known inputs, outputs, and operators.
2. Multiple operator types and exact operator counts.
3. Symbolic and dynamic input dimensions.
4. Missing optional metadata and honest fallbacks ("Not available").
5. Invalid or corrupted ONNX content.
6. Unsupported file extensions.
7. Oversized uploads.
8. Unusual but valid graph structures (multi-branch, multi-output, identity pass-through).
9. Switching between two different uploaded models without stale results.
10. Correct SHA-256 fingerprint calculation.
"""

import hashlib
import os
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest
from zmodel.core.config import MAX_UPLOAD_SIZE_BYTES
from zmodel.core.model_loader import (
    load_and_inspect_model,
    validate_file_constraints,
)


def _build_minimal_onnx_model(
    model_name: str = "TestModel",
    include_metadata: bool = True,
    dynamic_batch: bool = True,
) -> bytes:
    """Helper to synthesize a clean, valid ONNX model in memory."""
    batch_dim = "batch_size" if dynamic_batch else 1
    x = helper.make_tensor_value_info("x_input", TensorProto.FLOAT, [batch_dim, 4])
    y = helper.make_tensor_value_info("y_output", TensorProto.FLOAT, [batch_dim, 2])

    w_val = np.ones((4, 2), dtype=np.float32)
    b_val = np.zeros(2, dtype=np.float32)

    w_init = helper.make_tensor("w", TensorProto.FLOAT, [4, 2], w_val.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, [2], b_val.flatten().tolist())

    node_gemm = helper.make_node("Gemm", ["x_input", "w", "b"], ["gemm_out"], name="node_gemm")
    node_relu = helper.make_node("Relu", ["gemm_out"], ["y_output"], name="node_relu")

    graph = helper.make_graph(
        nodes=[node_gemm, node_relu],
        name=model_name,
        inputs=[x],
        outputs=[y],
        initializer=[w_init, b_init],
        doc_string="Test graph docstring" if include_metadata else "",
    )

    producer = "SyntheticUnitTests" if include_metadata else ""
    version_str = "1.0.0" if include_metadata else ""

    model = helper.make_model(
        graph,
        producer_name=producer,
        producer_version=version_str,
        opset_imports=[helper.make_opsetid("", 17)],
        doc_string="Model level documentation" if include_metadata else "",
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _build_unusual_multi_output_graph() -> bytes:
    """Build a multi-branch ONNX model with multiple outputs and Identity node."""
    x = helper.make_tensor_value_info("feat_in", TensorProto.FLOAT, ["batch", 8])
    out_main = helper.make_tensor_value_info("out_primary", TensorProto.FLOAT, ["batch", 8])
    out_aux = helper.make_tensor_value_info("out_secondary", TensorProto.FLOAT, ["batch", 8])

    # Node 1: Relu
    n_relu = helper.make_node("Relu", ["feat_in"], ["relu_out"], name="relu_branch")
    # Node 2: Identity pass-through to output 1
    n_id = helper.make_node("Identity", ["relu_out"], ["out_primary"], name="identity_tap")
    # Node 3: Abs on the same relu_out to output 2
    n_abs = helper.make_node("Abs", ["relu_out"], ["out_secondary"], name="abs_branch")

    graph = helper.make_graph(
        nodes=[n_relu, n_id, n_abs],
        name="MultiOutputGraph",
        inputs=[x],
        outputs=[out_main, out_aux],
        initializer=[],
    )
    model = helper.make_model(
        graph,
        producer_name="MultiBranchBuilder",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


# --- Test 1: Small valid model with known inputs, outputs, and operators ---
def test_small_valid_onnx_model():
    raw_bytes = _build_minimal_onnx_model(include_metadata=True, dynamic_batch=False)
    summary = load_and_inspect_model(raw_bytes, filename="test_mlp.onnx")

    assert summary.filename == "test_mlp.onnx"
    assert summary.is_valid is True
    assert summary.total_nodes == 2
    assert summary.distinct_operator_count == 2
    assert len(summary.inputs) == 1
    assert summary.inputs[0].name == "x_input"
    assert summary.inputs[0].data_type == "float32"
    assert summary.inputs[0].shape == [1, 4]
    assert summary.inputs[0].rank == 2

    assert len(summary.outputs) == 1
    assert summary.outputs[0].name == "y_output"
    assert summary.outputs[0].shape == [1, 2]
    assert summary.outputs[0].rank == 2


# --- Test 2: Multiple operator types and correct operator counts ---
def test_operator_counts_and_distribution():
    raw_bytes = _build_minimal_onnx_model()
    summary = load_and_inspect_model(raw_bytes, filename="counts.onnx")

    assert summary.total_nodes == 2
    assert summary.op_counts == {"Gemm": 1, "Relu": 1}
    assert summary.distinct_operator_count == 2
    # Verify node connections
    node_gemm = next(n for n in summary.nodes if n.op_type == "Gemm")
    node_relu = next(n for n in summary.nodes if n.op_type == "Relu")
    assert "node_relu" in node_gemm.successors
    assert "node_gemm" in node_relu.predecessors


# --- Test 3: Symbolic or dynamic input dimensions ---
def test_symbolic_dynamic_dimensions():
    raw_bytes = _build_minimal_onnx_model(dynamic_batch=True)
    summary = load_and_inspect_model(raw_bytes, filename="dynamic.onnx")

    inp = summary.inputs[0]
    assert inp.is_dynamic is True
    assert "batch_size" in inp.dynamic_dim_names
    assert inp.shape == ["batch_size", 4]
    assert inp.rank == 2
    assert "dynamic" in inp.dimension_description.lower()

    # Grounded warning generated for dynamic inputs
    warn_codes = [w.code for w in summary.warnings]
    assert "WARN_DYNAMIC_INPUTS" in warn_codes


# --- Test 4: Missing optional metadata ---
def test_missing_optional_metadata():
    raw_bytes = _build_minimal_onnx_model(include_metadata=False)
    summary = load_and_inspect_model(raw_bytes, filename="no_meta.onnx")

    assert summary.metadata.producer_name == "Not available"
    assert summary.metadata.producer_version == "Not available"
    assert summary.metadata.doc_string == "Not available"
    
    warn_codes = [w.code for w in summary.warnings]
    assert "WARN_MISSING_METADATA" in warn_codes


# --- Test 5: Invalid or corrupted ONNX content ---
def test_corrupted_onnx_bytes():
    corrupted_bytes = b"CORRUPTED_NOT_PROTOBUF_DATA_12345678"
    with pytest.raises(ValueError, match="Failed to parse ONNX protobuf"):
        load_and_inspect_model(corrupted_bytes, filename="broken.onnx")


# --- Test 6: Unsupported file extensions ---
def test_unsupported_extensions_rejected():
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_file_constraints("model.pt", 1024)
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_file_constraints("model.tflite", 1024)
    with pytest.raises(ValueError, match="Unsupported file format"):
        validate_file_constraints("model.bin", 1024)


# --- Test 7: Oversized uploads ---
def test_oversized_upload_rejected():
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        validate_file_constraints("giant_model.onnx", MAX_UPLOAD_SIZE_BYTES + 1)


# --- Test 8: Unusual but valid graph structure (Multi-branch, multi-output, Identity) ---
def test_unusual_multi_output_graph():
    raw_bytes = _build_unusual_multi_output_graph()
    summary = load_and_inspect_model(raw_bytes, filename="unusual.onnx")

    assert summary.is_valid is True
    assert summary.total_nodes == 3
    assert len(summary.outputs) == 2
    output_names = [o.name for o in summary.outputs]
    assert "out_primary" in output_names
    assert "out_secondary" in output_names

    # Check connection tracing on Identity and Abs nodes
    id_node = next(n for n in summary.nodes if n.op_type == "Identity")
    abs_node = next(n for n in summary.nodes if n.op_type == "Abs")
    assert "relu_branch" in id_node.predecessors
    assert "relu_branch" in abs_node.predecessors

    # Warnings for multi-output and zero-weights
    warn_codes = [w.code for w in summary.warnings]
    assert "WARN_MULTI_OUTPUT" in warn_codes
    assert "WARN_ZERO_WEIGHTS" in warn_codes


# --- Test 9: Switching between two different uploaded models without stale results ---
def test_switching_models_without_stale_results():
    bytes_a = _build_minimal_onnx_model(model_name="Model_A")
    bytes_b = _build_unusual_multi_output_graph()

    summary_a = load_and_inspect_model(bytes_a, filename="model_a.onnx")
    assert summary_a.filename == "model_a.onnx"
    assert summary_a.total_nodes == 2
    assert len(summary_a.outputs) == 1

    summary_b = load_and_inspect_model(bytes_b, filename="model_b.onnx")
    assert summary_b.filename == "model_b.onnx"
    assert summary_b.total_nodes == 3
    assert len(summary_b.outputs) == 2

    # Verify no leaked attributes between distinct model summaries
    assert summary_a.sha256_hash != summary_b.sha256_hash
    assert summary_a.total_nodes != summary_b.total_nodes


# --- Test 10: Correct SHA-256 fingerprint calculation ---
def test_sha256_fingerprint_calculation():
    raw_bytes = _build_minimal_onnx_model()
    expected_hash = hashlib.sha256(raw_bytes).hexdigest()
    summary = load_and_inspect_model(raw_bytes, filename="checksum_test.onnx")

    assert summary.sha256_hash == expected_hash
    assert len(summary.sha256_hash) == 64
