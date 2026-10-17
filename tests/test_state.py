"""Unit tests for state management."""

from unittest.mock import patch
import pytest
from zmodel.core.model_loader import (
    ModelMetadata,
    ModelSummary,
    OpsetInfo,
    TensorInfo,
)
from zmodel.core.state import (
    clear_active_model,
    get_active_model,
    init_session_state,
    navigate_to,
    set_active_model,
)


@pytest.fixture(autouse=True)
def mock_session_state():
    """Mock Streamlit session_state dict for pure Python tests."""
    state_dict = {}
    with patch("streamlit.session_state", state_dict):
        yield state_dict


def test_init_session_state(mock_session_state):
    init_session_state()
    assert mock_session_state["current_page"] == "overview"
    assert mock_session_state["active_model"] is None


def test_navigate_to(mock_session_state):
    init_session_state()
    navigate_to("inspector")
    assert mock_session_state["current_page"] == "inspector"


def test_set_and_clear_active_model(mock_session_state):
    init_session_state()
    fake_meta = ModelMetadata(
        producer_name="test",
        producer_version="1.0",
        model_version=1,
        ir_version=9,
        doc_string="",
        domain="",
        custom_metadata={},
    )
    fake_summary = ModelSummary(
        filename="dummy.onnx",
        file_size_bytes=1024,
        sha256_hash="abc",
        metadata=fake_meta,
        opsets=[OpsetInfo("ai.onnx", 17)],
        inputs=[TensorInfo("in", "float32", [1, 10], 2, False, [], "Rank 2 fixed-shape tensor")],
        outputs=[TensorInfo("out", "float32", [1, 2], 2, False, [], "Rank 2 fixed-shape tensor")],
        nodes=[],
        weights=[],
        op_counts={},
        distinct_operator_count=0,
        total_nodes=0,
        total_parameters=0,
        total_weight_bytes=0,
        has_external_data=False,
        external_data_tensors=[],
        warnings=[],
        is_valid=True,
        parsing_status="Successfully Parsed & Graph Constraints Verified",
        inference_status="Not Executed (Static Inspection Only)",
        compatibility_status="Not Analyzed (Scheduled for Future Phase)",
    )
    set_active_model(fake_summary, "Unit Test")
    assert get_active_model() is not None
    assert get_active_model().filename == "dummy.onnx"

    clear_active_model()
    assert get_active_model() is None
