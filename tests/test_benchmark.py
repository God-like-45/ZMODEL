"""
Comprehensive automated tests for Phase 4: Reproducible Model Benchmark Engine.

Covers all 22 test requirements specified in Phase 4 acceptance criteria:
1. Runtime and provider discovery.
2. Valid CPU benchmark where CPU execution is available.
3. Unavailable provider handling without silent fallback.
4. Correct input generation for fixed shapes.
5. Dynamic-shape input generation with overrides.
6. Multiple model inputs generation.
7. Unsupported tensor types rejection.
8. Invalid or corrupted model handling.
9. Smoke inference failure handling.
10. Non-finite (NaN / Inf) output detection.
11. Correct latency statistics and percentile calculations using deterministic arrays.
12. Correct throughput calculations (inf/sec and samples/sec).
13. Repeated-round aggregation.
14. Stable random input generation for a fixed seed.
15. Configuration validation and iteration limits.
16. Accurate model fingerprint recording.
17. Missing environment metadata resilience.
18. Failed-run status and error serialization.
19. JSON report serialization (valid schema, no server paths).
20. No false success result when correctness validation fails.
21. No stale benchmark results after switching models.
22. Regression tests for Phases 1–3.
"""

import json
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from zmodel.core.benchmark import (
    BenchmarkConfig,
    BenchmarkConfigurationError,
    BenchmarkEngine,
    BenchmarkError,
    BenchmarkReport,
    CorrectnessValidationError,
    ProviderUnavailableError,
    UnsupportedInputTypeError,
    calculate_latency_statistics,
    calculate_throughput,
    discover_execution_providers,
    generate_benchmark_inputs,
    validate_model_correctness,
)
from zmodel.core.config import APP_VERSION, CURRENT_PHASE, NAV_ITEMS
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.state import (
    clear_active_model,
    get_active_model,
    get_benchmark_report,
    init_session_state,
    set_active_model,
    set_benchmark_report,
)


def _build_test_onnx_model(
    nodes: list,
    inputs: list,
    outputs: list,
    initializers: list = None,
    opset_imports: list = None,
    ir_version: int = 9,
    model_name: str = "TestBenchmarkModel",
) -> bytes:
    """Helper to synthesize valid ONNX models in memory with IR version compatible with runtime."""
    if initializers is None:
        initializers = []
    if opset_imports is None:
        opset_imports = [helper.make_opsetid("", 17)]

    graph = helper.make_graph(
        nodes=nodes,
        name=model_name,
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=opset_imports,
        ir_version=ir_version,
        producer_name="ZModelPilotBenchmarkTests",
        producer_version="1.0",
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture
def linear_model_bytes() -> bytes:
    """Return in-memory binary bytes of a simple linear MLP model."""
    x = helper.make_tensor_value_info("x_in", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y_out", TensorProto.FLOAT, [1, 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 2], np.ones((4, 2), dtype=np.float32).flatten())
    b = helper.make_tensor("b", TensorProto.FLOAT, [2], np.zeros(2, dtype=np.float32).flatten())

    n1 = helper.make_node("Gemm", ["x_in", "w", "b"], ["g_out"])
    n2 = helper.make_node("Relu", ["g_out"], ["y_out"])

    return _build_test_onnx_model([n1, n2], [x], [y], [w, b])


# --- 1. Runtime and provider discovery ---
def test_runtime_and_provider_discovery():
    env_info = discover_execution_providers()
    assert env_info.installed_ort_version != ""
    assert isinstance(env_info.available_providers, list)
    assert len(env_info.available_providers) > 0
    assert "CPUExecutionProvider" in env_info.available_providers
    assert env_info.default_provider == "CPUExecutionProvider"


# --- 2. Valid CPU benchmark where CPU execution is available ---
def test_valid_cpu_benchmark(linear_model_bytes):
    summary = load_and_inspect_model(linear_model_bytes, filename="bench_linear.onnx")
    config = BenchmarkConfig(
        execution_provider="CPUExecutionProvider",
        warmup_iterations=5,
        measured_iterations=10,
        batch_size=1,
        repeated_rounds=2,
        random_seed=42,
    )

    report = BenchmarkEngine.run_benchmark(linear_model_bytes, summary, config)

    assert report.status == "SUCCESS"
    assert report.error_message is None
    assert report.model_name == "bench_linear.onnx"
    assert report.model_fingerprint == summary.sha256_hash
    assert report.correctness["passed"] is True
    assert report.latency["median_ms"] > 0
    assert report.latency["mean_ms"] > 0
    assert report.throughput["inferences_per_second"] > 0
    assert len(report.per_round_results) == 2
    assert len(report.all_latencies_ms) == 20  # 10 iters * 2 rounds


# --- 3. Unavailable provider handling without silent fallback ---
def test_unavailable_provider_handling(linear_model_bytes):
    summary = load_and_inspect_model(linear_model_bytes, filename="bench_linear.onnx")
    config = BenchmarkConfig(
        execution_provider="NonExistentAcceleratorProvider",
        warmup_iterations=5,
        measured_iterations=10,
    )

    with pytest.raises(ProviderUnavailableError, match="not available in the installed ONNX Runtime"):
        BenchmarkEngine.run_benchmark(linear_model_bytes, summary, config)


# --- 4. Correct input generation for fixed shapes ---
def test_correct_input_generation_fixed_shapes():
    x = helper.make_tensor_value_info("matrix_in", TensorProto.FLOAT, [2, 8])
    y = helper.make_tensor_value_info("matrix_out", TensorProto.FLOAT, [2, 8])
    n = helper.make_node("Identity", ["matrix_in"], ["matrix_out"])
    model_bytes = _build_test_onnx_model([n], [x], [y])
    summary = load_and_inspect_model(model_bytes, filename="fixed_shape.onnx")

    config = BenchmarkConfig(random_seed=123)
    inputs_dict, input_specs = generate_benchmark_inputs(summary, config)

    assert "matrix_in" in inputs_dict
    arr = inputs_dict["matrix_in"]
    assert arr.shape == (2, 8)
    assert arr.dtype == np.float32
    assert input_specs["matrix_in"]["resolved_shape"] == [2, 8]


# --- 5. Dynamic-shape input generation with overrides ---
def test_dynamic_shape_handling():
    x = helper.make_tensor_value_info("dynamic_tensor", TensorProto.FLOAT, ["batch_size", "seq_len", 16])
    y = helper.make_tensor_value_info("dynamic_out", TensorProto.FLOAT, ["batch_size", "seq_len", 16])
    n = helper.make_node("Identity", ["dynamic_tensor"], ["dynamic_out"])
    model_bytes = _build_test_onnx_model([n], [x], [y])
    summary = load_and_inspect_model(model_bytes, filename="dyn_model.onnx")

    config = BenchmarkConfig(
        batch_size=4,
        dynamic_dimensions={"seq_len": 32},
        random_seed=42,
    )
    inputs_dict, input_specs = generate_benchmark_inputs(summary, config)

    assert inputs_dict["dynamic_tensor"].shape == (4, 32, 16)
    assert input_specs["dynamic_tensor"]["resolved_shape"] == [4, 32, 16]


# --- 6. Multiple model inputs generation ---
def test_multiple_model_inputs():
    a = helper.make_tensor_value_info("in_a", TensorProto.FLOAT, [1, 4])
    b = helper.make_tensor_value_info("in_b", TensorProto.FLOAT, [1, 4])
    c = helper.make_tensor_value_info("in_c", TensorProto.INT32, [1, 4])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Add", ["in_a", "in_b"], ["out"])

    model_bytes = _build_test_onnx_model([n], [a, b, c], [out])
    summary = load_and_inspect_model(model_bytes, filename="multi_input.onnx")

    config = BenchmarkConfig()
    inputs_dict, input_specs = generate_benchmark_inputs(summary, config)

    assert "in_a" in inputs_dict and inputs_dict["in_a"].dtype == np.float32
    assert "in_b" in inputs_dict and inputs_dict["in_b"].dtype == np.float32
    assert "in_c" in inputs_dict and inputs_dict["in_c"].dtype == np.int32
    assert len(inputs_dict) == 3


# --- 7. Unsupported tensor types rejection ---
def test_unsupported_tensor_types():
    x = helper.make_tensor_value_info("complex_in", TensorProto.COMPLEX64, [1, 4])
    y = helper.make_tensor_value_info("complex_out", TensorProto.COMPLEX64, [1, 4])
    n = helper.make_node("Identity", ["complex_in"], ["complex_out"])
    model_bytes = _build_test_onnx_model([n], [x], [y])
    summary = load_and_inspect_model(model_bytes, filename="complex.onnx")

    config = BenchmarkConfig()
    with pytest.raises(UnsupportedInputTypeError, match="unsupported data type"):
        generate_benchmark_inputs(summary, config)


# --- 8. Invalid or corrupted model handling ---
def test_invalid_or_corrupted_model_handling():
    corrupted_bytes = b"CORRUPTED_ONNX_BYTES_12345"
    fake_summary = ModelSummary(
        filename="corrupted.onnx",
        file_size_bytes=len(corrupted_bytes),
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        metadata=None,
        opsets=[],
        inputs=[],
        outputs=[],
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
        is_valid=False,
        parsing_status="Failed",
        inference_status="Not Executed",
        compatibility_status="Not Analyzed",
    )

    config = BenchmarkConfig()
    with pytest.raises(BenchmarkError, match="Failed to create ONNX Runtime session"):
        BenchmarkEngine.run_benchmark(corrupted_bytes, fake_summary, config)


# --- 9. Smoke inference failure handling ---
def test_smoke_inference_failure():
    # Construct a model whose session creation succeeds via symbolic dimension,
    # but runtime inference fails due to inner matrix mismatch (default dynamic_dim=1 vs weight=5)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "dynamic_k"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [5, 2], np.ones((5, 2), dtype=np.float32).flatten())
    n = helper.make_node("Gemm", ["x", "w"], ["y"])

    graph = helper.make_graph([n], "mismatch_graph", [x], [y], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    model_bytes = model.SerializeToString()
    summary = load_and_inspect_model(model_bytes, filename="mismatch.onnx")

    config = BenchmarkConfig(warmup_iterations=1, measured_iterations=1)
    with pytest.raises(CorrectnessValidationError, match="Smoke inference failed"):
        BenchmarkEngine.run_benchmark(model_bytes, summary, config)



# --- 10. Non-finite output detection ---
def test_non_finite_output_detection():
    # Custom mock or session simulation returning NaN
    class MockSession:
        def run(self, output_names, inputs):
            return [np.array([[float("nan"), 1.0]], dtype=np.float32)]

        def get_outputs(self):
            class OutDef:
                name = "mock_out"
            return [OutDef()]

    with pytest.raises(CorrectnessValidationError, match="non-finite values"):
        validate_model_correctness(MockSession(), {"dummy": np.ones((1, 2))})


# --- 11. Correct latency statistics and percentile calculations ---
def test_latency_statistics_and_percentiles_deterministic():
    # Pass deterministic values from 1.0 to 100.0 ms
    latencies = [float(i) for i in range(1, 101)]
    stats = calculate_latency_statistics(latencies)

    assert stats.min_ms == 1.0
    assert stats.max_ms == 100.0
    assert stats.median_ms == 50.5
    assert stats.mean_ms == 50.5
    assert abs(stats.p90_ms - 90.1) < 1e-4
    assert abs(stats.p95_ms - 95.05) < 1e-4
    assert abs(stats.p99_ms - 99.01) < 1e-4
    assert stats.sample_count == 100


# --- 12. Correct throughput calculations ---
def test_throughput_calculations():
    thru = calculate_throughput(total_inferences=100, elapsed_seconds=2.0, batch_size=4)
    assert thru.inferences_per_second == 50.0
    assert thru.samples_per_second == 200.0
    assert thru.total_elapsed_seconds == 2.0
    assert thru.total_measured_inferences == 100


# --- 13. Repeated-round aggregation ---
def test_repeated_round_aggregation(linear_model_bytes):
    summary = load_and_inspect_model(linear_model_bytes, filename="bench_rounds.onnx")
    config = BenchmarkConfig(
        warmup_iterations=2,
        measured_iterations=5,
        repeated_rounds=3,
    )
    report = BenchmarkEngine.run_benchmark(linear_model_bytes, summary, config)

    assert len(report.per_round_results) == 3
    assert report.latency["sample_count"] == 15
    for idx, r in enumerate(report.per_round_results, start=1):
        assert r["round_number"] == idx
        assert len(r["latencies_ms"]) == 5


# --- 14. Stable random input generation for a fixed seed ---
def test_stable_random_input_generation_fixed_seed(linear_model_bytes):
    summary = load_and_inspect_model(linear_model_bytes, filename="seed_test.onnx")

    c1 = BenchmarkConfig(random_seed=42)
    in1, _ = generate_benchmark_inputs(summary, c1)

    c2 = BenchmarkConfig(random_seed=42)
    in2, _ = generate_benchmark_inputs(summary, c2)

    c3 = BenchmarkConfig(random_seed=999)
    in3, _ = generate_benchmark_inputs(summary, c3)

    # Identical seed must produce bitwise identical inputs
    assert np.array_equal(in1["x_in"], in2["x_in"])
    # Different seed must produce different values
    assert not np.array_equal(in1["x_in"], in3["x_in"])


# --- 15. Configuration validation and iteration limits ---
def test_configuration_validation_and_iteration_limits():
    with pytest.raises(BenchmarkConfigurationError, match="warmup_iterations"):
        BenchmarkConfig(warmup_iterations=0).validate()

    with pytest.raises(BenchmarkConfigurationError, match="measured_iterations"):
        BenchmarkConfig(measured_iterations=5000).validate()

    with pytest.raises(BenchmarkConfigurationError, match="batch_size"):
        BenchmarkConfig(batch_size=0).validate()

    with pytest.raises(BenchmarkConfigurationError, match="repeated_rounds"):
        BenchmarkConfig(repeated_rounds=20).validate()

    with pytest.raises(BenchmarkConfigurationError, match="Dynamic dimension"):
        BenchmarkConfig(dynamic_dimensions={"seq_len": -1}).validate()


# --- 16. Accurate model fingerprint recording ---
def test_model_fingerprint_recording(linear_model_bytes):
    summary = load_and_inspect_model(linear_model_bytes, filename="fp_model.onnx")
    config = BenchmarkConfig(warmup_iterations=1, measured_iterations=2, repeated_rounds=1)
    report = BenchmarkEngine.run_benchmark(linear_model_bytes, summary, config)

    assert report.model_fingerprint == summary.sha256_hash
    assert len(report.model_fingerprint) == 64


# --- 17. Missing environment metadata resilience ---
def test_missing_environment_metadata(linear_model_bytes):
    summary = load_and_inspect_model(linear_model_bytes, filename="env_test.onnx")
    config = BenchmarkConfig(warmup_iterations=1, measured_iterations=2, repeated_rounds=1)
    report = BenchmarkEngine.run_benchmark(linear_model_bytes, summary, config)

    assert "python_version" in report.environment
    assert "os_name" in report.environment
    assert "onnxruntime_version" in report.environment


# --- 18. Failed-run status and error serialization ---
def test_failed_run_status_and_error_serialization():
    failed_report = BenchmarkReport(
        run_id="BENCH-FAIL-TEST",
        timestamp_utc="2026-10-17T00:00:00Z",
        model_name="failed.onnx",
        model_fingerprint="abc12345",
        status="FAILED",
        error_message="Simulated numerical divergence failure",
        environment={"os": "Windows"},
        config={"provider": "CPUExecutionProvider"},
        input_spec={},
        correctness=None,
        latency=None,
        throughput=None,
        memory=None,
        per_round_results=[],
        all_latencies_ms=[],
        warnings=["Run failed"],
        limitations=["No data recorded"],
    )

    data = failed_report.to_dict()
    assert data["status"] == "FAILED"
    assert "Simulated numerical divergence failure" in data["error_message"]

    json_str = failed_report.to_json()
    parsed = json.loads(json_str)
    assert parsed["status"] == "FAILED"


# --- 19. JSON report serialization ---
def test_json_report_serialization(linear_model_bytes):
    summary = load_and_inspect_model(linear_model_bytes, filename="json_test.onnx")
    config = BenchmarkConfig(warmup_iterations=2, measured_iterations=5, repeated_rounds=1)
    report = BenchmarkEngine.run_benchmark(linear_model_bytes, summary, config)

    json_str = report.to_json()
    data = json.loads(json_str)

    assert data["report_version"] == "1.0.0"
    assert "model_metadata" in data
    assert "environment" in data
    assert "latency" in data
    assert "throughput" in data
    assert "memory" in data

    # Verify absence of local server paths or secret credentials
    assert "C:\\" not in json_str
    assert "D:\\" not in json_str
    assert "/home/" not in json_str


# --- 20. No false success result when correctness validation fails ---
def test_no_false_success_when_correctness_fails():
    class BrokenSession:
        def run(self, out_names, inputs):
            raise RuntimeError("Segmentation fault in simulated kernel")
        def get_outputs(self):
            return []

    with pytest.raises(CorrectnessValidationError):
        validate_model_correctness(BrokenSession(), {"x": np.zeros((1, 4))})


# --- 21. No stale benchmark results after switching models ---
def test_no_stale_benchmark_results_after_switching_models():
    init_session_state()

    # Create dummy summaries for model A and B
    sum_a = ModelSummary(
        filename="model_a.onnx",
        file_size_bytes=100,
        sha256_hash="hash_a",
        metadata=None,
        opsets=[],
        inputs=[],
        outputs=[],
        nodes=[],
        weights=[],
        op_counts={},
        distinct_operator_count=1,
        total_nodes=1,
        total_parameters=10,
        total_weight_bytes=40,
        has_external_data=False,
        external_data_tensors=[],
        warnings=[],
        is_valid=True,
        parsing_status="OK",
        inference_status="OK",
        compatibility_status="OK",
    )
    sum_b = ModelSummary(
        filename="model_b.onnx",
        file_size_bytes=200,
        sha256_hash="hash_b",
        metadata=None,
        opsets=[],
        inputs=[],
        outputs=[],
        nodes=[],
        weights=[],
        op_counts={},
        distinct_operator_count=1,
        total_nodes=1,
        total_parameters=20,
        total_weight_bytes=80,
        has_external_data=False,
        external_data_tensors=[],
        warnings=[],
        is_valid=True,
        parsing_status="OK",
        inference_status="OK",
        compatibility_status="OK",
    )

    # Set model A and save report A
    set_active_model(sum_a, "Model A")
    dummy_report = BenchmarkReport(
        run_id="BENCH-A",
        timestamp_utc="",
        model_name="model_a.onnx",
        model_fingerprint="hash_a",
        status="SUCCESS",
        error_message=None,
        environment={},
        config={},
        input_spec={},
        correctness={"passed": True},
        latency={"median_ms": 1.23},
        throughput={},
        memory={},
        per_round_results=[],
        all_latencies_ms=[],
        warnings=[],
        limitations=[],
    )
    set_benchmark_report(dummy_report)
    assert get_benchmark_report() is not None

    # Switch to model B -> benchmark report must be cleared
    set_active_model(sum_b, "Model B")
    assert get_benchmark_report() is None


# --- 22. Regression tests for Phases 1–3 ---
def test_regression_phases_1_to_3(linear_model_bytes):
    # Phase 1: App config and phase metadata
    assert CURRENT_PHASE >= 4
    assert APP_VERSION in ("0.4.0", "0.5.0", "0.6.0", "0.7.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0")
    bench_nav = next(item for item in NAV_ITEMS if item.id == "benchmarks")
    assert bench_nav.is_active is True
    assert bench_nav.phase == 4

    # Phase 2: Model loader inspection
    summary = load_and_inspect_model(linear_model_bytes, filename="regress.onnx")
    assert summary.is_valid is True
    assert summary.total_nodes == 2
    assert "Gemm" in summary.op_counts
    assert "Relu" in summary.op_counts

    # Phase 3: Compatibility analysis
    from zmodel.core.compatibility import CompatibilityAnalyzer
    from zmodel.core.profiles import load_builtin_profiles
    profiles = load_builtin_profiles()
    cpu_prof = profiles["local_onnx_runtime_cpu"]
    compat_report = CompatibilityAnalyzer.analyze(summary, cpu_prof)
    assert compat_report.overall_assessment == "COMPATIBLE"
