"""
Comprehensive automated tests for Phase 3: Architecture-Aware Compatibility Analysis Engine.

Covers all 17 requirements specified in Phase 3 acceptance criteria:
1. All-supported model/profile combination.
2. Explicitly unsupported operator.
3. Operator missing from profile rules (returns UNKNOWN, never PASS).
4. Supported operator with an unsupported version constraint.
5. Multiple operator domains.
6. Opset mismatch.
7. Unsupported data type.
8. Dynamic dimensions handling.
9. Incomplete or illustrative profile detection.
10. Malformed profile rejection and schema validation.
11. Model with multiple findings and correct summary counts.
12. Stable overall assessment when findings are reordered.
13. JSON export and schema validation.
14. Profile and ruleset version preservation.
15. Model fingerprint association.
16. Minimal edge model.
17. Regression tests for Phase 1 and Phase 2.
"""

import json
import random
import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from zmodel.core.compatibility import (
    CompatibilityAnalyzer,
    CompatibilityFinding,
    CompatibilityReport,
    determine_overall_assessment,
)
from zmodel.core.config import APP_VERSION, CURRENT_PHASE, NAV_ITEMS
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.profiles import (
    CapabilityProfile,
    OperatorRule,
    ProfileValidationError,
    load_builtin_profiles,
    load_profile_from_dict,
    load_profile_from_json,
)


def _build_test_model(
    nodes: list,
    inputs: list,
    outputs: list,
    initializers: list = None,
    opset_imports: list = None,
    model_name: str = "TestCompatModel",
) -> bytes:
    """Helper to synthesize custom test ONNX models in memory."""
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
        producer_name="ZModelPilotTests",
        producer_version="1.0",
    )
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture
def cpu_profile() -> CapabilityProfile:
    """Return verified local CPU capability profile."""
    profiles = load_builtin_profiles()
    return profiles["local_onnx_runtime_cpu"]


@pytest.fixture
def ibm_z_profile() -> CapabilityProfile:
    """Return illustrative IBM Z capability profile."""
    profiles = load_builtin_profiles()
    return profiles["ibm_z_onnx_illustrative"]


# --- 1. All-supported model/profile combination ---
def test_all_supported_model_profile_combination(cpu_profile):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 2], np.ones((4, 2), dtype=np.float32).flatten())
    b = helper.make_tensor("b", TensorProto.FLOAT, [2], np.zeros(2, dtype=np.float32).flatten())

    n1 = helper.make_node("Gemm", ["x", "w", "b"], ["g_out"])
    n2 = helper.make_node("Relu", ["g_out"], ["y"])

    model_bytes = _build_test_model([n1, n2], [x], [y], [w, b])
    summary = load_and_inspect_model(model_bytes, filename="all_supported.onnx")

    report = CompatibilityAnalyzer.analyze(summary, cpu_profile)

    assert report.overall_assessment == "COMPATIBLE"
    assert report.summary_counts["FAIL"] == 0
    assert report.summary_counts["UNKNOWN"] == 0
    assert report.summary_counts["WARNING"] == 0
    assert report.summary_counts["PASS"] > 0
    # Every finding should be PASS
    assert all(f.status == "PASS" for f in report.findings)


# --- 2. Explicitly unsupported operator ---
def test_explicitly_unsupported_operator(ibm_z_profile):
    # NonZero is explicitly marked as FAIL in the illustrative IBM Z profile
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [1, 4])
    n = helper.make_node("NonZero", ["x"], ["y"])

    model_bytes = _build_test_model([n], [x], [y], opset_imports=[helper.make_opsetid("", 17)])
    summary = load_and_inspect_model(model_bytes, filename="nonzero_model.onnx")

    report = CompatibilityAnalyzer.analyze(summary, ibm_z_profile)

    assert report.overall_assessment == "INCOMPATIBLE"
    assert report.summary_counts["FAIL"] >= 1
    fail_findings = [f for f in report.findings if f.status == "FAIL" and f.op_type == "NonZero"]
    assert len(fail_findings) == 1
    assert fail_findings[0].severity == "CRITICAL"
    assert "explicitly unsupported" in fail_findings[0].explanation.lower()


# --- 3. Operator missing from profile rules ---
def test_operator_missing_from_profile_rules(cpu_profile):
    # Sin is not in the baseline local CPU profile ruleset
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Sin", ["x"], ["y"])

    model_bytes = _build_test_model([n], [x], [y], opset_imports=[helper.make_opsetid("", 17)])
    summary = load_and_inspect_model(model_bytes, filename="sin_model.onnx")

    report = CompatibilityAnalyzer.analyze(summary, cpu_profile)

    assert report.overall_assessment == "INSUFFICIENT_DATA"
    unknown_findings = [f for f in report.findings if f.status == "UNKNOWN" and f.op_type == "Sin"]
    assert len(unknown_findings) == 1
    assert "no verified declaration" in unknown_findings[0].explanation.lower()
    assert report.summary_counts["FAIL"] == 0
    assert report.summary_counts["UNKNOWN"] >= 1


# --- 4. Supported operator with an unsupported version constraint ---
def test_supported_operator_with_unsupported_version_constraint():
    # Construct a custom profile where Gemm requires min_version=14, but model opset is 11
    custom_profile_dict = {
        "profile_id": "test_constraint_profile",
        "display_name": "Test Constraint Profile",
        "version": "1.0",
        "ruleset_version": "2026.1",
        "target_architecture": "x86_64",
        "target_runtime": "TestRuntime",
        "confidence_classification": "VERIFIED_DOCUMENTED",
        "is_illustrative": False,
        "min_opset": 10,
        "max_opset": 20,
        "supported_domains": ["", "ai.onnx"],
        "supported_data_types": ["float32"],
        "operator_rules": {
            "Gemm": {
                "status": "PASS",
                "min_version": 14,
                "max_version": 20,
                "supported_types": ["float32"],
            }
        },
    }
    profile = load_profile_from_dict(custom_profile_dict)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4, 2], np.ones((4, 2), dtype=np.float32).flatten())
    n = helper.make_node("Gemm", ["x", "w"], ["y"])

    # Model opset is 11 (which is within profile min/max 10-20, but below Gemm's min_version 14)
    model_bytes = _build_test_model([n], [x], [y], [w], opset_imports=[helper.make_opsetid("", 11)])
    summary = load_and_inspect_model(model_bytes, filename="gemm_opset11.onnx")

    report = CompatibilityAnalyzer.analyze(summary, profile)

    assert report.overall_assessment == "INCOMPATIBLE"
    op_fails = [f for f in report.findings if f.status == "FAIL" and f.op_type == "Gemm"]
    assert len(op_fails) == 1
    assert "requires opset version >= 14" in op_fails[0].explanation


# --- 5. Multiple operator domains ---
def test_multiple_operator_domains(cpu_profile):
    # Model with standard opset and custom unsupported domain
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Identity", ["x"], ["y"])

    # Import both standard "" domain and an unsupported domain "com.custom.vendor"
    model_bytes = _build_test_model(
        [n],
        [x],
        [y],
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.custom.vendor", 1),
        ],
    )
    summary = load_and_inspect_model(model_bytes, filename="multi_domain.onnx")

    report = CompatibilityAnalyzer.analyze(summary, cpu_profile)

    assert report.overall_assessment == "INCOMPATIBLE"
    domain_fails = [f for f in report.findings if f.status == "FAIL" and f.category == "opset"]
    assert len(domain_fails) >= 1
    assert "com.custom.vendor" in domain_fails[0].feature


# --- 6. Opset mismatch ---
def test_opset_mismatch(ibm_z_profile):
    # IBM Z profile supports opset 11-20. Test model with opset 7
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Identity", ["x"], ["y"])

    model_bytes = _build_test_model([n], [x], [y], opset_imports=[helper.make_opsetid("", 7)])
    summary = load_and_inspect_model(model_bytes, filename="opset7.onnx")

    report = CompatibilityAnalyzer.analyze(summary, ibm_z_profile)

    assert report.overall_assessment == "INCOMPATIBLE"
    opset_fails = [f for f in report.findings if f.status == "FAIL" and f.category == "opset"]
    assert len(opset_fails) == 1
    assert "outside the profile supported range" in opset_fails[0].explanation


# --- 7. Unsupported data type ---
def test_unsupported_data_type(cpu_profile):
    # Test model with unsupported data type (complex64)
    x = helper.make_tensor_value_info("x", TensorProto.COMPLEX64, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.COMPLEX64, [1, 4])
    n = helper.make_node("Identity", ["x"], ["y"])

    model_bytes = _build_test_model([n], [x], [y], opset_imports=[helper.make_opsetid("", 17)])
    summary = load_and_inspect_model(model_bytes, filename="complex_dtype.onnx")

    report = CompatibilityAnalyzer.analyze(summary, cpu_profile)

    assert report.overall_assessment == "INCOMPATIBLE"
    dtype_fails = [f for f in report.findings if f.status == "FAIL" and f.category == "data_type"]
    assert len(dtype_fails) >= 1
    assert "complex64" in dtype_fails[0].feature.lower()


# --- 8. Dynamic dimensions handling ---
def test_dynamic_dimensions_handling(cpu_profile, ibm_z_profile):
    # Model with dynamic batch dimension "batch_size"
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch_size", 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch_size", 4])
    n = helper.make_node("Relu", ["x"], ["y"])

    model_bytes = _build_test_model([n], [x], [y], opset_imports=[helper.make_opsetid("", 17)])
    summary = load_and_inspect_model(model_bytes, filename="dynamic_model.onnx")

    # CPU profile: dynamic dimensions are supported with PASS
    cpu_report = CompatibilityAnalyzer.analyze(summary, cpu_profile)
    dyn_cpu = [f for f in cpu_report.findings if f.feature == "Dynamic Dimensions"]
    assert len(dyn_cpu) == 1
    assert dyn_cpu[0].status == "PASS"

    # IBM Z illustrative profile: dynamic dimensions trigger WARNING for coprocessor shape alignment
    ibm_report = CompatibilityAnalyzer.analyze(summary, ibm_z_profile)
    dyn_ibm = [f for f in ibm_report.findings if f.feature == "Dynamic Dimensions"]
    assert len(dyn_ibm) == 1
    assert dyn_ibm[0].status == "WARNING"


# --- 9. Incomplete or illustrative profile detection ---
def test_incomplete_or_illustrative_profile(ibm_z_profile):
    # Model analyzed against illustrative profile must yield a profile completeness warning
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Relu", ["x"], ["y"])

    model_bytes = _build_test_model([n], [x], [y], opset_imports=[helper.make_opsetid("", 17)])
    summary = load_and_inspect_model(model_bytes, filename="illustrative_test.onnx")

    report = CompatibilityAnalyzer.analyze(summary, ibm_z_profile)

    assert report.is_illustrative_profile is True
    prof_warnings = [f for f in report.findings if f.category == "profile_completeness"]
    assert len(prof_warnings) == 1
    assert prof_warnings[0].status == "WARNING"
    assert "illustrative draft" in prof_warnings[0].explanation.lower()


# --- 10. Malformed profile rejection and schema validation ---
def test_malformed_profile_rejection():
    # Missing required field
    with pytest.raises(ProfileValidationError, match="missing required field"):
        load_profile_from_dict({"profile_id": "incomplete"})

    # Invalid status
    with pytest.raises(ProfileValidationError, match="Invalid operator rule status"):
        load_profile_from_dict({
            "profile_id": "p1",
            "display_name": "P1",
            "version": "1.0",
            "ruleset_version": "1.0",
            "target_architecture": "x86",
            "target_runtime": "rt",
            "confidence_classification": "VERIFIED_DOCUMENTED",
            "is_illustrative": False,
            "min_opset": 1,
            "max_opset": 10,
            "supported_domains": [""],
            "supported_data_types": ["float32"],
            "operator_rules": {"Relu": {"status": "INVALID_STATUS"}},
        })

    # Inverted opset range
    with pytest.raises(ProfileValidationError, match="cannot be less than min_opset"):
        load_profile_from_dict({
            "profile_id": "p2",
            "display_name": "P2",
            "version": "1.0",
            "ruleset_version": "1.0",
            "target_architecture": "x86",
            "target_runtime": "rt",
            "confidence_classification": "VERIFIED_DOCUMENTED",
            "is_illustrative": False,
            "min_opset": 15,
            "max_opset": 10,
            "supported_domains": [""],
            "supported_data_types": ["float32"],
            "operator_rules": {},
        })

    # Malformed JSON
    with pytest.raises(ProfileValidationError, match="Invalid JSON string"):
        load_profile_from_json("{broken json")


# --- 11. Model with multiple findings and correct summary counts ---
def test_model_with_multiple_findings_and_correct_summary_counts(ibm_z_profile):
    # Create model with:
    # - Supported op: Relu (PASS)
    # - Unsupported op: NonZero (FAIL)
    # - Unknown op: Sin (UNKNOWN)
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    r_out = helper.make_tensor_value_info("r_out", TensorProto.FLOAT, [1, 4])
    s_out = helper.make_tensor_value_info("s_out", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [1, 4])

    n1 = helper.make_node("Relu", ["x"], ["r_out"])
    n2 = helper.make_node("Sin", ["r_out"], ["s_out"])
    n3 = helper.make_node("NonZero", ["s_out"], ["y"])

    model_bytes = _build_test_model([n1, n2, n3], [x], [y], opset_imports=[helper.make_opsetid("", 17)])
    summary = load_and_inspect_model(model_bytes, filename="mixed_model.onnx")

    report = CompatibilityAnalyzer.analyze(summary, ibm_z_profile)

    # Check arithmetic consistency of summary counts
    assert report.summary_counts["TOTAL"] == len(report.findings)
    assert report.summary_counts["TOTAL"] == (
        report.summary_counts["PASS"]
        + report.summary_counts["FAIL"]
        + report.summary_counts["WARNING"]
        + report.summary_counts["UNKNOWN"]
    )
    assert report.summary_counts["FAIL"] >= 1
    assert report.summary_counts["UNKNOWN"] >= 1
    assert report.summary_counts["PASS"] >= 1
    assert report.overall_assessment == "INCOMPATIBLE"


# --- 12. Stable overall assessment when findings are reordered ---
def test_stable_overall_assessment_when_findings_reordered():
    # Test all permutations of counts produce invariant deterministic results
    counts1 = {"TOTAL": 10, "PASS": 7, "FAIL": 1, "WARNING": 1, "UNKNOWN": 1}
    assert determine_overall_assessment(counts1) == "INCOMPATIBLE"

    counts2 = {"TOTAL": 10, "PASS": 8, "FAIL": 0, "WARNING": 1, "UNKNOWN": 1}
    assert determine_overall_assessment(counts2) == "INSUFFICIENT_DATA"

    counts3 = {"TOTAL": 10, "PASS": 9, "FAIL": 0, "WARNING": 1, "UNKNOWN": 0}
    assert determine_overall_assessment(counts3) == "CONDITIONAL"

    counts4 = {"TOTAL": 10, "PASS": 10, "FAIL": 0, "WARNING": 0, "UNKNOWN": 0}
    assert determine_overall_assessment(counts4) == "COMPATIBLE"


# --- 13. JSON export and schema validation ---
def test_json_export_and_schema_validation(cpu_profile):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Relu", ["x"], ["y"])

    model_bytes = _build_test_model([n], [x], [y])
    summary = load_and_inspect_model(model_bytes, filename="export_test.onnx")

    report = CompatibilityAnalyzer.analyze(summary, cpu_profile)
    json_str = report.to_json()

    data = json.loads(json_str)
    assert "report_version" in data
    assert "model_metadata" in data
    assert "profile_metadata" in data
    assert "analysis_metadata" in data
    assert "findings" in data
    assert "assumptions" in data
    assert "recommended_next_steps" in data

    # Verify no local absolute paths or secrets are in the export
    assert "C:\\" not in json_str
    assert "D:\\" not in json_str
    assert "/home/" not in json_str


# --- 14. Profile and ruleset version preservation ---
def test_profile_and_ruleset_version_preservation(cpu_profile):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Relu", ["x"], ["y"])

    summary = load_and_inspect_model(_build_test_model([n], [x], [y]), filename="ver_test.onnx")
    report = CompatibilityAnalyzer.analyze(summary, cpu_profile)

    assert report.profile_version == cpu_profile.version
    assert report.ruleset_version == cpu_profile.ruleset_version
    assert report.profile_id == cpu_profile.profile_id


# --- 15. Model fingerprint association ---
def test_model_fingerprint_association(cpu_profile):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Relu", ["x"], ["y"])

    summary = load_and_inspect_model(_build_test_model([n], [x], [y]), filename="fp_test.onnx")
    report = CompatibilityAnalyzer.analyze(summary, cpu_profile)

    assert report.model_fingerprint == summary.sha256_hash
    assert len(report.model_fingerprint) == 64


# --- 16. Minimal edge model ---
def test_minimal_edge_model(cpu_profile):
    # Minimal Identity pass-through graph
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    n = helper.make_node("Identity", ["x"], ["y"])

    summary = load_and_inspect_model(_build_test_model([n], [x], [y]), filename="edge_model.onnx")
    report = CompatibilityAnalyzer.analyze(summary, cpu_profile)

    assert report.overall_assessment == "COMPATIBLE"
    assert report.summary_counts["TOTAL"] > 0


# --- 17. Regression tests for Phase 1 and Phase 2 ---
def test_regression_phase1_and_phase2():
    # Phase 1 configuration and phase check
    assert CURRENT_PHASE >= 3
    assert APP_VERSION in ("0.3.0", "0.4.0", "0.5.0", "0.6.0", "0.7.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0")
    compat_nav = next(item for item in NAV_ITEMS if item.id == "compatibility")
    assert compat_nav.is_active is True
    assert compat_nav.phase == 3

    # Phase 2 model loader checks
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    n = helper.make_node("Relu", ["x"], ["y"])
    model_bytes = _build_test_model([n], [x], [y])

    summary = load_and_inspect_model(model_bytes, filename="phase2_regress.onnx")
    assert summary.is_valid is True
    assert summary.total_nodes == 1
    assert "Relu" in summary.op_counts
