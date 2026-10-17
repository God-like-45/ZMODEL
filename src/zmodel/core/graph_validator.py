"""Layered correctness and numerical validation engine for ONNX model transformations.

Validates:
1. Structural integrity via onnx.checker, graph references, opset rules.
2. Input/output interface compatibility (names, types, dimensions).
3. Deterministic numerical equivalence under configurable atol/rtol tolerances.
"""

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

logger = logging.getLogger(__name__)

# Default numerical tolerances
DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-3
DEFAULT_TEST_SEED = 42


class ValidationStatus(str, Enum):
    """Validation outcome status."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_VERIFIED = "NOT_VERIFIED"


@dataclass
class OutputComparisonMetric:
    """Numerical comparison statistics for a single graph output tensor."""

    output_name: str
    shape: List[int]
    dtype: str
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    passed: bool
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Comprehensive validation report encompassing structural, interface, and numerical checks."""

    structural_status: ValidationStatus
    structural_errors: List[str] = field(default_factory=list)
    structural_warnings: List[str] = field(default_factory=list)
    interface_compatible: bool = True
    interface_diffs: List[str] = field(default_factory=list)
    numerical_status: ValidationStatus = ValidationStatus.NOT_VERIFIED
    numerical_details: str = ""
    metrics_per_output: List[OutputComparisonMetric] = field(default_factory=list)
    max_abs_error_across_outputs: Optional[float] = None
    max_rel_error_across_outputs: Optional[float] = None
    configured_atol: float = DEFAULT_ATOL
    configured_rtol: float = DEFAULT_RTOL
    test_seed: int = DEFAULT_TEST_SEED
    is_fully_verified: bool = False
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["structural_status"] = self.structural_status.value
        data["numerical_status"] = self.numerical_status.value
        data["metrics_per_output"] = [m.to_dict() for m in self.metrics_per_output]
        return data


class GraphValidator:
    """Executes layered structural, interface, and numerical validation on transformed models."""

    @classmethod
    def validate(
        cls,
        original_bytes: bytes,
        transformed_bytes: bytes,
        validate_numerical: bool = True,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        seed: int = DEFAULT_TEST_SEED,
    ) -> ValidationReport:
        """
        Run complete layered validation pipeline.
        If validate_numerical is False, executes only structural checks and marks
        numerical equivalence as NOT_VERIFIED.
        """
        # 1. Structural Validation
        struct_status, struct_errs, struct_warns, orig_proto, trans_proto = (
            cls._check_structural(original_bytes, transformed_bytes)
        )

        if struct_status == ValidationStatus.FAIL:
            return ValidationReport(
                structural_status=ValidationStatus.FAIL,
                structural_errors=struct_errs,
                structural_warnings=struct_warns,
                interface_compatible=False,
                numerical_status=ValidationStatus.NOT_VERIFIED,
                numerical_details="Numerical validation skipped due to structural validation failure.",
                is_fully_verified=False,
                rejection_reason=f"Structural check failed: {'; '.join(struct_errs)}",
            )

        # 2. Interface Compatibility Check
        interface_ok, interface_diffs = cls._check_interface(orig_proto, trans_proto)
        if not interface_ok:
            return ValidationReport(
                structural_status=ValidationStatus.FAIL,
                structural_errors=[f"Interface mismatch: {d}" for d in interface_diffs],
                structural_warnings=struct_warns,
                interface_compatible=False,
                interface_diffs=interface_diffs,
                numerical_status=ValidationStatus.NOT_VERIFIED,
                numerical_details="Numerical validation skipped due to unexpected interface change.",
                is_fully_verified=False,
                rejection_reason=f"Transformed model altered graph I/O signature: {'; '.join(interface_diffs)}",
            )

        # 3. Numerical Equivalence Validation
        if not validate_numerical:
            return ValidationReport(
                structural_status=struct_status,
                structural_errors=struct_errs,
                structural_warnings=struct_warns,
                interface_compatible=True,
                numerical_status=ValidationStatus.NOT_VERIFIED,
                numerical_details="Numerical equivalence was not requested (structural check only).",
                configured_atol=atol,
                configured_rtol=rtol,
                test_seed=seed,
                is_fully_verified=False,
            )

        return cls._check_numerical_equivalence(
            original_bytes,
            transformed_bytes,
            orig_proto,
            struct_status,
            struct_errs,
            struct_warns,
            atol,
            rtol,
            seed,
        )

    @classmethod
    def _check_structural(
        cls,
        orig_bytes: bytes,
        trans_bytes: bytes,
    ) -> Tuple[
        ValidationStatus,
        List[str],
        List[str],
        Optional[onnx.ModelProto],
        Optional[onnx.ModelProto],
    ]:
        """Verify ONNX model checker, opset imports, and graph integrity."""
        errors: List[str] = []
        warnings: List[str] = []

        try:
            orig_model = onnx.load_model_from_string(orig_bytes)
        except Exception as ex:
            return (
                ValidationStatus.FAIL,
                [f"Original model parsing failed: {str(ex)}"],
                warnings,
                None,
                None,
            )

        try:
            trans_model = onnx.load_model_from_string(trans_bytes)
        except Exception as ex:
            return (
                ValidationStatus.FAIL,
                [f"Transformed model parsing failed: {str(ex)}"],
                warnings,
                orig_model,
                None,
            )

        # Run ONNX checker
        try:
            onnx.checker.check_model(trans_model, full_check=True)
        except Exception as ex:
            errors.append(f"ONNX checker validation failed: {str(ex)}")

        # Verify graph inputs and outputs exist
        if len(trans_model.graph.input) == 0:
            errors.append("Transformed model has no graph inputs.")
        if len(trans_model.graph.output) == 0:
            errors.append("Transformed model has no graph outputs.")

        # Check for missing initializers
        init_names = {init.name for init in trans_model.graph.initializer}
        input_names = {inp.name for inp in trans_model.graph.input}
        node_outputs = set()
        for node in trans_model.graph.node:
            for out in node.output:
                if out:
                    node_outputs.add(out)

        for node in trans_model.graph.node:
            for inp in node.input:
                if inp and inp not in init_names and inp not in input_names and inp not in node_outputs:
                    # Input is not in inputs, initializers, or previous outputs
                    warnings.append(f"Node '{node.name}' has unlinked input reference '{inp}'.")

        status = ValidationStatus.FAIL if errors else (
            ValidationStatus.WARNING if warnings else ValidationStatus.PASS
        )
        return status, errors, warnings, orig_model, trans_model

    @classmethod
    def _check_interface(
        cls,
        orig_model: onnx.ModelProto,
        trans_model: onnx.ModelProto,
    ) -> Tuple[bool, List[str]]:
        """Verify that the transformed model preserves graph inputs and outputs signature."""
        diffs: List[str] = []

        orig_inputs = {inp.name: inp for inp in orig_model.graph.input}
        trans_inputs = {inp.name: inp for inp in trans_model.graph.input}

        orig_outputs = {out.name: out for out in orig_model.graph.output}
        trans_outputs = {out.name: out for out in trans_model.graph.output}

        # Initializers sometimes appear in graph.input in older opsets; ignore initializer inputs
        orig_inits = {init.name for init in orig_model.graph.initializer}
        trans_inits = {init.name for init in trans_model.graph.initializer}

        pure_orig_inputs = {k: v for k, v in orig_inputs.items() if k not in orig_inits}
        pure_trans_inputs = {k: v for k, v in trans_inputs.items() if k not in trans_inits}

        # Check input names
        if set(pure_orig_inputs.keys()) != set(pure_trans_inputs.keys()):
            diffs.append(
                f"Graph inputs differ. Original: {sorted(pure_orig_inputs.keys())}, "
                f"Transformed: {sorted(pure_trans_inputs.keys())}"
            )

        # Check output names
        if set(orig_outputs.keys()) != set(trans_outputs.keys()):
            diffs.append(
                f"Graph outputs differ. Original: {sorted(orig_outputs.keys())}, "
                f"Transformed: {sorted(trans_outputs.keys())}"
            )

        # Check types for matching inputs
        for name, orig_inp in pure_orig_inputs.items():
            if name in pure_trans_inputs:
                trans_inp = pure_trans_inputs[name]
                if orig_inp.type.tensor_type.elem_type != trans_inp.type.tensor_type.elem_type:
                    diffs.append(
                        f"Data type changed for input '{name}': original elem_type "
                        f"{orig_inp.type.tensor_type.elem_type} vs transformed {trans_inp.type.tensor_type.elem_type}"
                    )

        # Check types for matching outputs
        for name, orig_out in orig_outputs.items():
            if name in trans_outputs:
                trans_out = trans_outputs[name]
                if orig_out.type.tensor_type.elem_type != trans_out.type.tensor_type.elem_type:
                    diffs.append(
                        f"Data type changed for output '{name}': original elem_type "
                        f"{orig_out.type.tensor_type.elem_type} vs transformed {trans_out.type.tensor_type.elem_type}"
                    )

        return (len(diffs) == 0), diffs

    @classmethod
    def _generate_synthetic_input(
        cls,
        value_info: onnx.ValueInfoProto,
        rng: np.random.RandomState,
    ) -> Optional[np.ndarray]:
        """Generate a deterministic synthetic input tensor based on onnx ValueInfoProto."""
        tensor_type = value_info.type.tensor_type
        elem_type = tensor_type.elem_type

        # Extract dimensions, substituting 1 for dynamic/symbolic dimensions
        shape: List[int] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                # Symbolic dimension (e.g. 'batch_size') -> use 1 for deterministic validation
                shape.append(1)
            else:
                shape.append(1)

        if not shape:
            shape = [1]

        # Map ONNX elem_type to numpy dtype
        if elem_type == TensorProto.FLOAT:
            return rng.randn(*shape).astype(np.float32)
        elif elem_type == TensorProto.DOUBLE:
            return rng.randn(*shape).astype(np.float64)
        elif elem_type == TensorProto.INT64:
            return rng.randint(0, 10, size=shape, dtype=np.int64)
        elif elem_type == TensorProto.INT32:
            return rng.randint(0, 10, size=shape, dtype=np.int32)
        elif elem_type == TensorProto.INT8:
            return rng.randint(-5, 5, size=shape, dtype=np.int8)
        elif elem_type == TensorProto.UINT8:
            return rng.randint(0, 10, size=shape, dtype=np.uint8)
        elif elem_type == TensorProto.BOOL:
            return rng.choice([True, False], size=shape)
        else:
            return None

    @classmethod
    def _check_numerical_equivalence(
        cls,
        original_bytes: bytes,
        transformed_bytes: bytes,
        orig_model: onnx.ModelProto,
        struct_status: ValidationStatus,
        struct_errs: List[str],
        struct_warns: List[str],
        atol: float,
        rtol: float,
        seed: int,
    ) -> ValidationReport:
        """Run identical seeded inputs through original and transformed models and compare output tensors."""
        rng = np.random.RandomState(seed)

        # Identify required model inputs (excluding initializers)
        inits = {init.name for init in orig_model.graph.initializer}
        model_inputs = [inp for inp in orig_model.graph.input if inp.name not in inits]

        # Generate deterministic synthetic inputs
        feed_dict: Dict[str, np.ndarray] = {}
        for inp in model_inputs:
            arr = cls._generate_synthetic_input(inp, rng)
            if arr is None:
                return ValidationReport(
                    structural_status=struct_status,
                    structural_errors=struct_errs,
                    structural_warnings=struct_warns,
                    numerical_status=ValidationStatus.NOT_VERIFIED,
                    numerical_details=(
                        f"Unsupported tensor data type (elem_type={inp.type.tensor_type.elem_type}) "
                        f"for input '{inp.name}'. Numerical equivalence cannot be verified locally."
                    ),
                    configured_atol=atol,
                    configured_rtol=rtol,
                    test_seed=seed,
                    is_fully_verified=False,
                )
            feed_dict[inp.name] = arr

        # Execute both models
        try:
            orig_sess = ort.InferenceSession(original_bytes, providers=["CPUExecutionProvider"])
        except Exception as ex:
            return ValidationReport(
                structural_status=struct_status,
                structural_errors=struct_errs,
                structural_warnings=struct_warns,
                numerical_status=ValidationStatus.NOT_VERIFIED,
                numerical_details=f"Original model cannot execute on local CPU runtime: {str(ex)}",
                configured_atol=atol,
                configured_rtol=rtol,
                test_seed=seed,
                is_fully_verified=False,
            )

        try:
            trans_sess = ort.InferenceSession(transformed_bytes, providers=["CPUExecutionProvider"])
        except Exception as ex:
            return ValidationReport(
                structural_status=struct_status,
                structural_errors=struct_errs,
                structural_warnings=struct_warns,
                numerical_status=ValidationStatus.FAIL,
                numerical_details=f"Transformed model failed during runtime session initialization: {str(ex)}",
                configured_atol=atol,
                configured_rtol=rtol,
                test_seed=seed,
                is_fully_verified=False,
                rejection_reason=f"Transformed model runtime execution failed: {str(ex)}",
            )

        # Run inference
        try:
            orig_outputs = orig_sess.run(None, feed_dict)
            orig_out_names = [out.name for out in orig_sess.get_outputs()]
        except Exception as ex:
            return ValidationReport(
                structural_status=struct_status,
                structural_errors=struct_errs,
                structural_warnings=struct_warns,
                numerical_status=ValidationStatus.NOT_VERIFIED,
                numerical_details=f"Inference execution failed on original model: {str(ex)}",
                configured_atol=atol,
                configured_rtol=rtol,
                test_seed=seed,
                is_fully_verified=False,
            )

        try:
            trans_outputs = trans_sess.run(None, feed_dict)
            trans_out_names = [out.name for out in trans_sess.get_outputs()]
        except Exception as ex:
            return ValidationReport(
                structural_status=struct_status,
                structural_errors=struct_errs,
                structural_warnings=struct_warns,
                numerical_status=ValidationStatus.FAIL,
                numerical_details=f"Inference execution failed on transformed model: {str(ex)}",
                configured_atol=atol,
                configured_rtol=rtol,
                test_seed=seed,
                is_fully_verified=False,
                rejection_reason=f"Transformed model failed during inference: {str(ex)}",
            )

        # Compare outputs
        orig_map = dict(zip(orig_out_names, orig_outputs))
        trans_map = dict(zip(trans_out_names, trans_outputs))

        metrics: List[OutputComparisonMetric] = []
        all_passed = True
        max_abs_all: float = 0.0
        max_rel_all: float = 0.0

        for name, orig_val in orig_map.items():
            if name not in trans_map:
                all_passed = False
                metrics.append(
                    OutputComparisonMetric(
                        output_name=name,
                        shape=list(orig_val.shape),
                        dtype=str(orig_val.dtype),
                        max_absolute_error=float("inf"),
                        mean_absolute_error=float("inf"),
                        max_relative_error=float("inf"),
                        passed=False,
                        details=f"Output '{name}' missing from transformed model results.",
                    )
                )
                continue

            trans_val = trans_map[name]
            if orig_val.shape != trans_val.shape:
                all_passed = False
                metrics.append(
                    OutputComparisonMetric(
                        output_name=name,
                        shape=list(orig_val.shape),
                        dtype=str(orig_val.dtype),
                        max_absolute_error=float("inf"),
                        mean_absolute_error=float("inf"),
                        max_relative_error=float("inf"),
                        passed=False,
                        details=f"Shape mismatch: original {orig_val.shape} vs transformed {trans_val.shape}",
                    )
                )
                continue

            # Compute error statistics
            diff = np.abs(orig_val - trans_val)
            max_abs = float(np.max(diff))
            mean_abs = float(np.mean(diff))

            # Numerical stability: add epsilon to denominator
            rel_diff = diff / (np.abs(orig_val) + 1e-8)
            max_rel = float(np.max(rel_diff))

            max_abs_all = max(max_abs_all, max_abs)
            max_rel_all = max(max_rel_all, max_rel)

            passed = bool(np.allclose(orig_val, trans_val, rtol=rtol, atol=atol))
            if not passed:
                all_passed = False
                details = (
                    f"Exceeded tolerances (atol={atol}, rtol={rtol}). "
                    f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e}"
                )
            else:
                details = (
                    f"Passed within tolerances. "
                    f"max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, max_rel={max_rel:.6e}"
                )

            metrics.append(
                OutputComparisonMetric(
                    output_name=name,
                    shape=list(orig_val.shape),
                    dtype=str(orig_val.dtype),
                    max_absolute_error=round(max_abs, 8),
                    mean_absolute_error=round(mean_abs, 8),
                    max_relative_error=round(max_rel, 8),
                    passed=passed,
                    details=details,
                )
            )

        num_status = ValidationStatus.PASS if all_passed else ValidationStatus.FAIL
        is_verified = (struct_status == ValidationStatus.PASS and all_passed)
        rejection_reason = None
        if not all_passed:
            rejection_reason = (
                f"Numerical divergence: max absolute error {max_abs_all:.6e} exceeded tolerance {atol}."
            )

        num_details = (
            f"Evaluated {len(metrics)} output tensor(s) using seed={seed}. "
            f"Max absolute error: {max_abs_all:.6e}, Max relative error: {max_rel_all:.6e}. "
            f"Overall: {'PASS' if all_passed else 'FAIL'}."
        )

        return ValidationReport(
            structural_status=struct_status,
            structural_errors=struct_errs,
            structural_warnings=struct_warns,
            interface_compatible=True,
            numerical_status=num_status,
            numerical_details=num_details,
            metrics_per_output=metrics,
            max_abs_error_across_outputs=round(max_abs_all, 8),
            max_rel_error_across_outputs=round(max_rel_all, 8),
            configured_atol=atol,
            configured_rtol=rtol,
            test_seed=seed,
            is_fully_verified=is_verified,
            rejection_reason=rejection_reason,
        )
