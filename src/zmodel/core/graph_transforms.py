"""Registry and executors for graph-level ONNX transformations.

Implements real, evidence-backed model transformations:
1. ONNX static shape inference (onnx.shape_inference)
2. ONNX Runtime Basic Optimization (Constant folding, dead code removal)
3. ONNX Runtime Extended Optimization (Operator fusions: Gemm+Relu, Conv+Add)
4. ONNX Runtime All Optimizations (Layout transformations & full graph fusion)
5. Graph Simplification (Optional onnxsim tool with honest capability check)

Original models are treated as strictly immutable; each transform operates on an
isolated copy and outputs a newly fingerprinted artifact.
"""

import hashlib
import io
import logging
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import onnx
import onnxruntime as ort

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Enumeration of supported graph transformation strategies."""

    SHAPE_INFERENCE = "shape_inference"
    ORT_BASIC = "ort_basic"
    ORT_EXTENDED = "ort_extended"
    ORT_ALL = "ort_all"
    SIMPLIFICATION = "simplification"


@dataclass(frozen=True)
class TransformationDescriptor:
    """Metadata describing a graph transformation's capabilities, requirements, and safety."""

    transform_type: TransformationType
    name: str
    display_name: str
    description: str
    tool_name: str
    tool_version: str
    is_available: bool
    availability_reason: str
    safety_rating: str  # "SAFE", "TARGET_AWARE", "ENVIRONMENT_DEPENDENT"
    parameters_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transform_type": self.transform_type.value,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "is_available": self.is_available,
            "availability_reason": self.availability_reason,
            "safety_rating": self.safety_rating,
            "parameters_schema": self.parameters_schema,
        }


@dataclass
class TransformationResult:
    """Detailed outcome of executing a graph transformation."""

    transform_type: TransformationType
    success: bool
    transformed_bytes: Optional[bytes] = None
    transformed_hash: Optional[str] = None
    original_hash: str = ""
    graph_changed: bool = False
    diff_summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    tool_name: str = ""
    tool_version: str = ""
    environment_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["transform_type"] = self.transform_type.value
        # Exclude raw bytes from serializable dictionary
        data.pop("transformed_bytes", None)
        return data


def _get_onnx_version() -> str:
    return getattr(onnx, "__version__", "unknown")


def _get_ort_version() -> str:
    return getattr(ort, "__version__", "unknown")


def _is_onnxsim_available() -> Tuple[bool, str, str]:
    """Check if onnxsim is importable in the current environment."""
    try:
        import onnxsim
        ver = getattr(onnxsim, "__version__", "installed")
        return True, "onnxsim is installed and ready", ver
    except ImportError:
        return (
            False,
            "onnxsim package is not installed in the current environment. "
            "Install with 'pip install onnxsim' to enable graph simplification.",
            "not installed",
        )


def _compute_graph_diff(
    original_model: onnx.ModelProto,
    transformed_model: onnx.ModelProto,
) -> Dict[str, Any]:
    """Compute structural graph differences between original and transformed ONNX models."""
    orig_nodes = len(original_model.graph.node)
    trans_nodes = len(transformed_model.graph.node)

    orig_inits = len(original_model.graph.initializer)
    trans_inits = len(transformed_model.graph.initializer)

    orig_val_info = len(original_model.graph.value_info)
    trans_val_info = len(transformed_model.graph.value_info)

    # Op type distribution
    orig_ops: Dict[str, int] = {}
    for n in original_model.graph.node:
        orig_ops[n.op_type] = orig_ops.get(n.op_type, 0) + 1

    trans_ops: Dict[str, int] = {}
    for n in transformed_model.graph.node:
        trans_ops[n.op_type] = trans_ops.get(n.op_type, 0) + 1

    added_ops = {op: trans_ops[op] for op in trans_ops if op not in orig_ops}
    removed_ops = {op: orig_ops[op] for op in orig_ops if op not in trans_ops}

    node_delta = trans_nodes - orig_nodes
    init_delta = trans_inits - orig_inits
    val_info_delta = trans_val_info - orig_val_info

    # Graph is changed if nodes, inits, value_infos, or serialized graph bytes differ
    is_changed = (
        node_delta != 0
        or init_delta != 0
        or val_info_delta != 0
        or orig_ops != trans_ops
    )

    return {
        "nodes_before": orig_nodes,
        "nodes_after": trans_nodes,
        "node_delta": node_delta,
        "initializers_before": orig_inits,
        "initializers_after": trans_inits,
        "initializer_delta": init_delta,
        "value_infos_before": orig_val_info,
        "value_infos_after": trans_val_info,
        "value_info_delta": val_info_delta,
        "operators_before": orig_ops,
        "operators_after": trans_ops,
        "added_operators": added_ops,
        "removed_operators": removed_ops,
    }


class TransformationRegistry:
    """Registry of available model graph transformations and execution dispatch."""

    @classmethod
    def get_available_transformations(cls) -> List[TransformationDescriptor]:
        """Return all registered transformations with real-time capability checks."""
        onnx_ver = _get_onnx_version()
        ort_ver = _get_ort_version()
        sim_ok, sim_msg, sim_ver = _is_onnxsim_available()

        return [
            TransformationDescriptor(
                transform_type=TransformationType.SHAPE_INFERENCE,
                name="shape_inference",
                display_name="ONNX Static Shape & Type Inference",
                description=(
                    "Applies ONNX shape inference to propagate static shapes and types across all "
                    "intermediate graph value_info tensors. Essential for pre-deployment static sizing."
                ),
                tool_name="onnx.shape_inference",
                tool_version=onnx_ver,
                is_available=True,
                availability_reason="Native ONNX library available.",
                safety_rating="SAFE",
                parameters_schema={},
            ),
            TransformationDescriptor(
                transform_type=TransformationType.ORT_BASIC,
                name="ort_basic",
                display_name="ONNX Runtime Basic Optimization (Constant Folding)",
                description=(
                    "Evaluates constant subgraphs and performs dead-code elimination and redundant node removal. "
                    "Graph is simplified while strictly preserving mathematical behavior across all architectures."
                ),
                tool_name="onnxruntime",
                tool_version=ort_ver,
                is_available=True,
                availability_reason=f"ONNX Runtime {ort_ver} installed with ORT_ENABLE_BASIC support.",
                safety_rating="SAFE",
                parameters_schema={},
            ),
            TransformationDescriptor(
                transform_type=TransformationType.ORT_EXTENDED,
                name="ort_extended",
                display_name="ONNX Runtime Extended Optimization (Operator Fusion)",
                description=(
                    "Applies aggressive operator fusions (e.g. Gemm+Relu, Conv+Add+Relu, MatMul+Add) and "
                    "memory layout optimizations supported by the host runtime execution provider."
                ),
                tool_name="onnxruntime",
                tool_version=ort_ver,
                is_available=True,
                availability_reason=f"ONNX Runtime {ort_ver} installed with ORT_ENABLE_EXTENDED support.",
                safety_rating="ENVIRONMENT_DEPENDENT",
                parameters_schema={},
            ),
            TransformationDescriptor(
                transform_type=TransformationType.ORT_ALL,
                name="ort_all",
                display_name="ONNX Runtime Full Optimization (Layout & All Levels)",
                description=(
                    "Enables all ONNX Runtime optimizations, including hardware-specific memory layout "
                    "transformations and composite fusions."
                ),
                tool_name="onnxruntime",
                tool_version=ort_ver,
                is_available=True,
                availability_reason=f"ONNX Runtime {ort_ver} installed with ORT_ENABLE_ALL support.",
                safety_rating="ENVIRONMENT_DEPENDENT",
                parameters_schema={},
            ),
            TransformationDescriptor(
                transform_type=TransformationType.SIMPLIFICATION,
                name="simplification",
                display_name="ONNX Simplifier (onnxsim)",
                description=(
                    "Optimizes ONNX graphs by evaluating constant subgraphs, optimizing tensor shapes, "
                    "and cleaning redundant reshape/identity chains."
                ),
                tool_name="onnxsim",
                tool_version=sim_ver,
                is_available=sim_ok,
                availability_reason=sim_msg,
                safety_rating="TARGET_AWARE",
                parameters_schema={},
            ),
        ]

    @classmethod
    def get_descriptor(cls, transform_type: TransformationType) -> Optional[TransformationDescriptor]:
        """Look up descriptor by transformation type."""
        for desc in cls.get_available_transformations():
            if desc.transform_type == transform_type:
                return desc
        return None

    @classmethod
    def execute_transformation(
        cls,
        transform_type: TransformationType,
        model_bytes: bytes,
    ) -> TransformationResult:
        """
        Execute specified transformation on an isolated copy of the model.
        The input model_bytes is treated as immutable.
        """
        original_hash = hashlib.sha256(model_bytes).hexdigest()
        env_meta = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "onnx_version": _get_onnx_version(),
            "ort_version": _get_ort_version(),
        }

        # Validate input model
        try:
            original_model = onnx.load_model_from_string(model_bytes)
        except Exception as ex:
            return TransformationResult(
                transform_type=transform_type,
                success=False,
                original_hash=original_hash,
                error_message=f"Failed to parse source ONNX model: {str(ex)}",
                environment_metadata=env_meta,
            )

        start_time = time.perf_counter()

        if transform_type == TransformationType.SHAPE_INFERENCE:
            return cls._execute_shape_inference(original_model, original_hash, start_time, env_meta)
        elif transform_type in (
            TransformationType.ORT_BASIC,
            TransformationType.ORT_EXTENDED,
            TransformationType.ORT_ALL,
        ):
            return cls._execute_ort_optimization(
                transform_type, original_model, original_hash, start_time, env_meta
            )
        elif transform_type == TransformationType.SIMPLIFICATION:
            return cls._execute_simplification(original_model, original_hash, start_time, env_meta)
        else:
            return TransformationResult(
                transform_type=transform_type,
                success=False,
                original_hash=original_hash,
                error_message=f"Unknown or unsupported transformation type: '{transform_type}'",
                environment_metadata=env_meta,
            )

    @classmethod
    def _execute_shape_inference(
        cls,
        original_model: onnx.ModelProto,
        original_hash: str,
        start_time: float,
        env_meta: Dict[str, Any],
    ) -> TransformationResult:
        """Execute static shape and type inference."""
        tool_name = "onnx.shape_inference"
        tool_ver = _get_onnx_version()
        warnings: List[str] = []

        try:
            # Make isolated copy of model proto
            model_copy = onnx.ModelProto()
            model_copy.CopyFrom(original_model)

            inferred_model = onnx.shape_inference.infer_shapes(model_copy, check_type=True, strict_mode=False)

            diff = _compute_graph_diff(original_model, inferred_model)
            out_bytes = inferred_model.SerializeToString()
            out_hash = hashlib.sha256(out_bytes).hexdigest()
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            graph_changed = (out_hash != original_hash)
            if not graph_changed:
                warnings.append(
                    "Shape inference completed successfully, but all shapes were already fully inferred. "
                    "No additional value_info tensors were added."
                )

            return TransformationResult(
                transform_type=TransformationType.SHAPE_INFERENCE,
                success=True,
                transformed_bytes=out_bytes,
                transformed_hash=out_hash,
                original_hash=original_hash,
                graph_changed=graph_changed,
                diff_summary=diff,
                warnings=warnings,
                duration_ms=round(duration_ms, 2),
                tool_name=tool_name,
                tool_version=tool_ver,
                environment_metadata=env_meta,
            )
        except Exception as ex:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return TransformationResult(
                transform_type=TransformationType.SHAPE_INFERENCE,
                success=False,
                original_hash=original_hash,
                error_message=f"Shape inference error: {str(ex)}",
                duration_ms=round(duration_ms, 2),
                tool_name=tool_name,
                tool_version=tool_ver,
                environment_metadata=env_meta,
            )

    @classmethod
    def _execute_ort_optimization(
        cls,
        transform_type: TransformationType,
        original_model: onnx.ModelProto,
        original_hash: str,
        start_time: float,
        env_meta: Dict[str, Any],
    ) -> TransformationResult:
        """Execute ONNX Runtime graph optimization via optimized_model_filepath serialization."""
        tool_name = "onnxruntime"
        tool_ver = _get_ort_version()
        warnings: List[str] = []

        level_map = {
            TransformationType.ORT_BASIC: ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            TransformationType.ORT_EXTENDED: ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            TransformationType.ORT_ALL: ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        target_level = level_map[transform_type]

        if transform_type in (TransformationType.ORT_EXTENDED, TransformationType.ORT_ALL):
            warnings.append(
                f"Optimization level '{transform_type.value}' produces host-tailored operator fusions "
                f"under CPUExecutionProvider. Ensure compatibility reassessment before deploying to target s390x."
            )

        # Temporary files for input model and optimized output
        with tempfile.TemporaryDirectory(prefix="ort_opt_") as tmp_dir:
            in_path = os.path.join(tmp_dir, "input.onnx")
            out_path = os.path.join(tmp_dir, "optimized.onnx")

            try:
                # Write original model bytes to temp file
                with open(in_path, "wb") as f:
                    f.write(original_model.SerializeToString())

                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = target_level
                sess_opts.optimized_model_filepath = out_path

                # Initializing InferenceSession triggers optimization and writes optimized_model_filepath
                _ = ort.InferenceSession(
                    in_path,
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )

                if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                    # Some models with no optimizable patterns might not write out_path
                    # In that case, the graph is mathematically identical/unchanged
                    out_bytes = original_model.SerializeToString()
                    out_hash = hashlib.sha256(out_bytes).hexdigest()
                    diff = _compute_graph_diff(original_model, original_model)
                    warnings.append(
                        f"ONNX Runtime {transform_type.value} found no optimizable graph structures. "
                        "The model was not modified."
                    )
                    duration_ms = (time.perf_counter() - start_time) * 1000.0
                    return TransformationResult(
                        transform_type=transform_type,
                        success=True,
                        transformed_bytes=out_bytes,
                        transformed_hash=out_hash,
                        original_hash=original_hash,
                        graph_changed=False,
                        diff_summary=diff,
                        warnings=warnings,
                        duration_ms=round(duration_ms, 2),
                        tool_name=tool_name,
                        tool_version=tool_ver,
                        environment_metadata=env_meta,
                    )

                with open(out_path, "rb") as f:
                    out_bytes = f.read()

                out_hash = hashlib.sha256(out_bytes).hexdigest()
                optimized_model = onnx.load_model_from_string(out_bytes)
                diff = _compute_graph_diff(original_model, optimized_model)
                graph_changed = (out_hash != original_hash)

                if not graph_changed:
                    warnings.append(
                        f"Optimization level '{transform_type.value}' completed without changing node topology."
                    )

                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return TransformationResult(
                    transform_type=transform_type,
                    success=True,
                    transformed_bytes=out_bytes,
                    transformed_hash=out_hash,
                    original_hash=original_hash,
                    graph_changed=graph_changed,
                    diff_summary=diff,
                    warnings=warnings,
                    duration_ms=round(duration_ms, 2),
                    tool_name=tool_name,
                    tool_version=tool_ver,
                    environment_metadata=env_meta,
                )

            except Exception as ex:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                return TransformationResult(
                    transform_type=transform_type,
                    success=False,
                    original_hash=original_hash,
                    error_message=f"ONNX Runtime optimization failed ({transform_type.value}): {str(ex)}",
                    duration_ms=round(duration_ms, 2),
                    tool_name=tool_name,
                    tool_version=tool_ver,
                    environment_metadata=env_meta,
                )

    @classmethod
    def _execute_simplification(
        cls,
        original_model: onnx.ModelProto,
        original_hash: str,
        start_time: float,
        env_meta: Dict[str, Any],
    ) -> TransformationResult:
        """Execute onnxsim simplification if available."""
        tool_name = "onnxsim"
        avail, msg, tool_ver = _is_onnxsim_available()
        if not avail:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return TransformationResult(
                transform_type=TransformationType.SIMPLIFICATION,
                success=False,
                original_hash=original_hash,
                error_message=f"Transformation unavailable: {msg}",
                duration_ms=round(duration_ms, 2),
                tool_name=tool_name,
                tool_version=tool_ver,
                environment_metadata=env_meta,
            )

        try:
            import onnxsim

            model_copy = onnx.ModelProto()
            model_copy.CopyFrom(original_model)

            simplified_model, check = onnxsim.simplify(model_copy)
            if not check:
                raise ValueError("onnxsim check failed: simplified model does not validate.")

            out_bytes = simplified_model.SerializeToString()
            out_hash = hashlib.sha256(out_bytes).hexdigest()
            diff = _compute_graph_diff(original_model, simplified_model)
            graph_changed = (out_hash != original_hash)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            warnings = []
            if not graph_changed:
                warnings.append("onnxsim found no redundant nodes or constant subgraphs to simplify.")

            return TransformationResult(
                transform_type=TransformationType.SIMPLIFICATION,
                success=True,
                transformed_bytes=out_bytes,
                transformed_hash=out_hash,
                original_hash=original_hash,
                graph_changed=graph_changed,
                diff_summary=diff,
                warnings=warnings,
                duration_ms=round(duration_ms, 2),
                tool_name=tool_name,
                tool_version=tool_ver,
                environment_metadata=env_meta,
            )
        except Exception as ex:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return TransformationResult(
                transform_type=TransformationType.SIMPLIFICATION,
                success=False,
                original_hash=original_hash,
                error_message=f"onnxsim execution failed: {str(ex)}",
                duration_ms=round(duration_ms, 2),
                tool_name=tool_name,
                tool_version=tool_ver,
                environment_metadata=env_meta,
            )
