"""ONNX model parsing, schema inspection, and structural graph analysis engine."""

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import TensorProto, checker, shape_inference
from zmodel.core.config import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_FILE_EXTENSIONS,
)

# Mapping of ONNX TensorProto.DataType integer enum to human-readable strings
TENSOR_TYPE_MAP: Dict[int, str] = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.STRING: "string",
    TensorProto.BOOL: "bool",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.UINT32: "uint32",
    TensorProto.UINT64: "uint64",
    TensorProto.COMPLEX64: "complex64",
    TensorProto.COMPLEX128: "complex128",
    TensorProto.BFLOAT16: "bfloat16",
}


@dataclass
class InspectionWarning:
    """Grounded warning or advisory generated from actual graph inspection."""
    code: str
    level: str  # "advisory", "warning", "info"
    title: str
    message: str
    impact: str


@dataclass
class TensorInfo:
    """Detailed tensor signature for inputs, outputs, and intermediate value_info."""
    name: str
    data_type: str
    shape: List[Union[int, str]]
    rank: Optional[int]
    is_dynamic: bool
    dynamic_dim_names: List[str]
    dimension_description: str


@dataclass
class NodeInfo:
    """Detailed structural node information including upstream and downstream connections."""
    index: int
    name: str
    op_type: str
    domain: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)


@dataclass
class OpsetInfo:
    """Opset domain and version import declaration."""
    domain: str
    version: int


@dataclass
class WeightInfo:
    """Model parameter / initializer tensor metadata."""
    name: str
    data_type: str
    shape: List[int]
    num_elements: int
    size_bytes: int
    has_external_data: bool = False


@dataclass
class ModelMetadata:
    """Model-level identification and provenance metadata."""
    producer_name: str
    producer_version: str
    model_version: int
    ir_version: int
    doc_string: str
    domain: str
    custom_metadata: Dict[str, str]


@dataclass
class ModelSummary:
    """Comprehensive verified summary of an inspected ONNX computational graph."""
    filename: str
    file_size_bytes: int
    sha256_hash: str
    metadata: ModelMetadata
    opsets: List[OpsetInfo]
    inputs: List[TensorInfo]
    outputs: List[TensorInfo]
    nodes: List[NodeInfo]
    weights: List[WeightInfo]
    op_counts: Dict[str, int]
    distinct_operator_count: int
    total_nodes: int
    total_parameters: int
    total_weight_bytes: int
    has_external_data: bool
    external_data_tensors: List[str]
    warnings: List[InspectionWarning]
    is_valid: bool
    parsing_status: str
    inference_status: str
    compatibility_status: str


def validate_file_constraints(filename: str, file_size_bytes: int) -> None:
    """Validate filename extension and file size limits."""
    # Check extension
    is_valid_ext = any(filename.lower().endswith(ext) for ext in SUPPORTED_FILE_EXTENSIONS)
    if not is_valid_ext:
        allowed = ", ".join(SUPPORTED_FILE_EXTENSIONS)
        raise ValueError(
            f"Unsupported file format: '{filename}'. Z-ModelPilot currently accepts only {allowed} files."
        )

    # Check empty file
    if file_size_bytes <= 0:
        raise ValueError(f"Uploaded file '{filename}' is empty (0 bytes).")

    # Check size limit
    if file_size_bytes > MAX_UPLOAD_SIZE_BYTES:
        size_mb = file_size_bytes / (1024 * 1024)
        raise ValueError(
            f"File size ({size_mb:.1f} MB) exceeds configured maximum limit of {MAX_UPLOAD_SIZE_MB} MB."
        )


def _get_tensor_type_str(elem_type: int) -> str:
    return TENSOR_TYPE_MAP.get(elem_type, f"type_{elem_type}")


def _extract_shape_and_rank(
    type_proto: Any,
) -> Tuple[List[Union[int, str]], Optional[int], bool, List[str]]:
    """Extract dimensions, rank, dynamic status, and symbolic dimension names.
    
    Returns (shape_list, rank, is_dynamic, dynamic_dim_names).
    """
    dims: List[Union[int, str]] = []
    dynamic_names: List[str] = []
    is_dynamic = False

    if not type_proto.HasField("tensor_type"):
        return dims, None, False, []

    tensor_type = type_proto.tensor_type
    if not tensor_type.HasField("shape"):
        # Unranked tensor (shape unknown)
        return dims, None, True, ["unranked"]

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            param_name = dim.dim_param or "?"
            dims.append(param_name)
            dynamic_names.append(param_name)
            is_dynamic = True
        else:
            dims.append("?")
            dynamic_names.append("?")
            is_dynamic = True

    rank = len(dims)
    return dims, rank, is_dynamic, dynamic_names


def _parse_value_info(vi: onnx.ValueInfoProto) -> TensorInfo:
    """Parse a ValueInfoProto into rich TensorInfo."""
    elem_type = vi.type.tensor_type.elem_type if vi.type.HasField("tensor_type") else 0
    type_str = _get_tensor_type_str(elem_type)
    shape, rank, is_dynamic, dynamic_dims = _extract_shape_and_rank(vi.type)

    if rank is None:
        dim_desc = "Unranked tensor (shape undetermined)"
    elif rank == 0:
        dim_desc = "Scalar tensor (Rank 0)"
    elif is_dynamic:
        dyn_str = ", ".join(dynamic_dims)
        dim_desc = f"Rank {rank} tensor with dynamic axis ({dyn_str})"
    else:
        dim_desc = f"Rank {rank} fixed-shape tensor"

    return TensorInfo(
        name=vi.name,
        data_type=type_str,
        shape=shape,
        rank=rank,
        is_dynamic=is_dynamic,
        dynamic_dim_names=dynamic_dims,
        dimension_description=dim_desc,
    )


def _check_external_data(init: onnx.TensorProto) -> bool:
    """Determine whether an initializer references external data."""
    # In ONNX protobuf, data_location == 1 represents EXTERNAL
    if getattr(init, "data_location", None) == 1:
        return True
    if hasattr(init, "external_data") and len(init.external_data) > 0:
        return True
    return False


def _generate_inspection_warnings(
    metadata: ModelMetadata,
    inputs: List[TensorInfo],
    outputs: List[TensorInfo],
    nodes: List[NodeInfo],
    weights: List[WeightInfo],
    op_counts: Dict[str, int],
    has_external_data: bool,
    external_data_tensors: List[str],
) -> List[InspectionWarning]:
    """Generate grounded warnings and advisories based on actual graph inspection."""
    warnings: List[InspectionWarning] = []

    # 1. Missing optional metadata
    missing_meta_fields = []
    if metadata.producer_name == "Not available":
        missing_meta_fields.append("producer_name")
    if metadata.producer_version == "Not available":
        missing_meta_fields.append("producer_version")
    if metadata.doc_string == "Not available":
        missing_meta_fields.append("doc_string")

    if missing_meta_fields:
        warnings.append(
            InspectionWarning(
                code="WARN_MISSING_METADATA",
                level="info",
                title="Optional Metadata Absent",
                message=f"Model does not declare optional fields: {', '.join(missing_meta_fields)}.",
                impact="Does not prevent parsing or deployment, but reduces model provenance and documentation clarity.",
            )
        )

    # 2. Dynamic dimensions in inputs
    dyn_inputs = [inp for inp in inputs if inp.is_dynamic]
    if dyn_inputs:
        inp_details = ", ".join([f"{inp.name} ({inp.shape})" for inp in dyn_inputs])
        warnings.append(
            InspectionWarning(
                code="WARN_DYNAMIC_INPUTS",
                level="advisory",
                title="Dynamic Dimensions in Graph Inputs",
                message=f"Inputs with symbolic/variable dimensions: {inp_details}.",
                impact="Dynamic dimensions allow variable batch sizes or sequence lengths at runtime. However, some execution runtimes or hardware accelerators require static shape freezing.",
            )
        )

    # 3. External tensor data
    if has_external_data:
        warnings.append(
            InspectionWarning(
                code="WARN_EXTERNAL_DATA",
                level="warning",
                title="External Tensor Data References",
                message=f"{len(external_data_tensors)} initializers reference external data files ({', '.join(external_data_tensors[:5])}).",
                impact="The ONNX file does not contain all weight payloads internally. Auxiliary data files must accompany this model during inference.",
            )
        )

    # 4. Custom or non-standard operator domains
    custom_domains = set()
    for n in nodes:
        if n.domain and n.domain not in ("", "ai.onnx", "ai.onnx.ml"):
            custom_domains.add(n.domain)
    if custom_domains:
        warnings.append(
            InspectionWarning(
                code="WARN_CUSTOM_DOMAINS",
                level="warning",
                title="Non-Standard Operator Domains Detected",
                message=f"Model utilizes operators from custom domains: {', '.join(custom_domains)}.",
                impact="Operators outside the default 'ai.onnx' domain require specialized execution providers or custom runtime kernel extensions.",
            )
        )

    # 5. Multiple graph outputs
    if len(outputs) > 1:
        warnings.append(
            InspectionWarning(
                code="WARN_MULTI_OUTPUT",
                level="info",
                title="Multi-Output Graph Topology",
                message=f"Model graph emits {len(outputs)} distinct outputs: {', '.join([o.name for o in outputs])}.",
                impact="Deployment pipeline must bind buffers for each output tensor.",
            )
        )

    # 6. Models with no weights (e.g. topology-only models)
    if len(weights) == 0 and len(nodes) > 0:
        warnings.append(
            InspectionWarning(
                code="WARN_ZERO_WEIGHTS",
                level="info",
                title="No Parameter Initializers",
                message="Graph does not contain embedded weight initializers.",
                impact="Model is purely an operator topology or receives all weights dynamically through graph inputs.",
            )
        )

    return warnings


def load_and_inspect_model(
    source: Union[bytes, str],
    filename: str = "model.onnx",
) -> ModelSummary:
    """Load, strictly validate, and inspect an ONNX model from bytes or filepath.
    
    Operates in-memory and raises ValueError on validation or schema failure.
    """
    if isinstance(source, bytes):
        raw_bytes = source
    else:
        with open(source, "rb") as f:
            raw_bytes = f.read()

    file_size_bytes = len(raw_bytes)
    # Validate extension and size
    validate_file_constraints(filename, file_size_bytes)

    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Step 1: In-memory protobuf parsing
    try:
        model = onnx.load_model_from_string(raw_bytes)
    except Exception as e:
        raise ValueError(
            f"Failed to parse ONNX protobuf: The file does not appear to be a valid ONNX binary. ({str(e)})"
        ) from e

    # Step 2: Strict ONNX graph and constraint check
    try:
        checker.check_model(model)
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass  # Non-fatal if shape inference encounters custom ops
    except Exception as e:
        raise ValueError(f"ONNX Graph Integrity Verification Failed: {str(e)}") from e

    # Step 3: Metadata extraction with honest fallbacks
    custom_meta = {prop.key: prop.value for prop in model.metadata_props}
    metadata = ModelMetadata(
        producer_name=model.producer_name if model.producer_name else "Not available",
        producer_version=model.producer_version if model.producer_version else "Not available",
        model_version=model.model_version,
        ir_version=model.ir_version,
        doc_string=model.doc_string if model.doc_string else "Not available",
        domain=model.domain if model.domain else "Not available",
        custom_metadata=custom_meta,
    )

    # Step 4: Opset imports
    opsets = [
        OpsetInfo(
            domain=opset.domain or "ai.onnx (default)",
            version=opset.version,
        )
        for opset in model.opset_import
    ]

    # Step 5: Graph Inputs and Outputs
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    
    inputs: List[TensorInfo] = []
    for inp in graph.input:
        if inp.name not in initializer_names:
            inputs.append(_parse_value_info(inp))

    outputs: List[TensorInfo] = [_parse_value_info(out) for out in graph.output]

    # Step 6: Initializers / Weights & External Data Detection
    weights: List[WeightInfo] = []
    total_parameters = 0
    total_weight_bytes = 0
    external_data_tensors: List[str] = []

    for init in graph.initializer:
        dtype_str = _get_tensor_type_str(init.data_type)
        shape = list(init.dims)
        num_elem = int(np.prod(shape)) if shape else 1
        is_ext = _check_external_data(init)
        if is_ext:
            external_data_tensors.append(init.name)

        if init.raw_data:
            size_b = len(init.raw_data)
        else:
            element_size = 4
            if init.data_type in (TensorProto.FLOAT, TensorProto.INT32, TensorProto.UINT32):
                element_size = 4
            elif init.data_type in (TensorProto.DOUBLE, TensorProto.INT64, TensorProto.UINT64):
                element_size = 8
            elif init.data_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.INT16, TensorProto.UINT16):
                element_size = 2
            elif init.data_type in (TensorProto.INT8, TensorProto.UINT8, TensorProto.BOOL):
                element_size = 1
            size_b = num_elem * element_size

        total_parameters += num_elem
        total_weight_bytes += size_b

        weights.append(
            WeightInfo(
                name=init.name,
                data_type=dtype_str,
                shape=shape,
                num_elements=num_elem,
                size_bytes=size_b,
                has_external_data=is_ext,
            )
        )

    has_external_data = len(external_data_tensors) > 0

    # Step 7: Build Tensor Producer-Consumer Connection Graph
    tensor_producers: Dict[str, str] = {}
    tensor_consumers: Dict[str, List[str]] = defaultdict(list)

    # Register graph inputs as producers
    for inp in inputs:
        tensor_producers[inp.name] = "Graph Input"

    # Register initializers as producers
    for w in weights:
        tensor_producers[w.name] = "Weight Initializer"

    # Register node producers and consumers
    for idx, node in enumerate(graph.node):
        node_name = node.name or f"{node.op_type}_{idx}"
        for out_name in node.output:
            if out_name:
                tensor_producers[out_name] = node_name
        for in_name in node.input:
            if in_name:
                tensor_consumers[in_name].append(node_name)

    # Register graph outputs
    for out in outputs:
        tensor_consumers[out.name].append("Graph Output")

    # Step 8: Assemble Rich Nodes with Connections
    nodes: List[NodeInfo] = []
    op_counter: Counter = Counter()

    for idx, node in enumerate(graph.node):
        op_type = node.op_type
        op_counter[op_type] += 1
        node_name = node.name or f"{op_type}_{idx}"
        
        attrs: Dict[str, Any] = {}
        for attr in node.attribute:
            try:
                attrs[attr.name] = onnx.helper.get_attribute_value(attr)
            except Exception:
                attrs[attr.name] = str(attr)

        # Predecessors (nodes producing inputs to this node)
        predecessors: List[str] = []
        for in_name in node.input:
            prod = tensor_producers.get(in_name)
            if prod and prod not in ("Graph Input", "Weight Initializer", node_name) and prod not in predecessors:
                predecessors.append(prod)

        # Successors (nodes consuming outputs of this node)
        successors: List[str] = []
        for out_name in node.output:
            for cons in tensor_consumers.get(out_name, []):
                if cons not in ("Graph Output", node_name) and cons not in successors:
                    successors.append(cons)

        nodes.append(
            NodeInfo(
                index=idx,
                name=node_name,
                op_type=op_type,
                domain=node.domain or "ai.onnx",
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attrs,
                predecessors=predecessors,
                successors=successors,
            )
        )

    # Step 9: Generate Grounded Inspection Warnings
    warnings = _generate_inspection_warnings(
        metadata=metadata,
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        weights=weights,
        op_counts=dict(op_counter.most_common()),
        has_external_data=has_external_data,
        external_data_tensors=external_data_tensors,
    )

    return ModelSummary(
        filename=filename,
        file_size_bytes=file_size_bytes,
        sha256_hash=sha256_hash,
        metadata=metadata,
        opsets=opsets,
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        weights=weights,
        op_counts=dict(op_counter.most_common()),
        distinct_operator_count=len(op_counter),
        total_nodes=len(nodes),
        total_parameters=total_parameters,
        total_weight_bytes=total_weight_bytes,
        has_external_data=has_external_data,
        external_data_tensors=external_data_tensors,
        warnings=warnings,
        is_valid=True,
        parsing_status="Successfully Parsed & Graph Constraints Verified",
        inference_status="Not Executed (Static Inspection Only)",
        compatibility_status="Not Analyzed (Scheduled for Future Phase)",
    )
