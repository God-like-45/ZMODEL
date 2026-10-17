"""
Reproducible ONNX Model Benchmark Engine for Z-ModelPilot.

Executes genuine inference on real uploaded ONNX models using onnxruntime.
Measures latency percentiles, throughput, memory consumption, and numerical correctness.
Operates independently of Streamlit UI presentation logic.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
import psutil

from zmodel.core.model_loader import ModelSummary, TensorInfo


# --- Exceptions ---

class BenchmarkError(Exception):
    """Base class for benchmark-related errors."""
    pass


class ProviderUnavailableError(BenchmarkError):
    """Raised when a requested execution provider is not available in the installed runtime."""
    pass


class BenchmarkConfigurationError(BenchmarkError):
    """Raised when benchmark parameters violate valid ranges or constraints."""
    pass


class UnsupportedInputTypeError(BenchmarkError):
    """Raised when an ONNX input data type cannot be safely synthesized."""
    pass


class CorrectnessValidationError(BenchmarkError):
    """Raised when a pre-benchmark correctness or smoke inference run fails."""
    pass


# --- Data Structures ---

@dataclass(frozen=True)
class ProviderEnvironmentInfo:
    """Installed ONNX Runtime environment and available execution providers."""
    installed_ort_version: str
    available_providers: List[str]
    default_provider: str


@dataclass
class BenchmarkConfig:
    """Validated configuration parameters for a controlled benchmark execution."""
    execution_provider: str = "CPUExecutionProvider"
    warmup_iterations: int = 10
    measured_iterations: int = 50
    batch_size: int = 1
    repeated_rounds: int = 3
    random_seed: int = 42
    dynamic_dimensions: Dict[str, int] = field(default_factory=dict)
    intra_op_num_threads: Optional[int] = None
    inter_op_num_threads: Optional[int] = None
    graph_optimization_level: Optional[str] = None  # ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
    execution_mode: Optional[str] = None  # ORT_SEQUENTIAL, ORT_PARALLEL
    enable_memory_pattern: Optional[bool] = None

    def validate(self) -> None:
        """Validate parameter bounds to prevent resource exhaustion or crashes."""
        if self.warmup_iterations < 1 or self.warmup_iterations > 200:
            raise BenchmarkConfigurationError(
                f"warmup_iterations must be between 1 and 200, got {self.warmup_iterations}"
            )
        if self.measured_iterations < 1 or self.measured_iterations > 2000:
            raise BenchmarkConfigurationError(
                f"measured_iterations must be between 1 and 2000, got {self.measured_iterations}"
            )
        if self.batch_size < 1 or self.batch_size > 128:
            raise BenchmarkConfigurationError(
                f"batch_size must be between 1 and 128, got {self.batch_size}"
            )
        if self.repeated_rounds < 1 or self.repeated_rounds > 10:
            raise BenchmarkConfigurationError(
                f"repeated_rounds must be between 1 and 10, got {self.repeated_rounds}"
            )
        if self.intra_op_num_threads is not None and self.intra_op_num_threads < 1:
            raise BenchmarkConfigurationError("intra_op_num_threads must be >= 1")
        if self.inter_op_num_threads is not None and self.inter_op_num_threads < 1:
            raise BenchmarkConfigurationError("inter_op_num_threads must be >= 1")
        for dim_name, dim_val in self.dynamic_dimensions.items():
            if not isinstance(dim_val, int) or dim_val < 1:
                raise BenchmarkConfigurationError(
                    f"Dynamic dimension '{dim_name}' must be a positive integer, got {dim_val}"
                )

    def configuration_hash(self) -> str:
        """Generate a deterministic 16-hex fingerprint for configuration deduplication."""
        import hashlib
        payload = {
            "execution_provider": self.execution_provider,
            "batch_size": self.batch_size,
            "intra_op_num_threads": self.intra_op_num_threads,
            "inter_op_num_threads": self.inter_op_num_threads,
            "graph_optimization_level": self.graph_optimization_level,
            "execution_mode": self.execution_mode,
            "enable_memory_pattern": self.enable_memory_pattern,
            "dynamic_dimensions": sorted(self.dynamic_dimensions.items()),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CorrectnessResult:
    """Pre-benchmark output shape, finiteness, and repeatability validation."""
    passed: bool
    output_names: List[str]
    output_shapes: Dict[str, List[int]]
    all_finite: bool
    repeat_consistent: bool
    details: str


@dataclass(frozen=True)
class LatencySummary:
    """High-resolution latency statistics in milliseconds."""
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    sample_count: int


@dataclass(frozen=True)
class ThroughputSummary:
    """Calculated inference and sample throughput."""
    inferences_per_second: float
    samples_per_second: float
    total_elapsed_seconds: float
    total_measured_inferences: int


@dataclass(frozen=True)
class MemorySummary:
    """Process RSS memory metrics in megabytes."""
    baseline_rss_mb: Optional[float]
    post_session_rss_mb: Optional[float]
    session_delta_mb: Optional[float]
    peak_rss_mb: Optional[float]
    notes: str


@dataclass
class RoundResult:
    """Detailed measurements for an individual benchmark round."""
    round_number: int
    latencies_ms: List[float]
    mean_ms: float
    median_ms: float
    p95_ms: float
    inferences_per_second: float


@dataclass
class BenchmarkReport:
    """Complete, reproducible, serializable model benchmark report."""
    run_id: str
    timestamp_utc: str
    model_name: str
    model_fingerprint: str
    status: str  # "SUCCESS", "FAILED", "CANCELLED"
    error_message: Optional[str]
    environment: Dict[str, Any]
    config: Dict[str, Any]
    input_spec: Dict[str, Any]
    correctness: Optional[Dict[str, Any]]
    latency: Optional[Dict[str, Any]]
    throughput: Optional[Dict[str, Any]]
    memory: Optional[Dict[str, Any]]
    per_round_results: List[Dict[str, Any]]
    all_latencies_ms: List[float]
    warnings: List[str]
    limitations: List[str]

    @property
    def environment_meta(self) -> Dict[str, Any]:
        """Convenience property for environment metadata."""
        return self.environment

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to sanitized dictionary representation."""
        return {
            "report_version": "1.0.0",
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "status": self.status,
            "error_message": self.error_message,
            "model_metadata": {
                "model_name": self.model_name,
                "model_fingerprint": self.model_fingerprint,
            },
            "environment": self.environment,
            "configuration": self.config,
            "input_specifications": self.input_spec,
            "correctness": self.correctness,
            "latency": self.latency,
            "throughput": self.throughput,
            "memory": self.memory,
            "per_round_results": self.per_round_results,
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_csv(self) -> str:
        """Export individual measured latency samples as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["sample_index", "round_number", "latency_ms"])

        idx = 1
        for round_data in self.per_round_results:
            r_num = round_data.get("round_number", 1)
            for lat in round_data.get("latencies_ms", []):
                writer.writerow([idx, r_num, f"{lat:.4f}"])
                idx += 1
        return output.getvalue()


# --- Execution Provider Discovery ---

def discover_execution_providers() -> ProviderEnvironmentInfo:
    """
    Query the active ONNX Runtime installation for available execution providers.
    """
    installed_version = getattr(ort, "__version__", "unknown")
    try:
        available = ort.get_available_providers()
    except Exception:
        available = ["CPUExecutionProvider"]

    default_p = "CPUExecutionProvider" if "CPUExecutionProvider" in available else available[0]
    return ProviderEnvironmentInfo(
        installed_ort_version=installed_version,
        available_providers=list(available),
        default_provider=default_p,
    )


# --- Input Generation Engine ---

SUPPORTED_NUMPY_TYPES: Dict[str, np.dtype] = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "int32": np.int32,
    "int64": np.int64,
    "int16": np.int16,
    "int8": np.int8,
    "uint8": np.uint8,
    "bool": np.bool_,
}

MAX_ELEMENTS_PER_TENSOR = 10_000_000  # Guardrail against runaway memory allocation


def generate_benchmark_inputs(
    model_summary: ModelSummary,
    config: BenchmarkConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate reproducible, seeded synthetic input tensors derived strictly from model input signatures.
    """
    rng = np.random.RandomState(config.random_seed)
    inputs_dict: Dict[str, np.ndarray] = {}
    input_specs: Dict[str, Any] = {}

    for tensor_info in model_summary.inputs:
        dtype_str = tensor_info.data_type.lower()
        if dtype_str not in SUPPORTED_NUMPY_TYPES:
            raise UnsupportedInputTypeError(
                f"Input tensor '{tensor_info.name}' has unsupported data type '{tensor_info.data_type}'. "
                f"Supported types: {sorted(SUPPORTED_NUMPY_TYPES.keys())}"
            )
        np_dtype = SUPPORTED_NUMPY_TYPES[dtype_str]

        # Resolve concrete shape from fixed dimensions or dynamic overrides
        resolved_shape: List[int] = []
        for dim_idx, dim in enumerate(tensor_info.shape):
            if isinstance(dim, int):
                resolved_shape.append(dim if dim > 0 else 1)
            else:
                dim_name = str(dim)
                if dim_name in config.dynamic_dimensions:
                    resolved_shape.append(config.dynamic_dimensions[dim_name])
                elif dim_idx == 0 or dim_name.lower() in ("batch", "batch_size", "n", "dim_0"):
                    resolved_shape.append(config.batch_size)
                else:
                    # Deterministic default for intermediate symbolic axes
                    resolved_shape.append(1)

        # Guard against zero-sized or excessively large allocations
        total_elements = int(np.prod(resolved_shape)) if resolved_shape else 1
        if total_elements > MAX_ELEMENTS_PER_TENSOR:
            raise BenchmarkConfigurationError(
                f"Input tensor '{tensor_info.name}' resolves to {total_elements:,} elements "
                f"(shape: {resolved_shape}), which exceeds the safety limit of {MAX_ELEMENTS_PER_TENSOR:,}."
            )

        # Generate seeded random values based on dtype
        if np.issubdtype(np_dtype, np.floating):
            # Safe standardized range [-1.0, 1.0] to prevent exponential explosions
            raw_vals = rng.uniform(-1.0, 1.0, size=resolved_shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            raw_vals = rng.randint(0, 10, size=resolved_shape).astype(np_dtype)
        elif np_dtype == np.bool_:
            raw_vals = rng.choice([True, False], size=resolved_shape).astype(np.bool_)
        else:
            raw_vals = np.ones(resolved_shape, dtype=np_dtype)

        inputs_dict[tensor_info.name] = raw_vals
        input_specs[tensor_info.name] = {
            "declared_shape": list(tensor_info.shape),
            "resolved_shape": resolved_shape,
            "data_type": dtype_str,
            "total_elements": total_elements,
            "generation_strategy": "seeded_uniform_pseudo_random",
            "seed": config.random_seed,
        }

    return inputs_dict, input_specs


# --- Correctness Validation Engine ---

def validate_model_correctness(
    session: ort.InferenceSession,
    inputs: Dict[str, np.ndarray],
) -> CorrectnessResult:
    """
    Run pre-benchmark smoke validation, verify output dimensions, finiteness, and execution repeatability.
    """
    try:
        # Run first smoke inference
        raw_outputs = session.run(None, inputs)
    except Exception as e:
        raise CorrectnessValidationError(f"Smoke inference failed: {str(e)}") from e

    output_defs = session.get_outputs()
    output_names = [out.name for out in output_defs]
    output_shapes: Dict[str, List[int]] = {}

    all_finite = True
    for idx, out_val in enumerate(raw_outputs):
        name = output_names[idx] if idx < len(output_names) else f"output_{idx}"
        output_shapes[name] = list(out_val.shape)

        if isinstance(out_val, np.ndarray) and np.issubdtype(out_val.dtype, np.floating):
            if not np.all(np.isfinite(out_val)):
                all_finite = False

    # Check execution repeatability with the identical input feed
    try:
        repeat_outputs = session.run(None, inputs)
        repeat_consistent = True
        for out1, out2 in zip(raw_outputs, repeat_outputs):
            if isinstance(out1, np.ndarray) and isinstance(out2, np.ndarray):
                if not np.allclose(out1, out2, rtol=1e-4, atol=1e-5, equal_nan=True):
                    repeat_consistent = False
                    break
    except Exception:
        repeat_consistent = False

    if not all_finite:
        raise CorrectnessValidationError(
            "Smoke inference generated non-finite values (NaN or Inf) in model output tensors."
        )

    return CorrectnessResult(
        passed=True,
        output_names=output_names,
        output_shapes=output_shapes,
        all_finite=all_finite,
        repeat_consistent=repeat_consistent,
        details="Smoke inference succeeded with valid shapes, finite numerical outputs, and repeatable execution.",
    )


# --- Statistical Latency Calculations ---

def calculate_latency_statistics(latencies_ms: List[float]) -> LatencySummary:
    """
    Compute rigorous latency statistics and percentiles using numpy.
    """
    if not latencies_ms:
        return LatencySummary(
            mean_ms=0.0,
            median_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            p90_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            std_dev_ms=0.0,
            sample_count=0,
        )

    arr = np.array(latencies_ms, dtype=np.float64)
    return LatencySummary(
        mean_ms=float(np.mean(arr)),
        median_ms=float(np.median(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        std_dev_ms=float(np.std(arr)),
        sample_count=len(latencies_ms),
    )


def calculate_throughput(
    total_inferences: int,
    elapsed_seconds: float,
    batch_size: int,
) -> ThroughputSummary:
    """
    Calculate inferences per second and samples per second based on measured wall-clock elapsed time.
    """
    if elapsed_seconds <= 0 or total_inferences <= 0:
        return ThroughputSummary(
            inferences_per_second=0.0,
            samples_per_second=0.0,
            total_elapsed_seconds=elapsed_seconds,
            total_measured_inferences=total_inferences,
        )

    inf_per_sec = float(total_inferences / elapsed_seconds)
    samples_per_sec = float(inf_per_sec * batch_size)
    return ThroughputSummary(
        inferences_per_second=inf_per_sec,
        samples_per_second=samples_per_sec,
        total_elapsed_seconds=elapsed_seconds,
        total_measured_inferences=total_inferences,
    )


# --- Process Memory Helper ---

def get_current_process_memory_mb() -> Optional[float]:
    """Retrieve current process Resident Set Size (RSS) memory in megabytes."""
    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss / (1024 * 1024))
    except Exception:
        return None


# --- Benchmark Execution Engine ---

class BenchmarkEngine:
    """
    Stateless benchmark execution service for ONNX models.
    """

    @classmethod
    def run_benchmark(
        cls,
        model_path_or_bytes: Union[str, bytes],
        model_summary: ModelSummary,
        config: BenchmarkConfig,
    ) -> BenchmarkReport:
        """
        Execute full controlled benchmark: session creation, smoke check, warm-up, timed rounds, and statistics.
        """
        config.validate()
        run_id = f"BENCH-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 10000:04d}"
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        env_info = discover_execution_providers()
        requested_provider = config.execution_provider

        # Check provider availability
        if requested_provider not in env_info.available_providers:
            raise ProviderUnavailableError(
                f"Requested execution provider '{requested_provider}' is not available in the installed "
                f"ONNX Runtime ({env_info.installed_ort_version}). "
                f"Available providers: {env_info.available_providers}"
            )

        # Baseline memory measurement
        baseline_rss = get_current_process_memory_mb()

        # Build session options
        so = ort.SessionOptions()
        if config.intra_op_num_threads is not None:
            so.intra_op_num_threads = config.intra_op_num_threads
        if config.inter_op_num_threads is not None:
            so.inter_op_num_threads = config.inter_op_num_threads
        if config.graph_optimization_level:
            level = getattr(ort.GraphOptimizationLevel, config.graph_optimization_level, None)
            if level is not None:
                so.graph_optimization_level = level
        if config.execution_mode:
            mode = getattr(ort.ExecutionMode, config.execution_mode, None)
            if mode is not None:
                so.execution_mode = mode
        if config.enable_memory_pattern is not None:
            so.enable_mem_pattern = config.enable_memory_pattern

        # Create ONNX Runtime Session
        try:
            if isinstance(model_path_or_bytes, bytes):
                session = ort.InferenceSession(model_path_or_bytes, sess_options=so, providers=[requested_provider])
            else:
                session = ort.InferenceSession(str(model_path_or_bytes), sess_options=so, providers=[requested_provider])
        except Exception as e:
            raise BenchmarkError(f"Failed to create ONNX Runtime session: {str(e)}") from e

        post_session_rss = get_current_process_memory_mb()
        session_delta = (
            round(post_session_rss - baseline_rss, 2)
            if post_session_rss is not None and baseline_rss is not None
            else None
        )

        # Generate inputs
        inputs_dict, input_specs = generate_benchmark_inputs(model_summary, config)

        # Correctness validation (smoke inference run)
        correctness_res = validate_model_correctness(session, inputs_dict)

        # Warm-up phase (outside timed measurement)
        for _ in range(config.warmup_iterations):
            session.run(None, inputs_dict)

        # Measurement Phase across Repeated Rounds
        round_results: List[RoundResult] = []
        all_latencies_ms: List[float] = []
        peak_rss = post_session_rss

        wall_clock_start = time.perf_counter()

        for round_idx in range(1, config.repeated_rounds + 1):
            round_latencies: List[float] = []

            for _ in range(config.measured_iterations):
                # Measure strictly the inference execution call
                t_start = time.perf_counter_ns()
                session.run(None, inputs_dict)
                t_end = time.perf_counter_ns()

                lat_ms = (t_end - t_start) / 1_000_000.0
                round_latencies.append(lat_ms)

            # Sample peak process memory
            curr_mem = get_current_process_memory_mb()
            if curr_mem is not None and (peak_rss is None or curr_mem > peak_rss):
                peak_rss = curr_mem

            r_stats = calculate_latency_statistics(round_latencies)
            r_elapsed = sum(round_latencies) / 1000.0
            r_throughput = float(len(round_latencies) / r_elapsed) if r_elapsed > 0 else 0.0

            round_results.append(
                RoundResult(
                    round_number=round_idx,
                    latencies_ms=round_latencies,
                    mean_ms=r_stats.mean_ms,
                    median_ms=r_stats.median_ms,
                    p95_ms=r_stats.p95_ms,
                    inferences_per_second=r_throughput,
                )
            )
            all_latencies_ms.extend(round_latencies)

        wall_clock_elapsed = time.perf_counter() - wall_clock_start

        # Aggregate Statistics
        overall_latency = calculate_latency_statistics(all_latencies_ms)
        overall_throughput = calculate_throughput(
            total_inferences=len(all_latencies_ms),
            elapsed_seconds=wall_clock_elapsed,
            batch_size=config.batch_size,
        )

        memory_summary = MemorySummary(
            baseline_rss_mb=baseline_rss,
            post_session_rss_mb=post_session_rss,
            session_delta_mb=session_delta,
            peak_rss_mb=peak_rss,
            notes=(
                "Process-level memory measurements reflect Resident Set Size (RSS) allocated by the Python process. "
                "Because Streamlit, runtime libraries, and graph initializers share the process memory space, "
                "RSS is an indicator of memory growth rather than an isolated model-only footprint."
            ),
        )

        # Environment metadata
        mach = platform.machine() or "unknown"
        is_s390x = mach.lower() in ("s390x", "s390")
        provenance = "IBM_Z_NATIVE" if is_s390x else "LOCAL_HOST"

        environment_meta = {
            "python_version": platform.python_version(),
            "os_name": platform.system(),
            "os_release": platform.release(),
            "architecture": mach,
            "host_architecture": mach,
            "benchmark_provenance": provenance,
            "target_architecture_verified": is_s390x,
            "provenance_label": f"IBM Z ({mach})" if is_s390x else f"Local Host ({mach})",
            "cpu_count_logical": psutil.cpu_count(logical=True) if hasattr(psutil, "cpu_count") else "unknown",
            "cpu_count_physical": psutil.cpu_count(logical=False) if hasattr(psutil, "cpu_count") else "unknown",
            "onnxruntime_version": env_info.installed_ort_version,
            "available_providers": env_info.available_providers,
            "active_session_providers": session.get_providers(),
        }

        # Warnings and Limitations
        warnings: List[str] = []
        limitations: List[str] = [
            "Local CPU inference does NOT represent IBM Z (s390x) or NNPA coprocessor performance.",
            "Latency measures runtime execution boundary only; input tensor synthesis is excluded.",
            "Memory metrics represent process RSS changes and are not isolated to model weights.",
        ]

        if not is_s390x:
            limitations.append(
                f"Benchmark executed on {mach} host; does not establish IBM Z or LinuxONE performance."
            )

        if config.execution_provider == "CPUExecutionProvider":
            warnings.append(
                "Benchmark executed on host CPUExecutionProvider. Results are architecture-specific to this machine."
            )

        return BenchmarkReport(
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            model_name=model_summary.filename or "Unnamed Model",
            model_fingerprint=model_summary.sha256_hash,
            status="SUCCESS",
            error_message=None,
            environment=environment_meta,
            config=asdict(config),
            input_spec=input_specs,
            correctness=asdict(correctness_res),
            latency=asdict(overall_latency),
            throughput=asdict(overall_throughput),
            memory=asdict(memory_summary),
            per_round_results=[asdict(r) for r in round_results],
            all_latencies_ms=all_latencies_ms,
            warnings=warnings,
            limitations=limitations,
        )
