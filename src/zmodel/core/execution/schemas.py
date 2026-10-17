"""Standardized schemas, resource profiles, and timing contracts for Kubernetes worker jobs."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkerJobType(str, Enum):
    """Categorization of background tasks dispatched to workers."""

    BENCHMARK = "benchmark"
    OPTIMIZATION = "optimization"
    GRAPH_OPTIMIZATION = "graph_optimization"


class WorkerJobState(str, Enum):
    """Lifecycle states observed for worker jobs."""

    QUEUED = "QUEUED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ResourceProfile:
    """Configurable Kubernetes container compute resource allocation."""

    name: str
    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str
    active_deadline_seconds: int
    max_retries: int = 1
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Predefined, safety-bounded resource profiles
RESOURCE_PROFILES: Dict[str, ResourceProfile] = {
    "small_cpu": ResourceProfile(
        name="small_cpu",
        cpu_request="250m",
        cpu_limit="500m",
        memory_request="256Mi",
        memory_limit="512Mi",
        active_deadline_seconds=120,
        max_retries=1,
        description="Lightweight inference micro-benchmarks on small models (< 10 MB).",
    ),
    "standard_cpu": ResourceProfile(
        name="standard_cpu",
        cpu_request="500m",
        cpu_limit="1000m",
        memory_request="512Mi",
        memory_limit="1Gi",
        active_deadline_seconds=300,
        max_retries=1,
        description="Standard latency/throughput benchmarking with multiple rounds and warmups.",
    ),
    "graph_opt": ResourceProfile(
        name="graph_opt",
        cpu_request="1000m",
        cpu_limit="2000m",
        memory_request="1Gi",
        memory_limit="2Gi",
        active_deadline_seconds=600,
        max_retries=1,
        description="Graph-level optimization, operator fusion, constant folding, and numerical verification.",
    ),
    "large_opt": ResourceProfile(
        name="large_opt",
        cpu_request="1000m",
        cpu_limit="2000m",
        memory_request="2Gi",
        memory_limit="4Gi",
        active_deadline_seconds=900,
        max_retries=1,
        description="Multi-objective Pareto exploration sweeping multiple batch sizes and thread allocations.",
    ),
}


def get_resource_profile(profile_name: str) -> ResourceProfile:
    """Retrieve resource profile by name, falling back to standard_cpu if unknown."""
    return RESOURCE_PROFILES.get(profile_name.lower(), RESOURCE_PROFILES["standard_cpu"])


@dataclass
class TimingBreakdown:
    """
    Granular lifecycle timing distinguishing infrastructure delay from model inference.
    Kubernetes scheduling or container startup must never be conflated with inference latency.
    """

    queue_delay_seconds: float = 0.0
    startup_overhead_seconds: float = 0.0
    warmup_overhead_seconds: float = 0.0
    inference_duration_seconds: float = 0.0
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class WorkerJobSpec:
    """Immutable job specification provided to worker containers."""

    job_id: str
    job_type: WorkerJobType
    model_fingerprint: str
    model_filename: str = "model.onnx"
    input_artifact_id: Optional[str] = None
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_profile_id: Optional[str] = None
    resource_profile: str = "standard_cpu"
    timeout_seconds: int = 300
    max_retries: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["job_type"] = self.job_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerJobSpec":
        c_data = dict(data)
        if isinstance(c_data.get("job_type"), str):
            c_data["job_type"] = WorkerJobType(c_data["job_type"])
        return cls(**c_data)


@dataclass
class WorkerJobResult:
    """Structured result metadata published upon worker completion or failure."""

    job_id: str
    status: WorkerJobState
    model_fingerprint: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    worker_image: str = "zmodel-worker:latest"
    worker_node: Optional[str] = None
    pod_name: Optional[str] = None
    execution_environment: Dict[str, Any] = field(default_factory=dict)
    result_data: Optional[Dict[str, Any]] = None
    output_artifact_ids: List[str] = field(default_factory=list)
    error_category: Optional[str] = None
    error_message: Optional[str] = None
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["timing"] = self.timing.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerJobResult":
        c_data = dict(data)
        if isinstance(c_data.get("status"), str):
            c_data["status"] = WorkerJobState(c_data["status"])
        if isinstance(c_data.get("timing"), dict):
            c_data["timing"] = TimingBreakdown(**c_data["timing"])
        return cls(**c_data)
