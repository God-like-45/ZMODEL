"""Execution backend module supporting local in-process and Kubernetes worker execution."""

from zmodel.core.execution.backend import (
    BackendExecutionError,
    BackendHealth,
    BackendUnavailableError,
    ExecutionBackend,
    LocalExecutionBackend,
    get_execution_backend,
)
from zmodel.core.execution.k8s_adapter import KubernetesExecutionBackend
from zmodel.core.execution.schemas import (
    RESOURCE_PROFILES,
    ResourceProfile,
    TimingBreakdown,
    WorkerJobResult,
    WorkerJobSpec,
    WorkerJobState,
    WorkerJobType,
    get_resource_profile,
)

__all__ = [
    "ExecutionBackend",
    "LocalExecutionBackend",
    "KubernetesExecutionBackend",
    "get_execution_backend",
    "BackendHealth",
    "BackendExecutionError",
    "BackendUnavailableError",
    "WorkerJobType",
    "WorkerJobState",
    "ResourceProfile",
    "RESOURCE_PROFILES",
    "get_resource_profile",
    "TimingBreakdown",
    "WorkerJobSpec",
    "WorkerJobResult",
]
