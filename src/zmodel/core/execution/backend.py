"""Execution backend abstraction supporting local in-process threads and Kubernetes worker jobs."""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from zmodel.core.execution.schemas import (
    TimingBreakdown,
    WorkerJobResult,
    WorkerJobSpec,
    WorkerJobState,
    WorkerJobType,
)

logger = logging.getLogger(__name__)


class BackendExecutionError(Exception):
    """Raised when job submission or execution fails on a backend."""
    pass


class BackendUnavailableError(BackendExecutionError):
    """Raised when the requested execution backend is not reachable or configured."""
    pass


@dataclass
class BackendHealth:
    """Diagnostic health assessment for an execution backend."""

    backend_name: str
    is_available: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "is_available": self.is_available,
            "details": self.details,
            "error_message": self.error_message,
        }


class ExecutionBackend(ABC):
    """Abstract interface defining the execution contract for benchmark and optimization workloads."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the backend (e.g. 'local', 'kubernetes')."""
        pass

    @abstractmethod
    def submit_job(self, spec: WorkerJobSpec) -> str:
        """Submit a validated job specification for execution. Returns stable job_id."""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> WorkerJobState:
        """Query current execution status of a job."""
        pass

    @abstractmethod
    def get_job_result(self, job_id: str) -> Optional[WorkerJobResult]:
        """Retrieve completed result or failure details for a job."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel an active or queued job. Returns True if cancellation initiated."""
        pass

    @abstractmethod
    def cleanup_job(self, job_id: str) -> bool:
        """Remove completed or terminated job resources."""
        pass

    @abstractmethod
    def check_health(self) -> BackendHealth:
        """Inspect backend connectivity and resource availability."""
        pass


class LocalExecutionBackend(ExecutionBackend):
    """
    Default in-process execution backend using the application's bounded JobManager.
    Executes worker tasks in background threads for reliable zero-infrastructure local development.
    """

    def __init__(self):
        from zmodel.api.jobs import get_job_manager
        self.job_manager = get_job_manager()
        self._results: Dict[str, WorkerJobResult] = {}
        self._specs: Dict[str, WorkerJobSpec] = {}

    @property
    def name(self) -> str:
        return "local"

    def submit_job(self, spec: WorkerJobSpec) -> str:
        """Dispatch task to local ThreadPoolExecutor."""
        self._specs[spec.job_id] = spec
        enqueue_time = time.time()

        def _task_runner(progress_callback=None) -> Dict[str, Any]:
            start_time = time.time()
            queue_delay = start_time - enqueue_time
            started_iso = datetime.now(timezone.utc).isoformat()

            # Execute via worker service logic
            from zmodel.worker.runner import execute_worker_task
            try:
                result_data, out_artifacts, timing = execute_worker_task(spec, progress_callback=progress_callback)
                total_duration = time.time() - enqueue_time
                timing.queue_delay_seconds = queue_delay
                timing.total_duration_seconds = total_duration

                job_res = WorkerJobResult(
                    job_id=spec.job_id,
                    status=WorkerJobState.SUCCEEDED,
                    model_fingerprint=spec.model_fingerprint,
                    started_at=started_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    timing=timing,
                    worker_image="local-process",
                    worker_node="localhost",
                    pod_name=None,
                    execution_environment={"backend": "local", "threads": os.cpu_count()},
                    result_data=result_data,
                    output_artifact_ids=out_artifacts,
                )
                self._results[spec.job_id] = job_res
                return result_data
            except Exception as ex:
                total_duration = time.time() - enqueue_time
                timing = TimingBreakdown(queue_delay_seconds=queue_delay, total_duration_seconds=total_duration)
                job_res = WorkerJobResult(
                    job_id=spec.job_id,
                    status=WorkerJobState.FAILED,
                    model_fingerprint=spec.model_fingerprint,
                    started_at=started_iso,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    timing=timing,
                    worker_image="local-process",
                    worker_node="localhost",
                    pod_name=None,
                    execution_environment={"backend": "local"},
                    error_category="EXECUTION_ERROR",
                    error_message=str(ex),
                )
                self._results[spec.job_id] = job_res
                raise

        # Submit through JobManager to enforce concurrency bounds
        self.job_manager.submit_job(
            job_type=spec.job_type.value,
            task_fn=_task_runner,
            resource_id=spec.experiment_id or spec.run_id or spec.job_id,
        )
        return spec.job_id

    def get_job_status(self, job_id: str) -> WorkerJobState:
        if job_id in self._results:
            return self._results[job_id].status

        # Query underlying JobRecord
        job_rec = self.job_manager.get_job_by_resource_id(job_id) or self.job_manager.get_job(job_id)
        if not job_rec:
            # Check by spec
            if job_id in self._specs:
                return WorkerJobState.QUEUED
            return WorkerJobState.FAILED

        status_str = job_rec.status.value.upper()
        if status_str == "QUEUED":
            return WorkerJobState.QUEUED
        if status_str == "RUNNING":
            return WorkerJobState.RUNNING
        if status_str == "COMPLETED":
            return WorkerJobState.SUCCEEDED
        if status_str == "FAILED":
            return WorkerJobState.FAILED
        return WorkerJobState.QUEUED

    def get_job_result(self, job_id: str) -> Optional[WorkerJobResult]:
        return self._results.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        # In-process threads cannot be hard-killed safely, but status can be marked CANCELLED
        if job_id in self._results:
            if self._results[job_id].status in (WorkerJobState.QUEUED, WorkerJobState.SUBMITTED, WorkerJobState.RUNNING):
                self._results[job_id].status = WorkerJobState.CANCELLED
                return True
        if job_id in self._specs:
            self._specs.pop(job_id, None)
            return True
        return True

    def cleanup_job(self, job_id: str) -> bool:
        self._specs.pop(job_id, None)
        self._results.pop(job_id, None)
        return True

    def check_health(self) -> BackendHealth:
        return BackendHealth(
            backend_name="local",
            is_available=True,
            details={
                "concurrency": self.job_manager.concurrency,
                "max_queued": self.job_manager.max_queued,
                "active_jobs": len([j for j in self.job_manager.list_jobs().values() if j.status.value in ("QUEUED", "RUNNING")]),
            },
        )


_BACKEND_INSTANCE: Optional[ExecutionBackend] = None


def get_execution_backend(name: Optional[str] = None) -> ExecutionBackend:
    """
    Factory resolving the active execution backend.
    Honors EXECUTION_BACKEND environment variable ('local' or 'kubernetes').
    Never silently switches if 'kubernetes' was requested but unavailable.
    """
    global _BACKEND_INSTANCE
    target_name = (name or os.environ.get("EXECUTION_BACKEND", "local")).lower().strip()

    if target_name == "kubernetes":
        from zmodel.core.execution.k8s_adapter import KubernetesExecutionBackend
        return KubernetesExecutionBackend()

    if _BACKEND_INSTANCE is None or _BACKEND_INSTANCE.name != "local":
        _BACKEND_INSTANCE = LocalExecutionBackend()
    return _BACKEND_INSTANCE
