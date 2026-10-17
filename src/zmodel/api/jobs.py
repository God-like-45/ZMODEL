"""Bounded in-process background job execution manager for asynchronous benchmarks and optimizations."""

import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional
from zmodel.api.config import get_api_settings

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    """Stateful record tracking asynchronous execution of a benchmark or optimization job."""

    job_id: str
    job_type: str  # "benchmark" or "optimization"
    status: JobStatus = JobStatus.QUEUED
    created_at: str = field(default_factory=utc_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress_pct: int = 0
    progress_message: str = "Queued for execution"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    resource_id: Optional[str] = None  # run_id or experiment_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert job record to serializable dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress_pct": self.progress_pct,
            "progress_message": self.progress_message,
            "resource_id": self.resource_id,
            "error": self.error,
        }


def sanitize_error_message(err: Exception) -> str:
    """Sanitize exception message to avoid leaking filesystem paths or credentials."""
    msg = str(err)
    # Mask passwords
    msg = re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", msg)
    # Strip local filesystem paths like C:\Users\... or /home/...
    msg = re.sub(r"[A-Za-z]:\\[\w\\\.-]+", "[local_path]", msg)
    msg = re.sub(r"/(?:home|var|tmp|etc)/[\w/\.-]+", "[local_path]", msg)
    return msg


class JobQueueFullError(Exception):
    """Raised when the job queue exceeds maximum allowed pending jobs."""
    pass


class JobManager:
    """
    In-process, bounded job executor managing concurrency and execution limits.
    Jobs run in background threads without blocking HTTP request workers.
    """

    def __init__(self, concurrency: int = 2, max_queued: int = 20):
        self.concurrency = concurrency
        self.max_queued = max_queued
        self._executor = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="zmodel-job")
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit_job(
        self,
        job_type: str,
        task_fn: Callable[..., Dict[str, Any]],
        *args: Any,
        resource_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Submit a task for bounded background execution.
        Raises JobQueueFullError if pending queue capacity is reached.
        """
        with self._lock:
            # Count currently active and queued jobs
            pending = sum(1 for j in self._jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.RUNNING))
            if pending >= self.max_queued:
                raise JobQueueFullError(
                    f"Job queue capacity reached ({pending}/{self.max_queued} pending jobs). "
                    "Please wait for existing jobs to complete."
                )

            job_id = f"job-{uuid.uuid4().hex[:12]}"
            record = JobRecord(
                job_id=job_id,
                job_type=job_type,
                status=JobStatus.QUEUED,
                resource_id=resource_id,
            )
            self._jobs[job_id] = record

        # Dispatch execution
        self._executor.submit(self._run_wrapper, job_id, task_fn, *args, **kwargs)
        return job_id

    def _run_wrapper(self, job_id: str, task_fn: Callable[..., Dict[str, Any]], *args: Any, **kwargs: Any) -> None:
        """Thread wrapper that executes task, tracks progress, and updates status."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.RUNNING
            job.started_at = utc_now_iso()
            job.progress_pct = 10
            job.progress_message = "Execution started"

        def progress_callback(pct: int, msg: str) -> None:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id].progress_pct = max(0, min(100, pct))
                    self._jobs[job_id].progress_message = msg

        try:
            # Pass progress callback if accepted
            result = task_fn(*args, progress_callback=progress_callback, **kwargs)
            with self._lock:
                job = self._jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.completed_at = utc_now_iso()
                job.progress_pct = 100
                job.progress_message = "Execution completed successfully"
                job.result = result
                # Capture resource ID if present in result
                if result and isinstance(result, dict):
                    job.resource_id = (
                        result.get("run_id")
                        or result.get("experiment_id")
                        or job.resource_id
                    )
        except Exception as ex:
            logger.exception(f"Job {job_id} ({job.job_type}) failed: {ex}")
            with self._lock:
                job = self._jobs[job_id]
                job.status = JobStatus.FAILED
                job.completed_at = utc_now_iso()
                job.progress_pct = 100
                job.progress_message = "Execution failed"
                job.error = sanitize_error_message(ex)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Retrieve job record by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_by_resource_id(self, resource_id: str) -> Optional[JobRecord]:
        """Retrieve most recent job record associated with a resource ID (e.g. experiment_id or run_id)."""
        with self._lock:
            matching = [j for j in self._jobs.values() if j.resource_id == resource_id]
            if not matching:
                return None
            # Return newest job
            return sorted(matching, key=lambda j: j.created_at, reverse=True)[0]

    def list_jobs(self) -> Dict[str, JobRecord]:
        """Return shallow copy of all jobs."""
        with self._lock:
            return dict(self._jobs)

    def shutdown(self, wait: bool = False) -> None:
        """Shut down background executor."""
        self._executor.shutdown(wait=wait)
_JOB_MANAGER: Optional[JobManager] = None


def get_job_manager(reset: bool = False) -> JobManager:
    """Retrieve or initialize singleton JobManager, creating new instance if shut down."""
    global _JOB_MANAGER
    if reset or _JOB_MANAGER is None or _JOB_MANAGER._executor._shutdown:
        settings = get_api_settings()
        _JOB_MANAGER = JobManager(
            concurrency=settings.job_concurrency,
            max_queued=settings.max_queued_jobs,
        )
    return _JOB_MANAGER
