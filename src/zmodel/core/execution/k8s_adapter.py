"""Kubernetes Job execution adapter for isolated, resource-controlled worker workloads."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from zmodel.core.execution.backend import (
    BackendExecutionError,
    BackendHealth,
    BackendUnavailableError,
    ExecutionBackend,
)
from zmodel.core.execution.schemas import (
    RESOURCE_PROFILES,
    TimingBreakdown,
    WorkerJobResult,
    WorkerJobSpec,
    WorkerJobState,
    get_resource_profile,
)

logger = logging.getLogger(__name__)

# Kubernetes naming constraints: must be lowercase RFC 1123 label (< 63 chars)
def sanitize_k8s_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\-]", "-", name.lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned[:63]


class KubernetesExecutionBackend(ExecutionBackend):
    """
    Submits, tracks, and cleans up Kubernetes Jobs for isolated model benchmark and optimization tasks.
    Enforces container resource quotas, active execution deadlines, and non-root execution.
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        worker_image: Optional[str] = None,
        image_pull_policy: Optional[str] = None,
        batch_v1_api: Optional[Any] = None,
        core_v1_api: Optional[Any] = None,
    ):
        self.namespace = namespace or os.environ.get("K8S_NAMESPACE", "zmodel-jobs")
        self.worker_image = worker_image or os.environ.get("WORKER_IMAGE", "zmodel-worker:latest")
        self.image_pull_policy = image_pull_policy or os.environ.get("IMAGE_PULL_POLICY", "IfNotPresent")
        self.shared_storage_pvc = os.environ.get("SHARED_STORAGE_PVC", "zmodel-artifacts-pvc")

        # Injected or lazily initialized API clients
        self._batch_v1 = batch_v1_api
        self._core_v1 = core_v1_api
        self._initialized = batch_v1_api is not None

    @property
    def name(self) -> str:
        return "kubernetes"

    def _ensure_clients(self) -> None:
        """Initialize official kubernetes client from kubeconfig or in-cluster credentials."""
        if self._initialized:
            return

        try:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            except Exception:
                config.load_kube_config()
                logger.info("Loaded local kubeconfig")

            self._batch_v1 = client.BatchV1Api()
            self._core_v1 = client.CoreV1Api()
            self._initialized = True
        except Exception as ex:
            raise BackendUnavailableError(
                f"Kubernetes client initialization failed. Cluster is unreachable: {ex}"
            )

    def _build_job_manifest(self, spec: WorkerJobSpec) -> Dict[str, Any]:
        """Construct validated Kubernetes Job manifest with strict resource limits and security contexts."""
        profile = get_resource_profile(spec.resource_profile)
        job_name = sanitize_k8s_name(spec.job_id)

        # Labels adhering to Kubernetes recommended labels
        labels = {
            "app.kubernetes.io/name": "zmodel-worker",
            "app.kubernetes.io/part-of": "zmodel-pilot",
            "app.kubernetes.io/managed-by": "zmodel-control-plane",
            "zmodel.ai/job-id": spec.job_id,
            "zmodel.ai/job-type": spec.job_type.value,
        }

        # Safe environment variables
        env_vars = [
            {"name": "ZMODEL_JOB_ID", "value": spec.job_id},
            {"name": "ZMODEL_JOB_TYPE", "value": spec.job_type.value},
            {"name": "LOG_LEVEL", "value": os.environ.get("LOG_LEVEL", "INFO")},
            {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
            {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
        ]

        if "DATABASE_URL" in os.environ:
            env_vars.append({"name": "DATABASE_URL", "value": os.environ["DATABASE_URL"]})

        # Command executing non-interactive worker CLI
        spec_json_str = json.dumps(spec.to_dict())
        command = [
            "python",
            "-m",
            "zmodel.worker.main",
            "--spec-json",
            spec_json_str,
        ]

        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
                "labels": labels,
            },
            "spec": {
                "activeDeadlineSeconds": min(spec.timeout_seconds, profile.active_deadline_seconds),
                "backoffLimit": spec.max_retries,
                "ttlSecondsAfterFinished": 3600,
                "template": {
                    "metadata": {
                        "labels": labels,
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "runAsGroup": 1000,
                            "fsGroup": 1000,
                        },
                        "containers": [
                            {
                                "name": "worker",
                                "image": self.worker_image,
                                "imagePullPolicy": self.image_pull_policy,
                                "command": command,
                                "env": env_vars,
                                "resources": {
                                    "requests": {
                                        "cpu": profile.cpu_request,
                                        "memory": profile.memory_request,
                                    },
                                    "limits": {
                                        "cpu": profile.cpu_limit,
                                        "memory": profile.memory_limit,
                                    },
                                },
                                "securityContext": {
                                    "allowPrivilegeEscalation": False,
                                    "capabilities": {
                                        "drop": ["ALL"],
                                    },
                                },
                            }
                        ],
                    },
                },
            },
        }
        return manifest

    def submit_job(self, spec: WorkerJobSpec) -> str:
        """Submit a Kubernetes Job manifest using the official BatchV1Api."""
        self._ensure_clients()
        manifest = self._build_job_manifest(spec)
        job_name = manifest["metadata"]["name"]

        try:
            # Check for existing job with same name (idempotent duplicate submission protection)
            try:
                existing = self._batch_v1.read_namespaced_job(job_name, self.namespace)
                if existing:
                    logger.info(f"Kubernetes Job {job_name} already exists. Reusing.")
                    return spec.job_id
            except Exception:
                pass

            # Create Job
            self._batch_v1.create_namespaced_job(namespace=self.namespace, body=manifest)
            logger.info(f"Submitted Kubernetes Job {job_name} in namespace {self.namespace}")
            return spec.job_id
        except Exception as ex:
            logger.exception(f"Failed to submit Kubernetes Job for {spec.job_id}: {ex}")
            raise BackendExecutionError(f"Kubernetes Job submission failed: {ex}")

    def get_job_status(self, job_id: str) -> WorkerJobState:
        """Query Kubernetes Job and Pod status and map to WorkerJobState."""
        self._ensure_clients()
        job_name = sanitize_k8s_name(job_id)

        try:
            job = self._batch_v1.read_namespaced_job_status(job_name, self.namespace)
        except Exception as ex:
            logger.debug(f"Could not read Kubernetes Job status for {job_name}: {ex}")
            return WorkerJobState.FAILED

        # 1. Terminal states
        if job.status.succeeded:
            return WorkerJobState.SUCCEEDED

        if job.status.failed:
            # Check for deadline exceeded / timeout
            if job.status.conditions:
                for cond in job.status.conditions:
                    if getattr(cond, "reason", None) == "DeadlineExceeded":
                        return WorkerJobState.TIMED_OUT
            return WorkerJobState.FAILED

        # 2. Active states
        if job.status.active:
            # Inspect pods to distinguish SUBMITTED / QUEUED from RUNNING
            try:
                pods = self._core_v1.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=f"zmodel.ai/job-id={job_id}",
                )
                if pods and pods.items:
                    pod = pods.items[0]
                    phase = getattr(pod.status, "phase", "Pending")
                    if phase == "Running":
                        return WorkerJobState.RUNNING
                    if phase == "Pending":
                        return WorkerJobState.SUBMITTED
            except Exception:
                pass
            return WorkerJobState.RUNNING

        return WorkerJobState.SUBMITTED

    def get_job_result(self, job_id: str) -> Optional[WorkerJobResult]:
        """Retrieve structured result from pod logs or output if available."""
        self._ensure_clients()
        job_name = sanitize_k8s_name(job_id)
        current_status = self.get_job_status(job_id)

        # Extract pod logs to parse structured result if completed or failed
        result_json_str = None
        pod_name = None
        node_name = None

        try:
            pods = self._core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"zmodel.ai/job-id={job_id}",
            )
            if pods and pods.items:
                pod = pods.items[0]
                pod_name = pod.metadata.name
                node_name = getattr(pod.spec, "node_name", None)

                # Fetch log if container finished or is active
                logs = self._core_v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=self.namespace,
                    tail_lines=100,
                )
                if "--- WORKER_JOB_RESULT_START ---" in logs:
                    parts = logs.split("--- WORKER_JOB_RESULT_START ---")[1].split("--- WORKER_JOB_RESULT_END ---")
                    result_json_str = parts[0].strip()
        except Exception as ex:
            logger.debug(f"Could not retrieve pod logs for {job_id}: {ex}")

        if result_json_str:
            try:
                data = json.loads(result_json_str)
                return WorkerJobResult.from_dict(data)
            except Exception:
                pass

        # Fallback synthetic result from observed status
        return WorkerJobResult(
            job_id=job_id,
            status=current_status,
            model_fingerprint="unknown",
            pod_name=pod_name,
            worker_node=node_name,
            worker_image=self.worker_image,
        )

    def cancel_job(self, job_id: str) -> bool:
        """Delete Kubernetes Job with Foreground cascading deletion."""
        self._ensure_clients()
        job_name = sanitize_k8s_name(job_id)

        try:
            from kubernetes import client
            delete_opts = client.V1DeleteOptions(propagation_policy="Foreground")
            self._batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=delete_opts,
            )
            logger.info(f"Cancellation requested: deleted Kubernetes Job {job_name}")
            return True
        except Exception as ex:
            logger.warning(f"Could not delete Kubernetes Job {job_name} for cancellation: {ex}")
            return False

    def cleanup_job(self, job_id: str) -> bool:
        """Remove Kubernetes Job and associated Pods."""
        return self.cancel_job(job_id)

    def check_health(self) -> BackendHealth:
        """Inspect connectivity to the Kubernetes cluster."""
        try:
            self._ensure_clients()
            # Attempt read of namespace or API group
            self._batch_v1.get_api_resources()
            return BackendHealth(
                backend_name="kubernetes",
                is_available=True,
                details={
                    "namespace": self.namespace,
                    "worker_image": self.worker_image,
                },
            )
        except Exception as ex:
            return BackendHealth(
                backend_name="kubernetes",
                is_available=False,
                error_message=str(ex),
                details={"namespace": self.namespace},
            )
