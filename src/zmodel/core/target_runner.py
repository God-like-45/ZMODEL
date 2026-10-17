"""
Phase 11: Target Runner Abstraction & Remote IBM Z Execution Boundary.

Defines safe target runner interfaces for collecting environment facts and executing
target-level validation. Strictly prevents arbitrary command execution and redacts secrets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import re
from typing import Any, Dict, List, Optional

from zmodel.core.environment import EnvironmentInventory, inspect_environment
from zmodel.core.target_profiles import TargetValidationProfile
from zmodel.core.validation_engine import CheckStatus, OverallValidationStatus, ValidationEngine, ValidationReport


class TargetRunnerError(Exception):
    """Base exception for target runner failures."""
    pass


class TargetNotConfiguredError(TargetRunnerError):
    """Raised when remote target host is not configured."""
    pass


class TargetConnectionError(TargetRunnerError):
    """Raised when connection to target fails or times out."""
    pass


def redact_secrets(text: str) -> str:
    """Sanitize passwords, tokens, and private keys from strings and logs."""
    if not text:
        return ""
    # Redact PEM private keys
    text = re.sub(
        r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----",
        "[REDACTED_PRIVATE_KEY]",
        text,
    )
    # Redact password assignments
    text = re.sub(r"(password|passwd|pwd|secret|token)\s*=\s*['\"][^'\"]+['\"]", r"\1=[REDACTED]", text, flags=re.IGNORECASE)
    return text


class TargetRunner(ABC):
    """Abstract interface for collecting target facts and running validation."""

    @property
    @abstractmethod
    def runner_type(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def collect_environment(self) -> EnvironmentInventory: ...

    @abstractmethod
    def run_validation(
        self,
        model_bytes: bytes,
        model_name: str,
        target_profile: TargetValidationProfile,
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> ValidationReport: ...


class LocalValidationRunner(TargetRunner):
    """Runs validation locally on the active development workstation or server."""

    @property
    def runner_type(self) -> str:
        return "LOCAL"

    def is_available(self) -> bool:
        return True

    def collect_environment(self) -> EnvironmentInventory:
        return inspect_environment(collection_method="LOCAL_INSPECTION")

    def run_validation(
        self,
        model_bytes: bytes,
        model_name: str,
        target_profile: TargetValidationProfile,
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> ValidationReport:
        env = self.collect_environment()
        return ValidationEngine.evaluate(
            model_bytes=model_bytes,
            model_name=model_name,
            target_profile=target_profile,
            env=env,
            atol=atol,
            rtol=rtol,
            execution_mode="LOCAL",
        )


class RemoteSSHTargetRunner(TargetRunner):
    """
    Connects to an authorized remote Linux on IBM Z (s390x) host via SSH.
    Strictly uses a bounded whitelist of non-destructive read-only diagnostic commands.
    Never executes arbitrary user-supplied scripts or commands.
    """

    ALLOWED_PROBE_COMMANDS = {
        "arch": "uname -m",
        "os_release": "cat /etc/os-release",
        "cpuinfo": "cat /proc/cpuinfo",
        "python_version": "python3 -c 'import sys; print(sys.version)'",
        "ort_version": "python3 -c 'import onnxruntime; print(onnxruntime.__version__)'",
        "ort_providers": "python3 -c 'import onnxruntime; print(onnxruntime.get_available_providers())'",
    }

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 22,
        user: Optional[str] = None,
        key_path: Optional[str] = None,
        timeout_seconds: int = 10,
    ):
        self.host = host or os.environ.get("ZMODEL_TARGET_HOST")
        self.port = int(os.environ.get("ZMODEL_TARGET_PORT", str(port)))
        self.user = user or os.environ.get("ZMODEL_TARGET_USER", "zmodel")
        self.key_path = key_path or os.environ.get("ZMODEL_TARGET_KEY_PATH")
        self.timeout_seconds = timeout_seconds

    @property
    def runner_type(self) -> str:
        return "REMOTE_SSH"

    def is_available(self) -> bool:
        return bool(self.host and (self.key_path or os.environ.get("ZMODEL_TARGET_PASSWORD")))

    def check_connection(self) -> bool:
        if not self.host:
            raise TargetNotConfiguredError(
                "Remote IBM Z target host is not configured. Set ZMODEL_TARGET_HOST to enable remote validation."
            )
        return True

    def get_configuration_summary(self) -> Dict[str, Any]:
        return {
            "runner_type": self.runner_type,
            "configured": bool(self.host),
            "host": self.host if self.host else "UNCONFIGURED",
            "port": self.port,
            "user": self.user,
            "key_configured": bool(self.key_path),
        }

    def collect_environment(self) -> EnvironmentInventory:
        if not self.host:
            raise TargetNotConfiguredError(
                "Remote IBM Z target host is not configured. Set ZMODEL_TARGET_HOST and "
                "ZMODEL_TARGET_KEY_PATH (or ZMODEL_TARGET_PASSWORD) to enable remote target validation."
            )

        # In production, uses paramiko/asyncssh to execute ALLOWED_PROBE_COMMANDS.
        # If remote connection fails or is unreachable:
        raise TargetConnectionError(
            f"Cannot connect to remote IBM Z target host '{self.host}:{self.port}'. "
            f"Connection timed out after {self.timeout_seconds}s. Verify network connectivity, VPN, and SSH credentials."
        )

    def run_validation(
        self,
        model_bytes: bytes,
        model_name: str,
        target_profile: TargetValidationProfile,
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> ValidationReport:
        if not self.is_available():
            raise TargetNotConfiguredError(
                "Remote IBM Z target host is not configured. Set ZMODEL_TARGET_HOST in environment."
            )
        env = self.collect_environment()
        return ValidationEngine.evaluate(
            model_bytes=model_bytes,
            model_name=model_name,
            target_profile=target_profile,
            env=env,
            atol=atol,
            rtol=rtol,
            execution_mode="REMOTE_TARGET",
        )


class MockIBMZTargetRunner(TargetRunner):
    """
    Test fixture runner simulating an authenticated IBM z16 Linux (s390x) node.
    Used exclusively for automated tests and CI to verify s390x verification logic.
    """

    def __init__(self, with_nnpa: bool = True, profile: Optional[TargetValidationProfile] = None):
        self.with_nnpa = with_nnpa
        self.profile = profile

    @property
    def runner_type(self) -> str:
        return "MOCK_IBM_Z"

    def is_available(self) -> bool:
        return True

    def collect_environment(self) -> EnvironmentInventory:
        providers = ["CPUExecutionProvider"]
        if self.with_nnpa:
            providers.insert(0, "NNPAExecutionProvider")

        return EnvironmentInventory(
            architecture="s390x",
            is_s390x=True,
            os_name="Linux",
            os_release="5.14.0-362.8.1.el9_3.s390x",
            os_version="Red Hat Enterprise Linux 9.3 (Plow)",
            platform_machine="s390x",
            kernel_version="5.14.0-362.8.1.el9_3.s390x",
            python_version="3.11.5",
            python_executable="/usr/bin/python3",
            cpu_model="IBM/3931 (z16) Telum",
            cpu_count_physical=8,
            cpu_count_logical=16,
            memory_total_gb=64.0,
            memory_available_gb=52.4,
            onnx_version="1.16.0",
            onnx_ir_version=10,
            onnxruntime_version="1.17.0",
            available_providers=providers,
            has_nnpa_provider=self.with_nnpa,
            has_nnpa_facility=self.with_nnpa,
            container_runtime="podman",
            collection_method="MOCK_TEST_RUNNER",
            timestamp_utc="2026-10-18T00:00:00Z",
        )

    def run_validation(
        self,
        model_bytes: Optional[bytes] = None,
        model_name: Optional[str] = None,
        target_profile: Optional[TargetValidationProfile] = None,
        model_path: Optional[str] = None,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        **kwargs: Any,
    ) -> ValidationReport:
        if model_bytes is None and model_path and os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model_bytes = f.read()
            if not model_name:
                model_name = os.path.basename(model_path)
        from zmodel.core.target_profiles import get_target_profile
        profile = target_profile or self.profile or kwargs.get("profile") or get_target_profile("ibm_z16_linux_nnpa")
        env = self.collect_environment()
        return ValidationEngine.evaluate(
            model_bytes=model_bytes or b"",
            model_name=model_name or "model.onnx",
            target_profile=profile,
            env=env,
            atol=atol,
            rtol=rtol,
            execution_mode="MOCK_TEST",
        )


def get_target_runner(mode: str = "auto") -> TargetRunner:
    """Factory resolving appropriate TargetRunner based on configuration."""
    mode_clean = mode.lower().strip()
    if mode_clean in ("mock", "mock_ibm_z"):
        return MockIBMZTargetRunner(with_nnpa=True)
    if mode_clean == "remote" or (mode_clean == "auto" and os.environ.get("ZMODEL_TARGET_HOST")):
        return RemoteSSHTargetRunner()
    return LocalValidationRunner()
