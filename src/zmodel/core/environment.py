"""
Phase 11: IBM Z / Linux on IBM Z Environment Inventory & Diagnostics.

Safely captures verified, read-only system environment facts without fabricating
hardware metrics or assuming IBM Z compatibility based on partial clues.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import psutil
import sys
from typing import Any, Dict, List, Optional

try:
    import onnx
    ONNX_VERSION: Optional[str] = onnx.__version__
    ONNX_IR_VERSION: Optional[int] = onnx.IR_VERSION
except ImportError:
    ONNX_VERSION = None
    ONNX_IR_VERSION = None

try:
    import onnxruntime as ort
    ORT_VERSION: Optional[str] = ort.__version__
    ORT_PROVIDERS: List[str] = ort.get_available_providers()
except ImportError:
    ORT_VERSION = None
    ORT_PROVIDERS = []


@dataclass
class EnvironmentInventory:
    """Verified, read-only snapshot of a host or target validation environment."""
    architecture: str = "UNKNOWN"
    is_s390x: bool = False
    os_name: str = "UNKNOWN"
    os_release: str = "UNKNOWN"
    os_version: str = "UNKNOWN"
    platform_machine: str = "UNKNOWN"
    kernel_version: str = "UNKNOWN"
    python_version: str = "UNKNOWN"
    python_executable: str = "UNKNOWN"
    cpu_model: str = "UNKNOWN"
    cpu_count_physical: Optional[int] = None
    cpu_count_logical: Optional[int] = None
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    onnx_version: Optional[str] = None
    onnx_ir_version: Optional[int] = None
    onnxruntime_version: Optional[str] = None
    available_providers: List[str] = field(default_factory=list)
    has_nnpa_provider: bool = False
    has_nnpa_facility: bool = False
    container_runtime: str = "none"
    collection_method: str = "LOCAL_INSPECTION"
    timestamp_utc: str = ""
    environment_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.architecture == "s390x":
            self.is_s390x = True
        if not self.timestamp_utc:
            self.timestamp_utc = datetime.now(timezone.utc).isoformat()
        if not self.environment_fingerprint:
            # Deterministic hash of immutable machine facts
            facts = (
                f"{self.architecture}|{self.os_name}|{self.os_release}|"
                f"{self.python_version}|{self.onnxruntime_version}|"
                f"{','.join(sorted(self.available_providers))}|{self.has_nnpa_facility}"
            )
            self.environment_fingerprint = hashlib.sha256(facts.encode("utf-8")).hexdigest()

    @property
    def canonical_fingerprint(self) -> str:
        return self.environment_fingerprint

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentInventory:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def detect_container_runtime() -> str:
    """Safely detect if running inside Docker, Podman, or Kubernetes container."""
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    if Path("/run/.containerenv").exists():
        return "podman"
    if Path("/.dockerenv").exists():
        return "docker"
    try:
        cgroup_path = Path("/proc/1/cgroup")
        if cgroup_path.exists():
            content = cgroup_path.read_text(encoding="utf-8", errors="ignore")
            if "docker" in content:
                return "docker"
            if "kubepods" in content:
                return "kubernetes"
            if "containerd" in content:
                return "containerd"
    except Exception:
        pass
    return "none"


def detect_s390x_facilities() -> bool:
    """
    On Linux on IBM Z (s390x), inspect /proc/cpuinfo for facility 193 (NNPA).
    Returns False on non-s390x architectures or if facilities are inaccessible.
    """
    if platform.machine().lower() not in ("s390x", "s390"):
        return False

    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return False

    try:
        content = cpuinfo_path.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            line_clean = line.strip().lower()
            if line_clean.startswith("facilities"):
                tokens = line_clean.split(":", 1)[-1].strip().split()
                # Facility 193 indicates Neural Network Processing Assist (NNPA) on IBM z16
                if "193" in tokens or "nnpa" in tokens:
                    return True
    except Exception:
        pass
    return False


def get_cpu_model_name() -> str:
    """Safely query CPU brand or processor family."""
    if platform.system() == "Windows":
        return platform.processor() or "x86_64 Family"

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            content = cpuinfo.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip().lower()
                    if k in ("model name", "cpu model", "machine"):
                        return v.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "UNKNOWN"


def inspect_environment(collection_method: str = "LOCAL_INSPECTION") -> EnvironmentInventory:
    """
    Collect verified environment facts from active runtime.
    Missing, inaccessible, or unsupported attributes are marked UNKNOWN rather than fabricated.
    """
    mach = platform.machine() or "UNKNOWN"
    is_s390x = mach.lower() in ("s390x", "s390")

    mem = psutil.virtual_memory()
    total_mem = round(mem.total / (1024 ** 3), 2)
    avail_mem = round(mem.available / (1024 ** 3), 2)

    has_nnpa_prov = any("nnpa" in p.lower() or "ibmz" in p.lower() for p in ORT_PROVIDERS)
    has_nnpa_fac = detect_s390x_facilities()
    container_rt = detect_container_runtime()

    return EnvironmentInventory(
        architecture=mach,
        is_s390x=is_s390x,
        os_name=platform.system() or "UNKNOWN",
        os_release=platform.release() or "UNKNOWN",
        os_version=platform.version() or "UNKNOWN",
        platform_machine=mach,
        kernel_version=platform.uname().release or "UNKNOWN",
        python_version=platform.python_version() or "UNKNOWN",
        python_executable=sys.executable or "UNKNOWN",
        cpu_model=get_cpu_model_name(),
        cpu_count_physical=psutil.cpu_count(logical=False),
        cpu_count_logical=psutil.cpu_count(logical=True),
        memory_total_gb=total_mem,
        memory_available_gb=avail_mem,
        onnx_version=ONNX_VERSION,
        onnx_ir_version=ONNX_IR_VERSION,
        onnxruntime_version=ORT_VERSION,
        available_providers=list(ORT_PROVIDERS),
        has_nnpa_provider=has_nnpa_prov,
        has_nnpa_facility=has_nnpa_fac,
        container_runtime=container_rt,
        collection_method=collection_method,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
