"""System environment and runtime diagnostics.

Collects real, verifiable runtime attributes without fabricating metrics.
In Phase 1, target deployment architectures are strictly marked as 'Not verified'.
"""

import platform
import sys
from dataclasses import dataclass
from typing import List, Optional
import psutil

try:
    import onnx
    ONNX_VERSION = onnx.__version__
    ONNX_IR_VERSION = onnx.IR_VERSION
except ImportError:
    ONNX_VERSION = None
    ONNX_IR_VERSION = None

try:
    import onnxruntime as ort
    ORT_VERSION = ort.__version__
    ORT_PROVIDERS = ort.get_available_providers()
except ImportError:
    ORT_VERSION = None
    ORT_PROVIDERS = []


@dataclass
class SystemEnvironment:
    os_name: str
    os_release: str
    architecture: str
    python_version: str
    python_executable: str
    cpu_count_physical: Optional[int]
    cpu_count_logical: Optional[int]
    memory_total_gb: float
    memory_available_gb: float
    onnx_version: Optional[str]
    onnx_ir_version: Optional[int]
    onnxruntime_version: Optional[str]
    host_providers: List[str]
    # Deployment verification flags
    host_runtime_verified: bool
    target_hardware_verified: bool
    target_architecture_status: str


def get_system_environment() -> SystemEnvironment:
    """Retrieve verified environment attributes for the local host."""
    mem = psutil.virtual_memory()
    runtime_ok = bool(ONNX_VERSION and ORT_VERSION)
    
    return SystemEnvironment(
        os_name=platform.system(),
        os_release=platform.release(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        python_executable=sys.executable,
        cpu_count_physical=psutil.cpu_count(logical=False),
        cpu_count_logical=psutil.cpu_count(logical=True),
        memory_total_gb=round(mem.total / (1024 ** 3), 2),
        memory_available_gb=round(mem.available / (1024 ** 3), 2),
        onnx_version=ONNX_VERSION,
        onnx_ir_version=ONNX_IR_VERSION,
        onnxruntime_version=ORT_VERSION,
        host_providers=ORT_PROVIDERS,
        host_runtime_verified=runtime_ok,
        target_hardware_verified=False,
        target_architecture_status="Not verified (Phase 3)",
    )

