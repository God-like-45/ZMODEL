"""Application configuration and phase definitions."""

from dataclasses import dataclass
from typing import List

APP_NAME = "Z-ModelPilot"
APP_SUBTITLE = "Architecture-Aware AI Model Deployment Analysis"
APP_VERSION = "1.4.0"
CURRENT_PHASE = 12

# Environment and workspace configuration
ENVIRONMENT_TYPE = "Production-hardened release"
SUPPORTED_FILE_EXTENSIONS = [".onnx"]
MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


@dataclass(frozen=True)
class NavItem:
    id: str
    label: str
    phase: int
    is_active: bool
    status_label: str
    description: str


NAV_ITEMS: List[NavItem] = [
    NavItem(
        id="overview",
        label="Overview",
        phase=1,
        is_active=True,
        status_label="Implemented",
        description="Model deployment workspace, upload area, and workspace information.",
    ),
    NavItem(
        id="inspector",
        label="Model Inspector",
        phase=2,
        is_active=True,
        status_label="Implemented",
        description="Comprehensive ONNX computational graph breakdown, tensor signatures, operator topology, and metadata audit.",
    ),
    NavItem(
        id="compatibility",
        label="Compatibility",
        phase=3,
        is_active=True,
        status_label="Implemented",
        description="Target runtime and hardware capability profiling with operator coverage, opset rules, and explainable findings.",
    ),
    NavItem(
        id="benchmarks",
        label="Benchmarks",
        phase=4,
        is_active=True,
        status_label="Implemented",
        description="Empirical latency percentiles, throughput, correctness validation, and memory profiling.",
    ),
    NavItem(
        id="optimization",
        label="Optimization",
        phase=12,
        is_active=True,
        status_label="Implemented",
        description="Advanced multi-objective Pareto optimization, bounded search strategies, sensitivity analysis, and enterprise reporting.",
    ),
    NavItem(
        id="graph_optimization",
        label="Graph Optimization",
        phase=9,
        is_active=True,
        status_label="Implemented",
        description="ONNX graph-level transformation, constant folding, node fusion, shape inference, and numerical equivalence verification.",
    ),
    NavItem(
        id="workers",
        label="Worker Jobs",
        phase=10,
        is_active=True,
        status_label="Implemented",
        description="Kubernetes worker job orchestration, resource profiles, lifecycle monitoring, and scalable benchmark execution.",
    ),
    NavItem(
        id="validation",
        label="IBM Z Validation",
        phase=11,
        is_active=True,
        status_label="Implemented",
        description="Target environment facts, profile verification, 12-check validation engine, and hardware provenance.",
    ),
    NavItem(
        id="history",
        label="Experiment History",
        phase=6,
        is_active=True,
        status_label="Implemented",
        description="Durable PostgreSQL experiment tracking, reproducible run diffing, and model registry.",
    ),
    NavItem(
        id="settings",
        label="Settings & API",
        phase=7,
        is_active=True,
        status_label="Implemented",
        description="FastAPI REST API endpoints, OpenAPI documentation, execution bounds, and architecture boundaries.",
    ),
]
