"""Centralized configuration and environment settings for Z-ModelPilot FastAPI REST API."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from zmodel.core.config import MAX_UPLOAD_SIZE_MB, MAX_UPLOAD_SIZE_BYTES


@dataclass
class ApiSettings:
    """Configurable settings for the FastAPI REST API layer."""

    api_v1_prefix: str = "/api/v1"
    api_title: str = "Z-ModelPilot REST API"
    api_description: str = (
        "Enterprise-grade, architecture-aware AI model inspection, compatibility analysis, "
        "reproducible benchmarking, and multi-objective optimization API."
    )
    api_version: str = "0.7.0"
    api_host: str = os.environ.get("API_HOST", "0.0.0.0")
    api_port: int = int(os.environ.get("API_PORT", 8000))
    api_reload: bool = os.environ.get("API_RELOAD", "false").lower() in ("true", "1", "yes")
    environment: str = os.environ.get("ENV", os.environ.get("ENVIRONMENT", "development"))

    # Security & Resource Limits
    max_upload_size_mb: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", MAX_UPLOAD_SIZE_MB))
    max_upload_size_bytes: int = int(os.environ.get("MAX_UPLOAD_SIZE_BYTES", MAX_UPLOAD_SIZE_BYTES))
    max_benchmark_iterations: int = int(os.environ.get("MAX_BENCHMARK_ITERATIONS", 1000))
    max_benchmark_warmup: int = int(os.environ.get("MAX_BENCHMARK_WARMUP", 100))
    max_candidate_count: int = int(os.environ.get("MAX_CANDIDATE_COUNT", 50))
    max_optimization_candidates: int = int(os.environ.get("MAX_CANDIDATE_COUNT", 50))

    # Job Concurrency & Queue Limits
    job_concurrency: int = int(os.environ.get("JOB_CONCURRENCY", 2))
    max_queued_jobs: int = int(os.environ.get("MAX_QUEUED_JOBS", 20))

    # CORS
    cors_allowed_origins: List[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
            if s.strip()
        ]
    )
    allowed_cors_origins: str = os.environ.get("CORS_ALLOWED_ORIGINS", "*")

    # Temporary directory
    temp_dir: Optional[str] = os.environ.get("TEMP_DIR", None)

    # Logging level
    log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper()


_SETTINGS: Optional[ApiSettings] = None


def get_api_settings() -> ApiSettings:
    """Retrieve or initialize singleton API settings."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = ApiSettings()
    return _SETTINGS
