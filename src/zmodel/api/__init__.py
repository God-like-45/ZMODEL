"""Z-ModelPilot REST API Package (Phase 7: Production Hardening, FastAPI, Docker & CI/CD)."""

from zmodel.api.config import get_api_settings
from zmodel.api.app import create_app

__all__ = ["create_app", "get_api_settings"]
