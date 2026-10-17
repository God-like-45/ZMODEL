"""FastAPI Application Factory for Z-ModelPilot."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from zmodel.api.config import get_api_settings
from zmodel.api.jobs import get_job_manager, sanitize_error_message
from zmodel.api.routers import (
    benchmarks,
    compatibility,
    experiments,
    graph_optimizations,
    health,
    jobs,
    models,
    optimizations,
    validation,
)
from zmodel.api.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger("zmodel.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager for startup and shutdown routines."""
    settings = get_api_settings()
    logger.info(f"Starting Z-ModelPilot REST API v{settings.api_version} (env: {settings.environment})")
    yield
    logger.info("Shutting down Z-ModelPilot REST API background job manager...")
    get_job_manager().shutdown(wait=False)
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    settings = get_api_settings()

    app = FastAPI(
        title="Z-ModelPilot REST API",
        description=(
            "Architecture-Aware AI Deployment Optimizer REST API aligned with IBM AI on Z. "
            "Exposes model inspection, empirical onnxruntime benchmarking, compatibility ruleset "
            "evaluations, multi-objective optimization, and persistent PostgreSQL experiment history."
        ),
        version=settings.api_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Configure CORS
    allowed_origins = [o.strip() for o in settings.allowed_cors_origins.split(",") if o.strip()]
    if not allowed_origins:
        allowed_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = [
            ErrorDetail(
                field=" -> ".join(str(loc) for loc in err.get("loc", [])),
                message=err.get("msg", "Validation error"),
            )
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="VALIDATION_ERROR",
                message="Request payload failed structural or type validation.",
                details=details,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled server exception on {request.method} {request.url.path}: {exc}")
        safe_msg = sanitize_error_message(exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="INTERNAL_SERVER_ERROR",
                message=f"An unexpected server error occurred: {safe_msg}",
            ).model_dump(),
        )

    # Register Routers
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(compatibility.router)
    app.include_router(benchmarks.router)
    app.include_router(optimizations.router)
    app.include_router(graph_optimizations.router)
    app.include_router(jobs.router)
    app.include_router(experiments.router)
    app.include_router(validation.router)

    return app


# Default application instance for ASGI servers and test clients
app = create_app()

