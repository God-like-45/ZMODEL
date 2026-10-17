"""Health and readiness endpoints for container orchestrators and monitoring systems."""

import time
from fastapi import APIRouter, Response, status
from zmodel.api.config import get_api_settings
from zmodel.api.schemas import LiveResponse, ReadyResponse
from zmodel.db.config import check_database_connection

router = APIRouter(tags=["Health & System Diagnostics"])
_PROCESS_START_TIME = time.time()


@router.get(
    "/health/live",
    response_model=LiveResponse,
    summary="Process Liveness Probe",
    description="Indicates whether the FastAPI application process is running and able to respond to HTTP requests.",
)
def get_liveness() -> LiveResponse:
    settings = get_api_settings()
    return LiveResponse(
        status="alive",
        api_version=settings.api_version,
    )


@router.get(
    "/health/ready",
    response_model=ReadyResponse,
    summary="Readiness Probe",
    description=(
        "Checks underlying operational dependencies required for service execution, "
        "including database connectivity. Returns HTTP 200 when ready, or HTTP 503 when dependencies are unreachable."
    ),
)
def get_readiness(response: Response) -> ReadyResponse:
    db_status = check_database_connection(timeout_seconds=1)
    is_connected = db_status.get("connected", False)

    if not is_connected:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        readiness_status = "not_ready"
    else:
        readiness_status = "ready"

    return ReadyResponse(
        status=readiness_status,
        database_connected=is_connected,
        dialect=db_status.get("dialect", "unknown"),
        database_url_masked=db_status.get("database_url_masked", "Not Configured"),
        table_count=db_status.get("table_count", 0),
        tables=db_status.get("tables", []),
        error=db_status.get("error"),
    )
