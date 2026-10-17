"""
Phase 11: FastAPI Router for IBM Z / Linux on IBM Z Target Validation.

Provides endpoints for inspecting environment facts, target profile constraints,
running evidence-backed validation evaluations, and retrieving audit histories.
"""

from __future__ import annotations

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from zmodel.api.routers.benchmarks import resolve_model_bytes
from zmodel.api.schemas import (
    EnvironmentInventoryResponse,
    TargetValidationProfileResponse,
    ValidationEvaluateRequest,
    ValidationReportResponse,
    ValidationRunListResponse,
)
from zmodel.core.environment import inspect_environment
from zmodel.core.target_profiles import get_target_profile, load_target_profiles
from zmodel.core.target_runner import get_target_runner
from zmodel.core.validation_engine import ValidationReport
from zmodel.db.repositories import ValidationRepository
from zmodel.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/validation", tags=["IBM Z Validation & Deployment Readiness"])


@router.get(
    "/environment",
    response_model=EnvironmentInventoryResponse,
    summary="Inspect Environment Facts",
    description="Query verified, read-only system information, architecture (x86_64 vs s390x), and runtime providers.",
)
def get_environment_info() -> EnvironmentInventoryResponse:
    env = inspect_environment(collection_method="LOCAL_INSPECTION")
    return EnvironmentInventoryResponse(**env.to_dict())


@router.get(
    "/profiles",
    response_model=List[TargetValidationProfileResponse],
    summary="List Target Validation Profiles",
    description="List all versioned target architecture profiles, including IBM z16 with NNPA and Linux on IBM Z baseline.",
)
def list_target_profiles() -> List[TargetValidationProfileResponse]:
    profiles = load_target_profiles()
    return [TargetValidationProfileResponse(**p.to_dict()) for p in profiles.values()]


@router.get(
    "/profiles/{profile_id}",
    response_model=TargetValidationProfileResponse,
    summary="Get Target Validation Profile Details",
    description="Retrieve requirements, opset constraints, and evidence sources for a specific target profile.",
)
def get_profile_detail(profile_id: str) -> TargetValidationProfileResponse:
    try:
        profile = get_target_profile(profile_id)
        return TargetValidationProfileResponse(**profile.to_dict())
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ex),
        )


@router.post(
    "/evaluate",
    response_model=ValidationReportResponse,
    summary="Run Target Architecture Validation",
    description="Evaluate an ONNX model against a target validation profile across the 12 architecture, runtime, and correctness checks.",
)
def evaluate_target_validation(
    request: ValidationEvaluateRequest,
    db=Depends(get_db),
) -> ValidationReportResponse:
    # 1. Resolve Target Profile
    try:
        target_profile = get_target_profile(request.target_profile_id)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target profile error: {ex}",
        )

    # 2. Resolve Model Bytes
    try:
        model_bytes, filename = resolve_model_bytes(
            sample_model_name=request.sample_model_name,
            model_fingerprint=request.model_fingerprint,
        )
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to resolve model: {ex}",
        )

    # 3. Select Target Runner
    runner = get_target_runner(mode=request.execution_mode)

    # 4. Execute Validation Engine
    try:
        report = runner.run_validation(
            model_bytes=model_bytes,
            model_name=filename,
            target_profile=target_profile,
            atol=request.atol,
            rtol=request.rtol,
        )
    except Exception as ex:
        logger.exception(f"Target validation execution error: {ex}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation execution failed: {ex}",
        )

    # 5. Persist to PostgreSQL
    try:
        repo = ValidationRepository(session=db)
        repo.save_validation_report(report)
    except Exception as ex:
        logger.warning(f"Failed to persist validation report to database: {ex}")

    return ValidationReportResponse(**report.to_dict())


@router.get(
    "/runs",
    response_model=ValidationRunListResponse,
    summary="List Validation Run History",
    description="Retrieve paginated historical validation reports with optional filtering by status or profile.",
)
def list_validation_runs(
    model_fingerprint: Optional[str] = Query(None, description="Filter by model SHA-256 fingerprint"),
    target_profile_id: Optional[str] = Query(None, description="Filter by target profile ID"),
    overall_status: Optional[str] = Query(None, description="Filter by overall validation status"),
    limit: int = Query(50, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
    db=Depends(get_db),
) -> ValidationRunListResponse:
    repo = ValidationRepository(session=db)
    records, total = repo.list_validation_runs(
        model_fingerprint=model_fingerprint,
        target_profile_id=target_profile_id,
        overall_status=overall_status,
        limit=limit,
        offset=offset,
    )

    reports = []
    for r in records:
        reports.append(
            ValidationReportResponse(
                run_id=r.id,
                model_fingerprint=r.model_fingerprint,
                model_name="model.onnx",
                target_profile_id=r.target_profile_id,
                target_profile_version=r.target_profile_version,
                environment=EnvironmentInventoryResponse(**r.environment_info),
                overall_status=r.overall_status,
                status_rationale=r.status_rationale or "",
                checks=r.checks_json,
                summary_counts=r.summary_counts,
                numerical_correctness=r.numerical_correctness or {},
                benchmark_eligibility=r.benchmark_eligibility,
                execution_mode=r.execution_mode,
                timestamp_utc=r.created_at.isoformat() if r.created_at else "",
                provenance_badge=r.provenance_badge or "",
            )
        )

    return ValidationRunListResponse(
        runs=reports,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/runs/{run_id}",
    response_model=ValidationReportResponse,
    summary="Get Validation Run Detail",
    description="Retrieve complete audit record, check results, and diagnostic evidence for a specific validation run.",
)
def get_validation_run_detail(run_id: str, db=Depends(get_db)) -> ValidationReportResponse:
    repo = ValidationRepository(session=db)
    r = repo.get_validation_run(run_id)
    if not r:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Validation run '{run_id}' not found.",
        )

    return ValidationReportResponse(
        run_id=r.id,
        model_fingerprint=r.model_fingerprint,
        model_name="model.onnx",
        target_profile_id=r.target_profile_id,
        target_profile_version=r.target_profile_version,
        environment=EnvironmentInventoryResponse(**r.environment_info),
        overall_status=r.overall_status,
        status_rationale=r.status_rationale or "",
        checks=r.checks_json,
        summary_counts=r.summary_counts,
        numerical_correctness=r.numerical_correctness or {},
        benchmark_eligibility=r.benchmark_eligibility,
        execution_mode=r.execution_mode,
        timestamp_utc=r.created_at.isoformat() if r.created_at else "",
        provenance_badge=r.provenance_badge or "",
    )
