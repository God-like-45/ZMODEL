"""Compatibility analysis and capability profile REST endpoints."""

from typing import List
from fastapi import APIRouter, HTTPException, status
from zmodel.api.model_store import resolve_model_bytes
from zmodel.api.schemas import (
    CompatibilityAnalyzeRequest,
    CompatibilityAnalyzeResponse,
    CompatibilityFindingSchema,
    ProfileSummaryResponse,
)
from zmodel.core.compatibility import CompatibilityAnalyzer
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.profiles import load_builtin_profiles
from zmodel.db.repositories import CompatibilityRepository

router = APIRouter(prefix="/api/v1/compatibility", tags=["Compatibility Analysis"])


@router.get(
    "/profiles",
    response_model=List[ProfileSummaryResponse],
    summary="List Capability Profiles",
    description="Retrieve all available target runtime and hardware capability profiles with ruleset versions.",
)
def list_profiles() -> List[ProfileSummaryResponse]:
    profiles = load_builtin_profiles()
    summaries: List[ProfileSummaryResponse] = []
    for pid, p in sorted(profiles.items()):
        summaries.append(
            ProfileSummaryResponse(
                profile_id=p.profile_id,
                name=p.display_name,
                version=p.version,
                ruleset_version=p.ruleset_version,
                target_device=f"{p.target_architecture} / {p.target_runtime}",
                description=f"Evidence source: {p.evidence_source or 'standard documentation'}. Verified: {p.verification_date or 'N/A'}.",
                is_illustrative=p.is_illustrative,
                supported_opset_range={"min_opset": p.min_opset, "max_opset": p.max_opset},
                operator_rule_count=len(p.operator_rules),
            )
        )
    return summaries


@router.get(
    "/profiles/{profile_id}",
    response_model=ProfileSummaryResponse,
    summary="Get Capability Profile Details",
    description="Retrieve capability details and rule summary for a specific target profile.",
)
def get_profile_detail(profile_id: str) -> ProfileSummaryResponse:
    profiles = load_builtin_profiles()
    if profile_id not in profiles:
        available = list(profiles.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{profile_id}' not found. Available profiles: {available}",
        )
    p = profiles[profile_id]
    return ProfileSummaryResponse(
        profile_id=p.profile_id,
        name=p.display_name,
        version=p.version,
        ruleset_version=p.ruleset_version,
        target_device=f"{p.target_architecture} / {p.target_runtime}",
        description=f"Evidence source: {p.evidence_source or 'standard documentation'}. Verified: {p.verification_date or 'N/A'}.",
        is_illustrative=p.is_illustrative,
        supported_opset_range={"min_opset": p.min_opset, "max_opset": p.max_opset},
        operator_rule_count=len(p.operator_rules),
    )


@router.post(
    "/analyze",
    response_model=CompatibilityAnalyzeResponse,
    summary="Analyze Model Compatibility",
    description=(
        "Evaluate an ONNX model against a target capability profile (e.g. local ONNX Runtime CPU, "
        "Linux on IBM Z, or IBM WMLA NNPA draft). Returns deterministic, evidence-based PASS, "
        "FAIL, WARNING, and UNKNOWN findings without claiming unverified hardware acceleration."
    ),
)
def analyze_compatibility(request: CompatibilityAnalyzeRequest) -> CompatibilityAnalyzeResponse:
    profiles = load_builtin_profiles()
    if request.profile_id not in profiles:
        available = list(profiles.keys())
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown profile ID '{request.profile_id}'. Available profiles: {available}",
        )
    profile = profiles[request.profile_id]

    # Resolve model bytes safely
    raw_bytes, filename = resolve_model_bytes(
        model_fingerprint=request.model_fingerprint,
        sample_model_name=request.sample_model_name,
    )

    try:
        model_summary = load_and_inspect_model(raw_bytes, filename=filename)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to inspect model for compatibility analysis: {str(ex)}",
        ) from ex

    # Execute deterministic compatibility analysis using Phase 3 engine
    report = CompatibilityAnalyzer.analyze(model_summary, profile)

    # Persist in DB if reachable
    assessment_id = f"compat_{report.model_fingerprint[:12]}_{report.profile_id}"
    try:
        repo = CompatibilityRepository()
        saved = repo.save_assessment(report)
        assessment_id = saved.id
    except Exception:
        pass  # Graceful fallback if database is not running

    # Map findings
    finding_schemas = [
        CompatibilityFindingSchema(
            finding_id=f.finding_id,
            category=f.category,
            status=f.status,
            severity=f.severity,
            feature=f.feature,
            observed_value=f.observed_value,
            profile_rule=f.profile_rule,
            explanation=f.explanation,
            evidence_source=f.evidence_source,
            recommended_action=f.recommended_action,
            domain=f.domain,
            op_type=f.op_type,
        )
        for f in report.findings
    ]

    return CompatibilityAnalyzeResponse(
        assessment_id=assessment_id,
        model_name=report.model_name,
        model_fingerprint=report.model_fingerprint,
        profile_id=report.profile_id,
        profile_name=report.profile_name,
        profile_version=report.profile_version,
        ruleset_version=report.ruleset_version,
        overall_assessment=report.overall_assessment,
        summary_counts=report.summary_counts,
        findings=finding_schemas,
        assumptions=report.assumptions or [],
        unresolved_questions=report.unresolved_questions or [],
        recommended_next_steps=report.recommended_next_steps or [],
        is_illustrative_profile=report.is_illustrative_profile,
        timestamp_utc=report.analysis_timestamp,
    )
