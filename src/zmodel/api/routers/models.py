"""Model inspection and model registry REST endpoints."""

import os
from typing import Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from zmodel.api.config import get_api_settings
from zmodel.api.model_store import store_model_bytes
from zmodel.api.schemas import (
    ModelInspectionResponse,
    ModelListItem,
    ModelListResponse,
    ModelMetadataSchema,
    OpsetInfoSchema,
    TensorInfoSchema,
    WeightInfoSchema,
    InspectionWarningSchema,
)
from zmodel.core.config import SUPPORTED_FILE_EXTENSIONS
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.db.repositories import ModelRepository

router = APIRouter(prefix="/api/v1/models", tags=["Model Inspection & Registry"])


@router.post(
    "/inspect",
    response_model=ModelInspectionResponse,
    summary="Inspect ONNX Model Graph",
    description=(
        "Upload an ONNX model file for static graph inspection, tensor signature extraction, "
        "operator distribution analysis, SHA-256 fingerprinting, and structural validation. "
        "Does NOT execute inference or load weights onto hardware."
    ),
)
async def inspect_model(
    file: UploadFile = File(..., description="ONNX model binary (.onnx)"),
) -> ModelInspectionResponse:
    settings = get_api_settings()

    filename = file.filename or "model.onnx"
    _, ext = os.path.splitext(filename.lower())
    if ext not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Only {SUPPORTED_FILE_EXTENSIONS} files are accepted.",
        )

    # Read uploaded bytes with size validation
    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )
    if len(raw_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum upload size limit of {settings.max_upload_size_mb} MB.",
        )

    # Perform structural inspection using Phase 2 engine
    try:
        summary = load_and_inspect_model(raw_bytes, filename=filename)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model validation error: {str(ve)}",
        ) from ve
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to inspect model: {str(ex)}",
        ) from ex

    # Store bytes in memory cache for subsequent benchmark/optimization jobs
    store_model_bytes(summary.sha256_hash, raw_bytes)

    # Persist in Model Registry if database is reachable
    try:
        model_repo = ModelRepository()
        model_repo.register_or_update(summary)
    except Exception:
        pass  # Graceful tolerance if database is offline

    # Map to Pydantic response
    meta_schema = ModelMetadataSchema(
        producer_name=summary.metadata.producer_name,
        producer_version=summary.metadata.producer_version,
        model_version=summary.metadata.model_version,
        ir_version=summary.metadata.ir_version,
        doc_string=summary.metadata.doc_string,
        domain=summary.metadata.domain,
        custom_metadata=summary.metadata.custom_metadata,
    )

    opset_schemas = [
        OpsetInfoSchema(domain=op.domain or "ai.onnx", version=op.version)
        for op in summary.opsets
    ]

    input_schemas = [
        TensorInfoSchema(
            name=t.name,
            data_type=t.data_type,
            shape=t.shape,
            rank=t.rank,
            is_dynamic=t.is_dynamic,
            dynamic_dim_names=t.dynamic_dim_names,
            dimension_description=t.dimension_description,
        )
        for t in summary.inputs
    ]

    output_schemas = [
        TensorInfoSchema(
            name=t.name,
            data_type=t.data_type,
            shape=t.shape,
            rank=t.rank,
            is_dynamic=t.is_dynamic,
            dynamic_dim_names=t.dynamic_dim_names,
            dimension_description=t.dimension_description,
        )
        for t in summary.outputs
    ]

    weight_schemas = [
        WeightInfoSchema(
            name=w.name,
            data_type=w.data_type,
            shape=w.shape,
            num_elements=w.num_elements,
            size_bytes=w.size_bytes,
            has_external_data=w.has_external_data,
        )
        for w in summary.weights
    ]

    warning_schemas = [
        InspectionWarningSchema(
            code=w.code,
            level=w.level,
            title=w.title,
            message=w.message,
            impact=w.impact,
        )
        for w in summary.warnings
    ]

    return ModelInspectionResponse(
        filename=summary.filename,
        file_size_bytes=summary.file_size_bytes,
        sha256_fingerprint=summary.sha256_hash,
        ir_version=summary.metadata.ir_version,
        producer_name=summary.metadata.producer_name,
        producer_version=summary.metadata.producer_version,
        metadata=meta_schema,
        opset_imports=opset_schemas,
        inputs=input_schemas,
        outputs=output_schemas,
        weights=weight_schemas,
        total_nodes=summary.total_nodes,
        total_parameters=summary.total_parameters,
        total_weight_bytes=summary.total_weight_bytes,
        operator_counts=summary.op_counts,
        has_external_data=summary.has_external_data,
        external_data_tensors=summary.external_data_tensors,
        warnings=warning_schemas,
    )


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List Registered Models",
    description="Retrieve all previously inspected and persisted ONNX model identities from PostgreSQL.",
)
def list_models(
    limit: int = Query(50, ge=1, le=100, description="Page limit"),
    offset: int = Query(0, ge=0, description="Page offset"),
) -> ModelListResponse:
    try:
        repo = ModelRepository()
        models = repo.list_models(limit=limit, offset=offset)
        total = repo.count_models()
    except Exception:
        return ModelListResponse(models=[], total=0, limit=limit, offset=offset)

    items = [
        ModelListItem(
            fingerprint=m.fingerprint,
            original_filename=m.original_filename,
            file_size_bytes=m.file_size_bytes,
            ir_version=m.ir_version,
            producer_name=m.producer_name,
            first_seen_at=m.first_seen_at.isoformat() if m.first_seen_at else None,
            last_seen_at=m.last_seen_at.isoformat() if m.last_seen_at else None,
        )
        for m in models
    ]
    return ModelListResponse(models=items, total=total, limit=limit, offset=offset)


@router.get(
    "/{fingerprint}",
    response_model=ModelListItem,
    summary="Get Model Identity by Fingerprint",
    description="Retrieve metadata for a specific registered model by its SHA-256 fingerprint.",
)
def get_model(fingerprint: str) -> ModelListItem:
    clean_fp = fingerprint.strip().lower()
    try:
        repo = ModelRepository()
        m = repo.get_by_fingerprint(clean_fp)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query failed: {str(ex)}",
        ) from ex

    if not m:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with fingerprint '{clean_fp}' not found in registry.",
        )

    return ModelListItem(
        fingerprint=m.fingerprint,
        original_filename=m.original_filename,
        file_size_bytes=m.file_size_bytes,
        ir_version=m.ir_version,
        producer_name=m.producer_name,
        first_seen_at=m.first_seen_at.isoformat() if m.first_seen_at else None,
        last_seen_at=m.last_seen_at.isoformat() if m.last_seen_at else None,
    )
