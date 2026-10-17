"""Secure model bytes resolution and memory caching for API requests."""

import os
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, UploadFile, status
from zmodel.core.config import SUPPORTED_FILE_EXTENSIONS
from zmodel.api.config import get_api_settings

# Bounded in-memory model cache mapping SHA-256 fingerprint -> raw ONNX bytes
_MODEL_CACHE: Dict[str, bytes] = {}
_MAX_CACHE_ENTRIES = 20


def store_model_bytes(fingerprint: str, raw_bytes: bytes) -> None:
    """Store raw ONNX bytes in memory cache, evicting oldest if capacity reached."""
    if len(_MODEL_CACHE) >= _MAX_CACHE_ENTRIES and fingerprint not in _MODEL_CACHE:
        # Evict first key
        oldest_key = next(iter(_MODEL_CACHE))
        del _MODEL_CACHE[oldest_key]
    _MODEL_CACHE[fingerprint] = raw_bytes


def get_model_bytes(fingerprint: str) -> Optional[bytes]:
    """Retrieve raw ONNX bytes by SHA-256 fingerprint if present in cache."""
    return _MODEL_CACHE.get(fingerprint)


def clear_model_cache() -> None:
    """Clear in-memory model cache (useful for testing)."""
    _MODEL_CACHE.clear()


def resolve_model_bytes(
    model_fingerprint: Optional[str] = None,
    sample_model_name: Optional[str] = None,
    upload_file: Optional[UploadFile] = None,
) -> Tuple[bytes, str]:
    """
    Safely resolve ONNX model bytes and filename without exposing arbitrary filesystem access.

    Resolution order:
    1. Direct upload file (if provided)
    2. Sample model name from allowed sample_models directory
    3. Cached model bytes by SHA-256 fingerprint
    """
    settings = get_api_settings()

    # 1. Direct Upload
    if upload_file is not None and upload_file.filename:
        filename = upload_file.filename
        _, ext = os.path.splitext(filename.lower())
        if ext not in SUPPORTED_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{ext}'. Only {SUPPORTED_FILE_EXTENSIONS} files are supported.",
            )

        raw_bytes = upload_file.file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded model file is empty (0 bytes).",
            )
        if len(raw_bytes) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum upload size of {settings.max_upload_size_mb} MB.",
            )

        import hashlib
        fp = hashlib.sha256(raw_bytes).hexdigest()
        store_model_bytes(fp, raw_bytes)
        return raw_bytes, filename

    # 2. Sample Model Name
    if sample_model_name:
        # Prevent directory traversal attacks
        clean_name = os.path.basename(sample_model_name)
        if clean_name != sample_model_name or ".." in sample_model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid sample model name (directory traversal prohibited).",
            )
        sample_path = os.path.join("sample_models", clean_name)
        if not os.path.exists(sample_path) or not os.path.isfile(sample_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample model '{clean_name}' not found. Available samples: linear_classifier.onnx, conv_net.onnx.",
            )
        with open(sample_path, "rb") as f:
            raw_bytes = f.read()
        import hashlib
        fp = hashlib.sha256(raw_bytes).hexdigest()
        store_model_bytes(fp, raw_bytes)
        return raw_bytes, clean_name

    # 3. Model Fingerprint
    if model_fingerprint:
        clean_fp = model_fingerprint.strip().lower()
        cached = get_model_bytes(clean_fp)
        if cached:
            return cached, f"model_{clean_fp[:8]}.onnx"

        # Check sample models for matching fingerprint
        for sample_file in ["linear_classifier.onnx", "conv_net.onnx"]:
            sample_path = os.path.join("sample_models", sample_file)
            if os.path.exists(sample_path):
                with open(sample_path, "rb") as f:
                    s_bytes = f.read()
                import hashlib
                if hashlib.sha256(s_bytes).hexdigest() == clean_fp:
                    store_model_bytes(clean_fp, s_bytes)
                    return s_bytes, sample_file

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Model bytes for fingerprint '{clean_fp}' not found in active cache or sample library. "
                "Please upload the ONNX file directly in the request."
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Must provide an uploaded model file, a valid 'sample_model_name', or a recognized 'model_fingerprint'.",
    )
