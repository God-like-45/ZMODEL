"""Safe, isolated artifact storage and retrieval for transformed ONNX models.

Enforces immutability of source models, SHA-256 cryptographic fingerprinting,
strict path-traversal prevention, and controlled file access.
Model binaries are stored exclusively in the local artifact directory and
never persisted as raw binary blobs in PostgreSQL.
"""

import hashlib
import logging
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from zmodel.core.config import MAX_UPLOAD_SIZE_BYTES

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "artifacts",
    "transformed",
)


@dataclass(frozen=True)
class ArtifactInfo:
    """Immutable metadata record for a persisted model artifact."""

    artifact_id: str
    file_path: str
    file_size_bytes: int
    sha256_hash: str
    created_at: str
    suggested_filename: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ArtifactSecurityError(Exception):
    """Raised when an artifact path or identifier violates traversal safeguards."""
    pass


class ArtifactStore:
    """
    Controlled local artifact storage for transformed ONNX models.
    Guarantees isolation, path traversal prevention, and SHA-256 verification.
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_file_size_bytes: int = MAX_UPLOAD_SIZE_BYTES,
    ):
        self.storage_dir = os.path.abspath(storage_dir or DEFAULT_ARTIFACTS_DIR)
        self.max_file_size_bytes = max_file_size_bytes
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Create the storage directory if it does not exist."""
        os.makedirs(self.storage_dir, exist_ok=True)

    def _validate_safe_id(self, artifact_id: str) -> None:
        """Validate artifact ID to prevent path traversal characters."""
        if not artifact_id or not re.match(r"^[A-Za-z0-9_\-]+$", artifact_id):
            raise ArtifactSecurityError(
                f"Invalid artifact ID '{artifact_id}'. Must be alphanumeric with hyphens/underscores only."
            )

    def _validate_safe_path(self, target_path: str) -> None:
        """
        Confirm that target_path is strictly within storage_dir using realpath.
        Rejects directory traversal attempts (e.g. '../', symlink escapes).
        """
        real_target = os.path.realpath(target_path)
        real_root = os.path.realpath(self.storage_dir)

        # Allow target to equal root or be a direct subpath
        if not (real_target == real_root or real_target.startswith(real_root + os.sep)):
            raise ArtifactSecurityError(
                f"Path traversal detected: '{target_path}' is outside storage root '{self.storage_dir}'."
            )

    def store_artifact(
        self,
        model_bytes: bytes,
        parent_fingerprint: str,
        transformation_name: str,
        original_filename: str = "model.onnx",
    ) -> ArtifactInfo:
        """
        Safely persist transformed model bytes to an isolated file.
        Computes new SHA-256 fingerprint, generates a unique artifact ID,
        and ensures the source model is never touched.
        """
        if len(model_bytes) == 0:
            raise ValueError("Cannot store empty model artifact (0 bytes).")

        if len(model_bytes) > self.max_file_size_bytes:
            raise ValueError(
                f"Transformed model size ({len(model_bytes)} bytes) exceeds maximum limit "
                f"({self.max_file_size_bytes} bytes)."
            )

        sha256_hash = hashlib.sha256(model_bytes).hexdigest()
        clean_transform = re.sub(r"[^A-Za-z0-9_]", "_", transformation_name.lower())
        short_parent = parent_fingerprint[:8] if parent_fingerprint else "orig"
        uid = uuid.uuid4().hex[:8]

        artifact_id = f"tart_{short_parent}_{clean_transform}_{uid}"
        self._validate_safe_id(artifact_id)

        target_path = os.path.join(self.storage_dir, f"{artifact_id}.onnx")
        self._validate_safe_path(target_path)

        # Sanitize download filename
        base_name, _ = os.path.splitext(os.path.basename(original_filename))
        clean_base = re.sub(r"[^A-Za-z0-9_\-]", "_", base_name)
        suggested_filename = f"{clean_base}_{clean_transform}_{sha256_hash[:8]}.onnx"

        # Write atomically via temp file to avoid partial writes
        temp_fd, temp_path = tempfile.mkstemp(dir=self.storage_dir, prefix="tmp_art_")
        try:
            with os.fdopen(temp_fd, "wb") as f:
                f.write(model_bytes)
            # Atomic rename / replace
            if os.path.exists(target_path):
                os.remove(target_path)
            os.replace(temp_path, target_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        created_at = datetime.now(timezone.utc).isoformat()
        return ArtifactInfo(
            artifact_id=artifact_id,
            file_path=target_path,
            file_size_bytes=len(model_bytes),
            sha256_hash=sha256_hash,
            created_at=created_at,
            suggested_filename=suggested_filename,
        )

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactInfo]:
        """
        Retrieve artifact metadata by ID, verifying path safety and file existence.
        """
        self._validate_safe_id(artifact_id)
        target_path = os.path.join(self.storage_dir, f"{artifact_id}.onnx")
        self._validate_safe_path(target_path)

        if not os.path.isfile(target_path):
            return None

        stat = os.stat(target_path)
        with open(target_path, "rb") as f:
            content = f.read()
        sha256_hash = hashlib.sha256(content).hexdigest()

        return ArtifactInfo(
            artifact_id=artifact_id,
            file_path=target_path,
            file_size_bytes=stat.st_size,
            sha256_hash=sha256_hash,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            suggested_filename=f"{artifact_id}.onnx",
        )

    def read_artifact_bytes(
        self, artifact_id: str, expected_hash: Optional[str] = None
    ) -> Tuple[bytes, ArtifactInfo]:
        """
        Read artifact bytes directly with cryptographic verification.
        Raises FileNotFoundError if missing, ArtifactSecurityError on traversal,
        or ValueError if expected_hash is provided and does not match bytes on disk.
        """
        info = self.get_artifact(artifact_id)
        if info is None:
            raise FileNotFoundError(f"Artifact '{artifact_id}' not found.")

        self._validate_safe_path(info.file_path)
        with open(info.file_path, "rb") as f:
            data = f.read()

        # Integrity verification
        actual_hash = hashlib.sha256(data).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"Artifact integrity compromised for '{artifact_id}': "
                f"expected hash {expected_hash}, found {actual_hash}."
            )

        return data, info

    def delete_artifact(self, artifact_id: str) -> bool:
        """Safely delete an artifact file if it exists."""
        self._validate_safe_id(artifact_id)
        target_path = os.path.join(self.storage_dir, f"{artifact_id}.onnx")
        self._validate_safe_path(target_path)

        if os.path.isfile(target_path):
            os.remove(target_path)
            return True
        return False

    def list_artifacts(self) -> List[ArtifactInfo]:
        """List all valid artifacts in the storage directory."""
        results: List[ArtifactInfo] = []
        if not os.path.exists(self.storage_dir):
            return results

        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".onnx") and not fname.startswith("tmp_"):
                art_id = fname[:-5]
                try:
                    info = self.get_artifact(art_id)
                    if info:
                        results.append(info)
                except Exception as ex:
                    logger.warning("Could not read artifact %s: %s", fname, ex)
        return results


# Global singleton instance for application use
_GLOBAL_STORE: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get or create singleton ArtifactStore."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = ArtifactStore()
    return _GLOBAL_STORE
