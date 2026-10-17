"""
Capability Profile System for Z-ModelPilot.

Defines target-runtime capability profiles, operator rules, and validation logic.
Ensures capability profiles are strictly typed, versioned, validated, and untrusted-input safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class ProfileValidationError(Exception):
    """Raised when a capability profile fails schema or semantic validation."""
    pass


VALID_STATUSES: Set[str] = {"PASS", "FAIL", "WARNING", "UNKNOWN"}
VALID_CONFIDENCE_LEVELS: Set[str] = {
    "VERIFIED_DOCUMENTED",
    "VERIFIED_EXPERIMENTAL",
    "ILLUSTRATIVE_DRAFT",
    "UNVERIFIED_COMMUNITY",
}


@dataclass(frozen=True)
class OperatorRule:
    """Rules and constraints for a specific ONNX operator in a capability profile."""
    status: str
    min_version: Optional[int] = None
    max_version: Optional[int] = None
    supported_types: List[str] = field(default_factory=list)
    restricted_types: List[str] = field(default_factory=list)
    hardware_accelerated: Optional[str] = None
    evidence: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ProfileValidationError(
                f"Invalid operator rule status '{self.status}'. Must be one of {sorted(VALID_STATUSES)}."
            )
        if self.min_version is not None and self.min_version < 1:
            raise ProfileValidationError(f"min_version must be >= 1, got {self.min_version}")
        if self.max_version is not None and self.max_version < 1:
            raise ProfileValidationError(f"max_version must be >= 1, got {self.max_version}")
        if (
            self.min_version is not None
            and self.max_version is not None
            and self.min_version > self.max_version
        ):
            raise ProfileValidationError(
                f"min_version ({self.min_version}) cannot exceed max_version ({self.max_version})"
            )


@dataclass(frozen=True)
class CapabilityProfile:
    """Target-runtime capability profile specification."""
    profile_id: str
    display_name: str
    version: str
    ruleset_version: str
    target_architecture: str
    target_runtime: str
    confidence_classification: str
    is_illustrative: bool
    min_opset: int
    max_opset: int
    supported_domains: List[str]
    supported_data_types: List[str]
    unsupported_data_types: List[str]
    operator_rules: Dict[str, OperatorRule]
    known_limitations: List[str]
    evidence_source: str
    evidence_url: Optional[str]
    verification_date: str

    def __post_init__(self) -> None:
        if not self.profile_id or not isinstance(self.profile_id, str):
            raise ProfileValidationError("profile_id must be a non-empty string.")
        if not self.display_name:
            raise ProfileValidationError("display_name must be a non-empty string.")
        if not self.version:
            raise ProfileValidationError("version must be a non-empty string.")
        if not self.ruleset_version:
            raise ProfileValidationError("ruleset_version must be a non-empty string.")
        if self.min_opset < 1:
            raise ProfileValidationError(f"min_opset must be >= 1, got {self.min_opset}")
        if self.max_opset < self.min_opset:
            raise ProfileValidationError(
                f"max_opset ({self.max_opset}) cannot be less than min_opset ({self.min_opset})"
            )
        if self.confidence_classification not in VALID_CONFIDENCE_LEVELS:
            raise ProfileValidationError(
                f"Invalid confidence_classification '{self.confidence_classification}'. "
                f"Must be one of {sorted(VALID_CONFIDENCE_LEVELS)}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable dictionary."""
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "version": self.version,
            "ruleset_version": self.ruleset_version,
            "target_architecture": self.target_architecture,
            "target_runtime": self.target_runtime,
            "confidence_classification": self.confidence_classification,
            "is_illustrative": self.is_illustrative,
            "min_opset": self.min_opset,
            "max_opset": self.max_opset,
            "supported_domains": list(self.supported_domains),
            "supported_data_types": list(self.supported_data_types),
            "unsupported_data_types": list(self.unsupported_data_types),
            "operator_rules": {
                k: {
                    "status": v.status,
                    "min_version": v.min_version,
                    "max_version": v.max_version,
                    "supported_types": list(v.supported_types),
                    "restricted_types": list(v.restricted_types),
                    "hardware_accelerated": v.hardware_accelerated,
                    "evidence": v.evidence,
                    "notes": v.notes,
                }
                for k, v in self.operator_rules.items()
            },
            "known_limitations": list(self.known_limitations),
            "evidence_source": self.evidence_source,
            "evidence_url": self.evidence_url,
            "verification_date": self.verification_date,
        }


def parse_operator_rule(name: str, rule_data: Any) -> OperatorRule:
    """Parse and validate an operator rule dictionary."""
    if not isinstance(rule_data, dict):
        raise ProfileValidationError(f"Operator rule for '{name}' must be a dictionary.")

    status = rule_data.get("status")
    if not status or not isinstance(status, str):
        raise ProfileValidationError(f"Operator rule for '{name}' missing valid 'status' string.")

    return OperatorRule(
        status=status.upper(),
        min_version=rule_data.get("min_version"),
        max_version=rule_data.get("max_version"),
        supported_types=list(rule_data.get("supported_types", [])),
        restricted_types=list(rule_data.get("restricted_types", [])),
        hardware_accelerated=rule_data.get("hardware_accelerated"),
        evidence=str(rule_data.get("evidence", "")),
        notes=str(rule_data.get("notes", "")),
    )


def load_profile_from_dict(data: Dict[str, Any]) -> CapabilityProfile:
    """
    Validate schema and instantiate CapabilityProfile from a dictionary.
    
    Raises:
        ProfileValidationError: If any required field is missing or invalid.
    """
    if not isinstance(data, dict):
        raise ProfileValidationError("Profile data must be a dictionary.")

    required_keys = [
        "profile_id",
        "display_name",
        "version",
        "ruleset_version",
        "target_architecture",
        "target_runtime",
        "confidence_classification",
        "is_illustrative",
        "min_opset",
        "max_opset",
        "supported_domains",
        "supported_data_types",
        "operator_rules",
    ]

    for key in required_keys:
        if key not in data:
            raise ProfileValidationError(f"Profile definition missing required field: '{key}'")

    raw_op_rules = data["operator_rules"]
    if not isinstance(raw_op_rules, dict):
        raise ProfileValidationError("'operator_rules' must be a dictionary.")

    operator_rules: Dict[str, OperatorRule] = {}
    for op_name, op_rule_data in raw_op_rules.items():
        if not isinstance(op_name, str):
            raise ProfileValidationError("Operator rule keys must be strings.")
        operator_rules[op_name] = parse_operator_rule(op_name, op_rule_data)

    return CapabilityProfile(
        profile_id=str(data["profile_id"]),
        display_name=str(data["display_name"]),
        version=str(data["version"]),
        ruleset_version=str(data["ruleset_version"]),
        target_architecture=str(data["target_architecture"]),
        target_runtime=str(data["target_runtime"]),
        confidence_classification=str(data["confidence_classification"]),
        is_illustrative=bool(data["is_illustrative"]),
        min_opset=int(data["min_opset"]),
        max_opset=int(data["max_opset"]),
        supported_domains=list(data["supported_domains"]),
        supported_data_types=list(data["supported_data_types"]),
        unsupported_data_types=list(data.get("unsupported_data_types", [])),
        operator_rules=operator_rules,
        known_limitations=list(data.get("known_limitations", [])),
        evidence_source=str(data.get("evidence_source", "")),
        evidence_url=data.get("evidence_url"),
        verification_date=str(data.get("verification_date", "")),
    )


def load_profile_from_json(json_str: str) -> CapabilityProfile:
    """Parse JSON string and construct a CapabilityProfile."""
    try:
        data = json.loads(json_str)
    except Exception as e:
        raise ProfileValidationError(f"Invalid JSON string: {e}") from e
    return load_profile_from_dict(data)


def load_profile_from_file(file_path: Path) -> CapabilityProfile:
    """Load and validate a capability profile from a JSON file."""
    path = Path(file_path)
    if not path.is_file():
        raise ProfileValidationError(f"Profile file not found: {path}")
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        raise ProfileValidationError(f"Cannot read profile file {path}: {e}") from e
    return load_profile_from_json(content)


def load_builtin_profiles(profiles_dir: Optional[Path] = None) -> Dict[str, CapabilityProfile]:
    """
    Discover and load all valid capability profiles from the profiles directory.
    
    Returns a dictionary mapping profile_id -> CapabilityProfile.
    """
    if profiles_dir is None:
        # Default to workspace profiles_data directory
        profiles_dir = Path(__file__).resolve().parent.parent.parent.parent / "profiles_data"

    profiles: Dict[str, CapabilityProfile] = {}
    if not profiles_dir.exists() or not profiles_dir.is_dir():
        return profiles

    for profile_file in sorted(profiles_dir.glob("*.json")):
        try:
            profile = load_profile_from_file(profile_file)
            profiles[profile.profile_id] = profile
        except Exception:
            # Skip invalid or unparseable profile files safely
            continue

    return profiles
