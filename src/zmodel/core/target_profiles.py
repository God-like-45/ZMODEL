"""
Phase 11: Target Validation Profile Specifications & Loaders.

Encapsulates explicit, verifiable target constraints for IBM Z / Linux on IBM Z and local environments.
Distinguishes between configured requirements and observed runtime evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class TargetProfileError(Exception):
    """Raised when target profile is malformed or invalid."""
    pass


@dataclass(frozen=True)
class TargetRequirement:
    """A specific verifiable requirement in a target validation profile."""
    name: str
    category: str  # 'architecture', 'os', 'runtime', 'provider', 'hardware', 'model'
    expected_value: Any
    description: str
    is_mandatory: bool = True
    requires_target_hardware: bool = False
    remediation_hint: str = ""


@dataclass(frozen=True)
class TargetValidationProfile:
    """Configurable, versioned specification of target deployment requirements."""
    profile_id: str
    display_name: str
    version: str
    target_architecture: str  # 's390x', 'x86_64', etc.
    target_os: str            # 'Linux', 'any', etc.
    min_python_version: str
    max_python_version: str
    min_ort_version: Optional[str]
    required_execution_providers: List[str]
    required_facilities: List[str]  # e.g. ['facility_193_nnpa']
    accelerator_type: Optional[str] # e.g. 'IBM Integrated Accelerator for AI (NNPA)'
    min_opset: int
    max_opset: int
    supported_data_types: List[str]
    requires_target_hardware: bool
    confidence_classification: str  # 'TARGET_VERIFIED', 'UNVERIFIED_TARGET', 'LOCALLY_VERIFIED'
    evidence_source: str
    documentation_url: Optional[str] = None
    known_constraints: List[str] = field(default_factory=list)

    @property
    def architecture(self) -> str:
        return self.target_architecture

    @property
    def profile_name(self) -> str:
        return self.display_name

    @property
    def profile_version(self) -> str:
        return self.version

    @property
    def required_providers(self) -> List[str]:
        return self.required_execution_providers

    @property
    def supported_os(self) -> List[str]:
        return [self.target_os] if self.target_os else ["Linux"]

    @property
    def requires_accelerator(self) -> bool:
        return bool(self.accelerator_type or self.required_facilities)

    @property
    def required_accelerators(self) -> List[str]:
        accs = []
        if self.accelerator_type:
            accs.append(self.accelerator_type)
        if self.required_facilities:
            accs.extend([f"Facility {f}" for f in self.required_facilities])
        return accs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TargetValidationProfile:
        return cls(
            profile_id=str(data["profile_id"]),
            display_name=str(data["display_name"]),
            version=str(data.get("version", "1.0.0")),
            target_architecture=str(data.get("target_architecture", "s390x")),
            target_os=str(data.get("target_os", "Linux")),
            min_python_version=str(data.get("min_python_version", "3.9")),
            max_python_version=str(data.get("max_python_version", "3.12")),
            min_ort_version=data.get("min_ort_version"),
            required_execution_providers=list(data.get("required_execution_providers", ["CPUExecutionProvider"])),
            required_facilities=list(data.get("required_facilities", [])),
            accelerator_type=data.get("accelerator_type"),
            min_opset=int(data.get("min_opset", 7)),
            max_opset=int(data.get("max_opset", 21)),
            supported_data_types=list(data.get("supported_data_types", ["float32", "float16", "int64", "int32"])),
            requires_target_hardware=bool(data.get("requires_target_hardware", True)),
            confidence_classification=str(data.get("confidence_classification", "UNVERIFIED_TARGET")),
            evidence_source=str(data.get("evidence_source", "")),
            documentation_url=data.get("documentation_url"),
            known_constraints=list(data.get("known_constraints", [])),
        )


# Built-in default target profiles
BUILTIN_TARGET_PROFILES: Dict[str, TargetValidationProfile] = {
    "ibm_z16_linux_nnpa": TargetValidationProfile(
        profile_id="ibm_z16_linux_nnpa",
        display_name="IBM z16 / Linux on IBM Z (s390x) — Integrated Accelerator for AI (NNPA)",
        version="1.0.0",
        target_architecture="s390x",
        target_os="Linux",
        min_python_version="3.9",
        max_python_version="3.12",
        min_ort_version="1.16.0",
        required_execution_providers=["NNPAExecutionProvider", "CPUExecutionProvider"],
        required_facilities=["193"],
        accelerator_type="IBM Integrated Accelerator for AI (NNPA)",
        min_opset=11,
        max_opset=20,
        supported_data_types=["float32", "float16"],
        requires_target_hardware=True,
        confidence_classification="UNVERIFIED_TARGET",
        evidence_source="IBM z16 Technical Guide (SG24-8950) & IBM AI Toolkit for Linux on IBM Z",
        documentation_url="https://www.ibm.com/docs/en/linux-on-systems?topic=overview-artificial-intelligence",
        known_constraints=[
            "Requires Linux on IBM Z (s390x) running on IBM z16 (Machine Types 3931/3932) or LinuxONE 4.",
            "Requires s390x facility bit 193 active in kernel /proc/cpuinfo.",
            "Requires ONNX Runtime built with NNPAExecutionProvider support from IBM AI Toolkit.",
            "Unsupported operators automatically fall back to s390x CPUExecutionProvider with vector extensions (SIMD).",
        ],
    ),
    "ibm_z_linux_cpu": TargetValidationProfile(
        profile_id="ibm_z_linux_cpu",
        display_name="IBM Z / Linux on IBM Z (s390x) — CPU Execution Baseline",
        version="1.0.0",
        target_architecture="s390x",
        target_os="Linux",
        min_python_version="3.9",
        max_python_version="3.12",
        min_ort_version="1.14.0",
        required_execution_providers=["CPUExecutionProvider"],
        required_facilities=[],
        accelerator_type=None,
        min_opset=7,
        max_opset=21,
        supported_data_types=["float32", "float16", "int64", "int32", "uint8", "int8"],
        requires_target_hardware=True,
        confidence_classification="UNVERIFIED_TARGET",
        evidence_source="Linux on IBM Z ONNX Runtime Build Documentation",
        documentation_url="https://github.com/microsoft/onnxruntime/blob/main/BUILD.md",
        known_constraints=[
            "Requires Linux on IBM Z (s390x) LPAR or z/VM guest.",
            "Runs on s390x SIMD vector extensions without NNPA coprocessor offload.",
        ],
    ),
    "local_host_development": TargetValidationProfile(
        profile_id="local_host_development",
        display_name="Local Host Development Environment (x86_64 / ARM64 Baseline)",
        version="1.0.0",
        target_architecture="any",
        target_os="any",
        min_python_version="3.9",
        max_python_version="3.13",
        min_ort_version="1.12.0",
        required_execution_providers=["CPUExecutionProvider"],
        required_facilities=[],
        accelerator_type=None,
        min_opset=7,
        max_opset=21,
        supported_data_types=["float32", "float16", "int64", "int32", "uint8", "int8"],
        requires_target_hardware=False,
        confidence_classification="LOCALLY_VERIFIED",
        evidence_source="Local Python Virtual Environment & ONNX Runtime",
        documentation_url="https://onnxruntime.ai/",
        known_constraints=[
            "Verifies local functionality only.",
            "Does not establish or imply IBM Z compatibility or performance.",
        ],
    ),
}


def load_target_profiles(custom_dir: Optional[Path] = None) -> Dict[str, TargetValidationProfile]:
    """
    Load target profiles: merges builtins with any custom JSON profiles found in profiles_data/.
    """
    profiles = dict(BUILTIN_TARGET_PROFILES)

    search_dirs = []
    if custom_dir:
        search_dirs.append(custom_dir)
    default_dir = Path(__file__).resolve().parent.parent.parent.parent / "profiles_data"
    if default_dir.exists():
        search_dirs.append(default_dir)

    for pdir in search_dirs:
        for pfile in pdir.glob("*.target.json"):
            try:
                data = json.loads(pfile.read_text(encoding="utf-8"))
                p = TargetValidationProfile.from_dict(data)
                profiles[p.profile_id] = p
            except Exception:
                continue

    return profiles


def get_target_profile(profile_id: str, fallback_to_default: bool = False) -> TargetValidationProfile:
    """Retrieve profile by ID or raise TargetProfileError (or fallback to local if requested)."""
    profiles = load_target_profiles()
    if profile_id not in profiles:
        if fallback_to_default:
            return profiles.get("local_host_development", BUILTIN_TARGET_PROFILES["local_host_development"])
        raise TargetProfileError(
            f"Target profile '{profile_id}' not found. Available profiles: {sorted(profiles.keys())}"
        )
    return profiles[profile_id]
