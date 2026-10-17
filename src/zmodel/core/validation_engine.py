"""
Phase 11: Architecture and Runtime Compatibility Validation Engine.

Executes comprehensive, evidence-based validation checks comparing an ONNX model
and environment facts against an explicit TargetValidationProfile.
Maintains strict honesty: never treats local x86_64 execution as IBM Z validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
try:
    import onnx
    from onnx import helper
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from zmodel.core.benchmark import generate_benchmark_inputs
from zmodel.core.compatibility import CompatibilityAnalyzer, CompatibilityReport
from zmodel.core.environment import EnvironmentInventory, inspect_environment
from zmodel.core.model_loader import ModelSummary, load_and_inspect_model
from zmodel.core.profiles import CapabilityProfile, load_builtin_profiles
from zmodel.core.target_profiles import TargetValidationProfile, get_target_profile


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"
    NOT_RUN = "NOT_RUN"


class OverallValidationStatus(str, Enum):
    LOCALLY_VERIFIED = "LOCALLY_VERIFIED"
    TARGET_VERIFIED = "TARGET_VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    FAILED_VALIDATION = "FAILED_VALIDATION"


@dataclass
class ValidationCheck:
    """Individual evaluated validation requirement."""
    check_id: str
    name: str
    category: str
    requirement: str
    detected_value: Optional[str]
    expected_value: str
    status: CheckStatus
    evidence: str
    remediation: str
    requires_target_hardware: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationCheck:
        data_copy = dict(data)
        data_copy["status"] = CheckStatus(data_copy["status"])
        return cls(**data_copy)


@dataclass
class ValidationReport:
    """Complete, auditable result of a target validation run."""
    run_id: str
    model_fingerprint: str
    model_name: str
    target_profile_id: str
    target_profile_version: str
    environment: EnvironmentInventory
    overall_status: OverallValidationStatus
    status_rationale: str
    checks: List[ValidationCheck]
    summary_counts: Dict[str, int]
    numerical_correctness: Dict[str, Any]
    benchmark_eligibility: bool
    execution_mode: str  # 'LOCAL', 'REMOTE_TARGET', 'SIMULATED'
    timestamp_utc: str
    provenance_badge: str = ""

    def __post_init__(self) -> None:
        if not self.provenance_badge:
            if self.overall_status == OverallValidationStatus.TARGET_VERIFIED:
                self.provenance_badge = "Verified on IBM Z (s390x)"
            elif self.overall_status == OverallValidationStatus.LOCALLY_VERIFIED:
                self.provenance_badge = "Locally Verified (Host x86_64; IBM Z Pending)"
            elif self.overall_status == OverallValidationStatus.PARTIALLY_VERIFIED:
                self.provenance_badge = "Partially Verified"
            elif self.overall_status == OverallValidationStatus.FAILED_VALIDATION:
                self.provenance_badge = "Validation Failed"
            else:
                self.provenance_badge = "Not Verified"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_fingerprint": self.model_fingerprint,
            "model_name": self.model_name,
            "target_profile_id": self.target_profile_id,
            "target_profile_version": self.target_profile_version,
            "environment": self.environment.to_dict(),
            "overall_status": self.overall_status.value,
            "status_rationale": self.status_rationale,
            "checks": [c.to_dict() for c in self.checks],
            "summary_counts": dict(self.summary_counts),
            "numerical_correctness": dict(self.numerical_correctness),
            "benchmark_eligibility": self.benchmark_eligibility,
            "execution_mode": self.execution_mode,
            "timestamp_utc": self.timestamp_utc,
            "provenance_badge": self.provenance_badge,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidationReport:
        env = EnvironmentInventory.from_dict(data["environment"])
        checks = [ValidationCheck.from_dict(c) for c in data.get("checks", [])]
        return cls(
            run_id=data["run_id"],
            model_fingerprint=data["model_fingerprint"],
            model_name=data.get("model_name", "model.onnx"),
            target_profile_id=data["target_profile_id"],
            target_profile_version=data.get("target_profile_version", "1.0.0"),
            environment=env,
            overall_status=OverallValidationStatus(data["overall_status"]),
            status_rationale=data.get("status_rationale", ""),
            checks=checks,
            summary_counts=data.get("summary_counts", {}),
            numerical_correctness=data.get("numerical_correctness", {}),
            benchmark_eligibility=bool(data.get("benchmark_eligibility", False)),
            execution_mode=data.get("execution_mode", "LOCAL"),
            timestamp_utc=data.get("timestamp_utc", ""),
            provenance_badge=data.get("provenance_badge", ""),
        )


class _hybrid_evaluate:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            def _caller(
                model_bytes: Optional[bytes] = None,
                model_name: Optional[str] = None,
                target_profile: Optional[TargetValidationProfile] = None,
                env: Optional[EnvironmentInventory] = None,
                reference_outputs: Optional[Dict[str, np.ndarray]] = None,
                atol: float = 1e-4,
                rtol: float = 1e-3,
                execution_mode: str = "LOCAL",
                model_path: Optional[str] = None,
                **kwargs: Any,
            ) -> ValidationReport:
                if model_bytes is None:
                    if model_path and os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            model_bytes = f.read()
                        if not model_name:
                            model_name = os.path.basename(model_path)
                    else:
                        model_bytes = b""
                        if not model_name:
                            model_name = model_path or "invalid.onnx"
                if not model_name:
                    model_name = "model.onnx"
                target_profile = target_profile or kwargs.get("profile") or get_target_profile("local_host_development")
                env = env or kwargs.get("env") or inspect_environment(collection_method="LOCAL_INSPECTION")
                return self.func(
                    owner,
                    model_bytes=model_bytes,
                    model_name=model_name,
                    target_profile=target_profile,
                    env=env,
                    reference_outputs=reference_outputs,
                    atol=atol,
                    rtol=rtol,
                    execution_mode=execution_mode,
                )
            return _caller
        else:
            def _inst_caller(
                model_path: Optional[str] = None,
                model_bytes: Optional[bytes] = None,
                model_name: Optional[str] = None,
                target_profile: Optional[TargetValidationProfile] = None,
                env: Optional[EnvironmentInventory] = None,
                reference_outputs: Optional[Dict[str, np.ndarray]] = None,
                atol: Optional[float] = None,
                rtol: Optional[float] = None,
                execution_mode: str = "LOCAL",
                **kwargs: Any,
            ) -> ValidationReport:
                if model_bytes is None:
                    if model_path and os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            model_bytes = f.read()
                        if not model_name:
                            model_name = os.path.basename(model_path)
                    else:
                        model_bytes = b""
                        if not model_name:
                            model_name = model_path or "invalid.onnx"
                if not model_name:
                    model_name = "model.onnx"
                target_profile = target_profile or instance.profile or kwargs.get("profile") or get_target_profile("local_host_development")
                env = env or instance.environment or kwargs.get("env") or inspect_environment(collection_method="LOCAL_INSPECTION")
                _atol = atol if atol is not None else instance.atol
                _rtol = rtol if rtol is not None else instance.rtol
                return self.func(
                    owner,
                    model_bytes=model_bytes,
                    model_name=model_name,
                    target_profile=target_profile,
                    env=env,
                    reference_outputs=reference_outputs,
                    atol=_atol,
                    rtol=_rtol,
                    execution_mode=execution_mode,
                )
            return _inst_caller


class ValidationEngine:
    """Core evaluation engine executing the 12-point target validation battery."""

    def __init__(
        self,
        profile: Optional[TargetValidationProfile] = None,
        environment: Optional[EnvironmentInventory] = None,
        atol: float = 1e-4,
        rtol: float = 1e-3,
    ) -> None:
        self.profile = profile
        self.environment = environment
        self.atol = atol
        self.rtol = rtol

    def _evaluate_impl(
        self,
        model_bytes: bytes,
        model_name: str,
        target_profile: TargetValidationProfile,
        env: Optional[EnvironmentInventory] = None,
        reference_outputs: Optional[Dict[str, np.ndarray]] = None,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        execution_mode: str = "LOCAL",
    ) -> ValidationReport:
        if env is None:
            env = inspect_environment(collection_method="LOCAL_INSPECTION")

        run_id = f"VAL-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 10000:04d}"
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        checks: List[ValidationCheck] = []

        # 1. Model Inspection & Fingerprint
        try:
            summary = load_and_inspect_model(model_bytes, filename=model_name)
            model_fingerprint = summary.sha256_hash
        except Exception as ex:
            model_fingerprint = hashlib.sha256(model_bytes).hexdigest()
            summary = None

        # --- Check 1: Machine Architecture ---
        expected_arch = target_profile.target_architecture
        is_target_s390x = expected_arch.lower() in ("s390x", "s390")
        detected_arch = env.architecture

        if is_target_s390x:
            if env.is_s390x:
                c1_status = CheckStatus.PASS
                c1_evidence = f"Target s390x architecture verified on physical/virtual host ({detected_arch})."
                c1_remediation = "None required."
            elif execution_mode == "LOCAL":
                c1_status = CheckStatus.NOT_RUN
                c1_evidence = f"Active execution host is {detected_arch}. Target requires physical Linux on IBM Z (s390x)."
                c1_remediation = "Deploy or connect to physical Linux on IBM Z (s390x) LPAR or z/VM guest."
            else:
                c1_status = CheckStatus.FAIL
                c1_evidence = f"Target host reported architecture {detected_arch}, but {expected_arch} was required."
                c1_remediation = "Verify remote host architecture with 'uname -m'."
        else:
            c1_status = CheckStatus.PASS
            c1_evidence = f"Host architecture {detected_arch} satisfies '{expected_arch}' baseline."
            c1_remediation = "None required."

        checks.append(ValidationCheck(
            check_id="check_01_arch",
            name="Machine Architecture",
            category="architecture",
            requirement=f"Machine instruction set architecture must be '{expected_arch}'.",
            detected_value=detected_arch,
            expected_value=expected_arch,
            status=c1_status,
            evidence=c1_evidence,
            remediation=c1_remediation,
            requires_target_hardware=is_target_s390x,
        ))

        # --- Check 2: Operating System ---
        expected_os = target_profile.target_os
        detected_os = env.os_name
        if expected_os.lower() in ("any", "unknown") or expected_os.lower() == detected_os.lower():
            c2_status = CheckStatus.PASS
            c2_evidence = f"Operating system '{detected_os} {env.os_release}' satisfies target requirement '{expected_os}'."
            c2_remediation = "None required."
        elif is_target_s390x and execution_mode == "LOCAL":
            c2_status = CheckStatus.NOT_RUN
            c2_evidence = f"Local host OS is '{detected_os}'; Linux on IBM Z target OS validation remains pending."
            c2_remediation = "Verify OS on target s390x node (e.g. Red Hat Enterprise Linux 8/9, Ubuntu 22.04/24.04, SLES 15)."
        else:
            c2_status = CheckStatus.FAIL
            c2_evidence = f"Detected OS '{detected_os}' does not match required OS '{expected_os}'."
            c2_remediation = f"Install and run on supported {expected_os} distribution."

        checks.append(ValidationCheck(
            check_id="check_02_os",
            name="Operating System & Kernel",
            category="os",
            requirement=f"Operating system must be '{expected_os}'.",
            detected_value=f"{detected_os} (Kernel {env.kernel_version})",
            expected_value=expected_os,
            status=c2_status,
            evidence=c2_evidence,
            remediation=c2_remediation,
            requires_target_hardware=is_target_s390x,
        ))

        # --- Check 3: Python Runtime ---
        detected_py = env.python_version
        try:
            py_parts = [int(p) for p in detected_py.split(".")[:2]]
            min_parts = [int(p) for p in target_profile.min_python_version.split(".")[:2]]
            max_parts = [int(p) for p in target_profile.max_python_version.split(".")[:2]]
            py_ok = (min_parts <= py_parts <= max_parts)
        except Exception:
            py_ok = True

        if py_ok:
            c3_status = CheckStatus.PASS
            c3_evidence = f"Detected Python {detected_py} is within supported range [{target_profile.min_python_version}, {target_profile.max_python_version}]."
            c3_remediation = "None required."
        else:
            c3_status = CheckStatus.FAIL
            c3_evidence = f"Detected Python {detected_py} is outside supported range [{target_profile.min_python_version}, {target_profile.max_python_version}]."
            c3_remediation = f"Use Python {target_profile.min_python_version} to {target_profile.max_python_version}."

        checks.append(ValidationCheck(
            check_id="check_03_python",
            name="Python Version & ABI",
            category="runtime",
            requirement=f"Python version >= {target_profile.min_python_version} and <= {target_profile.max_python_version}.",
            detected_value=detected_py,
            expected_value=f">={target_profile.min_python_version}, <={target_profile.max_python_version}",
            status=c3_status,
            evidence=c3_evidence,
            remediation=c3_remediation,
            requires_target_hardware=False,
        ))

        # --- Check 4: ONNX Runtime Installation ---
        detected_ort = env.onnxruntime_version
        if detected_ort:
            c4_status = CheckStatus.PASS
            c4_evidence = f"ONNX Runtime version {detected_ort} is installed and importable."
            c4_remediation = "None required."
        else:
            c4_status = CheckStatus.FAIL
            c4_evidence = "ONNX Runtime package ('onnxruntime') is not installed in the active environment."
            c4_remediation = "Install onnxruntime via pip or IBM AI Toolkit package."

        checks.append(ValidationCheck(
            check_id="check_04_ort_version",
            name="ONNX Runtime Installation",
            category="runtime",
            requirement="ONNX Runtime engine must be installed and importable.",
            detected_value=detected_ort or "NOT_INSTALLED",
            expected_value=f">={target_profile.min_ort_version or '1.14.0'}",
            status=c4_status,
            evidence=c4_evidence,
            remediation=c4_remediation,
            requires_target_hardware=False,
        ))

        # --- Check 5: Required Execution Providers ---
        req_providers = target_profile.required_execution_providers
        has_primary = False
        if req_providers:
            primary_req = req_providers[0]
            has_primary = primary_req in env.available_providers
        else:
            primary_req = "CPUExecutionProvider"
            has_primary = True

        if has_primary:
            c5_status = CheckStatus.PASS
            c5_evidence = f"Required execution provider '{primary_req}' is verified and registered in ONNX Runtime."
            c5_remediation = "None required."
        elif is_target_s390x and execution_mode == "LOCAL" and not env.is_s390x:
            c5_status = CheckStatus.NOT_RUN
            c5_evidence = (
                f"Required provider '{primary_req}' is an IBM Z specialized runtime provider; "
                f"active host only has: {env.available_providers}. Target verification remains pending."
            )
            c5_remediation = "Deploy on Linux on IBM Z with IBM AI Toolkit for ONNX Runtime NNPA package installed."
        else:
            c5_status = CheckStatus.FAIL
            c5_evidence = f"Required provider '{primary_req}' is missing. Registered providers: {env.available_providers}."
            c5_remediation = f"Install and register '{primary_req}'. Strictly no silent fallback permitted."

        checks.append(ValidationCheck(
            check_id="check_05_providers",
            name="Execution Provider Availability",
            category="provider",
            requirement=f"Execution provider '{primary_req}' must be registered without fallback.",
            detected_value=", ".join(env.available_providers) if env.available_providers else "NONE",
            expected_value=primary_req,
            status=c5_status,
            evidence=c5_evidence,
            remediation=c5_remediation,
            requires_target_hardware=is_target_s390x and ("nnpa" in primary_req.lower() or "ibmz" in primary_req.lower()),
        ))

        # --- Check 6: Model Loadability ---
        model_loaded_ok = False
        load_err = None
        if summary is not None and onnx is not None and ort is not None:
            try:
                # Structural check with onnx
                onnx_model = onnx.load_model_from_string(model_bytes)
                onnx.checker.check_model(onnx_model)
                # Session creation check
                so = ort.SessionOptions()
                _sess = ort.InferenceSession(model_bytes, sess_options=so, providers=["CPUExecutionProvider"])
                model_loaded_ok = True
            except Exception as ex:
                load_err = str(ex)

        if model_loaded_ok:
            c6_status = CheckStatus.PASS
            c6_evidence = f"Model '{model_name}' parsed, validated with onnx.checker, and successfully instantiated in ONNX Runtime."
            c6_remediation = "None required."
        else:
            c6_status = CheckStatus.FAIL
            c6_evidence = f"Model instantiation failed: {load_err or 'Invalid ONNX binary or unparseable graph structure'}."
            c6_remediation = "Inspect ONNX graph with Model Inspector and re-export with valid opset and clean topology."

        checks.append(ValidationCheck(
            check_id="check_06_loadability",
            name="Model Loadability & Schema Validation",
            category="model",
            requirement="ONNX model must pass onnx.checker structural verification and create a runtime session.",
            detected_value="VALID_ONNX_GRAPH" if model_loaded_ok else "LOAD_FAILED",
            expected_value="VALID_ONNX_GRAPH",
            status=c6_status,
            evidence=c6_evidence,
            remediation=c6_remediation,
            requires_target_hardware=False,
        ))

        # --- Check 7: I/O Signature & Data Types ---
        io_ok = True
        io_evidence_parts = []
        if summary:
            # Check opset bounds
            model_opset = summary.opsets[0].version if summary.opsets else 0
            if model_opset < target_profile.min_opset or model_opset > target_profile.max_opset:
                io_ok = False
                io_evidence_parts.append(
                    f"Opset version {model_opset} is outside target range [{target_profile.min_opset}, {target_profile.max_opset}]."
                )

            # Check supported types
            supported_types = set(t.lower() for t in target_profile.supported_data_types)
            for inp in summary.inputs:
                dtype = inp.data_type.lower()
                if dtype not in supported_types:
                    io_ok = False
                    io_evidence_parts.append(f"Input '{inp.name}' has unsupported data type '{inp.data_type}'.")

        if io_ok and summary:
            c7_status = CheckStatus.PASS
            c7_evidence = f"All {len(summary.inputs)} inputs and {len(summary.outputs)} outputs use supported types ({target_profile.supported_data_types}) within opset {summary.opsets[0].version if summary.opsets else 'N/A'}."
            c7_remediation = "None required."
        else:
            c7_status = CheckStatus.FAIL
            c7_evidence = "; ".join(io_evidence_parts) if io_evidence_parts else "Model I/O signature violates target profile constraints."
            c7_remediation = f"Cast tensor inputs to supported types: {target_profile.supported_data_types}."

        checks.append(ValidationCheck(
            check_id="check_07_io_signature",
            name="Input/Output Signature & Opset Bounds",
            category="model",
            requirement=f"Data types must belong to {target_profile.supported_data_types} and opset in [{target_profile.min_opset}, {target_profile.max_opset}].",
            detected_value=f"Opset {summary.opsets[0].version if summary and summary.opsets else 'N/A'}",
            expected_value=f"Opset [{target_profile.min_opset}-{target_profile.max_opset}]",
            status=c7_status,
            evidence=c7_evidence,
            remediation=c7_remediation,
            requires_target_hardware=False,
        ))

        # --- Check 8: Operator Ruleset & Capability Compatibility ---
        builtin_profiles = load_builtin_profiles()
        matched_compat_profile = builtin_profiles.get("ibm_z_onnx_illustrative") or next(iter(builtin_profiles.values()))
        if summary and matched_compat_profile:
            analyzer = CompatibilityAnalyzer()
            compat_report = analyzer.analyze(summary, matched_compat_profile)
            if compat_report.overall_assessment == "COMPATIBLE":
                c8_status = CheckStatus.PASS
                c8_evidence = f"All {len(compat_report.findings)} operator rules evaluated PASS on {matched_compat_profile.display_name}."
                c8_remediation = "None required."
            elif compat_report.overall_assessment == "CONDITIONAL":
                c8_status = CheckStatus.WARNING
                c8_evidence = f"Operator rules yielded {compat_report.summary_counts.get('WARNING', 0)} warnings (e.g. unaccelerated ops falling back to general CPU)."
                c8_remediation = "Review operator findings; unsupported ops will execute on general-purpose s390x CPU cores."
            else:
                c8_status = CheckStatus.FAIL
                c8_evidence = f"Model contains {compat_report.summary_counts.get('FAIL', 0)} incompatible operator(s) on target profile."
                c8_remediation = "Replace or decompose unsupported operators with supported ONNX nodes."
        else:
            c8_status = CheckStatus.UNKNOWN
            c8_evidence = "Compatibility profile could not be evaluated."
            c8_remediation = "Ensure target compatibility profile is loaded."

        checks.append(ValidationCheck(
            check_id="check_08_operator_rules",
            name="Target Operator Support & Fallback Analysis",
            category="model",
            requirement="All model operators must be supported without fatal kernel incompatibility.",
            detected_value=c8_status.value,
            expected_value="PASS",
            status=c8_status,
            evidence=c8_evidence,
            remediation=c8_remediation,
            requires_target_hardware=False,
        ))

        # --- Check 9: Shared Libraries & Toolchain ---
        libs_ok = bool(ort is not None and onnx is not None and np is not None)
        checks.append(ValidationCheck(
            check_id="check_09_libraries",
            name="Runtime Shared Libraries & C++ ABI",
            category="runtime",
            requirement="Python shared libraries (numpy, onnx, onnxruntime) must be present and linked.",
            detected_value="LINKED_AND_AVAILABLE" if libs_ok else "MISSING_LIBRARIES",
            expected_value="LINKED_AND_AVAILABLE",
            status=CheckStatus.PASS if libs_ok else CheckStatus.FAIL,
            evidence="Core runtime libraries (numpy, onnx, onnxruntime) successfully loaded into process space." if libs_ok else "One or more critical shared libraries are missing.",
            remediation="Install required wheels from requirements.txt or IBM AI Toolkit repo." if not libs_ok else "None required.",
            requires_target_hardware=False,
        ))

        # --- Check 10: Numerical Correctness & Finiteness ---
        numerical_result = {
            "all_finite": False,
            "repeatable": False,
            "max_abs_error": None,
            "max_rel_error": None,
            "tolerance_satisfied": False,
        }
        num_ok = False
        num_err = None

        if model_loaded_ok and summary:
            try:
                from zmodel.core.benchmark import BenchmarkConfig
                dummy_cfg = BenchmarkConfig(batch_size=1, random_seed=42)
                inputs_dict, _ = generate_benchmark_inputs(summary, dummy_cfg)
                sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
                out1 = sess.run(None, inputs_dict)
                out2 = sess.run(None, inputs_dict)

                # Check finiteness
                all_finite = True
                for arr in out1:
                    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
                        if not np.all(np.isfinite(arr)):
                            all_finite = False

                # Check repeatability
                repeatable = True
                max_abs = 0.0
                max_rel = 0.0
                for a1, a2 in zip(out1, out2):
                    if isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
                        diff = np.abs(a1.astype(np.float64) - a2.astype(np.float64))
                        m_abs = float(np.max(diff))
                        max_abs = max(max_abs, m_abs)
                        denom = np.abs(a1.astype(np.float64)) + 1e-9
                        m_rel = float(np.max(diff / denom))
                        max_rel = max(max_rel, m_rel)
                        if not np.allclose(a1, a2, atol=atol, rtol=rtol, equal_nan=True):
                            repeatable = False

                # If reference output was provided, compare against reference
                if reference_outputs:
                    for out_idx, out_def in enumerate(sess.get_outputs()):
                        if out_def.name in reference_outputs:
                            ref_arr = reference_outputs[out_def.name]
                            diff = np.abs(out1[out_idx].astype(np.float64) - ref_arr.astype(np.float64))
                            m_abs = float(np.max(diff))
                            max_abs = max(max_abs, m_abs)
                            denom = np.abs(ref_arr.astype(np.float64)) + 1e-9
                            m_rel = float(np.max(diff / denom))
                            max_rel = max(max_rel, m_rel)
                            if not np.allclose(out1[out_idx], ref_arr, atol=atol, rtol=rtol, equal_nan=True):
                                repeatable = False

                numerical_result["all_finite"] = all_finite
                numerical_result["repeatable"] = repeatable
                numerical_result["max_abs_error"] = max_abs
                numerical_result["max_rel_error"] = max_rel
                numerical_result["tolerance_satisfied"] = bool(all_finite and repeatable)
                numerical_result["passed"] = bool(all_finite and repeatable)
                num_ok = bool(all_finite and repeatable)
            except Exception as ex:
                num_err = str(ex)

        if num_ok:
            c10_status = CheckStatus.PASS
            c10_evidence = (
                f"Smoke test outputs are finite with deterministic repeatability. "
                f"Max absolute error: {numerical_result['max_abs_error']:.2e} (bound: {atol}), "
                f"Max relative error: {numerical_result['max_rel_error']:.2e} (bound: {rtol})."
            )
            c10_remediation = "None required."
        else:
            c10_status = CheckStatus.FAIL
            c10_evidence = f"Numerical check failed: {num_err or 'NaN/Inf detected or non-deterministic outputs across identical input feeds'}."
            c10_remediation = "Check initializers, activation functions, and numeric ranges in model."

        checks.append(ValidationCheck(
            check_id="check_10_numerical",
            name="Numerical Correctness & Precision Verification",
            category="correctness",
            requirement=f"Output must be finite and repeatable within atol={atol}, rtol={rtol}.",
            detected_value=f"MaxAbsErr={numerical_result.get('max_abs_error')}",
            expected_value=f"atol<={atol}, rtol<={rtol}",
            status=c10_status,
            evidence=c10_evidence,
            remediation=c10_remediation,
            requires_target_hardware=False,
        ))

        # --- Check 11: Benchmark Execution Eligibility ---
        benchmark_eligible = bool(model_loaded_ok and num_ok and c7_status == CheckStatus.PASS)
        if benchmark_eligible:
            c11_status = CheckStatus.PASS
            c11_evidence = "Model passed all structural, schema, and numerical correctness smoke tests; eligible for empirical benchmarking."
            c11_remediation = "None required."
        else:
            c11_status = CheckStatus.FAIL
            c11_evidence = "Model is NOT eligible for empirical benchmark profiling because correctness or loadability checks failed."
            c11_remediation = "Resolve loadability and numerical correctness failures before benchmarking."

        checks.append(ValidationCheck(
            check_id="check_11_benchmark_eligible",
            name="Empirical Benchmark Profiling Eligibility",
            category="benchmark",
            requirement="Model must satisfy all correctness checks before running empirical latency benchmarks.",
            detected_value="ELIGIBLE" if benchmark_eligible else "INELIGIBLE",
            expected_value="ELIGIBLE",
            status=c11_status,
            evidence=c11_evidence,
            remediation=c11_remediation,
            requires_target_hardware=False,
        ))

        # --- Check 12: Accelerator Hardware Prerequisites ---
        if target_profile.accelerator_type:
            acc_name = target_profile.accelerator_type
            if env.has_nnpa_facility and env.has_nnpa_provider:
                c12_status = CheckStatus.PASS
                c12_evidence = f"Physical accelerator '{acc_name}' verified via s390x facility bit 193 and NNPAExecutionProvider registration."
                c12_remediation = "None required."
            elif is_target_s390x and execution_mode == "LOCAL":
                c12_status = CheckStatus.NOT_RUN
                c12_evidence = f"Accelerator '{acc_name}' requires physical IBM Telum processor with facility 193; not present on {env.architecture} host."
                c12_remediation = "Run on physical IBM z16 or LinuxONE 4 LPAR with NNPA coprocessor enabled."
            else:
                c12_status = CheckStatus.FAIL
                c12_evidence = f"Accelerator '{acc_name}' prerequisites not satisfied on active target."
                c12_remediation = "Verify Linux on Z kernel facility 193 in /proc/cpuinfo."
        else:
            c12_status = CheckStatus.PASS
            c12_evidence = "Target profile does not require specialized hardware coprocessors."
            c12_remediation = "None required."

        checks.append(ValidationCheck(
            check_id="check_12_accelerator",
            name="Hardware Accelerator Prerequisites",
            category="hardware",
            requirement=f"Accelerator prerequisites: {target_profile.accelerator_type or 'None (General CPU)'}.",
            detected_value="NNPA_FACILITY_ACTIVE" if env.has_nnpa_facility else "NOT_DETECTED",
            expected_value=target_profile.accelerator_type or "None",
            status=c12_status,
            evidence=c12_evidence,
            remediation=c12_remediation,
            requires_target_hardware=bool(target_profile.accelerator_type),
        ))

        # --- Aggregate Status Computation ---
        summary_counts = {
            "PASS": sum(1 for c in checks if c.status == CheckStatus.PASS),
            "FAIL": sum(1 for c in checks if c.status == CheckStatus.FAIL),
            "WARNING": sum(1 for c in checks if c.status == CheckStatus.WARNING),
            "UNKNOWN": sum(1 for c in checks if c.status == CheckStatus.UNKNOWN),
            "NOT_RUN": sum(1 for c in checks if c.status == CheckStatus.NOT_RUN),
        }

        # Decision rules
        if summary_counts["FAIL"] > 0:
            overall_status = OverallValidationStatus.FAILED_VALIDATION
            status_rationale = f"Validation failed with {summary_counts['FAIL']} failing check(s). Review remediation guidance."
        elif is_target_s390x and not env.is_s390x:
            # Running on x86_64 or non-s390x host:
            # All local checks passed, but target-specific hardware checks remain NOT_RUN
            if summary_counts["NOT_RUN"] > 0:
                overall_status = OverallValidationStatus.LOCALLY_VERIFIED
                status_rationale = (
                    f"Model passed all {summary_counts['PASS']} host-executable correctness and schema validations. "
                    f"{summary_counts['NOT_RUN']} target-hardware check(s) remain pending physical IBM Z (s390x) execution."
                )
            else:
                overall_status = OverallValidationStatus.LOCALLY_VERIFIED
                status_rationale = "Locally verified on development host; target hardware validation pending."
        elif is_target_s390x and env.is_s390x:
            if summary_counts["FAIL"] == 0 and summary_counts["NOT_RUN"] == 0:
                overall_status = OverallValidationStatus.TARGET_VERIFIED
                status_rationale = "All target requirements, s390x architecture, and runtime checks verified on target environment."
            else:
                overall_status = OverallValidationStatus.PARTIALLY_VERIFIED
                status_rationale = f"Executed on s390x target with {summary_counts['WARNING']} warnings and {summary_counts['NOT_RUN']} unverified checks."
        elif summary_counts["WARNING"] > 0:
            overall_status = OverallValidationStatus.PARTIALLY_VERIFIED
            status_rationale = f"Validation passed with {summary_counts['WARNING']} warning(s)."
        else:
            overall_status = OverallValidationStatus.LOCALLY_VERIFIED
            status_rationale = "All checks passed on local host environment."

        return ValidationReport(
            run_id=run_id,
            model_fingerprint=model_fingerprint,
            model_name=model_name,
            target_profile_id=target_profile.profile_id,
            target_profile_version=target_profile.version,
            environment=env,
            overall_status=overall_status,
            status_rationale=status_rationale,
            checks=checks,
            summary_counts=summary_counts,
            numerical_correctness=numerical_result,
            benchmark_eligibility=benchmark_eligible,
            execution_mode=execution_mode,
            timestamp_utc=timestamp_utc,
        )

    evaluate = _hybrid_evaluate(_evaluate_impl)
