"""
Architecture-Aware Compatibility Analysis Engine for Z-ModelPilot.

Compares Phase 2 ModelSummary against selectable capability profiles.
Generates explainable, evidence-backed compatibility reports with deterministic assessments.
Strictly separates static capability inspection from runtime validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from zmodel.core.model_loader import ModelSummary
from zmodel.core.profiles import CapabilityProfile, OperatorRule


@dataclass(frozen=True)
class CompatibilityFinding:
    """Individual compatibility finding with explicit evidence and recommendation."""
    finding_id: str
    category: str  # "operator", "opset", "data_type", "graph_feature", "profile_completeness"
    status: str    # "PASS", "FAIL", "WARNING", "UNKNOWN"
    severity: str  # "INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    feature: str
    observed_value: str
    profile_rule: Optional[str]
    explanation: str
    evidence_source: str
    recommended_action: str
    domain: Optional[str] = None
    op_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary representation."""
        return {
            "finding_id": self.finding_id,
            "category": self.category,
            "status": self.status,
            "severity": self.severity,
            "feature": self.feature,
            "observed_value": self.observed_value,
            "profile_rule": self.profile_rule,
            "explanation": self.explanation,
            "evidence_source": self.evidence_source,
            "recommended_action": self.recommended_action,
            "domain": self.domain,
            "op_type": self.op_type,
        }


@dataclass
class CompatibilityReport:
    """Complete, serializable compatibility analysis report."""
    model_name: str
    model_fingerprint: str
    profile_id: str
    profile_name: str
    profile_version: str
    ruleset_version: str
    target_architecture: str
    target_runtime: str
    confidence_classification: str
    analysis_timestamp: str
    overall_assessment: str  # "COMPATIBLE", "CONDITIONAL", "INCOMPATIBLE", "INSUFFICIENT_DATA"
    summary_counts: Dict[str, int]
    findings: List[CompatibilityFinding]
    unresolved_questions: List[str]
    assumptions: List[str]
    recommended_next_steps: List[str]
    is_illustrative_profile: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-safe dictionary (excluding secrets and absolute paths)."""
        return {
            "report_version": "1.0.0",
            "model_metadata": {
                "model_name": self.model_name,
                "model_fingerprint": self.model_fingerprint,
            },
            "profile_metadata": {
                "profile_id": self.profile_id,
                "profile_name": self.profile_name,
                "profile_version": self.profile_version,
                "ruleset_version": self.ruleset_version,
                "target_architecture": self.target_architecture,
                "target_runtime": self.target_runtime,
                "confidence_classification": self.confidence_classification,
                "is_illustrative_profile": self.is_illustrative_profile,
            },
            "analysis_metadata": {
                "timestamp_utc": self.analysis_timestamp,
                "overall_assessment": self.overall_assessment,
                "summary_counts": dict(self.summary_counts),
            },
            "findings": [f.to_dict() for f in self.findings],
            "unresolved_questions": list(self.unresolved_questions),
            "assumptions": list(self.assumptions),
            "recommended_next_steps": list(self.recommended_next_steps),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def determine_overall_assessment(summary_counts: Dict[str, int]) -> str:
    """
    Apply a deterministic assessment policy.
    
    Policy hierarchy:
    1. If fail_count > 0: "INCOMPATIBLE"
       At least one explicitly unsupported feature or strict violation was detected.
    2. If unknown_count > 0: "INSUFFICIENT_DATA"
       At least one operator or feature has no verified profile declaration.
       Absence of information cannot be treated as support.
    3. If warning_count > 0: "CONDITIONAL"
       All features are known and supported, but one or more conditional limitations,
       fallback warnings, or configuration dependencies require attention.
    4. Otherwise: "COMPATIBLE"
       All model features are explicitly supported under verified profile declarations.
    """
    fail_count = summary_counts.get("FAIL", 0)
    unknown_count = summary_counts.get("UNKNOWN", 0)
    warning_count = summary_counts.get("WARNING", 0)

    if fail_count > 0:
        return "INCOMPATIBLE"
    if unknown_count > 0:
        return "INSUFFICIENT_DATA"
    if warning_count > 0:
        return "CONDITIONAL"
    return "COMPATIBLE"


def _normalize_domain(domain: str) -> str:
    """Normalize domain strings so that empty string, 'ai.onnx', and 'ai.onnx (default)' are recognized consistently."""
    if not domain or domain.strip() in ("", "ai.onnx", "ai.onnx (default)"):
        return ""
    return domain.strip()


def _is_domain_supported(domain: str, supported_domains: List[str]) -> bool:
    """Check if domain matches any supported domain in profile."""
    norm_dom = _normalize_domain(domain)
    for sup in supported_domains:
        if _normalize_domain(sup) == norm_dom:
            return True
    return False


class CompatibilityAnalyzer:
    """
    Stateless analysis engine comparing an Inspected ModelSummary against a CapabilityProfile.
    """

    @classmethod
    def analyze(
        cls,
        model: ModelSummary,
        profile: CapabilityProfile,
    ) -> CompatibilityReport:
        """
        Execute full compatibility analysis and return a structured report.
        """
        findings: List[CompatibilityFinding] = []

        # 1. Profile Completeness & Illustrative Declaration Check
        cls._check_profile_completeness(profile, findings)

        # 2. Opset Compatibility Checks
        cls._check_opset_compatibility(model, profile, findings)

        # 3. Operator Support Checks
        cls._check_operators(model, profile, findings)

        # 4. Tensor Data Types Checks
        cls._check_tensor_data_types(model, profile, findings)

        # 5. Graph Features (Dynamic Shapes, External Data) Checks
        cls._check_graph_features(model, profile, findings)

        # Summarize counts
        summary_counts = {
            "TOTAL": len(findings),
            "PASS": sum(1 for f in findings if f.status == "PASS"),
            "FAIL": sum(1 for f in findings if f.status == "FAIL"),
            "WARNING": sum(1 for f in findings if f.status == "WARNING"),
            "UNKNOWN": sum(1 for f in findings if f.status == "UNKNOWN"),
        }

        overall_assessment = determine_overall_assessment(summary_counts)

        # Build assumptions and unresolved questions
        assumptions, unresolved, next_steps = cls._generate_insights(
            model, profile, findings, overall_assessment
        )

        return CompatibilityReport(
            model_name=model.filename or "Unnamed Model",
            model_fingerprint=model.sha256_hash,
            profile_id=profile.profile_id,
            profile_name=profile.display_name,
            profile_version=profile.version,
            ruleset_version=profile.ruleset_version,
            target_architecture=profile.target_architecture,
            target_runtime=profile.target_runtime,
            confidence_classification=profile.confidence_classification,
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            overall_assessment=overall_assessment,
            summary_counts=summary_counts,
            findings=findings,
            unresolved_questions=unresolved,
            assumptions=assumptions,
            recommended_next_steps=next_steps,
            is_illustrative_profile=profile.is_illustrative,
        )

    @classmethod
    def _check_profile_completeness(
        cls, profile: CapabilityProfile, findings: List[CompatibilityFinding]
    ) -> None:
        """Check if profile is illustrative or incomplete."""
        if profile.is_illustrative:
            findings.append(
                CompatibilityFinding(
                    finding_id="FIND-PROF-001",
                    category="profile_completeness",
                    status="WARNING",
                    severity="MEDIUM",
                    feature="Profile Verification Status",
                    observed_value=f"Profile '{profile.display_name}' v{profile.version}",
                    profile_rule=f"Confidence: {profile.confidence_classification}",
                    explanation=(
                        f"Target profile '{profile.display_name}' is an illustrative draft ruleset. "
                        "It demonstrates target architecture capability concepts but does NOT constitute "
                        "authoritative hardware verification or official vendor certification."
                    ),
                    evidence_source=profile.evidence_source,
                    recommended_action=(
                        "Treat compatibility findings as guidance only. Execute empirical tests "
                        "on actual target hardware before making deployment decisions."
                    ),
                )
            )

    @classmethod
    def _check_opset_compatibility(
        cls, model: ModelSummary, profile: CapabilityProfile, findings: List[CompatibilityFinding]
    ) -> None:
        """Evaluate model opset imports against profile domain and version limits."""
        for idx, opset in enumerate(model.opsets, start=1):
            domain = opset.domain
            version = opset.version
            finding_id = f"FIND-OPSET-{idx:03d}"

            # Check domain support
            if not _is_domain_supported(domain, profile.supported_domains):
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="opset",
                        status="FAIL",
                        severity="HIGH",
                        feature=f"Opset Domain '{domain}'",
                        observed_value=f"Domain: '{domain}', Version: {version}",
                        profile_rule=f"Supported domains: {profile.supported_domains}",
                        explanation=(
                            f"Operator domain '{domain}' is not supported by target profile "
                            f"'{profile.display_name}'."
                        ),
                        evidence_source=profile.evidence_source,
                        recommended_action=(
                            "Convert or export the model using standard supported domains, or "
                            "configure custom operator domain extensions on the target runtime."
                        ),
                        domain=domain,
                    )
                )
                continue

            # Check version range
            if version < profile.min_opset or version > profile.max_opset:
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="opset",
                        status="FAIL",
                        severity="HIGH",
                        feature=f"Opset Version for '{domain or 'ai.onnx'}'",
                        observed_value=f"Opset {version}",
                        profile_rule=f"Supported opset range: [{profile.min_opset}, {profile.max_opset}]",
                        explanation=(
                            f"Model opset {version} is outside the profile supported range "
                            f"[{profile.min_opset}, {profile.max_opset}]."
                        ),
                        evidence_source=profile.evidence_source,
                        recommended_action=(
                            f"Re-export or convert the model to an opset within "
                            f"[{profile.min_opset}, {profile.max_opset}]."
                        ),
                        domain=domain,
                    )
                )
            else:
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="opset",
                        status="PASS",
                        severity="INFO",
                        feature=f"Opset Version for '{domain or 'ai.onnx'}'",
                        observed_value=f"Opset {version}",
                        profile_rule=f"Supported opset range: [{profile.min_opset}, {profile.max_opset}]",
                        explanation=(
                            f"Model opset {version} for domain '{domain}' is within supported range "
                            f"[{profile.min_opset}, {profile.max_opset}]."
                        ),
                        evidence_source=profile.evidence_source,
                        recommended_action="No action required.",
                        domain=domain,
                    )
                )

    @classmethod
    def _check_operators(
        cls, model: ModelSummary, profile: CapabilityProfile, findings: List[CompatibilityFinding]
    ) -> None:
        """Evaluate each distinct operator type present in the inspected model graph."""
        # Find model's primary opset version for default/ai.onnx domain
        model_onnx_opset = 1
        for opset in model.opsets:
            if _normalize_domain(opset.domain) == "":
                model_onnx_opset = opset.version
                break

        # Map each unique op_type to its total count and observed domain(s)
        op_domains: Dict[str, Set[str]] = {}
        for node in model.nodes:
            op_domains.setdefault(node.op_type, set()).add(node.domain)

        for idx, (op_type, count) in enumerate(model.op_counts.items(), start=1):
            finding_id = f"FIND-OP-{idx:03d}"
            domains = sorted(op_domains.get(op_type, {""}))
            domain_str = ", ".join(domains) if domains else ""

            # Check if operator exists in profile rules
            if op_type not in profile.operator_rules:
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="operator",
                        status="UNKNOWN",
                        severity="MEDIUM",
                        feature=f"Operator: {op_type}",
                        observed_value=f"{count} occurrence(s), domain='{domain_str}'",
                        profile_rule=None,
                        explanation=(
                            f"Profile '{profile.display_name}' has no verified declaration or rule for "
                            f"operator '{op_type}'. Missing rule cannot be assumed as supported."
                        ),
                        evidence_source=profile.evidence_source,
                        recommended_action=(
                            f"Consult official target runtime documentation for operator '{op_type}', "
                            "or add a verified rule to the profile ruleset."
                        ),
                        domain=domain_str,
                        op_type=op_type,
                    )
                )
                continue

            rule = profile.operator_rules[op_type]

            # Check operator-specific version constraints
            if rule.min_version is not None and model_onnx_opset < rule.min_version:
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="operator",
                        status="FAIL",
                        severity="HIGH",
                        feature=f"Operator: {op_type} Version Constraint",
                        observed_value=f"Model opset={model_onnx_opset}, required min={rule.min_version}",
                        profile_rule=f"Min version: {rule.min_version}, Max version: {rule.max_version or 'any'}",
                        explanation=(
                            f"Operator '{op_type}' requires opset version >= {rule.min_version}, "
                            f"but model uses opset {model_onnx_opset}."
                        ),
                        evidence_source=rule.evidence or profile.evidence_source,
                        recommended_action=f"Upgrade model opset to at least {rule.min_version}.",
                        domain=domain_str,
                        op_type=op_type,
                    )
                )
                continue

            if rule.max_version is not None and model_onnx_opset > rule.max_version:
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="operator",
                        status="FAIL",
                        severity="HIGH",
                        feature=f"Operator: {op_type} Version Constraint",
                        observed_value=f"Model opset={model_onnx_opset}, allowed max={rule.max_version}",
                        profile_rule=f"Min version: {rule.min_version or 'any'}, Max version: {rule.max_version}",
                        explanation=(
                            f"Operator '{op_type}' in target profile is only supported up to opset "
                            f"{rule.max_version}, but model uses opset {model_onnx_opset}."
                        ),
                        evidence_source=rule.evidence or profile.evidence_source,
                        recommended_action=f"Downgrade or re-export model opset to at most {rule.max_version}.",
                        domain=domain_str,
                        op_type=op_type,
                    )
                )
                continue

            # Standard rule status
            status = rule.status
            if status == "PASS":
                severity = "INFO"
                action = "No action required."
            elif status == "FAIL":
                severity = "CRITICAL"
                action = f"Replace operator '{op_type}' with a supported equivalent or custom kernel."
            elif status == "WARNING":
                severity = "MEDIUM"
                action = "Review configuration and ensure prerequisite libraries/plugins are active."
            else:
                severity = "LOW"
                action = "Verify operator behavior on target runtime."

            hw_note = (
                f" [Hardware offload: {rule.hardware_accelerated}]"
                if rule.hardware_accelerated
                else ""
            )

            findings.append(
                CompatibilityFinding(
                    finding_id=finding_id,
                    category="operator",
                    status=status,
                    severity=severity,
                    feature=f"Operator: {op_type}",
                    observed_value=f"{count} occurrence(s), domain='{domain_str}'{hw_note}",
                    profile_rule=(
                        f"Status: {rule.status}, Version constraint: "
                        f"[{rule.min_version or 'any'}, {rule.max_version or 'any'}]"
                    ),
                    explanation=rule.notes or f"Evaluated according to profile rule for {op_type}.",
                    evidence_source=rule.evidence or profile.evidence_source,
                    recommended_action=action,
                    domain=domain_str,
                    op_type=op_type,
                )
            )

    @classmethod
    def _check_tensor_data_types(
        cls, model: ModelSummary, profile: CapabilityProfile, findings: List[CompatibilityFinding]
    ) -> None:
        """Check graph input and output tensor data types against profile capabilities."""
        checked_types = set()

        for idx, tensor in enumerate(model.inputs + model.outputs, start=1):
            dtype = tensor.data_type.lower()
            if dtype in checked_types:
                continue
            checked_types.add(dtype)

            finding_id = f"FIND-DTYPE-{idx:03d}"

            # Check if explicitly unsupported
            if dtype in [t.lower() for t in profile.unsupported_data_types]:
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="data_type",
                        status="FAIL",
                        severity="HIGH",
                        feature=f"Data Type '{dtype}'",
                        observed_value=f"Used in tensor '{tensor.name}'",
                        profile_rule=f"Explicitly unsupported data types: {profile.unsupported_data_types}",
                        explanation=(
                            f"Data type '{dtype}' is explicitly declared unsupported by target "
                            f"profile '{profile.display_name}'."
                        ),
                        evidence_source=profile.evidence_source,
                        recommended_action=(
                            f"Cast or quantize tensor '{tensor.name}' to a supported type: "
                            f"{profile.supported_data_types}."
                        ),
                    )
                )
            # Check if in supported types
            elif dtype not in [t.lower() for t in profile.supported_data_types]:
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="data_type",
                        status="UNKNOWN",
                        severity="MEDIUM",
                        feature=f"Data Type '{dtype}'",
                        observed_value=f"Used in tensor '{tensor.name}'",
                        profile_rule=f"Supported data types: {profile.supported_data_types}",
                        explanation=(
                            f"Data type '{dtype}' is not declared in target profile supported types. "
                            "Support is unverified."
                        ),
                        evidence_source=profile.evidence_source,
                        recommended_action=(
                            f"Verify whether target runtime supports '{dtype}', or convert to "
                            f"{profile.supported_data_types}."
                        ),
                    )
                )
            else:
                findings.append(
                    CompatibilityFinding(
                        finding_id=finding_id,
                        category="data_type",
                        status="PASS",
                        severity="INFO",
                        feature=f"Data Type '{dtype}'",
                        observed_value=f"Used in tensor '{tensor.name}'",
                        profile_rule=f"Supported data types: {profile.supported_data_types}",
                        explanation=f"Data type '{dtype}' is explicitly supported by target profile.",
                        evidence_source=profile.evidence_source,
                        recommended_action="No action required.",
                    )
                )

        # Intermediate tensor types observation
        findings.append(
            CompatibilityFinding(
                finding_id="FIND-DTYPE-INT",
                category="data_type",
                status="PASS",
                severity="INFO",
                feature="Intermediate Tensor Types",
                observed_value="Static graph inspection",
                profile_rule="Static boundary verification",
                explanation=(
                    "Static inspection verifies boundary graph inputs and outputs. "
                    "Intermediate tensor types depend on operator type propagation."
                ),
                evidence_source="Z-ModelPilot Static Analysis",
                recommended_action="Verify intermediate activations during empirical runtime validation.",
            )
        )

    @classmethod
    def _check_graph_features(
        cls, model: ModelSummary, profile: CapabilityProfile, findings: List[CompatibilityFinding]
    ) -> None:
        """Inspect dynamic shapes and external tensor references."""
        has_dynamic = any(t.is_dynamic for t in model.inputs + model.outputs)

        # Dynamic dimensions
        if has_dynamic:
            if profile.is_illustrative:
                # IBM Z NNPA often prefers static shapes for coprocessor offload
                findings.append(
                    CompatibilityFinding(
                        finding_id="FIND-FEAT-DYN",
                        category="graph_feature",
                        status="WARNING",
                        severity="MEDIUM",
                        feature="Dynamic Dimensions",
                        observed_value="Graph contains symbolic/dynamic dimensions",
                        profile_rule="Hardware coprocessor shape alignment",
                        explanation=(
                            "Target profile indicates accelerator coprocessor offload (e.g. NNPA) "
                            "is optimal or strictly required with static tensor shapes. Dynamic batching "
                            "or variable sequence lengths may cause fallback to host CPU execution."
                        ),
                        evidence_source=profile.evidence_source,
                        recommended_action=(
                            "If maximum hardware acceleration is desired, consider pinning batch size "
                            "and spatial dimensions to fixed values."
                        ),
                    )
                )
            else:
                findings.append(
                    CompatibilityFinding(
                        finding_id="FIND-FEAT-DYN",
                        category="graph_feature",
                        status="PASS",
                        severity="INFO",
                        feature="Dynamic Dimensions",
                        observed_value="Graph contains symbolic/dynamic dimensions",
                        profile_rule="Dynamic tensor support in CPU Execution Provider",
                        explanation=(
                            "Target runtime CPU provider supports variable/symbolic tensor dimensions."
                        ),
                        evidence_source=profile.evidence_source,
                        recommended_action=(
                            "Ensure host system memory allocation is sized for maximum expected batch size."
                        ),
                    )
                )
        else:
            findings.append(
                CompatibilityFinding(
                    finding_id="FIND-FEAT-DYN",
                    category="graph_feature",
                    status="PASS",
                    severity="INFO",
                    feature="Dynamic Dimensions",
                    observed_value="All input/output tensor dimensions are static",
                    profile_rule="Static dimension support",
                    explanation="Static tensor dimensions are fully supported without dynamic allocation overhead.",
                    evidence_source="Model graph metadata",
                    recommended_action="No action required.",
                )
            )

        # External data
        if model.has_external_data:
            findings.append(
                CompatibilityFinding(
                    finding_id="FIND-FEAT-EXT",
                    category="graph_feature",
                    status="WARNING",
                    severity="HIGH",
                    feature="External Tensor Data",
                    observed_value="Model references external tensor data files",
                    profile_rule="Deployment packaging constraint",
                    explanation=(
                        "Model initializers depend on external tensor storage. The external data files "
                        "must be packaged and located in the same relative path on the target runtime."
                    ),
                    evidence_source="ONNX Model Serialization Standard",
                    recommended_action="Ensure all external tensor data files accompany the model artifact.",
                )
            )

    @classmethod
    def _generate_insights(
        cls,
        model: ModelSummary,
        profile: CapabilityProfile,
        findings: List[CompatibilityFinding],
        overall_assessment: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate transparent assumptions, unresolved questions, and next steps."""
        assumptions: List[str] = [
            "Static analysis evaluates model structure against capability rules without executing inference.",
            f"Target runtime assumes {profile.target_runtime} on {profile.target_architecture}.",
            "Intermediate activation types are assumed compliant if boundary input/output types and operators pass.",
        ]

        unresolved: List[str] = []
        unknown_ops = [f.op_type for f in findings if f.category == "operator" and f.status == "UNKNOWN" and f.op_type]
        if unknown_ops:
            unresolved.append(
                f"Operator support for {', '.join(sorted(set(unknown_ops)))} is unverified in profile ruleset."
            )

        if profile.is_illustrative:
            unresolved.append(
                "Real hardware coprocessor offload (e.g. NNPA) cannot be confirmed without empirical execution on s390x."
            )

        if not unresolved:
            unresolved.append("No unresolved operator or opset questions detected under this profile.")

        next_steps: List[str] = []
        if overall_assessment == "INCOMPATIBLE":
            next_steps.append("Address FAIL findings: replace unsupported operators or re-export to supported opset.")
            next_steps.append("Export compatibility report to JSON and document remediation requirements.")
        elif overall_assessment == "INSUFFICIENT_DATA":
            next_steps.append("Investigate unknown operators in the target runtime official documentation.")
            next_steps.append("Update profile ruleset with authoritative evidence or test individual operators.")
        elif overall_assessment == "CONDITIONAL":
            next_steps.append("Review WARNING items (e.g. dynamic shape fallback or packaging requirements).")
            next_steps.append("Run empirical test in Phase 4 benchmark suite (when activated) to verify behavior.")
        else:
            next_steps.append("Proceed to empirical benchmarking on target environment (Phase 4).")
            next_steps.append("Export compatibility report for deployment audit trails.")

        return assumptions, unresolved, next_steps


# Alias for compatibility with Phase 9 graph optimization engine
CompatibilityEngine = CompatibilityAnalyzer
