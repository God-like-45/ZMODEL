"""Pydantic v2 API request and response schemas for Z-ModelPilot REST endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


def utc_now_str() -> str:
    """Return ISO format UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# Health Schemas
# ============================================================================

class LiveResponse(BaseModel):
    status: str = Field("alive", description="Process liveness state")
    api_version: str = Field(..., description="API semantic version")
    timestamp_utc: str = Field(default_factory=utc_now_str, description="Current UTC timestamp")


class ReadyResponse(BaseModel):
    status: str = Field(..., description="Service readiness status: ready, degraded, or not_ready")
    database_connected: bool = Field(..., description="Whether database connection check succeeded")
    dialect: str = Field(..., description="Underlying database dialect (postgresql, sqlite)")
    database_url_masked: str = Field(..., description="Target database URL with masked credentials")
    table_count: int = Field(..., description="Number of detected schema tables")
    tables: List[str] = Field(default_factory=list, description="List of detected table names")
    error: Optional[str] = Field(None, description="Database connection error if present")
    timestamp_utc: str = Field(default_factory=utc_now_str, description="Current UTC timestamp")


# ============================================================================
# Model Inspection Schemas
# ============================================================================

class TensorInfoSchema(BaseModel):
    name: str = Field(..., description="Tensor parameter or feature name")
    data_type: str = Field(..., description="Human-readable data type (e.g. float32, int64)")
    shape: List[Union[int, str]] = Field(..., description="Tensor dimensions, with symbolic names for dynamic axes")
    rank: Optional[int] = Field(None, description="Tensor rank (dimensionality)")
    is_dynamic: bool = Field(..., description="Whether tensor has dynamic or symbolic dimensions")
    dynamic_dim_names: List[str] = Field(default_factory=list, description="Names of symbolic dimensions")
    dimension_description: str = Field(..., description="Formatted description of shape dimensions")


class WeightInfoSchema(BaseModel):
    name: str = Field(..., description="Weight tensor initializer name")
    data_type: str = Field(..., description="Weight data type")
    shape: List[int] = Field(..., description="Shape dimensions")
    num_elements: int = Field(..., description="Total parameter count")
    size_bytes: int = Field(..., description="Total weight size in bytes")
    has_external_data: bool = Field(False, description="Whether weights are stored externally")


class NodeInfoSchema(BaseModel):
    name: str = Field(..., description="Node name")
    op_type: str = Field(..., description="ONNX operator type (e.g. Conv, MatMul, Relu)")
    domain: str = Field(..., description="Operator domain")
    inputs: List[str] = Field(default_factory=list, description="Input tensor names")
    outputs: List[str] = Field(default_factory=list, description="Output tensor names")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Operator attributes")


class InspectionWarningSchema(BaseModel):
    code: str = Field(..., description="Machine-readable warning code")
    level: str = Field(..., description="Severity level: info, advisory, or warning")
    title: str = Field(..., description="Human-readable warning title")
    message: str = Field(..., description="Detailed explanation of the observation")
    impact: str = Field(..., description="Operational impact on deployment")


class ModelMetadataSchema(BaseModel):
    producer_name: str = Field(..., description="Name of the model export tool")
    producer_version: str = Field(..., description="Version of the model export tool")
    model_version: int = Field(..., description="Model version integer")
    ir_version: int = Field(..., description="ONNX IR version")
    doc_string: str = Field(..., description="Documentation string if present")
    domain: str = Field(..., description="Model domain")
    custom_metadata: Dict[str, str] = Field(default_factory=dict, description="Custom metadata properties")


class OpsetInfoSchema(BaseModel):
    domain: str = Field(..., description="Opset domain")
    version: int = Field(..., description="Opset version integer")


class ModelInspectionResponse(BaseModel):
    filename: str = Field(..., description="Original model filename")
    file_size_bytes: int = Field(..., description="File size in bytes")
    sha256_fingerprint: str = Field(..., description="Exact SHA-256 fingerprint hex string")
    ir_version: int = Field(..., description="ONNX Intermediate Representation version")
    producer_name: str = Field(..., description="Producer software name")
    producer_version: str = Field(..., description="Producer software version")
    metadata: ModelMetadataSchema = Field(..., description="Extracted model metadata")
    opset_imports: List[OpsetInfoSchema] = Field(..., description="Imported ONNX opset domains and versions")
    inputs: List[TensorInfoSchema] = Field(..., description="Graph input tensor signatures")
    outputs: List[TensorInfoSchema] = Field(..., description="Graph output tensor signatures")
    weights: List[WeightInfoSchema] = Field(..., description="Initializers and weight summaries")
    total_nodes: int = Field(..., description="Total computational graph nodes")
    total_parameters: int = Field(..., description="Total weight parameters")
    total_weight_bytes: int = Field(..., description="Total weight size in bytes")
    operator_counts: Dict[str, int] = Field(..., description="Frequency counts per operator type")
    has_external_data: bool = Field(..., description="Whether model relies on external tensor files")
    external_data_tensors: List[str] = Field(default_factory=list, description="Tensors with external data")
    warnings: List[InspectionWarningSchema] = Field(default_factory=list, description="Graph inspection warnings")


class ModelListItem(BaseModel):
    fingerprint: str = Field(..., description="SHA-256 fingerprint")
    original_filename: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., description="Size in bytes")
    ir_version: Optional[int] = Field(None, description="ONNX IR version")
    producer_name: Optional[str] = Field(None, description="Producer name")
    first_seen_at: Optional[str] = Field(None, description="First registered timestamp")
    last_seen_at: Optional[str] = Field(None, description="Last activity timestamp")


class ModelListResponse(BaseModel):
    models: List[ModelListItem] = Field(..., description="List of registered models")
    total: int = Field(..., description="Total count matching query")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")


# ============================================================================
# Compatibility Schemas
# ============================================================================

class ProfileSummaryResponse(BaseModel):
    profile_id: str = Field(..., description="Unique profile identifier")
    name: str = Field(..., description="Display name")
    version: str = Field(..., description="Profile semantic version")
    ruleset_version: str = Field(..., description="Underlying ruleset version")
    target_device: str = Field(..., description="Target hardware or runtime device")
    description: str = Field(..., description="Summary of profile capabilities")
    is_illustrative: bool = Field(..., description="Whether this profile is an illustrative draft")
    supported_opset_range: Dict[str, int] = Field(..., description="Min and max supported opset versions")
    operator_rule_count: int = Field(..., description="Number of explicitly defined operator rules")


class CompatibilityFindingSchema(BaseModel):
    finding_id: str = Field(..., description="Unique finding ID")
    category: str = Field(..., description="Finding category: operator, opset, data_type, graph_feature")
    status: str = Field(..., description="Evaluation status: PASS, FAIL, WARNING, UNKNOWN")
    severity: str = Field(..., description="Severity level: INFO, LOW, MEDIUM, HIGH, CRITICAL")
    feature: str = Field(..., description="Evaluated feature or operator")
    observed_value: str = Field(..., description="Observed value in inspected model")
    profile_rule: Optional[str] = Field(None, description="Applicable profile rule constraint")
    explanation: str = Field(..., description="Detailed explanation of finding")
    evidence_source: str = Field(..., description="Source of evidence for assessment")
    recommended_action: str = Field(..., description="Recommended remediation or next step")
    domain: Optional[str] = Field(None, description="Operator domain")
    op_type: Optional[str] = Field(None, description="Operator type")


class CompatibilityAnalyzeRequest(BaseModel):
    model_fingerprint: Optional[str] = Field(None, description="SHA-256 fingerprint of previously uploaded model")
    sample_model_name: Optional[str] = Field(None, description="Sample model filename (e.g. linear_classifier.onnx)")
    profile_id: str = Field("local_onnx_runtime_cpu", description="Target capability profile ID")


class CompatibilityAnalyzeResponse(BaseModel):
    assessment_id: str = Field(..., description="Unique assessment ID")
    model_name: str = Field(..., description="Inspected model filename")
    model_fingerprint: str = Field(..., description="Model SHA-256 fingerprint")
    profile_id: str = Field(..., description="Evaluated capability profile ID")
    profile_name: str = Field(..., description="Evaluated capability profile name")
    profile_version: str = Field(..., description="Capability profile version")
    ruleset_version: str = Field(..., description="Capability ruleset version")
    overall_assessment: str = Field(..., description="Deterministic overall assessment: PASS, PASS_WITH_WARNINGS, FAIL, UNCERTAIN")
    summary_counts: Dict[str, int] = Field(..., description="Counts of findings by status and category")
    findings: List[CompatibilityFindingSchema] = Field(..., description="Itemized explainable findings")
    assumptions: List[str] = Field(default_factory=list, description="Explicit assumptions underlying evaluation")
    unresolved_questions: List[str] = Field(default_factory=list, description="Unresolved architecture questions")
    recommended_next_steps: List[str] = Field(default_factory=list, description="Recommended remediation actions")
    is_illustrative_profile: bool = Field(..., description="Whether target profile is an illustrative draft")
    timestamp_utc: str = Field(..., description="Execution timestamp in UTC")


# ============================================================================
# Benchmark Schemas
# ============================================================================

class BenchmarkConfigSchema(BaseModel):
    batch_size: int = Field(1, ge=1, le=128, description="Inference batch size")
    execution_provider: str = Field("CPUExecutionProvider", description="ONNX Runtime execution provider")
    intra_op_num_threads: Optional[int] = Field(None, ge=1, le=64, description="Intra-op thread count")
    inter_op_num_threads: Optional[int] = Field(None, ge=1, le=64, description="Inter-op thread count")
    warmup_iterations: int = Field(5, ge=0, le=100, description="Warmup inference iterations")
    measured_iterations: int = Field(20, ge=1, le=1000, description="Measured inference iterations")
    repeated_rounds: int = Field(1, ge=1, le=10, description="Rounds of measured iterations")
    random_seed: int = Field(42, description="Random seed for deterministic input generation")
    dynamic_dimensions: Optional[Dict[str, int]] = Field(None, description="Explicit values for symbolic dimensions")


class BenchmarkSubmitRequest(BaseModel):
    model_fingerprint: Optional[str] = Field(None, description="SHA-256 fingerprint of model in cache/DB")
    sample_model_name: Optional[str] = Field(None, description="Sample model name (e.g. linear_classifier.onnx)")
    config: BenchmarkConfigSchema = Field(default_factory=BenchmarkConfigSchema, description="Benchmark parameters")
    execution_backend: Optional[str] = Field(None, description="Execution backend: 'local' or 'kubernetes'")
    resource_profile: Optional[str] = Field(None, description="Compute resource profile: 'small_cpu', 'standard_cpu', 'large_opt'")


class BenchmarkJobResponse(BaseModel):
    job_id: str = Field(..., description="Asynchronous job tracking identifier")
    benchmark_id: Optional[str] = Field(None, description="Planned benchmark run ID")
    status: str = Field("QUEUED", description="Initial job status")
    created_at: str = Field(default_factory=utc_now_str, description="Creation timestamp")
    message: str = Field("Benchmark execution job queued", description="Status message")


class BenchmarkStatusResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    benchmark_id: Optional[str] = Field(None, description="Persisted benchmark run ID")
    status: str = Field(..., description="Current status: QUEUED, RUNNING, COMPLETED, FAILED")
    created_at: str = Field(..., description="Creation timestamp")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    progress_pct: int = Field(..., description="Progress percentage (0-100)")
    progress_message: str = Field(..., description="Progress details")
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Completed benchmark report if ready")


class BenchmarkRunItem(BaseModel):
    id: str = Field(..., description="Benchmark run ID")
    model_fingerprint: str = Field(..., description="Model SHA-256 fingerprint")
    execution_provider: str = Field(..., description="Execution provider")
    batch_size: int = Field(..., description="Batch size")
    intra_op_threads: Optional[int] = Field(None, description="Intra-op threads")
    status: str = Field(..., description="Execution status: SUCCESS or FAILED")
    latency_median_ms: Optional[float] = Field(None, description="Median latency in milliseconds")
    throughput_samples_sec: Optional[float] = Field(None, description="Throughput in samples/sec")
    peak_rss_mb: Optional[float] = Field(None, description="Peak RSS memory in MB")
    correctness_status: Optional[str] = Field(None, description="Correctness check status")
    created_at: str = Field(..., description="Creation timestamp")


class BenchmarkListResponse(BaseModel):
    runs: List[BenchmarkRunItem] = Field(..., description="List of benchmark executions")
    total: int = Field(..., description="Total runs matching query")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")


# ============================================================================
# Optimization Schemas
# ============================================================================

class HardConstraintSchema(BaseModel):
    name: str = Field(..., description="Constraint label")
    metric_field: str = Field(..., description="Target metric: median_latency_ms, p95_latency_ms, throughput_inf_sec, etc.")
    operator: str = Field(..., description="Comparison operator: <=, <, >=, >, ==")
    threshold: float = Field(..., description="Threshold numeric value")
    unit: str = Field("", description="Unit string (e.g. ms, inf/sec, MB)")


class OptimizationSubmitRequest(BaseModel):
    model_fingerprint: Optional[str] = Field(None, description="SHA-256 fingerprint of model")
    sample_model_name: Optional[str] = Field(None, description="Sample model name (e.g. linear_classifier.onnx)")
    profile_id: str = Field("local_onnx_runtime_cpu", description="Target capability profile ID")
    primary_objective: str = Field(
        "MINIMIZE_MEDIAN_LATENCY",
        description="Optimization objective: MINIMIZE_MEDIAN_LATENCY, MINIMIZE_P95_LATENCY, MAXIMIZE_THROUGHPUT, MINIMIZE_MEMORY, EXPLORE_TRADEOFFS",
    )
    hard_constraints: List[HardConstraintSchema] = Field(default_factory=list, description="Mandatory constraints")
    batch_sizes: List[int] = Field(default_factory=lambda: [1], description="Candidate batch sizes")
    intra_op_threads: List[int] = Field(default_factory=lambda: [1, 2], description="Candidate intra-op thread counts")
    inter_op_threads: List[int] = Field(default_factory=lambda: [1], description="Candidate inter-op thread counts")
    execution_providers: List[str] = Field(default_factory=lambda: ["CPUExecutionProvider"], description="Candidate execution providers")
    warmup_iterations: int = Field(5, ge=0, le=50, description="Warmup iterations per candidate")
    measured_iterations: int = Field(20, ge=1, le=200, description="Measured iterations per candidate")
    repeated_rounds: int = Field(1, ge=1, le=5, description="Benchmark rounds per candidate")
    max_candidates: int = Field(20, ge=1, le=50, description="Maximum candidates to generate")
    execution_backend: Optional[str] = Field(None, description="Execution backend: 'local' or 'kubernetes'")
    resource_profile: Optional[str] = Field(None, description="Compute resource profile: 'standard_cpu', 'large_opt'")


class OptimizationJobResponse(BaseModel):
    job_id: str = Field(..., description="Asynchronous job tracking identifier")
    experiment_id: str = Field(..., description="Planned optimization experiment ID")
    status: str = Field("QUEUED", description="Initial job status")
    created_at: str = Field(default_factory=utc_now_str, description="Creation timestamp")
    message: str = Field("Optimization search job queued", description="Status message")


class OptimizationStatusResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    experiment_id: str = Field(..., description="Optimization experiment ID")
    status: str = Field(..., description="Current status: QUEUED, RUNNING, COMPLETED, FAILED")
    created_at: str = Field(..., description="Creation timestamp")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    progress_pct: int = Field(..., description="Progress percentage (0-100)")
    progress_message: str = Field(..., description="Progress details")
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Completed optimization report if ready")


class CandidateEvaluationItem(BaseModel):
    candidate_id: str = Field(..., description="Candidate ID within experiment")
    benchmark_run_id: Optional[str] = Field(None, description="Associated benchmark run ID")
    status: str = Field(..., description="Candidate status: CANDIDATE, BENCHMARKED, REJECTED, INFEASIBLE, FAILED")
    rank: Optional[int] = Field(None, description="Objective rank (1 = best)")
    is_feasible: bool = Field(..., description="Whether candidate satisfied all hard constraints")
    is_pareto_optimal: bool = Field(..., description="Whether candidate belongs to Pareto frontier")
    is_baseline: bool = Field(..., description="Whether candidate is the baseline reference")
    speedup_vs_baseline: Optional[float] = Field(None, description="Speedup factor vs baseline (e.g. 1.25x)")
    latency_reduction_pct: Optional[float] = Field(None, description="Latency reduction percentage")
    measured_median_latency_ms: Optional[float] = Field(None, description="Empirical median latency in ms")
    measured_throughput_ips: Optional[float] = Field(None, description="Empirical throughput in inf/sec")
    measured_memory_mb: Optional[float] = Field(None, description="Empirical peak RSS in MB")
    configuration: Dict[str, Any] = Field(..., description="Candidate execution configuration")
    rejection_reason: Optional[str] = Field(None, description="Reason if rejected before benchmark")


# ============================================================================
# Phase 12: Advanced Optimization & Enterprise Reporting Schemas
# ============================================================================

class AdvancedOptimizationSubmitRequest(OptimizationSubmitRequest):
    search_strategy: str = Field("GRID", description="Search strategy: GRID, RANDOM, SUCCESSIVE_HALVING")
    recommendation_goal: str = Field("BALANCED_SCORE", description="Decision goal: MINIMIZE_LATENCY, MAXIMIZE_THROUGHPUT, MINIMIZE_MEMORY, BALANCED_SCORE")
    random_seed: int = Field(42, description="Random seed for deterministic reproducibility")
    graph_optimization_levels: Optional[List[str]] = Field(None, description="Graph optimization levels to explore: ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL")
    execution_modes: Optional[List[str]] = Field(None, description="Execution modes to explore: ORT_SEQUENTIAL, ORT_PARALLEL")


class ParetoCandidateItem(BaseModel):
    candidate_id: str = Field(..., description="Candidate ID")
    is_pareto_optimal: bool = Field(..., description="Whether candidate is on the Pareto frontier")
    median_latency_ms: Optional[float] = Field(None, description="Median latency in ms")
    p95_latency_ms: Optional[float] = Field(None, description="P95 latency in ms")
    throughput_inf_sec: Optional[float] = Field(None, description="Throughput in inf/sec")
    peak_memory_mb: Optional[float] = Field(None, description="Peak RSS memory in MB")
    configuration: Dict[str, Any] = Field(..., description="Configuration dict")


class ParetoFrontierResponse(BaseModel):
    experiment_id: str = Field(..., description="Experiment ID")
    pareto_frontier_ids: List[str] = Field(..., description="IDs of candidates on the Pareto frontier")
    frontier_candidates: List[ParetoCandidateItem] = Field(..., description="Detailed Pareto frontier candidates")
    total_evaluated: int = Field(..., description="Total candidates evaluated")
    pareto_count: int = Field(..., description="Count of Pareto optimal candidates")
    tradeoff_explanation: str = Field(..., description="Human-readable trade-off explanation")


class SensitivityAnalysisResponse(BaseModel):
    experiment_id: str = Field(..., description="Experiment ID")
    parameters: Dict[str, Any] = Field(..., description="Per-parameter sensitivity curves and elasticity")
    primary_influencing_parameter: Optional[str] = Field(None, description="Parameter showing highest variance on latency")
    summary_insight: str = Field(..., description="Summary of sensitivity findings")


class RecommendationResponse(BaseModel):
    experiment_id: str = Field(..., description="Experiment ID")
    goal: str = Field(..., description="User-selected recommendation goal")
    recommended_candidate_id: Optional[str] = Field(None, description="Winning candidate ID")
    recommended_config: Optional[Dict[str, Any]] = Field(None, description="Recommended configuration dict")
    is_target_verified: bool = Field(..., description="Whether validated directly on target IBM Z hardware")
    provenance_badge: str = Field(..., description="Hardware provenance badge")
    median_latency_ms: Optional[float] = Field(None, description="Recommended candidate median latency")
    throughput_inf_sec: Optional[float] = Field(None, description="Recommended candidate throughput")
    speedup_vs_baseline: Optional[float] = Field(None, description="Speedup relative to default baseline")
    latency_reduction_pct: Optional[float] = Field(None, description="Latency reduction percentage")
    reliability: Optional[Dict[str, Any]] = Field(None, description="Measurement reliability and dispersion stats")
    rationale: str = Field(..., description="Detailed evidence-backed selection rationale")
    unverified_requirements: List[str] = Field(default_factory=list, description="Target requirements pending hardware access")


class EnterpriseReportResponse(BaseModel):
    experiment_id: str = Field(..., description="Experiment ID")
    format: str = Field("markdown", description="Report format: markdown or json")
    report_content: str = Field(..., description="Complete audit report content")
    sha256_checksum: str = Field(..., description="Cryptographic SHA-256 integrity hash of report content")


# ============================================================================
# Experiments & History Schemas
# ============================================================================

class ExperimentListItem(BaseModel):
    id: str = Field(..., description="Experiment ID")
    model_name: str = Field(..., description="Model original filename")
    model_fingerprint: str = Field(..., description="Model SHA-256 fingerprint")
    profile_id: str = Field(..., description="Target profile ID")
    primary_objective: str = Field(..., description="Optimization objective")
    status: str = Field(..., description="Experiment status")
    best_candidate_id: Optional[str] = Field(None, description="Winning candidate ID")
    summary_counts: Dict[str, int] = Field(default_factory=dict, description="Candidate status counts")
    created_at: str = Field(..., description="Creation timestamp")


class ExperimentListResponse(BaseModel):
    experiments: List[ExperimentListItem] = Field(..., description="List of persisted experiments")
    total: int = Field(..., description="Total count matching query")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")


class BenchmarkComparisonRequest(BaseModel):
    run_id_a: str = Field(..., description="Baseline reference benchmark run ID (Run A)")
    run_id_b: str = Field(..., description="Candidate benchmark run ID to compare against Run A (Run B)")


class BenchmarkComparisonResponse(BaseModel):
    validity: str = Field(..., description="Comparison validity status: VALID, INCOMPATIBLE_MODELS, INCOMPATIBLE_WORKLOAD, INCOMPLETE_DATA")
    disqualification_reasons: List[str] = Field(default_factory=list, description="Reasons if comparison was disqualified")
    run_a_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of Run A configuration and metrics")
    run_b_summary: Optional[Dict[str, Any]] = Field(None, description="Summary of Run B configuration and metrics")
    delta_latency_ms: Optional[float] = Field(None, description="Latency difference (Run B - Run A) in ms")
    speedup_factor: Optional[float] = Field(None, description="Speedup factor (Run A / Run B)")
    latency_reduction_pct: Optional[float] = Field(None, description="Latency reduction percentage")
    delta_throughput_samples_sec: Optional[float] = Field(None, description="Throughput difference (Run B - Run A)")
    comparison_summary: str = Field(..., description="Human-readable comparison narrative")


# ============================================================================
# Error Response Schemas
# ============================================================================

class ErrorDetail(BaseModel):
    field: Optional[str] = Field(None, description="Field name where error occurred")
    message: str = Field(..., description="Error explanation")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="High-level error classification")
    message: str = Field(..., description="Detailed explanation of error")
    details: Optional[List[ErrorDetail]] = Field(None, description="Field-level validation details if applicable")
    timestamp_utc: str = Field(default_factory=utc_now_str, description="Error timestamp")


# ============================================================================
# Graph Optimization Schemas (Phase 9)
# ============================================================================

class GraphOptimizationSubmitRequest(BaseModel):
    """Payload to trigger an asynchronous graph optimization experiment."""
    model_fingerprint: Optional[str] = Field(None, description="SHA-256 fingerprint of preloaded/cached model")
    sample_model_name: Optional[str] = Field(None, description="Sample model filename from sample_models directory")
    selected_transforms: Optional[List[str]] = Field(
        default=["shape_inference", "ort_basic", "ort_extended", "ort_all"],
        description="List of transformation types to execute (shape_inference, ort_basic, ort_extended, ort_all, simplification)",
    )
    target_profile_id: Optional[str] = Field(None, description="Target architecture capability profile ID")
    validation_depth: str = Field("FULL", description="Validation depth: FULL (structural + numerical) or STRUCTURAL_ONLY")
    enable_benchmarking: bool = Field(True, description="Whether to empirically benchmark eligible transformed candidates")
    atol: float = Field(1e-4, description="Absolute numerical tolerance threshold")
    rtol: float = Field(1e-3, description="Relative numerical tolerance threshold")
    test_seed: int = Field(42, description="Random seed for deterministic numerical input tensors")
    execution_backend: Optional[str] = Field(None, description="Execution backend: 'local' or 'kubernetes'")
    resource_profile: Optional[str] = Field(None, description="Compute resource profile: 'graph_opt', 'standard_cpu'")


class GraphOptimizationSubmitResponse(BaseModel):
    job_id: str = Field(..., description="Background job ID for polling status")
    experiment_id: str = Field(..., description="Graph optimization experiment ID")
    status: str = Field(..., description="Initial job status (QUEUED or RUNNING)")
    message: str = Field(..., description="Status message")
    poll_url: str = Field(..., description="URL to poll job execution status")
    results_url: str = Field(..., description="URL to retrieve completed experiment results")


class GraphTransformationCandidateResponse(BaseModel):
    candidate_id: str = Field(..., description="Candidate identifier")
    transform_type: str = Field(..., description="Transformation method type")
    transform_name: str = Field(..., description="Display name of transformation")
    tool_name: str = Field(..., description="Transformation tool name")
    tool_version: str = Field(..., description="Transformation tool version")
    state: str = Field(..., description="Candidate lifecycle state")
    original_hash: str = Field(..., description="Original model SHA-256 fingerprint")
    transformed_hash: Optional[str] = Field(None, description="Transformed model SHA-256 fingerprint")
    artifact_id: Optional[str] = Field(None, description="Persisted artifact ID in artifact store")
    graph_changed: bool = Field(..., description="Whether model graph structure actually changed")
    model_size_bytes_before: int = Field(..., description="Original model size in bytes")
    model_size_bytes_after: Optional[int] = Field(None, description="Transformed model size in bytes")
    size_reduction_pct: Optional[float] = Field(None, description="Percentage size reduction")
    diff_summary: Optional[Dict[str, Any]] = Field(None, description="Node and initializer count differences")
    validation_report: Optional[Dict[str, Any]] = Field(None, description="Structural and numerical validation report")
    compatibility_report: Optional[Dict[str, Any]] = Field(None, description="Target architecture compatibility findings")
    compatibility_delta: Optional[Dict[str, Any]] = Field(None, description="Compatibility delta vs baseline")
    benchmark_report: Optional[Dict[str, Any]] = Field(None, description="Empirical benchmark report")
    speedup_factor: Optional[float] = Field(None, description="Speedup factor vs baseline (orig / trans)")
    latency_reduction_pct: Optional[float] = Field(None, description="Latency reduction percentage")
    throughput_improvement_pct: Optional[float] = Field(None, description="Throughput improvement percentage")
    recommendation_status: str = Field(..., description="Recommendation status category")
    recommendation_rationale: Optional[str] = Field(None, description="Evidence-backed rationale for recommendation")
    rejection_reason: Optional[str] = Field(None, description="Rejection reason if candidate failed validation")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated during transformation or evaluation")
    duration_ms: float = Field(0.0, description="Transformation execution duration in ms")
    artifact_download_url: Optional[str] = Field(None, description="Download URL if artifact was persisted and verified")


class GraphOptimizationExperimentResponse(BaseModel):
    id: str = Field(..., description="Experiment ID")
    model_name: str = Field(..., description="Original model filename")
    model_fingerprint: str = Field(..., description="Original model SHA-256 fingerprint")
    target_profile_id: Optional[str] = Field(None, description="Target architecture profile ID")
    target_profile_name: Optional[str] = Field(None, description="Target architecture profile display name")
    validation_depth: str = Field(..., description="Validation depth applied")
    benchmark_enabled: bool = Field(..., description="Whether empirical benchmarking was enabled")
    status: str = Field(..., description="Overall experiment status (SUCCESS, FAILED, RUNNING)")
    best_candidate_id: Optional[str] = Field(None, description="Winning candidate ID")
    summary_counts: Dict[str, int] = Field(default_factory=dict, description="Summary counts of candidate states")
    overall_recommendation: Optional[str] = Field(None, description="Executive recommendation summary")
    limitations: List[str] = Field(default_factory=list, description="Host vs target architectural limitations")
    error_message: Optional[str] = Field(None, description="Error message if experiment failed")
    created_at: str = Field(..., description="Creation timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    candidates: List[GraphTransformationCandidateResponse] = Field(default_factory=list, description="Evaluated candidates")


class GraphOptimizationStatusResponse(BaseModel):
    job_id: Optional[str] = Field(None, description="Job ID")
    experiment_id: str = Field(..., description="Experiment ID")
    status: str = Field(..., description="Execution status")
    progress_pct: int = Field(0, description="Completion progress percentage (0-100)")
    progress_message: str = Field("Processing", description="Current execution stage description")
    error: Optional[str] = Field(None, description="Error message if job failed")


class WorkerJobResponse(BaseModel):
    id: str = Field(..., description="Unique worker job ID")
    job_type: str = Field(..., description="Job type (benchmark, optimization, graph_optimization)")
    execution_backend: str = Field("local", description="Execution backend (local or kubernetes)")
    model_fingerprint: str = Field(..., description="SHA-256 fingerprint of target ONNX model")
    experiment_id: Optional[str] = Field(None, description="Associated experiment ID")
    run_id: Optional[str] = Field(None, description="Associated run ID")
    status: str = Field(..., description="Job lifecycle state (QUEUED, SUBMITTED, RUNNING, SUCCEEDED, FAILED, TIMED_OUT, CANCELLED)")
    resource_profile: str = Field("standard_cpu", description="Compute resource allocation profile")
    k8s_namespace: Optional[str] = Field(None, description="Kubernetes namespace if deployed to cluster")
    k8s_job_name: Optional[str] = Field(None, description="Kubernetes Job name")
    k8s_pod_name: Optional[str] = Field(None, description="Kubernetes Pod name")
    retry_count: int = Field(0, description="Number of retries attempted")
    max_retries: int = Field(1, description="Maximum allowed retries")
    worker_image: Optional[str] = Field(None, description="Worker container image version/digest")
    queue_delay_seconds: float = Field(0.0, description="Queue and scheduling delay in seconds")
    startup_overhead_seconds: float = Field(0.0, description="Container init and model resolution overhead in seconds")
    warmup_overhead_seconds: float = Field(0.0, description="Warmup duration in seconds")
    inference_duration_seconds: float = Field(0.0, description="Pure model inference duration in seconds")
    total_duration_seconds: float = Field(0.0, description="End-to-end total execution duration in seconds")
    result: Optional[Dict[str, Any]] = Field(None, description="Structured execution result")
    output_artifact_ids: Optional[List[str]] = Field(None, description="List of generated output artifact IDs")
    error_category: Optional[str] = Field(None, description="Sanitized error category if failed")
    error_message: Optional[str] = Field(None, description="Sanitized failure message")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    started_at: Optional[str] = Field(None, description="Execution start timestamp")
    completed_at: Optional[str] = Field(None, description="Execution completion timestamp")


class WorkerJobListResponse(BaseModel):
    jobs: List[WorkerJobResponse] = Field(default_factory=list, description="List of worker jobs")
    total: int = Field(..., description="Total matching jobs")
    limit: int = Field(..., description="Pagination limit")
    offset: int = Field(..., description="Pagination offset")


class WorkerJobCancelResponse(BaseModel):
    job_id: str = Field(..., description="Job ID")
    cancelled: bool = Field(..., description="Whether cancellation was accepted")
    message: str = Field(..., description="Cancellation status message")


class BackendHealthResponse(BaseModel):
    backend_name: str = Field(..., description="Execution backend name (local or kubernetes)")
    is_available: bool = Field(..., description="Whether backend is reachable and healthy")
    details: Dict[str, Any] = Field(default_factory=dict, description="Backend configuration and runtime details")
    error_message: Optional[str] = Field(None, description="Error message if backend is unreachable")


# ==============================================================================
# Phase 11: IBM Z / Linux on IBM Z Validation Schemas
# ==============================================================================

class EnvironmentInventoryResponse(BaseModel):
    architecture: str
    is_s390x: bool
    os_name: str
    os_release: str
    os_version: str
    platform_machine: str
    kernel_version: str
    python_version: str
    python_executable: str
    cpu_model: str
    cpu_count_physical: Optional[int]
    cpu_count_logical: Optional[int]
    memory_total_gb: float
    memory_available_gb: float
    onnx_version: Optional[str]
    onnx_ir_version: Optional[int]
    onnxruntime_version: Optional[str]
    available_providers: List[str]
    has_nnpa_provider: bool
    has_nnpa_facility: bool
    container_runtime: str
    collection_method: str
    timestamp_utc: str
    environment_fingerprint: str


class TargetValidationProfileResponse(BaseModel):
    profile_id: str
    display_name: str
    version: str
    target_architecture: str
    target_os: str
    min_python_version: str
    max_python_version: str
    min_ort_version: Optional[str]
    required_execution_providers: List[str]
    required_facilities: List[str]
    accelerator_type: Optional[str]
    min_opset: int
    max_opset: int
    supported_data_types: List[str]
    requires_target_hardware: bool
    confidence_classification: str
    evidence_source: str
    documentation_url: Optional[str] = None
    known_constraints: List[str] = Field(default_factory=list)


class ValidationEvaluateRequest(BaseModel):
    sample_model_name: Optional[str] = Field(None, description="Preset sample model name from sample_models/")
    model_fingerprint: Optional[str] = Field(None, description="SHA-256 fingerprint of pre-uploaded model")
    target_profile_id: str = Field("ibm_z16_linux_nnpa", description="Target validation profile ID")
    atol: float = Field(1e-4, ge=1e-7, le=1.0, description="Absolute error tolerance")
    rtol: float = Field(1e-3, ge=1e-7, le=1.0, description="Relative error tolerance")
    execution_mode: str = Field("LOCAL", description="Target execution mode ('LOCAL', 'REMOTE', 'MOCK')")


class ValidationCheckResponse(BaseModel):
    check_id: str
    name: str
    category: str
    requirement: str
    detected_value: Optional[str]
    expected_value: str
    status: str
    evidence: str
    remediation: str
    requires_target_hardware: bool


class ValidationReportResponse(BaseModel):
    run_id: str
    model_fingerprint: str
    model_name: str
    target_profile_id: str
    target_profile_version: str
    environment: EnvironmentInventoryResponse
    overall_status: str
    status_rationale: str
    checks: List[ValidationCheckResponse]
    summary_counts: Dict[str, int]
    numerical_correctness: Dict[str, Any]
    benchmark_eligibility: bool
    execution_mode: str
    timestamp_utc: str
    provenance_badge: str


class ValidationRunListResponse(BaseModel):
    runs: List[ValidationReportResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int



