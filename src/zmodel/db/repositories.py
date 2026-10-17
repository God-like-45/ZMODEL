"""Typed repository interfaces for persisting and querying models, benchmarks, and optimization experiments."""

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session, selectinload
from zmodel.core.benchmark import BenchmarkReport
from zmodel.core.compatibility import CompatibilityReport
from zmodel.core.model_loader import ModelSummary
from zmodel.core.optimizer import OptimizationReport
from zmodel.core.graph_optimizer import GraphOptimizationReport
from zmodel.db.models import (
    BenchmarkRunORM,
    CandidateEvaluationORM,
    CompatibilityAssessmentORM,
    GraphOptimizationExperimentORM,
    GraphTransformationCandidateORM,
    ModelIdentityORM,
    OptimizationExperimentORM,
    ValidationRunORM,
    WorkerJobORM,
    utc_now,
)
from zmodel.core.execution.schemas import WorkerJobSpec
from zmodel.core.validation_engine import ValidationReport
from zmodel.db.session import session_scope


class ModelRepository:
    """Data access repository for ONNX model identities."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def register_or_update(self, summary: ModelSummary) -> ModelIdentityORM:
        """
        Register a new model identity by SHA-256 fingerprint, or update last_seen_at if already registered.
        Guarantees that distinct files with the same name remain distinct records.
        """
        def _execute(s: Session) -> ModelIdentityORM:
            existing = s.get(ModelIdentityORM, summary.sha256_hash)
            if existing:
                existing.last_seen_at = utc_now()
                # Update filename if it changed
                if existing.original_filename != summary.filename:
                    existing.original_filename = summary.filename
                return existing

            opset_dict = {op.domain or "ai.onnx": op.version for op in summary.opsets} if summary.opsets else {}
            meta_dict = {
                "total_nodes": summary.total_nodes,
                "total_parameters": summary.total_parameters,
                "total_weight_bytes": summary.total_weight_bytes,
                "distinct_operators": summary.distinct_operator_count,
                "inputs": [
                    {"name": inp.name, "shape": inp.shape, "dtype": inp.data_type, "is_dynamic": inp.is_dynamic}
                    for inp in summary.inputs
                ],
                "outputs": [
                    {"name": out.name, "shape": out.shape, "dtype": out.data_type}
                    for out in summary.outputs
                ],
            }

            model = ModelIdentityORM(
                fingerprint=summary.sha256_hash,
                original_filename=summary.filename,
                file_size_bytes=summary.file_size_bytes,
                ir_version=summary.metadata.ir_version if summary.metadata else None,
                producer_name=summary.metadata.producer_name if summary.metadata else None,
                producer_version=summary.metadata.producer_version if summary.metadata else None,
                opset_imports=opset_dict,
                graph_metadata=meta_dict,
                first_seen_at=utc_now(),
                last_seen_at=utc_now(),
            )
            s.add(model)
            return model

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_by_fingerprint(self, fingerprint: str) -> Optional[ModelIdentityORM]:
        """Fetch model by exact SHA-256 fingerprint."""
        def _execute(s: Session) -> Optional[ModelIdentityORM]:
            return s.get(ModelIdentityORM, fingerprint)

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def list_models(self, limit: int = 50, offset: int = 0) -> List[ModelIdentityORM]:
        """List registered models ordered by last seen timestamp descending."""
        def _execute(s: Session) -> List[ModelIdentityORM]:
            stmt = select(ModelIdentityORM).order_by(desc(ModelIdentityORM.last_seen_at)).limit(limit).offset(offset)
            return list(s.scalars(stmt).all())

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def count_models(self) -> int:
        """Count total unique models registered."""
        def _execute(s: Session) -> int:
            return s.scalar(select(func.count(ModelIdentityORM.fingerprint))) or 0

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)


class CompatibilityRepository:
    """Data access repository for capability profile compatibility assessments."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def save_assessment(self, report: CompatibilityReport) -> CompatibilityAssessmentORM:
        """Save a compatibility assessment, ensuring model identity exists."""
        def _execute(s: Session) -> CompatibilityAssessmentORM:
            # Ensure model exists for foreign key referential integrity
            model = s.get(ModelIdentityORM, report.model_fingerprint)
            if not model:
                model = ModelIdentityORM(
                    fingerprint=report.model_fingerprint,
                    original_filename=report.model_name,
                    file_size_bytes=0,
                    first_seen_at=utc_now(),
                    last_seen_at=utc_now(),
                )
                s.add(model)
                s.flush()

            assessment_id = f"compat_{report.model_fingerprint[:12]}_{report.profile_id}_{int(datetime.now(timezone.utc).timestamp())}"
            existing = s.get(CompatibilityAssessmentORM, assessment_id)
            if existing:
                return existing

            record = CompatibilityAssessmentORM(
                id=assessment_id,
                model_fingerprint=report.model_fingerprint,
                profile_id=report.profile_id,
                profile_name=report.profile_name,
                profile_version=report.profile_version,
                ruleset_version=report.ruleset_version,
                overall_assessment=report.overall_assessment,
                summary_counts=dict(report.summary_counts),
                findings=[f.to_dict() for f in report.findings],
                assumptions=list(report.assumptions) if report.assumptions else [],
                unresolved_questions=list(report.unresolved_questions) if report.unresolved_questions else [],
                recommended_next_steps=list(report.recommended_next_steps) if report.recommended_next_steps else [],
                is_illustrative_profile=report.is_illustrative_profile,
                schema_version="1.0.0",
                assessed_at=utc_now(),
            )
            s.add(record)
            return record

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_assessment(self, assessment_id: str) -> Optional[CompatibilityAssessmentORM]:
        """Fetch assessment by ID."""
        def _execute(s: Session) -> Optional[CompatibilityAssessmentORM]:
            return s.get(CompatibilityAssessmentORM, assessment_id)

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def list_assessments(
        self,
        fingerprint: Optional[str] = None,
        profile_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[CompatibilityAssessmentORM], int]:
        """List assessments with optional filtering and pagination."""
        def _execute(s: Session) -> Tuple[List[CompatibilityAssessmentORM], int]:
            stmt = select(CompatibilityAssessmentORM)
            if fingerprint:
                stmt = stmt.where(CompatibilityAssessmentORM.model_fingerprint == fingerprint)
            if profile_id:
                stmt = stmt.where(CompatibilityAssessmentORM.profile_id == profile_id)

            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = s.scalar(count_stmt) or 0

            records = list(
                s.scalars(stmt.order_by(desc(CompatibilityAssessmentORM.assessed_at)).limit(limit).offset(offset)).all()
            )
            return records, total

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)


class BenchmarkRepository:
    """Data access repository for empirical Phase 4 benchmark runs."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def save_benchmark_report(self, report: BenchmarkReport) -> BenchmarkRunORM:
        """
        Persist a benchmark report idempotently.
        Ensures foreign key referential integrity and avoids fabricating values for missing metrics.
        """
        def _execute(s: Session) -> BenchmarkRunORM:
            # Check for existing run by ID (idempotent duplicate submission protection)
            existing = s.get(BenchmarkRunORM, report.run_id)
            if existing:
                return existing

            # Ensure model exists
            model = s.get(ModelIdentityORM, report.model_fingerprint)
            if not model:
                model = ModelIdentityORM(
                    fingerprint=report.model_fingerprint,
                    original_filename=report.model_name,
                    file_size_bytes=0,
                    first_seen_at=utc_now(),
                    last_seen_at=utc_now(),
                )
                s.add(model)
                s.flush()

            # Extract metrics cleanly (handling None explicitly without substituting 0.0)
            lat = report.latency or {}
            tp = report.throughput or {}
            mem = report.memory or {}
            corr = report.correctness or {}

            # Parse timestamps
            completed_time = None
            if report.timestamp_utc:
                try:
                    completed_time = datetime.fromisoformat(report.timestamp_utc)
                except ValueError:
                    completed_time = utc_now()

            record = BenchmarkRunORM(
                id=report.run_id,
                model_fingerprint=report.model_fingerprint,
                model_name=report.model_name,
                execution_provider=report.config.get("execution_provider", "CPUExecutionProvider"),
                configuration=report.config,
                input_specifications=report.input_spec,
                runtime_environment=report.environment,
                status=report.status,
                error_message=report.error_message,
                correctness_status="PASS" if corr.get("passed", False) else ("FAIL" if corr and not corr.get("passed", True) else ("FAIL" if report.status == "FAILED" else "UNKNOWN")),
                correctness_details=corr.get("details", ""),
                median_latency_ms=lat.get("median_ms"),
                mean_latency_ms=lat.get("mean_ms"),
                min_latency_ms=lat.get("min_ms"),
                max_latency_ms=lat.get("max_ms"),
                p90_latency_ms=lat.get("p90_ms"),
                p95_latency_ms=lat.get("p95_ms"),
                p99_latency_ms=lat.get("p99_ms"),
                std_dev_latency_ms=lat.get("std_dev_ms"),
                sample_count=lat.get("sample_count"),
                inferences_per_second=tp.get("inferences_per_second"),
                samples_per_second=tp.get("samples_per_second"),
                session_delta_mb=mem.get("session_delta_mb"),
                peak_rss_mb=mem.get("peak_rss_mb"),
                per_round_results=report.per_round_results,
                warnings=list(report.warnings) if report.warnings else [],
                limitations=list(report.limitations) if report.limitations else [],
                schema_version="1.0.0",
                created_at=completed_time or utc_now(),
                completed_at=completed_time or utc_now(),
            )
            s.add(record)
            return record

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_benchmark_run(self, run_id: str) -> Optional[BenchmarkRunORM]:
        """Fetch benchmark run by ID."""
        def _execute(s: Session) -> Optional[BenchmarkRunORM]:
            return s.get(BenchmarkRunORM, run_id)

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def list_benchmark_runs(
        self,
        fingerprint: Optional[str] = None,
        status: Optional[str] = None,
        provider: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[BenchmarkRunORM], int]:
        """List benchmark runs with optional filtering and pagination."""
        def _execute(s: Session) -> Tuple[List[BenchmarkRunORM], int]:
            stmt = select(BenchmarkRunORM)
            if fingerprint:
                stmt = stmt.where(BenchmarkRunORM.model_fingerprint == fingerprint)
            if status and status != "All":
                stmt = stmt.where(BenchmarkRunORM.status == status)
            if provider and provider != "All":
                stmt = stmt.where(BenchmarkRunORM.execution_provider == provider)

            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = s.scalar(count_stmt) or 0

            records = list(
                s.scalars(stmt.order_by(desc(BenchmarkRunORM.created_at)).limit(limit).offset(offset)).all()
            )
            return records, total

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def count_benchmark_runs(self) -> int:
        """Count total benchmark runs."""
        def _execute(s: Session) -> int:
            return s.scalar(select(func.count(BenchmarkRunORM.id))) or 0

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)


class OptimizationRepository:
    """Data access repository for Phase 5 optimization experiments and candidate evaluations."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def save_optimization_report(self, report: OptimizationReport) -> OptimizationExperimentORM:
        """
        Persist an optimization experiment and all candidate evaluations atomically within a single transaction.
        Preserves rejected and unbenchmarked candidates, constraint evaluations, and Pareto rankings.
        """
        def _execute(s: Session) -> OptimizationExperimentORM:
            # Check for existing experiment (idempotency)
            existing = s.get(OptimizationExperimentORM, report.experiment_id)
            if existing:
                return existing

            # Ensure model exists
            model = s.get(ModelIdentityORM, report.model_fingerprint)
            if not model:
                model = ModelIdentityORM(
                    fingerprint=report.model_fingerprint,
                    original_filename=report.model_name,
                    file_size_bytes=0,
                    first_seen_at=utc_now(),
                    last_seen_at=utc_now(),
                )
                s.add(model)
                s.flush()

            # Parse timestamps
            completed_time = None
            if report.timestamp_utc:
                try:
                    completed_time = datetime.fromisoformat(report.timestamp_utc)
                except ValueError:
                    completed_time = utc_now()

            experiment = OptimizationExperimentORM(
                id=report.experiment_id,
                model_fingerprint=report.model_fingerprint,
                model_name=report.model_name,
                profile_id=report.profile_id,
                profile_name=report.profile_name,
                profile_version=report.profile_version,
                primary_objective=report.primary_objective,
                status=report.status.value,
                request_payload=report.request.to_dict(),
                baseline_candidate_id=report.baseline_candidate_id,
                best_candidate_id=report.best_candidate_id,
                summary_counts=dict(report.summary_counts),
                recommendation_rationale=report.recommendation_rationale,
                pareto_frontier_ids=list(report.pareto_frontier_ids) if report.pareto_frontier_ids else [],
                warnings=list(report.warnings) if report.warnings else [],
                limitations=list(report.limitations) if report.limitations else [],
                environment=report.environment,
                schema_version="1.0.0",
                created_at=completed_time or utc_now(),
                completed_at=completed_time or utc_now(),
            )
            s.add(experiment)
            s.flush()

            # Save candidates
            for cand in report.candidates:
                bench_run_id = None
                if cand.benchmark_report:
                    bench_repo = BenchmarkRepository(session=s)
                    saved_run = bench_repo.save_benchmark_report(cand.benchmark_report)
                    bench_run_id = saved_run.id

                cand_metrics = cand.get_metrics_dict()
                eval_record = CandidateEvaluationORM(
                    id=f"{report.experiment_id}_{cand.candidate_id}",
                    experiment_id=report.experiment_id,
                    candidate_id=cand.candidate_id,
                    benchmark_run_id=bench_run_id,
                    configuration=cand.config.to_dict() if hasattr(cand.config, "to_dict") else asdict(cand.config),
                    generation_reason=cand.generation_reason,
                    status=cand.status.value,
                    rejection_reason=cand.rejection_reason,
                    compatibility_findings=cand.compatibility_findings,
                    constraint_evaluations=[ce.to_dict() for ce in cand.constraint_evaluations],
                    is_feasible=cand.is_feasible,
                    is_pareto_optimal=cand.is_pareto_optimal,
                    is_baseline=cand.is_baseline,
                    rank=cand.rank,
                    measured_median_latency_ms=cand_metrics.get("median_latency_ms"),
                    measured_throughput_ips=cand_metrics.get("throughput_inf_sec"),
                    measured_memory_mb=cand_metrics.get("peak_memory_mb"),
                    speedup_vs_baseline=cand.speedup_vs_baseline,
                    latency_reduction_pct=cand.latency_reduction_pct,
                    throughput_improvement_pct=cand.throughput_improvement_pct,
                    created_at=completed_time or utc_now(),
                )
                s.add(eval_record)

            return experiment

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def save_advanced_optimization_report(self, report_dict: Dict[str, Any]) -> OptimizationExperimentORM:
        """
        Persist a Phase 12 advanced optimization report including search strategy,
        sensitivity findings, recommendation decisions, and enterprise reports.
        """
        def _execute(s: Session) -> OptimizationExperimentORM:
            exp_id = report_dict["experiment_id"]
            existing = s.get(OptimizationExperimentORM, exp_id)
            if existing:
                return existing

            model = s.get(ModelIdentityORM, report_dict["model_fingerprint"])
            if not model:
                model = ModelIdentityORM(
                    fingerprint=report_dict["model_fingerprint"],
                    original_filename=report_dict.get("model_name", "model.onnx"),
                    file_size_bytes=0,
                    first_seen_at=utc_now(),
                    last_seen_at=utc_now(),
                )
                s.add(model)
                s.flush()

            completed_time = utc_now()
            if report_dict.get("timestamp_utc"):
                try:
                    completed_time = datetime.fromisoformat(report_dict["timestamp_utc"])
                except ValueError:
                    completed_time = utc_now()

            req_payload = dict(report_dict.get("request", {}))
            req_payload["sensitivity"] = report_dict.get("sensitivity")
            req_payload["recommendation"] = report_dict.get("recommendation")
            req_payload["markdown_report"] = report_dict.get("markdown_report")
            req_payload["json_report"] = report_dict.get("json_report")

            experiment = OptimizationExperimentORM(
                id=exp_id,
                model_fingerprint=report_dict["model_fingerprint"],
                model_name=report_dict.get("model_name", "model.onnx"),
                profile_id=report_dict.get("profile_id", "unknown_profile"),
                profile_name=report_dict.get("profile_name", "Target Profile"),
                profile_version=report_dict.get("profile_version", "1.0.0"),
                primary_objective=report_dict.get("primary_objective", "MINIMIZE_MEDIAN_LATENCY"),
                status="COMPLETED",
                request_payload=req_payload,
                baseline_candidate_id=report_dict.get("baseline_candidate_id"),
                best_candidate_id=report_dict.get("best_candidate_id"),
                summary_counts=report_dict.get("summary_counts", {}),
                recommendation_rationale=report_dict.get("recommendation_rationale"),
                pareto_frontier_ids=report_dict.get("pareto_frontier_ids", []),
                warnings=[],
                limitations=[],
                environment=report_dict.get("environment", {}),
                schema_version="2.0.0-phase12",
                created_at=completed_time,
                completed_at=completed_time,
            )
            s.add(experiment)
            s.flush()

            # Save candidate evaluations
            for c in report_dict.get("candidates", []):
                cfg = c.get("config", {})
                m = c.get("metrics", {}) or {}
                eval_record = CandidateEvaluationORM(
                    id=f"{exp_id}_{c.get('candidate_id')}",
                    experiment_id=exp_id,
                    candidate_id=c.get("candidate_id"),
                    benchmark_run_id=None,
                    configuration=cfg,
                    generation_reason=c.get("generation_reason", ""),
                    status=c.get("status", "BENCHMARKED"),
                    rejection_reason=c.get("rejection_reason"),
                    compatibility_findings=c.get("compatibility_findings", []),
                    constraint_evaluations=c.get("constraint_evaluations", []),
                    is_feasible=c.get("is_feasible", False),
                    is_pareto_optimal=c.get("is_pareto_optimal", False),
                    is_baseline=c.get("is_baseline", False),
                    rank=c.get("rank"),
                    measured_median_latency_ms=m.get("median_latency_ms"),
                    measured_throughput_ips=m.get("throughput_inf_sec"),
                    measured_memory_mb=m.get("peak_memory_mb"),
                    speedup_vs_baseline=c.get("speedup_vs_baseline"),
                    latency_reduction_pct=c.get("latency_reduction_pct"),
                    throughput_improvement_pct=c.get("throughput_improvement_pct"),
                    created_at=completed_time,
                )
                s.add(eval_record)

            return experiment

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_optimization_experiment(self, experiment_id: str) -> Optional[OptimizationExperimentORM]:
        """Fetch experiment by ID with all attached candidate evaluations."""
        def _execute(s: Session) -> Optional[OptimizationExperimentORM]:
            stmt = (
                select(OptimizationExperimentORM)
                .options(selectinload(OptimizationExperimentORM.candidates))
                .where(OptimizationExperimentORM.id == experiment_id)
            )
            return s.scalars(stmt).first()

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_candidates_for_experiment(self, experiment_id: str) -> List[CandidateEvaluationORM]:
        """Fetch all candidate evaluations for an experiment ordered by rank."""
        def _execute(s: Session) -> List[CandidateEvaluationORM]:
            stmt = (
                select(CandidateEvaluationORM)
                .where(CandidateEvaluationORM.experiment_id == experiment_id)
                .order_by(CandidateEvaluationORM.rank)
            )
            return list(s.scalars(stmt).all())

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def list_optimization_experiments(
        self,
        fingerprint: Optional[str] = None,
        status: Optional[str] = None,
        objective: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[OptimizationExperimentORM], int]:
        """List experiments with optional filtering and pagination."""
        def _execute(s: Session) -> Tuple[List[OptimizationExperimentORM], int]:
            stmt = select(OptimizationExperimentORM).options(selectinload(OptimizationExperimentORM.candidates))
            if fingerprint:
                stmt = stmt.where(OptimizationExperimentORM.model_fingerprint == fingerprint)
            if status and status != "All":
                stmt = stmt.where(OptimizationExperimentORM.status == status)
            if objective and objective != "All":
                stmt = stmt.where(OptimizationExperimentORM.primary_objective == objective)

            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = s.scalar(count_stmt) or 0

            records = list(
                s.scalars(stmt.order_by(desc(OptimizationExperimentORM.created_at)).limit(limit).offset(offset)).all()
            )
            return records, total

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def count_experiments(self) -> int:
        """Count total optimization experiments."""
        def _execute(s: Session) -> int:
            return s.scalar(select(func.count(OptimizationExperimentORM.id))) or 0

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)


class HistoryOverviewRepository:
    """Dashboard metrics aggregator for the Experiment History overview."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def get_overview_metrics(self) -> Dict[str, Any]:
        """Calculate counts and retrieve recent activity across all tables."""
        def _execute(s: Session) -> Dict[str, Any]:
            total_models = s.scalar(select(func.count(ModelIdentityORM.fingerprint))) or 0
            total_experiments = s.scalar(select(func.count(OptimizationExperimentORM.id))) or 0
            total_benchmarks = s.scalar(select(func.count(BenchmarkRunORM.id))) or 0

            # Experiment status counts
            exp_status_counts: Dict[str, int] = {}
            for status, count in s.execute(
                select(OptimizationExperimentORM.status, func.count(OptimizationExperimentORM.id)).group_by(
                    OptimizationExperimentORM.status
                )
            ):
                exp_status_counts[status] = count

            # Benchmark status counts
            bench_status_counts: Dict[str, int] = {}
            for status, count in s.execute(
                select(BenchmarkRunORM.status, func.count(BenchmarkRunORM.id)).group_by(BenchmarkRunORM.status)
            ):
                bench_status_counts[status] = count

            # Recent experiments
            recent_exps = list(
                s.scalars(
                    select(OptimizationExperimentORM)
                    .order_by(desc(OptimizationExperimentORM.created_at))
                    .limit(5)
                ).all()
            )

            # Recent benchmarks
            recent_benchs = list(
                s.scalars(
                    select(BenchmarkRunORM).order_by(desc(BenchmarkRunORM.created_at)).limit(5)
                ).all()
            )

            return {
                "total_models": total_models,
                "total_experiments": total_experiments,
                "total_benchmarks": total_benchmarks,
                "experiment_status_counts": exp_status_counts,
                "benchmark_status_counts": bench_status_counts,
                "recent_experiments": [e.to_dict() for e in recent_exps],
                "recent_benchmarks": [b.to_dict() for b in recent_benchs],
            }

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)


class GraphOptimizationRepository:
    """Data access repository for ONNX graph transformation experiments and candidates."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def create_experiment(self, report: GraphOptimizationReport) -> GraphOptimizationExperimentORM:
        """Persist a complete graph optimization experiment and its transformation candidates."""
        def _execute(s: Session) -> GraphOptimizationExperimentORM:
            # Ensure model identity exists to satisfy foreign key
            model = s.get(ModelIdentityORM, report.original_fingerprint)
            if not model:
                model = ModelIdentityORM(
                    fingerprint=report.original_fingerprint,
                    original_filename=report.model_name,
                    file_size_bytes=report.candidates[0].model_size_bytes_before if report.candidates else 0,
                    first_seen_at=utc_now(),
                    last_seen_at=utc_now(),
                )
                s.add(model)
                s.flush()

            exp = GraphOptimizationExperimentORM(
                id=report.experiment_id,
                model_fingerprint=report.original_fingerprint,
                model_name=report.model_name,
                target_profile_id=report.target_profile_id,
                target_profile_name=report.target_profile_name,
                validation_depth=report.validation_depth,
                benchmark_enabled=report.benchmark_enabled,
                status=report.status,
                best_candidate_id=report.best_candidate_id,
                summary_counts=report.summary_counts,
                overall_recommendation=report.overall_recommendation,
                limitations=report.limitations,
                error_message=report.error_message,
                created_at=datetime.fromisoformat(report.created_at) if report.created_at else utc_now(),
                completed_at=datetime.fromisoformat(report.completed_at) if report.completed_at else utc_now(),
            )
            s.add(exp)
            s.flush()

            for cand in report.candidates:
                cand_orm = GraphTransformationCandidateORM(
                    id=f"{exp.id}_{cand.candidate_id}",
                    experiment_id=exp.id,
                    candidate_id=cand.candidate_id,
                    transform_type=cand.transform_type.value if hasattr(cand.transform_type, "value") else str(cand.transform_type),
                    transform_name=cand.transform_name,
                    tool_name=cand.tool_name,
                    tool_version=cand.tool_version,
                    state=cand.state.value if hasattr(cand.state, "value") else str(cand.state),
                    original_hash=cand.original_hash,
                    transformed_hash=cand.transformed_hash,
                    artifact_id=cand.artifact_id,
                    graph_changed=cand.graph_changed,
                    model_size_bytes_before=cand.model_size_bytes_before,
                    model_size_bytes_after=cand.model_size_bytes_after,
                    size_reduction_pct=cand.size_reduction_pct,
                    diff_summary=cand.diff_summary,
                    validation_report=cand.validation_report.to_dict() if cand.validation_report else None,
                    compatibility_report=cand.compatibility_report,
                    compatibility_delta=cand.compatibility_delta,
                    benchmark_report=cand.benchmark_report,
                    speedup_factor=cand.speedup_factor,
                    latency_reduction_pct=cand.latency_reduction_pct,
                    throughput_improvement_pct=cand.throughput_improvement_pct,
                    recommendation_status=cand.recommendation_status.value if hasattr(cand.recommendation_status, "value") else str(cand.recommendation_status),
                    recommendation_rationale=cand.recommendation_rationale,
                    rejection_reason=cand.rejection_reason,
                    warnings=cand.warnings,
                    duration_ms=cand.duration_ms,
                    created_at=utc_now(),
                )
                s.add(cand_orm)

            s.flush()
            return exp

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_experiment(self, experiment_id: str) -> Optional[GraphOptimizationExperimentORM]:
        """Retrieve graph optimization experiment by ID with candidates preloaded."""
        def _execute(s: Session) -> Optional[GraphOptimizationExperimentORM]:
            stmt = (
                select(GraphOptimizationExperimentORM)
                .options(selectinload(GraphOptimizationExperimentORM.candidates))
                .where(GraphOptimizationExperimentORM.id == experiment_id)
            )
            return s.scalars(stmt).first()

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def list_experiments(
        self,
        model_fingerprint: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[GraphOptimizationExperimentORM]:
        """List graph optimization experiments, optionally filtered by model fingerprint."""
        def _execute(s: Session) -> List[GraphOptimizationExperimentORM]:
            stmt = select(GraphOptimizationExperimentORM).options(
                selectinload(GraphOptimizationExperimentORM.candidates)
            )
            if model_fingerprint:
                stmt = stmt.where(GraphOptimizationExperimentORM.model_fingerprint == model_fingerprint)
            stmt = stmt.order_by(desc(GraphOptimizationExperimentORM.created_at)).limit(limit).offset(offset)
            return list(s.scalars(stmt).all())

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_candidate(
        self, experiment_id: str, candidate_id: str
    ) -> Optional[GraphTransformationCandidateORM]:
        """Retrieve a specific transformation candidate by experiment ID and candidate ID."""
        def _execute(s: Session) -> Optional[GraphTransformationCandidateORM]:
            stmt = select(GraphTransformationCandidateORM).where(
                GraphTransformationCandidateORM.experiment_id == experiment_id,
                GraphTransformationCandidateORM.candidate_id == candidate_id,
            )
            return s.scalars(stmt).first()

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)


class WorkerJobRepository:
    """Data access repository for tracking asynchronous worker job lifecycles and results."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def create_job(self, spec: WorkerJobSpec, backend: str = "local") -> WorkerJobORM:
        """Persist a new worker job record idempotently."""
        def _execute(s: Session) -> WorkerJobORM:
            existing = s.get(WorkerJobORM, spec.job_id)
            if existing:
                return existing

            # Ensure model identity exists
            model = s.get(ModelIdentityORM, spec.model_fingerprint)
            if not model:
                model = ModelIdentityORM(
                    fingerprint=spec.model_fingerprint,
                    original_filename=spec.model_filename,
                    file_size_bytes=0,
                    first_seen_at=utc_now(),
                    last_seen_at=utc_now(),
                )
                s.add(model)
                s.flush()

            record = WorkerJobORM(
                id=spec.job_id,
                job_type=spec.job_type.value if hasattr(spec.job_type, "value") else str(spec.job_type),
                execution_backend=backend,
                model_fingerprint=spec.model_fingerprint,
                experiment_id=spec.experiment_id,
                run_id=spec.run_id,
                status="QUEUED",
                resource_profile=spec.resource_profile,
                max_retries=spec.max_retries,
                spec_json=spec.to_dict(),
                created_at=utc_now(),
            )
            s.add(record)
            s.flush()
            return record

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def update_job_status(
        self,
        job_id: str,
        status: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        k8s_namespace: Optional[str] = None,
        k8s_job_name: Optional[str] = None,
        k8s_pod_name: Optional[str] = None,
        worker_image: Optional[str] = None,
        timing: Optional[Any] = None,
        result_json: Optional[Dict[str, Any]] = None,
        output_artifact_ids: Optional[List[str]] = None,
        error_category: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[WorkerJobORM]:
        """Update job lifecycle status, timing metrics, and completed results."""
        def _execute(s: Session) -> Optional[WorkerJobORM]:
            job = s.get(WorkerJobORM, job_id)
            if not job:
                return None
            job.status = status
            if started_at:
                job.started_at = started_at
            if completed_at:
                job.completed_at = completed_at
            if k8s_namespace:
                job.k8s_namespace = k8s_namespace
            if k8s_job_name:
                job.k8s_job_name = k8s_job_name
            if k8s_pod_name:
                job.k8s_pod_name = k8s_pod_name
            if worker_image:
                job.worker_image = worker_image
            if timing:
                job.queue_delay_seconds = getattr(timing, "queue_delay_seconds", 0.0)
                job.startup_overhead_seconds = getattr(timing, "startup_overhead_seconds", 0.0)
                job.warmup_overhead_seconds = getattr(timing, "warmup_overhead_seconds", 0.0)
                job.inference_duration_seconds = getattr(timing, "inference_duration_seconds", 0.0)
                job.total_duration_seconds = getattr(timing, "total_duration_seconds", 0.0)
            if result_json:
                job.result_json = result_json
            if output_artifact_ids:
                job.output_artifact_ids = output_artifact_ids
            if error_category:
                job.error_category = error_category
            if error_message:
                job.error_message = error_message
            s.flush()
            return job

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_job(self, job_id: str) -> Optional[WorkerJobORM]:
        """Retrieve worker job by ID."""
        def _execute(s: Session) -> Optional[WorkerJobORM]:
            return s.get(WorkerJobORM, job_id)

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def list_jobs(
        self,
        job_type: Optional[str] = None,
        backend: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[WorkerJobORM], int]:
        """List worker jobs with optional filters and pagination."""
        def _execute(s: Session) -> Tuple[List[WorkerJobORM], int]:
            stmt = select(WorkerJobORM)
            if job_type:
                stmt = stmt.where(WorkerJobORM.job_type == job_type)
            if backend:
                stmt = stmt.where(WorkerJobORM.execution_backend == backend)
            if status:
                stmt = stmt.where(WorkerJobORM.status == status)

            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = s.scalar(count_stmt) or 0

            records = list(
                s.scalars(stmt.order_by(desc(WorkerJobORM.created_at)).limit(limit).offset(offset)).all()
            )
            return records, total

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)


class ValidationRepository:
    """Data access repository for IBM Z / Linux on IBM Z validation runs."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def save_validation_report(self, report: ValidationReport) -> ValidationRunORM:
        """
        Persist a full ValidationReport into PostgreSQL.
        Ensures foreign-key referential integrity to ModelIdentityORM.
        """
        def _execute(s: Session) -> ValidationRunORM:
            # Ensure model identity exists first
            model_record = s.get(ModelIdentityORM, report.model_fingerprint)
            if not model_record:
                model_record = ModelIdentityORM(
                    fingerprint=report.model_fingerprint,
                    original_filename=report.model_name or "model.onnx",
                    file_size_bytes=0,
                )
                s.add(model_record)
                s.flush()

            # Create or update validation run
            existing = s.get(ValidationRunORM, report.run_id)
            if existing:
                existing.overall_status = report.overall_status.value
                existing.status_rationale = report.status_rationale
                existing.provenance_badge = report.provenance_badge
                existing.summary_counts = report.summary_counts
                existing.checks_json = [c.to_dict() for c in report.checks]
                existing.numerical_correctness = report.numerical_correctness
                existing.benchmark_eligibility = report.benchmark_eligibility
                existing.environment_info = report.environment.to_dict()
                s.flush()
                return existing

            run_orm = ValidationRunORM(
                id=report.run_id,
                model_fingerprint=report.model_fingerprint,
                target_profile_id=report.target_profile_id,
                target_profile_version=report.target_profile_version,
                environment_fingerprint=report.environment.environment_fingerprint,
                execution_mode=report.execution_mode,
                overall_status=report.overall_status.value,
                status_rationale=report.status_rationale,
                provenance_badge=report.provenance_badge,
                benchmark_eligibility=report.benchmark_eligibility,
                summary_counts=report.summary_counts,
                checks_json=[c.to_dict() for c in report.checks],
                numerical_correctness=report.numerical_correctness,
                environment_info=report.environment.to_dict(),
            )
            s.add(run_orm)
            s.flush()
            return run_orm

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def get_validation_run(self, run_id: str) -> Optional[ValidationRunORM]:
        """Retrieve validation run by ID."""
        def _execute(s: Session) -> Optional[ValidationRunORM]:
            return s.get(ValidationRunORM, run_id)

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)

    def list_validation_runs(
        self,
        model_fingerprint: Optional[str] = None,
        target_profile_id: Optional[str] = None,
        overall_status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[ValidationRunORM], int]:
        """List validation runs with optional filtering and pagination."""
        def _execute(s: Session) -> Tuple[List[ValidationRunORM], int]:
            stmt = select(ValidationRunORM)
            if model_fingerprint:
                stmt = stmt.where(ValidationRunORM.model_fingerprint == model_fingerprint)
            if target_profile_id:
                stmt = stmt.where(ValidationRunORM.target_profile_id == target_profile_id)
            if overall_status:
                stmt = stmt.where(ValidationRunORM.overall_status == overall_status)

            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = s.scalar(count_stmt) or 0

            records = list(
                s.scalars(stmt.order_by(desc(ValidationRunORM.created_at)).limit(limit).offset(offset)).all()
            )
            return records, total

        if self._session:
            return _execute(self._session)
        with session_scope() as s:
            return _execute(s)



