"""Z-ModelPilot Database and Persistence Subsystem (Phase 6)."""

from zmodel.db.comparator import (
    BenchmarkComparisonResult,
    ComparisonValidity,
    MetricComparison,
    compare_benchmark_runs,
)
from zmodel.db.config import (
    check_database_connection,
    get_database_url,
    mask_database_url,
)
from zmodel.db.models import (
    Base,
    BenchmarkRunORM,
    CandidateEvaluationORM,
    CompatibilityAssessmentORM,
    ModelIdentityORM,
    OptimizationExperimentORM,
)
from zmodel.db.repositories import (
    BenchmarkRepository,
    CompatibilityRepository,
    HistoryOverviewRepository,
    ModelRepository,
    OptimizationRepository,
)
from zmodel.db.session import (
    get_engine,
    get_session_factory,
    init_db,
    reset_engine,
    session_scope,
)

__all__ = [
    "Base",
    "ModelIdentityORM",
    "CompatibilityAssessmentORM",
    "BenchmarkRunORM",
    "OptimizationExperimentORM",
    "CandidateEvaluationORM",
    "get_engine",
    "get_session_factory",
    "reset_engine",
    "session_scope",
    "init_db",
    "get_database_url",
    "mask_database_url",
    "check_database_connection",
    "ModelRepository",
    "CompatibilityRepository",
    "BenchmarkRepository",
    "OptimizationRepository",
    "HistoryOverviewRepository",
    "compare_benchmark_runs",
    "BenchmarkComparisonResult",
    "MetricComparison",
    "ComparisonValidity",
]
