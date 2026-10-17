import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st
from zmodel.core.model_loader import ModelSummary

logger = logging.getLogger(__name__)

_DB_AVAILABLE: Optional[bool] = None
_LAST_DB_CHECK: float = 0.0


def is_db_available() -> bool:
    """Check if the database is reachable with a 10-second cached result to avoid blocking."""
    global _DB_AVAILABLE, _LAST_DB_CHECK
    now = time.time()
    if _DB_AVAILABLE is not None and (now - _LAST_DB_CHECK) < 10.0:
        return _DB_AVAILABLE
    try:
        from zmodel.db.config import check_database_connection
        status = check_database_connection(timeout_seconds=1)
        _DB_AVAILABLE = status.get("connected", False)
    except Exception:
        _DB_AVAILABLE = False
    _LAST_DB_CHECK = now
    return _DB_AVAILABLE


def init_session_state() -> None:
    """Initialize default keys in Streamlit session state if absent."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "overview"

    if "active_model" not in st.session_state:
        st.session_state["active_model"] = None

    if "active_model_source" not in st.session_state:
        st.session_state["active_model_source"] = None

    if "activity_log" not in st.session_state:
        st.session_state["activity_log"] = []

    if "selected_profile_id" not in st.session_state:
        st.session_state["selected_profile_id"] = "local_onnx_runtime_cpu"

    if "active_compatibility_report" not in st.session_state:
        st.session_state["active_compatibility_report"] = None

    if "active_benchmark_report" not in st.session_state:
        st.session_state["active_benchmark_report"] = None

    if "active_optimization_report" not in st.session_state:
        st.session_state["active_optimization_report"] = None

    if "active_model_bytes" not in st.session_state:
        st.session_state["active_model_bytes"] = None

    if "selected_experiment_id" not in st.session_state:
        st.session_state["selected_experiment_id"] = None

    if "selected_benchmark_run_id" not in st.session_state:
        st.session_state["selected_benchmark_run_id"] = None

    if "comparison_run_ids" not in st.session_state:
        st.session_state["comparison_run_ids"] = []


def get_active_model() -> Optional[ModelSummary]:
    """Retrieve the currently loaded and validated model summary, if any."""
    return st.session_state.get("active_model", None)


def get_active_model_bytes() -> Optional[bytes]:
    """Retrieve the raw binary bytes of the active model, reading from disk if a known sample."""
    b = st.session_state.get("active_model_bytes", None)
    if b is not None:
        return b
    model = get_active_model()
    if model is not None and model.filename:
        sample_path = Path("sample_models") / model.filename
        if sample_path.exists() and sample_path.is_file():
            try:
                content = sample_path.read_bytes()
                st.session_state["active_model_bytes"] = content
                return content
            except Exception:
                pass
    return None


def set_active_model(
    model_summary: ModelSummary,
    source_description: str,
    model_bytes: Optional[bytes] = None,
) -> None:
    """Set the currently active model and synchronize with the persistent model registry."""
    st.session_state["active_model"] = model_summary
    st.session_state["active_model_source"] = source_description
    st.session_state["active_model_bytes"] = model_bytes
    # Clear stale reports from previous model
    st.session_state["active_compatibility_report"] = None
    st.session_state["active_benchmark_report"] = None
    st.session_state["active_optimization_report"] = None

    # Persist model identity to database if available
    if is_db_available():
        try:
            from zmodel.db.repositories import ModelRepository
            repo = ModelRepository()
            repo.register_or_update(model_summary)
        except Exception as ex:
            logger.debug(f"Persistence skipped for model identity: {ex}")

    # Record in activity log
    log_activity(
        action="Model Inspected",
        details=f"{model_summary.filename} ({model_summary.total_nodes} nodes, {model_summary.total_parameters:,} params)",
        status="Success",
    )


def clear_active_model() -> None:
    """Unload the active model."""
    if st.session_state.get("active_model") is not None:
        filename = st.session_state["active_model"].filename
        log_activity(
            action="Model Unloaded",
            details=f"Unloaded {filename}",
            status="Info",
        )
    st.session_state["active_model"] = None
    st.session_state["active_model_source"] = None
    st.session_state["active_model_bytes"] = None
    st.session_state["active_compatibility_report"] = None
    st.session_state["active_benchmark_report"] = None
    st.session_state["active_optimization_report"] = None


def get_selected_profile_id() -> str:
    """Get currently selected target capability profile id."""
    return st.session_state.get("selected_profile_id", "local_onnx_runtime_cpu")


def set_selected_profile_id(profile_id: str) -> None:
    """Set the selected target capability profile id."""
    st.session_state["selected_profile_id"] = profile_id


def get_compatibility_report() -> Optional[Any]:
    """Retrieve the cached compatibility analysis report, if any."""
    return st.session_state.get("active_compatibility_report", None)


def set_compatibility_report(report: Any) -> None:
    """Store the compatibility analysis report and persist to database."""
    st.session_state["active_compatibility_report"] = report
    if report is not None:
        if is_db_available():
            try:
                from zmodel.db.repositories import CompatibilityRepository
                repo = CompatibilityRepository()
                repo.save_assessment(report)
            except Exception as ex:
                logger.debug(f"Persistence skipped for compatibility assessment: {ex}")

        log_activity(
            action="Compatibility Analyzed",
            details=f"{report.model_name} against {report.profile_name} -> {report.overall_assessment}",
            status="Success",
        )


def get_benchmark_report() -> Optional[Any]:
    """Retrieve the cached benchmark report, if any."""
    return st.session_state.get("active_benchmark_report", None)


def set_benchmark_report(report: Any) -> None:
    """Store the benchmark report and persist to database."""
    st.session_state["active_benchmark_report"] = report
    if report is not None:
        if is_db_available():
            try:
                from zmodel.db.repositories import BenchmarkRepository
                repo = BenchmarkRepository()
                repo.save_benchmark_report(report)
            except Exception as ex:
                logger.debug(f"Persistence skipped for benchmark report: {ex}")

        med_lat = report.latency.get("median_ms", 0) if report.latency else 0
        log_activity(
            action="Model Benchmarked",
            details=f"{report.model_name} -> {report.status} ({med_lat:.2f} ms median)",
            status="Success" if report.status == "SUCCESS" else "Error",
        )


def clear_benchmark_report() -> None:
    """Clear active benchmark report."""
    st.session_state["active_benchmark_report"] = None


def get_optimization_report() -> Optional[Any]:
    """Retrieve the cached optimization report, if any."""
    return st.session_state.get("active_optimization_report", None)


def set_optimization_report(report: Any) -> None:
    """Store the optimization report and persist to database."""
    st.session_state["active_optimization_report"] = report
    if report is not None:
        if is_db_available():
            try:
                from zmodel.db.repositories import OptimizationRepository
                repo = OptimizationRepository()
                if isinstance(report, dict):
                    repo.save_advanced_optimization_report(report)
                else:
                    repo.save_optimization_report(report)
            except Exception as ex:
                logger.debug(f"Persistence skipped for optimization report: {ex}")

        if isinstance(report, dict):
            best_id = report.get("best_candidate_id") or "No feasible candidate"
            m_name = report.get("model_name", "Model")
            log_activity(
                action="Optimization Completed",
                details=f"{m_name} -> Phase 12 ({best_id})",
                status="Success" if report.get("best_candidate_id") else "Warning",
            )
        else:
            best_str = f"Best: {report.best_candidate_id}" if report.best_candidate_id else "No feasible candidate"
            log_activity(
                action="Optimization Completed",
                details=f"{report.model_name} -> {report.status.value} ({best_str})",
                status="Success" if report.best_candidate_id else "Warning",
            )


def clear_optimization_report() -> None:
    """Clear active optimization report."""
    st.session_state["active_optimization_report"] = None


def log_activity(action: str, details: str, status: str = "Success") -> None:
    """Add a structured record to the user's session activity log."""
    log = st.session_state.setdefault("activity_log", [])
    log.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "details": details,
            "status": status,
        },
    )
    # Keep last 50 activities
    if len(log) > 50:
        st.session_state["activity_log"] = log[:50]


def get_activity_log() -> List[Dict[str, Any]]:
    """Retrieve the chronological list of user activities in this session."""
    return st.session_state.get("activity_log", [])


def navigate_to(page_id: str) -> None:
    """Set active navigation page."""
    st.session_state["current_page"] = page_id
