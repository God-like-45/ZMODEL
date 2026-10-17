"""Worker task execution logic for isolated benchmark, optimization, and transformation jobs."""

import hashlib
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from zmodel.core.artifact_store import get_artifact_store
from zmodel.core.benchmark import BenchmarkConfig, BenchmarkEngine
from zmodel.core.execution.schemas import (
    TimingBreakdown,
    WorkerJobSpec,
    WorkerJobType,
)
from zmodel.core.graph_optimizer import GraphOptimizationEngine
from zmodel.core.graph_transforms import TransformationType
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.optimizer import (
    HardConstraint,
    OptimizationEngine,
    OptimizationObjective,
    OptimizationRequest,
)
from zmodel.core.profiles import load_builtin_profiles

logger = logging.getLogger(__name__)


def resolve_worker_model_bytes(spec: WorkerJobSpec) -> Tuple[bytes, str]:
    """
    Safely resolve and verify ONNX model bytes for worker execution.
    Verifies that the retrieved model exactly matches spec.model_fingerprint.
    """
    store = get_artifact_store()
    model_bytes: Optional[bytes] = None
    filename = spec.model_filename or "model.onnx"

    # 1. Attempt retrieval by input_artifact_id from ArtifactStore
    if spec.input_artifact_id:
        try:
            model_bytes = store.read_artifact_bytes(spec.input_artifact_id, expected_hash=spec.model_fingerprint)
        except Exception as ex:
            logger.debug(f"Could not read artifact {spec.input_artifact_id}: {ex}")

    # 2. Attempt retrieval from in-memory cache if running locally
    if model_bytes is None:
        from zmodel.api.model_store import get_model_bytes
        cached = get_model_bytes(spec.model_fingerprint)
        if cached:
            model_bytes = cached

    # 3. Check sample_models directory
    if model_bytes is None:
        sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "sample_models",
        )
        if os.path.exists(sample_dir):
            for fname in os.listdir(sample_dir):
                if fname.endswith(".onnx"):
                    fpath = os.path.join(sample_dir, fname)
                    with open(fpath, "rb") as f:
                        data = f.read()
                    if hashlib.sha256(data).hexdigest() == spec.model_fingerprint:
                        model_bytes = data
                        filename = fname
                        break

    # 4. Check transformed artifacts directory
    if model_bytes is None:
        if os.path.exists(store.storage_dir):
            for root, _, files in os.walk(store.storage_dir):
                for fname in files:
                    if fname.endswith(".onnx"):
                        fpath = os.path.join(root, fname)
                        with open(fpath, "rb") as f:
                            data = f.read()
                        if hashlib.sha256(data).hexdigest() == spec.model_fingerprint:
                            model_bytes = data
                            filename = fname
                            break
                if model_bytes is not None:
                    break

    if model_bytes is None:
        raise FileNotFoundError(
            f"Input model artifact with fingerprint {spec.model_fingerprint} could not be resolved."
        )

    # Enforce strict fingerprint integrity
    actual_hash = hashlib.sha256(model_bytes).hexdigest()
    if actual_hash != spec.model_fingerprint:
        raise ValueError(
            f"Model integrity violation: expected fingerprint {spec.model_fingerprint}, got {actual_hash}"
        )

    return model_bytes, filename


def execute_worker_task(
    spec: WorkerJobSpec,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[Dict[str, Any], List[str], TimingBreakdown]:
    """
    Execute a validated WorkerJobSpec using existing core engines.
    Returns: (result_dict, list_of_output_artifact_ids, TimingBreakdown)
    """
    t_start = time.time()
    if progress_callback:
        progress_callback(10, f"Resolving model for {spec.job_type.value} execution")

    # 1. Model Resolution (Startup overhead)
    model_bytes, filename = resolve_worker_model_bytes(spec)
    t_resolved = time.time()
    startup_overhead = t_resolved - t_start

    timing = TimingBreakdown(
        startup_overhead_seconds=startup_overhead,
    )
    output_artifact_ids: List[str] = []

    # 2. Dispatch by Job Type
    if spec.job_type == WorkerJobType.BENCHMARK:
        params = spec.parameters
        summary = load_and_inspect_model(model_bytes, filename=filename)
        b_cfg = BenchmarkConfig(
            execution_provider=params.get("execution_provider", "CPUExecutionProvider"),
            batch_size=int(params.get("batch_size", 1)),
            intra_op_num_threads=params.get("intra_op_num_threads"),
            inter_op_num_threads=params.get("inter_op_num_threads"),
            warmup_iterations=int(params.get("warmup_iterations", 5)),
            measured_iterations=int(params.get("measured_iterations", 20)),
            repeated_rounds=int(params.get("repeated_rounds", 3)),
            dynamic_dimensions=dict(params.get("dynamic_dimensions", {})),
        )

        t_bench_start = time.time()
        report = BenchmarkEngine.run_benchmark(
            model_path_or_bytes=model_bytes,
            model_summary=summary,
            config=b_cfg,
        )
        t_bench_end = time.time()

        # Pure inference measurement
        timing.inference_duration_seconds = t_bench_end - t_bench_start
        timing.warmup_overhead_seconds = 0.0

        # Persist report if database is accessible
        try:
            from zmodel.db.repositories import BenchmarkRepository
            from zmodel.db.session import session_scope
            with session_scope() as s:
                repo = BenchmarkRepository(session=s)
                repo.save_benchmark_report(report)
        except Exception as ex:
            logger.debug(f"Relational persistence skipped in worker: {ex}")

        result_dict = report.to_dict()

    elif spec.job_type == WorkerJobType.OPTIMIZATION:
        params = spec.parameters
        profiles = load_builtin_profiles()
        target_prof_id = spec.target_profile_id or params.get("profile_id") or "z16_integrated_ai"
        profile = profiles.get(target_prof_id, next(iter(profiles.values())))
        summary = load_and_inspect_model(model_bytes, filename=filename)

        raw_obj = params.get("primary_objective") or params.get("objective") or "MAXIMIZE_THROUGHPUT"
        try:
            objective = OptimizationObjective(raw_obj)
        except Exception:
            objective = OptimizationObjective.MAXIMIZE_THROUGHPUT

        hard_constraints = []
        for hc_data in params.get("hard_constraints", []):
            if isinstance(hc_data, dict):
                hard_constraints.append(
                    HardConstraint(
                        name=hc_data.get("name", ""),
                        metric_field=hc_data.get("metric_field", ""),
                        operator=hc_data.get("operator", "<="),
                        threshold=float(hc_data.get("threshold", 0.0)),
                        unit=hc_data.get("unit", ""),
                    )
                )

        opt_req = OptimizationRequest(
            model_fingerprint=summary.sha256_hash,
            profile_id=profile.profile_id,
            profile_version=profile.version,
            primary_objective=objective,
            hard_constraints=hard_constraints,
            max_candidates=int(params.get("max_candidates", 8)),
            batch_sizes=params.get("batch_sizes", [1, 2, 4]),
            intra_op_threads=params.get("intra_op_threads", [1, 2]),
            inter_op_threads=params.get("inter_op_threads", [1]),
            warmup_iterations=int(params.get("warmup_iterations", 3)),
            measured_iterations=int(params.get("measured_iterations", 10)),
            repeated_rounds=int(params.get("repeated_rounds", 3)),
            execution_providers=params.get("execution_providers", ["CPUExecutionProvider"]),
        )

        def engine_progress(stage: str, current: int, total: int):
            if progress_callback:
                pct = int((current / max(1, total)) * 100)
                progress_callback(pct, f"{stage}: {current}/{total}")

        t_opt_start = time.time()
        opt_report = OptimizationEngine.run_optimization(
            model_path_or_bytes=model_bytes,
            model_summary=summary,
            profile=profile,
            request=opt_req,
            progress_callback=engine_progress,
        )
        t_opt_end = time.time()
        timing.inference_duration_seconds = t_opt_end - t_opt_start

        # Persist report if database accessible
        try:
            from zmodel.db.repositories import OptimizationRepository
            from zmodel.db.session import session_scope
            with session_scope() as s:
                repo = OptimizationRepository(session=s)
                repo.save_experiment_report(opt_report)
        except Exception as ex:
            logger.debug(f"Relational persistence skipped in worker: {ex}")

        result_dict = opt_report.to_dict()

    elif spec.job_type == WorkerJobType.GRAPH_OPTIMIZATION:
        params = spec.parameters
        raw_transforms = params.get("transformations", ["shape_inference", "ort_basic", "ort_extended", "ort_all"])
        selected_transforms = []
        for t_str in raw_transforms:
            try:
                selected_transforms.append(TransformationType(t_str))
            except ValueError:
                pass

        val_depth = params.get("validation_depth", "FULL")
        enable_bench = bool(params.get("enable_benchmarking", True))

        engine = GraphOptimizationEngine(
            default_atol=float(params.get("atol", 1e-4)),
            default_rtol=float(params.get("rtol", 1e-3)),
        )

        t_graph_start = time.time()
        g_report = engine.run_optimization_experiment(
            model_bytes=model_bytes,
            original_filename=filename,
            selected_transforms=selected_transforms,
            target_profile_id=spec.target_profile_id or params.get("target_profile_id"),
            validation_depth=val_depth,
            enable_benchmarking=enable_bench,
            progress_callback=progress_callback,
        )
        t_graph_end = time.time()
        timing.inference_duration_seconds = t_graph_end - t_graph_start

        # Extract output artifact IDs
        for cand in g_report.candidates:
            if cand.artifact_id:
                output_artifact_ids.append(cand.artifact_id)

        # Persist report if database accessible
        try:
            from zmodel.db.repositories import GraphOptimizationRepository
            from zmodel.db.session import session_scope
            with session_scope() as s:
                repo = GraphOptimizationRepository(session=s)
                repo.create_experiment(g_report)
        except Exception as ex:
            logger.debug(f"Relational persistence skipped in worker: {ex}")

        result_dict = g_report.to_dict()

    else:
        raise ValueError(f"Unsupported WorkerJobType: {spec.job_type}")

    t_finish = time.time()
    timing.total_duration_seconds = t_finish - t_start
    if progress_callback:
        progress_callback(100, f"Completed {spec.job_type.value} execution")

    return result_dict, output_artifact_ids, timing
