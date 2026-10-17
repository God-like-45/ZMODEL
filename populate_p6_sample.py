"""Script to seed real benchmark and optimization experiments into the database for UI verification."""

import os
from pathlib import Path

# Enable SQLite fallback for local verification
os.environ["DB_ALLOW_SQLITE_FALLBACK"] = "true"
os.environ["SQLITE_DATABASE_PATH"] = "zmodel_experiments.db"

from zmodel.core.benchmark import BenchmarkConfig, BenchmarkEngine
from zmodel.core.compatibility import CompatibilityAnalyzer
from zmodel.core.model_loader import load_and_inspect_model
from zmodel.core.optimizer import (
    HardConstraint,
    OptimizationEngine,
    OptimizationObjective,
    OptimizationRequest,
)
from zmodel.core.profiles import load_builtin_profiles
from zmodel.db import (
    BenchmarkRepository,
    CompatibilityRepository,
    ModelRepository,
    OptimizationRepository,
    init_db,
)


def seed_data():
    print("Initializing database tables...")
    init_db()

    # Load model 1: linear_classifier.onnx
    model_path_1 = Path("sample_models/linear_classifier.onnx")
    m_bytes_1 = model_path_1.read_bytes()
    summary_1 = load_and_inspect_model(m_bytes_1, filename="linear_classifier.onnx")

    model_repo = ModelRepository()
    model_repo.register_or_update(summary_1)
    print(f"Registered model 1: {summary_1.filename} ({summary_1.sha256_hash[:12]})")

    # Load model 2: conv_net.onnx
    model_path_2 = Path("sample_models/conv_net.onnx")
    m_bytes_2 = model_path_2.read_bytes()
    summary_2 = load_and_inspect_model(m_bytes_2, filename="conv_net.onnx")
    model_repo.register_or_update(summary_2)
    print(f"Registered model 2: {summary_2.filename} ({summary_2.sha256_hash[:12]})")

    profiles = load_builtin_profiles()
    cpu_profile = profiles["local_onnx_runtime_cpu"]

    # Save compatibility assessment
    compat_rep = CompatibilityAnalyzer.analyze(summary_1, cpu_profile)
    compat_repo = CompatibilityRepository()
    compat_repo.save_assessment(compat_rep)
    print("Saved compatibility assessment.")

    # Run Benchmark 1 (Baseline: 1 thread)
    print("Running Benchmark 1 (1 thread)...")
    cfg_1 = BenchmarkConfig(batch_size=1, intra_op_num_threads=1, warmup_iterations=5, measured_iterations=20)
    rep_1 = BenchmarkEngine.run_benchmark(m_bytes_1, summary_1, cfg_1)
    bench_repo = BenchmarkRepository()
    bench_repo.save_benchmark_report(rep_1)
    print(f"Saved Benchmark 1: {rep_1.run_id} ({rep_1.latency['median_ms']:.4f} ms)")

    # Run Benchmark 2 (2 threads)
    print("Running Benchmark 2 (2 threads)...")
    cfg_2 = BenchmarkConfig(batch_size=1, intra_op_num_threads=2, warmup_iterations=5, measured_iterations=20)
    rep_2 = BenchmarkEngine.run_benchmark(m_bytes_1, summary_1, cfg_2)
    bench_repo.save_benchmark_report(rep_2)
    print(f"Saved Benchmark 2: {rep_2.run_id} ({rep_2.latency['median_ms']:.4f} ms)")

    # Run Benchmark 3 (conv_net model)
    print("Running Benchmark 3 (conv_net)...")
    cfg_3 = BenchmarkConfig(batch_size=1, intra_op_num_threads=1, warmup_iterations=2, measured_iterations=5)
    rep_3 = BenchmarkEngine.run_benchmark(m_bytes_2, summary_2, cfg_3)
    bench_repo.save_benchmark_report(rep_3)
    print(f"Saved Benchmark 3: {rep_3.run_id} (conv_net)")

    # Run Real Optimization Search
    print("Running Optimization Search...")
    opt_req = OptimizationRequest(
        model_fingerprint=summary_1.sha256_hash,
        profile_id="local_onnx_runtime_cpu",
        profile_version=cpu_profile.version,
        primary_objective=OptimizationObjective.MINIMIZE_MEDIAN_LATENCY,
        hard_constraints=[HardConstraint(name="Max Median Latency", metric_field="median_latency_ms", operator="<=", threshold=5.0, unit="ms")],
        batch_sizes=[1, 2],
        intra_op_threads=[1, 2],
        inter_op_threads=[1],
        warmup_iterations=5,
        measured_iterations=15,
        repeated_rounds=2,
    )
    opt_rep = OptimizationEngine.run_optimization(m_bytes_1, summary_1, cpu_profile, opt_req)
    opt_repo = OptimizationRepository()
    opt_repo.save_optimization_report(opt_rep)
    print(f"Saved Optimization Experiment: {opt_rep.experiment_id} (Best: {opt_rep.best_candidate_id})")

    print("Data seeding completed successfully!")


if __name__ == "__main__":
    seed_data()
