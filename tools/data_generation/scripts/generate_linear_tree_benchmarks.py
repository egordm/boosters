#!/usr/bin/env python3
"""Generate LightGBM linear tree models for benchmark comparison.

This script:
1. Creates synthetic linear regression data
2. Trains both standard and linear tree models
3. Saves models in text format for booste-rs loading
4. Measures training time for comparison

Usage:
    cd tools/data_generation && uv run python scripts/generate_linear_tree_benchmarks.py
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

# Output directory
BENCHMARK_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "test-cases" / "benchmark"


def generate_synthetic_linear(n_rows: int, n_cols: int, seed: int = 42, noise: float = 0.05):
    """Generate synthetic linear regression data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_rows, n_cols)).astype(np.float32)
    
    # True weights (linear relationship)
    weights = rng.standard_normal(n_cols).astype(np.float32)
    
    # Target with noise
    y = X @ weights + rng.normal(0, noise, size=n_rows).astype(np.float32)
    
    return X, y, weights


def train_and_save(name: str, n_rows: int, n_cols: int, n_trees: int = 50, 
                   max_depth: int = 6, linear_tree: bool = False, seed: int = 42):
    """Train a model and save it."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"  Rows: {n_rows}, Cols: {n_cols}, Trees: {n_trees}, Depth: {max_depth}")
    print(f"  Linear tree: {linear_tree}")
    print(f"{'='*60}")
    
    X, y, _ = generate_synthetic_linear(n_rows, n_cols, seed)
    
    params = {
        "objective": "regression",
        "metric": "l2",
        "num_iterations": n_trees,
        "learning_rate": 0.1,
        "max_depth": max_depth,
        "num_leaves": 2 ** max_depth,
        "min_data_in_leaf": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
    }
    
    if linear_tree:
        params["linear_tree"] = True
        params["linear_lambda"] = 0.01
    
    dataset = lgb.Dataset(X, label=y)
    
    # Warm up - train once to avoid first-run overhead
    warmup_params = {**params, "num_iterations": 5}
    _ = lgb.train(warmup_params, dataset)
    
    # Time training
    start = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        booster = lgb.train(params, dataset)
    elapsed = (time.perf_counter() - start) / n_runs
    
    print(f"  Training time: {elapsed*1000:.2f} ms (avg of {n_runs} runs)")
    
    # Save model
    model_path = BENCHMARK_DIR / f"{name}.lgb.txt"
    booster.save_model(str(model_path))
    print(f"  Saved: {model_path}")
    
    # Verify predictions
    preds = booster.predict(X[:10])
    print(f"  Sample predictions: {preds[:5]}")
    
    # Save timing info
    timing_path = BENCHMARK_DIR / f"{name}.timing.json"
    timing_data = {
        "name": name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_trees": n_trees,
        "max_depth": max_depth,
        "linear_tree": linear_tree,
        "training_time_ms": elapsed * 1000,
        "n_runs": n_runs,
    }
    with open(timing_path, "w") as f:
        json.dump(timing_data, f, indent=2)
    
    return elapsed


def main():
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    print("LightGBM Linear Tree Benchmark Model Generator")
    print("=" * 60)
    
    configs = [
        # Small dataset - for quick iterations
        ("bench_linear_small", 5_000, 50, 50, 6, True),
        ("bench_standard_small", 5_000, 50, 50, 6, False),
        
        # Medium dataset - primary comparison
        ("bench_linear_medium", 50_000, 100, 50, 6, True),
        ("bench_standard_medium", 50_000, 100, 50, 6, False),
    ]
    
    timings = []
    for name, n_rows, n_cols, n_trees, max_depth, linear_tree in configs:
        elapsed = train_and_save(name, n_rows, n_cols, n_trees, max_depth, linear_tree)
        timings.append({
            "name": name,
            "linear_tree": linear_tree,
            "training_time_ms": elapsed * 1000,
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Time Summary")
    print("=" * 60)
    print(f"{'Model':<30} {'Linear':>8} {'Time (ms)':>12}")
    print("-" * 60)
    for t in timings:
        linear_str = "Yes" if t["linear_tree"] else "No"
        print(f"{t['name']:<30} {linear_str:>8} {t['training_time_ms']:>12.2f}")
    
    # Calculate overhead
    print("\nOverhead Analysis:")
    for i in range(0, len(timings), 2):
        linear = timings[i]
        standard = timings[i + 1]
        overhead = (linear["training_time_ms"] / standard["training_time_ms"] - 1) * 100
        size = linear["name"].split("_")[-1]
        print(f"  {size}: Linear tree overhead = {overhead:+.1f}%")
    
    print("\nDone! Models saved to:", BENCHMARK_DIR)


if __name__ == "__main__":
    main()
