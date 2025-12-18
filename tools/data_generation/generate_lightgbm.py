#!/usr/bin/env python3
"""Generate LightGBM test models and expected predictions."""

import json
import math
import os
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification, make_regression

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "tests" / "test-cases" / "lightgbm" / "inference"


class NaNEncoder(json.JSONEncoder):
    """JSON encoder that converts NaN to null."""
    def default(self, obj):
        return super().default(obj)

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(self._convert_nans(obj), _one_shot)

    def _convert_nans(self, obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        if isinstance(obj, list):
            return [self._convert_nans(item) for item in obj]
        if isinstance(obj, dict):
            return {k: self._convert_nans(v) for k, v in obj.items()}
        return obj


def save_test_case(name: str, model: lgb.Booster, X: np.ndarray, y: np.ndarray, 
                   expected_raw: np.ndarray, expected_proba: np.ndarray | None = None):
    """Save model, input, and expected outputs."""
    output_dir = OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model in text format
    model.save_model(str(output_dir / "model.txt"))
    
    # Save model in JSON format
    model_json = model.dump_model()
    with open(output_dir / "model.json", "w") as f:
        json.dump(model_json, f, indent=2)
    
    # Save input data (use custom encoder for NaN)
    test_data = {
        "num_samples": X.shape[0],
        "num_features": X.shape[1],
        "data": X.tolist(),
        "labels": y.tolist() if y is not None else None,
    }
    with open(output_dir / "input.json", "w") as f:
        json.dump(test_data, f, indent=2, cls=NaNEncoder)
    
    # Save expected outputs
    expected = {
        "raw": expected_raw.tolist() if expected_raw.ndim == 1 else [row.tolist() for row in expected_raw],
    }
    if expected_proba is not None:
        expected["proba"] = expected_proba.tolist() if expected_proba.ndim == 1 else [row.tolist() for row in expected_proba]
    
    with open(output_dir / "expected.json", "w") as f:
        json.dump(expected, f, indent=2)
    
    print(f"Saved test case: {name}")
    print(f"  - Model: {output_dir / 'model.txt'}")
    print(f"  - Samples: {X.shape[0]}, Features: {X.shape[1]}")


def generate_regression_test():
    """Generate a regression test case."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    
    # Train model
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 10,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # Get predictions
    test_X = X[:20]  # Use first 20 samples for testing
    test_y = y[:20]
    raw_preds = model.predict(test_X, raw_score=True)
    
    save_test_case("regression", model, test_X, test_y, raw_preds)


def generate_binary_classification_test():
    """Generate a binary classification test case."""
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, 
                                n_redundant=2, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.int32)
    
    # Train model
    params = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 10,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # Get predictions
    test_X = X[:20]
    test_y = y[:20]
    raw_preds = model.predict(test_X, raw_score=True)
    proba_preds = model.predict(test_X, raw_score=False)
    
    save_test_case("binary_classification", model, test_X, test_y, raw_preds, proba_preds)


def generate_multiclass_test():
    """Generate a multiclass classification test case."""
    np.random.seed(42)
    X, y = make_classification(n_samples=150, n_features=10, n_informative=5, 
                                n_redundant=2, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.int32)
    
    # Train model
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "n_estimators": 10,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # Get predictions
    test_X = X[:20]
    test_y = y[:20]
    raw_preds = model.predict(test_X, raw_score=True)
    proba_preds = model.predict(test_X, raw_score=False)
    
    save_test_case("multiclass", model, test_X, test_y, raw_preds, proba_preds)


def generate_missing_values_test():
    """Generate a test case with missing values (NaN)."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    
    # Introduce missing values
    mask = np.random.random(X.shape) < 0.1
    X[mask] = np.nan
    
    # Train model
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 10,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=10)
    
    # Get predictions
    test_X = X[:20]
    test_y = y[:20]
    raw_preds = model.predict(test_X, raw_score=True)
    
    save_test_case("regression_missing", model, test_X, test_y, raw_preds)


def generate_small_tree_test():
    """Generate a simple test case with a small tree for manual verification."""
    np.random.seed(42)
    X, y = make_regression(n_samples=50, n_features=5, noise=0.1, random_state=42)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    
    # Train small model
    params = {
        "objective": "regression",
        "num_leaves": 4,  # Very small tree
        "learning_rate": 0.5,
        "n_estimators": 3,  # Very few trees
        "verbose": -1,
        "seed": 42,
        "min_data_in_leaf": 1,
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=3)
    
    # Get predictions
    test_X = X[:10]
    test_y = y[:10]
    raw_preds = model.predict(test_X, raw_score=True)
    
    save_test_case("small_tree", model, test_X, test_y, raw_preds)


# =============================================================================
# Benchmark Models
# =============================================================================

BENCHMARK_DIR = Path(__file__).parent.parent.parent / "tests" / "test-cases" / "benchmark"


def save_benchmark_model(name: str, model: lgb.Booster, num_features: int):
    """Save benchmark model in text format."""
    output_path = BENCHMARK_DIR / f"{name}.lgb.txt"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    print(f"Saved benchmark model: {name}")
    print(f"  - File: {output_path}")
    print(f"  - Features: {num_features}")
    print(f"  - Trees: {model.num_trees()}")


def generate_bench_small():
    """Small benchmark model: 20 trees, 5 features, num_leaves=8.
    
    Comparable to XGBoost bench_small: 20 trees, 5 features, max_depth=3.
    """
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=5, n_informative=4, noise=5.0, random_state=42)
    
    params = {
        "objective": "regression",
        "num_leaves": 8,  # ~equivalent to max_depth=3
        "learning_rate": 0.1,
        "feature_fraction": 1.0,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=20)
    save_benchmark_model("bench_small", model, num_features=5)


def generate_bench_medium():
    """Medium benchmark model: 100 trees, 50 features, num_leaves=16.
    
    Comparable to XGBoost bench_medium: 100 trees, 50 features, max_depth=4.
    """
    np.random.seed(42)
    X, y = make_regression(n_samples=2000, n_features=50, n_informative=30, noise=10.0, random_state=42)
    
    params = {
        "objective": "regression",
        "num_leaves": 16,  # ~equivalent to max_depth=4
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=100)
    save_benchmark_model("bench_medium", model, num_features=50)


def generate_bench_large():
    """Large benchmark model: 500 trees, 100 features, num_leaves=32.
    
    Comparable to XGBoost bench_large: 500 trees, 100 features, max_depth=5.
    """
    np.random.seed(42)
    X, y = make_regression(n_samples=5000, n_features=100, n_informative=50, noise=10.0, random_state=42)
    
    params = {
        "objective": "regression",
        "num_leaves": 32,  # ~equivalent to max_depth=5
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=500)
    save_benchmark_model("bench_large", model, num_features=100)


def generate_linear_tree_test():
    """Generate a test case with linear leaves."""
    np.random.seed(42)
    
    # Create data with a strong linear relationship that benefits from linear leaves
    X = np.random.randn(200, 5).astype(np.float64)
    # y is a linear combination plus some nonlinear structure
    y = 2.0 * X[:, 0] + 3.0 * X[:, 1] - 1.5 * X[:, 2] + 10.0
    # Add nonlinear component via interaction
    y += 0.5 * X[:, 0] * X[:, 1]
    y += np.random.randn(200) * 0.1  # Small noise
    
    # Train model with linear_tree=True
    params = {
        "objective": "regression",
        "num_leaves": 8,
        "learning_rate": 0.3,
        "n_estimators": 5,
        "verbose": -1,
        "seed": 42,
        "min_data_in_leaf": 10,
        "linear_tree": True,  # Enable linear leaves
    }
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=5)
    
    # Get predictions
    test_X = X[:20]
    test_y = y[:20]
    raw_preds = model.predict(test_X, raw_score=True)
    
    save_test_case("linear_tree", model, test_X, test_y, raw_preds)


def main():
    """Generate all test cases."""
    print(f"Output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Generating LightGBM Test Cases ===\n")
    
    generate_small_tree_test()
    generate_regression_test()
    generate_binary_classification_test()
    generate_multiclass_test()
    generate_missing_values_test()
    generate_linear_tree_test()
    
    print("\n=== Generating LightGBM Benchmark Models ===\n")
    
    generate_bench_small()
    generate_bench_medium()
    generate_bench_large()
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
