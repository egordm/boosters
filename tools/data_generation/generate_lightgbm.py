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
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
