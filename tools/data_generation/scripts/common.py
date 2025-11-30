"""Common utilities for test case generation.

Provides shared functionality for saving models, inputs, and expected predictions
across different model types (XGBoost, LightGBM, etc.).
"""
from pathlib import Path
import json
import numpy as np


def nan_to_null(obj):
    """Convert NaN values to None for JSON serialization."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, list):
        return [nan_to_null(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: nan_to_null(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return nan_to_null(obj.tolist())
    return obj


def save_test_input(path: Path, X_test: np.ndarray, feature_types: list[str] | None = None):
    """Save test input features to JSON.
    
    Args:
        path: Output file path
        X_test: Test features array
        feature_types: Optional list of feature types ('q' for quantitative, 'c' for categorical)
    """
    data = {
        "features": nan_to_null(X_test),
        "num_rows": len(X_test),
        "num_features": X_test.shape[1],
    }
    if feature_types:
        data["feature_types"] = feature_types
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_test_expected(
    path: Path,
    raw_predictions: np.ndarray,
    transformed_predictions: np.ndarray,
    objective: str,
    num_class: int = 0,
):
    """Save expected predictions to JSON.
    
    Args:
        path: Output file path
        raw_predictions: Raw margin predictions
        transformed_predictions: Transformed predictions (after sigmoid/softmax)
        objective: Objective function name
        num_class: Number of classes (0 for regression/binary)
    """
    with open(path, "w") as f:
        json.dump({
            "predictions": raw_predictions.tolist(),
            "predictions_transformed": transformed_predictions.tolist(),
            "objective": objective,
            "num_class": num_class,
        }, f, indent=2)


def print_success(name: str, num_cases: int, extra: str = ""):
    """Print success message for generated test case."""
    suffix = f" ({extra})" if extra else ""
    print(f"✓ {name}: model + {num_cases} test cases{suffix}")
