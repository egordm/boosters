"""Common utilities for test data generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rich.console import Console

if TYPE_CHECKING:
    from typing import Any

console = Console()


def nan_to_null(obj: Any) -> Any:
    """Convert NaN values to None for JSON serialization."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, list):
        return [nan_to_null(item) for item in obj]
    if isinstance(obj, dict):
        return {k: nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return nan_to_null(obj.tolist())
    return obj


def save_json(path: Path, data: dict) -> None:
    """Save data to JSON file."""
    path_obj = Path(path) if not isinstance(path, Path) else path
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w") as f:
        json.dump(data, f, indent=2)


def save_test_input(
    path: Path,
    x_test: np.ndarray,
    feature_types: list[str] | None = None,
) -> None:
    """Save test input features to JSON."""
    data = {
        "features": nan_to_null(x_test),
        "num_rows": len(x_test),
        "num_features": x_test.shape[1],
    }
    if feature_types:
        data["feature_types"] = feature_types
    save_json(path, data)


def save_test_expected(
    path: Path,
    raw_predictions: np.ndarray,
    transformed_predictions: np.ndarray,
    objective: str,
    num_class: int = 0,
) -> None:
    """Save expected predictions to JSON."""
    save_json(
        path,
        {
            "predictions": raw_predictions.tolist(),
            "predictions_transformed": transformed_predictions.tolist(),
            "objective": objective,
            "num_class": num_class,
        },
    )


def print_success(name: str, num_cases: int, extra: str = "") -> None:
    """Print success message for generated test case."""
    suffix = f" ({extra})" if extra else ""
    console.print(f"[green]✓[/green] {name}: model + {num_cases} test cases{suffix}")
