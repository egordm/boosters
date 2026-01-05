"""Metrics computation using sklearn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from boosters_eval.config import Task

# Classification threshold for binary classification
BINARY_THRESHOLD = 0.5

# Metrics where lower values are better
LOWER_BETTER_METRICS = frozenset({"rmse", "mae", "rmae", "logloss", "mlogloss", "peak_memory_mb"})


def compute_metrics(
    task: Task,
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    *,
    n_classes: int | None = None,
    sample_weight: NDArray[np.floating[Any]] | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics using sklearn.

    Args:
        task: The ML task type (regression, binary, multiclass).
        y_true: Ground truth labels.
        y_pred: Predictions (probabilities for classification).
        n_classes: Number of classes for multiclass.

    Returns:
        Dictionary of metric name to value.
    """
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.shape != np.asarray(y_true).shape:
            raise ValueError("sample_weight must have the same shape as y_true")
        if not np.all(np.isfinite(w)):
            raise ValueError("sample_weight must be finite")
        if np.any(w < 0):
            raise ValueError("sample_weight must be non-negative")
        if float(np.sum(w)) <= 0.0:
            # Treat all-zero weights as unweighted to avoid division by zero.
            sample_weight = None

    if task == Task.REGRESSION:
        # Relative MAE: MAE normalized by the mean absolute target value.
        # Useful for time-series forecasting where absolute scales vary.
        abs_y = np.abs(np.asarray(y_true, dtype=np.float64))
        denom = float(np.average(abs_y, weights=sample_weight)) if sample_weight is not None else float(np.mean(abs_y))
        denom = denom if denom > 0.0 else 1.0
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))),
            "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)),
            "rmae": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight) / denom),
            "r2": float(r2_score(y_true, y_pred, sample_weight=sample_weight)),
        }
    if task == Task.BINARY:
        y_pred_class = (np.asarray(y_pred) >= BINARY_THRESHOLD).astype(int)
        return {
            "logloss": float(log_loss(y_true, y_pred, labels=[0, 1], sample_weight=sample_weight)),
            "accuracy": float(accuracy_score(y_true, y_pred_class, sample_weight=sample_weight)),
        }
    # MULTICLASS
    y_pred_class = np.argmax(y_pred, axis=1)
    labels = list(range(n_classes)) if n_classes else None
    return {
        "mlogloss": float(log_loss(y_true, y_pred, labels=labels, sample_weight=sample_weight)),
        "accuracy": float(accuracy_score(y_true, y_pred_class, sample_weight=sample_weight)),
    }


def primary_metric(task: Task) -> str:
    """Get the primary metric for a task type."""
    return {
        Task.REGRESSION: "rmse",
        Task.BINARY: "logloss",
        Task.MULTICLASS: "mlogloss",
    }[task]


def is_lower_better(metric: str) -> bool:
    """Check if lower values are better for a metric."""
    return metric in LOWER_BETTER_METRICS
