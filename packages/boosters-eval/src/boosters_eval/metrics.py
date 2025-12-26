"""Metrics computation using sklearn."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)

from boosters_eval.datasets import Task


def compute_metrics(
    task: Task,
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    n_classes: int | None = None,
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
    if task == Task.REGRESSION:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    if task == Task.BINARY:
        y_pred_class = (np.asarray(y_pred) >= 0.5).astype(int)
        return {
            "logloss": float(log_loss(y_true, y_pred, labels=[0, 1])),
            "accuracy": float(accuracy_score(y_true, y_pred_class)),
        }
    # MULTICLASS
    y_pred_class = np.argmax(y_pred, axis=1)
    labels = list(range(n_classes)) if n_classes else None
    return {
        "mlogloss": float(log_loss(y_true, y_pred, labels=labels)),
        "accuracy": float(accuracy_score(y_true, y_pred_class)),
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
    return metric in ("rmse", "mae", "logloss", "mlogloss")
