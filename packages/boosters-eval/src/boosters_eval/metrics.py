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
LOWER_BETTER_METRICS = frozenset({
    "rmse",
    "mae",
    "rmae",
    "pinball",
    "rcrps",
    "logloss",
    "mlogloss",
    "peak_memory_mb",
})


def _validate_sample_weight(
    y_true: NDArray[np.floating[Any]],
    sample_weight: NDArray[np.floating[Any]] | None,
) -> NDArray[np.floating[Any]] | None:
    if sample_weight is None:
        return None

    w = np.asarray(sample_weight, dtype=np.float64)
    if w.shape != np.asarray(y_true).shape:
        raise ValueError("sample_weight must have the same shape as y_true")
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must be finite")
    if np.any(w < 0):
        raise ValueError("sample_weight must be non-negative")
    if float(np.sum(w)) <= 0.0:
        # Treat all-zero weights as unweighted to avoid division by zero.
        return None
    return w


def _weighted_quantile_1d(
    x: NDArray[np.floating[Any]],
    q: float,
    *,
    sample_weight: NDArray[np.floating[Any]] | None,
) -> float:
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")

    x_arr = np.asarray(x, dtype=np.float64)
    if sample_weight is None:
        return float(np.quantile(x_arr, q=q))

    w = np.asarray(sample_weight, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("x must be 1D")
    if w.shape != x_arr.shape:
        raise ValueError("sample_weight must have the same shape as x")

    order = np.argsort(x_arr)
    x_sorted = x_arr[order]
    w_sorted = w[order]

    w_sum = float(np.sum(w_sorted))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return float(np.quantile(x_sorted, q=q))

    cdf = np.cumsum(w_sorted) / w_sum
    return float(np.interp(q, cdf, x_sorted))


def pinball_loss(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    quantiles: NDArray[np.floating[Any]],
    *,
    sample_weight: NDArray[np.floating[Any]] | None = None,
) -> float:
    """Mean pinball loss averaged over samples and quantiles."""
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("y_true must be 1D")
    if pred.ndim != 2:
        raise ValueError("y_pred must be 2D for quantile regression")
    if pred.shape[0] != y.shape[0]:
        raise ValueError("y_pred must have the same number of rows as y_true")
    if pred.shape[1] != q.shape[0]:
        raise ValueError("y_pred second dimension must match len(quantiles)")
    if q.ndim != 1:
        raise ValueError("quantiles must be 1D")
    if np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError("quantiles must be strictly within (0, 1)")
    if not np.all(np.diff(q) > 0):
        raise ValueError("quantiles must be strictly increasing")

    w = _validate_sample_weight(y, sample_weight)

    err = y[:, None] - pred
    # rho_q(err) = q*err if err>=0 else (q-1)*err
    loss = np.where(err >= 0.0, q[None, :] * err, (q[None, :] - 1.0) * err)

    if w is None:
        return float(np.mean(loss))
    return float(np.sum(loss * w[:, None]) / (float(np.sum(w)) * float(len(q))))


def crps_from_quantiles(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    quantiles: NDArray[np.floating[Any]],
    *,
    sample_weight: NDArray[np.floating[Any]] | None = None,
) -> float:
    r"""Approximate CRPS from quantile predictions.

    Uses the identity $\mathrm{CRPS} = 2\int_0^1 \rho_u(y - q_u)\,du$ and
    numerically integrates over the provided quantile levels.
    """
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)

    if y.ndim != 1:
        raise ValueError("y_true must be 1D")
    if pred.ndim != 2:
        raise ValueError("y_pred must be 2D for quantile regression")
    if pred.shape[0] != y.shape[0]:
        raise ValueError("y_pred must have the same number of rows as y_true")
    if pred.shape[1] != q.shape[0]:
        raise ValueError("y_pred second dimension must match len(quantiles)")
    if q.ndim != 1:
        raise ValueError("quantiles must be 1D")
    if np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError("quantiles must be strictly within (0, 1)")
    if not np.all(np.diff(q) > 0):
        raise ValueError("quantiles must be strictly increasing")

    w = _validate_sample_weight(y, sample_weight)

    # Extend to [0, 1] by clamping the extreme predicted quantiles.
    q_ext = np.concatenate(([0.0], q, [1.0]))
    pred_ext = np.concatenate((pred[:, :1], pred, pred[:, -1:]), axis=1)

    err = y[:, None] - pred_ext
    loss = np.where(err >= 0.0, q_ext[None, :] * err, (q_ext[None, :] - 1.0) * err)
    crps_per_sample = 2.0 * np.trapezoid(loss, x=q_ext, axis=1)

    if w is None:
        return float(np.mean(crps_per_sample))
    return float(np.sum(crps_per_sample * w) / float(np.sum(w)))


def rcrps(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    quantiles: NDArray[np.floating[Any]],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    sample_weight: NDArray[np.floating[Any]] | None = None,
) -> float:
    """Relative CRPS: CRPS normalized by the (weighted) target range."""
    y = np.asarray(y_true, dtype=np.float64)
    w = _validate_sample_weight(y, sample_weight)

    y_range = _weighted_quantile_1d(y, upper_quantile, sample_weight=w) - _weighted_quantile_1d(
        y, lower_quantile, sample_weight=w
    )
    if not np.isfinite(y_range) or y_range == 0.0:
        return float("NaN")

    return float(crps_from_quantiles(y, y_pred, quantiles, sample_weight=w) / y_range)


def compute_metrics(
    task: Task,
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
    *,
    n_classes: int | None = None,
    sample_weight: NDArray[np.floating[Any]] | None = None,
    quantiles: NDArray[np.floating[Any]] | None = None,
) -> dict[str, float]:
    """Compute evaluation metrics using sklearn.

    Args:
        task: The ML task type (regression, binary, multiclass).
        y_true: Ground truth labels.
        y_pred: Predictions (probabilities for classification).
        n_classes: Number of classes for multiclass.
        sample_weight: Optional per-row weights.
        quantiles: Quantile levels for quantile regression (must match y_pred columns).

    Returns:
        Dictionary of metric name to value.
    """
    sample_weight = _validate_sample_weight(y_true, sample_weight)

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
    if task == Task.QUANTILE_REGRESSION:
        if quantiles is None:
            raise ValueError("quantiles must be provided for quantile regression metrics")
        q = np.asarray(quantiles, dtype=np.float64)
        return {
            "pinball": float(pinball_loss(y_true, y_pred, q, sample_weight=sample_weight)),
            "rcrps": float(rcrps(y_true, y_pred, q, sample_weight=sample_weight)),
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
        Task.QUANTILE_REGRESSION: "rcrps",
        Task.BINARY: "logloss",
        Task.MULTICLASS: "mlogloss",
    }[task]


def is_lower_better(metric: str) -> bool:
    """Check if lower values are better for a metric."""
    return metric in LOWER_BETTER_METRICS
