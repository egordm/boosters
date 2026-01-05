"""Small utilities shared across the evaluation framework."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from boosters_eval.config import LoadedDataset


def rolling_origin_splitter(
    data: LoadedDataset,
    seed: int,
    *,
    train_window: int,
    valid_window: int,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]] | None,
    NDArray[np.floating[Any]] | None,
]:
    """Time-series split with a rolling cut point.

    Picks a cut point (end of train) based on seed, then returns:
      train = [cut - train_window : cut]
      valid = [cut : cut + valid_window]

    This matches forecasting evaluation where the validation horizon directly
    follows the training window.
    """
    if train_window <= 0:
        raise ValueError("train_window must be positive")
    if valid_window <= 0:
        raise ValueError("valid_window must be positive")

    n = len(data.y)
    if n < train_window + valid_window:
        raise ValueError(
            "Time-series split requires enough samples for both windows; "
            f"need at least {train_window + valid_window}, got {n}."
        )

    rng = np.random.default_rng(seed)
    cut = int(rng.integers(low=train_window, high=(n - valid_window + 1)))

    train_start = cut - train_window
    train_end = cut
    valid_end = cut + valid_window

    return (
        data.x[train_start:train_end],
        data.x[train_end:valid_end],
        data.y[train_start:train_end],
        data.y[train_end:valid_end],
        (data.sample_weight[train_start:train_end] if data.sample_weight is not None else None),
        (data.sample_weight[train_end:valid_end] if data.sample_weight is not None else None),
    )


def train_valid_split(
    data: LoadedDataset,
    seed: int,
    *,
    validation_fraction: float = 0.2,
    max_samples: int | None = None,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]] | None,
    NDArray[np.floating[Any]] | None,
]:
    """Default dataset splitter.

    Uses sklearn's `train_test_split` under the hood and optionally subsamples the
    dataset first (useful for very large datasets).

    Splits sample weights when present.
    """
    x = data.x
    y = data.y
    w = data.sample_weight

    if max_samples is not None and len(y) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), max_samples, replace=False)
        x = np.asarray(x)[idx]
        y = np.asarray(y)[idx]
        if w is not None:
            w = np.asarray(w)[idx]

    if w is None:
        x_train, x_valid, y_train, y_valid = train_test_split(
            x,
            y,
            test_size=validation_fraction,
            random_state=seed,
        )
        return x_train, x_valid, y_train, y_valid, None, None

    x_train, x_valid, y_train, y_valid, w_train, w_valid = train_test_split(
        x,
        y,
        w,
        test_size=validation_fraction,
        random_state=seed,
    )
    return x_train, x_valid, y_train, y_valid, w_train, w_valid
