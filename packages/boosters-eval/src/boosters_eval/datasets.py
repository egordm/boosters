"""Dataset configurations for benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)

from boosters_eval.config import DatasetConfig, Task


def california_housing() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """California housing regression dataset."""
    data = fetch_california_housing()
    return data.data.astype(np.float32), data.target.astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]


def breast_cancer() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Breast cancer binary classification dataset."""
    data = load_breast_cancer()
    return data.data.astype(np.float32), data.target.astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]


def iris() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Iris multiclass classification dataset."""
    data = load_iris()
    return data.data.astype(np.float32), data.target.astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]


def covertype() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Covertype multiclass classification dataset (subsampled)."""
    data = fetch_covtype()
    return data.data.astype(np.float32), (data.target - 1).astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]


def _synthetic_regression(
    n_samples: int = 10000,
    n_features: int = 50,
    noise: float = 0.1,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Generate synthetic regression dataset."""
    x, y = make_regression(  # pyright: ignore[reportAssignmentType]
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=42,
    )
    return x.astype(np.float32), y.astype(np.float32)


def _synthetic_classification(
    n_samples: int = 10000,
    n_features: int = 50,
    n_classes: int = 2,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Generate synthetic classification dataset."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=min(n_features // 2, n_classes * 2),
        n_redundant=n_features // 4,
        random_state=42,
    )
    return x.astype(np.float32), y.astype(np.float32)


# Dataset loader factories
def synthetic_regression_small() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Small synthetic regression dataset."""
    return _synthetic_regression(2000, 50)


def synthetic_regression_medium() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Medium synthetic regression dataset."""
    return _synthetic_regression(10000, 100)


def synthetic_binary_small() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Small synthetic binary classification dataset."""
    return _synthetic_classification(2000, 50, 2)


def synthetic_binary_medium() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Medium synthetic binary classification dataset."""
    return _synthetic_classification(10000, 100, 2)


def synthetic_multi_small() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Small synthetic multiclass classification dataset."""
    return _synthetic_classification(2000, 50, 5)


# Pre-defined dataset configurations
DATASETS: dict[str, DatasetConfig] = {
    # Regression
    "california": DatasetConfig(
        name="california",
        task=Task.REGRESSION,
        loader=california_housing,
    ),
    "synthetic_reg_small": DatasetConfig(
        name="synthetic_reg_small",
        task=Task.REGRESSION,
        loader=synthetic_regression_small,
    ),
    "synthetic_reg_medium": DatasetConfig(
        name="synthetic_reg_medium",
        task=Task.REGRESSION,
        loader=synthetic_regression_medium,
    ),
    # Binary classification
    "breast_cancer": DatasetConfig(
        name="breast_cancer",
        task=Task.BINARY,
        loader=breast_cancer,
    ),
    "synthetic_bin_small": DatasetConfig(
        name="synthetic_bin_small",
        task=Task.BINARY,
        loader=synthetic_binary_small,
    ),
    "synthetic_bin_medium": DatasetConfig(
        name="synthetic_bin_medium",
        task=Task.BINARY,
        loader=synthetic_binary_medium,
    ),
    # Multiclass classification
    "iris": DatasetConfig(
        name="iris",
        task=Task.MULTICLASS,
        loader=iris,
        n_classes=3,
    ),
    "covertype": DatasetConfig(
        name="covertype",
        task=Task.MULTICLASS,
        loader=covertype,
        n_classes=7,
        subsample=50000,
    ),
    "synthetic_multi_small": DatasetConfig(
        name="synthetic_multi_small",
        task=Task.MULTICLASS,
        loader=synthetic_multi_small,
        n_classes=5,
    ),
}


def get_datasets_by_task(task: Task) -> dict[str, DatasetConfig]:
    """Get all datasets for a specific task."""
    return {k: v for k, v in DATASETS.items() if v.task == task}
