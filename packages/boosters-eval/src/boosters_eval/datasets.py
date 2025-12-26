"""Dataset configurations for benchmarks."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 (used at runtime by pydantic)
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray  # noqa: TC002 (used at runtime by pydantic)
from pydantic import BaseModel, ConfigDict
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)


class Task(str, Enum):
    """Machine learning task type."""

    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"


class BoosterType(str, Enum):
    """Type of gradient boosting model."""

    GBDT = "gbdt"  # Standard gradient boosted decision trees
    GBLINEAR = "gblinear"  # Linear booster (XGBoost)
    LINEAR_TREES = "linear_trees"  # GBDT with linear leaves (LightGBM)


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    task: Task
    loader: Callable[[], tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]]
    n_classes: int | None = None
    subsample: int | None = None  # Subsample large datasets


class TrainingConfig(BaseModel):
    """Training hyperparameters - consistent across libraries."""

    model_config = ConfigDict(frozen=True)

    n_trees: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_threads: int = 1


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark comparison."""

    model_config = ConfigDict(frozen=True)

    name: str
    dataset: DatasetConfig
    training: TrainingConfig = TrainingConfig()
    booster_type: BoosterType = BoosterType.GBDT
    libraries: list[str] = ["xgboost", "lightgbm"]


# =============================================================================
# Dataset Loaders (using sklearn)
# =============================================================================


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
    """Covertype multiclass classification dataset."""
    data = fetch_covtype()
    return data.data.astype(np.float32), (data.target - 1).astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]


def synthetic_regression(
    n_samples: int = 10000, n_features: int = 50, noise: float = 0.1
) -> Callable[[], tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]]:
    """Factory for synthetic regression datasets."""

    def loader() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        x, y = make_regression(  # pyright: ignore[reportAssignmentType]
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=42,
        )
        return x.astype(np.float32), y.astype(np.float32)

    return loader


def synthetic_classification(
    n_samples: int = 10000,
    n_features: int = 50,
    n_classes: int = 2,
) -> Callable[[], tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]]:
    """Factory for synthetic classification datasets."""

    def loader() -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=min(n_features // 2, n_classes * 2),
            n_redundant=n_features // 4,
            random_state=42,
        )
        return x.astype(np.float32), y.astype(np.float32)

    return loader


# =============================================================================
# Pre-defined Dataset Configurations
# =============================================================================

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
        loader=synthetic_regression(2000, 50),
    ),
    "synthetic_reg_medium": DatasetConfig(
        name="synthetic_reg_medium",
        task=Task.REGRESSION,
        loader=synthetic_regression(10000, 100),
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
        loader=synthetic_classification(2000, 50, 2),
    ),
    "synthetic_bin_medium": DatasetConfig(
        name="synthetic_bin_medium",
        task=Task.BINARY,
        loader=synthetic_classification(10000, 100, 2),
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
        loader=synthetic_classification(2000, 50, 5),
        n_classes=5,
    ),
}


def get_datasets_by_task(task: Task) -> dict[str, DatasetConfig]:
    """Get all datasets for a specific task."""
    return {k: v for k, v in DATASETS.items() if v.task == task}
