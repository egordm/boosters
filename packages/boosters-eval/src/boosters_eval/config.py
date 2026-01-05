"""Configuration models for the evaluation framework.

This module defines all Pydantic models for benchmark configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from sklearn.model_selection import train_test_split


class Task(str, Enum):
    """Machine learning task type."""

    REGRESSION = "regression"
    QUANTILE_REGRESSION = "quantile_regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"


class BoosterType(str, Enum):
    """Type of gradient boosting model."""

    GBDT = "gbdt"
    GBLINEAR = "gblinear"
    LINEAR_TREES = "linear_trees"


class GrowthStrategy(str, Enum):
    """Tree growth strategy."""

    DEPTHWISE = "depthwise"
    LEAFWISE = "leafwise"


def sklearn_split(
    data: LoadedDataset,
    seed: int,
    *,
    validation_fraction: float = 0.2,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]] | None,
    NDArray[np.floating[Any]] | None,
]:
    """Default dataset splitter using sklearn's train_test_split.

    Splits sample weights when present.
    """
    if data.sample_weight is None:
        x_train, x_valid, y_train, y_valid = train_test_split(
            data.x,
            data.y,
            test_size=validation_fraction,
            random_state=seed,
        )
        return x_train, x_valid, y_train, y_valid, None, None

    x_train, x_valid, y_train, y_valid, w_train, w_valid = train_test_split(
        data.x,
        data.y,
        data.sample_weight,
        test_size=validation_fraction,
        random_state=seed,
    )
    return x_train, x_valid, y_train, y_valid, w_train, w_valid


DEFAULT_SPLITTER = cast(
    Callable[
        ["LoadedDataset", int],
        tuple[
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]] | None,
            NDArray[np.floating[Any]] | None,
        ],
    ],
    partial(sklearn_split, validation_fraction=0.2),
)


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    task: Task
    loader: Callable[[], LoadedDataset]
    n_classes: int | None = None
    # Quantile regression: quantiles used for training + evaluation.
    quantiles: list[float] | None = None
    subsample: int | None = None
    # If False (typically time-series), random subsampling is disabled.
    allow_random_subsample: bool = True
    # Optional per-dataset metric override (e.g., "mape" for forecasting datasets)
    primary_metric: str | None = None
    supported_booster_types: list[BoosterType] | None = None
    training_overrides: TrainingOverrides | None = None
    splitter: Callable[
        [LoadedDataset, int],
        tuple[
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]] | None,
            NDArray[np.floating[Any]] | None,
        ],
    ] = DEFAULT_SPLITTER

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, v: list[float] | None) -> list[float] | None:
        """Validate and normalize dataset quantiles.

        Ensures values are finite, in (0, 1), and returns a sorted unique list.
        """
        if v is None:
            return None

        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("quantiles must be a non-empty 1D list")
        if not np.all(np.isfinite(arr)):
            raise ValueError("quantiles must be finite")
        if np.any(arr <= 0.0) or np.any(arr >= 1.0):
            raise ValueError("quantiles must be strictly within (0, 1)")

        # Sort and de-duplicate to ensure a stable output layout.
        arr = np.unique(np.sort(arr))
        return [float(x) for x in arr.tolist()]


class LoadedDataset(BaseModel):
    """Loaded dataset with optional metadata.

    This allows benchmarks to implement non-random splitting (e.g. time-series
    forecasting) while keeping the rest of the pipeline unchanged.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    x: NDArray[np.floating[Any]]
    y: NDArray[np.floating[Any]]
    sample_weight: NDArray[np.floating[Any]] | None = None
    feature_names: list[str] | None = None
    categorical_features: list[int] = []

    # Optional precomputed chronological split indices for time-series datasets.
    # If present, split is trivial: train = [:train_end], valid = [train_end:valid_end].
    train_end: int | None = None
    valid_end: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingOverrides:
    """Partial TrainingConfig override.

    Fields set to None are not applied.
    """

    n_estimators: int | None = None
    max_depth: int | None = None
    learning_rate: float | None = None
    quantiles: list[float] | None = None
    reg_lambda: float | None = None
    reg_alpha: float | None = None
    min_child_weight: float | None = None
    min_samples_leaf: int | None = None
    subsample: float | None = None
    colsample_bytree: float | None = None
    max_bins: int | None = None
    n_threads: int | None = None
    growth_strategy: GrowthStrategy | None = None
    linear_l2: float | None = None
    linear_max_features: int | None = None

    def to_update_dict(self) -> dict[str, Any]:
        """Convert override values to an update dict.

        Only fields with non-None values are included.
        """
        update: dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if value is not None:
                update[field_name] = value
        return update

    def merged(self, other: TrainingOverrides | None) -> TrainingOverrides:
        """Merge another override on top of this one."""
        if other is None:
            return self
        merged = self.to_update_dict()
        merged.update(other.to_update_dict())
        return TrainingOverrides(**merged)


class TrainingConfig(BaseModel):
    """Training hyperparameters - consistent across libraries.

    Parameter alignment notes:
    - learning_rate: 0.1 (XGBoost default is 0.3, but 0.1 is standard)
    - reg_lambda/reg_alpha: 0.0 (no regularization for fair quality comparison)
    - max_depth: 6 (same across all libraries)
    - For LightGBM, num_leaves is computed as 2^max_depth - 1 to match depth-wise
    - growth_strategy: LEAFWISE by default (LightGBM and boosters preferred mode)
    - max_bins: 256 (binning resolution for histograms)

    Linear trees parameters:
    - linear_l2: L2 regularization for linear leaf coefficients
    - linear_max_features: Maximum number of features for linear leaves
    """

    model_config = ConfigDict(frozen=True)

    # Core parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    # Quantile regression: list of quantiles in (0, 1).
    # Only used when the dataset task is QUANTILE_REGRESSION.
    quantiles: list[float] | None = None
    reg_lambda: float = 0.0  # L2 regularization - 0 for fair comparison
    reg_alpha: float = 0.0  # L1 regularization - 0 for fair comparison
    min_child_weight: float = 1.0  # Minimum hessian sum in leaf
    min_samples_leaf: int = 1  # Minimum samples in leaf (boosters/LightGBM only)
    subsample: float = 1.0  # Row subsampling (default: no subsampling)
    colsample_bytree: float = 1.0
    max_bins: int = 256  # Binning resolution for histograms
    n_threads: int = 1
    growth_strategy: GrowthStrategy = GrowthStrategy.LEAFWISE

    # Linear trees parameters (aligned with LightGBM and boosters)
    linear_l2: float = 0.01  # L2 regularization for linear leaf coefficients
    linear_max_features: int | None = None  # Max features per linear leaf (None = use path)

    @property
    def num_leaves(self) -> int:
        """Compute num_leaves from max_depth for LightGBM depth-wise equivalence."""
        return (2**self.max_depth) - 1

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        """Validate learning rate is positive."""
        if v <= 0:
            raise ValueError("learning_rate must be positive")
        return v

    @field_validator("n_estimators")
    @classmethod
    def validate_n_estimators(cls, v: int) -> int:
        """Validate n_estimators is positive."""
        if v <= 0:
            raise ValueError("n_estimators must be positive")
        return v


def _apply_training_overrides(
    base: TrainingConfig,
    overrides: TrainingOverrides | None,
) -> TrainingConfig:
    if overrides is None:
        return base
    return base.model_copy(update=overrides.to_update_dict())


def resolve_training_config(
    *,
    dataset: DatasetConfig,
    suite: SuiteConfig | None = None,
    base: TrainingConfig | None = None,
) -> TrainingConfig:
    """Resolve final TrainingConfig for a dataset run.

    Merge order is base → dataset overrides → suite overrides.
    """
    resolved = base or TrainingConfig()
    resolved = _apply_training_overrides(resolved, dataset.training_overrides)
    if suite is not None:
        resolved = _apply_training_overrides(resolved, suite.to_training_overrides())
    return resolved


class BenchmarkConfig(BaseModel):
    """Configuration for a single benchmark run."""

    model_config = ConfigDict(frozen=True)

    name: str
    dataset: DatasetConfig
    training: TrainingConfig = TrainingConfig()
    booster_type: BoosterType = BoosterType.GBDT
    libraries: list[str] = ["boosters", "xgboost", "lightgbm"]


class SuiteConfig(BaseModel):
    """Configuration for a benchmark suite."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    datasets: list[str]
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    quantiles: list[float] | None = None
    reg_lambda: float = 0.0  # L2 regularization
    reg_alpha: float = 0.0  # L1 regularization
    seeds: list[int] = [42, 1379, 2716]
    libraries: list[str] = ["boosters", "xgboost", "lightgbm"]
    booster_type: BoosterType = BoosterType.GBDT
    booster_types: list[BoosterType] | None = None  # If set, runs multiple booster types
    growth_strategy: GrowthStrategy = GrowthStrategy.LEAFWISE  # Default to leafwise

    # Optional structured overrides (useful for parameters not exposed as SuiteConfig fields).
    training_overrides: TrainingOverrides | None = None

    def to_training_config(self) -> TrainingConfig:
        """Convert suite config to training config."""
        return _apply_training_overrides(TrainingConfig(), self.to_training_overrides())

    def to_training_overrides(self) -> TrainingOverrides:
        """Convert *explicitly set* suite fields to TrainingOverrides.

        This prevents defaults from unintentionally overriding per-dataset overrides.
        """
        update: dict[str, Any] = {}
        for field_name in (
            "n_estimators",
            "max_depth",
            "learning_rate",
            "quantiles",
            "reg_lambda",
            "reg_alpha",
            "growth_strategy",
        ):
            if field_name in self.model_fields_set:
                update[field_name] = getattr(self, field_name)

        overrides = TrainingOverrides(**update)
        return overrides.merged(self.training_overrides)

    def get_booster_types(self) -> list[BoosterType]:
        """Get list of booster types to run."""
        if self.booster_types:
            return self.booster_types
        return [self.booster_type]
