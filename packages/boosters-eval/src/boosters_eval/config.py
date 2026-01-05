"""Configuration models for the evaluation framework.

This module defines all Pydantic models for benchmark configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from boosters_eval.utils import train_valid_split


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


class DatasetConfig(BaseModel):
    """Configuration for a dataset."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    task: Task
    loader: Callable[[], LoadedDataset]
    n_classes: int | None = None
    # Quantile regression: quantiles used for training + evaluation.
    quantiles: list[float] | None = None
    # Optional per-dataset metric override (e.g., "mape" for forecasting datasets)
    primary_metric: str | None = None
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig())
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
    ] = partial(train_valid_split, validation_fraction=0.2)

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


# Base training defaults per model family (booster type).
#
# This is the first layer in config resolution:
#   (1) base per model → (2) dataset → (3) suite → (4) CLI overrides
#
# Note: The current defaults are intentionally identical across booster types.
# This keeps existing baselines stable while still centralizing the place where
# model-type-specific defaults should live (e.g. GBLinear vs GBDT).
DEFAULT_BASE_TRAINING_BY_BOOSTER_TYPE: dict[BoosterType, TrainingConfig] = {
    BoosterType.GBDT: TrainingConfig(),
    BoosterType.GBLINEAR: TrainingConfig(),
    BoosterType.LINEAR_TREES: TrainingConfig(),
}


def resolve_training_config(
    *,
    booster_type: BoosterType,
    dataset: DatasetConfig,
    suite: SuiteConfig | None = None,
    cli: TrainingConfig | None = None,
    base_by_booster_type: dict[BoosterType, TrainingConfig] | None = None,
) -> TrainingConfig:
    """Resolve final TrainingConfig for a dataset run.

    Merge order is base (per model) → dataset overrides → suite overrides → CLI overrides.
    """
    base_map = DEFAULT_BASE_TRAINING_BY_BOOSTER_TYPE
    if base_by_booster_type is not None:
        base_map = {**base_map, **base_by_booster_type}

    resolved = base_map[booster_type]
    resolved = resolved.model_copy(update=dataset.training.model_dump(exclude_unset=True))
    if suite is not None:
        resolved = resolved.model_copy(update=suite.training.model_dump(exclude_unset=True))
    if cli is not None:
        resolved = resolved.model_copy(update=cli.model_dump(exclude_unset=True))
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
    training: TrainingConfig = Field(default_factory=lambda: TrainingConfig())
    seeds: list[int] = [42, 1379, 2716]
    libraries: list[str] = ["boosters", "xgboost", "lightgbm"]
    booster_type: BoosterType = BoosterType.GBDT
    booster_types: list[BoosterType] | None = None  # If set, runs multiple booster types

    def get_booster_types(self) -> list[BoosterType]:
        """Get list of booster types to run."""
        if self.booster_types:
            return self.booster_types
        return [self.booster_type]
