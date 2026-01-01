"""Configuration models for the evaluation framework.

This module defines all Pydantic models for benchmark configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator


class Task(str, Enum):
    """Machine learning task type."""

    REGRESSION = "regression"
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
    loader: Callable[[], tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]]
    n_classes: int | None = None
    subsample: int | None = None
    # Optional per-dataset metric override (e.g., "mape" for forecasting datasets)
    primary_metric: str | None = None


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
    reg_lambda: float = 0.0  # L2 regularization - 0 for fair comparison
    reg_alpha: float = 0.0  # L1 regularization - 0 for fair comparison
    min_child_weight: float = 1.0  # Minimum hessian sum in leaf
    min_samples_leaf: int = 1  # Minimum samples in leaf (boosters/LightGBM only)
    subsample: float = 1.0
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

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    datasets: list[str]
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    reg_lambda: float = 0.0  # L2 regularization
    reg_alpha: float = 0.0  # L1 regularization
    seeds: list[int] = [42, 1379, 2716]
    libraries: list[str] = ["boosters", "xgboost", "lightgbm"]
    booster_type: BoosterType = BoosterType.GBDT
    booster_types: list[BoosterType] | None = None  # If set, runs multiple booster types
    growth_strategy: GrowthStrategy = GrowthStrategy.LEAFWISE  # Default to leafwise

    def to_training_config(self) -> TrainingConfig:
        """Convert suite config to training config."""
        return TrainingConfig(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            growth_strategy=self.growth_strategy,
        )

    def get_booster_types(self) -> list[BoosterType]:
        """Get list of booster types to run."""
        if self.booster_types:
            return self.booster_types
        return [self.booster_type]
