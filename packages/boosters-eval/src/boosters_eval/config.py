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


class TrainingConfig(BaseModel):
    """Training hyperparameters - consistent across libraries.

    Parameter alignment notes:
    - learning_rate: 0.1 (XGBoost default is 0.3, but 0.1 is standard)
    - reg_lambda/reg_alpha: 0.0 (no regularization for fair quality comparison)
    - max_depth: 6 (same across all libraries)
    - For LightGBM, num_leaves is computed as 2^max_depth - 1 to match depth-wise
    """

    model_config = ConfigDict(frozen=True)

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    reg_lambda: float = 0.0  # L2 regularization - 0 for fair comparison
    reg_alpha: float = 0.0  # L1 regularization - 0 for fair comparison
    min_child_weight: float = 1.0  # Minimum hessian sum in leaf
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_threads: int = 1
    growth_strategy: GrowthStrategy = GrowthStrategy.DEPTHWISE

    @property
    def num_leaves(self) -> int:
        """Compute num_leaves from max_depth for LightGBM depth-wise equivalence."""
        return (2 ** self.max_depth) - 1

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
    seeds: list[int] = [42, 1379, 2716]
    libraries: list[str] = ["boosters", "xgboost", "lightgbm"]
    booster_type: BoosterType = BoosterType.GBDT
    growth_strategy: GrowthStrategy = GrowthStrategy.DEPTHWISE

    def to_training_config(self) -> TrainingConfig:
        """Convert suite config to training config."""
        return TrainingConfig(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            growth_strategy=self.growth_strategy,
        )
