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
    """Training hyperparameters - consistent across libraries."""

    model_config = ConfigDict(frozen=True)

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    n_threads: int = 1
    growth_strategy: GrowthStrategy = GrowthStrategy.DEPTHWISE

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
    seeds: list[int] = [42, 1379, 2716]
    libraries: list[str] = ["boosters", "xgboost", "lightgbm"]
    booster_type: BoosterType = BoosterType.GBDT
