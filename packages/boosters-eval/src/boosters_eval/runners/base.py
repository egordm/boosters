"""Runner base types shared by all backends."""

from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from boosters_eval.config import BenchmarkConfig, Task
from boosters_eval.results import BenchmarkResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class RunData:
    """Inputs for a single benchmark run.

    Groups train/validation arrays together with optional feature metadata.
    """

    x_train: NDArray[np.floating[Any]]
    y_train: NDArray[np.floating[Any]]
    x_valid: NDArray[np.floating[Any]]
    y_valid: NDArray[np.floating[Any]]
    categorical_features: list[int]
    feature_names: list[str] | None
    sample_weight_train: NDArray[np.floating[Any]] | None = None
    sample_weight_valid: NDArray[np.floating[Any]] | None = None

    @cached_property
    def has_nans(self) -> bool:
        """Check if any feature array contains NaN values."""
        return bool(np.isnan(self.x_train).any() or np.isnan(self.x_valid).any())


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable context for a benchmark run.

    Captures all configuration needed to train/predict without
    re-reading from nested config objects.
    """

    # Core training parameters
    n_estimators: int
    learning_rate: float
    max_depth: int
    num_leaves: int
    reg_lambda: float
    reg_alpha: float
    min_child_weight: float
    min_samples_leaf: int
    subsample: float
    colsample_bytree: float
    max_bins: int
    n_threads: int
    seed: int

    # Task info
    task: Task
    n_classes: int | None
    quantiles: np.ndarray | None

    # Linear trees
    linear_l2: float
    linear_max_features: int | None

    # Dataset metadata
    dataset_name: str
    config_name: str
    primary_metric: str | None

    # Run options
    timing_mode: bool = False
    measure_memory: bool = False

    @classmethod
    def from_config(
        cls,
        config: BenchmarkConfig,
        seed: int,
        *,
        timing_mode: bool = False,
        measure_memory: bool = False,
    ) -> RunContext:
        """Create a RunContext from a BenchmarkConfig."""
        tc = config.training

        # Resolve quantiles for quantile regression
        quantiles: np.ndarray | None = None
        if config.dataset.task == Task.QUANTILE_REGRESSION:
            q = config.dataset.quantiles or tc.quantiles or [0.1, 0.5, 0.9]
            quantiles = np.asarray(q, dtype=np.float64)

        return cls(
            n_estimators=tc.n_estimators,
            learning_rate=tc.learning_rate,
            max_depth=tc.max_depth,
            num_leaves=tc.num_leaves,
            reg_lambda=tc.reg_lambda,
            reg_alpha=tc.reg_alpha,
            min_child_weight=tc.min_child_weight,
            min_samples_leaf=tc.min_samples_leaf,
            subsample=tc.subsample,
            colsample_bytree=tc.colsample_bytree,
            max_bins=tc.max_bins,
            n_threads=tc.n_threads,
            seed=seed,
            task=config.dataset.task,
            n_classes=config.dataset.n_classes,
            quantiles=quantiles,
            linear_l2=tc.linear_l2,
            linear_max_features=tc.linear_max_features,
            dataset_name=config.dataset.name,
            config_name=config.name,
            primary_metric=config.dataset.primary_metric,
            timing_mode=timing_mode,
            measure_memory=measure_memory,
        )


class Runner(ABC):
    """Base class for benchmark runners."""

    name: str

    @classmethod
    @abstractmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if this runner supports the given configuration."""
        ...

    @classmethod
    @abstractmethod
    def run(
        cls,
        config: BenchmarkConfig,
        data: RunData,
        seed: int,
        *,
        timing_mode: bool = False,
        measure_memory: bool = False,
    ) -> BenchmarkResult:
        """Run training and return results."""
        ...


def _maybe_start_memory(*, measure_memory: bool) -> None:
    if measure_memory:
        tracemalloc.start()


def _maybe_get_peak_memory(*, measure_memory: bool) -> float | None:
    if not measure_memory:
        return None

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


def _timed_call(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start
