"""Runner base types shared by all backends."""

from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
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


def resolve_quantiles(config: BenchmarkConfig) -> np.ndarray | None:
    """Resolve quantiles for a run.

    Composition-first helper: reads from `BenchmarkConfig` instead of
    duplicating a flattened context object.
    """
    if config.dataset.task != Task.QUANTILE_REGRESSION:
        return None

    q = config.dataset.quantiles or config.training.quantiles or [0.1, 0.5, 0.9]
    return np.asarray(q, dtype=np.float64)


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
