"""Boosters evaluation framework.

A simple, extensible framework for benchmarking gradient boosting libraries.

Example:
    >>> from boosters_eval import BenchmarkConfig, BenchmarkSuite, DATASETS
    >>> config = BenchmarkConfig(
    ...     name="test",
    ...     dataset=DATASETS["california"],
    ...     libraries=["xgboost", "lightgbm"],
    ... )
    >>> suite = BenchmarkSuite([config], seeds=[42])
    >>> _ = suite.run(verbose=False)  # doctest: +SKIP
"""

from boosters_eval.datasets import (
    DATASETS,
    BenchmarkConfig,
    BoosterType,
    DatasetConfig,
    Task,
    TrainingConfig,
    get_datasets_by_task,
)
from boosters_eval.metrics import compute_metrics, is_lower_better, primary_metric
from boosters_eval.results import BenchmarkResult
from boosters_eval.runners import Runner, get_available_runners, get_runner, register_runner
from boosters_eval.suite import BenchmarkSuite, run_all_combinations

__all__ = [
    "DATASETS",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BoosterType",
    "DatasetConfig",
    "Runner",
    "Task",
    "TrainingConfig",
    "compute_metrics",
    "get_available_runners",
    "get_datasets_by_task",
    "get_runner",
    "is_lower_better",
    "primary_metric",
    "register_runner",
    "run_all_combinations",
]

__version__ = "0.1.0"
