"""Boosters evaluation framework.

A simple, extensible framework for benchmarking gradient boosting libraries.

Example:
    >>> from boosters_eval import compare, QUICK_SUITE
    >>> results = compare(["california"], seeds=[42])  # doctest: +SKIP
"""

from boosters_eval.baseline import (
    Baseline,
    BaselineResult,
    MetricStats,
    RegressionReport,
    check_baseline,
    load_baseline,
    record_baseline,
)
from boosters_eval.config import (
    BenchmarkConfig,
    BoosterType,
    DatasetConfig,
    GrowthStrategy,
    Task,
    TrainingConfig,
)
from boosters_eval.datasets import DATASETS, get_datasets_by_task
from boosters_eval.metrics import compute_metrics, is_lower_better, primary_metric
from boosters_eval.results import BenchmarkError, BenchmarkResult, ResultCollection
from boosters_eval.runners import Runner, get_available_runners, get_runner
from boosters_eval.suite import FULL_SUITE, MINIMAL_SUITE, QUICK_SUITE, compare, run_suite

__all__ = [
    # Main API
    "compare",
    "run_suite",
    # Suite constants
    "QUICK_SUITE",
    "FULL_SUITE",
    "MINIMAL_SUITE",
    # Configuration
    "BenchmarkConfig",
    "BoosterType",
    "DatasetConfig",
    "GrowthStrategy",
    "Task",
    "TrainingConfig",
    # Datasets
    "DATASETS",
    "get_datasets_by_task",
    # Results
    "BenchmarkError",
    "BenchmarkResult",
    "ResultCollection",
    # Metrics
    "compute_metrics",
    "is_lower_better",
    "primary_metric",
    # Runners
    "Runner",
    "get_available_runners",
    "get_runner",
    # Baseline
    "Baseline",
    "BaselineResult",
    "MetricStats",
    "RegressionReport",
    "check_baseline",
    "load_baseline",
    "record_baseline",
]

__version__ = "0.1.0"
