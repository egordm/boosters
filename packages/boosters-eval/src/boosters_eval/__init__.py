"""Boosters evaluation framework.

A simple, extensible framework for benchmarking gradient boosting libraries.

Example:
    >>> from boosters_eval import compare, QUICK_SUITE
    >>> results = compare(["california"], seeds=[42])  # doctest: +SKIP
"""

from boosters_eval.baseline import (
    Baseline,
    BaselineResult,
    MetricChange,
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
from boosters_eval.reports import (
    LibraryVersions,
    MachineInfo,
    ReportMetadata,
    generate_report,
    get_library_versions,
    get_machine_info,
    is_significant,
    render_report,
)
from boosters_eval.results import (
    BenchmarkError,
    BenchmarkResult,
    ResultCollection,
    TASK_METRICS,
    TIMING_METRICS,
)
from boosters_eval.runners import Runner, get_available_runners, get_runner
from boosters_eval.suite import (
    ABLATION_SUITES,
    FULL_SUITE,
    MINIMAL_SUITE,
    QUICK_SUITE,
    compare,
    create_ablation_suite,
    run_ablation,
    run_suite,
)

__all__ = [
    # Main API
    "compare",
    "run_suite",
    "run_ablation",
    "create_ablation_suite",
    # Suite constants
    "QUICK_SUITE",
    "FULL_SUITE",
    "MINIMAL_SUITE",
    "ABLATION_SUITES",
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
    "TASK_METRICS",
    "TIMING_METRICS",
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
    "MetricChange",
    "MetricStats",
    "RegressionReport",
    "check_baseline",
    "load_baseline",
    "record_baseline",
    # Report
    "LibraryVersions",
    "MachineInfo",
    "ReportMetadata",
    "generate_report",
    "get_library_versions",
    "get_machine_info",
    "is_significant",
    "render_report",
]

__version__ = "0.1.0"
