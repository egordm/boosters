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
    TASK_METRICS,
    TIMING_METRICS,
    BenchmarkError,
    BenchmarkResult,
    ResultCollection,
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

__all__ = (
    "ABLATION_SUITES",
    "DATASETS",
    "FULL_SUITE",
    "MINIMAL_SUITE",
    "QUICK_SUITE",
    "TASK_METRICS",
    "TIMING_METRICS",
    "Baseline",
    "BaselineResult",
    "BenchmarkConfig",
    "BenchmarkError",
    "BenchmarkResult",
    "BoosterType",
    "DatasetConfig",
    "GrowthStrategy",
    "LibraryVersions",
    "MachineInfo",
    "MetricChange",
    "MetricStats",
    "RegressionReport",
    "ReportMetadata",
    "ResultCollection",
    "Runner",
    "Task",
    "TrainingConfig",
    "check_baseline",
    "compare",
    "compute_metrics",
    "create_ablation_suite",
    "generate_report",
    "get_available_runners",
    "get_datasets_by_task",
    "get_library_versions",
    "get_machine_info",
    "get_runner",
    "is_lower_better",
    "is_significant",
    "load_baseline",
    "primary_metric",
    "record_baseline",
    "render_report",
    "run_ablation",
    "run_suite",
)


__version__ = "0.1.0"
