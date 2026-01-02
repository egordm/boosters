"""Baseline schema and regression detection."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from boosters_eval.metrics import is_lower_better
from boosters_eval.results import ResultCollection

# Current schema version
SCHEMA_VERSION = 1


class MetricStats(BaseModel):
    """Statistics for a metric across seeds."""

    model_config = ConfigDict(frozen=True)

    mean: float
    std: float
    n: int


class BaselineResult(BaseModel):
    """Result entry in a baseline."""

    model_config = ConfigDict(frozen=True)

    config_name: str
    library: str
    task: str
    booster_type: str
    dataset_name: str
    metrics: dict[str, MetricStats]


class Baseline(BaseModel):
    """Versioned baseline with results."""

    model_config = ConfigDict(frozen=True)

    schema_version: int
    created_at: str
    git_sha: str | None = None
    boosters_version: str | None = None
    results: list[BaselineResult]

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: int) -> int:
        """Validate schema version."""
        if v > SCHEMA_VERSION:
            raise ValueError(f"Baseline schema version {v} is newer than supported version {SCHEMA_VERSION}")
        return v


class MetricChange(TypedDict):
    """A metric that changed between baseline and current."""

    config: str
    metric: str
    baseline: float
    current: float


class RegressionReport(BaseModel):
    """Report of regressions found."""

    model_config = ConfigDict(frozen=True)

    has_regressions: bool
    regressions: list[MetricChange]
    improvements: list[MetricChange]
    tolerance: float


def get_git_sha() -> str | None:
    """Get current git SHA if in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_boosters_version() -> str | None:
    """Get boosters library version."""
    try:
        import boosters
    except (ImportError, AttributeError):
        return None
    else:
        return boosters.__version__


def record_baseline(
    results: ResultCollection,
    output_path: Path | None = None,
) -> Baseline:
    """Record current results as a baseline.

    Args:
        results: Results to record.
        output_path: Optional path to save baseline JSON.

    Returns:
        Baseline object.
    """
    # Aggregate results by (config_name, library, dataset_name, task, booster_type)
    aggregated: dict[tuple[str, str, str, str, str], list[dict[str, float]]] = {}

    for r in results.results:
        key = (r.config_name, r.library, r.dataset_name, r.task, r.booster_type)
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(r.metrics)

    # Compute stats for each group
    baseline_results: list[BaselineResult] = []

    for key, metrics_list in aggregated.items():
        config_name, library, dataset_name, task, booster_type = key

        # Compute mean/std for each metric
        metric_stats: dict[str, MetricStats] = {}
        metric_keys = metrics_list[0].keys()

        for metric_key in metric_keys:
            values = [m[metric_key] for m in metrics_list]
            mean = float(np.mean(values))
            std = float(np.std(values))
            n = len(values)
            metric_stats[metric_key] = MetricStats(mean=mean, std=std, n=n)

        baseline_results.append(
            BaselineResult(
                config_name=config_name,
                library=library,
                task=task,
                booster_type=booster_type,
                dataset_name=dataset_name,
                metrics=metric_stats,
            )
        )

    baseline = Baseline(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(UTC).isoformat(),
        git_sha=get_git_sha(),
        boosters_version=get_boosters_version(),
        results=baseline_results,
    )

    if output_path:
        output_path.write_text(baseline.model_dump_json(indent=2))

    return baseline


def load_baseline(path: Path) -> Baseline:
    """Load baseline from JSON file.

    Args:
        path: Path to baseline JSON.

    Returns:
        Baseline object.

    Raises:
        ValidationError: If baseline is invalid.
    """
    data = json.loads(path.read_text())
    return Baseline(**data)


def is_regression(
    current: float,
    baseline: float,
    metric: str,
    tolerance: float = 0.02,
) -> bool:
    """Check if current value is a regression from baseline.

    Args:
        current: Current metric value.
        baseline: Baseline metric value.
        metric: Metric name (to determine direction).
        tolerance: Relative tolerance (0.02 = 2%).

    Returns:
        True if this is a regression beyond tolerance.
    """
    # Calculate the absolute difference allowed
    # Use absolute value of baseline for tolerance calculation to handle negatives correctly
    abs_tolerance = abs(baseline) * tolerance

    # Handle edge case where baseline is very close to zero
    if abs_tolerance < 1e-10:
        abs_tolerance = tolerance

    if is_lower_better(metric):
        # For lower-is-better: regression if current is significantly higher
        return current > baseline + abs_tolerance
    # For higher-is-better: regression if current is significantly lower
    return current < baseline - abs_tolerance


def check_baseline(
    results: ResultCollection,
    baseline: Baseline,
    tolerance: float = 0.02,
) -> RegressionReport:
    """Compare current results against baseline.

    Args:
        results: Current results.
        baseline: Baseline to compare against.
        tolerance: Relative tolerance for regression detection.

    Returns:
        RegressionReport with detected regressions.
    """
    # Build lookup for baseline results
    baseline_lookup: dict[tuple[str, str], BaselineResult] = {}
    for br in baseline.results:
        key = (br.config_name, br.library)
        baseline_lookup[key] = br

    # Aggregate current results
    current_agg: dict[tuple[str, str], list[dict[str, float]]] = {}
    for r in results.results:
        key = (r.config_name, r.library)
        if key not in current_agg:
            current_agg[key] = []
        current_agg[key].append(r.metrics)

    regressions: list[MetricChange] = []
    improvements: list[MetricChange] = []

    for key, metrics_list in current_agg.items():
        config_name, _library = key
        baseline_result = baseline_lookup.get(key)

        if baseline_result is None:
            # New config, not in baseline - skip
            continue

        for metric_key in metrics_list[0].keys():
            values = [m[metric_key] for m in metrics_list]
            current_mean = float(np.mean(values))

            baseline_stats = baseline_result.metrics.get(metric_key)
            if baseline_stats is None:
                continue

            baseline_mean = baseline_stats.mean

            if is_regression(current_mean, baseline_mean, metric_key, tolerance):
                regressions.append(
                    MetricChange(
                        config=config_name,
                        metric=metric_key,
                        baseline=baseline_mean,
                        current=current_mean,
                    )
                )
            elif is_regression(baseline_mean, current_mean, metric_key, tolerance):
                # Improvement (baseline regressed compared to current)
                improvements.append(
                    MetricChange(
                        config=config_name,
                        metric=metric_key,
                        baseline=baseline_mean,
                        current=current_mean,
                    )
                )

    return RegressionReport(
        has_regressions=len(regressions) > 0,
        regressions=regressions,
        improvements=improvements,
        tolerance=tolerance,
    )
