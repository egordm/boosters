"""Benchmark result models and collection."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from scipy import stats

from boosters_eval.config import Task
from boosters_eval.metrics import LOWER_BETTER_METRICS, primary_metric

# Metrics relevant to each task type
TASK_METRICS: dict[Task, list[str]] = {
    Task.REGRESSION: ["rmse", "mae", "r2"],
    Task.BINARY: ["logloss", "accuracy"],
    Task.MULTICLASS: ["mlogloss", "accuracy"],
}

# Timing metrics included in all reports
TIMING_METRICS: list[str] = ["train_time_s", "predict_time_s"]

# Memory metrics (included when measured)
MEMORY_METRICS: list[str] = ["peak_memory_mb"]


class BenchmarkResult(BaseModel):
    """Results from a single benchmark run."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    config_name: str
    library: str
    seed: int
    task: str
    booster_type: str
    dataset_name: str
    metrics: dict[str, float]
    train_time_s: float | None = None
    predict_time_s: float | None = None
    peak_memory_mb: float | None = None
    # Per-dataset primary metric override (e.g., "mape" for forecasting)
    dataset_primary_metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for DataFrame."""
        result: dict[str, Any] = {
            "config": self.config_name,
            "library": self.library,
            "dataset": self.dataset_name,
            "booster": self.booster_type,
            "seed": self.seed,
            "task": self.task,
        }
        result.update(self.metrics)
        if self.train_time_s is not None:
            result["train_time_s"] = self.train_time_s
        if self.predict_time_s is not None:
            result["predict_time_s"] = self.predict_time_s
        if self.peak_memory_mb is not None:
            result["peak_memory_mb"] = self.peak_memory_mb
        if self.dataset_primary_metric is not None:
            result["dataset_primary_metric"] = self.dataset_primary_metric
        return result


class BenchmarkError(BaseModel):
    """Error from a failed benchmark run."""

    model_config = ConfigDict(frozen=True)

    config_name: str
    library: str
    seed: int
    error_type: str
    error_message: str
    dataset_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary."""
        return {
            "config": self.config_name,
            "library": self.library,
            "dataset": self.dataset_name,
            "seed": self.seed,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def derive_seed(base_seed: int, config_name: str, library: str) -> int:
    """Derive a deterministic seed from base seed, config, and library.

    Formula: hash((base_seed, config_name, library)) % 2^32
    """
    key = f"{base_seed}:{config_name}:{library}"
    hash_bytes = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "little")


class ResultCollection:
    """Collection of benchmark results with aggregation and export."""

    def __init__(
        self,
        results: list[BenchmarkResult] | None = None,
        errors: list[BenchmarkError] | None = None,
    ) -> None:
        """Initialize result collection."""
        self.results = results or []
        self.errors = errors or []

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a result to the collection."""
        self.results.append(result)

    def add_error(self, error: BenchmarkError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def errors_dataframe(self) -> pd.DataFrame:
        """Convert errors to pandas DataFrame."""
        if not self.errors:
            return pd.DataFrame()
        return pd.DataFrame([e.to_dict() for e in self.errors])

    def filter(
        self,
        *,
        library: str | list[str] | None = None,
        dataset: str | list[str] | None = None,
        task: str | list[str] | None = None,
    ) -> ResultCollection:
        """Filter results by criteria."""
        filtered = self.results

        if library is not None:
            libs = [library] if isinstance(library, str) else library
            filtered = [r for r in filtered if r.library in libs]

        if dataset is not None:
            datasets = [dataset] if isinstance(dataset, str) else dataset
            filtered = [r for r in filtered if r.dataset_name in datasets]

        if task is not None:
            tasks = [task] if isinstance(task, str) else task
            filtered = [r for r in filtered if r.task in tasks]

        return ResultCollection(results=filtered, errors=self.errors)

    def summary(self, group_by: list[str] | None = None) -> pd.DataFrame:
        """Aggregate results with mean ± std.

        Args:
            group_by: Columns to group by. Defaults to ["dataset", "booster", "library"].

        Returns:
            DataFrame with aggregated statistics.
        """
        df = self.to_dataframe()
        if df.empty:
            return df

        group_by = group_by or ["dataset", "booster", "library"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [c for c in numeric_cols if c not in [*group_by, "seed"]]

        agg_funcs = {col: ["mean", "std", "count"] for col in metric_cols}
        summary = df.groupby(group_by, as_index=False).agg(agg_funcs)

        # Flatten column names
        summary.columns = [  # pyright: ignore[reportAttributeAccessIssue]
            f"{col}_{agg}" if agg else col for col, agg in summary.columns
        ]

        return pd.DataFrame(summary)

    def summary_by_task(self) -> dict[Task, pd.DataFrame]:
        """Get separate summary tables for each task type.

        Returns task-specific tables with only relevant metrics.
        """
        result = {}
        df = self.to_dataframe()
        if df.empty:
            return result

        for task in Task:
            task_df = df[df["task"] == task.value]
            if task_df.empty:
                continue

            # Get relevant metrics for this task (including memory if measured)
            metrics = TASK_METRICS[task] + TIMING_METRICS + MEMORY_METRICS
            group_cols = ["dataset", "library"]

            # Build aggregation for relevant metrics only
            numeric_cols = task_df.select_dtypes(include=[np.number]).columns
            relevant_cols = [c for c in numeric_cols if c in metrics]

            if not relevant_cols:
                continue

            agg_funcs = {col: ["mean", "std"] for col in relevant_cols}
            summary = task_df.groupby(group_cols, as_index=False).agg(agg_funcs)

            # Flatten column names
            summary.columns = [  # pyright: ignore[reportAttributeAccessIssue]
                f"{col}_{agg}" if agg else col for col, agg in summary.columns
            ]

            result[task] = pd.DataFrame(summary)

        return result

    def summary_by_dataset(self) -> dict[str, pd.DataFrame]:
        """Get separate summary tables for each dataset.

        Returns per-dataset tables with relevant metrics based on dataset's
        primary_metric override or task default.
        """
        result: dict[str, pd.DataFrame] = {}
        df = self.to_dataframe()
        if df.empty:
            return result

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            if dataset_df.empty:
                continue

            # Get task type for this dataset
            task_value = dataset_df["task"].iloc[0]  # pyright: ignore[reportAttributeAccessIssue]
            task = Task(task_value)

            # Get relevant metrics for this task
            metrics = TASK_METRICS[task] + TIMING_METRICS + MEMORY_METRICS
            group_cols = ["dataset", "booster", "library"]

            # Build aggregation for relevant metrics only
            numeric_cols = dataset_df.select_dtypes(include=[np.number]).columns
            relevant_cols = [c for c in numeric_cols if c in metrics]

            if not relevant_cols:
                continue

            agg_funcs = {col: ["mean", "std"] for col in relevant_cols}
            summary = dataset_df.groupby(group_cols, as_index=False).agg(agg_funcs)

            # Flatten column names
            summary.columns = [  # pyright: ignore[reportAttributeAccessIssue]
                f"{col}_{agg}" if agg else col for col, agg in summary.columns
            ]

            result[str(dataset)] = pd.DataFrame(summary)

        return result

    def get_raw_values_by_library(
        self,
        task: Task,
        dataset: str,
        metric: str,
        booster_type: str | None = None,
    ) -> dict[str, list[float]]:
        """Get raw metric values per library for significance testing.

        Args:
            task: Task type to filter by.
            dataset: Dataset name to filter by.
            metric: Metric name to extract values for.
            booster_type: Optional booster type to filter by. If None, includes all.

        Returns a dict mapping library name to list of values across seeds.
        """
        result: dict[str, list[float]] = {}
        for r in self.results:
            if r.task != task.value or r.dataset_name != dataset:
                continue
            # Filter by booster type if specified
            if booster_type is not None and r.booster_type != booster_type:
                continue
            # Check if metric is in the result
            if metric in r.metrics:
                val = r.metrics[metric]
            elif metric == "train_time_s" and r.train_time_s is not None:
                val = r.train_time_s
            elif metric == "predict_time_s" and r.predict_time_s is not None:
                val = r.predict_time_s
            elif metric == "peak_memory_mb" and r.peak_memory_mb is not None:
                val = r.peak_memory_mb
            else:
                continue
            if r.library not in result:
                result[r.library] = []
            result[r.library].append(val)
        return result

    def get_primary_metric_for_dataset(self, dataset: str) -> str:
        """Get primary metric for a dataset (override or task default).

        Checks if any result for this dataset has a dataset_primary_metric set,
        otherwise falls back to task-based primary metric.
        """
        for r in self.results:
            if r.dataset_name == dataset:
                if r.dataset_primary_metric:
                    return r.dataset_primary_metric
                # Fall back to task default
                return primary_metric(Task(r.task))
        # Default
        return "rmse"

    def is_significantly_better(
        self,
        values_best: list[float],
        values_second: list[float],
        alpha: float = 0.05,
    ) -> bool:
        """Test if best library is significantly better than second-best using Welch's t-test."""
        if len(values_best) < 2 or len(values_second) < 2:
            # Not enough data for statistical test
            return False
        _, p_value = stats.ttest_ind(values_best, values_second, equal_var=False)
        p_value_f = float(p_value)  # pyright: ignore[reportArgumentType]
        return p_value_f < alpha

    def format_summary_table(
        self,
        task: Task,
        *,
        highlight_best: bool = True,
        require_significance: bool = True,
        alpha: float = 0.05,
        precision: int = 4,
    ) -> str:
        """Format a summary table for a specific task with significance-aware highlighting.

        Args:
            task: Task type to format.
            highlight_best: Bold the best value in each metric.
            require_significance: Only bold if statistically significant (Welch's t-test).
            alpha: Significance level for t-test (default 0.05).
            precision: Decimal precision for values.

        Returns:
            Markdown formatted string (using pandas to_markdown).
        """
        summaries = self.summary_by_task()
        if task not in summaries:
            return f"No results for {task.value} task."

        df = summaries[task]
        metrics = TASK_METRICS[task] + TIMING_METRICS + MEMORY_METRICS

        # Determine which metrics are actually present
        present_metrics = [m for m in metrics if f"{m}_mean" in df.columns]

        if not present_metrics:
            return f"No metrics for {task.value} task."

        # Build output DataFrame with formatted values
        output_rows: list[dict[str, str]] = []

        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]

            # For each metric, find best and determine if significant
            significant_winners: dict[str, str | None] = {}
            for metric in present_metrics:
                mean_col = f"{metric}_mean"
                valid = dataset_df.dropna(subset=[mean_col])  # pyright: ignore[reportCallIssue]
                if len(valid) < 2:
                    # Can't compare if fewer than 2 libraries
                    significant_winners[metric] = None
                    continue

                lower_better = metric in LOWER_BETTER_METRICS or metric.endswith("_time_s")
                if lower_better:
                    sorted_df = valid.sort_values(mean_col, ascending=True)
                else:
                    sorted_df = valid.sort_values(mean_col, ascending=False)

                best_lib = str(sorted_df.iloc[0]["library"])
                second_lib = str(sorted_df.iloc[1]["library"])

                if not highlight_best:
                    significant_winners[metric] = None
                elif not require_significance:
                    # Highlight best without significance check
                    significant_winners[metric] = best_lib
                else:
                    # Check statistical significance
                    raw_values = self.get_raw_values_by_library(task, str(dataset), metric)
                    best_vals = raw_values.get(best_lib, [])
                    second_vals = raw_values.get(second_lib, [])

                    if self.is_significantly_better(best_vals, second_vals, alpha):
                        significant_winners[metric] = best_lib
                    else:
                        # Not significant - don't highlight (tie)
                        significant_winners[metric] = None

            # Format rows for this dataset
            for _, row in dataset_df.iterrows():
                lib = str(row["library"])
                row_dict: dict[str, str] = {"Dataset": str(dataset), "Library": lib}

                for metric in present_metrics:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"

                    mean_val = row[mean_col]
                    std_val = row.get(std_col, np.nan)

                    if bool(pd.isna(mean_val)):
                        row_dict[metric] = "-"
                    else:
                        # Format value with optional std
                        if std_val is None or bool(pd.isna(std_val)):
                            val_str = f"{mean_val:.{precision}f}"
                        else:
                            std_val_f = float(std_val)
                            if std_val_f == 0.0:
                                val_str = f"{mean_val:.{precision}f}"
                            else:
                                val_str = f"{mean_val:.{precision}f}±{std_val_f:.{precision}f}"

                        # Bold only if this library is the significant winner
                        if significant_winners.get(metric) == lib:
                            val_str = f"**{val_str}**"

                        row_dict[metric] = val_str

                output_rows.append(row_dict)

        # Create DataFrame and use to_markdown
        output_df = pd.DataFrame(output_rows)
        # Reorder columns
        col_order = ["Dataset", "Library", *present_metrics]
        output_df = output_df[[c for c in col_order if c in output_df.columns]]

        return str(output_df.to_markdown(index=False))

    def format_dataset_table(
        self,
        dataset: str,
        *,
        highlight_best: bool = True,
        require_significance: bool = True,
        alpha: float = 0.05,
        precision: int = 4,
    ) -> str:
        """Format a summary table for a specific dataset with significance-aware highlighting.

        This is the per-dataset version that respects dataset-specific primary metrics.

        Args:
            dataset: Dataset name to format.
            highlight_best: Bold the best value in each metric.
            require_significance: Only bold if statistically significant (Welch's t-test).
            alpha: Significance level for t-test (default 0.05).
            precision: Decimal precision for values.

        Returns:
            Markdown formatted string (using pandas to_markdown).
        """
        summaries = self.summary_by_dataset()
        if dataset not in summaries:
            return f"No results for {dataset} dataset."

        df = summaries[dataset]

        # Get task type for this dataset to determine relevant metrics
        task = None
        for r in self.results:
            if r.dataset_name == dataset:
                task = Task(r.task)
                break
        if task is None:
            return f"No results for {dataset} dataset."

        metrics = TASK_METRICS[task] + TIMING_METRICS + MEMORY_METRICS

        # Determine which metrics are actually present
        present_metrics = [m for m in metrics if f"{m}_mean" in df.columns]

        if not present_metrics:
            return f"No metrics for {dataset} dataset."

        # Build output DataFrame with formatted values
        output_rows: list[dict[str, str]] = []

        for booster in df["booster"].unique():
            booster_df = df[df["booster"] == booster]

            # For each metric, find best and determine if significant
            significant_winners: dict[str, str | None] = {}
            for metric in present_metrics:
                mean_col = f"{metric}_mean"
                valid = booster_df.dropna(subset=[mean_col])  # pyright: ignore[reportCallIssue]
                if len(valid) < 2:
                    # Can't compare if fewer than 2 libraries
                    significant_winners[metric] = None
                    continue

                lower_better = metric in LOWER_BETTER_METRICS or metric.endswith("_time_s")
                if lower_better:
                    sorted_df = valid.sort_values(mean_col, ascending=True)
                else:
                    sorted_df = valid.sort_values(mean_col, ascending=False)

                best_lib = str(sorted_df.iloc[0]["library"])
                second_lib = str(sorted_df.iloc[1]["library"])

                if not highlight_best:
                    significant_winners[metric] = None
                elif not require_significance:
                    # Highlight best without significance check
                    significant_winners[metric] = best_lib
                else:
                    # Check statistical significance (filter by booster type)
                    raw_values = self.get_raw_values_by_library(task, dataset, metric, str(booster))
                    best_vals = raw_values.get(best_lib, [])
                    second_vals = raw_values.get(second_lib, [])

                    if self.is_significantly_better(best_vals, second_vals, alpha):
                        significant_winners[metric] = best_lib
                    else:
                        # Not significant - don't highlight (tie)
                        significant_winners[metric] = None

            # Format rows for this booster
            for _, row in booster_df.iterrows():
                lib = str(row["library"])
                row_dict: dict[str, str] = {"Booster": str(booster), "Library": lib}

                for metric in present_metrics:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"

                    mean_val = row[mean_col]
                    std_val = row.get(std_col, np.nan)

                    if bool(pd.isna(mean_val)):
                        row_dict[metric] = "-"
                    else:
                        # Format value with optional std
                        if std_val is None or bool(pd.isna(std_val)):
                            val_str = f"{mean_val:.{precision}f}"
                        else:
                            std_val_f = float(std_val)
                            if std_val_f == 0.0:
                                val_str = f"{mean_val:.{precision}f}"
                            else:
                                val_str = f"{mean_val:.{precision}f}±{std_val_f:.{precision}f}"

                        # Bold only if this library is the significant winner
                        if significant_winners.get(metric) == lib:
                            val_str = f"**{val_str}**"

                        row_dict[metric] = val_str

                output_rows.append(row_dict)

        # Create DataFrame and use to_markdown
        output_df = pd.DataFrame(output_rows)
        # Reorder columns
        col_order = ["Booster", "Library", *present_metrics]
        output_df = output_df[[c for c in col_order if c in output_df.columns]]

        return str(output_df.to_markdown(index=False))

    def to_markdown(
        self,
        *,
        precision: int = 4,
        highlight_best: bool = True,
        require_significance: bool = True,
        alpha: float = 0.05,
        group_by_dataset: bool = True,
    ) -> str:
        """Generate markdown tables grouped by dataset or task type.

        Args:
            precision: Decimal precision for values.
            highlight_best: Bold the best value in each metric.
            require_significance: Only highlight if statistically significant.
            alpha: Significance level for Welch's t-test.
            group_by_dataset: If True, group by dataset. Otherwise by task.

        Returns:
            Markdown formatted string with sections per dataset or task.
        """
        if group_by_dataset:
            return self._to_markdown_by_dataset(
                precision=precision,
                highlight_best=highlight_best,
                require_significance=require_significance,
                alpha=alpha,
            )
        return self._to_markdown_by_task(
            precision=precision,
            highlight_best=highlight_best,
            require_significance=require_significance,
            alpha=alpha,
        )

    def _to_markdown_by_dataset(
        self,
        *,
        precision: int = 4,
        highlight_best: bool = True,
        require_significance: bool = True,
        alpha: float = 0.05,
    ) -> str:
        """Generate markdown tables grouped by dataset."""
        summaries = self.summary_by_dataset()
        if not summaries:
            return "No results to display."

        sections = []

        # Get task info for each dataset
        dataset_tasks: dict[str, Task] = {}
        for r in self.results:
            if r.dataset_name not in dataset_tasks:
                dataset_tasks[r.dataset_name] = Task(r.task)

        for dataset in sorted(summaries.keys()):
            task = dataset_tasks.get(dataset, Task.REGRESSION)
            primary = self.get_primary_metric_for_dataset(dataset)
            sections.append(f"### {dataset} ({task.value}, primary: {primary})")
            sections.append("")
            sections.append(
                self.format_dataset_table(
                    dataset,
                    highlight_best=highlight_best,
                    require_significance=require_significance,
                    alpha=alpha,
                    precision=precision,
                )
            )
            sections.append("")

        return "\n".join(sections)

    def _to_markdown_by_task(
        self,
        *,
        precision: int = 4,
        highlight_best: bool = True,
        require_significance: bool = True,
        alpha: float = 0.05,
    ) -> str:
        """Generate markdown tables grouped by task type."""
        summaries = self.summary_by_task()
        if not summaries:
            return "No results to display."

        sections = []

        # Task display order and names
        task_names = {
            Task.REGRESSION: "Regression",
            Task.BINARY: "Binary Classification",
            Task.MULTICLASS: "Multiclass Classification",
        }

        for task in [Task.REGRESSION, Task.BINARY, Task.MULTICLASS]:
            if task not in summaries:
                continue

            sections.append(f"### {task_names[task]}")
            sections.append("")
            sections.append(
                self.format_summary_table(
                    task,
                    highlight_best=highlight_best,
                    require_significance=require_significance,
                    alpha=alpha,
                    precision=precision,
                )
            )
            sections.append("")

        return "\n".join(sections)

    def to_json(self) -> str:
        """Export results to JSON string."""
        data = {
            "results": [r.model_dump() for r in self.results],
            "errors": [e.model_dump() for e in self.errors],
        }
        return json.dumps(data, indent=2)

    def to_csv(self) -> str:
        """Export results to CSV string."""
        df = self.to_dataframe()
        return df.to_csv(index=False)

    @classmethod
    def from_json(cls, json_str: str) -> ResultCollection:
        """Load results from JSON string."""
        data = json.loads(json_str)
        results = [BenchmarkResult(**r) for r in data.get("results", [])]
        errors = [BenchmarkError(**e) for e in data.get("errors", [])]
        return cls(results=results, errors=errors)

    def __len__(self) -> int:
        """Return number of results."""
        return len(self.results)
