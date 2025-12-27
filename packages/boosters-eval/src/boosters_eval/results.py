"""Benchmark result models and collection."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

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

    def format_summary_table(
        self,
        task: Task,
        *,
        highlight_best: bool = True,
        precision: int = 4,
    ) -> str:
        """Format a summary table for a specific task with highlighting.

        Args:
            task: Task type to format.
            highlight_best: Bold the best value in each metric.
            precision: Decimal precision for values.

        Returns:
            Markdown formatted string.
        """
        summaries = self.summary_by_task()
        if task not in summaries:
            return f"No results for {task.value} task."

        df = summaries[task]
        metrics = TASK_METRICS[task] + TIMING_METRICS + MEMORY_METRICS

        # Build formatted table
        lines = []
        header = ["Dataset", "Library"]

        # Add metric columns (without _mean suffix for cleaner display)
        for metric in metrics:
            mean_col = f"{metric}_mean"
            if mean_col in df.columns:
                header.append(metric)

        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")

        # Group by dataset to find best values
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]

            # Find best library for each metric
            best_values: dict[str, tuple[str, float]] = {}
            for metric in metrics:
                mean_col = f"{metric}_mean"
                if mean_col not in dataset_df.columns:
                    continue

                valid = dataset_df.dropna(subset=[mean_col])
                if valid.empty:
                    continue

                lower_better = metric in LOWER_BETTER_METRICS or metric.endswith("_time_s")
                if lower_better:
                    best_idx = valid[mean_col].idxmin()
                else:
                    best_idx = valid[mean_col].idxmax()

                best_lib = valid.loc[best_idx, "library"]  # pyright: ignore[reportArgumentType]
                best_val = valid.loc[best_idx, mean_col]  # pyright: ignore[reportArgumentType]
                best_values[metric] = (best_lib, best_val)

            # Format rows
            for _, row in dataset_df.iterrows():
                lib = row["library"]
                row_data = [str(dataset), str(lib)]

                for metric in metrics:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"

                    if mean_col not in df.columns:
                        continue

                    mean_val = row[mean_col]
                    std_val = row.get(std_col, np.nan)

                    if pd.isna(mean_val):
                        row_data.append("-")
                    else:
                        # Format value with optional std
                        if pd.isna(std_val) or std_val == 0:
                            val_str = f"{mean_val:.{precision}f}"
                        else:
                            val_str = f"{mean_val:.{precision}f}±{std_val:.{precision}f}"

                        # Bold if best
                        if highlight_best and metric in best_values:
                            best_lib, _ = best_values[metric]
                            if lib == best_lib:
                                val_str = f"**{val_str}**"

                        row_data.append(val_str)

                lines.append("| " + " | ".join(row_data) + " |")

        return "\n".join(lines)

    def to_markdown(self, precision: int = 4, highlight_best: bool = True) -> str:
        """Generate markdown tables grouped by task type.

        Args:
            precision: Decimal precision for values.
            highlight_best: Bold the best value in each metric.

        Returns:
            Markdown formatted string with sections per task.
        """
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
            sections.append(self.format_summary_table(task, highlight_best=highlight_best, precision=precision))
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
