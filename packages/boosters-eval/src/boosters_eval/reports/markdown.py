"""Markdown report formatter.

Generates full benchmark reports with environment info, configuration,
and results tables grouped by task type with only primary metric.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from boosters_eval.config import Task, TrainingConfig
from boosters_eval.metrics import LOWER_BETTER_METRICS
from boosters_eval.reports.base import (
    ReportMetadata,
    get_git_sha,
    get_library_versions,
    get_machine_info,
)

if TYPE_CHECKING:
    from boosters_eval.results import ResultCollection

# Task display order and names
TASK_ORDER = [Task.REGRESSION, Task.BINARY, Task.MULTICLASS]
TASK_NAMES = {
    Task.REGRESSION: "Regression",
    Task.BINARY: "Binary Classification",
    Task.MULTICLASS: "Multiclass Classification",
}


def format_dataset_table(
    results: ResultCollection,
    dataset: str,
    *,
    highlight_best: bool = True,
    require_significance: bool = True,
    alpha: float = 0.05,
    precision: int = 4,
    show_timing: bool = True,
) -> str:
    """Format a summary table for a specific dataset.

    Shows only primary metric (+ optional timing).

    Args:
        results: ResultCollection with benchmark data
        dataset: Dataset name to format
        highlight_best: Bold the best value
        require_significance: Only bold if statistically significant
        alpha: Significance level for t-test
        precision: Decimal precision for values
        show_timing: Whether to include train_time_s

    Returns:
        Markdown formatted table string
    """
    summaries = results.summary_by_dataset()
    if dataset not in summaries:
        return f"No results for {dataset} dataset."

    df = summaries[dataset]

    # Get task type and primary metric
    task: Task | None = None
    for r in results.results:
        if r.dataset_name == dataset:
            task = Task(r.task)
            break
    if task is None:
        return f"No results for {dataset} dataset."

    pm = results._get_primary_metric_for_dataset(dataset)
    pm_mean_col = f"{pm}_mean"
    pm_std_col = f"{pm}_std"

    if pm_mean_col not in df.columns:
        return f"No {pm} metric for {dataset} dataset."

    # Build output rows
    output_rows: list[dict[str, str]] = []

    for booster in df["booster"].unique():
        booster_df = df[df["booster"] == booster]

        # Find best library for primary metric
        best_lib_pm: str | None = None
        best_lib_time: str | None = None

        valid = booster_df.dropna(subset=[pm_mean_col])  # pyright: ignore[reportCallIssue]
        if len(valid) >= 2:
            lower_better = pm in LOWER_BETTER_METRICS
            if lower_better:
                sorted_df = valid.sort_values(pm_mean_col, ascending=True)
            else:
                sorted_df = valid.sort_values(pm_mean_col, ascending=False)

            best_lib_pm = str(sorted_df.iloc[0]["library"])

            if require_significance and highlight_best:
                second_lib = str(sorted_df.iloc[1]["library"])
                raw_values = results._get_raw_values_by_library(task, dataset, pm, booster)
                best_vals = raw_values.get(best_lib_pm, [])
                second_vals = raw_values.get(second_lib, [])
                if not results._is_significantly_better(best_vals, second_vals, alpha):
                    best_lib_pm = None

        if show_timing and "train_time_s_mean" in df.columns:
            valid_time = booster_df.dropna(subset=["train_time_s_mean"])  # pyright: ignore[reportCallIssue]
            if len(valid_time) >= 2:
                sorted_time = valid_time.sort_values("train_time_s_mean", ascending=True)
                best_lib_time = str(sorted_time.iloc[0]["library"])

                if require_significance and highlight_best:
                    second_time_lib = str(sorted_time.iloc[1]["library"])
                    raw_time = results._get_raw_values_by_library(task, dataset, "train_time_s", booster)
                    best_time_vals = raw_time.get(best_lib_time, [])
                    second_time_vals = raw_time.get(second_time_lib, [])
                    if not results._is_significantly_better(best_time_vals, second_time_vals, alpha):
                        best_lib_time = None

        # Format rows
        for _, row in booster_df.iterrows():
            lib = str(row["library"])
            row_dict: dict[str, str] = {"Booster": str(booster), "Library": lib}

            # Primary metric
            mean_val = row[pm_mean_col]
            std_val = row.get(pm_std_col, np.nan) if pm_std_col in row.index else np.nan

            if bool(pd.isna(mean_val)):
                row_dict[pm] = "-"
            else:
                if std_val is None or bool(pd.isna(std_val)):
                    val_str = f"{mean_val:.{precision}f}"
                else:
                    std_val_f = float(std_val)
                    if std_val_f == 0.0:
                        val_str = f"{mean_val:.{precision}f}"
                    else:
                        val_str = f"{mean_val:.{precision}f}±{std_val_f:.{precision}f}"

                if highlight_best and best_lib_pm == lib:
                    val_str = f"**{val_str}**"
                row_dict[pm] = val_str

            # Timing
            if show_timing and "train_time_s_mean" in df.columns:
                time_mean = row.get("train_time_s_mean", np.nan)
                time_std = row.get("train_time_s_std", np.nan)

                if bool(pd.isna(time_mean)):
                    row_dict["train_time_s"] = "-"
                else:
                    if time_std is None or bool(pd.isna(time_std)):
                        time_str = f"{time_mean:.{precision}f}"
                    else:
                        time_std_f = float(time_std)
                        if time_std_f == 0.0:
                            time_str = f"{time_mean:.{precision}f}"
                        else:
                            time_str = f"{time_mean:.{precision}f}±{time_std_f:.{precision}f}"

                    if highlight_best and best_lib_time == lib:
                        time_str = f"**{time_str}**"
                    row_dict["train_time_s"] = time_str

            output_rows.append(row_dict)

    # Create DataFrame and use to_markdown
    output_df = pd.DataFrame(output_rows)
    col_order = ["Booster", "Library", pm]
    if show_timing and "train_time_s" in output_df.columns:
        col_order.append("train_time_s")
    output_df = output_df[[c for c in col_order if c in output_df.columns]]

    return str(output_df.to_markdown(index=False))


def render_report(
    results: ResultCollection,
    metadata: ReportMetadata,
    *,
    require_significance: bool = True,
) -> str:
    """Render a benchmark report as markdown.

    Groups results by task type, showing only primary metric per dataset.

    Args:
        results: Benchmark results to include
        metadata: Report metadata
        require_significance: Only bold winners if statistically significant

    Returns:
        Markdown string
    """
    lines = [
        f"# {metadata.title}",
        "",
        f"Generated: {metadata.created_at}",
        "",
        "## Environment",
        "",
        "| Property | Value |",
        "|----------|-------|",
        f"| Machine | {metadata.machine.cpu} |",
        f"| Cores | {metadata.machine.cores} |",
        f"| Memory | {metadata.machine.memory_gb} GB |",
        f"| OS | {metadata.machine.os} |",
    ]

    if metadata.git_sha:
        lines.append(f"| Git SHA | {metadata.git_sha} |")

    lines.extend([
        "",
        "## Library Versions",
        "",
        "| Library | Version |",
        "|---------|---------|",
        f"| Python | {metadata.library_versions.python} |",
    ])

    lv = metadata.library_versions
    if lv.boosters:
        lines.append(f"| boosters | {lv.boosters} |")
    if lv.xgboost:
        lines.append(f"| xgboost | {lv.xgboost} |")
    if lv.lightgbm:
        lines.append(f"| lightgbm | {lv.lightgbm} |")
    if lv.numpy:
        lines.append(f"| numpy | {lv.numpy} |")

    lines.extend([
        "",
        "## Configuration",
        "",
        f"- **Suite**: {metadata.suite_name}",
        f"- **Seeds**: {metadata.n_seeds}",
    ])

    # Add training parameters relevant to the booster types being run
    tc = metadata.training_config
    booster_types = metadata.booster_types or []
    if tc is not None:
        # Common parameters
        lines.append(f"- **n_estimators**: {tc.n_estimators}")
        lines.append(f"- **learning_rate**: {tc.learning_rate}")

        # Tree-based parameters (GBDT or LINEAR_TREES only)
        has_tree_based = any(bt in ("gbdt", "linear_trees") for bt in booster_types)
        if has_tree_based or not booster_types:
            lines.append(f"- **max_depth**: {tc.max_depth}")
            lines.append(f"- **growth_strategy**: {tc.growth_strategy.value}")
            lines.append(f"- **max_bins**: {tc.max_bins}")
            lines.append(f"- **min_samples_leaf**: {tc.min_samples_leaf}")

        # Regularization
        lines.append(f"- **reg_lambda (L2)**: {tc.reg_lambda}")
        lines.append(f"- **reg_alpha (L1)**: {tc.reg_alpha}")

        # Linear tree parameters
        if "linear_trees" in booster_types:
            lines.append(f"- **linear_l2**: {tc.linear_l2}")

    if booster_types:
        lines.append(f"- **booster_types**: {', '.join(booster_types)}")

    lines.extend([
        "",
        "## Results",
        "",
    ])

    # Group datasets by task type
    dataset_tasks: dict[str, Task] = {}
    for r in results.results:
        if r.dataset_name not in dataset_tasks:
            dataset_tasks[r.dataset_name] = Task(r.task)

    task_datasets: dict[Task, list[str]] = {task: [] for task in TASK_ORDER}
    for dataset, task in dataset_tasks.items():
        if task in task_datasets:
            task_datasets[task].append(dataset)

    # Render by task type
    for task in TASK_ORDER:
        datasets = sorted(task_datasets.get(task, []))
        if not datasets:
            continue

        lines.append(f"### {TASK_NAMES[task]}")
        lines.append("")

        for dataset in datasets:
            pm = results._get_primary_metric_for_dataset(dataset)
            lines.append(f"**{dataset}** (primary: {pm})")
            lines.append("")
            lines.append(
                format_dataset_table(
                    results,
                    dataset,
                    highlight_best=True,
                    require_significance=require_significance,
                )
            )
            lines.append("")

    lines.extend([
        "## Reproducing",
        "",
        "```bash",
        f"boosters-eval {metadata.suite_name.lower()}",
        "```",
        "",
        "---",
        "",
        "*Best values per metric are **bolded**. Lower is better for loss/time metrics.*",
    ])

    return "\n".join(lines)


def generate_report(
    results: ResultCollection,
    suite_name: str,
    output_path: Path | None = None,
    title: str = "Benchmark Report",
    *,
    training_config: TrainingConfig | None = None,
    booster_types: list[str] | None = None,
) -> str:
    """Generate a benchmark report.

    Args:
        results: Benchmark results
        suite_name: Name of the suite that was run
        output_path: Optional path to save the report
        title: Report title
        training_config: Training configuration used for the benchmark
        booster_types: List of booster types evaluated

    Returns:
        Rendered markdown report
    """
    # Collect metadata
    machine = get_machine_info()
    library_versions = get_library_versions()
    git_sha = get_git_sha()

    # Count seeds
    seeds = set()
    for result in results.results:
        seeds.add(result.seed)

    metadata = ReportMetadata(
        title=title,
        created_at=datetime.now(UTC).isoformat(),
        git_sha=git_sha,
        machine=machine,
        library_versions=library_versions,
        suite_name=suite_name,
        n_seeds=len(seeds),
        training_config=training_config,
        booster_types=booster_types,
    )

    report = render_report(results, metadata)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report
