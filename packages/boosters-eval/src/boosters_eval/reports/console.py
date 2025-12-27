"""Console output formatter using Rich tables.

Shows results grouped by task type, sorted, with only primary metric visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from boosters_eval.config import Task
from boosters_eval.metrics import LOWER_BETTER_METRICS

if TYPE_CHECKING:
    from boosters_eval.results import ResultCollection

console = Console()

# Task display order
TASK_ORDER = [Task.REGRESSION, Task.BINARY, Task.MULTICLASS]


def format_results_terminal(
    results: ResultCollection,
    *,
    require_significance: bool = True,
    show_timing: bool = True,
) -> None:
    """Display results as Rich tables grouped by task type.

    Shows only primary metric per dataset (+ optional timing).
    Datasets are sorted by task type for clarity.

    Args:
        results: ResultCollection to display
        require_significance: Only highlight winners if statistically significant
        show_timing: Whether to show train_time_s column
    """
    summaries = results.summary_by_dataset()

    if not summaries:
        console.print("[yellow]No results to display.[/yellow]")
        return

    # Group datasets by task type
    task_datasets: dict[Task, list[str]] = {task: [] for task in TASK_ORDER}
    dataset_tasks: dict[str, Task] = {}

    for r in results.results:
        if r.dataset_name not in dataset_tasks:
            task = Task(r.task)
            dataset_tasks[r.dataset_name] = task
            task_datasets[task].append(r.dataset_name)

    # Display grouped by task
    for task in TASK_ORDER:
        datasets = sorted(task_datasets.get(task, []))
        if not datasets:
            continue

        # Task header
        task_names = {
            Task.REGRESSION: "Regression",
            Task.BINARY: "Binary Classification",
            Task.MULTICLASS: "Multiclass Classification",
        }
        console.print(f"\n[bold cyan]{task_names[task]}[/bold cyan]")

        for dataset in datasets:
            if dataset not in summaries:
                continue

            df = summaries[dataset]

            # Get primary metric for this dataset
            pm = results._get_primary_metric_for_dataset(dataset)
            pm_mean_col = f"{pm}_mean"
            pm_std_col = f"{pm}_std"

            if pm_mean_col not in df.columns:
                continue

            # Create table for this dataset
            table = Table(title=f"{dataset}", show_header=True, header_style="bold")
            table.add_column("Booster", style="cyan", no_wrap=True)
            table.add_column("Library", style="green")
            table.add_column(pm, justify="right")

            if show_timing and "train_time_s_mean" in df.columns:
                table.add_column("train_time_s", justify="right")

            # Find best values per booster
            for booster in df["booster"].unique():
                booster_df = df[df["booster"] == booster]

                # Determine best library for primary metric
                best_lib_pm: str | None = None
                best_lib_time: str | None = None

                valid = booster_df.dropna(subset=[pm_mean_col])
                if len(valid) >= 2:
                    lower_better = pm in LOWER_BETTER_METRICS
                    best_idx = valid[pm_mean_col].idxmin() if lower_better else valid[pm_mean_col].idxmax()
                    best_lib_pm = str(valid.loc[best_idx, "library"])  # pyright: ignore[reportArgumentType]

                if show_timing and "train_time_s_mean" in df.columns:
                    valid_time = booster_df.dropna(subset=["train_time_s_mean"])
                    if len(valid_time) >= 2:
                        best_time_idx = valid_time["train_time_s_mean"].idxmin()
                        best_lib_time = str(valid_time.loc[best_time_idx, "library"])  # pyright: ignore[reportArgumentType]

                # Add rows
                for _, row in booster_df.iterrows():
                    lib = str(row["library"])
                    row_data = [str(booster), lib]

                    # Primary metric
                    mean_val = row[pm_mean_col]
                    std_val = row.get(pm_std_col, np.nan) if pm_std_col in row.index else np.nan

                    if pd.isna(mean_val):
                        row_data.append("-")
                    else:
                        if pd.isna(std_val) or std_val == 0:
                            val_str = f"{mean_val:.4f}"
                        else:
                            val_str = f"{mean_val:.4f}±{std_val:.4f}"

                        if best_lib_pm == lib:
                            val_str = f"[bold green]{val_str}[/bold green]"
                        row_data.append(val_str)

                    # Timing
                    if show_timing and "train_time_s_mean" in df.columns:
                        time_mean = row.get("train_time_s_mean", np.nan)
                        time_std = row.get("train_time_s_std", np.nan)

                        if pd.isna(time_mean):
                            row_data.append("-")
                        else:
                            if pd.isna(time_std) or time_std == 0:
                                time_str = f"{time_mean:.4f}"
                            else:
                                time_str = f"{time_mean:.4f}±{time_std:.4f}"

                            if best_lib_time == lib:
                                time_str = f"[bold green]{time_str}[/bold green]"
                            row_data.append(time_str)

                    table.add_row(*row_data)

            console.print(table)
