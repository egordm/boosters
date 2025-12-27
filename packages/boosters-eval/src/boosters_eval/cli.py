"""CLI for running benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from boosters_eval.config import BoosterType, Task, TrainingConfig
from boosters_eval.datasets import DATASETS
from boosters_eval.runners import get_available_runners
from boosters_eval.suite import FULL_SUITE, QUICK_SUITE, compare, run_suite

app = typer.Typer(
    name="boosters-eval",
    help="Evaluate and compare gradient boosting libraries.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def quick(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path for results."),
    ] = None,
) -> None:
    """Run the quick benchmark suite (small datasets, few iterations)."""
    console.print("[bold]Running quick benchmark suite[/bold]\n")

    results = run_suite(QUICK_SUITE)
    markdown = results.to_markdown()
    console.print(markdown)

    if output:
        output.write_text(markdown)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def full(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path for results."),
    ] = None,
) -> None:
    """Run the full benchmark suite (all datasets, more seeds)."""
    console.print("[bold]Running full benchmark suite[/bold]\n")

    results = run_suite(FULL_SUITE)
    markdown = results.to_markdown()
    console.print(markdown)

    if output:
        output.write_text(markdown)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command(name="compare")
def compare_cmd(
    datasets: Annotated[
        Optional[list[str]],
        typer.Option(
            "--dataset",
            "-d",
            help="Datasets to benchmark (can specify multiple).",
        ),
    ] = None,
    libraries: Annotated[
        Optional[list[str]],
        typer.Option("--library", "-l", help="Libraries to compare. Default: all available."),
    ] = None,
    booster: Annotated[
        str,
        typer.Option("--booster", "-b", help="Booster type: gbdt, gblinear, linear_trees."),
    ] = "gbdt",
    n_estimators: Annotated[
        int,
        typer.Option("--trees", "-t", help="Number of trees/rounds."),
    ] = 100,
    max_depth: Annotated[
        int,
        typer.Option("--depth", help="Max tree depth."),
    ] = 6,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Learning rate."),
    ] = 0.1,
    seeds: Annotated[
        int,
        typer.Option("--seeds", "-s", help="Number of random seeds."),
    ] = 3,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path."),
    ] = None,
) -> None:
    """Compare libraries on selected datasets."""
    available = get_available_runners()
    libs = libraries or available
    libs = [lib for lib in libs if lib in available]

    if not libs:
        console.print("[red]No libraries available![/red]")
        raise typer.Exit(1)

    try:
        booster_type = BoosterType(booster)
    except ValueError:
        console.print(f"[red]Invalid booster type: {booster}[/red]")
        console.print(f"Valid options: {[b.value for b in BoosterType]}")
        raise typer.Exit(1) from None

    # Default datasets
    dataset_names = datasets or ["california", "breast_cancer", "iris"]

    # Filter valid datasets
    valid_datasets = [d for d in dataset_names if d in DATASETS]
    if not valid_datasets:
        console.print("[red]No valid datasets specified![/red]")
        console.print(f"Available: {list(DATASETS.keys())}")
        raise typer.Exit(1)

    training = TrainingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )

    console.print("[bold]Running comparison[/bold]")
    console.print(f"  Datasets: {valid_datasets}")
    console.print(f"  Libraries: {libs}")
    console.print(f"  Booster: {booster_type.value}")
    console.print()

    results = compare(
        datasets=valid_datasets,
        libraries=libs,
        seeds=list(range(seeds)),
        n_estimators=training.n_estimators,
        booster_type=booster_type,
    )

    markdown = results.to_markdown()
    console.print(markdown)

    if output:
        output.write_text(markdown)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def list_datasets() -> None:
    """List available datasets."""
    table = Table(title="Available Datasets")
    table.add_column("Name", style="cyan")
    table.add_column("Task", style="green")
    table.add_column("Classes", style="yellow")

    for name, config in DATASETS.items():
        classes = str(config.n_classes) if config.n_classes else "-"
        table.add_row(name, config.task.value, classes)

    console.print(table)


@app.command()
def list_libraries() -> None:
    """List available benchmark libraries."""
    available = get_available_runners()

    table = Table(title="Available Libraries")
    table.add_column("Library", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Supported Boosters", style="yellow")

    library_info = {
        "boosters": "gbdt, gblinear",
        "xgboost": "gbdt, gblinear",
        "lightgbm": "gbdt, linear_trees",
    }

    for lib, boosters_supported in library_info.items():
        status = "Available" if lib in available else "Not installed"
        style = "green" if lib in available else "red"
        table.add_row(lib, f"[{style}]{status}[/{style}]", boosters_supported)

    console.print(table)


@app.command()
def list_tasks() -> None:
    """List supported ML task types."""
    table = Table(title="Supported Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Primary Metric", style="yellow")

    task_info = [
        (Task.REGRESSION, "Predict continuous values", "RMSE"),
        (Task.BINARY, "Binary classification", "Log Loss"),
        (Task.MULTICLASS, "Multi-class classification", "Multi-class Log Loss"),
    ]

    for task, desc, metric in task_info:
        table.add_row(task.value, desc, metric)

    console.print(table)


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
