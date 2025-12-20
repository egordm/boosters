"""CLI for running benchmarks."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from boosters_eval.datasets import DATASETS, BenchmarkConfig, BoosterType, Task, TrainingConfig
from boosters_eval.runners import get_available_runners
from boosters_eval.suite import BenchmarkSuite, run_all_combinations

app = typer.Typer(
    name="boosters-eval",
    help="Evaluate and compare gradient boosting libraries.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def compare(
    datasets: list[str] = typer.Option(
        ["california", "breast_cancer", "iris"],
        "--dataset",
        "-d",
        help="Datasets to benchmark (can specify multiple).",
    ),
    libraries: list[str] = typer.Option(
        None,
        "--library",
        "-l",
        help="Libraries to compare. Default: all available.",
    ),
    booster: str = typer.Option(
        "gbdt",
        "--booster",
        "-b",
        help="Booster type: gbdt, gblinear, linear_trees.",
    ),
    n_trees: int = typer.Option(100, "--trees", "-t", help="Number of trees."),
    max_depth: int = typer.Option(6, "--depth", help="Max tree depth."),
    learning_rate: float = typer.Option(0.1, "--lr", help="Learning rate."),
    seeds: int = typer.Option(3, "--seeds", "-s", help="Number of random seeds."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (markdown).",
    ),
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

    training = TrainingConfig(
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )

    configs: list[BenchmarkConfig] = []
    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            console.print(f"[yellow]Unknown dataset: {dataset_name}[/yellow]")
            continue

        configs.append(
            BenchmarkConfig(
                name=f"{dataset_name}/{booster}",
                dataset=DATASETS[dataset_name],
                training=training,
                booster_type=booster_type,
                libraries=libs,
            )
        )

    if not configs:
        console.print("[red]No valid datasets specified![/red]")
        raise typer.Exit(1)

    console.print("[bold]Running benchmarks[/bold]")
    console.print(f"  Datasets: {[c.dataset.name for c in configs]}")
    console.print(f"  Libraries: {libs}")
    console.print(f"  Booster: {booster_type.value}")
    console.print()

    suite = BenchmarkSuite(configs, seeds=list(range(seeds)))
    suite.run(verbose=True)

    report = suite.report()
    console.print(report)

    if output:
        output.write_text(report)
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command()
def all(
    tasks: list[str] = typer.Option(
        None,
        "--task",
        "-t",
        help="Task types to run: regression, binary, multiclass. Default: all.",
    ),
    boosters: list[str] = typer.Option(
        ["gbdt"],
        "--booster",
        "-b",
        help="Booster types: gbdt, gblinear, linear_trees.",
    ),
    libraries: list[str] = typer.Option(
        None,
        "--library",
        "-l",
        help="Libraries to compare. Default: all available.",
    ),
    seeds: int = typer.Option(3, "--seeds", "-s", help="Number of random seeds."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (markdown).",
    ),
) -> None:
    """Run benchmarks on all datasets with specified configuration."""
    available = get_available_runners()
    libs = libraries or available
    libs = [lib for lib in libs if lib in available]

    if not libs:
        console.print("[red]No libraries available![/red]")
        raise typer.Exit(1)

    # Filter datasets by task
    task_filter = None
    if tasks:
        task_filter = [Task(t) for t in tasks]

    ds_names = list(DATASETS.keys())
    if task_filter:
        ds_names = [name for name, ds in DATASETS.items() if ds.task in task_filter]

    booster_types = [BoosterType(b) for b in boosters]

    console.print("[bold]Running comprehensive benchmarks[/bold]")
    console.print(f"  Datasets: {ds_names}")
    console.print(f"  Libraries: {libs}")
    console.print(f"  Boosters: {boosters}")
    console.print()

    suite = run_all_combinations(
        datasets=ds_names,
        booster_types=booster_types,
        libraries=libs,
        seeds=list(range(seeds)),
        verbose=True,
    )

    report = suite.report()
    console.print(report)

    if output:
        output.write_text(report)
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

    library_boosters = {
        "xgboost": "gbdt, gblinear",
        "lightgbm": "gbdt, linear_trees",
        "catboost": "gbdt",
    }

    for lib in ["xgboost", "lightgbm", "catboost"]:
        status = "✓ Available" if lib in available else "✗ Not installed"
        style = "green" if lib in available else "red"
        table.add_row(lib, f"[{style}]{status}[/{style}]", library_boosters.get(lib, ""))

    console.print(table)


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
