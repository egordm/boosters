"""CLI for running benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from boosters_eval.baseline import check_baseline, load_baseline, record_baseline
from boosters_eval.config import BoosterType, Task, TrainingConfig
from boosters_eval.datasets import DATASETS
from boosters_eval.metrics import LOWER_BETTER_METRICS
from boosters_eval.reports import format_results_terminal, generate_report
from boosters_eval.results import ResultCollection
from boosters_eval.runners import get_available_runners
from boosters_eval.suite import FULL_SUITE, QUICK_SUITE, compare, run_suite

app = typer.Typer(
    name="boosters-eval",
    help="Evaluate and compare gradient boosting libraries.",
    no_args_is_help=True,
)
baseline_app = typer.Typer(help="Manage baselines for regression detection.")
app.add_typer(baseline_app, name="baseline")
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

    # Terminal display: compact Rich tables
    format_results_terminal(results)

    # File output: full report with training config
    if output:
        tc = QUICK_SUITE.to_training_config()
        booster_types = QUICK_SUITE.get_booster_types()
        generate_report(
            results,
            suite_name="quick",
            output_path=output,
            title="Quick Benchmark Report",
            training_config=tc,
            booster_types=[bt.value for bt in booster_types],
        )
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def full(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path for results."),
    ] = None,
    booster: Annotated[
        Optional[str],
        typer.Option(
            "--booster",
            "-b",
            help="Booster type to run: gbdt, gblinear, linear_trees, or 'all'. Default: all.",
        ),
    ] = None,
) -> None:
    """Run the full benchmark suite (all datasets, more seeds).

    By default runs ALL booster types (GBDT, GBLinear, Linear Trees).
    Use --booster to select a specific model type.
    """
    # Determine booster types to run
    if booster is None or booster.lower() == "all":
        # Run all booster types (default)
        booster_types = [BoosterType.GBDT, BoosterType.GBLINEAR, BoosterType.LINEAR_TREES]
        booster_label = "all booster types"
        suite = FULL_SUITE
    else:
        try:
            booster_type = BoosterType(booster.lower())
        except ValueError:
            console.print(f"[red]Unknown booster type: {booster}[/red]")
            console.print(f"Valid options: {', '.join(b.value for b in BoosterType)}, all")
            raise typer.Exit(1)  # noqa: B904
        booster_types = [booster_type]
        booster_label = booster_type.value
        # Override to run only selected booster type
        suite = FULL_SUITE.model_copy(update={"booster_types": booster_types})

    console.print(f"[bold]Running full benchmark suite ({booster_label})[/bold]\n")

    results = run_suite(suite)

    # Terminal display: compact Rich tables
    format_results_terminal(results)

    # File output: full report with training config
    if output:
        tc = suite.to_training_config()
        generate_report(
            results,
            suite_name="full",
            output_path=output,
            title="Full Benchmark Report",
            training_config=tc,
            booster_types=[bt.value for bt in booster_types],
        )
        console.print(f"[green]Results saved to {output}[/green]")


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

    # Terminal display: compact Rich tables
    format_results_terminal(results)

    # File output: markdown with task sections
    if output:
        markdown = results.to_markdown()
        output.write_text(markdown)
        console.print(f"[green]Results saved to {output}[/green]")


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


@baseline_app.command(name="record")
def baseline_record(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output path for baseline JSON."),
    ],
    suite: Annotated[
        str,
        typer.Option("--suite", "-s", help="Suite to run: quick or full."),
    ] = "quick",
) -> None:
    """Record current results as a baseline."""
    if suite == "quick":
        benchmark_suite = QUICK_SUITE
    elif suite == "full":
        benchmark_suite = FULL_SUITE
    else:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        console.print("Valid options: quick, full")
        raise typer.Exit(1)

    console.print(f"[bold]Recording baseline using {suite} suite[/bold]\n")
    results = run_suite(benchmark_suite)

    baseline = record_baseline(results, output_path=output)
    console.print(f"\n[green]Baseline saved to {output}[/green]")
    console.print(f"  Results: {len(baseline.results)}")
    console.print(f"  Git SHA: {baseline.git_sha or 'unknown'}")


@baseline_app.command(name="check")
def baseline_check(
    baseline_file: Annotated[
        Path,
        typer.Argument(help="Path to baseline JSON file."),
    ],
    suite: Annotated[
        str,
        typer.Option("--suite", "-s", help="Suite to run: quick or full."),
    ] = "quick",
    tolerance: Annotated[
        float,
        typer.Option("--tolerance", "-t", help="Regression tolerance (0.02 = 2%)."),
    ] = 0.02,
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression", help="Exit with code 1 if regression detected."),
    ] = True,
) -> None:
    """Check current results against a baseline for regressions."""
    if not baseline_file.exists():
        console.print(f"[red]Baseline file not found: {baseline_file}[/red]")
        raise typer.Exit(2)

    if suite == "quick":
        benchmark_suite = QUICK_SUITE
    elif suite == "full":
        benchmark_suite = FULL_SUITE
    else:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Checking against baseline: {baseline_file}[/bold]")
    console.print(f"  Suite: {suite}")
    console.print(f"  Tolerance: {tolerance:.1%}\n")

    # Load baseline
    baseline = load_baseline(baseline_file)

    # Run current benchmarks
    results = run_suite(benchmark_suite)

    # Check for regressions
    report = check_baseline(results, baseline, tolerance=tolerance)

    if report.has_regressions:
        console.print("[bold red]Regressions detected![/bold red]\n")

        table = Table(title="Regressions")
        table.add_column("Config", style="cyan")
        table.add_column("Metric", style="yellow")
        table.add_column("Baseline", style="green")
        table.add_column("Current", style="red")
        table.add_column("Change", style="red")

        for reg in report.regressions:
            change_pct = ((reg["current"] - reg["baseline"]) / reg["baseline"]) * 100
            table.add_row(
                reg["config"],
                reg["metric"],
                f"{reg['baseline']:.4f}",
                f"{reg['current']:.4f}",
                f"{change_pct:+.1f}%",
            )

        console.print(table)

        if fail_on_regression:
            raise typer.Exit(1)
    else:
        console.print("[bold green]No regressions detected![/bold green]")

        if report.improvements:
            console.print(f"\n[green]Improvements: {len(report.improvements)}[/green]")


@app.command(name="report")
def report_cmd(
    suite: Annotated[
        str,
        typer.Option("--suite", "-s", help="Suite to run: quick or full."),
    ] = "quick",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for report."),
    ] = None,
    title: Annotated[
        str,
        typer.Option("--title", "-t", help="Report title."),
    ] = "Benchmark Report",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print report without saving."),
    ] = False,
) -> None:
    """Generate a benchmark report."""
    if suite == "quick":
        benchmark_suite = QUICK_SUITE
    elif suite == "full":
        benchmark_suite = FULL_SUITE
    else:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        console.print("Valid options: quick, full")
        raise typer.Exit(1)

    console.print(f"[bold]Generating report using {suite} suite[/bold]\n")
    results = run_suite(benchmark_suite)

    # Determine output path
    save_path = None if dry_run else output

    # Get training config for report
    tc = benchmark_suite.to_training_config()
    booster_types = benchmark_suite.get_booster_types()

    report = generate_report(
        results,
        suite_name=suite,
        output_path=save_path,
        title=title,
        training_config=tc,
        booster_types=[bt.value for bt in booster_types],
    )

    if dry_run or not output:
        console.print(report)
    else:
        console.print(f"\n[green]Report saved to {output}[/green]")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
