"""CLI for running benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from boosters_eval.baseline import check_baseline, load_baseline, record_baseline
from boosters_eval.config import BoosterType, Task, TrainingConfig
from boosters_eval.datasets import DATASETS
from boosters_eval.reports import format_results_terminal, generate_report
from boosters_eval.results import ResultCollection
from boosters_eval.runners import get_available_runners
from boosters_eval.suite import (
    ABLATION_SUITES,
    FULL_SUITE,
    MINIMAL_SUITE,
    QUICK_SUITE,
    compare,
    run_ablation,
    run_suite,
)

DEFAULT_BASELINES: dict[str, Path] = {
    "minimal": Path("tests/baselines/minimal.json"),
    "quick": Path("tests/baselines/quick.json"),
    "full": Path("tests/baselines/full.json"),
}

app = typer.Typer(
    name="boosters-eval",
    help="Evaluate and compare gradient boosting libraries.",
    no_args_is_help=True,
)
baseline_app = typer.Typer(help="Manage baselines for regression detection.")
app.add_typer(baseline_app, name="baseline")
console = Console()


def _find_baseline_path(*, suite: str, baseline_path: Path | None = None) -> Path:
    """Find baseline file, checking common locations.

    Args:
        suite: Suite name used to determine the default baseline path.
        baseline_path: Explicit path to use, or None to auto-detect.

    Returns:
        Path to baseline file.

    Raises:
        typer.Exit: If baseline file not found.
    """
    if baseline_path is not None:
        if baseline_path.exists():
            return baseline_path
        console.print(f"[red]Baseline file not found: {baseline_path}[/red]")
        raise typer.Exit(2)

    default_baseline = DEFAULT_BASELINES.get(suite)
    if default_baseline is None:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        console.print(f"Valid options: {', '.join(sorted(DEFAULT_BASELINES.keys()))}")
        raise typer.Exit(2)

    # Try default locations
    candidates = [
        default_baseline,
        Path.cwd() / default_baseline,
        Path(__file__).parent.parent.parent.parent.parent.parent / default_baseline,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    console.print("[red]No baseline file found![/red]")
    console.print(f"Expected at: {default_baseline}")
    console.print("Run 'boosters-eval baseline record' to create one.")
    raise typer.Exit(2)


@app.command()
def quick(
    output: Annotated[
        Path | None,
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
        Path | None,
        typer.Option("--output", "-o", help="Output file path for results."),
    ] = None,
    booster: Annotated[
        str | None,
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


@app.command()
def check(
    suite: Annotated[
        str,
        typer.Option("--suite", "-s", help="Suite to run: minimal, quick, or full."),
    ] = "minimal",
    baseline_file: Annotated[
        Path | None,
        typer.Option("--baseline", "-b", help="Path to baseline JSON file."),
    ] = None,
    tolerance: Annotated[
        float,
        typer.Option("--tolerance", "-t", help="Regression tolerance (0.02 = 2%)."),
    ] = 0.02,
    fail_on_regression: Annotated[
        bool,
        typer.Option("--fail-on-regression/--no-fail-on-regression", help="Exit with code 1 if regression detected."),
    ] = True,
) -> None:
    """Run regression check against recorded baseline.

    Runs the full benchmark suite and compares results against a baseline.
    Reports regressions (quality got worse) and improvements (quality got better).

    Exit codes:
      0 - No regressions detected
      1 - Regressions detected (when --fail-on-regression is set)
      2 - Baseline file not found
    """
    baseline_path = _find_baseline_path(suite=suite, baseline_path=baseline_file)

    console.print("[bold]Running regression check[/bold]")
    console.print(f"  Baseline: {baseline_path}")
    console.print(f"  Tolerance: {tolerance:.1%}\n")

    # Load baseline
    baseline = load_baseline(baseline_path)
    console.print(f"  Baseline recorded: {baseline.created_at}")
    console.print(f"  Baseline git SHA: {baseline.git_sha or 'unknown'}")
    console.print()

    if suite == "minimal":
        benchmark_suite = MINIMAL_SUITE
    elif suite == "quick":
        benchmark_suite = QUICK_SUITE
    elif suite == "full":
        benchmark_suite = FULL_SUITE
    else:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        console.print("Valid options: minimal, quick, full")
        raise typer.Exit(1)

    # Run current benchmarks
    results = run_suite(benchmark_suite)

    # Check for regressions
    report = check_baseline(results, baseline, tolerance=tolerance)

    if report.improvements:
        console.print(f"[bold green]✓ Improvements detected: {len(report.improvements)}[/bold green]\n")

        table = Table(title="Improvements", style="green")
        table.add_column("Config", style="cyan")
        table.add_column("Metric", style="yellow")
        table.add_column("Baseline", style="dim")
        table.add_column("Current", style="green")
        table.add_column("Change", style="green")

        for imp in report.improvements:
            change_pct = ((imp["current"] - imp["baseline"]) / imp["baseline"]) * 100
            table.add_row(
                imp["config"],
                imp["metric"],
                f"{imp['baseline']:.4f}",
                f"{imp['current']:.4f}",
                f"{change_pct:+.1f}%",
            )

        console.print(table)
        console.print()

    if report.has_regressions:
        console.print(f"[bold red]✗ Regressions detected: {len(report.regressions)}[/bold red]\n")

        table = Table(title="Regressions", style="red")
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
        console.print("[bold green]✓ No regressions detected![/bold green]")


@app.command()
def ablation(
    study: Annotated[
        str,
        typer.Argument(help="Ablation study to run: depth, lr, growth."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output file path for results."),
    ] = None,
) -> None:
    """Run an ablation study comparing different hyperparameter settings.

    Available studies:
    - depth: Compare max_depth values (4, 6, 8)
    - lr: Compare learning rates (0.01, 0.1, 0.3)
    - growth: Compare growth strategies (depthwise, leafwise)
    """
    if study not in ABLATION_SUITES:
        console.print(f"[red]Unknown ablation study: {study}[/red]")
        console.print(f"Available: {', '.join(ABLATION_SUITES.keys())}")
        raise typer.Exit(1)

    console.print(f"[bold]Running ablation study: {study}[/bold]\n")

    # Run all variants in the ablation study
    variant_results = run_ablation(study)

    # Display results for each variant
    for variant_name, results in variant_results.items():
        console.print(f"\n[bold cyan]{variant_name}[/bold cyan]")
        format_results_terminal(results, require_significance=False)

    # Save combined report if output specified
    if output:
        # Combine all results
        combined = ResultCollection()
        for results in variant_results.values():
            for r in results.results:
                combined.add_result(r)
            for e in results.errors:
                combined.add_error(e)

        # Get training config from first variant
        first_suite = ABLATION_SUITES[study][0]
        tc = first_suite.to_training_config()
        booster_types = first_suite.get_booster_types()

        generate_report(
            combined,
            suite_name=f"ablation_{study}",
            output_path=output,
            title=f"Ablation Study: {study.upper()}",
            training_config=tc,
            booster_types=[bt.value for bt in booster_types],
        )
        console.print(f"\n[green]Results saved to {output}[/green]")


@app.command(name="list-ablations")
def list_ablations() -> None:
    """List available ablation studies."""
    table = Table(title="Available Ablation Studies")
    table.add_column("Name", style="cyan")
    table.add_column("Variants", style="green")
    table.add_column("Description", style="yellow")

    study_descriptions = {
        "depth": "Compare tree max_depth (4, 6, 8)",
        "lr": "Compare learning rates (0.01, 0.1, 0.3)",
        "growth": "Compare growth strategies (depthwise, leafwise)",
    }

    for name, suites in ABLATION_SUITES.items():
        variants = ", ".join(s.name.split("_")[-1] for s in suites)
        desc = study_descriptions.get(name, "")
        table.add_row(name, variants, desc)

    console.print(table)


@app.command(name="compare")
def compare_cmd(
    datasets: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Datasets to benchmark (can specify multiple).",
        ),
    ] = None,
    libraries: Annotated[
        list[str] | None,
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
        Path | None,
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
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output path for baseline JSON (default depends on --suite).",
        ),
    ] = None,
    suite: Annotated[
        str,
        typer.Option("--suite", "-s", help="Suite to run: minimal, quick or full."),
    ] = "minimal",
) -> None:
    """Record current results as a baseline.

    By default, records results from the minimal suite to tests/baselines/minimal.json.
    This baseline is used by 'boosters-eval check' for regression testing.
    """
    if suite == "minimal":
        benchmark_suite = MINIMAL_SUITE
    elif suite == "quick":
        benchmark_suite = QUICK_SUITE
    elif suite == "full":
        benchmark_suite = FULL_SUITE
    else:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        console.print("Valid options: minimal, quick, full")
        raise typer.Exit(1)

    # Default output path
    output_path = output or DEFAULT_BASELINES[suite]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Recording baseline using {suite} suite[/bold]\n")
    results = run_suite(benchmark_suite)

    baseline = record_baseline(results, output_path=output_path)
    console.print(f"\n[green]Baseline saved to {output_path}[/green]")
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
        typer.Option("--suite", "-s", help="Suite to run: minimal, quick or full."),
    ] = "minimal",
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

    if suite == "minimal":
        benchmark_suite = MINIMAL_SUITE
    elif suite == "quick":
        benchmark_suite = QUICK_SUITE
    elif suite == "full":
        benchmark_suite = FULL_SUITE
    else:
        console.print(f"[red]Unknown suite: {suite}[/red]")
        console.print("Valid options: minimal, quick, full")
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
        Path | None,
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
