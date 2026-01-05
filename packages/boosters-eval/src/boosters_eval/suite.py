"""Benchmark suite execution."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from boosters_eval.config import (
    BenchmarkConfig,
    BoosterType,
    GrowthStrategy,
    SuiteConfig,
    TrainingConfig,
    resolve_training_config,
)
from boosters_eval.datasets import DATASETS
from boosters_eval.preprocessing import prepare_run_data
from boosters_eval.results import BenchmarkError, ResultCollection

console = Console()


# =============================================================================
# Pre-defined Suite Configurations
# =============================================================================

QUICK_SUITE = SuiteConfig(
    name="quick",
    description="Quick benchmark suite for development (3 seeds, 2 datasets, 50 trees)",
    datasets=["california", "breast_cancer"],
    training=TrainingConfig(n_estimators=50),
    seeds=[42, 1379, 2716],
    libraries=["boosters", "xgboost", "lightgbm"],
)

FULL_SUITE = SuiteConfig(
    name="full",
    description="Full benchmark suite (5 seeds, all datasets, 100 trees, all booster types)",
    datasets=list(DATASETS.keys()),
    training=TrainingConfig(n_estimators=100),
    seeds=[42, 1379, 2716, 4053, 5390],
    libraries=["boosters", "xgboost", "lightgbm"],
    booster_types=[BoosterType.GBDT, BoosterType.GBLINEAR, BoosterType.LINEAR_TREES],
)

MINIMAL_SUITE = SuiteConfig(
    name="minimal",
    description="Minimal suite for CI (1 seed, 2 small datasets)",
    datasets=["synthetic_reg_small", "synthetic_bin_small"],
    training=TrainingConfig(n_estimators=10),
    seeds=[42],
    libraries=["boosters", "xgboost", "lightgbm"],
)


# =============================================================================
# Ablation Suites
# =============================================================================

# Depth ablation: compare different max_depth values
ABLATION_DEPTH = [
    SuiteConfig(
        name="ablation_depth_4",
        description="Depth ablation: max_depth=4",
        datasets=["california", "breast_cancer", "iris"],
        training=TrainingConfig(n_estimators=100, max_depth=4),
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_depth_6",
        description="Depth ablation: max_depth=6",
        datasets=["california", "breast_cancer", "iris"],
        training=TrainingConfig(n_estimators=100, max_depth=6),
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_depth_8",
        description="Depth ablation: max_depth=8",
        datasets=["california", "breast_cancer", "iris"],
        training=TrainingConfig(n_estimators=100, max_depth=8),
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
]

# Learning rate ablation: compare different learning rates
ABLATION_LR = [
    SuiteConfig(
        name="ablation_lr_0.01",
        description="Learning rate ablation: lr=0.01",
        datasets=["california", "breast_cancer", "iris"],
        training=TrainingConfig(n_estimators=100, learning_rate=0.01),
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_lr_0.1",
        description="Learning rate ablation: lr=0.1",
        datasets=["california", "breast_cancer", "iris"],
        training=TrainingConfig(n_estimators=100, learning_rate=0.1),
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_lr_0.3",
        description="Learning rate ablation: lr=0.3",
        datasets=["california", "breast_cancer", "iris"],
        training=TrainingConfig(n_estimators=100, learning_rate=0.3),
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
]

# Growth strategy ablation: depthwise vs leafwise
ABLATION_GROWTH = [
    SuiteConfig(
        name="ablation_growth_depthwise",
        description="Growth strategy ablation: depthwise",
        datasets=["california", "breast_cancer", "iris"],
        training=TrainingConfig(n_estimators=100, growth_strategy=GrowthStrategy.DEPTHWISE),
        seeds=[42, 1379, 2716],
        libraries=["boosters", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_growth_leafwise",
        description="Growth strategy ablation: leafwise",
        datasets=["california", "breast_cancer", "iris"],
        training=TrainingConfig(n_estimators=100, growth_strategy=GrowthStrategy.LEAFWISE),
        seeds=[42, 1379, 2716],
        libraries=["boosters", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
]

# All ablation suites by name
ABLATION_SUITES: dict[str, list[SuiteConfig]] = {
    "depth": ABLATION_DEPTH,
    "lr": ABLATION_LR,
    "growth": ABLATION_GROWTH,
}


def run_ablation(
    ablation_name: str,
    *,
    verbose: bool = True,
) -> dict[str, ResultCollection]:
    """Run an ablation study and return results keyed by variant name.

    Args:
        ablation_name: Name of the ablation study (depth, lr, growth).
        verbose: Print progress information.

    Returns:
        Dictionary mapping variant name to ResultCollection.
    """
    if ablation_name not in ABLATION_SUITES:
        raise ValueError(f"Unknown ablation: {ablation_name}. Available: {list(ABLATION_SUITES.keys())}")

    suites = ABLATION_SUITES[ablation_name]
    results: dict[str, ResultCollection] = {}

    for suite in suites:
        if verbose:
            console.print(f"[bold]Running {suite.name}[/bold]")
        collection = run_suite(suite, verbose=verbose)
        results[suite.name] = collection

    return results


def create_ablation_suite(
    name: str,
    base_suite: SuiteConfig,
    variants: dict[str, dict[str, Any]],
) -> list[SuiteConfig]:
    """Create ablation suites from a base suite and variants.

    Args:
        name: Base name for the ablation suites.
        base_suite: Base suite configuration.
        variants: Dictionary of variant name -> config overrides.

    Returns:
        List of SuiteConfig objects, one per variant.

    Example:
        >>> from boosters_eval.suite import QUICK_SUITE, create_ablation_suite
        >>> variants = {
        ...     "depth_4": {"max_depth": 4},
        ...     "depth_8": {"max_depth": 8},
        ... }
        >>> suites = create_ablation_suite("depth_ablation", QUICK_SUITE, variants)
        >>> len(suites)
        2
    """
    result = []
    for variant_name, overrides in variants.items():
        suite_dict = base_suite.model_dump()
        suite_dict["name"] = f"{name}_{variant_name}"
        suite_dict["description"] = f"{base_suite.description} - {variant_name}"
        training_dict = dict(suite_dict.get("training") or {})

        for k, v in overrides.items():
            if k in TrainingConfig.model_fields:
                training_dict[k] = v
            else:
                suite_dict[k] = v

        suite_dict["training"] = training_dict
        result.append(SuiteConfig(**suite_dict))
    return result


# =============================================================================
# Suite Execution
# =============================================================================


def run_suite(
    suite: SuiteConfig,
    *,
    verbose: bool = True,
    timing_mode: bool = False,
    measure_memory: bool = False,
    cli_training: TrainingConfig | None = None,
) -> ResultCollection:
    """Run a benchmark suite and return collected results.

    Args:
        suite: Suite configuration to run.
        verbose: Print progress information.
        timing_mode: Enable timing mode with warmup runs.
        measure_memory: Measure peak memory usage.
        cli_training: Training config overrides applied after suite/dataset/base.

    Returns:
        ResultCollection with all results and errors.
    """
    from boosters_eval.runners import get_available_runners, get_runner

    collection = ResultCollection()
    available_runners = get_available_runners()

    # Filter to available libraries
    libraries = [lib for lib in suite.libraries if lib in available_runners]
    if not libraries:
        if verbose:
            console.print("[yellow]No libraries available to run![/yellow]")
        return collection

    # Get all booster types to run
    booster_types = suite.get_booster_types()

    configs: list[BenchmarkConfig] = []
    for ds_name in suite.datasets:
        if ds_name not in DATASETS:
            if verbose:
                console.print(f"[yellow]Unknown dataset: {ds_name}[/yellow]")
            continue

        ds = DATASETS[ds_name]
        for booster_type in booster_types:
            training = resolve_training_config(
                booster_type=booster_type,
                dataset=ds,
                suite=suite,
                cli=cli_training,
            )
            configs.append(
                BenchmarkConfig(
                    name=f"{ds_name}/{booster_type.value}",
                    dataset=ds,
                    training=training,
                    booster_type=booster_type,
                    libraries=libraries,
                )
            )

    if verbose:
        console.print(f"[bold]Running suite: {suite.name}[/bold]")
        console.print(f"  Datasets: {len(suite.datasets)}")
        console.print(f"  Booster types: {[bt.value for bt in booster_types]}")
        console.print(f"  Libraries: {libraries}")
        console.print(f"  Seeds: {len(suite.seeds)}")
        console.print()

    total_runs = len(configs) * len(suite.seeds) * len(libraries)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=not verbose,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=total_runs)

        for config in configs:
            for seed in suite.seeds:
                # Load and split data
                loaded = config.dataset.loader()
                run_data = prepare_run_data(config=config, loaded=loaded, seed=seed)

                for library in config.libraries:
                    progress.update(
                        task,
                        description=f"{config.name} | {library} | seed={seed}",
                    )

                    try:
                        runner = get_runner(library)
                        if not runner.supports(config):
                            progress.advance(task)
                            continue

                        result = runner.run(
                            config=config,
                            data=run_data,
                            seed=seed,
                            timing_mode=timing_mode,
                            measure_memory=measure_memory,
                        )
                        collection.add_result(result)

                    except Exception as e:
                        error = BenchmarkError(
                            config_name=config.name,
                            library=library,
                            seed=seed,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            dataset_name=config.dataset.name,
                        )
                        collection.add_error(error)

                    progress.advance(task)

    if verbose and collection.errors:
        console.print(f"\n[yellow]Completed with {len(collection.errors)} errors[/yellow]")

    return collection


def compare(
    datasets: list[str] | None = None,
    *,
    libraries: list[str] | None = None,
    seeds: list[int] | None = None,
    training: TrainingConfig | None = None,
    booster_type: BoosterType = BoosterType.GBDT,
    booster_types: list[BoosterType] | None = None,
    verbose: bool = True,
) -> ResultCollection:
    """Compare libraries on specified datasets.

    This is a convenience function that creates and runs a suite.

    Args:
        datasets: Dataset names to benchmark. Default: ["california", "breast_cancer"].
        libraries: Libraries to compare. Default: all available.
        seeds: Random seeds to use. Default: [42, 1379, 2716].
        training: Training config overrides from the caller (applied last, like CLI overrides).
        booster_type: Type of booster. Default: GBDT.
        booster_types: If set, runs these booster types instead of a single booster.
        verbose: Print progress. Default: True.

    Returns:
        ResultCollection with all results.

    Example:
        >>> from boosters_eval import compare
        >>> results = compare(["california"], seeds=[42])  # doctest: +SKIP
        >>> print(results.to_markdown())  # doctest: +SKIP
    """
    from boosters_eval.runners import get_available_runners

    datasets = datasets or ["california", "breast_cancer"]
    libraries = libraries or get_available_runners()
    seeds = seeds or [42, 1379, 2716]
    suite = SuiteConfig(
        name="compare",
        description="Ad-hoc comparison",
        datasets=datasets,
        seeds=seeds,
        libraries=libraries,
        booster_type=booster_type,
        booster_types=booster_types,
    )

    return run_suite(suite, verbose=verbose, cli_training=training)
