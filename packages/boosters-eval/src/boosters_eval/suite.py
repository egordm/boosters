"""Benchmark suite execution."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from sklearn.model_selection import train_test_split

from boosters_eval.config import (
    BenchmarkConfig,
    BoosterType,
    GrowthStrategy,
    SuiteConfig,
)
from boosters_eval.datasets import DATASETS
from boosters_eval.results import BenchmarkError, ResultCollection

console = Console()


# =============================================================================
# Pre-defined Suite Configurations
# =============================================================================

QUICK_SUITE = SuiteConfig(
    name="quick",
    description="Quick benchmark suite for development (3 seeds, 2 datasets, 50 trees)",
    datasets=["california", "breast_cancer"],
    n_estimators=50,
    seeds=[42, 1379, 2716],
    libraries=["boosters", "xgboost", "lightgbm"],
)

FULL_SUITE = SuiteConfig(
    name="full",
    description="Full benchmark suite (5 seeds, all datasets, 100 trees, all booster types)",
    datasets=list(DATASETS.keys()),
    n_estimators=100,
    seeds=[42, 1379, 2716, 4053, 5390],
    libraries=["boosters", "xgboost", "lightgbm"],
    booster_types=[BoosterType.GBDT, BoosterType.GBLINEAR, BoosterType.LINEAR_TREES],
)

MINIMAL_SUITE = SuiteConfig(
    name="minimal",
    description="Minimal suite for CI (1 seed, 2 small datasets)",
    datasets=["synthetic_reg_small", "synthetic_bin_small"],
    n_estimators=10,
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
        n_estimators=100,
        max_depth=4,
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_depth_6",
        description="Depth ablation: max_depth=6",
        datasets=["california", "breast_cancer", "iris"],
        n_estimators=100,
        max_depth=6,
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_depth_8",
        description="Depth ablation: max_depth=8",
        datasets=["california", "breast_cancer", "iris"],
        n_estimators=100,
        max_depth=8,
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
        n_estimators=100,
        learning_rate=0.01,
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_lr_0.1",
        description="Learning rate ablation: lr=0.1",
        datasets=["california", "breast_cancer", "iris"],
        n_estimators=100,
        learning_rate=0.1,
        seeds=[42, 1379, 2716],
        libraries=["boosters", "xgboost", "lightgbm"],
        booster_type=BoosterType.GBDT,
    ),
    SuiteConfig(
        name="ablation_lr_0.3",
        description="Learning rate ablation: lr=0.3",
        datasets=["california", "breast_cancer", "iris"],
        n_estimators=100,
        learning_rate=0.3,
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
        n_estimators=100,
        seeds=[42, 1379, 2716],
        libraries=["boosters", "lightgbm"],
        booster_type=BoosterType.GBDT,
        growth_strategy=GrowthStrategy.DEPTHWISE,
    ),
    SuiteConfig(
        name="ablation_growth_leafwise",
        description="Growth strategy ablation: leafwise",
        datasets=["california", "breast_cancer", "iris"],
        n_estimators=100,
        seeds=[42, 1379, 2716],
        libraries=["boosters", "lightgbm"],
        booster_type=BoosterType.GBDT,
        growth_strategy=GrowthStrategy.LEAFWISE,
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
        for k, v in overrides.items():
            suite_dict[k] = v
        result.append(SuiteConfig(**suite_dict))
    return result


# =============================================================================
# Suite Execution
# =============================================================================


def run_suite(
    suite: SuiteConfig,
    *,
    validation_fraction: float = 0.2,
    verbose: bool = True,
    timing_mode: bool = False,
    measure_memory: bool = False,
) -> ResultCollection:
    """Run a benchmark suite and return collected results.

    Args:
        suite: Suite configuration to run.
        validation_fraction: Fraction of data to use for validation.
        verbose: Print progress information.
        timing_mode: Enable timing mode with warmup runs.
        measure_memory: Measure peak memory usage.

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

    # Build configs from suite's training parameters
    training = suite.to_training_config()

    # Get all booster types to run
    booster_types = suite.get_booster_types()

    configs: list[BenchmarkConfig] = []
    for ds_name in suite.datasets:
        if ds_name not in DATASETS:
            if verbose:
                console.print(f"[yellow]Unknown dataset: {ds_name}[/yellow]")
            continue

        configs.extend(
            BenchmarkConfig(
                name=f"{ds_name}/{booster_type.value}",
                dataset=DATASETS[ds_name],
                training=training,
                booster_type=booster_type,
                libraries=libraries,
            )
            for booster_type in booster_types
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
                x, y = config.dataset.loader()

                # Subsample if needed
                if config.dataset.subsample and len(y) > config.dataset.subsample:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(len(y), config.dataset.subsample, replace=False)
                    x, y = x[idx], y[idx]

                x_train, x_valid, y_train, y_valid = train_test_split(
                    x, y, test_size=validation_fraction, random_state=seed
                )

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
                            x_train=x_train,
                            y_train=y_train,
                            x_valid=x_valid,
                            y_valid=y_valid,
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
    n_estimators: int = 100,
    booster_type: BoosterType = BoosterType.GBDT,
    verbose: bool = True,
) -> ResultCollection:
    """Compare libraries on specified datasets.

    This is a convenience function that creates and runs a suite.

    Args:
        datasets: Dataset names to benchmark. Default: ["california", "breast_cancer"].
        libraries: Libraries to compare. Default: all available.
        seeds: Random seeds to use. Default: [42, 1379, 2716].
        n_estimators: Number of trees/rounds. Default: 100.
        booster_type: Type of booster. Default: GBDT.
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
        n_estimators=n_estimators,
        seeds=seeds,
        libraries=libraries,
        booster_type=booster_type,
    )

    return run_suite(suite, verbose=verbose)
