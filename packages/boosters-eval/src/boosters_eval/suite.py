"""Benchmark suite for running evaluations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from boosters_eval.datasets import BenchmarkConfig, BoosterType, Task
from boosters_eval.metrics import is_lower_better, primary_metric
from boosters_eval.results import BenchmarkResult


class BenchmarkSuite:
    """A collection of benchmark configurations to run."""

    def __init__(
        self,
        configs: list[BenchmarkConfig],
        seeds: list[int] | None = None,
        validation_fraction: float = 0.2,
    ):
        self.configs = configs
        self.seeds = seeds or [42, 1379, 2716, 4053, 5390]
        self.validation_fraction = validation_fraction
        self._results: list[BenchmarkResult] = []

    def run(self, verbose: bool = True) -> pd.DataFrame:
        """Run all benchmarks and return results as DataFrame."""
        from boosters_eval.runners import get_runner

        self._results = []

        for config in self.configs:
            if verbose:
                print(f"\n=== {config.name} ===")

            for seed in self.seeds:
                # Load and split data
                x, y = config.dataset.loader()

                # Subsample if needed
                if config.dataset.subsample and len(y) > config.dataset.subsample:
                    rng = np.random.default_rng(seed)
                    idx = rng.choice(len(y), config.dataset.subsample, replace=False)
                    x, y = x[idx], y[idx]

                x_train, x_valid, y_train, y_valid = train_test_split(
                    x, y, test_size=self.validation_fraction, random_state=seed
                )

                for library in config.libraries:
                    try:
                        runner = get_runner(library)
                    except ImportError:
                        if verbose:
                            print(f"  {library}: not available")
                        continue

                    if not runner.supports(config):
                        if verbose:
                            print(f"  {library}: config not supported")
                        continue

                    result = runner.run(
                        config=config,
                        x_train=x_train,
                        y_train=y_train,
                        x_valid=x_valid,
                        y_valid=y_valid,
                        seed=seed,
                    )
                    self._results.append(result)

                    if verbose:
                        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in result.metrics.items())
                        time_str = (
                            f", time={result.train_time_s:.3f}s" if result.train_time_s else ""
                        )
                        print(f"  {library} (seed={seed}): {metrics_str}{time_str}")

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if not self._results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self._results])

    def summary(self, group_by: list[str] | None = None) -> pd.DataFrame:
        """Aggregate results with mean ± std.

        Args:
            group_by: Columns to group by. Defaults to ["config", "library"].

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
        summary.columns = [f"{col}_{agg}" if agg else col for col, agg in summary.columns]

        return summary

    def report(self, precision: int = 4) -> str:
        """Generate a formatted report with metrics as columns and libraries as rows.

        Args:
            precision: Decimal precision for values.

        Returns:
            Formatted string report suitable for console and markdown.
        """
        df = self.to_dataframe()
        if df.empty:
            return "No results to display."

        lines: list[str] = []

        # Group by dataset + booster
        for (dataset, booster), group_df in df.groupby(["dataset", "booster"]):
            task = Task(group_df["task"].iloc[0])
            _ = primary_metric(task)  # Available for future use

            lines.append(f"\n## {dataset} ({booster})\n")

            # Get metric columns (exclude non-metrics)
            exclude = {"config", "library", "dataset", "booster", "seed", "task"}
            metric_cols = [c for c in group_df.columns if c not in exclude]

            # Aggregate by library
            agg = group_df.groupby("library")[metric_cols].agg(["mean", "std", "count"])

            # Build table data: library as index, metrics as columns
            table_data = []
            best_values: dict[str, tuple[str, float]] = {}  # metric -> (lib, value)

            for lib in agg.index:
                row = {"Library": lib}
                for col in metric_cols:
                    mean = agg.loc[lib, (col, "mean")]
                    std = agg.loc[lib, (col, "std")]
                    count = agg.loc[lib, (col, "count")]

                    if count == 1 or std == 0:
                        row[col] = f"{mean:.{precision}f}"
                    else:
                        row[col] = f"{mean:.{precision}f} ± {std:.{precision}f}"

                    # Track best
                    if col not in best_values:
                        best_values[col] = (lib, mean)
                    else:
                        current_best = best_values[col][1]
                        if is_lower_better(col):
                            if mean < current_best:
                                best_values[col] = (lib, mean)
                        elif mean > current_best:
                            best_values[col] = (lib, mean)

                table_data.append(row)

            # Bold the best values
            for row in table_data:
                lib = row["Library"]
                for col in metric_cols:
                    if best_values.get(col, (None,))[0] == lib:
                        row[col] = f"**{row[col]}**"

            # Convert to DataFrame for tabulate
            table_df = pd.DataFrame(table_data)
            table_df = table_df.set_index("Library")

            # Rename columns for display
            display_cols = {}
            for col in metric_cols:
                suffix = "↓" if is_lower_better(col) else "↑"
                display_cols[col] = f"{col} {suffix}"
            table_df = table_df.rename(columns=display_cols)

            lines.append(tabulate(table_df, headers="keys", tablefmt="pipe", stralign="right"))

            # Runtime summary
            if "train_time_s" in metric_cols:
                lines.append("\n**Runtime Statistics:**")
                for lib in agg.index:
                    mean_time = agg.loc[lib, ("train_time_s", "mean")]
                    count = agg.loc[lib, ("train_time_s", "count")]
                    total_time = mean_time * count
                    lines.append(f"- {lib}: avg {mean_time:.3f}s, total {total_time:.3f}s")

        return "\n".join(lines)

    def to_markdown(self, metric: str | None = None, precision: int = 4) -> str:
        """Generate markdown report.

        Args:
            metric: Specific metric to highlight. If None, uses task primary metric.
            precision: Decimal precision for values.

        Returns:
            Markdown formatted string.
        """
        return self.report(precision=precision)


def run_all_combinations(
    datasets: list[str] | None = None,
    booster_types: list[BoosterType] | None = None,
    libraries: list[str] | None = None,
    seeds: list[int] | None = None,
    training: dict | None = None,
    verbose: bool = True,
) -> BenchmarkSuite:
    """Run benchmarks for all valid combinations of datasets, boosters, and libraries.

    Args:
        datasets: Dataset names to use. Default: all.
        booster_types: Booster types to test. Default: [GBDT].
        libraries: Libraries to compare. Default: all available.
        seeds: Random seeds. Default: [42, 1379, 2716].
        training: Training config overrides.
        verbose: Print progress.

    Returns:
        BenchmarkSuite with results.
    """
    from boosters_eval.datasets import DATASETS, TrainingConfig
    from boosters_eval.runners import get_available_runners

    datasets = datasets or list(DATASETS.keys())
    booster_types = booster_types or [BoosterType.GBDT]
    libraries = libraries or get_available_runners()
    seeds = seeds or [42, 1379, 2716]
    training_config = TrainingConfig(**(training or {}))

    configs: list[BenchmarkConfig] = []

    for ds_name in datasets:
        if ds_name not in DATASETS:
            continue
        ds = DATASETS[ds_name]

        for bt in booster_types:
            configs.append(
                BenchmarkConfig(
                    name=f"{ds_name}/{bt.value}",
                    dataset=ds,
                    training=training_config,
                    booster_type=bt,
                    libraries=libraries,
                )
            )

    suite = BenchmarkSuite(configs, seeds=seeds)
    suite.run(verbose=verbose)
    return suite
