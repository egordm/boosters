"""Report generation with machine fingerprinting."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import psutil
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table
from scipy import stats

from boosters_eval.config import BoosterType, Task, TrainingConfig
from boosters_eval.metrics import LOWER_BETTER_METRICS, primary_metric
from boosters_eval.results import TIMING_METRICS, ResultCollection

if TYPE_CHECKING:
    pass

console = Console()


class MachineInfo(BaseModel):
    """Machine information for reproducibility.

    Simplified to show just the essentials:
    - Machine type (CPU model)
    - Core count and memory for context
    """

    model_config = ConfigDict(frozen=True)

    cpu: str
    cores: int
    memory_gb: float
    os: str


class LibraryVersions(BaseModel):
    """Library versions for reproducibility."""

    model_config = ConfigDict(frozen=True)

    python: str
    boosters: Optional[str] = None
    xgboost: Optional[str] = None
    lightgbm: Optional[str] = None
    numpy: Optional[str] = None


def get_machine_info() -> MachineInfo:
    """Collect machine information (simplified)."""
    # CPU info - try platform.processor first, fallback for Linux
    cpu = platform.processor()
    if not cpu and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu = line.split(":")[1].strip()
                        break
        except (OSError, IndexError):
            pass
    cpu = cpu or "Unknown"

    # Memory
    memory_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    return MachineInfo(
        cpu=cpu,
        cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        memory_gb=memory_gb,
        os=f"{platform.system()} {platform.release()}",
    )


def get_library_versions() -> LibraryVersions:
    """Collect versions of key libraries."""
    versions = LibraryVersions(python=platform.python_version())

    # boosters
    try:
        import boosters

        versions = versions.model_copy(update={"boosters": getattr(boosters, "__version__", "unknown")})
    except ImportError:
        pass

    # xgboost
    try:
        import xgboost

        versions = versions.model_copy(update={"xgboost": getattr(xgboost, "__version__", "unknown")})
    except ImportError:
        pass

    # lightgbm
    try:
        import lightgbm

        versions = versions.model_copy(update={"lightgbm": getattr(lightgbm, "__version__", "unknown")})
    except ImportError:
        pass

    # numpy
    try:
        import numpy

        versions = versions.model_copy(update={"numpy": numpy.__version__})
    except ImportError:
        pass

    return versions


class ReportMetadata(BaseModel):
    """Metadata for a benchmark report."""

    model_config = ConfigDict(frozen=True)

    title: str
    created_at: str
    git_sha: Optional[str] = None
    machine: MachineInfo
    library_versions: LibraryVersions
    suite_name: str
    n_seeds: int
    # Training configuration for reproducibility
    training_config: Optional[TrainingConfig] = None
    booster_types: Optional[list[str]] = None


def get_git_sha() -> Optional[str]:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_significant(
    values1: list[float],
    values2: list[float],
    alpha: float = 0.05,
) -> bool:
    """Test if two sets of values are significantly different using Welch's t-test.

    Args:
        values1: First set of metric values (e.g., from boosters)
        values2: Second set of metric values (e.g., from xgboost)
        alpha: Significance level (default 0.05)

    Returns:
        True if difference is statistically significant (p < alpha)
    """
    # Need at least 2 samples per group
    if len(values1) < 2 or len(values2) < 2:
        return False

    # All identical values - no variance
    if len(set(values1)) == 1 and len(set(values2)) == 1:
        return values1[0] != values2[0]

    try:
        result = stats.ttest_ind(values1, values2, equal_var=False)
        return bool(result.pvalue < alpha)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False


def render_report(
    results: ResultCollection,
    metadata: ReportMetadata,
    require_significance: bool = True,
) -> str:
    """Render a benchmark report as markdown.

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
        if has_tree_based or not booster_types:  # Show if no booster types specified
            lines.append(f"- **max_depth**: {tc.max_depth}")
            lines.append(f"- **growth_strategy**: {tc.growth_strategy.value}")
            lines.append(f"- **max_bins**: {tc.max_bins}")
            lines.append(f"- **min_samples_leaf**: {tc.min_samples_leaf}")

        # Regularization (applies to all)
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

    # Add dataset-grouped summary tables with best values highlighted
    formatted = results.to_markdown(highlight_best=True)
    lines.append(formatted)

    lines.extend([
        "",
        "## Reproducing",
        "",
        "```bash",
        f"boosters-eval {metadata.suite_name.lower()}",
        "```",
        "",
    ])

    lines.extend([
        "---",
        "",
        "*Best values per metric are **bolded**. Lower is better for loss/time metrics.*",
    ])

    return "\n".join(lines)


def format_results_terminal(
    results: ResultCollection,
    *,
    require_significance: bool = True,
) -> None:
    """Display results as Rich tables grouped by dataset.

    Uses the same data and logic as the markdown formatter, just with Rich styling.
    Shows all metrics relevant to each dataset's task type.

    Args:
        results: ResultCollection to display
        require_significance: Only highlight winners if statistically significant
    """
    from boosters_eval.results import TASK_METRICS, TIMING_METRICS, MEMORY_METRICS

    summaries = results.summary_by_dataset()

    if not summaries:
        console.print("[yellow]No results to display.[/yellow]")
        return

    for dataset in sorted(summaries.keys()):
        df = summaries[dataset]

        # Get task type for this dataset
        task = None
        for r in results.results:
            if r.dataset_name == dataset:
                task = Task(r.task)
                break
        if task is None:
            continue

        # Get primary metric for this dataset
        primary = results._get_primary_metric_for_dataset(dataset)

        # All metrics relevant to this dataset
        metrics = TASK_METRICS[task] + TIMING_METRICS + MEMORY_METRICS

        # Create table for this dataset
        title = f"{dataset} ({task.value}, primary: {primary})"
        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Booster", style="cyan", no_wrap=True)
        table.add_column("Library", style="green")

        # Add metric columns that are present
        present_metrics = [m for m in metrics if f"{m}_mean" in df.columns]
        for metric in present_metrics:
            table.add_column(metric, justify="right")

        # Find best values per booster
        for booster in df["booster"].unique():
            booster_df = df[df["booster"] == booster]

            # Determine best library for each metric
            best_libs: dict[str, str] = {}
            for metric in present_metrics:
                mean_col = f"{metric}_mean"
                valid = booster_df.dropna(subset=[mean_col])
                if len(valid) < 2:
                    continue
                lower_better = metric in LOWER_BETTER_METRICS or metric.endswith("_time_s")
                best_idx = valid[mean_col].idxmin() if lower_better else valid[mean_col].idxmax()
                best_libs[metric] = str(valid.loc[best_idx, "library"])  # pyright: ignore[reportArgumentType]

            # Add rows
            for _, row in booster_df.iterrows():
                import pandas as pd

                lib = str(row["library"])
                row_data = [str(booster), lib]

                for metric in present_metrics:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"

                    mean_val = row[mean_col]
                    std_val = row.get(std_col, np.nan) if std_col in row.index else np.nan

                    if pd.isna(mean_val):
                        row_data.append("-")
                    else:
                        if pd.isna(std_val) or std_val == 0:
                            val_str = f"{mean_val:.4f}"
                        else:
                            val_str = f"{mean_val:.4f}±{std_val:.4f}"

                        # Highlight best (simplified - use same logic as markdown)
                        if best_libs.get(metric) == lib:
                            val_str = f"[bold green]{val_str}[/bold green]"
                        row_data.append(val_str)

                table.add_row(*row_data)

        console.print(table)
        console.print()


def generate_report(
    results: ResultCollection,
    suite_name: str,
    output_path: Optional[Path] = None,
    title: str = "Benchmark Report",
    *,
    training_config: Optional[TrainingConfig] = None,
    booster_types: Optional[list[str]] = None,
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
        created_at=datetime.now().isoformat(),
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
