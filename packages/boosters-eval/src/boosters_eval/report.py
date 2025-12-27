"""Report generation with machine fingerprinting."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import psutil
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table
from scipy import stats

from boosters_eval.config import Task, TrainingConfig
from boosters_eval.metrics import LOWER_BETTER_METRICS, primary_metric
from boosters_eval.results import TIMING_METRICS, ResultCollection

if TYPE_CHECKING:
    pass

console = Console()


class MachineInfo(BaseModel):
    """Machine information for reproducibility."""

    model_config = ConfigDict(frozen=True)

    cpu: str
    cores: int
    memory_gb: float
    os: str
    python_version: str
    blas_backend: Optional[str] = None


def get_machine_info() -> MachineInfo:
    """Collect machine information."""
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

    # BLAS backend (best effort via numpy)
    blas_backend = None
    try:
        import numpy as np

        config = np.__config__.show()
        if isinstance(config, str):
            if "openblas" in config.lower():
                blas_backend = "OpenBLAS"
            elif "mkl" in config.lower():
                blas_backend = "MKL"
            elif "accelerate" in config.lower():
                blas_backend = "Accelerate"
    except Exception:  # noqa: BLE001
        pass

    return MachineInfo(
        cpu=cpu,
        cores=psutil.cpu_count(logical=False) or psutil.cpu_count() or 1,
        memory_gb=memory_gb,
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        blas_backend=blas_backend,
    )


class ReportMetadata(BaseModel):
    """Metadata for a benchmark report."""

    model_config = ConfigDict(frozen=True)

    title: str
    created_at: str
    git_sha: Optional[str] = None
    boosters_version: Optional[str] = None
    machine: MachineInfo
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
        return bool(result.pvalue < alpha)
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
        f"| CPU | {metadata.machine.cpu} |",
        f"| Cores | {metadata.machine.cores} |",
        f"| Memory | {metadata.machine.memory_gb} GB |",
        f"| OS | {metadata.machine.os} |",
        f"| Python | {metadata.machine.python_version} |",
    ]

    if metadata.machine.blas_backend:
        lines.append(f"| BLAS | {metadata.machine.blas_backend} |")
    if metadata.git_sha:
        lines.append(f"| Git SHA | {metadata.git_sha} |")
    if metadata.boosters_version:
        lines.append(f"| Boosters | {metadata.boosters_version} |")

    lines.extend([
        "",
        "## Configuration",
        "",
        f"- **Suite**: {metadata.suite_name}",
        f"- **Seeds**: {metadata.n_seeds}",
    ])

    # Add training parameters if available
    tc = metadata.training_config
    if tc is not None:
        lines.append(f"- **n_estimators**: {tc.n_estimators}")
        lines.append(f"- **max_depth**: {tc.max_depth}")
        lines.append(f"- **learning_rate**: {tc.learning_rate}")
        lines.append(f"- **reg_lambda (L2)**: {tc.reg_lambda}")
        lines.append(f"- **reg_alpha (L1)**: {tc.reg_alpha}")
        lines.append(f"- **growth_strategy**: {tc.growth_strategy.value}")
    if metadata.booster_types:
        lines.append(f"- **booster_types**: {', '.join(metadata.booster_types)}")

    lines.extend([
        "",
        "## Results",
        "",
    ])

    # Add task-grouped summary tables with best values highlighted
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


def format_results_terminal(results: ResultCollection) -> None:
    """Display results as Rich tables grouped by task type.

    Shows only the primary metric for each task type plus timing metrics.
    - Regression: rmse + timing
    - Binary: logloss + timing
    - Multiclass: mlogloss + timing

    Args:
        results: ResultCollection to display
    """
    summaries = results.summary_by_task()

    task_names = {
        Task.REGRESSION: "Regression Results",
        Task.BINARY: "Binary Classification Results",
        Task.MULTICLASS: "Multiclass Classification Results",
    }

    for task in [Task.REGRESSION, Task.BINARY, Task.MULTICLASS]:
        if task not in summaries:
            continue

        df = summaries[task]
        # Show only primary metric + timing (reduced from showing all metrics)
        metrics = [primary_metric(task)] + TIMING_METRICS

        # Create compact table for this task
        table = Table(title=task_names[task], show_header=True, header_style="bold")
        table.add_column("Dataset", style="cyan", no_wrap=True)
        table.add_column("Library", style="green")

        # Add only relevant metric columns
        for metric in metrics:
            mean_col = f"{metric}_mean"
            if mean_col in df.columns:
                table.add_column(metric, justify="right")

        # Find best values per dataset
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]

            # Find best for each metric
            best_libs: dict[str, str] = {}
            for metric in metrics:
                mean_col = f"{metric}_mean"
                if mean_col not in dataset_df.columns:
                    continue
                valid = dataset_df.dropna(subset=[mean_col])
                if valid.empty:
                    continue
                lower_better = metric in LOWER_BETTER_METRICS or metric.endswith("_time_s")
                best_idx = valid[mean_col].idxmin() if lower_better else valid[mean_col].idxmax()
                best_libs[metric] = str(valid.loc[best_idx, "library"])  # pyright: ignore[reportArgumentType]

            # Add rows
            for _, row in dataset_df.iterrows():
                import pandas as pd

                lib = str(row["library"])
                row_data = [str(dataset), lib]

                for metric in metrics:
                    mean_col = f"{metric}_mean"
                    if mean_col not in df.columns:
                        continue

                    mean_val = row[mean_col]
                    if pd.isna(mean_val):
                        row_data.append("-")
                    else:
                        val_str = f"{mean_val:.4f}"
                        # Highlight best
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
    git_sha = get_git_sha()

    # Get boosters version
    boosters_version = None
    try:
        import boosters

        boosters_version = getattr(boosters, "__version__", None)
    except ImportError:
        pass

    # Count seeds
    seeds = set()
    for result in results.results:
        seeds.add(result.seed)

    metadata = ReportMetadata(
        title=title,
        created_at=datetime.now().isoformat(),
        git_sha=git_sha,
        boosters_version=boosters_version,
        machine=machine,
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
