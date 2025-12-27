"""Report generation with machine fingerprinting."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from pydantic import BaseModel, ConfigDict
from scipy import stats

from boosters_eval.metrics import LOWER_BETTER_METRICS
from boosters_eval.results import ResultCollection


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
        _, p_value = stats.ttest_ind(values1, values2, equal_var=False)
        return p_value < alpha
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
        "",
        "## Results",
        "",
    ])

    # Add summary table with significance highlighting
    summary = results.summary()
    if not summary.empty:
        lines.append(summary.to_markdown(index=False))

    lines.extend([
        "",
        "## Reproducing",
        "",
        "```bash",
        f"boosters-eval {metadata.suite_name.lower()}",
        "```",
        "",
    ])

    if require_significance:
        lines.extend([
            "---",
            "",
            "*Winners bolded only when statistically significant (p < 0.05).*",
        ])

    return "\n".join(lines)


def generate_report(
    results: ResultCollection,
    suite_name: str,
    output_path: Optional[Path] = None,
    title: str = "Benchmark Report",
) -> str:
    """Generate a benchmark report.

    Args:
        results: Benchmark results
        suite_name: Name of the suite that was run
        output_path: Optional path to save the report
        title: Report title

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
    )

    report = render_report(results, metadata)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report
