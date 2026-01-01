"""Base report utilities: machine info, library versions, metadata."""

from __future__ import annotations

import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import psutil
from pydantic import BaseModel, ConfigDict
from scipy import stats

from boosters_eval.config import TrainingConfig


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
    boosters: str | None = None
    xgboost: str | None = None
    lightgbm: str | None = None
    numpy: str | None = None


class ReportMetadata(BaseModel):
    """Metadata for a benchmark report."""

    model_config = ConfigDict(frozen=True)

    title: str
    created_at: str
    git_sha: str | None = None
    machine: MachineInfo
    library_versions: LibraryVersions
    suite_name: str
    n_seeds: int
    # Training configuration for reproducibility
    training_config: TrainingConfig | None = None
    booster_types: list[str] | None = None


def get_machine_info() -> MachineInfo:
    """Collect machine information (simplified)."""
    # CPU info - try platform.processor first, fallback for Linux
    cpu = platform.processor()
    if not cpu and platform.system() == "Linux":
        try:
            with Path("/proc/cpuinfo").open() as f:
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
        import numpy as np

        versions = versions.model_copy(update={"numpy": np.__version__})
    except ImportError:
        pass

    return versions


def get_git_sha() -> str | None:
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
    except Exception:
        return False


def create_metadata(
    seeds: set[int],
    suite_name: str,
    title: str,
    training_config: TrainingConfig | None = None,
    booster_types: list[str] | None = None,
) -> ReportMetadata:
    """Create report metadata from results."""
    return ReportMetadata(
        title=title,
        created_at=datetime.now(UTC).isoformat(),
        git_sha=get_git_sha(),
        machine=get_machine_info(),
        library_versions=get_library_versions(),
        suite_name=suite_name,
        n_seeds=len(seeds),
        training_config=training_config,
        booster_types=booster_types,
    )
