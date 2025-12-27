"""Benchmark result models and collection."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from boosters_eval.config import Task


class BenchmarkResult(BaseModel):
    """Results from a single benchmark run."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    config_name: str
    library: str
    seed: int
    task: str
    booster_type: str
    dataset_name: str
    metrics: dict[str, float]
    train_time_s: float | None = None
    predict_time_s: float | None = None
    peak_memory_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for DataFrame."""
        result: dict[str, Any] = {
            "config": self.config_name,
            "library": self.library,
            "dataset": self.dataset_name,
            "booster": self.booster_type,
            "seed": self.seed,
            "task": self.task,
        }
        result.update(self.metrics)
        if self.train_time_s is not None:
            result["train_time_s"] = self.train_time_s
        if self.predict_time_s is not None:
            result["predict_time_s"] = self.predict_time_s
        if self.peak_memory_mb is not None:
            result["peak_memory_mb"] = self.peak_memory_mb
        return result


class BenchmarkError(BaseModel):
    """Error from a failed benchmark run."""

    model_config = ConfigDict(frozen=True)

    config_name: str
    library: str
    seed: int
    error_type: str
    error_message: str
    dataset_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary."""
        return {
            "config": self.config_name,
            "library": self.library,
            "dataset": self.dataset_name,
            "seed": self.seed,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def derive_seed(base_seed: int, config_name: str, library: str) -> int:
    """Derive a deterministic seed from base seed, config, and library.

    Formula: hash((base_seed, config_name, library)) % 2^32
    """
    key = f"{base_seed}:{config_name}:{library}"
    hash_bytes = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(hash_bytes[:4], "little")


class ResultCollection:
    """Collection of benchmark results with aggregation and export."""

    def __init__(
        self,
        results: list[BenchmarkResult] | None = None,
        errors: list[BenchmarkError] | None = None,
    ) -> None:
        """Initialize result collection."""
        self.results = results or []
        self.errors = errors or []

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a result to the collection."""
        self.results.append(result)

    def add_error(self, error: BenchmarkError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.results])

    def errors_dataframe(self) -> pd.DataFrame:
        """Convert errors to pandas DataFrame."""
        if not self.errors:
            return pd.DataFrame()
        return pd.DataFrame([e.to_dict() for e in self.errors])

    def filter(
        self,
        *,
        library: str | list[str] | None = None,
        dataset: str | list[str] | None = None,
        task: str | list[str] | None = None,
    ) -> ResultCollection:
        """Filter results by criteria."""
        filtered = self.results

        if library is not None:
            libs = [library] if isinstance(library, str) else library
            filtered = [r for r in filtered if r.library in libs]

        if dataset is not None:
            datasets = [dataset] if isinstance(dataset, str) else dataset
            filtered = [r for r in filtered if r.dataset_name in datasets]

        if task is not None:
            tasks = [task] if isinstance(task, str) else task
            filtered = [r for r in filtered if r.task in tasks]

        return ResultCollection(results=filtered, errors=self.errors)

    def summary(self, group_by: list[str] | None = None) -> pd.DataFrame:
        """Aggregate results with mean ± std.

        Args:
            group_by: Columns to group by. Defaults to ["dataset", "booster", "library"].

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
        summary.columns = [  # pyright: ignore[reportAttributeAccessIssue]
            f"{col}_{agg}" if agg else col for col, agg in summary.columns
        ]

        return pd.DataFrame(summary)

    def to_markdown(self, precision: int = 4) -> str:
        """Generate markdown table from results.

        Args:
            precision: Decimal precision for values.

        Returns:
            Markdown formatted string.
        """
        df = self.to_dataframe()
        if df.empty:
            return "No results to display."

        summary = self.summary()
        return summary.to_markdown(index=False, floatfmt=f".{precision}f")

    def to_json(self) -> str:
        """Export results to JSON string."""
        data = {
            "results": [r.model_dump() for r in self.results],
            "errors": [e.model_dump() for e in self.errors],
        }
        return json.dumps(data, indent=2)

    def to_csv(self) -> str:
        """Export results to CSV string."""
        df = self.to_dataframe()
        return df.to_csv(index=False)

    @classmethod
    def from_json(cls, json_str: str) -> ResultCollection:
        """Load results from JSON string."""
        data = json.loads(json_str)
        results = [BenchmarkResult(**r) for r in data.get("results", [])]
        errors = [BenchmarkError(**e) for e in data.get("errors", [])]
        return cls(results=results, errors=errors)

    def __len__(self) -> int:
        """Return number of results."""
        return len(self.results)
