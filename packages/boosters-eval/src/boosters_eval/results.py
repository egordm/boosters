"""Benchmark result model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from boosters_eval.datasets import Task


class BenchmarkResult(BaseModel):
    """Results from a single benchmark run."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    config_name: str
    library: str
    seed: int
    task: Task
    booster_type: str
    dataset_name: str
    metrics: dict[str, float]
    train_time_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for DataFrame."""
        result = {
            "config": self.config_name,
            "library": self.library,
            "dataset": self.dataset_name,
            "booster": self.booster_type,
            "seed": self.seed,
            "task": self.task.value,
        }
        result.update(self.metrics)
        if self.train_time_s is not None:
            result["train_time_s"] = self.train_time_s
        return result
