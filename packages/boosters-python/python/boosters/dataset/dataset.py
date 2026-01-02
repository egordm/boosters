"""Dataset: training/prediction data container.

This extends the Rust `Dataset` binding with Python-friendly construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import boosters._boosters_rs as _rs

from .builder import DatasetBuilder, LabelsInput, WeightsInput

if TYPE_CHECKING:
    import pandas as pd

_RustDataset = _rs.Dataset  # type: ignore[attr-defined]

__all__ = ["Dataset"]


class Dataset(_RustDataset):
    """Dataset holding features, labels, and optional metadata."""

    was_converted: bool

    def __new__(  # noqa: PYI034
        cls,
        features: NDArray[np.generic] | pd.DataFrame | object,
        labels: LabelsInput = None,
        weights: WeightsInput = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> Dataset:
        """Create a Dataset from numpy arrays or a pandas DataFrame."""
        builder = DatasetBuilder()

        builder.add_features(
            features,  # type: ignore[arg-type]
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

        was_converted = True
        if isinstance(features, np.ndarray):
            was_converted = not (features.dtype == np.float32 and features.flags.c_contiguous)

        builder.labels(labels)
        builder.weights(weights)

        instance = builder.build()
        instance.was_converted = was_converted  # type: ignore[attr-defined]
        return instance  # type: ignore[return-value]

    def __init__(
        self,
        features: NDArray[np.generic] | pd.DataFrame | object,
        labels: LabelsInput = None,
        weights: WeightsInput = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> None:
        """No-op: all initialization is performed in `__new__`."""

    @classmethod
    def builder(cls) -> DatasetBuilder:
        """Create a builder for step-by-step dataset construction."""
        return DatasetBuilder()
