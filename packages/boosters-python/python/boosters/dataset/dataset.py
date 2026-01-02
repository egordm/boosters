"""Dataset: training/prediction data container.

This extends the Rust `Dataset` binding with Python-friendly construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, overload

import numpy as np
from numpy.typing import NDArray

from boosters._boosters_rs import Dataset as _RustDataset
from boosters.types import (
    LabelsInput,
    PandasDataFrame,
    SparseMatrix,
    WeightsInput,
    is_pandas_dataframe,
    is_sparse_matrix,
)

from .builder import DatasetBuilder

__all__ = ["Dataset"]


class Dataset(_RustDataset):
    """Dataset holding features, labels, and optional metadata."""

    was_converted: bool

    @overload
    def __new__(
        cls,
        features: NDArray[np.generic],
        labels: LabelsInput = None,
        weights: WeightsInput = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> Dataset: ...

    @overload
    def __new__(
        cls,
        features: PandasDataFrame,
        labels: LabelsInput = None,
        weights: WeightsInput = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> Dataset: ...

    @overload
    def __new__(
        cls,
        features: SparseMatrix,
        labels: LabelsInput = None,
        weights: WeightsInput = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> Dataset: ...

    def __new__(  # noqa: PYI034
        cls,
        features: object,
        labels: LabelsInput = None,
        weights: WeightsInput = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> Dataset:
        """Create a Dataset from numpy arrays or a pandas DataFrame."""
        builder = DatasetBuilder()

        if isinstance(features, np.ndarray):
            builder.add_features(features, feature_names=feature_names, categorical_features=categorical_features)
        elif is_pandas_dataframe(features):
            if feature_names is not None:
                raise ValueError("feature_names is not supported for pandas DataFrame inputs")
            builder.add_features(features, categorical_features=categorical_features)
        elif is_sparse_matrix(features):
            builder.add_features(features, feature_names=feature_names, categorical_features=categorical_features)
        else:
            raise TypeError(
                f"expected numpy.ndarray, pandas.DataFrame, or scipy sparse matrix; got {type(features).__name__}"
            )

        was_converted = True
        if isinstance(features, np.ndarray):
            was_converted = not (features.dtype == np.float32 and features.flags.c_contiguous)

        builder.labels(labels)
        builder.weights(weights)

        instance = cast(Dataset, builder.build())
        instance.was_converted = was_converted
        return instance

    def __init__(
        self,
        features: object,
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
