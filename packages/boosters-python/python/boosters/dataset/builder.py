"""DatasetBuilder: incremental dataset construction.

This is a thin Python layer extending the Rust `DatasetBuilder`.
It owns input normalization and feature parsing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast, overload

import numpy as np
from numpy.typing import NDArray

from boosters._boosters_rs import DatasetBuilder as _RustDatasetBuilder
from boosters._boosters_rs import Feature as _RustFeature
from boosters.types import (
    DenseValuesInput,
    LabelsInput,
    PandasDataFrame,
    SparseMatrix,
    WeightsInput,
    is_pandas_dataframe,
    is_sparse_matrix,
)

from .feature import Feature

__all__ = ["DatasetBuilder"]


def _validate_categorical_features(categorical_features: Sequence[int] | None, n_features: int) -> None:
    if categorical_features is None:
        return
    if not categorical_features:
        return

    idx = np.asarray(categorical_features, dtype=np.int64)
    min_idx = int(idx.min())
    max_idx = int(idx.max())
    if min_idx < 0 or max_idx >= n_features:
        raise ValueError(f"categorical indices out of range [0, {n_features - 1}]")


class DatasetBuilder(_RustDatasetBuilder):
    """Builder for constructing datasets with flexible feature types."""

    __slots__ = ("_n_samples",)

    _n_samples: int | None

    def __new__(cls) -> DatasetBuilder:  # noqa: PYI034
        """Create an empty builder with Python-side metadata."""
        self = cast(DatasetBuilder, super().__new__(cls))
        self._n_samples = None
        return self  # Necessary for proper typecasting.

    def _set_or_check_n_samples(self, n_samples: int) -> None:
        if self._n_samples is None:
            self._n_samples = int(n_samples)
            return
        if int(n_samples) != self._n_samples:
            raise ValueError(f"n_samples mismatch: expected {self._n_samples}, got {int(n_samples)}")

    def add_feature(self, feature: _RustFeature) -> DatasetBuilder:
        """Add a single feature column."""
        self._set_or_check_n_samples(feature.n_samples)
        _RustDatasetBuilder.add_feature(self, feature)
        return self

    @overload
    def add_features(
        self,
        features: NDArray[np.generic],
        *,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> DatasetBuilder: ...

    @overload
    def add_features(
        self,
        features: PandasDataFrame,
        *,
        categorical_features: Sequence[int] | None = None,
    ) -> DatasetBuilder: ...

    @overload
    def add_features(
        self,
        features: SparseMatrix,
        *,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> DatasetBuilder: ...

    def add_features(
        self,
        features: NDArray[np.generic] | PandasDataFrame | SparseMatrix,
        *,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> DatasetBuilder:
        """Add features from a numpy array, pandas DataFrame, or scipy sparse matrix."""
        if isinstance(features, np.ndarray):
            return self.add_features_from_array(
                features,
                feature_names=feature_names,
                categorical_features=categorical_features,
            )

        if is_pandas_dataframe(features):
            if feature_names is not None:
                raise ValueError("feature_names is not supported for pandas DataFrame inputs")
            return self.add_features_from_dataframe(features, categorical_features=categorical_features)

        if is_sparse_matrix(features):
            return self.add_features_from_sparse(
                features,
                feature_names=feature_names,
                categorical_features=categorical_features,
            )

        raise TypeError(
            f"expected numpy.ndarray, pandas.DataFrame, or scipy sparse matrix; got {type(features).__name__}"
        )

    def add_features_from_array(
        self,
        features: NDArray[np.generic],
        *,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> DatasetBuilder:
        """Add multiple dense features from a 2D numpy array."""
        if features.ndim != 2:
            raise ValueError(f"features must be 2D array, got {features.ndim}D")
        if features.shape[0] == 0:
            raise ValueError("features must have at least one sample")

        n_features = int(features.shape[1])
        _validate_categorical_features(categorical_features, n_features)
        cat_set = set(categorical_features or [])
        names = list(feature_names) if feature_names else [None] * n_features

        if len(names) != n_features:
            raise ValueError(f"feature_names length mismatch: expected {n_features}, got {len(names)}")

        for j in range(n_features):
            col = np.ascontiguousarray(features[:, j], dtype=np.float32)
            self.add_feature(Feature.from_dense(col, name=names[j], categorical=(j in cat_set)))

        return self

    def add_features_from_dataframe(
        self,
        df: PandasDataFrame,
        *,
        categorical_features: Sequence[int] | None = None,
    ) -> DatasetBuilder:
        """Add multiple features from a pandas DataFrame."""
        if not is_pandas_dataframe(df):
            raise TypeError(f"expected pandas DataFrame, got {type(df).__name__}")

        n_features = int(df.shape[1])
        _validate_categorical_features(categorical_features, n_features)
        cat_set = set(categorical_features or [])

        for i, col in enumerate(df.columns):
            cat_override: bool | None = (i in cat_set) if categorical_features is not None else None
            self.add_feature(Feature.from_dense(df[col], name=str(col), categorical=cat_override))

        return self

    def add_features_from_sparse(
        self,
        sparse: SparseMatrix,
        *,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> DatasetBuilder:
        """Add multiple sparse features from a scipy sparse matrix."""
        if not is_sparse_matrix(sparse):
            raise TypeError(f"expected scipy sparse matrix, got {type(sparse).__name__}")

        csc = sparse.tocsc()
        n_samples, n_features = csc.shape
        _validate_categorical_features(categorical_features, n_features)
        cat_set = set(categorical_features or [])
        names = list(feature_names) if feature_names else [None] * n_features

        if len(names) != n_features:
            raise ValueError(f"feature_names length mismatch: expected {n_features}, got {len(names)}")

        for j in range(n_features):
            start = int(csc.indptr[j])
            end = int(csc.indptr[j + 1])
            indices = np.ascontiguousarray(csc.indices[start:end], dtype=np.uint32)
            values = np.ascontiguousarray(csc.data[start:end], dtype=np.float32)
            self.add_feature(Feature.from_sparse(indices, values, n_samples, name=names[j], categorical=(j in cat_set)))

        return self

    def add_features_from_dict(
        self,
        features: Mapping[str, object],
        *,
        n_samples: int | None = None,
    ) -> DatasetBuilder:
        """Add features from a dict of column name -> data."""
        for name, data in features.items():
            if is_sparse_matrix(data):
                self.add_feature(Feature.from_sparse(cast(SparseMatrix, data), name=str(name)))
            elif isinstance(data, tuple) and len(data) == 2:
                if n_samples is None:
                    raise ValueError("n_samples is required for (indices, values) tuple format")
                indices, values = data
                self.add_feature(Feature.from_sparse(indices, values, n_samples, name=str(name)))
            else:
                self.add_feature(Feature.from_dense(cast(DenseValuesInput, data), name=str(name)))

        return self

    def labels(self, labels: LabelsInput) -> DatasetBuilder:
        """Set target labels (1D or 2D)."""
        if labels is None:
            return self

        arr = np.asarray(labels, dtype=np.float32)
        if arr.ndim == 1:
            if self._n_samples is not None and arr.shape[0] != self._n_samples:
                raise ValueError(f"labels shape mismatch: expected {self._n_samples}, got {arr.shape[0]}")
            _RustDatasetBuilder.labels_1d(self, np.ascontiguousarray(arr, dtype=np.float32))
            return self
        if arr.ndim != 2:
            raise ValueError(f"labels must be 1D or 2D array, got {arr.ndim}D")

        if self._n_samples is None:
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
            _RustDatasetBuilder.labels_2d(self, np.ascontiguousarray(arr, dtype=np.float32))
            return self

        if arr.shape[0] == self._n_samples:
            arr = arr.T
        elif arr.shape[1] != self._n_samples:
            raise ValueError(f"labels shape mismatch: no dimension matches {self._n_samples} samples")
        _RustDatasetBuilder.labels_2d(self, np.ascontiguousarray(arr, dtype=np.float32))
        return self

    def weights(self, weights: WeightsInput) -> DatasetBuilder:
        """Set sample weights."""
        if weights is None:
            return self

        arr = np.asarray(weights, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"weights must be 1D array, got {arr.ndim}D")
        if self._n_samples is not None and arr.shape[0] != self._n_samples:
            raise ValueError(f"weights shape mismatch: expected ({self._n_samples},), got {arr.shape}")
        _RustDatasetBuilder.weights(self, np.ascontiguousarray(arr, dtype=np.float32))
        return self
