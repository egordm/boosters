"""Feature type for single feature columns.

This module provides `Feature`, a thin Python layer over the Rust `Feature`.
It owns all Python-side coercion (numpy / pandas / scipy) and keeps the Rust
bindings strict and fast.

Supported dense inputs:
- numpy.ndarray (1D)
- pandas.Series (auto-categorical detection)
- Python list/sequence

Supported sparse inputs:
- scipy sparse column vector (n,1) or (1,n)
- Raw indices + values + n_samples
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from boosters._boosters_rs import Feature as _RustFeature
from boosters.types import DenseValuesInput, PandasSeries, SparseMatrix, is_pandas_series, is_sparse_matrix

__all__ = ["Feature"]


def _series_to_dense(
    s: PandasSeries,
    *,
    categorical: bool | None,
) -> tuple[NDArray[np.float32], bool]:
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

    is_cat_dtype = isinstance(s.dtype, pd.CategoricalDtype) or str(s.dtype) == "category"

    if categorical is None:
        categorical = bool(
            is_cat_dtype or is_string_dtype(s.dtype) or is_object_dtype(s.dtype) or is_bool_dtype(s.dtype)
        )

    if categorical:
        codes = s.cat.codes if is_cat_dtype else s.astype("category").cat.codes
        return np.ascontiguousarray(codes.to_numpy(), dtype=np.float32), True

    try:
        return np.ascontiguousarray(s.to_numpy(), dtype=np.float32), False
    except (ValueError, TypeError) as e:
        raise TypeError(
            "Cannot convert pandas Series to float32. "
            "Pass categorical=True for string/object data, or convert manually."
        ) from e


def _sparse_to_indices_values(
    sparse: SparseMatrix,
) -> tuple[NDArray[np.uint32], NDArray[np.float32], int]:
    import scipy.sparse

    if not scipy.sparse.issparse(sparse):
        raise TypeError("expected scipy sparse matrix")

    shape = sparse.shape
    if len(shape) != 2:
        raise ValueError(f"expected 2D sparse matrix, got shape={shape}")

    if shape[1] == 1:
        csc = sparse.tocsc()
        n_samples = int(shape[0])
        start = int(csc.indptr[0])
        end = int(csc.indptr[1])
        indices = np.ascontiguousarray(csc.indices[start:end], dtype=np.uint32)
        values = np.ascontiguousarray(csc.data[start:end], dtype=np.float32)
        return indices, values, n_samples

    if shape[0] == 1:
        csr = sparse.tocsr()
        n_samples = int(shape[1])
        start = int(csr.indptr[0])
        end = int(csr.indptr[1])
        indices = np.ascontiguousarray(csr.indices[start:end], dtype=np.uint32)
        values = np.ascontiguousarray(csr.data[start:end], dtype=np.float32)
        return indices, values, n_samples

    raise ValueError(f"expected sparse column vector (n,1) or (1,n), got shape={shape}")


class Feature(_RustFeature):
    """A single feature column with metadata.

    Use:
    - `Feature.from_dense(values, ...)`
    - `Feature.from_sparse(...)`

    Note: This class is intended to extend the Rust binding and provide
    Python-friendly constructors.
    """

    @classmethod
    def from_dense(
        cls,
        values: DenseValuesInput,
        *,
        name: str | None = None,
        categorical: bool | None = None,
    ) -> Feature:
        """Create a dense feature from an array/Series/list.

        `categorical=None` enables pandas dtype auto-detection.
        """
        is_cat: bool

        if is_pandas_series(values):
            arr, is_cat = _series_to_dense(values, categorical=categorical)
        elif isinstance(values, np.ndarray):
            arr = values
            if arr.ndim == 2 and 1 in arr.shape:
                arr = arr.reshape(-1)
            if arr.ndim != 1:
                raise ValueError(f"expected 1D array, got {arr.ndim}D")
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            is_cat = bool(categorical) if categorical is not None else False
        elif isinstance(values, Sequence):
            arr = np.ascontiguousarray(values, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"expected 1D sequence, got {arr.ndim}D after conversion")
            is_cat = bool(categorical) if categorical is not None else False
        else:
            raise TypeError(f"from_dense() expected ndarray, Series, or list; got {type(values).__name__}")

        # We intentionally call the Rust constructor here.
        return cast(Feature, cls.dense(arr, name=name, categorical=is_cat))

    @classmethod
    def from_sparse(
        cls,
        indices: NDArray[np.integer] | Sequence[int] | SparseMatrix,
        values: NDArray[np.floating] | Sequence[float] | None = None,
        n_samples: int | None = None,
        *,
        name: str | None = None,
        default: float = 0.0,
        categorical: bool = False,
    ) -> Feature:
        """Create a sparse feature.

        Accepts either:
        - a scipy sparse column vector, or
        - (`indices`, `values`, `n_samples`).
        """
        if is_sparse_matrix(indices):
            if values is not None or n_samples is not None:
                raise TypeError("from_sparse(scipy_sparse) takes no values/n_samples arguments")
            idx, vals, n = _sparse_to_indices_values(indices)
            return cast(Feature, cls.sparse(idx, vals, n, name=name, default=default, categorical=categorical))

        if values is None or n_samples is None:
            raise TypeError("from_sparse(indices, values, n_samples, ...) requires all three arguments")

        idx = np.ascontiguousarray(indices, dtype=np.uint32)
        vals = np.ascontiguousarray(values, dtype=np.float32)

        if idx.ndim != 1:
            raise ValueError(f"indices must be 1D, got {idx.ndim}D")
        if vals.ndim != 1:
            raise ValueError(f"values must be 1D, got {vals.ndim}D")
        if len(idx) != len(vals):
            raise ValueError(f"indices and values must have same length: {len(idx)} vs {len(vals)}")

        return cast(
            Feature,
            cls.sparse(idx, vals, int(n_samples), name=name, default=default, categorical=categorical),
        )
