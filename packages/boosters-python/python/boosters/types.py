"""Common type aliases and type guards for boosters.

This module centralizes typing helpers so we avoid duplicating type aliases
across the codebase.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Tree Building Types
# =============================================================================
# GrowthStrategy is now a proper enum exported from the Rust core.
# Import it from config module for backwards compatibility.
from boosters._boosters_rs import GrowthStrategy

if TYPE_CHECKING:
    import pandas as pd

    type PandasSeries = pd.Series
    type PandasDataFrame = pd.DataFrame
else:
    type PandasSeries = object
    type PandasDataFrame = object


class SparseCSCMatrix(Protocol):
    """CSC-form sparse matrix view used by our helpers.

    This is a minimal structural type (Protocol) to avoid tight coupling to
    SciPy's exact classes and to keep type checking stable.
    """

    shape: tuple[int, int]
    indptr: NDArray[np.integer]
    indices: NDArray[np.integer]
    data: NDArray[np.floating]


class SparseCSRMatrix(Protocol):
    """CSR-form sparse matrix view used by our helpers."""

    shape: tuple[int, int]
    indptr: NDArray[np.integer]
    indices: NDArray[np.integer]
    data: NDArray[np.floating]


class SparseMatrix(Protocol):
    """Sparse matrix input supported by boosters.

    This mirrors the subset of SciPy sparse matrix APIs we rely on.
    """

    shape: tuple[int, int]

    def tocsc(self) -> SparseCSCMatrix:
        """Convert to CSC for efficient column access."""
        ...

    def tocsr(self) -> SparseCSRMatrix:
        """Convert to CSR for efficient row access."""
        ...


def is_pandas_series(obj: object) -> TypeGuard[PandasSeries]:
    """Return True if obj is a pandas Series (and pandas is installed)."""
    try:
        import pandas as pd

        return isinstance(obj, pd.Series)
    except ImportError:
        return False


def is_pandas_dataframe(obj: object) -> TypeGuard[PandasDataFrame]:
    """Return True if obj is a pandas DataFrame (and pandas is installed)."""
    try:
        import pandas as pd

        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False


def is_sparse_matrix(obj: object) -> TypeGuard[SparseMatrix]:
    """Return True if obj is a scipy.sparse matrix/array (and scipy is installed)."""
    try:
        import scipy.sparse

        return scipy.sparse.issparse(obj)
    except ImportError:
        return False


type DenseValuesInput = NDArray[np.generic] | PandasSeries | Sequence[float]
type SparseValuesInput = NDArray[np.floating] | Sequence[float]
type SparseIndicesInput = NDArray[np.integer] | Sequence[int]
type LabelsInput = NDArray[np.floating] | Sequence[float] | None
type WeightsInput = NDArray[np.floating] | Sequence[float] | None
type FeaturesInput = NDArray[np.generic] | PandasDataFrame | SparseMatrix


__all__ = [
    "DenseValuesInput",
    "FeaturesInput",
    "GrowthStrategy",
    "LabelsInput",
    "PandasDataFrame",
    "PandasSeries",
    "SparseCSCMatrix",
    "SparseCSRMatrix",
    "SparseIndicesInput",
    "SparseMatrix",
    "SparseValuesInput",
    "WeightsInput",
    "is_pandas_dataframe",
    "is_pandas_series",
    "is_sparse_matrix",
]
