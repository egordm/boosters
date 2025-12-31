"""Data types for gradient boosting.

This module provides dataset wrappers for training and evaluation.

Types:
    - Dataset: Training/prediction dataset with features, labels, weights
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from boosters._boosters_rs import Dataset as _RustDataset

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

__all__: list[str] = [
    "Dataset",
]


# =============================================================================
# Type definitions for array-like inputs
# =============================================================================


@runtime_checkable
class DataFrameProtocol(Protocol):
    """Protocol for DataFrame-like objects (pandas, polars, etc.)."""

    @property
    def columns(self) -> Any:
        """Column names."""
        ...

    @property
    def dtypes(self) -> Any:
        """Column dtypes."""
        ...

    @property
    def values(self) -> NDArray[Any]:
        """Underlying numpy array."""
        ...

    def to_numpy(self) -> NDArray[Any]:
        """Convert to numpy array."""
        ...


@runtime_checkable
class SparseMatrixProtocol(Protocol):
    """Protocol for scipy sparse matrices."""

    def toarray(self) -> NDArray[Any]:
        """Convert to dense array."""
        ...


# Type alias for features input
FeaturesInput = NDArray[Any] | DataFrameProtocol | SparseMatrixProtocol | Sequence[Sequence[float]]
LabelsInput = NDArray[Any] | Sequence[float] | None
WeightsInput = NDArray[Any] | Sequence[float] | None
GroupsInput = NDArray[Any] | Sequence[int] | None


# =============================================================================
# DataFrame extraction utilities
# =============================================================================


def _is_pandas_dataframe(obj: object) -> bool:
    """Check if object is a pandas DataFrame."""
    try:
        import pandas as pd  # noqa: PLC0415 (lazy import for optional dependency)

        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False


def _is_polars_dataframe(obj: object) -> bool:
    """Check if object is a polars DataFrame."""
    try:
        import polars as pl  # noqa: PLC0415 (lazy import for optional dependency)

        return isinstance(obj, (pl.DataFrame, pl.LazyFrame))
    except ImportError:
        return False


def _extract_pandas_dataframe(
    df: pd.DataFrame,
) -> tuple[NDArray[np.float32], list[str], list[int]]:
    """Extract features from a pandas DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Tuple of (features_array, feature_names, categorical_indices).
    """
    import pandas as pd  # noqa: PLC0415 (lazy import for optional dependency)

    # Get feature names
    feature_names = [str(c) for c in df.columns]

    # Detect categorical columns
    categorical_indices: list[int] = []
    for i, dtype in enumerate(df.dtypes):
        if isinstance(dtype, pd.CategoricalDtype) or str(dtype) == "category":
            categorical_indices.append(i)

    # Convert to numpy array
    values = df.to_numpy()

    # Ensure float32 and C-contiguous
    features = np.ascontiguousarray(values, dtype=np.float32)

    return features, feature_names, categorical_indices


def _extract_polars_dataframe(
    df: pl.DataFrame | pl.LazyFrame,
) -> tuple[NDArray[np.float32], list[str], list[int]]:
    """Extract features from a polars DataFrame.

    Args:
        df: Input polars DataFrame or LazyFrame.

    Returns:
        Tuple of (features_array, feature_names, categorical_indices).
    """
    import polars as pl  # noqa: PLC0415 (lazy import for optional dependency)

    # Collect if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Get feature names
    feature_names = df.columns

    # Detect categorical columns (polars uses Categorical and Enum)
    categorical_indices: list[int] = []
    for i, dtype in enumerate(df.dtypes):
        if dtype in (pl.Categorical, pl.Enum):
            categorical_indices.append(i)

    # Convert to numpy
    features = df.to_numpy().astype(np.float32, order="C")

    return features, feature_names, categorical_indices


def _extract_numpy_array(arr: NDArray[Any]) -> tuple[NDArray[np.float32], bool]:
    """Extract and validate a numpy array.

    Args:
        arr: Input array.

    Returns:
        Tuple of (C-contiguous float32 array, was_converted flag).

    Raises:
        ValueError: If array is not 2D or has no samples.
    """
    if arr.ndim != 2:
        raise ValueError(f"features must be 2D array, got {arr.ndim}D")

    if arr.shape[0] == 0:
        raise ValueError("features must have at least one sample")

    # Check if already in optimal format
    if arr.dtype == np.float32 and arr.flags.c_contiguous:
        return arr, False

    return np.ascontiguousarray(arr, dtype=np.float32), True


def _is_sparse_matrix(obj: object) -> bool:
    """Check if object is a scipy sparse matrix."""
    try:
        import scipy.sparse  # noqa: PLC0415 (lazy import for optional dependency)

        return scipy.sparse.issparse(obj)
    except ImportError:
        return False


# =============================================================================
# Dataset class
# =============================================================================


class Dataset(_RustDataset):
    """Dataset holding features, labels, and optional metadata.

    This class extends the Rust Dataset with Python-friendly constructors.
    Data is converted to C-contiguous float32 arrays for efficient processing.

    Supports multiple input types:
    - NumPy arrays (preferred, zero-copy when possible)
    - Pandas DataFrames (auto-detects feature names and categorical columns)
    - Polars DataFrames (auto-detects feature names and categorical columns)
    - Any array-like that can be converted to NumPy

    Args:
        features: 2D array-like of shape (n_samples, n_features).
            Accepts NumPy arrays, pandas/polars DataFrames, or any array-like.
        labels: Optional 1D array of shape (n_samples,).
        weights: Optional 1D array of sample weights (n_samples,).
        groups: Optional 1D array of group labels for ranking.
        feature_names: Optional list of feature names.
            Auto-detected from DataFrames if not provided.
        categorical_features: Optional list of categorical feature indices.
            Auto-detected from DataFrame categorical dtypes if not provided.

    Raises:
        ValueError: If data is invalid (shape mismatch, Inf values, etc.).
        TypeError: If data types are unsupported.
        NotImplementedError: If sparse matrices are passed (not yet supported).

    Example:
        >>> import numpy as np
        >>> from boosters import Dataset
        >>>
        >>> # From NumPy array
        >>> X = np.random.rand(100, 10).astype(np.float32)
        >>> y = np.random.rand(100).astype(np.float32)
        >>> dataset = Dataset(X, y)
        >>>
        >>> # From pandas DataFrame
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> dataset = Dataset(df)
        >>> print(dataset.feature_names)  # ['a', 'b']
    """

    # Python-only attributes
    _groups: NDArray[np.int32] | None
    _was_converted: bool

    # Dataset constructor is intentionally complex to handle multiple input types
    # (numpy, pandas, polars, lists) with validation and type detection
    def __new__(  # noqa: C901, PLR0912, PLR0915, PYI034
        cls,
        features: FeaturesInput,
        labels: LabelsInput = None,
        weights: WeightsInput = None,
        groups: GroupsInput = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> Dataset:
        """Create a new Dataset instance."""
        # Handle sparse matrices
        if _is_sparse_matrix(features):
            raise NotImplementedError(
                "Sparse matrices are not yet supported. Convert to dense array with "
                ".toarray() or use a DataFrame. Sparse support is planned."
            )

        # Extract features based on type
        detected_cats: list[int] = []
        detected_names: list[str] | None = None
        was_converted = False

        if _is_pandas_dataframe(features):
            features_arr, detected_names, detected_cats = _extract_pandas_dataframe(
                features  # type: ignore[arg-type]
            )
            was_converted = True
        elif _is_polars_dataframe(features):
            features_arr, detected_names, detected_cats = _extract_polars_dataframe(
                features  # type: ignore[arg-type]
            )
            was_converted = True
        elif isinstance(features, np.ndarray):
            features_arr, was_converted = _extract_numpy_array(features)
        else:
            # Try to convert to numpy array (handles lists, tuples, etc.)
            try:
                tmp_arr = np.asarray(features, dtype=np.float32)
                features_arr, _ = _extract_numpy_array(tmp_arr)
                was_converted = True
            except (ValueError, TypeError) as e:
                type_name = type(features).__name__
                raise TypeError(
                    f"Cannot convert {type_name} to feature array. "
                    f"Expected numpy array, pandas/polars DataFrame, or array-like."
                ) from e

        # Use detected names if not provided
        if feature_names is None and detected_names is not None:
            feature_names = detected_names

        # Merge categorical features (user-provided + auto-detected)
        all_cats: list[int] = list(categorical_features) if categorical_features else []
        for cat in detected_cats:
            if cat not in all_cats:
                all_cats.append(cat)
        all_cats.sort()

        # Validate categorical feature indices
        n_features = features_arr.shape[1]
        for cat_idx in all_cats:
            if cat_idx < 0 or cat_idx >= n_features:
                raise ValueError(f"categorical feature index {cat_idx} is out of range [0, {n_features - 1}]")

        # Convert labels if provided
        labels_2d: NDArray[np.float32] | None = None
        if labels is not None:
            labels_arr = np.asarray(labels, dtype=np.float32)
            if labels_arr.ndim == 1:
                labels_2d = labels_arr.reshape(1, -1)
            elif labels_arr.ndim == 2:
                labels_2d = np.ascontiguousarray(labels_arr, dtype=np.float32)
            else:
                raise ValueError(f"labels must be 1D or 2D array, got {labels_arr.ndim}D")

            if labels_2d.shape[1] != features_arr.shape[0]:
                raise ValueError(
                    f"labels shape mismatch: expected {features_arr.shape[0]} samples, got {labels_2d.shape[1]}"
                )
            if not np.all(np.isfinite(labels_2d)):
                raise ValueError("labels contain NaN or Inf values")

        # Convert weights if provided
        weights_arr: NDArray[np.float32] | None = None
        if weights is not None:
            weights_arr = np.asarray(weights, dtype=np.float32)
            if weights_arr.ndim != 1:
                raise ValueError(f"weights must be 1D array, got {weights_arr.ndim}D")
            if weights_arr.shape[0] != features_arr.shape[0]:
                raise ValueError(
                    f"weights shape mismatch: expected {features_arr.shape[0]} samples, got {weights_arr.shape[0]}"
                )

        # Create the Rust instance
        instance = cast(
            "Dataset",
            _RustDataset.__new__(
                cls,
                features=features_arr,
                labels=labels_2d,
                weights=weights_arr,
                feature_names=list(feature_names) if feature_names else None,
                categorical_features=all_cats if all_cats else None,
            ),
        )

        # Store Python-only attributes (on newly created instance)
        instance._was_converted = was_converted  # noqa: SLF001
        instance._groups = None  # noqa: SLF001

        # Handle groups if provided
        if groups is not None:
            groups_arr = np.asarray(groups, dtype=np.int32)
            if groups_arr.ndim != 1:
                raise ValueError(f"groups must be 1D array, got {groups_arr.ndim}D")
            if groups_arr.shape[0] != features_arr.shape[0]:
                raise ValueError(
                    f"groups shape mismatch: expected {features_arr.shape[0]} samples, got {groups_arr.shape[0]}"
                )
            instance._groups = groups_arr  # noqa: SLF001

        return instance

    # Skip __init__ since all initialization is done in __new__
    def __init__(
        self,
        features: FeaturesInput,
        labels: LabelsInput = None,
        weights: WeightsInput = None,
        groups: GroupsInput = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> None:
        """No-op init - all initialization done in __new__."""

    @property
    def groups(self) -> NDArray[np.int32] | None:
        """Group labels for ranking (if provided)."""
        return self._groups

    @property
    def has_groups(self) -> bool:
        """Whether groups are present."""
        return self._groups is not None

    @property
    def was_converted(self) -> bool:
        """Whether data was converted during construction.

        True if Python performed a copy/conversion (e.g., from DataFrame
        or non-float32 array). False if zero-copy was possible.
        """
        return self._was_converted

    def __repr__(self) -> str:
        """Return string representation."""
        cats = len(self.categorical_features)
        return (
            f"Dataset(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"has_labels={self.has_labels}, categorical_features={cats})"
        )
