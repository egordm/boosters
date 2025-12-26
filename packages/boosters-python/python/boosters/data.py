"""Data types for gradient boosting.

This module provides dataset wrappers for training and evaluation.

Types:
    - Dataset: Training/prediction dataset with features, labels, weights
    - EvalSet: Named evaluation set for validation during training
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from boosters._boosters_rs import Dataset as _RustDataset
from boosters._boosters_rs import EvalSet

if TYPE_CHECKING:
    import pandas as pd

__all__: list[str] = [
    "Dataset",
    "EvalSet",
]


def _extract_dataframe(
    df: pd.DataFrame,
) -> tuple[NDArray[np.float32], list[str] | None, list[int]]:
    """Extract features from a pandas DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (features_array, feature_names, categorical_indices).
    """
    import pandas as pd

    # Get feature names
    feature_names = list(df.columns.astype(str))

    # Detect categorical columns
    categorical_indices: list[int] = []
    for i, dtype in enumerate(df.dtypes):
        if isinstance(dtype, pd.CategoricalDtype) or str(dtype) == "category":
            categorical_indices.append(i)

    # Convert to numpy array
    values = df.values

    # Ensure float32 and C-contiguous
    if values.dtype == np.float32 and values.flags.c_contiguous:
        features = values
    elif values.dtype in (np.float32, np.float64):
        features = np.ascontiguousarray(values, dtype=np.float32)
    else:
        features = np.ascontiguousarray(values, dtype=np.float32)

    return features, feature_names, categorical_indices


def _extract_numpy_array(
    arr: NDArray,
) -> tuple[NDArray[np.float32], bool]:
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

    # Ensure float32 and C-contiguous
    if arr.dtype == np.float32 and arr.flags.c_contiguous:
        return arr, False
    else:
        return np.ascontiguousarray(arr, dtype=np.float32), True


def _is_sparse_matrix(obj: object) -> bool:
    """Check if object is a scipy sparse matrix."""
    try:
        import scipy.sparse

        return scipy.sparse.issparse(obj)
    except ImportError:
        return False


def _is_dataframe(obj: object) -> bool:
    """Check if object is a pandas DataFrame."""
    try:
        import pandas as pd

        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False


class Dataset:
    """Dataset holding features, labels, and optional metadata.

    This class wraps NumPy arrays or pandas DataFrames for use with boosters models.
    Data is converted to C-contiguous float32 arrays for efficient processing.

    Args:
        features: 2D NumPy array or pandas DataFrame of shape (n_samples, n_features).
        labels: Optional 1D array of shape (n_samples,).
        weights: Optional 1D array of sample weights (n_samples,).
        groups: Optional 1D array of group labels for ranking.
        feature_names: Optional list of feature names. Auto-detected from DataFrames.
        categorical_features: Optional list of categorical feature indices.
            Auto-detected from pandas categorical dtype.

    Raises:
        ValueError: If data is invalid (shape mismatch, Inf values, etc.).
        TypeError: If data types are unsupported.
        NotImplementedError: If sparse matrices are passed (not yet supported).

    Example:
        >>> import numpy as np
        >>> from boosters import Dataset
        >>>
        >>> X = np.random.rand(100, 10).astype(np.float32)
        >>> y = np.random.rand(100).astype(np.float32)
        >>>
        >>> dataset = Dataset(X, y)
        >>> print(f"Samples: {dataset.n_samples}, Features: {dataset.n_features}")
    """

    _inner: _RustDataset

    def __init__(
        self,
        features: ArrayLike,
        labels: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        groups: ArrayLike | None = None,
        feature_names: Sequence[str] | None = None,
        categorical_features: Sequence[int] | None = None,
    ) -> None:
        # Handle sparse matrices
        if _is_sparse_matrix(features):
            raise NotImplementedError(
                "Sparse matrices are not yet supported. Convert to dense array with "
                ".toarray() or use a pandas DataFrame. Sparse support is planned for "
                "a future release."
            )

        # Extract features based on type
        detected_cats: list[int] = []
        detected_names: list[str] | None = None
        python_converted = False

        if _is_dataframe(features):
            features_arr, detected_names, detected_cats = _extract_dataframe(
                features  # type: ignore[arg-type]  # Checked by _is_dataframe
            )
            python_converted = True  # DataFrame always converts
        elif isinstance(features, np.ndarray):
            features_arr, python_converted = _extract_numpy_array(features)
        else:
            # Try to convert to numpy array
            try:
                tmp_arr = np.asarray(features, dtype=np.float32)
                features_arr, python_converted = _extract_numpy_array(tmp_arr)
                python_converted = True  # Always converted from non-ndarray
            except (ValueError, TypeError) as e:
                type_name = type(features).__name__
                raise TypeError(f"expected numpy array or pandas DataFrame, got {type_name}") from e

        # Store conversion flag for the property
        self._python_converted = python_converted

        # Use detected names if not provided
        if feature_names is None:
            feature_names = detected_names

        # Merge categorical features
        all_cats = list(categorical_features) if categorical_features else []
        for cat in detected_cats:
            if cat not in all_cats:
                all_cats.append(cat)
        all_cats.sort()

        # Convert labels if provided
        labels_arr = None
        if labels is not None:
            labels_arr = np.asarray(labels, dtype=np.float32)
            if labels_arr.ndim != 1:
                raise ValueError(f"labels must be 1D array, got {labels_arr.ndim}D")
            if labels_arr.shape[0] != features_arr.shape[0]:
                raise ValueError(
                    f"labels shape mismatch: expected {features_arr.shape[0]} samples, "
                    f"got {labels_arr.shape[0]}"
                )
            if not np.all(np.isfinite(labels_arr)):
                raise ValueError("labels contain NaN or Inf values")

        # Convert weights if provided
        weights_arr = None
        if weights is not None:
            weights_arr = np.asarray(weights, dtype=np.float32)
            if weights_arr.ndim != 1:
                raise ValueError(f"weights must be 1D array, got {weights_arr.ndim}D")
            if weights_arr.shape[0] != features_arr.shape[0]:
                raise ValueError(
                    f"weights shape mismatch: expected {features_arr.shape[0]} samples, "
                    f"got {weights_arr.shape[0]}"
                )

        # Convert groups if provided
        groups_arr = None
        if groups is not None:
            groups_arr = np.asarray(groups, dtype=np.int32)
            if groups_arr.ndim != 1:
                raise ValueError(f"groups must be 1D array, got {groups_arr.ndim}D")
            if groups_arr.shape[0] != features_arr.shape[0]:
                raise ValueError(
                    f"groups shape mismatch: expected {features_arr.shape[0]} samples, "
                    f"got {groups_arr.shape[0]}"
                )

        # Create Rust Dataset with pre-validated arrays
        self._inner = _RustDataset(
            features=features_arr,
            labels=labels_arr,
            weights=weights_arr,
            groups=groups_arr,
            feature_names=list(feature_names) if feature_names else None,
            categorical_features=all_cats if all_cats else None,
        )

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return self._inner.n_samples

    @property
    def n_features(self) -> int:
        """Number of features in the dataset."""
        return self._inner.n_features

    @property
    def has_labels(self) -> bool:
        """Whether labels are present."""
        return self._inner.has_labels

    @property
    def has_weights(self) -> bool:
        """Whether weights are present."""
        return self._inner.has_weights

    @property
    def has_groups(self) -> bool:
        """Whether groups are present."""
        return self._inner.has_groups

    @property
    def feature_names(self) -> list[str] | None:
        """Feature names if provided."""
        return self._inner.feature_names

    @property
    def categorical_features(self) -> list[int]:
        """Indices of categorical features."""
        return self._inner.categorical_features

    @property
    def was_converted(self) -> bool:
        """Whether data was converted (not zero-copy).

        True if either Python or Rust performed a copy/conversion.
        """
        return self._python_converted or self._inner.was_converted

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the features array as (n_samples, n_features)."""
        return self._inner.shape

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"Dataset(n_samples={self.n_samples}, n_features={self.n_features}, "
            f"has_labels={self.has_labels}, categorical_features={len(self.categorical_features)})"
        )
