"""Tests for Dataset and EvalSet types.

Focuses on:
- Core construction behavior (not just "does it accept values")
- Validation that protects users from errors
- Edge cases and conversions
"""

import numpy as np
import pytest

from boosters import Dataset, EvalSet


class TestDatasetConstruction:
    """Tests for Dataset construction and conversion behavior."""

    def test_basic_construction(self) -> None:
        """Test basic dataset construction with features only."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        ds = Dataset(X)
        assert ds.n_samples == 100
        assert ds.n_features == 10
        assert ds.shape == (100, 10)

    def test_with_labels_and_weights(self) -> None:
        """Test dataset construction with labels and weights."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        y = rng.random(100).astype(np.float32)
        w = rng.random(100).astype(np.float32)
        ds = Dataset(X, y, weights=w)
        assert ds.has_labels
        assert ds.has_weights

    def test_c_contiguous_zero_copy(self) -> None:
        """Test that C-contiguous float32 is zero-copy."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        assert X.flags.c_contiguous
        ds = Dataset(X)
        assert not ds.was_converted

    def test_f_contiguous_converted(self) -> None:
        """Test that F-contiguous arrays are converted."""
        rng = np.random.default_rng(42)
        X = np.asfortranarray(rng.random((100, 10)).astype(np.float32))  # noqa: N806
        assert not X.flags.c_contiguous
        ds = Dataset(X)
        assert ds.was_converted

    def test_float64_converted(self) -> None:
        """Test that float64 arrays are converted to float32."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float64)  # noqa: N806
        ds = Dataset(X)
        assert ds.was_converted


class TestDatasetValidation:
    """Tests for Dataset validation that protects users from errors."""

    def test_labels_shape_mismatch_rejected(self) -> None:
        """Mismatched labels shape raises clear error."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        y = rng.random(50).astype(np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            Dataset(X, y)

    def test_weights_shape_mismatch_rejected(self) -> None:
        """Mismatched weights shape raises clear error."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        y = rng.random(100).astype(np.float32)
        w = rng.random(50).astype(np.float32)
        with pytest.raises(ValueError, match="shape mismatch"):
            Dataset(X, y, weights=w)

    def test_1d_features_rejected(self) -> None:
        """1D features array raises clear error."""
        rng = np.random.default_rng(42)
        X = rng.random(100).astype(np.float32)  # noqa: N806
        with pytest.raises(ValueError, match="2D array"):
            Dataset(X)

    def test_nan_labels_rejected(self) -> None:
        """NaN in labels raises clear error."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        y = rng.random(100).astype(np.float32)
        y[0] = np.nan
        with pytest.raises(ValueError, match="NaN or Inf"):
            Dataset(X, y)

    def test_inf_labels_rejected(self) -> None:
        """Inf in labels raises clear error."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        y = rng.random(100).astype(np.float32)
        y[0] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            Dataset(X, y)

    def test_invalid_categorical_index_rejected(self) -> None:
        """Out-of-range categorical index raises clear error."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        with pytest.raises(ValueError, match="out of range"):
            Dataset(X, categorical_features=[15])

    def test_nan_features_allowed(self) -> None:
        """NaN in features is allowed (treated as missing)."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        X[0, 0] = np.nan
        ds = Dataset(X)
        assert ds.n_samples == 100

    def test_sparse_matrix_raises_not_implemented(self) -> None:
        """Sparse matrices raise helpful error message."""
        pytest.importorskip("scipy")
        from scipy.sparse import csr_matrix

        rng = np.random.default_rng(42)
        X_dense = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        X_sparse = csr_matrix(X_dense)  # noqa: N806

        with pytest.raises(NotImplementedError, match="Sparse matrices are not yet supported"):
            Dataset(X_sparse)


class TestDatasetCategoricalFeatures:
    """Tests for categorical feature handling."""

    def test_categorical_features_sorted(self) -> None:
        """Categorical feature indices are sorted."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 10)).astype(np.float32)  # noqa: N806
        ds = Dataset(X, categorical_features=[7, 0, 3])
        assert ds.categorical_features == [0, 3, 7]


class TestEvalSet:
    """Tests for EvalSet type."""

    def test_basic_construction(self) -> None:
        """EvalSet wraps a dataset with a name."""
        rng = np.random.default_rng(42)
        X = rng.random((50, 10)).astype(np.float32)  # noqa: N806
        y = rng.random(50).astype(np.float32)
        ds = Dataset(X, y)
        es = EvalSet(ds, "validation")
        assert es.name == "validation"
        assert es.dataset.n_samples == 50
