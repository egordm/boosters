"""Tests for Dataset and EvalSet types."""

import numpy as np
import pytest

import boosters as bst
from boosters import Dataset, EvalSet


class TestDataset:
    """Tests for Dataset type."""

    def test_basic_construction(self) -> None:
        """Test basic dataset construction with features only."""
        X = np.random.rand(100, 10).astype(np.float32)
        ds = Dataset(X)

        assert ds.n_samples == 100
        assert ds.n_features == 10
        assert ds.shape == (100, 10)
        assert not ds.has_labels
        assert not ds.has_weights
        assert not ds.has_groups
        assert ds.feature_names is None
        assert ds.categorical_features == []

    def test_construction_with_labels(self) -> None:
        """Test dataset construction with labels."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        ds = Dataset(X, y)

        assert ds.n_samples == 100
        assert ds.n_features == 10
        assert ds.has_labels
        assert not ds.has_weights

    def test_construction_with_weights(self) -> None:
        """Test dataset construction with weights."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        w = np.random.rand(100).astype(np.float32)
        ds = Dataset(X, y, weights=w)

        assert ds.has_labels
        assert ds.has_weights

    def test_construction_with_groups(self) -> None:
        """Test dataset construction with groups (for ranking)."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        # Groups must be 1D array
        g = np.repeat(np.arange(10), 10).astype(np.int32)
        ds = Dataset(X, y, groups=g)

        assert ds.has_labels
        assert ds.has_groups

    def test_construction_with_feature_names(self) -> None:
        """Test dataset construction with feature names."""
        X = np.random.rand(100, 5).astype(np.float32)
        names = ["a", "b", "c", "d", "e"]
        ds = Dataset(X, feature_names=names)

        assert ds.feature_names == names

    def test_construction_with_categorical_features(self) -> None:
        """Test dataset construction with categorical feature indices."""
        X = np.random.rand(100, 10).astype(np.float32)
        cat_features = [0, 3, 7]
        ds = Dataset(X, categorical_features=cat_features)

        # Should be sorted
        assert ds.categorical_features == [0, 3, 7]

    def test_float64_accepted(self) -> None:
        """Test that float64 arrays are accepted."""
        X = np.random.rand(100, 10).astype(np.float64)
        y = np.random.rand(100).astype(np.float64)
        ds = Dataset(X, y)

        assert ds.n_samples == 100
        assert ds.n_features == 10

    def test_c_contiguous_zero_copy(self) -> None:
        """Test that C-contiguous float32 is zero-copy."""
        X = np.random.rand(100, 10).astype(np.float32)
        assert X.flags.c_contiguous
        ds = Dataset(X)

        assert not ds.was_converted

    def test_f_contiguous_converted(self) -> None:
        """Test that F-contiguous arrays are converted."""
        X = np.asfortranarray(np.random.rand(100, 10).astype(np.float32))
        assert X.flags.f_contiguous
        assert not X.flags.c_contiguous
        ds = Dataset(X)

        # Conversion should happen
        assert ds.was_converted

    def test_repr(self) -> None:
        """Test string representation."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        ds = Dataset(X, y, categorical_features=[0, 1])

        r = repr(ds)
        assert "n_samples=100" in r
        assert "n_features=10" in r
        assert "has_labels=true" in r  # Rust uses lowercase boolean
        assert "categorical_features=2" in r

    # Validation tests

    def test_invalid_labels_shape(self) -> None:
        """Test that mismatched labels shape raises error."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)  # Wrong size

        with pytest.raises(ValueError, match="shape mismatch"):
            Dataset(X, y)

    def test_invalid_weights_shape(self) -> None:
        """Test that mismatched weights shape raises error."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        w = np.random.rand(50).astype(np.float32)  # Wrong size

        with pytest.raises(ValueError, match="shape mismatch"):
            Dataset(X, y, weights=w)

    def test_invalid_1d_features(self) -> None:
        """Test that 1D features raises error."""
        X = np.random.rand(100).astype(np.float32)

        # 1D arrays are not recognized as valid numpy arrays for features
        with pytest.raises(TypeError):
            Dataset(X)

    def test_invalid_labels_nan(self) -> None:
        """Test that NaN in labels raises error."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        y[0] = np.nan

        with pytest.raises(ValueError, match="NaN or Inf"):
            Dataset(X, y)

    def test_invalid_labels_inf(self) -> None:
        """Test that Inf in labels raises error."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)
        y[0] = np.inf

        with pytest.raises(ValueError, match="NaN or Inf"):
            Dataset(X, y)

    def test_invalid_categorical_index(self) -> None:
        """Test that out-of-range categorical index raises error."""
        X = np.random.rand(100, 10).astype(np.float32)

        with pytest.raises(ValueError, match="out of range"):
            Dataset(X, categorical_features=[15])

    def test_nan_features_allowed(self) -> None:
        """Test that NaN in features is allowed (treated as missing)."""
        X = np.random.rand(100, 10).astype(np.float32)
        X[0, 0] = np.nan
        ds = Dataset(X)

        assert ds.n_samples == 100

    def test_empty_features_rejected(self) -> None:
        """Test that empty features raise error."""
        X = np.random.rand(0, 10).astype(np.float32)

        # Empty arrays fail in shape detection path
        with pytest.raises((ValueError, TypeError)):
            Dataset(X)


class TestEvalSet:
    """Tests for EvalSet type."""

    def test_basic_construction(self) -> None:
        """Test basic EvalSet construction."""
        X = np.random.rand(50, 10).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        ds = Dataset(X, y)

        es = EvalSet("validation", ds)
        assert es.name == "validation"

    def test_dataset_access(self) -> None:
        """Test accessing the underlying dataset."""
        X = np.random.rand(50, 10).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        ds = Dataset(X, y)

        es = EvalSet("test", ds)
        retrieved = es.dataset

        # Should get a usable dataset
        assert retrieved.n_samples == 50
        assert retrieved.n_features == 10

    def test_repr(self) -> None:
        """Test string representation."""
        X = np.random.rand(50, 10).astype(np.float32)
        y = np.random.rand(50).astype(np.float32)
        ds = Dataset(X, y)
        es = EvalSet("validation", ds)

        r = repr(es)
        assert "validation" in r
        assert "n_samples=50" in r


class TestDataImports:
    """Test that data types are properly exported."""

    def test_import_from_main(self) -> None:
        """Test importing from main boosters module."""
        assert hasattr(bst, "Dataset")
        assert hasattr(bst, "EvalSet")

    def test_import_from_data(self) -> None:
        """Test importing from boosters.data module."""
        from boosters.data import Dataset as D
        from boosters.data import EvalSet as E

        assert D is Dataset
        assert E is EvalSet
