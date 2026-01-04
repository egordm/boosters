"""Unit tests for dataset Feature conversions."""

from typing import Any, cast

import numpy as np
import pytest

from boosters import Feature


class TestFeatureDense:
    def test_from_dense_numpy(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        f = Feature.from_dense(x, name="x")
        assert f.name == "x"
        assert f.n_samples == 3
        assert not f.is_sparse
        assert not f.categorical

    def test_from_dense_sparse_rejected(self) -> None:
        pytest.importorskip("scipy")
        from scipy.sparse import csr_matrix

        col = csr_matrix([[0.0], [1.5], [0.0], [2.0]], dtype=np.float32)
        with pytest.raises(TypeError, match="from_dense"):
            Feature.from_dense(cast(Any, col), name="x")


class TestFeaturePandas:
    def test_pandas_object_auto_categorical(self) -> None:
        pd = pytest.importorskip("pandas")
        s = pd.Series(["red", "blue", "red"], name="color")

        f = Feature.from_dense(s, name="color")
        assert f.categorical
        assert f.name == "color"
        assert f.n_samples == 3

    def test_pandas_object_categorical_false_raises(self) -> None:
        pd = pytest.importorskip("pandas")
        s = pd.Series(["red", "blue", "red"], name="color")

        with pytest.raises(TypeError, match="Cannot convert pandas Series"):
            Feature.from_dense(s, name="color", categorical=False)


class TestFeatureSparse:
    def test_from_sparse_accepts_sparse_column_vector(self) -> None:
        pytest.importorskip("scipy")
        from scipy.sparse import csr_matrix

        col = csr_matrix([[0.0], [1.5], [0.0], [2.0]], dtype=np.float32)
        f = Feature.from_sparse(cast(Any, col), name="x")
        assert f.is_sparse
        assert f.name == "x"
        assert f.n_samples == 4

    def test_from_sparse_raw_indices_values(self) -> None:
        idx = np.array([0, 3], dtype=np.uint32)
        vals = np.array([1.0, 2.0], dtype=np.float32)
        f = Feature.from_sparse(idx, vals, 5, name="x")
        assert f.is_sparse
        assert f.n_samples == 5
