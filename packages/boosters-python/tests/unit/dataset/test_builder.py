"""Unit tests for DatasetBuilder convenience behavior."""

from __future__ import annotations

import numpy as np
import pytest

from boosters import DatasetBuilder, Feature


class TestDatasetBuilder:
    def test_add_feature_dense(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = DatasetBuilder()
        b.add_feature(Feature.from_dense(x, name="x"))
        d = b.labels([0.0, 1.0, 0.0]).build()

        assert d.n_features == 1
        assert d.n_samples == 3

    def test_add_feature_feature_object(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        f = Feature.from_dense(x, name="x")

        b = DatasetBuilder()
        b.add_feature(f)
        d = b.labels([0.0, 1.0, 0.0]).build()

        assert d.n_features == 1
        assert d.n_samples == 3


class TestDatasetBuilderSparse:
    def test_add_features_from_sparse_column_major(self) -> None:
        scipy = pytest.importorskip("scipy")
        sparse = scipy.sparse

        x_sparse = sparse.csr_matrix(
            [[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]],
            dtype=np.float32,
        )

        b = DatasetBuilder()
        b.add_features_from_sparse(x_sparse)
        d = b.labels([0.0, 1.0, 0.0]).build()

        assert d.n_features == 2
        assert d.n_samples == 3
