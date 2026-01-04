"""Unit tests for Dataset construction behavior."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from boosters import Dataset


class TestDatasetNumpy:
    def test_dataset_from_array_no_conversion(self) -> None:
        rng = np.random.default_rng(0)
        x = np.ascontiguousarray(rng.standard_normal((10, 3), dtype=np.float32))
        y = np.ascontiguousarray(rng.standard_normal(10, dtype=np.float32))

        d = Dataset(x, y)

        assert d.n_samples == 10
        assert d.n_features == 3
        assert d.was_converted is False

    def test_dataset_from_array_converts_dtype(self) -> None:
        rng = np.random.default_rng(0)
        x = np.ascontiguousarray(rng.standard_normal((10, 3)).astype(np.float64))
        y = np.ascontiguousarray(rng.standard_normal(10).astype(np.float64))

        d = Dataset(x, y)

        assert d.n_samples == 10
        assert d.n_features == 3
        assert d.was_converted is True

    def test_dataset_rejects_non_2d_features(self) -> None:
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        with pytest.raises(ValueError, match="features must be 2D"):
            Dataset(cast(Any, x), y)


class TestDatasetPandas:
    def test_dataset_from_dataframe(self) -> None:
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "y": [1.0, 2.0, 3.0],
        })
        labels = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        d = Dataset(df, labels)

        assert d.n_samples == 3
        assert d.n_features == 2

    def test_dataset_dataframe_object_column_auto_categorical(self) -> None:
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame({
            "x": [0.0, 1.0, 2.0],
            "color": ["red", "blue", "red"],
        })
        labels = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        d = Dataset(df, labels)

        # Rust Dataset exposes categorical features by index.
        assert d.categorical_features == [1]


class TestDatasetWeights:
    def test_dataset_accepts_weights(self) -> None:
        rng = np.random.default_rng(0)
        x = np.ascontiguousarray(rng.standard_normal((10, 3), dtype=np.float32))
        y = np.ascontiguousarray(rng.standard_normal(10, dtype=np.float32))
        w = np.ones(10, dtype=np.float32)

        d = Dataset(x, y, weights=w)

        assert d.n_samples == 10
        assert d.n_features == 3
