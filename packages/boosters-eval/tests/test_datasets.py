"""Tests for the dataset system."""

from __future__ import annotations

import time

import numpy as np
import pytest

from boosters_eval.config import Task
from boosters_eval.datasets import (
    DATASETS,
    breast_cancer,
    california_housing,
    get_datasets_by_task,
    iris,
    synthetic_binary_small,
    synthetic_regression_small,
)


class TestDatasetRegistry:
    """Tests for the DATASETS registry."""

    def test_datasets_not_empty(self) -> None:
        """Test that DATASETS is populated."""
        assert len(DATASETS) > 0
        assert len(DATASETS) >= 9  # At least 9 datasets

    def test_required_datasets_present(self) -> None:
        """Test that required datasets are present."""
        required = ["california", "breast_cancer", "iris", "covertype"]
        for name in required:
            assert name in DATASETS, f"Missing required dataset: {name}"

    def test_dataset_configs_valid(self) -> None:
        """Test that all dataset configs have required fields."""
        for name, config in DATASETS.items():
            assert config.name == name
            assert config.task in (Task.REGRESSION, Task.BINARY, Task.MULTICLASS)
            assert callable(config.loader)


class TestDatasetLoading:
    """Tests for dataset loading."""

    def test_california_housing_shape(self) -> None:
        """Test California housing dataset returns correct shape."""
        x, y = california_housing()
        assert x.shape == (20640, 8)
        assert y.shape == (20640,)
        assert x.dtype == np.float32
        assert y.dtype == np.float32

    def test_breast_cancer_shape(self) -> None:
        """Test Breast cancer dataset returns correct shape."""
        x, y = breast_cancer()
        assert x.shape == (569, 30)
        assert y.shape == (569,)
        assert x.dtype == np.float32

    def test_iris_shape(self) -> None:
        """Test Iris dataset returns correct shape."""
        x, y = iris()
        assert x.shape == (150, 4)
        assert y.shape == (150,)
        # Check multiclass labels
        assert set(y) == {0.0, 1.0, 2.0}

    def test_synthetic_regression_small_shape(self) -> None:
        """Test synthetic regression small dataset."""
        x, y = synthetic_regression_small()
        assert x.shape == (2000, 50)
        assert y.shape == (2000,)

    def test_synthetic_binary_small_shape(self) -> None:
        """Test synthetic binary classification small dataset."""
        x, y = synthetic_binary_small()
        assert x.shape == (2000, 50)
        assert y.shape == (2000,)
        assert set(y) == {0.0, 1.0}


class TestDatasetCaching:
    """Tests for dataset caching."""

    def test_dataset_caching_performance(self) -> None:
        """Test that second load is faster (cached)."""
        # Clear any existing cache
        synthetic_regression_small.cache_clear()

        # First load (cold)
        start = time.perf_counter()
        _ = synthetic_regression_small()
        first_time = time.perf_counter() - start

        # Second load (cached)
        start = time.perf_counter()
        _ = synthetic_regression_small()
        second_time = time.perf_counter() - start

        # Cached load should be much faster
        # Use a ratio check - second should be at least 10x faster if cached
        # But we also accept if first was already fast (< 1ms means caching is irrelevant)
        if first_time > 0.001:  # Only test if first load was slow enough to measure
            assert second_time < first_time / 5, (
                f"Caching not working: first={first_time:.4f}s, second={second_time:.4f}s"
            )

    def test_cache_returns_same_data(self) -> None:
        """Test that cached data is identical."""
        synthetic_regression_small.cache_clear()

        x1, y1 = synthetic_regression_small()
        x2, y2 = synthetic_regression_small()

        # Should be the exact same arrays (same object)
        assert x1 is x2
        assert y1 is y2


class TestGetDatasetsByTask:
    """Tests for get_datasets_by_task function."""

    def test_regression_datasets(self) -> None:
        """Test filtering by regression task."""
        datasets = get_datasets_by_task(Task.REGRESSION)
        assert len(datasets) >= 3
        assert "california" in datasets
        assert all(d.task == Task.REGRESSION for d in datasets.values())

    def test_binary_datasets(self) -> None:
        """Test filtering by binary classification task."""
        datasets = get_datasets_by_task(Task.BINARY)
        assert len(datasets) >= 2
        assert "breast_cancer" in datasets
        assert all(d.task == Task.BINARY for d in datasets.values())

    def test_multiclass_datasets(self) -> None:
        """Test filtering by multiclass task."""
        datasets = get_datasets_by_task(Task.MULTICLASS)
        assert len(datasets) >= 2
        assert "iris" in datasets
        assert all(d.task == Task.MULTICLASS for d in datasets.values())


class TestDatasetMetadata:
    """Tests for dataset metadata."""

    def test_multiclass_n_classes(self) -> None:
        """Test that multiclass datasets have n_classes set."""
        for name, config in DATASETS.items():
            if config.task == Task.MULTICLASS:
                assert config.n_classes is not None, f"Missing n_classes for {name}"
                assert config.n_classes >= 3, f"Invalid n_classes for {name}"

    def test_covertype_has_subsample(self) -> None:
        """Test that covertype has subsample set."""
        assert DATASETS["covertype"].subsample == 50000
