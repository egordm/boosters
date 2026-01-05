"""Tests for the dataset system."""

from __future__ import annotations

from boosters_eval.config import Task
from boosters_eval.datasets import (
    DATASETS,
    get_datasets_by_task,
)


class TestDatasetRegistry:
    """Tests for the DATASETS registry."""

    def test_datasets_not_empty(self) -> None:
        """Test that DATASETS is populated."""
        assert len(DATASETS) >= 9

    def test_required_datasets_present(self) -> None:
        """Test that required datasets are present."""
        required = ["california", "breast_cancer", "iris", "covertype"]
        for name in required:
            assert name in DATASETS, f"Missing required dataset: {name}"

    def test_dataset_configs_valid(self) -> None:
        """Test that all dataset configs have required fields."""
        for name, config in DATASETS.items():
            assert config.name == name
            assert config.task in (Task.REGRESSION, Task.QUANTILE_REGRESSION, Task.BINARY, Task.MULTICLASS)
            assert callable(config.loader)


class TestGetDatasetsByTask:
    """Tests for get_datasets_by_task function."""

    def test_filter_by_task(self) -> None:
        """Test filtering datasets by task type."""
        regression = get_datasets_by_task(Task.REGRESSION)
        quantile = get_datasets_by_task(Task.QUANTILE_REGRESSION)
        binary = get_datasets_by_task(Task.BINARY)
        multiclass = get_datasets_by_task(Task.MULTICLASS)

        assert "california" in regression
        assert "liander_energy_forecasting" in quantile
        assert "breast_cancer" in binary
        assert "iris" in multiclass

        assert all(d.task == Task.REGRESSION for d in regression.values())
        assert all(d.task == Task.QUANTILE_REGRESSION for d in quantile.values())
        assert all(d.task == Task.BINARY for d in binary.values())
        assert all(d.task == Task.MULTICLASS for d in multiclass.values())


class TestDatasetMetadata:
    """Tests for dataset metadata."""

    def test_multiclass_has_n_classes(self) -> None:
        """Test that multiclass datasets have n_classes set."""
        for name, config in DATASETS.items():
            if config.task == Task.MULTICLASS:
                assert config.n_classes is not None, f"Missing n_classes for {name}"
                assert config.n_classes >= 3
