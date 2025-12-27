"""Tests for metrics computation."""

from __future__ import annotations

import numpy as np
import pytest

from boosters_eval.config import Task
from boosters_eval.metrics import (
    compute_metrics,
    is_lower_better,
    primary_metric,
)


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_rmse_known_value(self) -> None:
        """Test RMSE with known value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        # RMSE = sqrt((0^2 + 0^2 + 1^2) / 3) = sqrt(1/3) ≈ 0.577
        metrics = compute_metrics(Task.REGRESSION, y_true, y_pred)
        assert pytest.approx(metrics["rmse"], rel=1e-3) == 0.577

    @pytest.mark.parametrize(
        "task,expected_metrics",
        [
            (Task.REGRESSION, ["rmse", "mae", "r2"]),
            (Task.BINARY, ["logloss", "accuracy"]),
            (Task.MULTICLASS, ["mlogloss", "accuracy"]),
        ],
    )
    def test_returns_expected_metrics(self, task: Task, expected_metrics: list[str]) -> None:
        """Test that each task returns expected metric keys."""
        if task == Task.REGRESSION:
            y_true = np.array([1.0, 2.0, 3.0])
            y_pred = np.array([1.1, 2.0, 2.9])
            metrics = compute_metrics(task, y_true, y_pred)
        elif task == Task.BINARY:
            y_true = np.array([0, 1, 0, 1])
            y_pred = np.array([0.3, 0.7, 0.4, 0.6])
            metrics = compute_metrics(task, y_true, y_pred)
        else:
            y_true = np.array([0, 1, 2])
            y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            metrics = compute_metrics(task, y_true, y_pred, n_classes=3)

        for metric in expected_metrics:
            assert metric in metrics
            assert np.isfinite(metrics[metric])

    def test_perfect_binary_accuracy(self) -> None:
        """Test accuracy = 1.0 for perfect binary predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = compute_metrics(Task.BINARY, y_true, y_pred)
        assert metrics["accuracy"] == 1.0

    def test_perfect_multiclass_accuracy(self) -> None:
        """Test accuracy = 1.0 for perfect multiclass predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        metrics = compute_metrics(Task.MULTICLASS, y_true, y_pred, n_classes=3)
        assert metrics["accuracy"] == 1.0


class TestPrimaryMetricAndDirection:
    """Tests for primary_metric and is_lower_better functions."""

    @pytest.mark.parametrize(
        "task,expected_primary",
        [
            (Task.REGRESSION, "rmse"),
            (Task.BINARY, "logloss"),
            (Task.MULTICLASS, "mlogloss"),
        ],
    )
    def test_primary_metric(self, task: Task, expected_primary: str) -> None:
        """Test primary metric for each task."""
        assert primary_metric(task) == expected_primary

    @pytest.mark.parametrize(
        "metric,expected_lower",
        [
            ("rmse", True),
            ("mae", True),
            ("logloss", True),
            ("mlogloss", True),
            ("accuracy", False),
            ("r2", False),
        ],
    )
    def test_metric_direction(self, metric: str, expected_lower: bool) -> None:
        """Test is_lower_better for various metrics."""
        assert is_lower_better(metric) == expected_lower
