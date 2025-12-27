"""Tests for metrics computation."""

from __future__ import annotations

import numpy as np
import pytest

from boosters_eval.config import Task
from boosters_eval.metrics import (
    LOWER_BETTER_METRICS,
    compute_metrics,
    is_lower_better,
    primary_metric,
)


class TestComputeMetricsRegression:
    """Tests for regression metrics."""

    def test_rmse_known_value(self) -> None:
        """Test RMSE with known value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        # RMSE = sqrt((0^2 + 0^2 + 1^2) / 3) = sqrt(1/3) ≈ 0.577
        metrics = compute_metrics(Task.REGRESSION, y_true, y_pred)
        assert pytest.approx(metrics["rmse"], rel=1e-3) == 0.577

    def test_mae_known_value(self) -> None:
        """Test MAE with known value."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        # MAE = (0 + 0 + 1) / 3 ≈ 0.333
        metrics = compute_metrics(Task.REGRESSION, y_true, y_pred)
        assert pytest.approx(metrics["mae"], rel=1e-3) == 0.333

    def test_r2_perfect(self) -> None:
        """Test R² for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(Task.REGRESSION, y_true, y_pred)
        assert metrics["r2"] == 1.0

    def test_regression_returns_all_metrics(self) -> None:
        """Test regression returns expected metrics."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.0, 2.9])
        metrics = compute_metrics(Task.REGRESSION, y_true, y_pred)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics


class TestComputeMetricsBinary:
    """Tests for binary classification metrics."""

    def test_perfect_prediction(self) -> None:
        """Test accuracy = 1.0 for perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])  # Probabilities
        metrics = compute_metrics(Task.BINARY, y_true, y_pred)
        assert metrics["accuracy"] == 1.0

    def test_logloss_bounded(self) -> None:
        """Test log loss is positive and finite."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.3, 0.7, 0.4, 0.6])
        metrics = compute_metrics(Task.BINARY, y_true, y_pred)
        assert 0 < metrics["logloss"] < 10
        assert np.isfinite(metrics["logloss"])

    def test_binary_returns_all_metrics(self) -> None:
        """Test binary returns expected metrics."""
        y_true = np.array([0, 1])
        y_pred = np.array([0.3, 0.7])
        metrics = compute_metrics(Task.BINARY, y_true, y_pred)
        assert "logloss" in metrics
        assert "accuracy" in metrics


class TestComputeMetricsMulticlass:
    """Tests for multiclass classification metrics."""

    def test_perfect_prediction(self) -> None:
        """Test accuracy = 1.0 for perfect predictions."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        metrics = compute_metrics(Task.MULTICLASS, y_true, y_pred, n_classes=3)
        assert metrics["accuracy"] == 1.0

    def test_mlogloss_bounded(self) -> None:
        """Test multiclass log loss is positive and finite."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]])
        metrics = compute_metrics(Task.MULTICLASS, y_true, y_pred, n_classes=3)
        assert 0 < metrics["mlogloss"] < 10
        assert np.isfinite(metrics["mlogloss"])

    def test_multiclass_returns_all_metrics(self) -> None:
        """Test multiclass returns expected metrics."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        metrics = compute_metrics(Task.MULTICLASS, y_true, y_pred, n_classes=3)
        assert "mlogloss" in metrics
        assert "accuracy" in metrics


class TestPrimaryMetric:
    """Tests for primary_metric function."""

    def test_regression_primary(self) -> None:
        """Test primary metric for regression."""
        assert primary_metric(Task.REGRESSION) == "rmse"

    def test_binary_primary(self) -> None:
        """Test primary metric for binary."""
        assert primary_metric(Task.BINARY) == "logloss"

    def test_multiclass_primary(self) -> None:
        """Test primary metric for multiclass."""
        assert primary_metric(Task.MULTICLASS) == "mlogloss"


class TestIsLowerBetter:
    """Tests for is_lower_better function."""

    def test_rmse_lower_better(self) -> None:
        """Test RMSE is lower-is-better."""
        assert is_lower_better("rmse") is True

    def test_mae_lower_better(self) -> None:
        """Test MAE is lower-is-better."""
        assert is_lower_better("mae") is True

    def test_logloss_lower_better(self) -> None:
        """Test logloss is lower-is-better."""
        assert is_lower_better("logloss") is True

    def test_accuracy_higher_better(self) -> None:
        """Test accuracy is higher-is-better."""
        assert is_lower_better("accuracy") is False

    def test_r2_higher_better(self) -> None:
        """Test R² is higher-is-better."""
        assert is_lower_better("r2") is False

    def test_constant_contains_expected(self) -> None:
        """Test LOWER_BETTER_METRICS constant contains expected metrics."""
        assert "rmse" in LOWER_BETTER_METRICS
        assert "mae" in LOWER_BETTER_METRICS
        assert "logloss" in LOWER_BETTER_METRICS
        assert "mlogloss" in LOWER_BETTER_METRICS
