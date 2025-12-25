"""Tests for evaluation metric types."""

import pytest

import boosters as bst


class TestParameterlessMetrics:
    """Tests for metrics without parameters."""

    def test_rmse_creation(self) -> None:
        """Rmse can be instantiated."""
        metric = bst.Rmse()
        assert metric is not None

    def test_rmse_repr(self) -> None:
        """Rmse has meaningful repr."""
        metric = bst.Rmse()
        assert "Rmse" in repr(metric)

    def test_mae_creation(self) -> None:
        """Mae can be instantiated."""
        metric = bst.Mae()
        assert metric is not None

    def test_mae_repr(self) -> None:
        """Mae has meaningful repr."""
        metric = bst.Mae()
        assert "Mae" in repr(metric)

    def test_mape_creation(self) -> None:
        """Mape can be instantiated."""
        metric = bst.Mape()
        assert metric is not None

    def test_mape_repr(self) -> None:
        """Mape has meaningful repr."""
        metric = bst.Mape()
        assert "Mape" in repr(metric)

    def test_logloss_creation(self) -> None:
        """LogLoss can be instantiated."""
        metric = bst.LogLoss()
        assert metric is not None

    def test_logloss_repr(self) -> None:
        """LogLoss has meaningful repr."""
        metric = bst.LogLoss()
        assert "LogLoss" in repr(metric)

    def test_auc_creation(self) -> None:
        """Auc can be instantiated."""
        metric = bst.Auc()
        assert metric is not None

    def test_auc_repr(self) -> None:
        """Auc has meaningful repr."""
        metric = bst.Auc()
        assert "Auc" in repr(metric)

    def test_accuracy_creation(self) -> None:
        """Accuracy can be instantiated."""
        metric = bst.Accuracy()
        assert metric is not None

    def test_accuracy_repr(self) -> None:
        """Accuracy has meaningful repr."""
        metric = bst.Accuracy()
        assert "Accuracy" in repr(metric)


class TestNdcg:
    """Tests for Ndcg metric with at parameter."""

    def test_default_at(self) -> None:
        """Ndcg has default at=10."""
        metric = bst.Ndcg()
        assert metric.at == 10

    def test_custom_at(self) -> None:
        """Ndcg accepts custom at value."""
        metric = bst.Ndcg(at=5)
        assert metric.at == 5

    def test_repr_includes_at(self) -> None:
        """Ndcg repr includes at value."""
        metric = bst.Ndcg(at=20)
        r = repr(metric)
        assert "Ndcg" in r
        assert "20" in r

    def test_invalid_at_zero(self) -> None:
        """Ndcg rejects at < 1."""
        with pytest.raises((ValueError, Exception)):
            bst.Ndcg(at=0)


class TestMetricTypeAlias:
    """Tests for Metric type alias."""

    def test_rmse_is_metric(self) -> None:
        """Rmse is a valid Metric."""
        metric: bst.Metric = bst.Rmse()
        assert metric is not None

    def test_ndcg_is_metric(self) -> None:
        """Ndcg is a valid Metric."""
        metric: bst.Metric = bst.Ndcg(at=5)
        assert metric is not None

    def test_auc_is_metric(self) -> None:
        """Auc is a valid Metric."""
        metric: bst.Metric = bst.Auc()
        assert metric is not None


class TestExportsFromPackage:
    """Tests that all metrics are exported from top-level package."""

    def test_all_metrics_exported(self) -> None:
        """All metric types are accessible from boosters package."""
        # Parameterless
        assert hasattr(bst, "Rmse")
        assert hasattr(bst, "Mae")
        assert hasattr(bst, "Mape")
        assert hasattr(bst, "LogLoss")
        assert hasattr(bst, "Auc")
        assert hasattr(bst, "Accuracy")

        # Parameterized
        assert hasattr(bst, "Ndcg")

        # Type alias
        assert hasattr(bst, "Metric")
