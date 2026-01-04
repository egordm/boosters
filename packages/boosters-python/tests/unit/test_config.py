"""Tests for GBDTConfig and GBLinearConfig validation.

Focuses on validation that catches user errors early.
"""

from typing import Any, cast

import pytest

from boosters import GBDTConfig, GBLinearConfig, GrowthStrategy, Metric, Objective


class TestGBDTConfigValidation:
    """Tests for GBDTConfig parameter validation."""

    def test_n_estimators_must_be_positive(self) -> None:
        """n_estimators=0 raises error."""
        with pytest.raises(ValueError, match="n_estimators"):
            GBDTConfig(n_estimators=0)

    def test_learning_rate_must_be_positive(self) -> None:
        """learning_rate<=0 raises error."""
        with pytest.raises(ValueError, match="learning_rate"):
            GBDTConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate"):
            GBDTConfig(learning_rate=-0.1)

    def test_subsample_must_be_in_0_1(self) -> None:
        """subsample must be in (0, 1]."""
        with pytest.raises(ValueError, match="subsample"):
            GBDTConfig(subsample=0.0)
        with pytest.raises(ValueError, match="subsample"):
            GBDTConfig(subsample=1.5)

    def test_colsample_bytree_must_be_in_0_1(self) -> None:
        """colsample_bytree must be in (0, 1]."""
        with pytest.raises(ValueError, match="colsample_bytree"):
            GBDTConfig(colsample_bytree=0.0)
        with pytest.raises(ValueError, match="colsample_bytree"):
            GBDTConfig(colsample_bytree=1.5)

    def test_l1_must_be_non_negative(self) -> None:
        """l1<0 raises error."""
        with pytest.raises(ValueError, match="l1"):
            GBDTConfig(l1=-0.1)

    def test_l2_must_be_non_negative(self) -> None:
        """l2<0 raises error."""
        with pytest.raises(ValueError, match="l2"):
            GBDTConfig(l2=-0.1)

    def test_invalid_objective_type_rejected(self) -> None:
        """Non-Objective types are rejected."""
        with pytest.raises(TypeError):
            GBDTConfig(objective=cast(Any, "squared"))

    def test_invalid_metric_type_rejected(self) -> None:
        """Non-Metric types are rejected."""
        with pytest.raises(TypeError):
            GBDTConfig(metric=cast(Any, "rmse"))
        with pytest.raises(TypeError):
            GBDTConfig(metric=cast(Any, Objective.squared()))


class TestGBLinearConfigValidation:
    """Tests for GBLinearConfig parameter validation."""

    def test_n_estimators_must_be_positive(self) -> None:
        """n_estimators=0 raises error."""
        with pytest.raises(ValueError, match="n_estimators"):
            GBLinearConfig(n_estimators=0)

    def test_learning_rate_must_be_positive(self) -> None:
        """learning_rate<=0 raises error."""
        with pytest.raises(ValueError, match="learning_rate"):
            GBLinearConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate"):
            GBLinearConfig(learning_rate=-0.1)

    def test_l1_must_be_non_negative(self) -> None:
        """l1<0 raises error."""
        with pytest.raises(ValueError, match="l1"):
            GBLinearConfig(l1=-0.1)

    def test_l2_must_be_non_negative(self) -> None:
        """l2<0 raises error."""
        with pytest.raises(ValueError, match="l2"):
            GBLinearConfig(l2=-0.1)


class TestGBDTConfigCombinations:
    """Test realistic configuration combinations."""

    def test_binary_classification(self) -> None:
        """Binary classification config works."""
        config = GBDTConfig(
            n_estimators=100,
            learning_rate=0.1,
            objective=Objective.logistic(),
            metric=Metric.logloss(),
        )
        assert config.objective == Objective.logistic()

    def test_multiclass_classification(self) -> None:
        """Multiclass classification config works."""
        config = GBDTConfig(
            objective=Objective.softmax(n_classes=5),
            metric=Metric.accuracy_at(threshold=0.5),
        )
        assert config.objective == Objective.softmax(n_classes=5)

    def test_leafwise_growth(self) -> None:
        """Leafwise growth strategy works."""
        config = GBDTConfig(growth_strategy=GrowthStrategy.Leafwise)
        assert config.growth_strategy == GrowthStrategy.Leafwise

    def test_linear_leaves(self) -> None:
        """Linear leaves config works."""
        config = GBDTConfig(linear_leaves=True, linear_l2=0.1)
        assert config.linear_leaves is True
        assert config.linear_l2 == pytest.approx(0.1)


class TestGBLinearConfigCombinations:
    """Test realistic GBLinear configuration combinations."""

    def test_elastic_net_style(self) -> None:
        """Elastic net style regularization works."""
        config = GBLinearConfig(l1=0.3, l2=0.7)
        assert config.l1 == pytest.approx(0.3)
        assert config.l2 == pytest.approx(0.7)
