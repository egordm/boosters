"""Tests for GBLinearConfig."""

import pytest

from boosters import (
    GBLinearConfig,
    HuberLoss,
    LogisticLoss,
    LogLoss,
    Mae,
    Rmse,
    SquaredLoss,
)


class TestGBLinearConfigDefaults:
    """Test that GBLinearConfig has sensible defaults."""

    def test_default_construction(self):
        """Default GBLinearConfig should have required fields set."""
        config = GBLinearConfig()

        # Just verify fields exist and have valid types
        assert isinstance(config.n_estimators, int)
        assert config.n_estimators > 0
        assert isinstance(config.learning_rate, float)
        assert config.learning_rate > 0
        assert isinstance(config.l1, float)
        assert config.l1 >= 0
        assert isinstance(config.l2, float)
        assert config.l2 >= 0
        assert config.early_stopping_rounds is None or isinstance(config.early_stopping_rounds, int)
        assert isinstance(config.seed, int)

    def test_default_objective(self):
        """Default objective should be present."""
        config = GBLinearConfig()
        # Default should be some objective
        assert config.objective is not None

    def test_default_metric_is_none(self):
        """Default metric should be None (auto-selected based on objective)."""
        config = GBLinearConfig()
        assert config.metric is None


class TestGBLinearConfigCustomization:
    """Test customization of GBLinearConfig."""

    def test_custom_n_estimators(self):
        """Custom n_estimators should be stored."""
        config = GBLinearConfig(n_estimators=500)

        assert config.n_estimators == 500

    def test_custom_learning_rate(self):
        """Custom learning_rate should be stored."""
        config = GBLinearConfig(learning_rate=0.1)

        assert config.learning_rate == 0.1

    def test_custom_objective(self):
        """Custom objective should be stored."""
        config = GBLinearConfig(objective=HuberLoss(delta=2.0))

        assert isinstance(config.objective, HuberLoss)
        assert config.objective.delta == 2.0

    def test_custom_metric(self):
        """Custom metric should be stored."""
        config = GBLinearConfig(metric=Mae())

        assert isinstance(config.metric, Mae)

    def test_custom_l1_regularization(self):
        """Custom L1 regularization should be stored."""
        config = GBLinearConfig(l1=0.5)

        assert config.l1 == 0.5

    def test_custom_l2_regularization(self):
        """Custom L2 regularization should be stored."""
        config = GBLinearConfig(l2=2.0)

        assert config.l2 == 2.0

    def test_early_stopping(self):
        """Early stopping rounds can be set."""
        config = GBLinearConfig(early_stopping_rounds=50)

        assert config.early_stopping_rounds == 50

    def test_custom_seed(self):
        """Custom seed should be stored."""
        config = GBLinearConfig(seed=12345)

        assert config.seed == 12345


class TestGBLinearConfigValidation:
    """Test validation in GBLinearConfig."""

    def test_n_estimators_must_be_positive(self):
        """n_estimators=0 should raise error."""
        with pytest.raises(ValueError, match="n_estimators"):
            GBLinearConfig(n_estimators=0)

    def test_learning_rate_must_be_positive(self):
        """learning_rate<=0 should raise error."""
        with pytest.raises(ValueError, match="learning_rate"):
            GBLinearConfig(learning_rate=0.0)

        with pytest.raises(ValueError, match="learning_rate"):
            GBLinearConfig(learning_rate=-0.1)

    def test_l1_must_be_non_negative(self):
        """l1<0 should raise error."""
        with pytest.raises(ValueError, match="l1"):
            GBLinearConfig(l1=-0.1)

    def test_l2_must_be_non_negative(self):
        """l2<0 should raise error."""
        with pytest.raises(ValueError, match="l2"):
            GBLinearConfig(l2=-0.1)

    def test_invalid_objective_rejected(self):
        """Non-objective types should be rejected."""
        with pytest.raises(TypeError):
            GBLinearConfig(objective="invalid")  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            GBLinearConfig(objective=123)  # type: ignore[arg-type]

    def test_invalid_metric_rejected(self):
        """Non-metric types should be rejected."""
        with pytest.raises(TypeError):
            GBLinearConfig(metric="invalid")  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            GBLinearConfig(metric=SquaredLoss())  # type: ignore[arg-type]


class TestGBLinearConfigRepr:
    """Test string representation."""

    def test_repr_contains_key_info(self):
        """Repr should contain key configuration info."""
        config = GBLinearConfig(n_estimators=200, learning_rate=0.1, l1=0.5, l2=2.5)
        repr_str = repr(config)

        assert "GBLinearConfig" in repr_str
        assert "200" in repr_str
        assert "0.1" in repr_str
        assert "0.5" in repr_str
        assert "2.5" in repr_str


class TestGBLinearConfigCombinations:
    """Test realistic configuration combinations."""

    def test_classification_config(self):
        """Test a typical binary classification configuration."""
        config = GBLinearConfig(
            n_estimators=100,
            learning_rate=0.3,
            objective=LogisticLoss(),
            metric=LogLoss(),
            l2=0.1,
        )

        assert isinstance(config.objective, LogisticLoss)
        assert isinstance(config.metric, LogLoss)

    def test_sparse_regression_config(self):
        """Test regression with L1 for sparsity."""
        config = GBLinearConfig(
            n_estimators=200,
            learning_rate=0.1,
            objective=SquaredLoss(),
            metric=Rmse(),
            l1=1.0,  # High L1 for sparse weights
            l2=0.0,
            early_stopping_rounds=10,
        )

        assert config.l1 == 1.0
        assert config.l2 == 0.0
        assert config.early_stopping_rounds == 10

    def test_elastic_net_style_config(self):
        """Test elastic net style with both L1 and L2."""
        config = GBLinearConfig(
            n_estimators=100,
            learning_rate=0.5,
            l1=0.3,
            l2=0.7,
        )

        assert config.l1 == 0.3
        assert config.l2 == 0.7
