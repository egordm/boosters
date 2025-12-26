"""Tests for GBDTConfig with flat structure."""

import pytest

from boosters import (
    GBDTConfig,
    GrowthStrategy,
    Metric,
    Objective,
)


class TestGBDTConfigDefaults:
    """Test that GBDTConfig has sensible defaults."""

    def test_default_construction(self):
        """Default GBDTConfig should have required fields set."""
        config = GBDTConfig()

        # Just verify fields exist and have valid types
        assert isinstance(config.n_estimators, int)
        assert config.n_estimators > 0
        assert isinstance(config.learning_rate, float)
        assert config.learning_rate > 0
        assert config.early_stopping_rounds is None or isinstance(config.early_stopping_rounds, int)
        assert isinstance(config.seed, int)

    def test_default_objective(self):
        """Default objective should be present."""
        config = GBDTConfig()
        # Default should be Squared
        assert config.objective is not None
        assert config.objective == Objective.Squared()

    def test_default_metric_is_none(self):
        """Default metric should be None (auto-selected based on objective)."""
        config = GBDTConfig()
        assert config.metric is None

    def test_default_tree_params(self):
        """Default tree params should be set."""
        config = GBDTConfig()
        assert isinstance(config.max_depth, int)
        assert isinstance(config.n_leaves, int)
        assert isinstance(config.growth_strategy, GrowthStrategy)

    def test_default_regularization_params(self):
        """Default regularization params should be set."""
        config = GBDTConfig()
        assert isinstance(config.l1, float)
        assert isinstance(config.l2, float)
        assert isinstance(config.min_child_weight, float)
        assert isinstance(config.min_gain_to_split, float)

    def test_default_sampling_params(self):
        """Default sampling params should be set."""
        config = GBDTConfig()
        assert isinstance(config.subsample, float)
        assert isinstance(config.colsample_bytree, float)
        assert isinstance(config.colsample_bylevel, float)

    def test_default_linear_leaves_disabled(self):
        """Default linear_leaves should be False (disabled)."""
        config = GBDTConfig()
        assert config.linear_leaves is False


class TestGBDTConfigCustomization:
    """Test customization of GBDTConfig."""

    def test_custom_n_estimators(self):
        """Custom n_estimators should be stored."""
        config = GBDTConfig(n_estimators=500)
        assert config.n_estimators == 500

    def test_custom_learning_rate(self):
        """Custom learning_rate should be stored."""
        config = GBDTConfig(learning_rate=0.1)
        assert config.learning_rate == 0.1

    def test_custom_objective(self):
        """Custom objective should be stored."""
        config = GBDTConfig(objective=Objective.huber(delta=2.0))
        assert config.objective == Objective.Huber(delta=2.0)
        assert config.objective.delta == 2.0  # type: ignore[attr-defined]

    def test_custom_metric(self):
        """Custom metric should be stored."""
        config = GBDTConfig(metric=Metric.mae())
        assert config.metric == Metric.Mae()

    def test_custom_tree_params(self):
        """Custom tree params should be stored."""
        config = GBDTConfig(max_depth=8, n_leaves=64, growth_strategy=GrowthStrategy.Leafwise)
        assert config.max_depth == 8
        assert config.n_leaves == 64
        assert config.growth_strategy == GrowthStrategy.Leafwise

    def test_custom_regularization(self):
        """Custom regularization params should be stored."""
        config = GBDTConfig(l1=0.5, l2=2.0, min_child_weight=5.0, min_gain_to_split=0.1)
        assert config.l1 == 0.5
        assert config.l2 == 2.0
        assert config.min_child_weight == 5.0
        assert config.min_gain_to_split == 0.1

    def test_custom_sampling(self):
        """Custom sampling params should be stored."""
        config = GBDTConfig(subsample=0.8, colsample_bytree=0.9, colsample_bylevel=0.7)
        assert config.subsample == 0.8
        assert config.colsample_bytree == 0.9
        assert config.colsample_bylevel == 0.7

    def test_linear_leaves_enabled(self):
        """Linear leaves can be enabled with params."""
        config = GBDTConfig(linear_leaves=True, linear_l2=0.1, linear_l1=0.05)
        assert config.linear_leaves is True
        assert config.linear_l2 == 0.1
        assert config.linear_l1 == 0.05

    def test_early_stopping(self):
        """Early stopping rounds can be set."""
        config = GBDTConfig(early_stopping_rounds=50)
        assert config.early_stopping_rounds == 50

    def test_custom_seed(self):
        """Custom seed should be stored."""
        config = GBDTConfig(seed=12345)
        assert config.seed == 12345


class TestGBDTConfigValidation:
    """Test validation in GBDTConfig."""

    def test_n_estimators_must_be_positive(self):
        """n_estimators=0 should raise error."""
        with pytest.raises(ValueError, match="n_estimators"):
            GBDTConfig(n_estimators=0)

    def test_learning_rate_must_be_positive(self):
        """learning_rate<=0 should raise error."""
        with pytest.raises(ValueError, match="learning_rate"):
            GBDTConfig(learning_rate=0.0)

        with pytest.raises(ValueError, match="learning_rate"):
            GBDTConfig(learning_rate=-0.1)

    def test_invalid_objective_rejected(self):
        """Non-objective types should be rejected."""
        with pytest.raises(TypeError):
            GBDTConfig(objective="invalid")  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            GBDTConfig(objective=123)  # type: ignore[arg-type]

    def test_invalid_metric_rejected(self):
        """Non-metric types should be rejected."""
        with pytest.raises(TypeError):
            GBDTConfig(metric="invalid")  # type: ignore[arg-type]

        with pytest.raises(TypeError):
            GBDTConfig(metric=Objective.squared())  # type: ignore[arg-type]


class TestGBDTConfigRepr:
    """Test string representation."""

    def test_repr_contains_key_info(self):
        """Repr should contain key configuration info."""
        config = GBDTConfig(n_estimators=200, learning_rate=0.1)
        repr_str = repr(config)

        assert "GBDTConfig" in repr_str
        assert "200" in repr_str
        assert "0.1" in repr_str


class TestGBDTConfigCombinations:
    """Test realistic configuration combinations."""

    def test_classification_config(self):
        """Test a typical binary classification configuration."""
        config = GBDTConfig(
            n_estimators=100,
            learning_rate=0.1,
            objective=Objective.logistic(),
            metric=Metric.logloss(),
            max_depth=6,
            l2=1.0,
        )

        assert config.objective == Objective.Logistic()
        assert config.metric == Metric.LogLoss()

    def test_multiclass_config(self):
        """Test a multiclass classification configuration."""
        config = GBDTConfig(
            n_estimators=100,
            learning_rate=0.1,
            objective=Objective.softmax(n_classes=5),
            metric=Metric.accuracy(),
        )

        assert config.objective == Objective.Softmax(n_classes=5)
        assert config.objective.n_classes == 5  # type: ignore[attr-defined]

    def test_regression_with_subsampling(self):
        """Test regression config with subsampling."""
        config = GBDTConfig(
            n_estimators=500,
            learning_rate=0.05,
            objective=Objective.squared(),
            metric=Metric.rmse(),
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
        )

        assert config.subsample == 0.8
        assert config.early_stopping_rounds == 20
