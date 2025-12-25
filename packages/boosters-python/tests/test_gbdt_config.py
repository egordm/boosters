"""Tests for GBDTConfig."""

import pytest

from boosters import (
    Accuracy,
    CategoricalConfig,
    EFBConfig,
    GBDTConfig,
    HuberLoss,
    LinearLeavesConfig,
    LogisticLoss,
    LogLoss,
    Mae,
    RegularizationConfig,
    Rmse,
    SamplingConfig,
    SoftmaxLoss,
    SquaredLoss,
    TreeConfig,
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
        # Default should be some objective (likely SquaredLoss)
        assert config.objective is not None

    def test_default_metric_is_none(self):
        """Default metric should be None (auto-selected based on objective)."""
        config = GBDTConfig()
        assert config.metric is None

    def test_default_tree_config(self):
        """Default tree config should be present."""
        config = GBDTConfig()
        assert isinstance(config.tree, TreeConfig)

    def test_default_regularization_config(self):
        """Default regularization config should be present."""
        config = GBDTConfig()
        assert isinstance(config.regularization, RegularizationConfig)

    def test_default_sampling_config(self):
        """Default sampling config should be present."""
        config = GBDTConfig()
        assert isinstance(config.sampling, SamplingConfig)

    def test_default_categorical_config(self):
        """Default categorical config should be present."""
        config = GBDTConfig()
        assert isinstance(config.categorical, CategoricalConfig)

    def test_default_efb_config(self):
        """Default EFB config should be present."""
        config = GBDTConfig()
        assert isinstance(config.efb, EFBConfig)

    def test_default_linear_leaves_none(self):
        """Default linear_leaves should be None (disabled)."""
        config = GBDTConfig()
        assert config.linear_leaves is None


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
        config = GBDTConfig(objective=HuberLoss(delta=2.0))

        assert isinstance(config.objective, HuberLoss)
        assert config.objective.delta == 2.0

    def test_custom_metric(self):
        """Custom metric should be stored."""
        config = GBDTConfig(metric=Mae())

        assert isinstance(config.metric, Mae)

    def test_custom_tree_config(self):
        """Custom tree config should replace default."""
        config = GBDTConfig(tree=TreeConfig(max_depth=8, n_leaves=64))

        assert config.tree.max_depth == 8
        assert config.tree.n_leaves == 64

    def test_custom_regularization(self):
        """Custom regularization should replace default."""
        config = GBDTConfig(regularization=RegularizationConfig(l1=0.5, l2=2.0))

        assert config.regularization.l1 == 0.5
        assert config.regularization.l2 == 2.0

    def test_custom_sampling(self):
        """Custom sampling should replace default."""
        config = GBDTConfig(sampling=SamplingConfig(subsample=0.8, colsample=0.9))

        assert config.sampling.subsample == 0.8
        assert config.sampling.colsample == 0.9

    def test_custom_categorical(self):
        """Custom categorical config should replace default."""
        config = GBDTConfig(categorical=CategoricalConfig(max_categories=128))

        assert config.categorical.max_categories == 128

    def test_custom_efb(self):
        """Custom EFB config should replace default."""
        config = GBDTConfig(efb=EFBConfig(enable=False))

        assert config.efb.enable is False

    def test_linear_leaves_enabled(self):
        """Linear leaves can be enabled with config."""
        config = GBDTConfig(linear_leaves=LinearLeavesConfig(enable=True, l2=0.1))

        assert config.linear_leaves is not None
        assert config.linear_leaves.enable is True
        assert config.linear_leaves.l2 == 0.1

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
            GBDTConfig(metric=SquaredLoss())  # type: ignore[arg-type]


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
            objective=LogisticLoss(),
            metric=LogLoss(),
            tree=TreeConfig(max_depth=6),
            regularization=RegularizationConfig(l2=1.0),
        )

        assert isinstance(config.objective, LogisticLoss)
        assert isinstance(config.metric, LogLoss)

    def test_multiclass_config(self):
        """Test a multiclass classification configuration."""
        config = GBDTConfig(
            n_estimators=100,
            learning_rate=0.1,
            objective=SoftmaxLoss(n_classes=5),
            metric=Accuracy(),
        )

        assert isinstance(config.objective, SoftmaxLoss)
        assert config.objective.n_classes == 5

    def test_regression_with_subsampling(self):
        """Test regression config with subsampling."""
        config = GBDTConfig(
            n_estimators=500,
            learning_rate=0.05,
            objective=SquaredLoss(),
            metric=Rmse(),
            sampling=SamplingConfig(subsample=0.8, colsample=0.8),
            early_stopping_rounds=20,
        )

        assert config.sampling.subsample == 0.8
        assert config.early_stopping_rounds == 20
