"""Tests for configuration types."""

import pytest

import boosters
from boosters import GBDTConfig, GBLinearConfig, GrowthStrategy


class TestGrowthStrategy:
    """Tests for GrowthStrategy enum."""

    def test_default_is_depthwise(self):
        """Default growth strategy should be depthwise."""
        config = GBDTConfig()
        assert config.growth_strategy == GrowthStrategy.Depthwise

    def test_leafwise_valid(self):
        """Leafwise growth strategy is valid."""
        config = GBDTConfig(growth_strategy=GrowthStrategy.Leafwise)
        assert config.growth_strategy == GrowthStrategy.Leafwise

    def test_repr(self):
        """Test GrowthStrategy repr."""
        assert "Depthwise" in repr(GrowthStrategy.Depthwise)
        assert "Leafwise" in repr(GrowthStrategy.Leafwise)


class TestGBDTConfig:
    """Tests for GBDTConfig with flat structure."""

    def test_default_construction(self):
        """Test GBDTConfig has sensible defaults."""
        config = GBDTConfig()
        # Core boosting params
        assert isinstance(config.n_estimators, int)
        assert config.n_estimators > 0
        assert isinstance(config.learning_rate, float)
        assert config.learning_rate > 0

        # Tree structure
        assert isinstance(config.max_depth, int)
        assert isinstance(config.n_leaves, int)
        assert config.growth_strategy in (GrowthStrategy.Depthwise, GrowthStrategy.Leafwise)

        # Regularization
        assert isinstance(config.l1, float)
        assert config.l1 >= 0
        assert isinstance(config.l2, float)
        assert config.l2 >= 0
        assert isinstance(config.min_child_weight, float)
        assert isinstance(config.min_gain_to_split, float)

        # Sampling
        assert isinstance(config.subsample, float)
        assert 0 < config.subsample <= 1
        assert isinstance(config.colsample_bytree, float)
        assert isinstance(config.colsample_bylevel, float)

    def test_custom_values(self):
        """Test GBDTConfig accepts custom values."""
        config = GBDTConfig(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=8,
            l2=2.0,
            subsample=0.8,
        )
        assert config.n_estimators == 500
        assert config.learning_rate == 0.1
        assert config.max_depth == 8
        assert config.l2 == 2.0
        assert config.subsample == 0.8

    def test_invalid_n_estimators(self):
        """Test n_estimators=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_estimators"):
            GBDTConfig(n_estimators=0)

    def test_invalid_learning_rate(self):
        """Test learning_rate<=0 raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate"):
            GBDTConfig(learning_rate=0.0)
        with pytest.raises(ValueError, match="learning_rate"):
            GBDTConfig(learning_rate=-0.1)

    def test_invalid_subsample(self):
        """Test invalid subsample raises ValueError."""
        with pytest.raises(ValueError, match="subsample"):
            GBDTConfig(subsample=0.0)
        with pytest.raises(ValueError, match="subsample"):
            GBDTConfig(subsample=1.5)

    def test_invalid_colsample_bytree(self):
        """Test invalid colsample_bytree raises ValueError."""
        with pytest.raises(ValueError, match="colsample_bytree"):
            GBDTConfig(colsample_bytree=0.0)
        with pytest.raises(ValueError, match="colsample_bytree"):
            GBDTConfig(colsample_bytree=1.5)

    def test_invalid_l1(self):
        """Test negative l1 raises ValueError."""
        with pytest.raises(ValueError, match="l1"):
            GBDTConfig(l1=-0.1)

    def test_invalid_l2(self):
        """Test negative l2 raises ValueError."""
        with pytest.raises(ValueError, match="l2"):
            GBDTConfig(l2=-0.1)

    def test_linear_leaves_params(self):
        """Test linear leaves parameters."""
        config = GBDTConfig(linear_leaves=True, linear_l2=0.1, linear_l1=0.05)
        assert config.linear_leaves is True
        assert config.linear_l2 == 0.1
        assert config.linear_l1 == 0.05

    def test_repr(self):
        """Test GBDTConfig repr is informative."""
        config = GBDTConfig()
        assert "GBDTConfig" in repr(config)


class TestGBLinearConfig:
    """Tests for GBLinearConfig."""

    def test_default_construction(self):
        """Test GBLinearConfig has sensible defaults."""
        config = GBLinearConfig()
        assert isinstance(config.n_estimators, int)
        assert config.n_estimators > 0
        assert isinstance(config.learning_rate, float)
        assert config.learning_rate > 0
        assert isinstance(config.l1, float)
        assert config.l1 >= 0
        assert isinstance(config.l2, float)
        assert config.l2 >= 0

    def test_custom_values(self):
        """Test GBLinearConfig accepts custom values."""
        config = GBLinearConfig(n_estimators=200, learning_rate=0.3, l1=0.1, l2=2.0)
        assert config.n_estimators == 200
        assert config.learning_rate == 0.3
        assert config.l1 == 0.1
        assert config.l2 == 2.0


class TestExportsFromPackage:
    """Test config types are exported from boosters package."""

    def test_main_configs_exported(self):
        """Test main config types are exported from boosters."""
        assert hasattr(boosters, "GBDTConfig")
        assert hasattr(boosters, "GBLinearConfig")
        assert hasattr(boosters, "GrowthStrategy")
