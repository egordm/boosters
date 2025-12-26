"""Tests for configuration types."""

import pytest

import boosters
from boosters import (
    CategoricalConfig,
    EFBConfig,
    GrowthStrategy,
    LinearLeavesConfig,
    RegularizationConfig,
    SamplingConfig,
    TreeConfig,
)


class TestTreeConfig:
    """Tests for TreeConfig."""

    def test_default_construction(self):
        """Test TreeConfig has defaults set (not checking specific values)."""
        config = TreeConfig()
        # Just verify fields are accessible and have valid types
        assert isinstance(config.max_depth, int)
        assert isinstance(config.n_leaves, int)
        assert config.n_leaves > 0
        assert isinstance(config.min_samples_leaf, int)
        assert isinstance(config.min_gain_to_split, float)
        assert config.growth_strategy in (GrowthStrategy.Depthwise, GrowthStrategy.Leafwise)

    def test_custom_values(self):
        """Test TreeConfig accepts custom values."""
        config = TreeConfig(max_depth=6, n_leaves=64)
        assert config.max_depth == 6
        assert config.n_leaves == 64

    def test_growth_strategy_leafwise(self):
        """Test leafwise growth strategy is valid."""
        config = TreeConfig(growth_strategy=GrowthStrategy.Leafwise)
        assert config.growth_strategy == GrowthStrategy.Leafwise

    def test_invalid_growth_strategy(self):
        """Test invalid growth strategy raises TypeError (enum doesn't accept strings)."""
        with pytest.raises(TypeError):
            # Intentionally passing invalid value to test runtime validation
            TreeConfig(growth_strategy="invalid")  # type: ignore[arg-type]

    def test_invalid_min_gain_to_split(self):
        """Test negative min_gain_to_split raises ValueError."""
        with pytest.raises(ValueError, match="min_gain_to_split"):
            TreeConfig(min_gain_to_split=-1.0)

    def test_repr(self):
        """Test TreeConfig repr is informative."""
        config = TreeConfig()
        assert "TreeConfig" in repr(config)
        assert "max_depth" in repr(config)


class TestRegularizationConfig:
    """Tests for RegularizationConfig."""

    def test_default_construction(self):
        """Test RegularizationConfig has defaults set (not checking specific values)."""
        config = RegularizationConfig()
        # Just verify fields are accessible and have valid types
        assert isinstance(config.l1, float)
        assert config.l1 >= 0
        assert isinstance(config.l2, float)
        assert config.l2 >= 0
        assert isinstance(config.min_hessian, float)

    def test_custom_values(self):
        """Test RegularizationConfig accepts custom values."""
        config = RegularizationConfig(l1=0.5, l2=2.0, min_hessian=0.1)
        assert config.l1 == 0.5
        assert config.l2 == 2.0
        assert config.min_hessian == 0.1

    def test_invalid_l1(self):
        """Test negative l1 raises ValueError."""
        with pytest.raises(ValueError, match="l1"):
            RegularizationConfig(l1=-0.1)

    def test_invalid_l2(self):
        """Test negative l2 raises ValueError."""
        with pytest.raises(ValueError, match="l2"):
            RegularizationConfig(l2=-0.1)


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_construction(self):
        """Test SamplingConfig has defaults set (not checking specific values)."""
        config = SamplingConfig()
        # Just verify fields are accessible and have valid types
        assert isinstance(config.subsample, float)
        assert 0 < config.subsample <= 1
        assert isinstance(config.colsample, float)
        assert 0 < config.colsample <= 1
        assert isinstance(config.colsample_bylevel, float)
        assert isinstance(config.goss_alpha, float)
        assert isinstance(config.goss_beta, float)

    def test_custom_values(self):
        """Test SamplingConfig accepts custom values."""
        config = SamplingConfig(subsample=0.8, colsample=0.8)
        assert config.subsample == 0.8
        assert config.colsample == 0.8

    def test_invalid_subsample_too_high(self):
        """Test subsample > 1 raises ValueError."""
        with pytest.raises(ValueError, match="subsample"):
            SamplingConfig(subsample=1.5)

    def test_invalid_subsample_zero(self):
        """Test subsample = 0 raises ValueError."""
        with pytest.raises(ValueError, match="subsample"):
            SamplingConfig(subsample=0.0)

    def test_invalid_colsample(self):
        """Test invalid colsample raises ValueError."""
        with pytest.raises(ValueError, match="colsample"):
            SamplingConfig(colsample=1.5)


class TestCategoricalConfig:
    """Tests for CategoricalConfig."""

    def test_default_construction(self):
        """Test CategoricalConfig has defaults set (not checking specific values)."""
        config = CategoricalConfig()
        # Just verify fields are accessible and have valid types
        assert isinstance(config.max_categories, int)
        assert config.max_categories > 0
        assert isinstance(config.min_category_count, int)
        assert isinstance(config.max_onehot, int)

    def test_custom_values(self):
        """Test CategoricalConfig accepts custom values."""
        config = CategoricalConfig(max_categories=64)
        assert config.max_categories == 64

    def test_invalid_max_categories(self):
        """Test zero max_categories raises ValueError."""
        with pytest.raises(ValueError, match="max_categories"):
            CategoricalConfig(max_categories=0)


class TestEFBConfig:
    """Tests for EFBConfig."""

    def test_default_construction(self):
        """Test EFBConfig has defaults set (not checking specific values)."""
        config = EFBConfig()
        # Just verify fields are accessible and have valid types
        assert isinstance(config.enable, bool)
        assert isinstance(config.max_conflict_rate, float)
        assert 0 <= config.max_conflict_rate < 1

    def test_custom_values(self):
        """Test EFBConfig accepts custom values."""
        config = EFBConfig(enable=False, max_conflict_rate=0.1)
        assert config.enable is False
        assert config.max_conflict_rate == 0.1

    def test_invalid_max_conflict_rate(self):
        """Test max_conflict_rate >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_conflict_rate"):
            EFBConfig(max_conflict_rate=1.0)


class TestLinearLeavesConfig:
    """Tests for LinearLeavesConfig."""

    def test_default_construction(self):
        """Test LinearLeavesConfig has defaults set (not checking specific values)."""
        config = LinearLeavesConfig()
        # Just verify fields are accessible and have valid types
        assert isinstance(config.enable, bool)
        assert isinstance(config.l2, float)
        assert config.l2 >= 0
        assert isinstance(config.l1, float)
        assert isinstance(config.max_iter, int)
        assert config.max_iter > 0
        assert isinstance(config.tolerance, float)
        assert config.tolerance > 0
        assert isinstance(config.min_samples, int)

    def test_custom_values(self):
        """Test LinearLeavesConfig accepts custom values."""
        config = LinearLeavesConfig(enable=True, l2=0.1)
        assert config.enable is True
        assert config.l2 == 0.1

    def test_invalid_tolerance(self):
        """Test non-positive tolerance raises ValueError."""
        with pytest.raises(ValueError, match="tolerance"):
            LinearLeavesConfig(tolerance=0.0)


class TestExportsFromPackage:
    """Test config types are exported from boosters package."""

    def test_all_configs_exported(self):
        """Test all config types are exported from boosters."""
        assert hasattr(boosters, "TreeConfig")
        assert hasattr(boosters, "RegularizationConfig")
        assert hasattr(boosters, "SamplingConfig")
        assert hasattr(boosters, "CategoricalConfig")
        assert hasattr(boosters, "EFBConfig")
        assert hasattr(boosters, "LinearLeavesConfig")
