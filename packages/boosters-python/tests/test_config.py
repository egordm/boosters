"""Tests for configuration types."""

import pytest

import boosters
from boosters import (
    CategoricalConfig,
    EFBConfig,
    LinearLeavesConfig,
    RegularizationConfig,
    SamplingConfig,
    TreeConfig,
)


class TestTreeConfig:
    """Tests for TreeConfig."""

    def test_default_values(self):
        """Test TreeConfig has correct defaults."""
        config = TreeConfig()
        assert config.max_depth == -1
        assert config.n_leaves == 31
        assert config.min_samples_leaf == 1
        assert config.min_gain_to_split == 0.0
        assert config.growth_strategy == "depthwise"

    def test_custom_values(self):
        """Test TreeConfig accepts custom values."""
        config = TreeConfig(max_depth=6, n_leaves=64)
        assert config.max_depth == 6
        assert config.n_leaves == 64

    def test_growth_strategy_leafwise(self):
        """Test leafwise growth strategy is valid."""
        config = TreeConfig(growth_strategy="leafwise")
        assert config.growth_strategy == "leafwise"

    def test_invalid_growth_strategy(self):
        """Test invalid growth strategy raises ValueError."""
        with pytest.raises(ValueError, match="growth_strategy"):
            TreeConfig(growth_strategy="invalid")

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

    def test_default_values(self):
        """Test RegularizationConfig has correct defaults."""
        config = RegularizationConfig()
        assert config.l1 == 0.0
        assert config.l2 == 1.0
        assert config.min_hessian == 1.0

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

    def test_default_values(self):
        """Test SamplingConfig has correct defaults."""
        config = SamplingConfig()
        assert config.subsample == 1.0
        assert config.colsample == 1.0
        assert config.colsample_bylevel == 1.0
        assert config.goss_alpha == 0.0
        assert config.goss_beta == 0.0

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

    def test_default_values(self):
        """Test CategoricalConfig has correct defaults."""
        config = CategoricalConfig()
        assert config.max_categories == 256
        assert config.min_category_count == 10
        assert config.max_onehot == 4

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

    def test_default_values(self):
        """Test EFBConfig has correct defaults."""
        config = EFBConfig()
        assert config.enable is True
        assert config.max_conflict_rate == 0.0

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

    def test_default_values(self):
        """Test LinearLeavesConfig has correct defaults."""
        config = LinearLeavesConfig()
        assert config.enable is False
        assert config.l2 == 0.01
        assert config.l1 == 0.0
        assert config.max_iter == 10
        assert config.tolerance == 1e-6
        assert config.min_samples == 50

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
