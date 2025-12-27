"""Tests for configuration dataclasses validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from boosters_eval.config import (
    BenchmarkConfig,
    BoosterType,
    DatasetConfig,
    GrowthStrategy,
    SuiteConfig,
    Task,
    TrainingConfig,
)


class TestTaskEnum:
    """Tests for Task enum."""

    def test_task_values(self) -> None:
        """Test Task enum has expected values."""
        assert Task.REGRESSION.value == "regression"
        assert Task.BINARY.value == "binary"
        assert Task.MULTICLASS.value == "multiclass"

    def test_task_is_string_enum(self) -> None:
        """Test Task can be used as string in comparisons."""
        # Task inherits from str, so equality works with string values
        assert Task.REGRESSION == "regression"
        assert Task.BINARY == "binary"


class TestBoosterTypeEnum:
    """Tests for BoosterType enum."""

    def test_booster_type_values(self) -> None:
        """Test BoosterType enum has expected values."""
        assert BoosterType.GBDT.value == "gbdt"
        assert BoosterType.GBLINEAR.value == "gblinear"
        assert BoosterType.LINEAR_TREES.value == "linear_trees"


class TestGrowthStrategyEnum:
    """Tests for GrowthStrategy enum."""

    def test_growth_strategy_values(self) -> None:
        """Test GrowthStrategy enum has expected values."""
        assert GrowthStrategy.DEPTHWISE.value == "depthwise"
        assert GrowthStrategy.LEAFWISE.value == "leafwise"


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are populated correctly."""
        config = TrainingConfig()
        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
        assert config.reg_lambda == 1.0
        assert config.reg_alpha == 0.0
        assert config.min_child_weight == 1.0
        assert config.subsample == 1.0
        assert config.colsample_bytree == 1.0
        assert config.n_threads == 1
        assert config.growth_strategy == GrowthStrategy.DEPTHWISE

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        config = TrainingConfig(
            n_estimators=50,
            max_depth=8,
            learning_rate=0.05,
            growth_strategy=GrowthStrategy.LEAFWISE,
        )
        assert config.n_estimators == 50
        assert config.max_depth == 8
        assert config.learning_rate == 0.05
        assert config.growth_strategy == GrowthStrategy.LEAFWISE

    def test_invalid_learning_rate_negative(self) -> None:
        """Test negative learning_rate raises ValidationError."""
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.1)

    def test_invalid_learning_rate_zero(self) -> None:
        """Test zero learning_rate raises ValidationError."""
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0.0)

    def test_invalid_n_estimators_zero(self) -> None:
        """Test zero n_estimators raises ValidationError."""
        with pytest.raises(ValidationError, match="n_estimators must be positive"):
            TrainingConfig(n_estimators=0)

    def test_invalid_n_estimators_negative(self) -> None:
        """Test negative n_estimators raises ValidationError."""
        with pytest.raises(ValidationError, match="n_estimators must be positive"):
            TrainingConfig(n_estimators=-10)

    def test_frozen(self) -> None:
        """Test TrainingConfig is immutable."""
        config = TrainingConfig()
        with pytest.raises(ValidationError):
            config.n_estimators = 50  # type: ignore[misc]

    def test_config_roundtrip(self) -> None:
        """Test serialize/deserialize produces same config."""
        config = TrainingConfig(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
        )
        data = config.model_dump()
        restored = TrainingConfig(**data)
        assert restored == config


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_minimal_config(self) -> None:
        """Test DatasetConfig with minimal required fields."""
        config = DatasetConfig(
            name="test",
            task=Task.REGRESSION,
            loader=lambda: (None, None),  # type: ignore[return-value]
        )
        assert config.name == "test"
        assert config.task == Task.REGRESSION
        assert config.n_classes is None
        assert config.subsample is None

    def test_multiclass_config(self) -> None:
        """Test DatasetConfig for multiclass task."""
        config = DatasetConfig(
            name="iris",
            task=Task.MULTICLASS,
            loader=lambda: (None, None),  # type: ignore[return-value]
            n_classes=3,
        )
        assert config.n_classes == 3


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_minimal_config(self) -> None:
        """Test BenchmarkConfig with minimal required fields."""
        dataset = DatasetConfig(
            name="test",
            task=Task.REGRESSION,
            loader=lambda: (None, None),  # type: ignore[return-value]
        )
        config = BenchmarkConfig(name="test-config", dataset=dataset)
        assert config.name == "test-config"
        assert config.training.n_estimators == 100  # default TrainingConfig
        assert config.booster_type == BoosterType.GBDT

    def test_custom_training(self) -> None:
        """Test BenchmarkConfig with custom TrainingConfig."""
        dataset = DatasetConfig(
            name="test",
            task=Task.BINARY,
            loader=lambda: (None, None),  # type: ignore[return-value]
        )
        training = TrainingConfig(n_estimators=50)
        config = BenchmarkConfig(
            name="custom",
            dataset=dataset,
            training=training,
            booster_type=BoosterType.GBLINEAR,
        )
        assert config.training.n_estimators == 50
        assert config.booster_type == BoosterType.GBLINEAR


class TestSuiteConfig:
    """Tests for SuiteConfig dataclass."""

    def test_default_values(self) -> None:
        """Test SuiteConfig with defaults."""
        config = SuiteConfig(
            name="test",
            description="Test suite",
            datasets=["california"],
        )
        assert config.n_estimators == 100
        assert config.seeds == [42, 1379, 2716]
        assert config.libraries == ["boosters", "xgboost", "lightgbm"]
        assert config.booster_type == BoosterType.GBDT

    def test_custom_suite(self) -> None:
        """Test SuiteConfig with custom values."""
        config = SuiteConfig(
            name="custom",
            description="Custom suite",
            datasets=["california", "breast_cancer"],
            n_estimators=50,
            seeds=[42],
            libraries=["boosters"],
        )
        assert config.n_estimators == 50
        assert config.seeds == [42]
        assert config.libraries == ["boosters"]
