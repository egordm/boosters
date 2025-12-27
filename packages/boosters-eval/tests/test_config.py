"""Tests for configuration validation - critical path tests only."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from boosters_eval.config import (
    BenchmarkConfig,
    BoosterType,
    DatasetConfig,
    SuiteConfig,
    Task,
    TrainingConfig,
)


class TestTrainingConfigValidation:
    """Tests for TrainingConfig validation - critical path."""

    def test_invalid_learning_rate(self) -> None:
        """Test that invalid learning_rate values raise ValidationError."""
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-0.1)

        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0.0)

    def test_invalid_n_estimators(self) -> None:
        """Test that invalid n_estimators values raise ValidationError."""
        with pytest.raises(ValidationError, match="n_estimators must be positive"):
            TrainingConfig(n_estimators=0)

        with pytest.raises(ValidationError, match="n_estimators must be positive"):
            TrainingConfig(n_estimators=-10)

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


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig composition."""

    def test_config_composition(self) -> None:
        """Test BenchmarkConfig correctly composes nested configs."""
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
        assert config.dataset.task == Task.BINARY


class TestSuiteConfig:
    """Tests for SuiteConfig."""

    def test_suite_config(self) -> None:
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
        assert len(config.datasets) == 2
