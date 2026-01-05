"""Tests for runner system."""

from __future__ import annotations

import numpy as np
import pytest

from boosters_eval.config import (
    BenchmarkConfig,
    BoosterType,
    DatasetConfig,
    LoadedDataset,
    Task,
    TrainingConfig,
)
from boosters_eval.runners import (
    BoostersRunner,
    LightGBMRunner,
    RunData,
    XGBoostRunner,
    get_available_runners,
    get_runner,
)


def make_config(
    task: Task = Task.REGRESSION,
    booster_type: BoosterType = BoosterType.GBDT,
    *,
    quantiles: list[float] | None = None,
) -> BenchmarkConfig:
    """Helper to create a test config."""

    def loader() -> LoadedDataset:
        rng = np.random.default_rng(123)
        x = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)
        return LoadedDataset(x=x, y=y, feature_names=[f"f{i}" for i in range(x.shape[1])])

    return BenchmarkConfig(
        name="test/config",
        dataset=DatasetConfig(
            name="test_ds",
            task=task,
            loader=loader,
            n_classes=3 if task == Task.MULTICLASS else None,
            quantiles=quantiles,
        ),
        training=TrainingConfig(n_estimators=5, max_depth=3),
        booster_type=booster_type,
    )


def make_data(
    task: Task = Task.REGRESSION,
    n_samples: int = 100,
    n_features: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper to create train/validation data."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n_samples, n_features))

    if task == Task.REGRESSION:
        y = rng.standard_normal(n_samples)
    elif task == Task.BINARY:
        y = rng.integers(0, 2, n_samples).astype(float)
    else:
        y = rng.integers(0, 3, n_samples).astype(float)

    split = n_samples * 4 // 5
    return x[:split], x[split:], y[:split], y[split:]


class TestRunnerRegistry:
    """Tests for runner registry."""

    def test_get_available_runners(self) -> None:
        """Test that available runners returns list."""
        runners = get_available_runners()
        assert isinstance(runners, list)
        # All three should be available since they're mandatory dependencies
        assert "boosters" in runners
        assert "xgboost" in runners
        assert "lightgbm" in runners

    def test_get_runner_boosters(self) -> None:
        """Test getting boosters runner."""
        runner = get_runner("boosters")
        assert runner is BoostersRunner

    def test_get_runner_xgboost(self) -> None:
        """Test getting xgboost runner."""
        runner = get_runner("xgboost")
        assert runner is XGBoostRunner

    def test_get_runner_lightgbm(self) -> None:
        """Test getting lightgbm runner."""
        runner = get_runner("lightgbm")
        assert runner is LightGBMRunner

    def test_get_runner_unknown(self) -> None:
        """Test getting unknown runner raises KeyError."""
        with pytest.raises(KeyError, match="Unknown runner"):
            get_runner("catboost")


class TestRunnerSupports:
    """Tests for runner supports() method."""

    def test_boosters_supports_gbdt(self) -> None:
        """Test boosters supports GBDT."""
        config = make_config(booster_type=BoosterType.GBDT)
        assert BoostersRunner.supports(config)

    def test_boosters_supports_gblinear(self) -> None:
        """Test boosters supports GBLinear."""
        config = make_config(booster_type=BoosterType.GBLINEAR)
        assert BoostersRunner.supports(config)

    def test_boosters_supports_linear_trees(self) -> None:
        """Test boosters supports LINEAR_TREES."""
        config = make_config(booster_type=BoosterType.LINEAR_TREES)
        assert BoostersRunner.supports(config)

    def test_xgboost_supports_gbdt(self) -> None:
        """Test xgboost supports GBDT."""
        config = make_config(booster_type=BoosterType.GBDT)
        assert XGBoostRunner.supports(config)

    def test_xgboost_supports_gblinear(self) -> None:
        """Test xgboost supports GBLinear."""
        config = make_config(booster_type=BoosterType.GBLINEAR)
        assert XGBoostRunner.supports(config)

    def test_xgboost_not_supports_linear_trees(self) -> None:
        """Test xgboost does not support LINEAR_TREES."""
        config = make_config(booster_type=BoosterType.LINEAR_TREES)
        assert not XGBoostRunner.supports(config)

    def test_lightgbm_supports_gbdt(self) -> None:
        """Test lightgbm supports GBDT."""
        config = make_config(booster_type=BoosterType.GBDT)
        assert LightGBMRunner.supports(config)

    def test_lightgbm_supports_linear_trees(self) -> None:
        """Test lightgbm supports LINEAR_TREES."""
        config = make_config(booster_type=BoosterType.LINEAR_TREES)
        assert LightGBMRunner.supports(config)


class TestBoostersRunner:
    """Tests for BoostersRunner."""

    def test_regression(self) -> None:
        """Test boosters runner on regression task."""
        config = make_config(task=Task.REGRESSION)
        x_train, x_valid, y_train, y_valid = make_data(Task.REGRESSION)

        result = BoostersRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert result.library == "boosters"
        assert "rmse" in result.metrics
        assert result.train_time_s is not None
        assert result.train_time_s > 0

    def test_binary(self) -> None:
        """Test boosters runner on binary classification."""
        config = make_config(task=Task.BINARY)
        x_train, x_valid, y_train, y_valid = make_data(Task.BINARY)

        result = BoostersRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert "logloss" in result.metrics
        assert "accuracy" in result.metrics

    def test_multiclass(self) -> None:
        """Test boosters runner on multiclass classification."""
        config = make_config(task=Task.MULTICLASS)
        x_train, x_valid, y_train, y_valid = make_data(Task.MULTICLASS)

        result = BoostersRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert "mlogloss" in result.metrics
        assert "accuracy" in result.metrics

    def test_quantile_regression(self) -> None:
        """Test boosters runner on quantile regression."""
        config = make_config(task=Task.QUANTILE_REGRESSION, quantiles=[0.95, 0.5, 0.05])
        x_train, x_valid, y_train, y_valid = make_data(Task.REGRESSION)

        result = BoostersRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert "pinball" in result.metrics
        assert "rcrps" in result.metrics


class TestXGBoostRunner:
    """Tests for XGBoostRunner."""

    def test_regression(self) -> None:
        """Test xgboost runner on regression task."""
        config = make_config(task=Task.REGRESSION)
        x_train, x_valid, y_train, y_valid = make_data(Task.REGRESSION)

        result = XGBoostRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert result.library == "xgboost"
        assert "rmse" in result.metrics
        assert np.isfinite(result.metrics["rmse"])

    def test_binary(self) -> None:
        """Test xgboost runner on binary classification."""
        config = make_config(task=Task.BINARY)
        x_train, x_valid, y_train, y_valid = make_data(Task.BINARY)

        result = XGBoostRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert "logloss" in result.metrics
        assert np.isfinite(result.metrics["logloss"])

    def test_gblinear(self) -> None:
        """Test xgboost runner with gblinear booster."""
        config = make_config(booster_type=BoosterType.GBLINEAR)
        x_train, x_valid, y_train, y_valid = make_data(Task.REGRESSION)

        result = XGBoostRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert result.booster_type == "gblinear"

    def test_quantile_regression_gbtree(self) -> None:
        """Test xgboost runner on quantile regression (gbtree multi-output)."""
        config = make_config(task=Task.QUANTILE_REGRESSION, booster_type=BoosterType.GBDT, quantiles=[0.95, 0.5, 0.05])
        x_train, x_valid, y_train, y_valid = make_data(Task.REGRESSION)

        result = XGBoostRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert "pinball" in result.metrics
        assert "rcrps" in result.metrics
        assert np.isfinite(result.metrics["rcrps"])

    def test_quantile_regression_gblinear(self) -> None:
        """Test xgboost runner on quantile regression (gblinear native multi-quantile)."""
        config = make_config(
            task=Task.QUANTILE_REGRESSION,
            booster_type=BoosterType.GBLINEAR,
            quantiles=[0.1, 0.5, 0.9],
        )
        x_train, x_valid, y_train, y_valid = make_data(Task.REGRESSION)

        result = XGBoostRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert result.booster_type == "gblinear"
        assert "pinball" in result.metrics
        assert "rcrps" in result.metrics


class TestLightGBMRunner:
    """Tests for LightGBMRunner."""

    def test_regression(self) -> None:
        """Test lightgbm runner on regression task."""
        config = make_config(task=Task.REGRESSION)
        x_train, x_valid, y_train, y_valid = make_data(Task.REGRESSION)

        result = LightGBMRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert result.library == "lightgbm"
        assert "rmse" in result.metrics
        assert np.isfinite(result.metrics["rmse"])

    def test_binary(self) -> None:
        """Test lightgbm runner on binary classification."""
        config = make_config(task=Task.BINARY)
        x_train, x_valid, y_train, y_valid = make_data(Task.BINARY)

        result = LightGBMRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert "logloss" in result.metrics
        assert np.isfinite(result.metrics["logloss"])

    def test_linear_trees(self) -> None:
        """Test lightgbm runner with linear trees."""
        config = make_config(booster_type=BoosterType.LINEAR_TREES)
        x_train, x_valid, y_train, y_valid = make_data(Task.REGRESSION)

        result = LightGBMRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
        )

        assert result.booster_type == "linear_trees"


class TestTimingAndMemory:
    """Tests for timing and memory measurement."""

    def test_timing_mode(self) -> None:
        """Test timing mode runs without error."""
        config = make_config()
        x_train, x_valid, y_train, y_valid = make_data()

        result = BoostersRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
            timing_mode=True,
        )

        assert result.train_time_s is not None
        assert result.predict_time_s is not None

    def test_memory_measurement(self) -> None:
        """Test memory measurement captures peak memory."""
        config = make_config()
        x_train, x_valid, y_train, y_valid = make_data()

        result = BoostersRunner.run(
            config,
            RunData(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                categorical_features=[],
                feature_names=None,
            ),
            seed=42,
            measure_memory=True,
        )

        assert result.peak_memory_mb is not None
        assert result.peak_memory_mb >= 0
