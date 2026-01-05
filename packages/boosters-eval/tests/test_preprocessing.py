"""Tests for dataset preprocessing."""

from __future__ import annotations

import numpy as np

from boosters_eval.config import BenchmarkConfig, BoosterType, DatasetConfig, LoadedDataset, Task, TrainingConfig
from boosters_eval.preprocessing import prepare_run_data


def _make_loaded_with_categoricals() -> LoadedDataset:
    x = np.arange(100, dtype=np.float32).reshape(20, 5)
    y = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    return LoadedDataset(
        x=x,
        y=y,
        feature_names=["a", "b", "c", "d", "e"],
        categorical_features=[1, 3],
    )


def test_prepare_run_data_drops_categoricals_for_gblinear() -> None:
    loaded = _make_loaded_with_categoricals()

    cfg = BenchmarkConfig(
        name="test",
        dataset=DatasetConfig(name="ds", task=Task.REGRESSION, loader=lambda: loaded),
        training=TrainingConfig(n_estimators=5),
        booster_type=BoosterType.GBLINEAR,
        libraries=["boosters"],
    )

    run_data = prepare_run_data(config=cfg, loaded=loaded, seed=42)
    assert run_data.x_train.shape[1] == 3
    assert run_data.categorical_features == []
    assert run_data.feature_names == ["a", "c", "e"]


def test_prepare_run_data_scales_regression_targets() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 5), dtype=np.float32)
    y = (10.0 + 5.0 * rng.standard_normal(100)).astype(np.float32)
    loaded = LoadedDataset(x=x, y=y)

    cfg = BenchmarkConfig(
        name="test",
        dataset=DatasetConfig(name="ds", task=Task.REGRESSION, loader=lambda: loaded),
        training=TrainingConfig(n_estimators=5),
        booster_type=BoosterType.GBDT,
        libraries=["boosters"],
    )

    run_data = prepare_run_data(config=cfg, loaded=loaded, seed=42)

    assert np.isfinite(run_data.y_train).all()
    assert abs(float(run_data.y_train.mean())) < 1e-6
    assert abs(float(run_data.y_train.std()) - 1.0) < 1e-6


def test_prepare_run_data_scales_quantile_regression_targets() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 5), dtype=np.float32)
    y = (10.0 + 5.0 * rng.standard_normal(100)).astype(np.float32)
    loaded = LoadedDataset(x=x, y=y)

    cfg = BenchmarkConfig(
        name="test",
        dataset=DatasetConfig(name="ds", task=Task.QUANTILE_REGRESSION, loader=lambda: loaded, quantiles=[0.1, 0.5]),
        training=TrainingConfig(n_estimators=5),
        booster_type=BoosterType.GBLINEAR,
        libraries=["boosters"],
    )

    run_data = prepare_run_data(config=cfg, loaded=loaded, seed=42)
    assert np.isfinite(run_data.y_train).all()
    assert abs(float(run_data.y_train.mean())) < 1e-6
