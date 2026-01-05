"""Benchmark runner for the boosters library."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from boosters_eval.config import BenchmarkConfig, BoosterType, Task
from boosters_eval.metrics import compute_metrics
from boosters_eval.results import BenchmarkResult
from boosters_eval.runners.base import (
    RunData,
    Runner,
    _maybe_get_peak_memory,
    _maybe_start_memory,
    resolve_quantiles,
)

if TYPE_CHECKING:
    import boosters


def _create_objective(config: BenchmarkConfig) -> boosters.Objective:
    """Create boosters Objective from the benchmark config."""
    import boosters as bst

    match config.dataset.task:
        case Task.REGRESSION:
            return bst.Objective.squared()
        case Task.QUANTILE_REGRESSION:
            quantiles_arr = resolve_quantiles(config)
            quantiles = quantiles_arr.tolist() if quantiles_arr is not None else [0.5]
            return bst.Objective.pinball(quantiles)
        case Task.BINARY:
            return bst.Objective.logistic()
        case Task.MULTICLASS:
            return bst.Objective.softmax(config.dataset.n_classes or 3)


def _create_gbdt_config(
    config: BenchmarkConfig,
    *,
    seed: int,
    linear_leaves: bool = False,
) -> boosters.GBDTConfig:
    """Create GBDTConfig from the benchmark config."""
    import boosters as bst

    tc = config.training

    growth = (
        bst.GrowthStrategy.Leafwise
        if config.dataset.task != Task.MULTICLASS  # Default to leafwise except multiclass
        else bst.GrowthStrategy.Depthwise
    )

    return bst.GBDTConfig(
        n_estimators=tc.n_estimators,
        max_depth=tc.max_depth,
        learning_rate=tc.learning_rate,
        l2=tc.reg_lambda,
        l1=tc.reg_alpha,
        min_child_weight=tc.min_child_weight,
        min_samples_leaf=tc.min_samples_leaf,
        subsample=tc.subsample,
        colsample_bytree=tc.colsample_bytree,
        max_bins=tc.max_bins,
        growth_strategy=growth,
        objective=_create_objective(config),
        seed=seed,
        linear_leaves=linear_leaves,
        linear_l2=tc.linear_l2 if linear_leaves else 0.01,
    )


def _create_gblinear_config(config: BenchmarkConfig, *, seed: int) -> boosters.GBLinearConfig:
    """Create GBLinearConfig from the benchmark config."""
    import boosters as bst

    tc = config.training

    return bst.GBLinearConfig(
        n_estimators=tc.n_estimators,
        learning_rate=tc.learning_rate,
        l2=tc.reg_lambda,
        l1=tc.reg_alpha,
        update_strategy=bst.GBLinearUpdateStrategy.Sequential,
        objective=_create_objective(config),
        seed=seed,
    )


def _create_dataset(
    x: np.ndarray,
    y: np.ndarray,
    *,
    weights: np.ndarray | None,
    categorical_features: list[int],
    feature_names: list[str] | None,
) -> boosters.Dataset:
    """Create a boosters Dataset with proper dtypes."""
    import boosters as bst

    return bst.Dataset(
        features=x.astype("float32"),
        labels=y.astype("float32"),
        weights=weights.astype("float32") if weights is not None else None,
        categorical_features=categorical_features,
        feature_names=feature_names,
    )


class BoostersRunner(Runner):
    """Runner for the boosters library."""

    name = "boosters"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if this runner supports the given configuration."""
        # Note: Boosters GBLinear supports NaN feature values (treated as missing).
        return config.booster_type in (
            BoosterType.GBDT,
            BoosterType.GBLINEAR,
            BoosterType.LINEAR_TREES,
        )

    @classmethod
    def run(
        cls,
        config: BenchmarkConfig,
        data: RunData,
        seed: int,
        *,
        timing_mode: bool = False,
        measure_memory: bool = False,
    ) -> BenchmarkResult:
        """Train and evaluate a boosters model."""
        import boosters as bst

        tc = config.training
        task = config.dataset.task
        n_classes = config.dataset.n_classes
        quantiles = resolve_quantiles(config)

        # Boosters GBLinear supports NaN feature values (treated as missing/0 contribution).

        # Create datasets
        train_ds = _create_dataset(
            data.x_train,
            data.y_train,
            weights=data.sample_weight_train,
            categorical_features=data.categorical_features,
            feature_names=data.feature_names,
        )
        valid_ds = _create_dataset(
            data.x_valid,
            data.y_valid,
            weights=data.sample_weight_valid,
            categorical_features=data.categorical_features,
            feature_names=data.feature_names,
        )

        # Optional warmup for timing
        warmup_ds = None
        if timing_mode:
            warmup_ds = _create_dataset(
                data.x_train[:100],
                data.y_train[:100],
                weights=data.sample_weight_train[:100] if data.sample_weight_train is not None else None,
                categorical_features=data.categorical_features,
                feature_names=data.feature_names,
            )

        # Train
        _maybe_start_memory(measure_memory=measure_memory)

        start_train = time.perf_counter()

        match config.booster_type:
            case BoosterType.GBDT:
                model_config = _create_gbdt_config(config, seed=seed, linear_leaves=False)
                if warmup_ds is not None:
                    _ = bst.GBDTModel.train(warmup_ds, config=model_config, n_threads=tc.n_threads)
                model = bst.GBDTModel.train(train_ds, config=model_config, n_threads=tc.n_threads)
            case BoosterType.LINEAR_TREES:
                model_config = _create_gbdt_config(config, seed=seed, linear_leaves=True)
                if warmup_ds is not None:
                    _ = bst.GBDTModel.train(warmup_ds, config=model_config, n_threads=tc.n_threads)
                model = bst.GBDTModel.train(train_ds, config=model_config, n_threads=tc.n_threads)
            case BoosterType.GBLINEAR:
                model_config = _create_gblinear_config(config, seed=seed)
                if warmup_ds is not None:
                    _ = bst.GBLinearModel.train(warmup_ds, config=model_config, n_threads=tc.n_threads)
                model = bst.GBLinearModel.train(train_ds, config=model_config, n_threads=tc.n_threads)

        train_time = time.perf_counter() - start_train

        # Predict
        start_predict = time.perf_counter()
        y_pred = model.predict(valid_ds, n_threads=tc.n_threads)
        predict_time = time.perf_counter() - start_predict

        peak_memory = _maybe_get_peak_memory(measure_memory=measure_memory)

        # Compute metrics
        metrics = compute_metrics(
            task=task,
            y_true=data.y_valid,
            y_pred=np.asarray(y_pred),
            n_classes=n_classes,
            sample_weight=data.sample_weight_valid,
            quantiles=quantiles,
        )

        return BenchmarkResult(
            config_name=config.name,
            library=cls.name,
            seed=seed,
            task=task.value,
            booster_type=config.booster_type.value,
            dataset_name=config.dataset.name,
            metrics=metrics,
            train_time_s=train_time,
            predict_time_s=predict_time,
            peak_memory_mb=peak_memory,
            dataset_primary_metric=config.dataset.primary_metric,
        )
