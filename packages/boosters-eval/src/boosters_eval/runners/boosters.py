"""Benchmark runner for the boosters library."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from boosters_eval.config import BenchmarkConfig, BoosterType, Task
from boosters_eval.metrics import compute_metrics
from boosters_eval.results import BenchmarkResult
from boosters_eval.runners.base import (
    RunContext,
    RunData,
    Runner,
    _maybe_get_peak_memory,
    _maybe_start_memory,
)

if TYPE_CHECKING:
    import boosters


def _create_objective(ctx: RunContext) -> boosters.Objective:
    """Create boosters Objective from run context."""
    import boosters as bst

    match ctx.task:
        case Task.REGRESSION:
            return bst.Objective.squared()
        case Task.QUANTILE_REGRESSION:
            quantiles = ctx.quantiles.tolist() if ctx.quantiles is not None else [0.5]
            return bst.Objective.pinball(quantiles)
        case Task.BINARY:
            return bst.Objective.logistic()
        case Task.MULTICLASS:
            return bst.Objective.softmax(ctx.n_classes or 3)


def _create_gbdt_config(ctx: RunContext, *, linear_leaves: bool = False) -> boosters.GBDTConfig:
    """Create GBDTConfig from run context."""
    import boosters as bst

    growth = (
        bst.GrowthStrategy.Leafwise
        if ctx.task != Task.MULTICLASS  # Default to leafwise except multiclass
        else bst.GrowthStrategy.Depthwise
    )

    return bst.GBDTConfig(
        n_estimators=ctx.n_estimators,
        max_depth=ctx.max_depth,
        learning_rate=ctx.learning_rate,
        l2=ctx.reg_lambda,
        l1=ctx.reg_alpha,
        min_child_weight=ctx.min_child_weight,
        min_samples_leaf=ctx.min_samples_leaf,
        subsample=ctx.subsample,
        colsample_bytree=ctx.colsample_bytree,
        max_bins=ctx.max_bins,
        growth_strategy=growth,
        objective=_create_objective(ctx),
        seed=ctx.seed,
        linear_leaves=linear_leaves,
        linear_l2=ctx.linear_l2 if linear_leaves else 0.01,
    )


def _create_gblinear_config(ctx: RunContext) -> boosters.GBLinearConfig:
    """Create GBLinearConfig from run context."""
    import boosters as bst

    return bst.GBLinearConfig(
        n_estimators=ctx.n_estimators,
        learning_rate=ctx.learning_rate,
        l2=ctx.reg_lambda,
        l1=ctx.reg_alpha,
        update_strategy=bst.GBLinearUpdateStrategy.Sequential,
        objective=_create_objective(ctx),
        seed=ctx.seed,
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
        # Boosters GBLinear does not support NaN - checked at run time via data.has_nans
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

        ctx = RunContext.from_config(config, seed, timing_mode=timing_mode, measure_memory=measure_memory)

        # Boosters GBLinear does not support NaN values
        if config.booster_type == BoosterType.GBLINEAR and data.has_nans:
            raise ValueError("Boosters GBLinear does not support NaN values in features")

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

        # Determine model type and config
        match config.booster_type:
            case BoosterType.GBDT:
                model_cls = bst.GBDTModel
                model_config = _create_gbdt_config(ctx, linear_leaves=False)
            case BoosterType.LINEAR_TREES:
                model_cls = bst.GBDTModel
                model_config = _create_gbdt_config(ctx, linear_leaves=True)
            case BoosterType.GBLINEAR:
                model_cls = bst.GBLinearModel
                model_config = _create_gblinear_config(ctx)

        # Optional warmup for timing
        if ctx.timing_mode:
            warmup_ds = _create_dataset(
                data.x_train[:100],
                data.y_train[:100],
                weights=data.sample_weight_train[:100] if data.sample_weight_train is not None else None,
                categorical_features=data.categorical_features,
                feature_names=data.feature_names,
            )
            _ = model_cls.train(warmup_ds, config=model_config, n_threads=ctx.n_threads)

        # Train
        _maybe_start_memory(measure_memory=ctx.measure_memory)

        start_train = time.perf_counter()
        model = model_cls.train(train_ds, config=model_config, n_threads=ctx.n_threads)
        train_time = time.perf_counter() - start_train

        # Predict
        start_predict = time.perf_counter()
        y_pred = model.predict(valid_ds, n_threads=ctx.n_threads)
        predict_time = time.perf_counter() - start_predict

        peak_memory = _maybe_get_peak_memory(measure_memory=ctx.measure_memory)

        # Compute metrics
        metrics = compute_metrics(
            task=ctx.task,
            y_true=data.y_valid,
            y_pred=np.asarray(y_pred),
            n_classes=ctx.n_classes,
            sample_weight=data.sample_weight_valid,
            quantiles=ctx.quantiles,
        )

        return BenchmarkResult(
            config_name=ctx.config_name,
            library=cls.name,
            seed=seed,
            task=ctx.task.value,
            booster_type=config.booster_type.value,
            dataset_name=ctx.dataset_name,
            metrics=metrics,
            train_time_s=train_time,
            predict_time_s=predict_time,
            peak_memory_mb=peak_memory,
            dataset_primary_metric=ctx.primary_metric,
        )
