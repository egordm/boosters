"""Benchmark runner for LightGBM."""

from __future__ import annotations

import time
from typing import Any

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

# =============================================================================
# Quantile Regression Helpers
# =============================================================================


def _augment_with_quantiles(
    x: np.ndarray,
    *,
    quantiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat rows in block-by-quantile layout and append quantile indicator feature.

    Layout: [all rows for q0] [all rows for q1] ...
    This matches common LightGBM custom-objective examples and enables
    reshape-based gradient computation.

    Returns:
        (x_augmented, quantile_per_row)
    """
    n_rows = x.shape[0]
    n_quantiles = len(quantiles)

    # Stack copies and append quantile column
    x_rep = np.concatenate([x] * n_quantiles, axis=0)
    q_rep = np.repeat(quantiles.astype(np.float32), repeats=n_rows)
    x_aug = np.column_stack([x_rep, q_rep])

    return x_aug, q_rep


def _pinball_loss_objective(quantiles: np.ndarray) -> Any:
    """Create a LightGBM custom objective for multi-quantile pinball loss.

    Assumes block-by-quantile data layout from `_augment_with_quantiles`.
    Uses constant Hessian approximation (pinball loss is piecewise linear).
    """
    alphas = quantiles.astype(np.float64)
    n_quantiles = len(alphas)

    def _objective(preds: np.ndarray, train_data: Any) -> tuple[np.ndarray, np.ndarray]:
        y_true = train_data.get_label().astype(np.float64)
        pred = preds.astype(np.float64)

        # Reshape as (n_quantiles, n_rows)
        y2 = y_true.reshape(n_quantiles, -1)
        p2 = pred.reshape(n_quantiles, -1)
        residual = y2 - p2

        # Gradient: 1_{residual<0} - alpha  (derivative of pinball loss w.r.t. pred)
        grad = ((residual < 0.0).astype(np.float64) - alphas.reshape(-1, 1)).ravel()
        hess = np.ones_like(grad)

        # Apply sample weights if present
        raw_weights = train_data.get_weight()
        if raw_weights is not None:
            weights = np.asarray(raw_weights, dtype=np.float64)
            if weights.size > 0:
                grad *= weights
                hess *= weights

        return grad, hess

    return _objective


# =============================================================================
# Parameter Building
# =============================================================================


def _build_lgb_params(ctx: RunContext, *, for_quantile: bool = False) -> dict[str, Any]:
    """Build LightGBM parameter dict from run context."""
    params: dict[str, Any] = {
        "boosting_type": "gbdt",
        "learning_rate": ctx.learning_rate,
        "max_depth": ctx.max_depth,
        "num_leaves": ctx.num_leaves,
        "lambda_l1": ctx.reg_alpha,
        "lambda_l2": ctx.reg_lambda,
        "min_sum_hessian_in_leaf": ctx.min_child_weight,
        "min_data_in_leaf": ctx.min_samples_leaf,
        "bagging_fraction": ctx.subsample,
        "bagging_freq": 1 if ctx.subsample < 1.0 else 0,
        "feature_fraction": ctx.colsample_bytree,
        "max_bin": ctx.max_bins,
        "n_jobs": ctx.n_threads,
        "seed": ctx.seed,
        "verbose": -1,
        "force_col_wise": True,
        "early_stopping_round": None,
    }

    if for_quantile:
        params["metric"] = "None"
    else:
        match ctx.task:
            case Task.REGRESSION:
                params["objective"] = "regression"
                params["metric"] = "rmse"
            case Task.BINARY:
                params["objective"] = "binary"
                params["metric"] = "binary_logloss"
            case Task.MULTICLASS:
                params["objective"] = "multiclass"
                params["metric"] = "multi_logloss"
                params["num_class"] = ctx.n_classes

    return params


# =============================================================================
# Training Functions
# =============================================================================


def _run_quantile(
    ctx: RunContext,
    data: RunData,
    *,
    linear_tree: bool = False,
) -> BenchmarkResult:
    """Run quantile regression with custom pinball loss objective."""
    import lightgbm as lgb

    quantiles = ctx.quantiles
    if quantiles is None:
        raise ValueError("Quantiles must be specified for quantile regression")

    quantiles = np.unique(np.sort(quantiles))
    n_quantiles = len(quantiles)
    n_rows_valid = len(data.y_valid)

    # Augment data with quantile feature
    x_train_aug, _ = _augment_with_quantiles(data.x_train, quantiles=quantiles)
    y_train_aug = np.tile(data.y_train, reps=n_quantiles)
    w_train_aug = np.tile(data.sample_weight_train, reps=n_quantiles) if data.sample_weight_train is not None else None

    x_valid_aug, _ = _augment_with_quantiles(data.x_valid, quantiles=quantiles)
    y_valid_aug = np.tile(data.y_valid, reps=n_quantiles)
    w_valid_aug = np.tile(data.sample_weight_valid, reps=n_quantiles) if data.sample_weight_valid is not None else None

    # Build feature names with quantile indicator
    feature_names: list[str] | str = "auto"
    if data.feature_names is not None:
        feature_names = [*data.feature_names, "quantile"]

    # Create datasets
    dtrain = lgb.Dataset(
        x_train_aug,
        label=y_train_aug,
        weight=w_train_aug,
        feature_name=feature_names,
        categorical_feature=data.categorical_features,
        free_raw_data=False,
        params={"verbose": -1},
    )
    dvalid = lgb.Dataset(
        x_valid_aug,
        label=y_valid_aug,
        weight=w_valid_aug,
        feature_name=feature_names,
        categorical_feature=data.categorical_features,
        reference=dtrain,
        free_raw_data=False,
        params={"verbose": -1},
    )

    # Build params
    params = _build_lgb_params(ctx, for_quantile=True)
    params["objective"] = _pinball_loss_objective(quantiles)

    if linear_tree:
        params["linear_tree"] = True
        params["linear_lambda"] = ctx.linear_l2

    # Monotone constraint on quantile feature (last column)
    if isinstance(feature_names, list):
        params["monotone_constraints"] = [0] * (len(feature_names) - 1) + [1]

    # Optional warmup
    if ctx.timing_mode:
        dtrain_small = lgb.Dataset(
            x_train_aug[: 100 * n_quantiles],
            label=y_train_aug[: 100 * n_quantiles],
            weight=w_train_aug[: 100 * n_quantiles] if w_train_aug is not None else None,
            feature_name=feature_names,
            categorical_feature=data.categorical_features,
            free_raw_data=False,
            params={"verbose": -1},
        )
        lgb.train(
            params,
            dtrain_small,
            num_boost_round=min(5, ctx.n_estimators),
            callbacks=[lgb.log_evaluation(0)],
        )

    # Train
    _maybe_start_memory(measure_memory=ctx.measure_memory)

    start_train = time.perf_counter()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=ctx.n_estimators,
        valid_sets=[dvalid],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    train_time = time.perf_counter() - start_train

    # Predict
    start_predict = time.perf_counter()
    y_pred_aug = model.predict(x_valid_aug)
    predict_time = time.perf_counter() - start_predict

    # Reshape predictions from block layout to (n_rows, n_quantiles)
    y_pred = np.asarray(y_pred_aug, dtype=np.float32).reshape(n_quantiles, n_rows_valid).T

    peak_memory = _maybe_get_peak_memory(measure_memory=ctx.measure_memory)

    metrics = compute_metrics(
        task=Task.QUANTILE_REGRESSION,
        y_true=data.y_valid,
        y_pred=y_pred,
        n_classes=None,
        sample_weight=data.sample_weight_valid,
        quantiles=quantiles,
    )

    return BenchmarkResult(
        config_name=ctx.config_name,
        library=LightGBMRunner.name,
        seed=ctx.seed,
        task=Task.QUANTILE_REGRESSION.value,
        booster_type=BoosterType.LINEAR_TREES.value if linear_tree else BoosterType.GBDT.value,
        dataset_name=ctx.dataset_name,
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        peak_memory_mb=peak_memory,
        dataset_primary_metric=ctx.primary_metric,
    )


def _run_standard(
    ctx: RunContext,
    data: RunData,
    *,
    linear_tree: bool = False,
) -> BenchmarkResult:
    """Run standard regression/classification with LightGBM."""
    import lightgbm as lgb

    params = _build_lgb_params(ctx, for_quantile=False)

    if linear_tree:
        params["linear_tree"] = True
        params["linear_lambda"] = ctx.linear_l2

    feature_names = data.feature_names or "auto"

    # Create datasets
    dtrain = lgb.Dataset(
        data.x_train,
        label=data.y_train,
        weight=data.sample_weight_train,
        feature_name=feature_names,
        categorical_feature=data.categorical_features,
        params={"verbose": -1},
    )
    dvalid = lgb.Dataset(
        data.x_valid,
        label=data.y_valid,
        weight=data.sample_weight_valid,
        feature_name=feature_names,
        categorical_feature=data.categorical_features,
        reference=dtrain,
        params={"verbose": -1},
    )

    # Optional warmup
    if ctx.timing_mode:
        dtrain_small = lgb.Dataset(
            data.x_train[:100],
            label=data.y_train[:100],
            weight=data.sample_weight_train[:100] if data.sample_weight_train is not None else None,
            feature_name=feature_names,
            categorical_feature=data.categorical_features,
            params={"verbose": -1},
        )
        lgb.train(
            params,
            dtrain_small,
            num_boost_round=min(5, ctx.n_estimators),
            callbacks=[lgb.log_evaluation(0)],
        )

    # Train
    _maybe_start_memory(measure_memory=ctx.measure_memory)

    start_train = time.perf_counter()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=ctx.n_estimators,
        valid_sets=[dvalid],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    train_time = time.perf_counter() - start_train

    # Predict
    start_predict = time.perf_counter()
    y_pred = model.predict(data.x_valid)
    predict_time = time.perf_counter() - start_predict

    peak_memory = _maybe_get_peak_memory(measure_memory=ctx.measure_memory)

    metrics = compute_metrics(
        task=ctx.task,
        y_true=data.y_valid,
        y_pred=np.asarray(y_pred, dtype=np.float32),
        n_classes=ctx.n_classes,
        sample_weight=data.sample_weight_valid,
    )

    return BenchmarkResult(
        config_name=ctx.config_name,
        library=LightGBMRunner.name,
        seed=ctx.seed,
        task=ctx.task.value,
        booster_type=BoosterType.LINEAR_TREES.value if linear_tree else BoosterType.GBDT.value,
        dataset_name=ctx.dataset_name,
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        peak_memory_mb=peak_memory,
        dataset_primary_metric=ctx.primary_metric,
    )


# =============================================================================
# Runner
# =============================================================================


class LightGBMRunner(Runner):
    """Runner for LightGBM.

    Parameter mapping from canonical TrainingConfig:
    - learning_rate -> learning_rate
    - max_depth -> max_depth (also sets num_leaves = 2^max_depth - 1)
    - reg_lambda -> lambda_l2 (L2 regularization)
    - reg_alpha -> lambda_l1 (L1 regularization)
    - min_child_weight -> min_sum_hessian_in_leaf
    - subsample -> bagging_fraction (requires bagging_freq=1)
    - colsample_bytree -> feature_fraction
    """

    name = "lightgbm"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if this runner supports the given configuration."""
        return config.booster_type in (BoosterType.GBDT, BoosterType.LINEAR_TREES)

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
        """Train and evaluate a LightGBM model."""
        ctx = RunContext.from_config(config, seed, timing_mode=timing_mode, measure_memory=measure_memory)
        linear_tree = config.booster_type == BoosterType.LINEAR_TREES

        if ctx.task == Task.QUANTILE_REGRESSION:
            return _run_quantile(ctx, data, linear_tree=linear_tree)

        return _run_standard(ctx, data, linear_tree=linear_tree)
