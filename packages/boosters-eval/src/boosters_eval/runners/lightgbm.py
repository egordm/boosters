"""Benchmark runner for LightGBM."""

from __future__ import annotations

import time
from typing import Any

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

# =============================================================================
# Quantile Regression Helpers
# =============================================================================


def _pinball_loss_objective(alpha: float) -> Any:
    """Create a LightGBM custom objective for single-quantile pinball loss.

    Uses constant Hessian approximation (pinball loss is piecewise linear).
    """

    def _objective(preds: np.ndarray, train_data: Any) -> tuple[np.ndarray, np.ndarray]:
        y_true = train_data.get_label().astype(np.float64)
        pred = preds.astype(np.float64)
        residual = y_true - pred

        # Gradient: 1_{residual<0} - alpha  (derivative of pinball loss w.r.t. pred)
        grad = (residual < 0.0).astype(np.float64) - alpha
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


def _build_lgb_params(config: BenchmarkConfig, *, seed: int, for_quantile: bool = False) -> dict[str, Any]:
    """Build LightGBM parameter dict from config."""
    tc = config.training
    params: dict[str, Any] = {
        "boosting_type": "gbdt",
        "learning_rate": tc.learning_rate,
        "max_depth": tc.max_depth,
        "num_leaves": tc.num_leaves,
        "lambda_l1": tc.reg_alpha,
        "lambda_l2": tc.reg_lambda,
        "min_sum_hessian_in_leaf": tc.min_child_weight,
        "min_data_in_leaf": tc.min_samples_leaf,
        "bagging_fraction": tc.subsample,
        "bagging_freq": 1 if tc.subsample < 1.0 else 0,
        "feature_fraction": tc.colsample_bytree,
        "max_bin": tc.max_bins,
        "n_jobs": tc.n_threads,
        "seed": seed,
        "verbose": -1,
        "force_col_wise": True,
        "early_stopping_round": None,
    }

    if for_quantile:
        params["metric"] = "None"
    else:
        match config.dataset.task:
            case Task.REGRESSION:
                params["objective"] = "regression"
                params["metric"] = "rmse"
            case Task.BINARY:
                params["objective"] = "binary"
                params["metric"] = "binary_logloss"
            case Task.MULTICLASS:
                params["objective"] = "multiclass"
                params["metric"] = "multi_logloss"
                params["num_class"] = config.dataset.n_classes

    return params


# =============================================================================
# Training Functions
# =============================================================================


def _run_quantile(
    config: BenchmarkConfig,
    data: RunData,
    *,
    linear_tree: bool = False,
    seed: int,
    timing_mode: bool,
    measure_memory: bool,
) -> BenchmarkResult:
    """Run quantile regression with one model per quantile (one_output_per_tree strategy).

    This matches XGBoost's multi_strategy='one_output_per_tree' approach for fair comparison.
    Each quantile gets its own tree per boosting round.
    """
    import lightgbm as lgb

    quantiles = resolve_quantiles(config)
    if quantiles is None:
        raise ValueError("Quantiles must be specified for quantile regression")

    quantiles = np.unique(np.sort(quantiles))

    feature_names: list[str] | str = "auto"
    if data.feature_names is not None:
        feature_names = list(data.feature_names)

    # Create training dataset (shared across all quantiles)
    dtrain = lgb.Dataset(
        data.x_train,
        label=data.y_train,
        weight=data.sample_weight_train,
        feature_name=feature_names,
        categorical_feature=data.categorical_features,
        free_raw_data=False,
        params={"verbose": -1},
    )
    dvalid = lgb.Dataset(
        data.x_valid,
        label=data.y_valid,
        weight=data.sample_weight_valid,
        feature_name=feature_names,
        categorical_feature=data.categorical_features,
        reference=dtrain,
        free_raw_data=False,
        params={"verbose": -1},
    )

    # Optional warmup (single quantile)
    if timing_mode:
        dtrain_small = lgb.Dataset(
            data.x_train[:100],
            label=data.y_train[:100],
            weight=data.sample_weight_train[:100] if data.sample_weight_train is not None else None,
            feature_name=feature_names,
            categorical_feature=data.categorical_features,
            free_raw_data=False,
            params={"verbose": -1},
        )
        warmup_params = _build_lgb_params(config, seed=seed, for_quantile=True)
        warmup_params["objective"] = _pinball_loss_objective(float(quantiles[0]))
        if linear_tree:
            warmup_params["linear_tree"] = True
            warmup_params["linear_lambda"] = config.training.linear_l2
        lgb.train(
            warmup_params,
            dtrain_small,
            num_boost_round=min(5, config.training.n_estimators),
            callbacks=[lgb.log_evaluation(0)],
        )

    # Train one model per quantile (one_output_per_tree strategy)
    _maybe_start_memory(measure_memory=measure_memory)

    models: list[Any] = []
    start_train = time.perf_counter()

    for q_idx, alpha in enumerate(quantiles):
        params = _build_lgb_params(config, seed=seed + q_idx, for_quantile=True)
        params["objective"] = _pinball_loss_objective(float(alpha))

        if linear_tree:
            params["linear_tree"] = True
            params["linear_lambda"] = config.training.linear_l2

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=config.training.n_estimators,
            valid_sets=[dvalid],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        models.append(model)

    train_time = time.perf_counter() - start_train

    # Predict with all models
    start_predict = time.perf_counter()
    y_preds = [model.predict(data.x_valid) for model in models]
    predict_time = time.perf_counter() - start_predict

    # Stack predictions: shape (n_rows, n_quantiles)
    y_pred = np.column_stack(y_preds).astype(np.float32)

    peak_memory = _maybe_get_peak_memory(measure_memory=measure_memory)

    metrics = compute_metrics(
        task=Task.QUANTILE_REGRESSION,
        y_true=data.y_valid,
        y_pred=y_pred,
        n_classes=None,
        sample_weight=data.sample_weight_valid,
        quantiles=quantiles,
    )

    return BenchmarkResult(
        config_name=config.name,
        library=LightGBMRunner.name,
        seed=seed,
        task=Task.QUANTILE_REGRESSION.value,
        booster_type=BoosterType.LINEAR_TREES.value if linear_tree else BoosterType.GBDT.value,
        dataset_name=config.dataset.name,
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        peak_memory_mb=peak_memory,
        dataset_primary_metric=config.dataset.primary_metric,
    )


def _run_standard(
    config: BenchmarkConfig,
    data: RunData,
    *,
    linear_tree: bool = False,
    seed: int,
    timing_mode: bool,
    measure_memory: bool,
) -> BenchmarkResult:
    """Run standard regression/classification with LightGBM."""
    import lightgbm as lgb

    params = _build_lgb_params(config, seed=seed, for_quantile=False)

    if linear_tree:
        params["linear_tree"] = True
        params["linear_lambda"] = config.training.linear_l2

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
    if timing_mode:
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
            num_boost_round=min(5, config.training.n_estimators),
            callbacks=[lgb.log_evaluation(0)],
        )

    # Train
    _maybe_start_memory(measure_memory=measure_memory)

    start_train = time.perf_counter()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=config.training.n_estimators,
        valid_sets=[dvalid],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    train_time = time.perf_counter() - start_train

    # Predict
    start_predict = time.perf_counter()
    y_pred = model.predict(data.x_valid)
    predict_time = time.perf_counter() - start_predict

    peak_memory = _maybe_get_peak_memory(measure_memory=measure_memory)

    metrics = compute_metrics(
        task=config.dataset.task,
        y_true=data.y_valid,
        y_pred=np.asarray(y_pred, dtype=np.float32),
        n_classes=config.dataset.n_classes,
        sample_weight=data.sample_weight_valid,
    )

    return BenchmarkResult(
        config_name=config.name,
        library=LightGBMRunner.name,
        seed=seed,
        task=config.dataset.task.value,
        booster_type=BoosterType.LINEAR_TREES.value if linear_tree else BoosterType.GBDT.value,
        dataset_name=config.dataset.name,
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        peak_memory_mb=peak_memory,
        dataset_primary_metric=config.dataset.primary_metric,
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
        linear_tree = config.booster_type == BoosterType.LINEAR_TREES

        if config.dataset.task == Task.QUANTILE_REGRESSION:
            return _run_quantile(
                config,
                data,
                linear_tree=linear_tree,
                seed=seed,
                timing_mode=timing_mode,
                measure_memory=measure_memory,
            )

        return _run_standard(
            config,
            data,
            linear_tree=linear_tree,
            seed=seed,
            timing_mode=timing_mode,
            measure_memory=measure_memory,
        )
