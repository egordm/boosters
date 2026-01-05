"""Benchmark runner for XGBoost."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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


@dataclass(frozen=True, slots=True)
class _CategoricalData:
    """Prepared dataframes for XGBoost categorical features."""

    train: pd.DataFrame
    valid: pd.DataFrame
    train_small: pd.DataFrame
    enabled: bool = True


def _prepare_categoricals(data: RunData) -> _CategoricalData | None:
    """Convert categorical features to pandas Categorical dtype if needed.

    Returns None if no categorical features are present.
    """
    if not data.categorical_features:
        return None

    col_names = data.feature_names or [f"f{i}" for i in range(data.x_train.shape[1])]
    columns = pd.Index(col_names)

    def to_categorical_df(x: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(x, columns=columns)
        for idx in data.categorical_features:
            col = col_names[idx]
            df[col] = pd.Categorical(df[col].astype("int32"))
        return df

    return _CategoricalData(
        train=to_categorical_df(data.x_train),
        valid=to_categorical_df(data.x_valid),
        train_small=to_categorical_df(data.x_train[:100]),
    )


def _build_xgb_params(ctx: RunContext, *, booster: str) -> dict[str, Any]:
    """Build XGBoost parameter dict from run context."""
    params: dict[str, Any] = {
        "booster": booster,
        "n_estimators": ctx.n_estimators,
        "learning_rate": ctx.learning_rate,
        "reg_lambda": ctx.reg_lambda,
        "reg_alpha": ctx.reg_alpha,
        "n_jobs": ctx.n_threads,
        "random_state": ctx.seed,
        "verbosity": 0,
    }

    if booster == "gbtree":
        params |= {
            "max_depth": ctx.max_depth,
            "min_child_weight": ctx.min_child_weight,
            "subsample": ctx.subsample,
            "colsample_bytree": ctx.colsample_bytree,
            "max_bin": ctx.max_bins,
            "tree_method": "hist",
        }

    return params


def _run_quantile(
    ctx: RunContext,
    data: RunData,
    *,
    booster: str,
) -> BenchmarkResult:
    """Run quantile regression with XGBoost's built-in objective."""
    import xgboost as xgb

    quantiles = ctx.quantiles
    if quantiles is None:
        raise ValueError("Quantiles must be specified for quantile regression")

    n_quantiles = len(quantiles)
    cat_data = _prepare_categoricals(data) if booster == "gbtree" else None

    # Build params with quantile objective
    params = _build_xgb_params(ctx, booster=booster) | {
        "multi_strategy": "one_output_per_tree",
        "objective": "reg:quantileerror",
        "quantile_alpha": quantiles.tolist(),
        "disable_default_eval_metric": True,
    }
    if cat_data:
        params["enable_categorical"] = True

    # Select train/valid data
    x_train = cat_data.train if cat_data else data.x_train
    x_valid = cat_data.valid if cat_data else data.x_valid
    x_small = cat_data.train_small if cat_data else data.x_train[:100]

    # Optional warmup
    if ctx.timing_mode:
        warmup = xgb.XGBRegressor(**{**params, "n_estimators": min(5, ctx.n_estimators)})
        warmup.fit(
            x_small,
            data.y_train[:100],
            sample_weight=data.sample_weight_train[:100] if data.sample_weight_train is not None else None,
            verbose=False,
        )

    # Train
    _maybe_start_memory(measure_memory=ctx.measure_memory)

    start_train = time.perf_counter()
    model = xgb.XGBRegressor(**params)
    model.fit(
        x_train,
        data.y_train,
        sample_weight=data.sample_weight_train,
        eval_set=[(x_valid, data.y_valid)],
        sample_weight_eval_set=[data.sample_weight_valid],
        verbose=False,
    )
    train_time = time.perf_counter() - start_train

    # Predict
    start_predict = time.perf_counter()
    y_pred = model.predict(x_valid)
    predict_time = time.perf_counter() - start_predict

    # Reshape predictions to (n_rows, n_quantiles)
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(len(data.y_valid), n_quantiles)

    peak_memory = _maybe_get_peak_memory(measure_memory=ctx.measure_memory)

    metrics = compute_metrics(
        task=Task.QUANTILE_REGRESSION,
        y_true=data.y_valid,
        y_pred=y_pred_arr,
        n_classes=None,
        sample_weight=data.sample_weight_valid,
        quantiles=quantiles,
    )

    return BenchmarkResult(
        config_name=ctx.config_name,
        library=XGBoostRunner.name,
        seed=ctx.seed,
        task=Task.QUANTILE_REGRESSION.value,
        booster_type=BoosterType.GBDT.value if booster == "gbtree" else BoosterType.GBLINEAR.value,
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
    booster: str,
) -> BenchmarkResult:
    """Run standard regression/classification with XGBoost."""
    import xgboost as xgb

    cat_data = _prepare_categoricals(data) if booster == "gbtree" else None

    # Build objective-specific params
    match ctx.task:
        case Task.REGRESSION:
            objective = "reg:squarederror"
        case Task.BINARY:
            objective = "binary:logistic"
        case Task.MULTICLASS:
            objective = "multi:softprob"
        case _:
            raise ValueError(f"Unexpected task: {ctx.task}")

    # Use low-level xgb.train for standard tasks (matches original behavior)
    params: dict[str, Any] = {
        "booster": booster,
        "eta": ctx.learning_rate,
        "max_depth": ctx.max_depth if booster == "gbtree" else 0,
        "lambda": ctx.reg_lambda,
        "alpha": ctx.reg_alpha,
        "subsample": ctx.subsample,
        "colsample_bytree": ctx.colsample_bytree,
        "max_bin": ctx.max_bins,
        "tree_method": "hist",
        "nthread": ctx.n_threads,
        "seed": ctx.seed,
        "verbosity": 0,
        "objective": objective,
    }

    if booster == "gbtree":
        params["min_child_weight"] = ctx.min_child_weight

    if ctx.task == Task.MULTICLASS:
        params["num_class"] = ctx.n_classes

    if cat_data:
        params["enable_categorical"] = True

    # Create DMatrix objects
    x_train = cat_data.train if cat_data else data.x_train
    x_valid = cat_data.valid if cat_data else data.x_valid
    x_small = cat_data.train_small if cat_data else data.x_train[:100]

    dtrain = xgb.DMatrix(
        x_train,
        label=data.y_train,
        weight=data.sample_weight_train,
        enable_categorical=bool(cat_data),
    )
    dvalid = xgb.DMatrix(
        x_valid,
        label=data.y_valid,
        weight=data.sample_weight_valid,
        enable_categorical=bool(cat_data),
    )

    # Optional warmup
    if ctx.timing_mode:
        dtrain_small = xgb.DMatrix(
            x_small,
            label=data.y_train[:100],
            weight=data.sample_weight_train[:100] if data.sample_weight_train is not None else None,
            enable_categorical=bool(cat_data),
        )
        xgb.train(params, dtrain_small, num_boost_round=min(5, ctx.n_estimators))

    # Train
    _maybe_start_memory(measure_memory=ctx.measure_memory)

    start_train = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=ctx.n_estimators)
    train_time = time.perf_counter() - start_train

    # Predict
    start_predict = time.perf_counter()
    y_pred = model.predict(dvalid)
    predict_time = time.perf_counter() - start_predict

    peak_memory = _maybe_get_peak_memory(measure_memory=ctx.measure_memory)

    metrics = compute_metrics(
        task=ctx.task,
        y_true=data.y_valid,
        y_pred=y_pred,
        n_classes=ctx.n_classes,
        sample_weight=data.sample_weight_valid,
    )

    return BenchmarkResult(
        config_name=ctx.config_name,
        library=XGBoostRunner.name,
        seed=ctx.seed,
        task=ctx.task.value,
        booster_type=BoosterType.GBDT.value if booster == "gbtree" else BoosterType.GBLINEAR.value,
        dataset_name=ctx.dataset_name,
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        peak_memory_mb=peak_memory,
        dataset_primary_metric=ctx.primary_metric,
    )


class XGBoostRunner(Runner):
    """Runner for XGBoost."""

    name = "xgboost"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if this runner supports the given configuration."""
        return config.booster_type in (BoosterType.GBDT, BoosterType.GBLINEAR)

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
        """Train and evaluate an XGBoost model."""
        ctx = RunContext.from_config(config, seed, timing_mode=timing_mode, measure_memory=measure_memory)
        booster = "gbtree" if config.booster_type == BoosterType.GBDT else "gblinear"

        if ctx.task == Task.QUANTILE_REGRESSION:
            return _run_quantile(ctx, data, booster=booster)

        return _run_standard(ctx, data, booster=booster)
