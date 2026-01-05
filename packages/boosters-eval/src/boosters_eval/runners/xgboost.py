"""Benchmark runner for XGBoost."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from boosters_eval.config import BenchmarkConfig, BoosterType, Task
from boosters_eval.metrics import compute_metrics
from boosters_eval.results import BenchmarkResult
from boosters_eval.runners.base import RunData, Runner, _maybe_get_peak_memory, _maybe_start_memory


def _pinball_loss_multi_objective(
    *,
    quantiles: np.ndarray,
    n_quantiles: int,
) -> Any:
    # xgboost.train calls objective(preds, dtrain)
    # where both preds and dtrain.get_label() are flattened 1D.
    def _objective(preds: np.ndarray, dtrain: Any) -> tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(dtrain.get_label())
        weights = np.asarray(dtrain.get_weight())
        if weights.size == 0:
            weights = None

        y_true = y_true.reshape(-1)
        preds = np.asarray(preds).reshape(-1)

        n_items = y_true.shape[0]
        n_rows = n_items // n_quantiles

        y_true_2d = y_true.reshape(n_rows, n_quantiles)
        preds_2d = preds.reshape(n_rows, n_quantiles)

        errors = preds_2d - y_true_2d
        left_mask = errors < 0
        right_mask = ~left_mask

        gradient = (quantiles * left_mask + (1 - quantiles) * right_mask) * errors
        hessian = np.ones_like(preds_2d)

        if weights is not None:
            w2d = weights.reshape(n_rows, 1)
            gradient *= w2d
            hessian *= w2d

        return gradient / n_quantiles, hessian / n_quantiles

    _objective.__name__ = "pinball_loss_multi_objective"  # pyright: ignore[reportAttributeAccessIssue]
    return _objective


def _reshape_quantile_predictions(y_pred: np.ndarray, *, n_rows: int, n_quantiles: int) -> np.ndarray:
    arr = np.asarray(y_pred)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1 and arr.size == n_rows * n_quantiles:
        return arr.reshape(n_rows, n_quantiles)
    raise ValueError(f"Unexpected quantile prediction shape {arr.shape}")


def _prepare_xgbtree_inputs_for_categoricals(data: RunData) -> tuple[Any, Any, Any, bool]:
    if not data.categorical_features:
        return data.x_train, data.x_valid, data.x_train[:100], False

    import pandas as pd

    col_names = data.feature_names or [f"f{i}" for i in range(data.x_train.shape[1])]
    columns = pd.Index(col_names)

    train_df = pd.DataFrame(data.x_train, columns=columns)
    valid_df = pd.DataFrame(data.x_valid, columns=columns)
    small_df = pd.DataFrame(data.x_train[:100], columns=columns)

    for idx in data.categorical_features:
        col = col_names[idx]
        train_df[col] = pd.Categorical(train_df[col].astype("int32"))
        valid_df[col] = pd.Categorical(valid_df[col].astype("int32"))
        small_df[col] = pd.Categorical(small_df[col].astype("int32"))

    return train_df, valid_df, small_df, True


def _run_quantile_gbtree(
    *,
    config: BenchmarkConfig,
    data: RunData,
    seed: int,
    timing_mode: bool,
    measure_memory: bool,
    quantiles_arr: np.ndarray,
) -> BenchmarkResult:
    import xgboost as xgb

    tc = config.training
    n_quantiles = int(quantiles_arr.shape[0])

    x_train, x_valid, x_train_small, enable_categorical = _prepare_xgbtree_inputs_for_categoricals(data)

    y_train = np.repeat(data.y_train[:, np.newaxis], repeats=n_quantiles, axis=1)
    y_valid = np.repeat(data.y_valid[:, np.newaxis], repeats=n_quantiles, axis=1)

    dtrain = xgb.DMatrix(
        x_train,
        label=y_train,
        weight=data.sample_weight_train,
        enable_categorical=enable_categorical,
    )
    dvalid = xgb.DMatrix(
        x_valid,
        label=y_valid,
        weight=data.sample_weight_valid,
        enable_categorical=enable_categorical,
    )

    params: dict[str, Any] = {
        "booster": "gbtree",
        "multi_strategy": "one_output_per_tree",
        "eta": tc.learning_rate,
        "max_depth": tc.max_depth,
        "lambda": tc.reg_lambda,
        "alpha": tc.reg_alpha,
        "subsample": tc.subsample,
        "colsample_bytree": tc.colsample_bytree,
        "max_bin": tc.max_bins,
        "tree_method": "hist",
        "nthread": tc.n_threads,
        "seed": seed,
        "verbosity": 0,
        "min_child_weight": tc.min_child_weight,
        # Needed for the non-degenerate hessian approximation.
        "max_delta_step": 0.1,
    }
    if enable_categorical:
        params["enable_categorical"] = True

    obj = _pinball_loss_multi_objective(
        quantiles=quantiles_arr,
        n_quantiles=n_quantiles,
    )

    if timing_mode:
        dtrain_small = xgb.DMatrix(
            x_train_small,
            label=y_train[:100],
            weight=(data.sample_weight_train[:100] if data.sample_weight_train is not None else None),
            enable_categorical=enable_categorical,
        )
        xgb.train(params, dtrain_small, num_boost_round=min(5, tc.n_estimators), obj=obj)

    _maybe_start_memory(measure_memory=measure_memory)

    start_train = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=tc.n_estimators, obj=obj)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    y_pred = model.predict(dvalid)
    predict_time = time.perf_counter() - start_predict

    y_pred_arr = _reshape_quantile_predictions(
        y_pred,
        n_rows=data.y_valid.shape[0],
        n_quantiles=n_quantiles,
    )

    peak_memory = _maybe_get_peak_memory(measure_memory=measure_memory)

    metrics = compute_metrics(
        task=config.dataset.task,
        y_true=data.y_valid,
        y_pred=y_pred_arr,
        n_classes=config.dataset.n_classes,
        sample_weight=data.sample_weight_valid,
        quantiles=quantiles_arr,
    )

    return BenchmarkResult(
        config_name=config.name,
        library=XGBoostRunner.name,
        seed=seed,
        task=config.dataset.task.value,
        booster_type=config.booster_type.value,
        dataset_name=config.dataset.name,
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        peak_memory_mb=peak_memory,
        dataset_primary_metric=config.dataset.primary_metric,
    )


def _run_quantile_gblinear(
    *,
    config: BenchmarkConfig,
    data: RunData,
    seed: int,
    timing_mode: bool,
    measure_memory: bool,
    quantiles_arr: np.ndarray,
) -> BenchmarkResult:
    import xgboost as xgb

    tc = config.training
    n_quantiles = int(quantiles_arr.shape[0])

    params = {
        "booster": "gblinear",
        "n_estimators": tc.n_estimators,
        "learning_rate": tc.learning_rate,
        "reg_lambda": tc.reg_lambda,
        "reg_alpha": tc.reg_alpha,
        "n_jobs": tc.n_threads,
        "random_state": seed,
        "verbosity": 0,
        "objective": "reg:quantileerror",
        "quantile_alpha": [float(q) for q in quantiles_arr.tolist()],
    }

    if timing_mode:
        warmup = xgb.XGBRegressor(**{**params, "n_estimators": min(5, tc.n_estimators)})
        warmup.fit(
            data.x_train[:100],
            data.y_train[:100],
            sample_weight=(data.sample_weight_train[:100] if data.sample_weight_train is not None else None),
            verbose=False,
        )

    _maybe_start_memory(measure_memory=measure_memory)

    start_train = time.perf_counter()
    model = xgb.XGBRegressor(**params)
    model.fit(
        data.x_train,
        data.y_train,
        sample_weight=data.sample_weight_train,
        eval_set=[(data.x_valid, data.y_valid)],
        sample_weight_eval_set=[data.sample_weight_valid],
        verbose=False,
    )
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    y_pred = model.predict(data.x_valid)
    predict_time = time.perf_counter() - start_predict

    y_pred_arr = _reshape_quantile_predictions(
        y_pred,
        n_rows=data.y_valid.shape[0],
        n_quantiles=n_quantiles,
    )

    peak_memory = _maybe_get_peak_memory(measure_memory=measure_memory)

    metrics = compute_metrics(
        task=config.dataset.task,
        y_true=data.y_valid,
        y_pred=y_pred_arr,
        n_classes=config.dataset.n_classes,
        sample_weight=data.sample_weight_valid,
        quantiles=quantiles_arr,
    )

    return BenchmarkResult(
        config_name=config.name,
        library=XGBoostRunner.name,
        seed=seed,
        task=config.dataset.task.value,
        booster_type=config.booster_type.value,
        dataset_name=config.dataset.name,
        metrics=metrics,
        train_time_s=train_time,
        predict_time_s=predict_time,
        peak_memory_mb=peak_memory,
        dataset_primary_metric=config.dataset.primary_metric,
    )


def _run_standard(
    *,
    config: BenchmarkConfig,
    data: RunData,
    seed: int,
    timing_mode: bool,
    measure_memory: bool,
) -> BenchmarkResult:
    import xgboost as xgb

    task = config.dataset.task
    tc = config.training
    booster = "gbtree" if config.booster_type == BoosterType.GBDT else "gblinear"

    params = {
        "booster": booster,
        "eta": tc.learning_rate,
        "max_depth": tc.max_depth if booster == "gbtree" else 0,
        "lambda": tc.reg_lambda,
        "alpha": tc.reg_alpha,
        "subsample": tc.subsample,
        "colsample_bytree": tc.colsample_bytree,
        "max_bin": tc.max_bins,
        "tree_method": "hist",
        "nthread": tc.n_threads,
        "seed": seed,
        "verbosity": 0,
    }

    if booster == "gbtree":
        params["min_child_weight"] = tc.min_child_weight

    use_categoricals = bool(data.categorical_features) and booster == "gbtree"
    if use_categoricals:
        import pandas as pd

        col_names = data.feature_names or [f"f{i}" for i in range(data.x_train.shape[1])]
        columns = pd.Index(col_names)
        train_df = pd.DataFrame(data.x_train, columns=columns)
        valid_df = pd.DataFrame(data.x_valid, columns=columns)

        for idx in data.categorical_features:
            col = col_names[idx]
            train_df[col] = pd.Categorical(train_df[col].astype("int32"))
            valid_df[col] = pd.Categorical(valid_df[col].astype("int32"))

        dtrain = xgb.DMatrix(train_df, label=data.y_train, weight=data.sample_weight_train, enable_categorical=True)
        dvalid = xgb.DMatrix(valid_df, label=data.y_valid, weight=data.sample_weight_valid, enable_categorical=True)
        params["enable_categorical"] = True
    else:
        dtrain = xgb.DMatrix(data.x_train, label=data.y_train, weight=data.sample_weight_train)
        dvalid = xgb.DMatrix(data.x_valid, label=data.y_valid, weight=data.sample_weight_valid)

    if task == Task.REGRESSION:
        params["objective"] = "reg:squarederror"
    elif task == Task.BINARY:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = config.dataset.n_classes

    if timing_mode:
        if use_categoricals:
            import pandas as pd

            col_names = data.feature_names or [f"f{i}" for i in range(data.x_train.shape[1])]
            small_df = pd.DataFrame(data.x_train[:100], columns=pd.Index(col_names))
            for idx in data.categorical_features:
                col = col_names[idx]
                small_df[col] = pd.Categorical(small_df[col].astype("int32"))
            dtrain_small = xgb.DMatrix(
                small_df,
                label=data.y_train[:100],
                weight=(data.sample_weight_train[:100] if data.sample_weight_train is not None else None),
                enable_categorical=True,
            )
        else:
            dtrain_small = xgb.DMatrix(
                data.x_train[:100],
                label=data.y_train[:100],
                weight=(data.sample_weight_train[:100] if data.sample_weight_train is not None else None),
            )
        xgb.train(params, dtrain_small, num_boost_round=min(5, tc.n_estimators))

    _maybe_start_memory(measure_memory=measure_memory)

    start_train = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=tc.n_estimators)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    y_pred = model.predict(dvalid)
    predict_time = time.perf_counter() - start_predict

    peak_memory = _maybe_get_peak_memory(measure_memory=measure_memory)

    metrics = compute_metrics(
        task=task,
        y_true=data.y_valid,
        y_pred=y_pred,
        n_classes=config.dataset.n_classes,
        sample_weight=data.sample_weight_valid,
    )

    return BenchmarkResult(
        config_name=config.name,
        library=XGBoostRunner.name,
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


class XGBoostRunner(Runner):
    """Runner for XGBoost."""

    name = "xgboost"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Return whether this runner can run the given benchmark config."""
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
    ) -> BenchmarkResult:  # type: ignore[override]
        """Train and evaluate an XGBoost model for this benchmark config."""
        if config.dataset.task == Task.QUANTILE_REGRESSION:
            quantiles = config.dataset.quantiles or config.training.quantiles or [0.1, 0.5, 0.9]
            quantiles_arr = np.asarray(quantiles, dtype=np.float64)
            if config.booster_type == BoosterType.GBDT:
                return _run_quantile_gbtree(
                    config=config,
                    data=data,
                    seed=seed,
                    timing_mode=timing_mode,
                    measure_memory=measure_memory,
                    quantiles_arr=quantiles_arr,
                )
            return _run_quantile_gblinear(
                config=config,
                data=data,
                seed=seed,
                timing_mode=timing_mode,
                measure_memory=measure_memory,
                quantiles_arr=quantiles_arr,
            )

        return _run_standard(
            config=config,
            data=data,
            seed=seed,
            timing_mode=timing_mode,
            measure_memory=measure_memory,
        )
