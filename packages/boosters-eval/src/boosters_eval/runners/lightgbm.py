"""Benchmark runner for LightGBM."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from boosters_eval.config import BenchmarkConfig, BoosterType, Task
from boosters_eval.metrics import compute_metrics
from boosters_eval.results import BenchmarkResult
from boosters_eval.runners.base import RunData, Runner, _maybe_get_peak_memory, _maybe_start_memory


class LightGBMRunner(Runner):
    """Runner for LightGBM.

    Parameter mapping from canonical TrainingConfig:
    - learning_rate -> learning_rate
    - max_depth -> max_depth (also sets num_leaves = 2^max_depth - 1 for depth-wise equivalence)
    - reg_lambda -> lambda_l2 (L2 regularization)
    - reg_alpha -> lambda_l1 (L1 regularization)
    - min_child_weight -> min_sum_hessian_in_leaf (also set min_data_in_leaf=1 to match XGBoost)
    - subsample -> bagging_fraction (requires bagging_freq=1)
    - colsample_bytree -> feature_fraction
    """

    name = "lightgbm"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Return whether this runner can run the given benchmark config."""
        if config.dataset.task == Task.QUANTILE_REGRESSION:
            return False
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
    ) -> BenchmarkResult:  # type: ignore[override]
        """Train and evaluate a LightGBM model for this benchmark config."""
        import lightgbm as lgb

        task = config.dataset.task
        tc = config.training

        num_leaves = tc.num_leaves

        params: dict[str, Any] = {
            "boosting_type": "gbdt",
            "learning_rate": tc.learning_rate,
            "max_depth": tc.max_depth,
            "num_leaves": num_leaves,
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

        if config.booster_type == BoosterType.LINEAR_TREES:
            params["linear_tree"] = True
            params["linear_lambda"] = tc.linear_l2

        if task == Task.REGRESSION:
            params["objective"] = "regression"
            params["metric"] = "rmse"
        elif task == Task.BINARY:
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
        else:
            params["objective"] = "multiclass"
            params["metric"] = "multi_logloss"
            params["num_class"] = config.dataset.n_classes

        feature_names = data.feature_names or "auto"

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

        if timing_mode:
            dtrain_small = lgb.Dataset(
                data.x_train[:100],
                label=data.y_train[:100],
                weight=(data.sample_weight_train[:100] if data.sample_weight_train is not None else None),
                feature_name=feature_names,
                categorical_feature=data.categorical_features,
                params={"verbose": -1},
            )
            lgb.train(params, dtrain_small, num_boost_round=min(5, tc.n_estimators), callbacks=[lgb.log_evaluation(0)])

        _maybe_start_memory(measure_memory=measure_memory)

        start_train = time.perf_counter()
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=tc.n_estimators,
            valid_sets=[dvalid],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        train_time = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        y_pred = model.predict(data.x_valid)
        predict_time = time.perf_counter() - start_predict

        y_pred_arr = np.asarray(y_pred, dtype=np.float32)
        peak_memory = _maybe_get_peak_memory(measure_memory=measure_memory)

        metrics = compute_metrics(
            task=task,
            y_true=data.y_valid,
            y_pred=y_pred_arr,
            n_classes=config.dataset.n_classes,
            sample_weight=data.sample_weight_valid,
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
