"""Benchmark runner for the boosters library."""

from __future__ import annotations

import numpy as np

from boosters_eval.config import BenchmarkConfig, BoosterType, Task
from boosters_eval.metrics import compute_metrics
from boosters_eval.results import BenchmarkResult
from boosters_eval.runners.base import RunData, Runner, _maybe_get_peak_memory, _maybe_start_memory


class BoostersRunner(Runner):
    """Runner for the boosters library."""

    name = "boosters"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Return whether this runner can run the given benchmark config."""
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
    ) -> BenchmarkResult:  # type: ignore[override]
        """Train and evaluate a boosters model for this benchmark config."""
        import boosters

        task = config.dataset.task
        tc = config.training

        quantiles_for_task: np.ndarray | None = None

        if task == Task.REGRESSION:
            objective = boosters.Objective.squared()
        elif task == Task.QUANTILE_REGRESSION:
            quantiles = config.dataset.quantiles or tc.quantiles or [0.1, 0.5, 0.9]
            quantiles_for_task = np.asarray(quantiles, dtype=np.float64)
            objective = boosters.Objective.pinball(list(quantiles))
        elif task == Task.BINARY:
            objective = boosters.Objective.logistic()
        else:
            objective = boosters.Objective.softmax(config.dataset.n_classes or 3)

        gbdt_config: boosters.GBDTConfig | None = None
        gblinear_config: boosters.GBLinearConfig | None = None

        if config.booster_type == BoosterType.GBDT:
            gbdt_config = boosters.GBDTConfig(
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
                growth_strategy=(
                    boosters.GrowthStrategy.Leafwise
                    if tc.growth_strategy.value == "leafwise"
                    else boosters.GrowthStrategy.Depthwise
                ),
                objective=objective,
                seed=seed,
            )
            is_gblinear = False
        elif config.booster_type == BoosterType.LINEAR_TREES:
            gbdt_config = boosters.GBDTConfig(
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
                growth_strategy=(
                    boosters.GrowthStrategy.Leafwise
                    if tc.growth_strategy.value == "leafwise"
                    else boosters.GrowthStrategy.Depthwise
                ),
                linear_leaves=True,
                linear_l2=tc.linear_l2,
                objective=objective,
                seed=seed,
            )
            is_gblinear = False
        else:
            gblinear_config = boosters.GBLinearConfig(
                n_estimators=tc.n_estimators,
                learning_rate=tc.learning_rate,
                l2=tc.reg_lambda,
                l1=tc.reg_alpha,
                update_strategy=boosters.GBLinearUpdateStrategy.Sequential,
                objective=objective,
                seed=seed,
            )
            is_gblinear = True

        if is_gblinear and gblinear_config is None:
            raise RuntimeError("Internal error: expected GBLinearConfig")
        if not is_gblinear and gbdt_config is None:
            raise RuntimeError("Internal error: expected GBDTConfig")

        train_ds = boosters.Dataset(
            features=data.x_train.astype("float32"),
            labels=data.y_train.astype("float32"),
            weights=(data.sample_weight_train.astype("float32") if data.sample_weight_train is not None else None),
            categorical_features=data.categorical_features,
            feature_names=data.feature_names,
        )
        valid_ds = boosters.Dataset(
            features=data.x_valid.astype("float32"),
            labels=data.y_valid.astype("float32"),
            weights=(data.sample_weight_valid.astype("float32") if data.sample_weight_valid is not None else None),
            categorical_features=data.categorical_features,
            feature_names=data.feature_names,
        )

        if timing_mode:
            warmup_ds = boosters.Dataset(
                features=data.x_train[:100].astype("float32"),
                labels=data.y_train[:100].astype("float32"),
                weights=(
                    data.sample_weight_train[:100].astype("float32") if data.sample_weight_train is not None else None
                ),
                categorical_features=data.categorical_features,
                feature_names=data.feature_names,
            )
            if is_gblinear:
                _ = boosters.GBLinearModel.train(warmup_ds, config=gblinear_config, n_threads=tc.n_threads)
            else:
                _ = boosters.GBDTModel.train(warmup_ds, config=gbdt_config, n_threads=tc.n_threads)

        _maybe_start_memory(measure_memory=measure_memory)

        import time

        start_train = time.perf_counter()
        if is_gblinear:
            model = boosters.GBLinearModel.train(train_ds, config=gblinear_config, n_threads=tc.n_threads)
        else:
            model = boosters.GBDTModel.train(train_ds, config=gbdt_config, n_threads=tc.n_threads)
        train_time = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        y_pred = model.predict(valid_ds, n_threads=tc.n_threads)
        predict_time = time.perf_counter() - start_predict

        y_pred_arr = np.asarray(y_pred)
        peak_memory = _maybe_get_peak_memory(measure_memory=measure_memory)

        metrics = compute_metrics(
            task=task,
            y_true=data.y_valid,
            y_pred=y_pred_arr,
            n_classes=config.dataset.n_classes,
            sample_weight=data.sample_weight_valid,
            quantiles=quantiles_for_task,
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
