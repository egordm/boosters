"""Benchmark runners for different libraries."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from boosters_eval.datasets import BenchmarkConfig, BoosterType, Task
from boosters_eval.metrics import compute_metrics
from boosters_eval.results import BenchmarkResult


class Runner(ABC):
    """Base class for benchmark runners."""

    name: str

    @classmethod
    @abstractmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if this runner supports the given configuration."""
        ...

    @classmethod
    @abstractmethod
    def run(
        cls,
        config: BenchmarkConfig,
        x_train: NDArray[np.floating[Any]],
        y_train: NDArray[np.floating[Any]],
        x_valid: NDArray[np.floating[Any]],
        y_valid: NDArray[np.floating[Any]],
        seed: int,
    ) -> BenchmarkResult:
        """Run training and return results."""
        ...


class XGBoostRunner(Runner):
    """Runner for XGBoost library."""

    name = "xgboost"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        return config.booster_type in (BoosterType.GBDT, BoosterType.GBLINEAR)

    @classmethod
    def run(
        cls,
        config: BenchmarkConfig,
        x_train: NDArray[np.floating[Any]],
        y_train: NDArray[np.floating[Any]],
        x_valid: NDArray[np.floating[Any]],
        y_valid: NDArray[np.floating[Any]],
        seed: int,
    ) -> BenchmarkResult:
        import xgboost as xgb

        task = config.dataset.task
        tc = config.training
        booster = "gbtree" if config.booster_type == BoosterType.GBDT else "gblinear"

        params: dict[str, Any] = {
            "booster": booster,
            "eta": tc.learning_rate,
            "max_depth": tc.max_depth if booster == "gbtree" else 0,
            "lambda": tc.reg_lambda,
            "alpha": tc.reg_alpha,
            "subsample": tc.subsample,
            "colsample_bytree": tc.colsample_bytree,
            "nthread": tc.n_threads,
            "seed": seed,
            "verbosity": 0,
        }

        if booster == "gbtree":
            params["min_child_weight"] = tc.min_child_weight

        if task == Task.REGRESSION:
            params["objective"] = "reg:squarederror"
        elif task == Task.BINARY:
            params["objective"] = "binary:logistic"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = config.dataset.n_classes

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid)

        start = time.perf_counter()
        model = xgb.train(params, dtrain, num_boost_round=tc.n_trees)
        train_time = time.perf_counter() - start

        y_pred = model.predict(dvalid)
        metrics = compute_metrics(task, y_valid, y_pred, config.dataset.n_classes)

        return BenchmarkResult(
            config_name=config.name,
            library=cls.name,
            seed=seed,
            task=task,
            booster_type=config.booster_type.value,
            dataset_name=config.dataset.name,
            metrics=metrics,
            train_time_s=train_time,
        )


class LightGBMRunner(Runner):
    """Runner for LightGBM library."""

    name = "lightgbm"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        return config.booster_type in (BoosterType.GBDT, BoosterType.LINEAR_TREES)

    @classmethod
    def run(
        cls,
        config: BenchmarkConfig,
        x_train: NDArray[np.floating[Any]],
        y_train: NDArray[np.floating[Any]],
        x_valid: NDArray[np.floating[Any]],
        y_valid: NDArray[np.floating[Any]],
        seed: int,
    ) -> BenchmarkResult:
        import lightgbm as lgb

        task = config.dataset.task
        tc = config.training

        params: dict[str, Any] = {
            "boosting_type": "gbdt",
            "learning_rate": tc.learning_rate,
            "max_depth": tc.max_depth,
            "reg_lambda": tc.reg_lambda,
            "reg_alpha": tc.reg_alpha,
            "subsample": tc.subsample,
            "colsample_bytree": tc.colsample_bytree,
            "min_child_weight": tc.min_child_weight,
            "n_jobs": tc.n_threads,
            "seed": seed,
            "verbose": -1,
            "force_col_wise": True,
        }

        if config.booster_type == BoosterType.LINEAR_TREES:
            params["linear_tree"] = True

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

        dtrain = lgb.Dataset(x_train, label=y_train, params={"verbose": -1})
        dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain, params={"verbose": -1})

        start = time.perf_counter()
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=tc.n_trees,
            valid_sets=[dvalid],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        train_time = time.perf_counter() - start

        y_pred = model.predict(x_valid)
        metrics = compute_metrics(task, y_valid, y_pred, config.dataset.n_classes) # pyright: ignore[reportArgumentType]

        return BenchmarkResult(
            config_name=config.name,
            library=cls.name,
            seed=seed,
            task=task,
            booster_type=config.booster_type.value,
            dataset_name=config.dataset.name,
            metrics=metrics,
            train_time_s=train_time,
        )


class CatBoostRunner(Runner):
    """Runner for CatBoost library."""

    name = "catboost"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        return config.booster_type == BoosterType.GBDT

    @classmethod
    def run(
        cls,
        config: BenchmarkConfig,
        x_train: NDArray[np.floating[Any]],
        y_train: NDArray[np.floating[Any]],
        x_valid: NDArray[np.floating[Any]],
        y_valid: NDArray[np.floating[Any]],
        seed: int,
    ) -> BenchmarkResult:
        from catboost import Pool, train

        task = config.dataset.task
        tc = config.training

        params: dict[str, Any] = {
            "iterations": tc.n_trees,
            "learning_rate": tc.learning_rate,
            "depth": tc.max_depth,
            "l2_leaf_reg": tc.reg_lambda,
            "subsample": tc.subsample,
            "rsm": tc.colsample_bytree,
            "min_child_samples": int(tc.min_child_weight),
            "thread_count": tc.n_threads,
            "random_seed": seed,
            "verbose": 0,
            "allow_writing_files": False,
        }

        if task == Task.REGRESSION:
            params["loss_function"] = "RMSE"
        elif task == Task.BINARY:
            params["loss_function"] = "Logloss"
        else:
            params["loss_function"] = "MultiClass"
            params["classes_count"] = config.dataset.n_classes

        pool_train = Pool(x_train, label=y_train)
        pool_valid = Pool(x_valid, label=y_valid)

        start = time.perf_counter()
        model = train(params, pool_train, eval_set=pool_valid, verbose=0)
        train_time = time.perf_counter() - start

        if task == Task.REGRESSION:
            y_pred = model.predict(pool_valid)
        else:
            y_pred = model.predict(pool_valid, prediction_type="Probability")

        metrics = compute_metrics(task, y_valid, y_pred, config.dataset.n_classes)

        return BenchmarkResult(
            config_name=config.name,
            library=cls.name,
            seed=seed,
            task=task,
            booster_type=config.booster_type.value,
            dataset_name=config.dataset.name,
            metrics=metrics,
            train_time_s=train_time,
        )


# Registry
_RUNNERS: dict[str, type[Runner]] = {
    "xgboost": XGBoostRunner,
    "lightgbm": LightGBMRunner,
    "catboost": CatBoostRunner,
}


def get_runner(name: str) -> type[Runner]:
    """Get a runner by name."""
    if name not in _RUNNERS:
        raise KeyError(f"Unknown runner: {name}")

    runner_cls = _RUNNERS[name]

    if name == "xgboost":
        import xgboost  # noqa: F401
    elif name == "lightgbm":
        import lightgbm  # noqa: F401
    elif name == "catboost":
        import catboost  # noqa: F401

    return runner_cls


def get_available_runners() -> list[str]:
    """Get list of runners with available dependencies."""
    available = []
    for name in _RUNNERS:
        try:
            get_runner(name)
            available.append(name)
        except ImportError:
            pass
    return available


def register_runner(name: str, runner_cls: type[Runner]) -> None:
    """Register a custom runner."""
    _RUNNERS[name] = runner_cls
