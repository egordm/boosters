"""Benchmark runners for different libraries."""

from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

from boosters_eval.config import BenchmarkConfig, BoosterType, Task
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
        *,
        timing_mode: bool = False,
        measure_memory: bool = False,
    ) -> BenchmarkResult:
        """Run training and return results."""
        ...


def _measure_memory_context() -> tuple[None, None] | tuple[float, float]:
    """Start memory measurement."""
    tracemalloc.start()
    return None, None


def _get_peak_memory() -> float:
    """Get peak memory in MB and stop tracing."""
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024 / 1024


class BoostersRunner(Runner):
    """Runner for the boosters library."""

    name = "boosters"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if boosters supports the given configuration."""
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
        *,
        timing_mode: bool = False,
        measure_memory: bool = False,
    ) -> BenchmarkResult:
        """Run boosters training and return results."""
        import boosters  # noqa: PLC0415

        task = config.dataset.task
        tc = config.training

        # Map task to objective
        if task == Task.REGRESSION:
            objective = boosters.Objective.squared()
        elif task == Task.BINARY:
            objective = boosters.Objective.logistic()
        else:
            objective = boosters.Objective.softmax(config.dataset.n_classes or 3)

        # Build config based on booster type
        if config.booster_type == BoosterType.GBDT:
            model_config = boosters.GBDTConfig(
                n_estimators=tc.n_estimators,
                max_depth=tc.max_depth,
                learning_rate=tc.learning_rate,
                reg_lambda=tc.reg_lambda,
                reg_alpha=tc.reg_alpha,
                min_child_weight=tc.min_child_weight,
                subsample=tc.subsample,
                colsample_bytree=tc.colsample_bytree,
                objective=objective,
                seed=seed,
                n_threads=tc.n_threads,
            )
            model = boosters.GBDT(model_config)
        else:  # GBLINEAR
            model_config = boosters.GBLinearConfig(
                n_rounds=tc.n_estimators,
                learning_rate=tc.learning_rate,
                reg_lambda=tc.reg_lambda,
                reg_alpha=tc.reg_alpha,
                objective=objective,
                seed=seed,
                n_threads=tc.n_threads,
            )
            model = boosters.GBLinear(model_config)

        # Warmup for timing mode
        if timing_mode:
            model.fit(x_train[:100], y_train[:100])

        # Memory tracking
        if measure_memory:
            _measure_memory_context()

        # Train
        start_train = time.perf_counter()
        model.fit(x_train, y_train)
        train_time = time.perf_counter() - start_train

        # Predict
        start_predict = time.perf_counter()
        y_pred = model.predict(x_valid)
        predict_time = time.perf_counter() - start_predict

        peak_memory = _get_peak_memory() if measure_memory else None

        # Compute metrics
        metrics = compute_metrics(task, y_valid, y_pred, config.dataset.n_classes)

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
        )


class XGBoostRunner(Runner):
    """Runner for XGBoost library."""

    name = "xgboost"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if XGBoost supports the given configuration."""
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
        *,
        timing_mode: bool = False,
        measure_memory: bool = False,
    ) -> BenchmarkResult:
        """Run XGBoost training and return results."""
        import xgboost as xgb  # noqa: PLC0415

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

        # Warmup for timing mode
        if timing_mode:
            dtrain_small = xgb.DMatrix(x_train[:100], label=y_train[:100])
            xgb.train(params, dtrain_small, num_boost_round=5)

        if measure_memory:
            _measure_memory_context()

        start_train = time.perf_counter()
        model = xgb.train(params, dtrain, num_boost_round=tc.n_estimators)
        train_time = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        y_pred = model.predict(dvalid)
        predict_time = time.perf_counter() - start_predict

        peak_memory = _get_peak_memory() if measure_memory else None

        metrics = compute_metrics(task, y_valid, y_pred, config.dataset.n_classes)

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
        )


class LightGBMRunner(Runner):
    """Runner for LightGBM library."""

    name = "lightgbm"

    @classmethod
    def supports(cls, config: BenchmarkConfig) -> bool:
        """Check if LightGBM supports the given configuration."""
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
        *,
        timing_mode: bool = False,
        measure_memory: bool = False,
    ) -> BenchmarkResult:
        """Run LightGBM training and return results."""
        import lightgbm as lgb  # noqa: PLC0415

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

        # Warmup for timing mode
        if timing_mode:
            dtrain_small = lgb.Dataset(x_train[:100], label=y_train[:100], params={"verbose": -1})
            lgb.train(
                params,
                dtrain_small,
                num_boost_round=5,
                callbacks=[lgb.log_evaluation(period=0)],
            )

        if measure_memory:
            _measure_memory_context()

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
        y_pred = model.predict(x_valid)
        predict_time = time.perf_counter() - start_predict

        peak_memory = _get_peak_memory() if measure_memory else None

        metrics = compute_metrics(task, y_valid, y_pred, config.dataset.n_classes)  # pyright: ignore[reportArgumentType]

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
        )


# Registry of runners
_RUNNERS: dict[str, type[Runner]] = {
    "boosters": BoostersRunner,
    "xgboost": XGBoostRunner,
    "lightgbm": LightGBMRunner,
}


def get_runner(name: str) -> type[Runner]:
    """Get a runner by name.

    Args:
        name: Runner name (boosters, xgboost, lightgbm).

    Returns:
        Runner class.

    Raises:
        KeyError: If runner not found.
        ImportError: If runner's library not installed.
    """
    if name not in _RUNNERS:
        raise KeyError(f"Unknown runner: {name}")

    runner_cls = _RUNNERS[name]

    # Verify library is available
    if name == "xgboost":
        import xgboost  # noqa: F401, PLC0415
    elif name == "lightgbm":
        import lightgbm  # noqa: F401, PLC0415
    elif name == "boosters":
        import boosters  # noqa: F401, PLC0415

    return runner_cls


def get_available_runners() -> list[str]:
    """Get list of runners with available dependencies.

    Returns:
        List of available runner names.
    """
    available = []
    for name in _RUNNERS:
        try:
            get_runner(name)
            available.append(name)
        except ImportError:
            pass
    return available
