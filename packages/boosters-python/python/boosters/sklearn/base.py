"""Base classes and utilities for sklearn integration.

This module provides shared logic for sklearn-compatible estimators,
including kwargs→config conversion, base classes, and validation utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from boosters import (
    CategoricalConfig,
    EFBConfig,
    GBDTConfig,
    GBLinearConfig,
    LinearLeavesConfig,
    RegularizationConfig,
    SamplingConfig,
    TreeConfig,
)
from boosters.types import GrowthStrategy

# =============================================================================
# sklearn Compatibility Layer
# =============================================================================

# Check if sklearn is available
try:
    from sklearn.base import BaseEstimator as _SklearnBaseEstimator
    from sklearn.base import ClassifierMixin as _SklearnClassifierMixin
    from sklearn.base import RegressorMixin as _SklearnRegressorMixin
    from sklearn.utils.validation import (
        check_array as _sklearn_check_array,
    )
    from sklearn.utils.validation import (
        check_is_fitted as _sklearn_check_is_fitted,
    )
    from sklearn.utils.validation import (
        check_X_y as _sklearn_check_X_y,
    )

    SKLEARN_AVAILABLE = True

    # Use sklearn implementations
    BaseEstimator: type = _SklearnBaseEstimator
    ClassifierMixin: type = _SklearnClassifierMixin
    RegressorMixin: type = _SklearnRegressorMixin

    def check_array(X: Any, **kwargs: Any) -> NDArray[np.floating[Any]]:
        """Validate array input."""
        return _sklearn_check_array(X, **kwargs)  # type: ignore[return-value]

    def check_X_y(X: Any, y: Any, **kwargs: Any) -> tuple[NDArray[np.floating[Any]], NDArray[Any]]:
        """Validate X and y inputs."""
        return _sklearn_check_X_y(X, y, **kwargs)  # type: ignore[return-value]

    def check_is_fitted(estimator: Any, attributes: list[str] | None = None) -> None:
        """Check if estimator is fitted."""
        _sklearn_check_is_fitted(estimator, attributes)

except ImportError:
    # Create dummy classes if sklearn is not available
    SKLEARN_AVAILABLE = False

    class _DummyBaseEstimator:
        """Dummy base class when sklearn is not installed.

        Provides minimal get_params/set_params for sklearn compatibility.
        """

        def get_params(self, deep: bool = True) -> dict[str, Any]:
            """Get parameters for this estimator."""
            params = {}
            for key in self.__init__.__code__.co_varnames[1:]:  # Skip 'self'
                if hasattr(self, key):
                    params[key] = getattr(self, key)
            return params

        def set_params(self, **params: Any) -> _DummyBaseEstimator:
            """Set parameters for this estimator."""
            for key, value in params.items():
                setattr(self, key, value)
            return self

    class _DummyRegressorMixin:
        """Dummy mixin when sklearn is not installed."""

        pass

    class _DummyClassifierMixin:
        """Dummy mixin when sklearn is not installed."""

        pass

    BaseEstimator: type = _DummyBaseEstimator  # type: ignore[no-redef]
    ClassifierMixin: type = _DummyClassifierMixin  # type: ignore[no-redef]
    RegressorMixin: type = _DummyRegressorMixin  # type: ignore[no-redef]

    def check_array(X: Any, **kwargs: Any) -> NDArray[np.floating[Any]]:  # noqa: D103
        return np.asarray(X, dtype=np.float32)

    def check_X_y(  # noqa: D103
        X: Any, y: Any, **kwargs: Any
    ) -> tuple[NDArray[np.floating[Any]], NDArray[Any]]:
        # Don't convert y to float32 since it might be categorical
        return np.asarray(X, dtype=np.float32), np.asarray(y)

    def check_is_fitted(  # noqa: D103
        estimator: Any, attributes: list[str] | None = None
    ) -> None:
        pass


# =============================================================================
# Config Builders
# =============================================================================


def build_gbdt_config(
    *,
    objective: str,
    metric: str | None = None,
    n_classes: int | None = None,
    # Top-level
    n_estimators: int = 100,
    learning_rate: float = 0.3,
    early_stopping_rounds: int | None = None,
    seed: int = 42,
    # Tree (XGBoost-style names → core names)
    max_depth: int = -1,
    max_leaves: int = 31,  # → n_leaves
    min_child_weight: float = 1.0,  # → min_hessian
    gamma: float = 0.0,  # → min_gain_to_split
    grow_strategy: GrowthStrategy = "depthwise",  # → growth_strategy
    # Regularization (XGBoost-style names)
    reg_alpha: float = 0.0,  # → l1
    reg_lambda: float = 1.0,  # → l2
    # Sampling (sklearn-friendly names)
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,  # → colsample
    goss_top_rate: float = 0.0,  # → goss_alpha
    goss_other_rate: float = 0.0,  # → goss_beta
    # Categorical (sklearn-friendly names)
    min_samples_cat: int = 10,
    max_cat_threshold: int = 256,
    # EFB (sklearn-friendly names)
    enable_efb: bool = True,
    max_conflict_rate: float = 0.0,
    # Linear leaves (sklearn-friendly names)
    enable_linear_leaves: bool = False,
    linear_leaves_l2: float = 0.01,
) -> GBDTConfig:
    """Build a GBDTConfig from flat kwargs.

    This function takes sklearn/XGBoost-style flat keyword arguments and constructs
    the nested config structure expected by the core API.

    Returns:
        GBDTConfig with all nested configs populated.
    """
    from boosters import LogisticLoss, LogLoss, Rmse, SoftmaxLoss, SquaredLoss

    # Map objective string to objective object
    objective_map: dict[str, Any] = {
        "regression:squarederror": SquaredLoss(),
        "binary:logistic": LogisticLoss(),
        "multi:softmax": SoftmaxLoss(n_classes=n_classes or 2),
    }
    obj = objective_map.get(objective)
    if obj is None:
        raise ValueError(f"Unknown objective: {objective}")

    # Map metric string to metric object
    metric_obj = None
    if metric:
        metric_map: dict[str, Any] = {
            "rmse": Rmse(),
            "logloss": LogLoss(),
            "mlogloss": LogLoss(),
        }
        metric_obj = metric_map.get(metric)
        if metric_obj is None:
            raise ValueError(f"Unknown metric: {metric}")

    # Map XGBoost-style names to core names
    tree = TreeConfig(
        max_depth=max_depth,
        n_leaves=max_leaves,  # XGBoost: max_leaves → core: n_leaves
        min_samples_leaf=1,  # Not exposed in sklearn API, use default
        min_gain_to_split=gamma,  # XGBoost: gamma → core: min_gain_to_split
        growth_strategy=grow_strategy,  # XGBoost: grow_strategy → core: growth_strategy
    )

    regularization = RegularizationConfig(
        l1=reg_alpha,  # XGBoost: reg_alpha → core: l1
        l2=reg_lambda,  # XGBoost: reg_lambda → core: l2
        min_hessian=min_child_weight,  # XGBoost: min_child_weight → core: min_hessian
    )

    # Map sklearn-style params to core API params
    sampling = SamplingConfig(
        subsample=subsample,
        colsample=colsample_bytree,  # sklearn uses colsample_bytree
        goss_alpha=goss_top_rate,  # sklearn uses goss_top_rate
        goss_beta=goss_other_rate,  # sklearn uses goss_other_rate
    )

    categorical = CategoricalConfig(
        max_categories=max_cat_threshold,  # sklearn uses max_cat_threshold
        min_category_count=min_samples_cat,  # sklearn uses min_samples_cat
    )

    efb = EFBConfig(
        enable=enable_efb,  # core uses enable, not enable_efb
        max_conflict_rate=max_conflict_rate,
    )

    linear_leaves = LinearLeavesConfig(
        enable=enable_linear_leaves,  # core uses enable
        l2=linear_leaves_l2,
    )

    return GBDTConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        objective=obj,
        metric=metric_obj,
        tree=tree,
        regularization=regularization,
        sampling=sampling,
        categorical=categorical,
        efb=efb,
        linear_leaves=linear_leaves,
    )


def build_gblinear_config(
    *,
    objective: Any,
    metric: Any = None,
    n_estimators: int = 100,
    learning_rate: float = 0.5,
    l1: float = 0.0,
    l2: float = 1.0,
    early_stopping_rounds: int | None = None,
    seed: int = 42,
) -> GBLinearConfig:
    """Build a GBLinearConfig from flat kwargs.

    Returns:
        GBLinearConfig with all parameters populated.
    """
    return GBLinearConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        l1=l1,
        l2=l2,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        objective=objective,
        metric=metric,
    )


__all__ = [
    "SKLEARN_AVAILABLE",
    "BaseEstimator",
    "ClassifierMixin",
    "GrowthStrategy",
    "RegressorMixin",
    "build_gbdt_config",
    "build_gblinear_config",
    "check_X_y",
    "check_array",
    "check_is_fitted",
]
