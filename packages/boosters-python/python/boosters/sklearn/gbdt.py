"""Gradient Boosted Decision Tree sklearn-compatible estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from boosters import (
    CategoricalConfig,
    EFBConfig,
    GBDTConfig,
    GBDTModel,
    LinearLeavesConfig,
    LogisticLoss,
    LogLoss,
    RegularizationConfig,
    Rmse,
    SamplingConfig,
    SoftmaxLoss,
    SquaredLoss,
    TreeConfig,
)
from boosters.data import Dataset, EvalSet
from boosters.types import GrowthStrategy

from .base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    check_array,
    check_is_fitted,
    check_X_y,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Config Builder
# =============================================================================


def _build_config(
    *,
    objective: str,
    metric: str | None = None,
    n_classes: int | None = None,
    n_estimators: int = 100,
    learning_rate: float = 0.3,
    early_stopping_rounds: int | None = None,
    seed: int = 42,
    max_depth: int = -1,
    max_leaves: int = 31,
    min_child_weight: float = 1.0,
    gamma: float = 0.0,
    grow_strategy: GrowthStrategy = "depthwise",
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    goss_top_rate: float = 0.0,
    goss_other_rate: float = 0.0,
    min_samples_cat: int = 10,
    max_cat_threshold: int = 256,
    enable_efb: bool = True,
    max_conflict_rate: float = 0.0,
    enable_linear_leaves: bool = False,
    linear_leaves_l2: float = 0.01,
) -> GBDTConfig:
    """Build a GBDTConfig from flat kwargs."""
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

    return GBDTConfig(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        objective=obj,
        metric=metric_obj,
        tree=TreeConfig(
            max_depth=max_depth,
            n_leaves=max_leaves,
            min_samples_leaf=1,
            min_gain_to_split=gamma,
            growth_strategy=grow_strategy,
        ),
        regularization=RegularizationConfig(
            l1=reg_alpha,
            l2=reg_lambda,
            min_hessian=min_child_weight,
        ),
        sampling=SamplingConfig(
            subsample=subsample,
            colsample=colsample_bytree,
            goss_alpha=goss_top_rate,
            goss_beta=goss_other_rate,
        ),
        categorical=CategoricalConfig(
            max_categories=max_cat_threshold,
            min_category_count=min_samples_cat,
        ),
        efb=EFBConfig(
            enable=enable_efb,
            max_conflict_rate=max_conflict_rate,
        ),
        linear_leaves=LinearLeavesConfig(
            enable=enable_linear_leaves,
            l2=linear_leaves_l2,
        ),
    )


# =============================================================================
# Estimators
# =============================================================================


class GBDTRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Gradient Boosted Decision Tree Regressor.

    A sklearn-compatible wrapper around GBDTModel for regression.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=6
        Maximum depth of each tree.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight (hessian) in a child node.
    max_leaves : int, default=31
        Maximum number of leaves per tree.
    grow_strategy : {"depthwise", "lossguide"}, default="depthwise"
        Tree growing strategy.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns for each tree.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    gamma : float, default=0.0
        Minimum loss reduction required for split.
    reg_alpha : float, default=0.0
        L1 regularization on weights.
    reg_lambda : float, default=1.0
        L2 regularization on weights.
    early_stopping_rounds : int or None, default=None
        Stop if no improvement for this many rounds.
    seed : int, default=42
        Random seed.

    Attributes:
    ----------
    model_ : GBDTModel
        The fitted core model.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importance scores (gain-based).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        max_leaves: int = 31,
        grow_strategy: GrowthStrategy = "depthwise",
        colsample_bytree: float = 1.0,
        subsample: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        n_threads: int = 0,
        verbose: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_leaves = max_leaves
        self.grow_strategy = grow_strategy
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.n_threads = n_threads  # Not yet used, for API compatibility
        self.verbose = verbose  # Not yet used, for API compatibility

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        sample_weight: NDArray[np.float32] | None = None,
    ) -> GBDTRegressor:
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        eval_set : list of tuples, optional
            Validation sets as (X, y) pairs for early stopping.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns:
        -------
        self : GBDTRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float32, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        config = _build_config(
            objective="regression:squarederror",
            metric="rmse",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            max_leaves=self.max_leaves,
            grow_strategy=self.grow_strategy,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.seed,
        )

        train_data = Dataset(X, y, weights=sample_weight)

        valid_list = None
        if eval_set is not None:
            valid_list = []
            for i, (X_val, y_val) in enumerate(eval_set):
                X_val = check_array(X_val, dtype=np.float32)
                y_val = np.asarray(y_val, dtype=np.float32)
                val_ds = Dataset(X_val, y_val)
                valid_list.append(EvalSet(f"valid_{i}", val_ds))

        self.model_ = GBDTModel(config=config)
        self.model_.fit(train_data, valid=valid_list)

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns:
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, dtype=np.float32)
        preds = self.model_.predict(X)
        # Squeeze from (n_samples, 1) to (n_samples,) for sklearn compatibility
        return np.squeeze(preds, axis=-1)

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        """Return feature importances (gain-based)."""
        check_is_fitted(self, ["model_"])
        return self.model_.feature_importance(importance_type="gain")


class GBDTClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Gradient Boosted Decision Tree Classifier.

    A sklearn-compatible wrapper around GBDTModel for classification.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Learning rate.
    max_depth : int, default=6
        Maximum depth of each tree.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight in a child node.
    max_leaves : int, default=31
        Maximum number of leaves per tree.
    grow_strategy : {"depthwise", "lossguide"}, default="depthwise"
        Tree growing strategy.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns for each tree.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    gamma : float, default=0.0
        Minimum loss reduction required for split.
    reg_alpha : float, default=0.0
        L1 regularization.
    reg_lambda : float, default=1.0
        L2 regularization.
    early_stopping_rounds : int or None, default=None
        Stop if no improvement for this many rounds.
    seed : int, default=42
        Random seed.

    Attributes:
    ----------
    model_ : GBDTModel
        The fitted core model.
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importance scores.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        max_leaves: int = 31,
        grow_strategy: GrowthStrategy = "depthwise",
        colsample_bytree: float = 1.0,
        subsample: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        n_threads: int = 0,
        verbose: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_leaves = max_leaves
        self.grow_strategy = grow_strategy
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.n_threads = n_threads  # Not yet used, for API compatibility
        self.verbose = verbose  # Not yet used, for API compatibility

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        sample_weight: NDArray[np.float32] | None = None,
    ) -> GBDTClassifier:
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target class labels.
        eval_set : list of tuples, optional
            Validation sets as (X, y) pairs.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns:
        -------
        self : GBDTClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        # Label encoding
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._label_to_idx[c] for c in y], dtype=np.float32)

        if self.n_classes_ == 2:
            objective = "binary:logistic"
            metric = "logloss"
        else:
            objective = "multi:softmax"
            metric = "mlogloss"

        config = _build_config(
            objective=objective,
            metric=metric,
            n_classes=self.n_classes_,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            max_leaves=self.max_leaves,
            grow_strategy=self.grow_strategy,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.seed,
        )

        train_data = Dataset(X, y_encoded, weights=sample_weight)

        valid_list = None
        if eval_set is not None:
            valid_list = []
            for i, (X_val, y_val) in enumerate(eval_set):
                X_val = check_array(X_val, dtype=np.float32)
                y_val_encoded = np.array([self._label_to_idx[c] for c in y_val], dtype=np.float32)
                val_ds = Dataset(X_val, y_val_encoded)
                valid_list.append(EvalSet(f"valid_{i}", val_ds))

        self.model_ = GBDTModel(config=config)
        self.model_.fit(train_data, valid=valid_list)

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns:
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["model_", "classes_"])
        proba = self.predict_proba(X)

        if self.n_classes_ == 2:
            indices = (proba[:, 1] >= 0.5).astype(int)
        else:
            indices = np.argmax(proba, axis=1)

        return self.classes_[indices]

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns:
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probability estimates.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, dtype=np.float32)
        preds = self.model_.predict(X)

        if self.n_classes_ == 2:
            # Binary: preds is (n_samples, 1), squeeze and make 2-column
            preds_1d = np.squeeze(preds, axis=-1)
            proba = np.column_stack([1 - preds_1d, preds_1d])
        else:
            # Multiclass: preds is already (n_samples, n_classes)
            proba = preds

        return proba

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        """Return feature importances (gain-based)."""
        check_is_fitted(self, ["model_"])
        return self.model_.feature_importance(importance_type="gain")


__all__ = ["GBDTClassifier", "GBDTRegressor"]
