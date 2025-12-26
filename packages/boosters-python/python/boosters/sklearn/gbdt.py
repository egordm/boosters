"""Gradient Boosted Decision Tree sklearn-compatible estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self, TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from boosters import GBDTConfig, GBDTModel, GrowthStrategy, ImportanceType, Metric, Objective
from boosters.data import Dataset, EvalSet

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GBDTClassifier", "GBDTRegressor"]

T = TypeVar("T", bound="_GBDTEstimatorBase")


# =============================================================================
# Base Estimator
# =============================================================================


class _GBDTEstimatorBase(BaseEstimator, ABC):  # type: ignore[misc]
    """Base class for GBDT estimators.

    Handles common initialization, config creation, and fitting logic.
    Subclasses define task-specific behavior (regression vs classification).
    """

    # Instance attributes (declared for type checking)
    model_: GBDTModel
    n_features_in_: int
    _config: GBDTConfig

    @classmethod
    @abstractmethod
    def _get_default_objective(cls) -> Objective:
        """Return the default objective for this estimator type."""
        ...

    @classmethod
    @abstractmethod
    def _get_default_metric(cls) -> Metric | None:
        """Return the default metric for this estimator type."""
        ...

    @classmethod
    @abstractmethod
    def _validate_objective(cls, objective: Objective) -> None:
        """Validate objective is appropriate for this estimator type.

        Raises:
            ValueError: If objective is not valid for this estimator type.
        """
        ...

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        max_leaves: int = 31,
        grow_strategy: GrowthStrategy = GrowthStrategy.Depthwise,
        colsample_bytree: float = 1.0,
        subsample: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        n_threads: int = 0,
        verbose: int = 1,
        objective: Objective | None = None,
        metric: Metric | None = None,
    ) -> None:
        # Store all parameters (sklearn convention)
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
        self.n_threads = n_threads
        self.verbose = verbose
        self.objective = objective
        self.metric = metric

        # Validate and create config immediately
        obj = objective if objective is not None else self._get_default_objective()
        met = metric if metric is not None else self._get_default_metric()
        self._validate_objective(obj)

        self._config = GBDTConfig(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            objective=obj,
            metric=met,
            growth_strategy=grow_strategy,
            max_depth=max_depth,
            n_leaves=max_leaves,
            min_child_weight=min_child_weight,
            min_gain_to_split=gamma,
            l1=reg_alpha,
            l2=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )

    @abstractmethod
    def _prepare_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare targets for training.

        For regressors, this simply casts to float32.
        For classifiers, this performs label encoding.
        """
        ...

    @abstractmethod
    def _prepare_eval_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare evaluation set targets."""
        ...

    def fit(
        self: T,
        X: NDArray[Any],
        y: NDArray[Any],
        eval_set: list[tuple[NDArray[Any], NDArray[Any]]] | list[EvalSet] | None = None,
        sample_weight: NDArray[np.float32] | None = None,
    ) -> T:
        """Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        eval_set : list of tuples or list of EvalSet, optional
            Validation sets. Can be:
            - list of (X, y) tuples (auto-named as "valid_0", "valid_1", ...)
            - list of EvalSet objects (with custom names)
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        # Prepare targets (handles label encoding for classifiers)
        y_prepared = self._prepare_targets(y)

        train_data = Dataset(X, y_prepared, weights=sample_weight)
        valid_list = self._build_eval_sets(eval_set)

        self.model_ = GBDTModel(config=self._config)
        self.model_.fit(train_data, valid=valid_list)

        return self

    def _build_eval_sets(
        self, eval_set: list[tuple[NDArray[Any], NDArray[Any]]] | list[EvalSet] | None
    ) -> list[EvalSet] | None:
        """Build evaluation sets from user input."""
        if eval_set is None:
            return None

        valid_list: list[EvalSet] = []
        for i, item in enumerate(eval_set):
            if isinstance(item, EvalSet):
                valid_list.append(item)
            else:
                X_val, y_val = item
                X_val = check_array(X_val, dtype=np.float32)
                y_val_prepared = self._prepare_eval_targets(y_val)
                val_ds = Dataset(X_val, y_val_prepared)
                valid_list.append(EvalSet(val_ds, f"valid_{i}"))

        return valid_list

    def predict(self, X: NDArray[Any]) -> NDArray[np.float32]:
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, dtype=np.float32)
        preds: NDArray[np.float32] = self.model_.predict(Dataset(X))
        return np.squeeze(preds, axis=-1)

    @property
    def feature_importances_(self) -> NDArray[np.float32]:
        """Return feature importances (gain-based)."""
        check_is_fitted(self, ["model_"])
        return self.model_.feature_importance(ImportanceType.Gain)

    def get_feature_importance(
        self, importance_type: ImportanceType = ImportanceType.Gain
    ) -> NDArray[np.float32]:
        """Get feature importance scores.

        Parameters
        ----------
        importance_type : ImportanceType, default=ImportanceType.Gain
            Type of feature importance to compute.

        Returns
        -------
        importance : ndarray of shape (n_features,)
            Feature importance scores.
        """
        check_is_fitted(self, ["model_"])
        return self.model_.feature_importance(importance_type)


# =============================================================================
# Regressor
# =============================================================================


class GBDTRegressor(_GBDTEstimatorBase, RegressorMixin):  # type: ignore[misc]
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
    grow_strategy : GrowthStrategy, default=GrowthStrategy.Depthwise
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
    n_threads : int, default=0
        Number of threads (0 = auto).
    objective : Objective or None, default=None
        Loss function. Must be a regression objective (e.g., Objective.squared()).
        If None, uses Objective.squared().
    metric : Metric or None, default=None
        Evaluation metric. If None, uses Metric.rmse().

    Attributes
    ----------
    model_ : GBDTModel
        The fitted core model.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importance scores (gain-based).
    """

    _REGRESSION_KEYWORDS = ("squared", "absolute", "huber", "quantile", "tweedie", "poisson", "gamma")
    _CLASSIFICATION_KEYWORDS = ("logistic", "softmax", "cross")

    @classmethod
    def _get_default_objective(cls) -> Objective:
        return Objective.squared()

    @classmethod
    def _get_default_metric(cls) -> Metric | None:
        return Metric.rmse()

    @classmethod
    def _validate_objective(cls, objective: Objective) -> None:
        obj_name = str(objective).lower()
        if any(x in obj_name for x in cls._CLASSIFICATION_KEYWORDS):
            raise ValueError(
                f"GBDTRegressor requires a regression objective, got {objective}. "
                f"Use Objective.squared(), Objective.absolute(), etc. "
                f"For classification, use GBDTClassifier instead."
            )

    def _prepare_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare regression targets."""
        return np.asarray(y, dtype=np.float32)

    def _prepare_eval_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare evaluation set targets for regression."""
        return np.asarray(y, dtype=np.float32)


# =============================================================================
# Classifier
# =============================================================================


class GBDTClassifier(_GBDTEstimatorBase, ClassifierMixin):  # type: ignore[misc]
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
    grow_strategy : GrowthStrategy, default=GrowthStrategy.Depthwise
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
    n_threads : int, default=0
        Number of threads (0 = auto).
    objective : Objective or None, default=None
        Loss function. Must be a classification objective.
        If None, auto-detects: Objective.logistic() for binary,
        Objective.softmax() for multiclass.
    metric : Metric or None, default=None
        Evaluation metric. If None, uses Metric.logloss().

    Attributes
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

    # Additional instance attributes for classifier
    classes_: NDArray[Any]
    n_classes_: int
    _label_to_idx: Mapping[Any, int]

    _REGRESSION_KEYWORDS = ("squared", "absolute", "huber", "quantile", "tweedie", "poisson", "gamma")

    @classmethod
    def _get_default_objective(cls) -> Objective:
        return Objective.logistic()

    @classmethod
    def _get_default_metric(cls) -> Metric | None:
        return Metric.logloss()

    @classmethod
    def _validate_objective(cls, objective: Objective) -> None:
        obj_name = str(objective).lower()
        if any(x in obj_name for x in cls._REGRESSION_KEYWORDS):
            raise ValueError(
                f"GBDTClassifier requires a classification objective, got {objective}. "
                f"Use Objective.logistic() for binary or Objective.softmax() for multiclass. "
                f"For regression, use GBDTRegressor instead."
            )

    def _prepare_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare classification targets with label encoding."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        return np.array([self._label_to_idx[c] for c in y], dtype=np.float32)

    def _prepare_eval_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare evaluation set targets with label encoding."""
        return np.array([self._label_to_idx[c] for c in y], dtype=np.float32)

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
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

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float32]:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probability estimates.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, dtype=np.float32)
        preds: NDArray[np.float32] = self.model_.predict(Dataset(X))

        if self.n_classes_ == 2:
            preds_1d = np.squeeze(preds, axis=-1)
            proba: NDArray[np.float32] = np.column_stack([1 - preds_1d, preds_1d])
        else:
            proba = preds

        return proba
