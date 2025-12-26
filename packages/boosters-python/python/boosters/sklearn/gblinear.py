"""Gradient Boosted Linear sklearn-compatible estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from boosters import GBLinearConfig, GBLinearModel, Metric, Objective
from boosters.data import Dataset, EvalSet

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GBLinearClassifier", "GBLinearRegressor"]

T = TypeVar("T", bound="_GBLinearEstimatorBase")


# =============================================================================
# Base Estimator
# =============================================================================


class _GBLinearEstimatorBase(BaseEstimator, ABC):  # type: ignore[misc]
    """Base class for GBLinear estimators.

    Handles common initialization, config creation, and fitting logic.
    Subclasses define task-specific behavior (regression vs classification).
    """

    # Instance attributes (declared for type checking)
    model_: GBLinearModel
    n_features_in_: int
    _config: GBLinearConfig

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
    def _is_valid_objective(cls, objective: Objective) -> bool:
        """Check if the objective is valid for this estimator type."""
        ...

    @classmethod
    def _get_invalid_objective_message(cls, objective: Objective) -> str:
        """Return error message for invalid objective."""
        return f"Invalid objective {objective} for {cls.__name__}"

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        l1: float = 0.0,
        l2: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        verbose: int = 1,
        objective: Objective | None = None,
        metric: Metric | None = None,
    ) -> None:
        # Store all parameters (sklearn convention)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.verbose = verbose
        self.objective = objective
        self.metric = metric

        # Validate and create config immediately
        obj = objective if objective is not None else self._get_default_objective()
        met = metric if metric is not None else self._get_default_metric()

        # Validate objective type
        if not self._is_valid_objective(obj):
            raise ValueError(self._get_invalid_objective_message(obj))

        self._config = self._create_config(obj, met)

    def _create_config(self, objective: Objective, metric: Metric | None) -> GBLinearConfig:
        """Create config with given objective and metric."""
        return GBLinearConfig(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.seed,
            objective=objective,
            metric=metric,
            l1=self.l1,
            l2=self.l2,
        )

    @abstractmethod
    def _prepare_targets(
        self, y: NDArray[Any]
    ) -> tuple[NDArray[np.float32], Objective, Metric | None]:
        """Prepare targets for training.

        Returns:
            Tuple of (y_prepared, objective, metric).
            For classifiers, this may change objective based on n_classes.
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
        y_prepared, objective, metric = self._prepare_targets(y)

        # Recreate config only if objective changed (e.g., multiclass detection)
        if objective != self._config.objective or metric != self._config.metric:
            self._config = self._create_config(objective, metric)

        train_data = Dataset(X, y_prepared, weights=sample_weight)
        valid_list = self._build_eval_sets(eval_set)

        self.model_ = GBLinearModel(config=self._config)
        self.model_.fit(train_data, eval_set=valid_list)

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
    def coef_(self) -> NDArray[np.float32]:
        """Coefficient weights."""
        check_is_fitted(self, ["model_"])
        return self.model_.coef_

    @property
    def intercept_(self) -> NDArray[np.float32]:
        """Intercept (bias) term."""
        check_is_fitted(self, ["model_"])
        return self.model_.intercept_


# =============================================================================
# Regressor
# =============================================================================


class GBLinearRegressor(_GBLinearEstimatorBase, RegressorMixin):  # type: ignore[misc]
    """Gradient Boosted Linear Regressor.

    A sklearn-compatible wrapper around GBLinearModel for linear regression.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.5
        Step size for weight updates.
    l1 : float, default=0.0
        L1 regularization (alpha).
    l2 : float, default=1.0
        L2 regularization (lambda).
    early_stopping_rounds : int or None, default=None
        Stop if no improvement for this many rounds.
    seed : int, default=42
        Random seed.
    objective : Objective or None, default=None
        Loss function. Must be a regression objective.
        If None, uses Objective.squared().
    metric : Metric or None, default=None
        Evaluation metric. If None, uses Metric.rmse().

    Attributes
    ----------
    model_ : GBLinearModel
        The fitted core model.
    coef_ : ndarray of shape (n_features,)
        Coefficient weights.
    intercept_ : ndarray of shape (1,)
        Intercept (bias) term.
    n_features_in_ : int
        Number of features seen during fit.
    """

    _CLASSIFICATION_KEYWORDS = ("logistic", "softmax", "cross")

    @classmethod
    def _get_default_objective(cls) -> Objective:
        return Objective.squared()

    @classmethod
    def _get_default_metric(cls) -> Metric | None:
        return Metric.rmse()

    @classmethod
    def _is_valid_objective(cls, objective: Objective) -> bool:
        obj_name = str(objective).lower()
        return not any(x in obj_name for x in cls._CLASSIFICATION_KEYWORDS)

    @classmethod
    def _get_invalid_objective_message(cls, objective: Objective) -> str:
        return (
            f"GBLinearRegressor requires a regression objective, got {objective}. "
            f"Use Objective.squared(), etc. "
            f"For classification, use GBLinearClassifier instead."
        )

    def _prepare_targets(
        self, y: NDArray[Any]
    ) -> tuple[NDArray[np.float32], Objective, Metric | None]:
        """Prepare regression targets."""
        y_arr = np.asarray(y, dtype=np.float32)
        obj = self.objective if self.objective is not None else self._get_default_objective()
        met = self.metric if self.metric is not None else self._get_default_metric()
        return y_arr, obj, met

    def _prepare_eval_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare evaluation set targets for regression."""
        return np.asarray(y, dtype=np.float32)


# =============================================================================
# Classifier
# =============================================================================


class GBLinearClassifier(_GBLinearEstimatorBase, ClassifierMixin):  # type: ignore[misc]
    """Gradient Boosted Linear Classifier.

    A sklearn-compatible wrapper around GBLinearModel for classification.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.5
        Step size for weight updates.
    l1 : float, default=0.0
        L1 regularization.
    l2 : float, default=1.0
        L2 regularization.
    early_stopping_rounds : int or None, default=None
        Stop if no improvement for this many rounds.
    seed : int, default=42
        Random seed.
    objective : Objective or None, default=None
        Loss function. Must be a classification objective.
        If None, auto-detects: Objective.logistic() for binary,
        Objective.softmax() for multiclass.
    metric : Metric or None, default=None
        Evaluation metric. If None, uses Metric.logloss().

    Attributes
    ----------
    model_ : GBLinearModel
        The fitted core model.
    classes_ : ndarray
        Unique class labels.
    coef_ : ndarray
        Coefficient weights.
    intercept_ : ndarray
        Intercept terms.
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
    def _is_valid_objective(cls, objective: Objective) -> bool:
        obj_name = str(objective).lower()
        return not any(x in obj_name for x in cls._REGRESSION_KEYWORDS)

    @classmethod
    def _get_invalid_objective_message(cls, objective: Objective) -> str:
        return (
            f"GBLinearClassifier requires a classification objective, got {objective}. "
            f"Use Objective.logistic() for binary or Objective.softmax() for multiclass. "
            f"For regression, use GBLinearRegressor instead."
        )

    def _prepare_targets(
        self, y: NDArray[Any]
    ) -> tuple[NDArray[np.float32], Objective, Metric | None]:
        """Prepare classification targets with label encoding."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._label_to_idx[c] for c in y], dtype=np.float32)

        # Determine objective based on number of classes (if not specified)
        if self.objective is not None:
            obj = self.objective
        elif self.n_classes_ == 2:
            obj = Objective.logistic()
        else:
            obj = Objective.softmax(n_classes=self.n_classes_)

        met = self.metric if self.metric is not None else Metric.logloss()
        return y_encoded, obj, met

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
