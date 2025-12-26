"""Gradient Boosted Linear sklearn-compatible estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from boosters import GBLinearConfig, GBLinearModel, Metric, Objective
from boosters.data import Dataset, EvalSet

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# Base Estimator
# =============================================================================


class _GBLinearEstimatorBase(BaseEstimator, ABC):  # type: ignore[misc]
    """Base class for GBLinear estimators.

    Handles common initialization, config creation, and fitting logic.
    Subclasses define task-specific behavior (regression vs classification).
    """

    # Subclasses must define these
    _default_objective: Objective
    _default_metric: Metric | None

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
        self._validate_and_create_config()

    def _validate_and_create_config(self) -> None:
        """Validate parameters and create the config object."""
        # Use provided objective or default
        obj = self.objective if self.objective is not None else self._default_objective
        met = self.metric if self.metric is not None else self._default_metric

        # Validate objective is appropriate for this estimator type
        self._validate_objective(obj)

        # Create config - this will validate all numeric parameters
        self._config = GBLinearConfig(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.seed,
            objective=obj,
            metric=met,
            l1=self.l1,
            l2=self.l2,
        )

    @abstractmethod
    def _validate_objective(self, objective: Objective) -> None:
        """Validate that the objective is appropriate for this estimator."""
        ...

    @abstractmethod
    def _prepare_targets(
        self, y: NDArray
    ) -> tuple[NDArray[np.float32], Objective, Metric | None]:
        """Prepare targets for training. Returns (y_prepared, objective, metric)."""
        ...

    @abstractmethod
    def _prepare_eval_targets(self, y: NDArray) -> NDArray[np.float32]:
        """Prepare evaluation set targets."""
        ...

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray,
        eval_set: list[tuple[NDArray, NDArray]] | list[EvalSet] | None = None,
        sample_weight: NDArray[np.float32] | None = None,
    ) -> _GBLinearEstimatorBase:
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

        # Update config if objective/metric changed (e.g., multiclass detection)
        if objective != self._config.objective or metric != self._config.metric:
            self._config = GBLinearConfig(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                early_stopping_rounds=self.early_stopping_rounds,
                seed=self.seed,
                objective=objective,
                metric=metric,
                l1=self.l1,
                l2=self.l2,
            )

        train_data = Dataset(X, y_prepared, weights=sample_weight)

        # Build eval sets
        valid_list = self._build_eval_sets(eval_set)

        self.model_ = GBLinearModel(config=self._config)
        self.model_.fit(train_data, eval_set=valid_list)

        return self

    def _build_eval_sets(
        self, eval_set: list[tuple[NDArray, NDArray]] | list[EvalSet] | None
    ) -> list[EvalSet] | None:
        """Build evaluation sets from user input."""
        if eval_set is None:
            return None

        valid_list = []
        for i, item in enumerate(eval_set):
            if isinstance(item, EvalSet):
                # User provided EvalSet with custom name
                valid_list.append(item)
            else:
                # Tuple of (X, y) - auto-name
                X_val, y_val = item
                X_val = check_array(X_val, dtype=np.float32)
                y_val_prepared = self._prepare_eval_targets(y_val)
                val_ds = Dataset(X_val, y_val_prepared)
                valid_list.append(EvalSet(val_ds, f"valid_{i}"))

        return valid_list

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
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
        preds = self.model_.predict(Dataset(X))
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

    _default_objective = Objective.squared()
    _default_metric = Metric.rmse()

    def _validate_objective(self, objective: Objective) -> None:
        """Validate that the objective is a regression objective."""
        obj_name = str(objective).lower()
        if any(x in obj_name for x in ["logistic", "softmax", "cross"]):
            raise ValueError(
                f"GBLinearRegressor requires a regression objective, got {objective}. "
                f"Use Objective.squared(), etc. "
                f"For classification, use GBLinearClassifier instead."
            )

    def _prepare_targets(
        self, y: NDArray
    ) -> tuple[NDArray[np.float32], Objective, Metric | None]:
        """Prepare regression targets."""
        y = np.asarray(y, dtype=np.float32)
        obj = self.objective if self.objective is not None else self._default_objective
        met = self.metric if self.metric is not None else self._default_metric
        return y, obj, met

    def _prepare_eval_targets(self, y: NDArray) -> NDArray[np.float32]:
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

    _default_objective = Objective.logistic()
    _default_metric = Metric.logloss()

    def _validate_objective(self, objective: Objective) -> None:
        """Validate that the objective is a classification objective."""
        obj_name = str(objective).lower()
        if any(x in obj_name for x in ["squared", "absolute", "huber", "quantile", "tweedie"]):
            raise ValueError(
                f"GBLinearClassifier requires a classification objective, got {objective}. "
                f"Use Objective.logistic() for binary or Objective.softmax() for multiclass. "
                f"For regression, use GBLinearRegressor instead."
            )

    def _prepare_targets(
        self, y: NDArray
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

    def _prepare_eval_targets(self, y: NDArray) -> NDArray[np.float32]:
        """Prepare evaluation set targets with label encoding."""
        return np.array([self._label_to_idx[c] for c in y], dtype=np.float32)

    def predict(self, X: NDArray[np.float32]) -> NDArray:
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

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
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
        preds = self.model_.predict(Dataset(X))

        if self.n_classes_ == 2:
            preds_1d = np.squeeze(preds, axis=-1)
            proba = np.column_stack([1 - preds_1d, preds_1d])
        else:
            proba = preds

        return proba


__all__ = ["GBLinearClassifier", "GBLinearRegressor"]
