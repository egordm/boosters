"""Gradient Boosted Linear sklearn-compatible estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self, TypeVar

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from boosters import Dataset, GBLinearConfig, GBLinearModel, Metric, Objective

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["GBLinearClassifier", "GBLinearRegressor"]

T = TypeVar("T", bound="_GBLinearEstimatorBase")


# =============================================================================
# Base Estimator
# =============================================================================


class _GBLinearEstimatorBase(BaseEstimator, ABC):
    """Base class for GBLinear estimators.

    Handles common initialization, config creation, and fitting logic.
    Subclasses define task-specific behavior (regression vs classification).
    """

    # Instance attributes (declared for type checking)
    model_: GBLinearModel
    n_features_in_: int

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

    def __init__(  # noqa: PLR0913 (sklearn estimators have many hyperparameters)
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        l1: float = 0.0,
        l2: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        n_threads: int = 0,
        verbose: int = 1,
        objective: Objective | None = None,
        metric: Metric | None = None,
    ) -> None:
        # Store all parameters (sklearn convention)
        # Config is built at fit() time from these attributes to support set_params()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.n_threads = n_threads
        self.verbose = verbose
        self.objective = objective
        self.metric = metric

    def _build_config(self, objective: Objective | None = None) -> GBLinearConfig:
        """Build config from current attributes.

        Called at fit() time to ensure set_params() changes are reflected.

        Parameters
        ----------
        objective : Objective, optional
            Override objective (used by classifier for multiclass).
        """
        obj = objective if objective is not None else self.objective
        if obj is None:
            obj = self._get_default_objective()
        met = self.metric if self.metric is not None else self._get_default_metric()
        self._validate_objective(obj)

        return GBLinearConfig(
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
    def _prepare_targets(self, y: NDArray[Any]) -> tuple[NDArray[np.float32], Objective | None]:
        """Prepare targets for training.

        For regressors, this simply casts to float32.
        For classifiers, this performs label encoding.

        Returns:
        -------
        y_prepared : ndarray of shape (n_samples,)
            Prepared targets.
        objective_override : Objective or None
            Objective to use (e.g., softmax for multiclass), or None to use default.
        """
        ...

    @abstractmethod
    def _prepare_eval_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare evaluation set targets."""
        ...

    def fit(
        self,
        X: NDArray[Any],  # noqa: N803 (sklearn convention for feature matrix)
        y: NDArray[Any],
        eval_set: tuple[NDArray[Any], NDArray[Any]] | None = None,
        sample_weight: NDArray[np.float32] | None = None,
    ) -> Self:
        """Fit the estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        eval_set : tuple of (X, y), optional
            Validation set as (X_val, y_val) tuple.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns:
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float32)  # noqa: N806 (sklearn convention)
        self.n_features_in_ = X.shape[1]

        # Prepare targets (handles label encoding for classifiers)
        # Also returns objective override for multiclass classification
        y_prepared, objective_override = self._prepare_targets(y)

        # Build config at fit time to respect set_params() changes
        config = self._build_config(objective=objective_override)

        train_data = Dataset(X, y_prepared, weights=sample_weight)
        val_data = self._build_val_set(eval_set)

        self.model_ = GBLinearModel.train(
            train_data,
            config=config,
            val_set=val_data,
            n_threads=self.n_threads,
        )

        return self

    def _build_val_set(self, eval_set: tuple[NDArray[Any], NDArray[Any]] | None) -> Dataset | None:
        """Build validation dataset from user input."""
        if eval_set is None:
            return None

        X_val, y_val = eval_set  # noqa: N806 (sklearn convention)
        X_val = check_array(X_val, dtype=np.float32)  # noqa: N806 (sklearn convention)
        y_val_prepared = self._prepare_eval_targets(y_val)
        return Dataset(X_val, y_val_prepared)

    def predict(self, X: NDArray[Any]) -> NDArray[np.float32]:  # noqa: N803 (sklearn convention)
        """Predict using the fitted model.

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
        X = check_array(X, dtype=np.float32)  # noqa: N806 (sklearn convention)
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


class GBLinearRegressor(_GBLinearEstimatorBase, RegressorMixin):
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

    Attributes:
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
    def _validate_objective(cls, objective: Objective) -> None:
        obj_name = str(objective).lower()
        if any(x in obj_name for x in cls._CLASSIFICATION_KEYWORDS):
            raise ValueError(
                f"GBLinearRegressor requires a regression objective, got {objective}. "
                f"Use Objective.squared(), etc. "
                f"For classification, use GBLinearClassifier instead."
            )

    def _prepare_targets(self, y: NDArray[Any]) -> tuple[NDArray[np.float32], Objective | None]:
        """Prepare regression targets."""
        return np.asarray(y, dtype=np.float32), None

    def _prepare_eval_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare evaluation set targets for regression."""
        return np.asarray(y, dtype=np.float32)


# =============================================================================
# Classifier
# =============================================================================


class GBLinearClassifier(_GBLinearEstimatorBase, ClassifierMixin):
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

    Attributes:
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

    _REGRESSION_KEYWORDS = (
        "squared",
        "absolute",
        "huber",
        "quantile",
        "tweedie",
        "poisson",
        "gamma",
    )

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
                f"GBLinearClassifier requires a classification objective, got {objective}. "
                f"Use Objective.logistic() for binary or Objective.softmax() for multiclass. "
                f"For regression, use GBLinearRegressor instead."
            )

    def _prepare_targets(self, y: NDArray[Any]) -> tuple[NDArray[np.float32], Objective | None]:
        """Prepare classification targets with label encoding.

        Returns softmax objective override for multiclass if user didn't specify.
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}

        # Auto-switch to softmax for multiclass (if user didn't specify objective)
        objective_override: Objective | None = None
        if self.n_classes_ > 2 and self.objective is None:
            objective_override = Objective.softmax(self.n_classes_)

        y_encoded = np.array([self._label_to_idx[c] for c in y], dtype=np.float32)
        return y_encoded, objective_override

    def _prepare_eval_targets(self, y: NDArray[Any]) -> NDArray[np.float32]:
        """Prepare evaluation set targets with label encoding."""
        return np.array([self._label_to_idx[c] for c in y], dtype=np.float32)

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:  # noqa: N803 (sklearn convention)
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

        # Binary vs multiclass classification threshold
        if self.n_classes_ == 2:  # noqa: SIM108 (ternary less readable here)
            indices = (proba[:, 1] >= 0.5).astype(int)
        else:
            indices = np.argmax(proba, axis=1)

        return self.classes_[indices]

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.float32]:  # noqa: N803 (sklearn convention)
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
        X = check_array(X, dtype=np.float32)  # noqa: N806 (sklearn convention)
        preds: NDArray[np.float32] = self.model_.predict(Dataset(X))

        if self.n_classes_ == 2:
            preds_1d = np.squeeze(preds, axis=-1)
            proba: NDArray[np.float32] = np.column_stack([1 - preds_1d, preds_1d])
        else:
            proba = preds

        return proba
