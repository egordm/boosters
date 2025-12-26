"""Gradient Boosted Linear sklearn-compatible estimators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from boosters import GBLinearModel
from boosters._boosters_rs import EvalSet
from boosters.data import Dataset
from boosters.metrics import Rmse
from boosters.objectives import LogisticLoss, SoftmaxLoss, SquaredLoss

from .base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    build_gblinear_config,
    check_array,
    check_is_fitted,
    check_X_y,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GBLinearRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
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

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        l1: float = 0.0,
        l2: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.float32],
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        sample_weight: NDArray[np.float32] | None = None,
    ) -> GBLinearRegressor:
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
        self : GBLinearRegressor
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float32, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        config = build_gblinear_config(
            objective=SquaredLoss(),
            metric=Rmse(),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            l1=self.l1,
            l2=self.l2,
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
                valid_list.append(EvalSet(f"valid_{i}", val_ds._inner))

        self.model_ = GBLinearModel(config=config)
        self.model_.fit(train_data._inner, eval_set=valid_list)

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
        return self.model_.predict(X)

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


class GBLinearClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
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

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        l1: float = 0.0,
        l2: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray,
        eval_set: list[tuple[NDArray, NDArray]] | None = None,
        sample_weight: NDArray[np.float32] | None = None,
    ) -> GBLinearClassifier:
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
        self : GBLinearClassifier
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._label_to_idx[c] for c in y], dtype=np.float32)

        if self.n_classes_ == 2:
            objective = LogisticLoss()
        else:
            objective = SoftmaxLoss(n_classes=self.n_classes_)

        config = build_gblinear_config(
            objective=objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            l1=self.l1,
            l2=self.l2,
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
                valid_list.append(EvalSet(f"valid_{i}", val_ds._inner))

        self.model_ = GBLinearModel(config=config)
        self.model_.fit(train_data._inner, eval_set=valid_list)

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
            proba = np.column_stack([1 - preds, preds])
        else:
            proba = preds

        return proba

    @property
    def coef_(self) -> NDArray[np.float32]:
        """Coefficient weights."""
        check_is_fitted(self, ["model_"])
        return self.model_.coef_

    @property
    def intercept_(self) -> NDArray[np.float32]:
        """Intercept terms."""
        check_is_fitted(self, ["model_"])
        return self.model_.intercept_


__all__ = ["GBLinearClassifier", "GBLinearRegressor"]
