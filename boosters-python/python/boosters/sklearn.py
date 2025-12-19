"""Sklearn-compatible wrappers for boosters models.

These wrappers provide full sklearn compatibility including:
- get_params() / set_params() for GridSearchCV
- clone() support
- Pipeline integration
- Cross-validation support

Example:
    >>> from boosters.sklearn import GBDTRegressor, GBDTClassifier
    >>> from sklearn.model_selection import cross_val_score

    >>> model = GBDTRegressor(n_estimators=100, learning_rate=0.1)
    >>> scores = cross_val_score(model, X, y, cv=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    pass

_HAS_SKLEARN = False
_sklearn_check_X_y = None
_sklearn_check_array = None
_sklearn_check_is_fitted = None
_SklearnBaseEstimator: type = object
_SklearnRegressorMixin: type = object
_SklearnClassifierMixin: type = object

try:
    from sklearn.base import (
        BaseEstimator as _SklearnBaseEstimator,  # type: ignore[misc]
    )
    from sklearn.base import (
        ClassifierMixin as _SklearnClassifierMixin,  # type: ignore[misc]
    )
    from sklearn.base import (
        RegressorMixin as _SklearnRegressorMixin,  # type: ignore[misc]
    )
    from sklearn.utils.validation import (
        check_array as _sklearn_check_array,  # type: ignore[misc]
    )
    from sklearn.utils.validation import (
        check_is_fitted as _sklearn_check_is_fitted,  # type: ignore[misc]
    )
    from sklearn.utils.validation import (
        check_X_y as _sklearn_check_X_y,  # type: ignore[misc]
    )
    _HAS_SKLEARN = True
except ImportError:
    pass

from boosters import GBDTBooster, GBLinearBooster


def _check_sklearn() -> None:
    """Raise ImportError if sklearn is not available."""
    if not _HAS_SKLEARN:
        raise ImportError(
            "sklearn is required for sklearn wrappers. "
            "Install with: pip install boosters[sklearn]"
        )


def _validate_X_y(
    x: ArrayLike, y: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert X and y using sklearn utilities."""
    assert _sklearn_check_X_y is not None
    return _sklearn_check_X_y(x, y, dtype=np.float32, accept_sparse=False)  # type: ignore[return-value]


def _validate_X(x: ArrayLike) -> np.ndarray:
    """Validate and convert X using sklearn utilities."""
    assert _sklearn_check_array is not None
    return _sklearn_check_array(x, dtype=np.float32)  # type: ignore[return-value]


def _validate_is_fitted(estimator: object, attr: str) -> None:
    """Check if estimator is fitted."""
    assert _sklearn_check_is_fitted is not None
    _sklearn_check_is_fitted(estimator, attr)


class GBDTRegressor(_SklearnRegressorMixin, _SklearnBaseEstimator):  # type: ignore[misc]
    """Sklearn-compatible GBDT regressor.

    This wrapper provides full sklearn compatibility including get_params(),
    set_params(), clone(), and integration with sklearn pipelines and
    cross-validation utilities.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.3
        Learning rate (shrinkage).
    max_depth : int, default=6
        Maximum tree depth.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight in a child.
    reg_lambda : float, default=1.0
        L2 regularization on leaf weights.
    reg_alpha : float, default=0.0
        L1 regularization on leaf weights.
    gamma : float, default=0.0
        Minimum loss reduction for split.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns for each tree.
    max_bin : int, default=256
        Maximum histogram bins.
    random_state : int, default=0
        Random seed.

    Attributes:
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during fit.
    booster_ : GBDTBooster
        The underlying booster object after fitting.

    Example:
    -------
    >>> from boosters.sklearn import GBDTRegressor
    >>> from sklearn.model_selection import cross_val_score
    >>>
    >>> model = GBDTRegressor(n_estimators=100, learning_rate=0.1)
    >>> scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        max_bin: int = 256,
        random_state: int = 0,
    ):
        _check_sklearn()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.random_state = random_state

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        categorical_features: list[int] | None = None,
    ) -> GBDTRegressor:
        """Fit the model to training data.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        feature_names : list of str, optional
            Feature names.
        categorical_features : list of int, optional
            Indices of categorical features.

        Returns:
        -------
        self : GBDTRegressor
            Fitted estimator.
        """
        X, y_arr = _validate_X_y(x, y)

        self.n_features_in_ = X.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        elif hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)

        # Create and fit booster
        self.booster_ = GBDTBooster(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            objective="squared_error",
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            max_bin=self.max_bin,
            random_state=self.random_state,
        )

        self.booster_.fit(
            X, y_arr,
            sample_weight=sample_weight,
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

        return self

    def predict(self, X) -> np.ndarray:
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns:
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        _validate_is_fitted(self, "booster_")
        X = _validate_X(X)
        return self.booster_.predict(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances (gain-based).

        Returns:
        -------
        importances : ndarray of shape (n_features_in_,)
            Feature importance scores.
        """
        _validate_is_fitted(self, "booster_")
        importance_dict = self.booster_.feature_importance(importance_type="gain")

        # Convert dict to array
        n_features = self.n_features_in_
        importances = np.zeros(n_features)
        for key, value in importance_dict.items():
            if isinstance(key, int):
                importances[key] = value
            elif hasattr(self, 'feature_names_in_'):
                # Find index by name
                idx = np.where(self.feature_names_in_ == key)[0]
                if len(idx) > 0:
                    importances[idx[0]] = value

        return importances


class GBDTClassifier(_SklearnClassifierMixin, _SklearnBaseEstimator):  # type: ignore[misc]
    """Sklearn-compatible GBDT classifier.

    For binary classification, uses logistic loss.
    For multiclass, uses softmax loss with auto-detected number of classes.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.3
        Learning rate (shrinkage).
    max_depth : int, default=6
        Maximum tree depth.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight in a child.
    reg_lambda : float, default=1.0
        L2 regularization on leaf weights.
    reg_alpha : float, default=0.0
        L1 regularization on leaf weights.
    gamma : float, default=0.0
        Minimum loss reduction for split.
    subsample : float, default=1.0
        Subsample ratio of training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns for each tree.
    max_bin : int, default=256
        Maximum histogram bins.
    random_state : int, default=0
        Random seed.

    Attributes:
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels known to the classifier.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during fit.
    booster_ : GBDTBooster
        The underlying booster object after fitting.

    Example:
    -------
    >>> from boosters.sklearn import GBDTClassifier
    >>> from sklearn.model_selection import cross_val_score
    >>>
    >>> model = GBDTClassifier(n_estimators=100, learning_rate=0.1)
    >>> scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        max_bin: int = 256,
        random_state: int = 0,
    ):
        _check_sklearn()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.max_bin = max_bin
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        sample_weight: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        categorical_features: list[int] | None = None,
    ):
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        feature_names : list of str, optional
            Feature names.
        categorical_features : list of int, optional
            Indices of categorical features.

        Returns:
        -------
        self : GBDTClassifier
            Fitted estimator.
        """
        X, y = _validate_X_y(X, y)

        self.n_features_in_ = X.shape[1]
        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        elif hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns)

        # Determine classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Encode labels to 0, 1, ... n_classes-1
        self._label_encoder = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._label_encoder[c] for c in y], dtype=np.float32)

        # Choose objective
        if self.n_classes_ == 2:
            objective = "binary:logistic"
            num_class = None
        else:
            objective = "multi:softmax"
            num_class = self.n_classes_

        # Create and fit booster
        self.booster_ = GBDTBooster(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            objective=objective,
            num_class=num_class,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            max_bin=self.max_bin,
            random_state=self.random_state,
        )

        self.booster_.fit(
            X, y_encoded,
            sample_weight=sample_weight,
            feature_names=feature_names,
            categorical_features=categorical_features,
        )

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns:
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        _validate_is_fitted(self, "booster_")
        X = _validate_X(X)

        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns:
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        _validate_is_fitted(self, "booster_")
        X = _validate_X(X)

        raw_preds = self.booster_.predict(X)

        if self.n_classes_ == 2:
            # Binary: apply sigmoid and return [1-p, p]
            proba_1 = 1 / (1 + np.exp(-raw_preds))
            return np.column_stack([1 - proba_1, proba_1])
        else:
            # Multiclass: reshape and apply softmax
            n_samples = X.shape[0]
            raw_preds = raw_preds.reshape(n_samples, self.n_classes_)
            # Softmax
            exp_preds = np.exp(raw_preds - np.max(raw_preds, axis=1, keepdims=True))
            return exp_preds / exp_preds.sum(axis=1, keepdims=True)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances (gain-based).

        Returns:
        -------
        importances : ndarray of shape (n_features_in_,)
            Feature importance scores.
        """
        _validate_is_fitted(self, "booster_")
        importance_dict = self.booster_.feature_importance(importance_type="gain")

        n_features = self.n_features_in_
        importances = np.zeros(n_features)
        for key, value in importance_dict.items():
            if isinstance(key, int):
                importances[key] = value
            elif hasattr(self, 'feature_names_in_'):
                idx = np.where(self.feature_names_in_ == key)[0]
                if len(idx) > 0:
                    importances[idx[0]] = value

        return importances


class GBLinearRegressor(_SklearnRegressorMixin, _SklearnBaseEstimator):  # type: ignore[misc]
    """Sklearn-compatible linear booster regressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.5
        Learning rate.
    reg_lambda : float, default=0.0
        L2 regularization.
    reg_alpha : float, default=0.0
        L1 regularization.
    random_state : int, default=0
        Random seed.

    Attributes:
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    booster_ : GBLinearBooster
        The underlying booster object after fitting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        reg_lambda: float = 0.0,
        reg_alpha: float = 0.0,
        random_state: int = 0,
    ):
        _check_sklearn()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state

    def fit(self, X, y, sample_weight: np.ndarray | None = None):
        """Fit the model."""
        X, y = _validate_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self.booster_ = GBLinearBooster(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            objective="squared_error",
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
        )

        self.booster_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict regression target for X."""
        _validate_is_fitted(self, "booster_")
        X = _validate_X(X)
        return self.booster_.predict(X)

    @property
    def coef_(self) -> np.ndarray:
        """Model weights."""
        _validate_is_fitted(self, "booster_")
        return self.booster_.weights

    @property
    def intercept_(self) -> float:
        """Model bias."""
        _validate_is_fitted(self, "booster_")
        return self.booster_.bias[0]


class GBLinearClassifier(_SklearnClassifierMixin, _SklearnBaseEstimator):  # type: ignore[misc]
    """Sklearn-compatible linear booster classifier.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.5
        Learning rate.
    reg_lambda : float, default=0.0
        L2 regularization.
    reg_alpha : float, default=0.0
        L1 regularization.
    random_state : int, default=0
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        reg_lambda: float = 0.0,
        reg_alpha: float = 0.0,
        random_state: int = 0,
    ):
        _check_sklearn()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state

    def fit(self, X, y, sample_weight: np.ndarray | None = None):
        """Fit the classifier."""
        X, y = _validate_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ != 2:
            raise ValueError(
                f"GBLinearClassifier only supports binary classification. "
                f"Got {self.n_classes_} classes."
            )

        self._label_encoder = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._label_encoder[c] for c in y], dtype=np.float32)

        self.booster_ = GBLinearBooster(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            objective="binary:logistic",
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
        )

        self.booster_.fit(X, y_encoded, sample_weight=sample_weight)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        _validate_is_fitted(self, "booster_")
        X = _validate_X(X)

        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        _validate_is_fitted(self, "booster_")
        X = _validate_X(X)

        raw_preds = self.booster_.predict(X)
        proba_1 = 1 / (1 + np.exp(-raw_preds))
        return np.column_stack([1 - proba_1, proba_1])

    @property
    def coef_(self) -> np.ndarray:
        """Model weights."""
        _validate_is_fitted(self, "booster_")
        return self.booster_.weights

    @property
    def intercept_(self) -> float:
        """Model bias."""
        _validate_is_fitted(self, "booster_")
        return self.booster_.bias[0]


__all__ = [
    "GBDTClassifier",
    "GBDTRegressor",
    "GBLinearClassifier",
    "GBLinearRegressor",
]
