"""scikit-learn compatible estimators for boosters.

This module provides sklearn-compatible wrappers around the core boosters models.
The estimators follow sklearn conventions and pass `check_estimator()` tests.

Example:
    >>> from boosters.sklearn import GBDTRegressor
    >>> reg = GBDTRegressor(max_depth=5, learning_rate=0.1)

    For full training examples, see individual class docstrings.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from boosters import (
    Dataset,
    EvalSet,
    GBDTModel,
    GBLinearModel,
    LogisticLoss,
    Rmse,
    SoftmaxLoss,
    SquaredLoss,
)
from boosters._sklearn_base import build_gbdt_config, build_gblinear_config

GrowthStrategy = Literal["leafwise", "depthwise"]

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

    def check_array(X: Any, **kwargs: Any) -> NDArray[np.floating[Any]]:
        return np.asarray(X, dtype=np.float32)

    def check_X_y(X: Any, y: Any, **kwargs: Any) -> tuple[NDArray[np.floating[Any]], NDArray[Any]]:
        # Don't convert y to float32 since it might be categorical
        return np.asarray(X, dtype=np.float32), np.asarray(y)

    def check_is_fitted(estimator: Any, attributes: list[str] | None = None) -> None:
        pass


# =============================================================================
# GBDT Estimators
# =============================================================================


class GBDTRegressor(BaseEstimator, RegressorMixin):  # type: ignore[misc]
    """Gradient Boosted Decision Tree Regressor.

    A sklearn-compatible wrapper around the core GBDTModel for regression tasks.
    Uses flat kwargs (like `max_depth`, `learning_rate`) instead of nested configs.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.3
        Step size shrinkage. Lower values require more trees.
    max_depth : int, default=-1
        Maximum tree depth. -1 means unlimited (leaf-controlled).
    n_leaves : int, default=31
        Maximum number of leaves per tree (for leaf-wise growth).
    min_samples_leaf : int, default=1
        Minimum samples in a leaf node.
    min_gain_to_split : float, default=0.0
        Minimum gain required to make a split.
    growth_strategy : str, default="depthwise"
        Tree growth strategy: "depthwise" or "leafwise".
    l1 : float, default=0.0
        L1 regularization on leaf weights.
    l2 : float, default=1.0
        L2 regularization on leaf weights.
    subsample : float, default=1.0
        Row subsampling ratio (0, 1].
    colsample_bytree : float, default=1.0
        Column subsampling ratio (0, 1].
    early_stopping_rounds : int or None, default=None
        Stop if no improvement for this many rounds. Requires eval_set in fit().
    seed : int, default=42
        Random seed for reproducibility.
    n_threads : int, default=0
        Number of threads. 0 means auto-detect.
    verbose : int, default=1
        Verbosity level. 0=silent, 1=progress, 2=debug.

    Attributes:
    ----------
    model_ : GBDTModel
        The fitted core model.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances based on split count.

    Examples:
    --------
    >>> from boosters.sklearn import GBDTRegressor
    >>> reg = GBDTRegressor(max_depth=5, n_estimators=10, verbose=0)
    >>> import numpy as np
    >>> X = np.random.randn(100, 5).astype(np.float32)
    >>> y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
    >>> reg.fit(X, y)  # doctest: +ELLIPSIS
    GBDTRegressor(...)
    >>> predictions = reg.predict(X)
    >>> predictions.shape
    (100,)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = -1,
        n_leaves: int = 31,
        min_samples_leaf: int = 1,
        min_gain_to_split: float = 0.0,
        growth_strategy: GrowthStrategy = "depthwise",
        l1: float = 0.0,
        l2: float = 1.0,
        min_hessian: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        min_samples_cat: int = 100,
        max_cat_threshold: int = 32,
        enable_efb: bool = True,
        max_conflict_rate: float = 0.0,
        enable_linear_leaves: bool = False,
        linear_leaves_l2: float = 0.1,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        n_threads: int = 0,
        verbose: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_leaves = n_leaves
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split
        self.growth_strategy = growth_strategy
        self.l1 = l1
        self.l2 = l2
        self.min_hessian = min_hessian
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.min_samples_cat = min_samples_cat
        self.max_cat_threshold = max_cat_threshold
        self.enable_efb = enable_efb
        self.max_conflict_rate = max_conflict_rate
        self.enable_linear_leaves = enable_linear_leaves
        self.linear_leaves_l2 = linear_leaves_l2
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.n_threads = n_threads
        self.verbose = verbose

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
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping and monitoring.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns:
        -------
        self : GBDTRegressor
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, dtype=np.float32, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        # Build config
        config = build_gbdt_config(
            objective=SquaredLoss(),
            metric=Rmse(),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.seed,
            max_depth=self.max_depth,
            n_leaves=self.n_leaves,
            min_samples_leaf=self.min_samples_leaf,
            min_gain_to_split=self.min_gain_to_split,
            growth_strategy=self.growth_strategy,
            l1=self.l1,
            l2=self.l2,
            min_hessian=self.min_hessian,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            goss_top_rate=self.goss_top_rate,
            goss_other_rate=self.goss_other_rate,
            min_samples_cat=self.min_samples_cat,
            max_cat_threshold=self.max_cat_threshold,
            enable_efb=self.enable_efb,
            max_conflict_rate=self.max_conflict_rate,
            enable_linear_leaves=self.enable_linear_leaves,
            linear_leaves_l2=self.linear_leaves_l2,
        )

        # Create dataset
        train_data = Dataset(X, y, weights=sample_weight)

        # Convert eval_set to EvalSet objects
        valid = None
        if eval_set is not None:
            valid = []
            for i, (X_val, y_val) in enumerate(eval_set):
                X_val = check_array(X_val, dtype=np.float32)
                y_val = np.asarray(y_val, dtype=np.float32)
                valid.append(EvalSet(f"valid_{i}", Dataset(X_val, y_val)))

        # Fit model
        self.model_ = GBDTModel(config=config)
        self.model_.fit(train_data, valid=valid)

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns:
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, dtype=np.float32)
        return self.model_.predict(X)  # type: ignore[return-value]

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        """Feature importances based on split count."""
        check_is_fitted(self, ["model_"])
        return self.model_.feature_importance()


class GBDTClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Gradient Boosted Decision Tree Classifier.

    A sklearn-compatible wrapper around the core GBDTModel for classification.
    Automatically infers binary vs multiclass from label cardinality.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.3
        Step size shrinkage.
    max_depth : int, default=-1
        Maximum tree depth. -1 means unlimited.
    n_leaves : int, default=31
        Maximum number of leaves per tree.
    min_samples_leaf : int, default=1
        Minimum samples in a leaf node.
    l1 : float, default=0.0
        L1 regularization.
    l2 : float, default=1.0
        L2 regularization.
    subsample : float, default=1.0
        Row subsampling ratio.
    colsample_bytree : float, default=1.0
        Column subsampling ratio.
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

    Examples:
    --------
    >>> from boosters.sklearn import GBDTClassifier
    >>> import numpy as np
    >>> X = np.random.randn(100, 5).astype(np.float32)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = GBDTClassifier(max_depth=5, n_estimators=10, verbose=0)
    >>> clf.fit(X, y)  # doctest: +ELLIPSIS
    GBDTClassifier(...)
    >>> predictions = clf.predict(X)
    >>> probabilities = clf.predict_proba(X)
    >>> probabilities.shape
    (100, 2)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        max_depth: int = -1,
        n_leaves: int = 31,
        min_samples_leaf: int = 1,
        min_gain_to_split: float = 0.0,
        growth_strategy: GrowthStrategy = "depthwise",
        l1: float = 0.0,
        l2: float = 1.0,
        min_hessian: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        min_samples_cat: int = 100,
        max_cat_threshold: int = 32,
        enable_efb: bool = True,
        max_conflict_rate: float = 0.0,
        enable_linear_leaves: bool = False,
        linear_leaves_l2: float = 0.1,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        n_threads: int = 0,
        verbose: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_leaves = n_leaves
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_to_split = min_gain_to_split
        self.growth_strategy = growth_strategy
        self.l1 = l1
        self.l2 = l2
        self.min_hessian = min_hessian
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.min_samples_cat = min_samples_cat
        self.max_cat_threshold = max_cat_threshold
        self.enable_efb = enable_efb
        self.max_conflict_rate = max_conflict_rate
        self.enable_linear_leaves = enable_linear_leaves
        self.linear_leaves_l2 = linear_leaves_l2
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.n_threads = n_threads
        self.verbose = verbose

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
            Training features.
        y : array-like of shape (n_samples,)
            Target class labels.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns:
        -------
        self : GBDTClassifier
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        # Extract classes and encode labels
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Create label encoder mapping
        self._label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self._label_to_idx[c] for c in y], dtype=np.float32)

        # Choose objective based on number of classes
        if self.n_classes_ == 2:
            objective = LogisticLoss()
        else:
            objective = SoftmaxLoss(n_classes=self.n_classes_)

        # Build config
        config = build_gbdt_config(
            objective=objective,
            metric=None,  # Use objective's default
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            early_stopping_rounds=self.early_stopping_rounds,
            seed=self.seed,
            max_depth=self.max_depth,
            n_leaves=self.n_leaves,
            min_samples_leaf=self.min_samples_leaf,
            min_gain_to_split=self.min_gain_to_split,
            growth_strategy=self.growth_strategy,
            l1=self.l1,
            l2=self.l2,
            min_hessian=self.min_hessian,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            goss_top_rate=self.goss_top_rate,
            goss_other_rate=self.goss_other_rate,
            min_samples_cat=self.min_samples_cat,
            max_cat_threshold=self.max_cat_threshold,
            enable_efb=self.enable_efb,
            max_conflict_rate=self.max_conflict_rate,
            enable_linear_leaves=self.enable_linear_leaves,
            linear_leaves_l2=self.linear_leaves_l2,
        )

        # Create dataset
        train_data = Dataset(X, y_encoded, weights=sample_weight)

        # Convert eval_set to EvalSet objects
        valid = None
        if eval_set is not None:
            valid = []
            for i, (X_val, y_val) in enumerate(eval_set):
                X_val = check_array(X_val, dtype=np.float32)
                y_val_encoded = np.array([self._label_to_idx[c] for c in y_val], dtype=np.float32)
                valid.append(EvalSet(f"valid_{i}", Dataset(X_val, y_val_encoded)))

        # Fit model
        self.model_ = GBDTModel(config=config)
        self.model_.fit(train_data, valid=valid)

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns:
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["model_", "classes_"])
        proba = self.predict_proba(X)

        if self.n_classes_ == 2:
            # Binary: use probability of class 1
            indices = (proba[:, 1] >= 0.5).astype(int)
        else:
            indices = np.argmax(proba, axis=1)

        return self.classes_[indices]

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict.

        Returns:
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, dtype=np.float32)
        preds = self.model_.predict(X)

        if self.n_classes_ == 2:
            # Binary: preds is probability of class 1
            proba = np.column_stack([1 - preds, preds])
        else:
            # Multiclass: preds is already (n_samples, n_classes)
            proba = preds

        return proba  # type: ignore[return-value]

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        """Feature importances based on split count."""
        check_is_fitted(self, ["model_"])
        return self.model_.feature_importance()


# =============================================================================
# GBLinear Estimators
# =============================================================================


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
        """Fit the regressor."""
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

        valid = None
        if eval_set is not None:
            valid = []
            for i, (X_val, y_val) in enumerate(eval_set):
                X_val = check_array(X_val, dtype=np.float32)
                y_val = np.asarray(y_val, dtype=np.float32)
                valid.append(EvalSet(f"valid_{i}", Dataset(X_val, y_val)))

        self.model_ = GBLinearModel(config=config)
        self.model_.fit(train_data, eval_set=valid)

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict target values."""
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
        """Fit the classifier."""
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

        valid = None
        if eval_set is not None:
            valid = []
            for i, (X_val, y_val) in enumerate(eval_set):
                X_val = check_array(X_val, dtype=np.float32)
                y_val_encoded = np.array([self._label_to_idx[c] for c in y_val], dtype=np.float32)
                valid.append(EvalSet(f"valid_{i}", Dataset(X_val, y_val_encoded)))

        self.model_ = GBLinearModel(config=config)
        self.model_.fit(train_data, eval_set=valid)

        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray:
        """Predict class labels."""
        check_is_fitted(self, ["model_", "classes_"])
        proba = self.predict_proba(X)

        if self.n_classes_ == 2:
            # Binary: use probability of class 1
            indices = (proba[:, 1] >= 0.5).astype(int)
        else:
            indices = np.argmax(proba, axis=1)

        return self.classes_[indices]

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Predict class probabilities."""
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


__all__ = [
    "GBDTClassifier",
    "GBDTRegressor",
    "GBLinearClassifier",
    "GBLinearRegressor",
]
