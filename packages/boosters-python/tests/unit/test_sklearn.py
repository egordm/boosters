"""Tests for sklearn-compatible estimators.

Focuses on:
- Sklearn API compliance (get_params, set_params, clone)
- Integration with sklearn utilities (cross_val_score, Pipeline, GridSearchCV)
- Core classification/regression behavior
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boosters import Objective
from boosters.sklearn import (
    GBDTClassifier,
    GBDTRegressor,
    GBLinearClassifier,
    GBLinearRegressor,
)


def make_regression_data(n_samples: int = 100, n_features: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple regression data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = X[:, 0] * 2 + X[:, 1] + rng.standard_normal(n_samples).astype(np.float32) * 0.1
    return X, y


def make_binary_data(n_samples: int = 100, n_features: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


class TestGBDTRegressorSklearn:
    """Tests for GBDTRegressor sklearn compliance."""

    def test_get_set_params(self) -> None:
        """get_params/set_params work correctly."""
        reg = GBDTRegressor(n_estimators=50, max_depth=5)
        params = reg.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 5

        reg.set_params(learning_rate=0.2)
        assert reg.learning_rate == 0.2

    def test_fit_predict(self) -> None:
        """Basic fit/predict works."""
        X, y = make_regression_data()
        reg = GBDTRegressor(n_estimators=20, verbose=0)
        reg.fit(X, y)

        assert reg.n_features_in_ == 5
        preds = reg.predict(X)
        assert preds.shape == (100,)

        # Should have reasonable correlation with truth
        corr = np.corrcoef(preds, y)[0, 1]
        assert corr > 0.8

    def test_feature_importances(self) -> None:
        """feature_importances_ property works."""
        X, y = make_regression_data()
        reg = GBDTRegressor(n_estimators=20, verbose=0)
        reg.fit(X, y)

        importance = reg.feature_importances_
        assert importance.shape == (5,)
        assert importance.sum() > 0


class TestGBDTClassifierSklearn:
    """Tests for GBDTClassifier sklearn compliance."""

    def test_binary_classification(self) -> None:
        """Binary classification works."""
        X, y = make_binary_data()
        clf = GBDTClassifier(n_estimators=20, verbose=0)
        clf.fit(X, y)

        assert clf.n_classes_ == 2
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

        # Reasonable accuracy
        accuracy = (preds == y).mean()
        assert accuracy > 0.7

    def test_predict_proba(self) -> None:
        """predict_proba returns valid probabilities."""
        X, y = make_binary_data()
        clf = GBDTClassifier(n_estimators=20, verbose=0)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_multiclass(self) -> None:
        """Multiclass classification requires explicit softmax objective."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((150, 5)).astype(np.float32)
        y = np.zeros(150, dtype=int)
        y[X[:, 0] > 0.5] = 1
        y[X[:, 0] < -0.5] = 2

        # Must specify softmax with correct n_classes
        clf = GBDTClassifier(n_estimators=30, verbose=0, objective=Objective.softmax(n_classes=3))
        clf.fit(X, y)

        assert clf.n_classes_ == 3
        proba = clf.predict_proba(X)
        assert proba.shape == (150, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_string_labels(self) -> None:
        """String labels are handled correctly."""
        X, _ = make_binary_data()
        y = np.where(X[:, 0] > 0, "positive", "negative")

        clf = GBDTClassifier(n_estimators=20, verbose=0)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert all(p in ["positive", "negative"] for p in preds)


class TestGBLinearSklearn:
    """Tests for GBLinear sklearn estimators."""

    def test_regressor_fit_predict(self) -> None:
        """GBLinearRegressor fit/predict works."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3)).astype(np.float32)
        true_weights = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        y = X @ true_weights + 1.0

        reg = GBLinearRegressor(n_estimators=100, learning_rate=0.3)
        reg.fit(X, y.astype(np.float32))

        preds = reg.predict(X)
        corr = np.corrcoef(preds, y)[0, 1]
        assert corr > 0.95

    def test_regressor_coef_intercept(self) -> None:
        """coef_ and intercept_ properties work."""
        X, y = make_regression_data()
        reg = GBLinearRegressor(n_estimators=50)
        reg.fit(X, y)

        assert reg.coef_.shape[0] == 5
        assert reg.intercept_.shape == (1,)

    def test_classifier_binary(self) -> None:
        """GBLinearClassifier binary classification works."""
        X, y = make_binary_data()
        clf = GBLinearClassifier(n_estimators=100, learning_rate=0.3)
        clf.fit(X, y)

        preds = clf.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy > 0.7

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)


class TestSklearnIntegration:
    """Tests for sklearn utility integration."""

    def test_clone(self) -> None:
        """Estimator can be cloned."""
        X, y = make_regression_data(50)
        reg = GBDTRegressor(n_estimators=10, verbose=0)
        reg.fit(X, y)

        reg2 = clone(reg)
        assert not hasattr(reg2, "model_")
        reg2.fit(X, y)
        assert hasattr(reg2, "model_")

    def test_cross_val_score(self) -> None:
        """Works with cross_val_score."""
        X, y = make_regression_data()
        reg = GBDTRegressor(n_estimators=10, verbose=0)
        scores = cross_val_score(reg, X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        assert all(s > 0 for s in scores)

    def test_pipeline(self) -> None:
        """Works in sklearn Pipeline."""
        X, y = make_binary_data()
        X = X * 100  # Large scale

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GBDTClassifier(n_estimators=10, verbose=0)),
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == 100

    def test_grid_search(self) -> None:
        """Works with GridSearchCV."""
        X, y = make_regression_data(60)
        reg = GBDTRegressor(verbose=0)
        param_grid = {"n_estimators": [5, 10], "max_depth": [2, 3]}

        grid = GridSearchCV(reg, param_grid, cv=2, scoring="r2")
        grid.fit(X, y)

        assert grid.best_params_ is not None
        assert hasattr(grid.best_estimator_, "model_")
