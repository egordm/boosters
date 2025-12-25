"""Tests for sklearn-compatible estimators."""

import numpy as np
import pytest

from boosters.sklearn import (
    GBDTClassifier,
    GBDTRegressor,
    GBLinearClassifier,
    GBLinearRegressor,
)

# Check if sklearn is available for additional compatibility tests
try:
    from sklearn.utils.estimator_checks import parametrize_with_checks

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# GBDTRegressor Tests
# =============================================================================


class TestGBDTRegressor:
    """Tests for GBDTRegressor."""

    def test_init_defaults(self) -> None:
        """Test default parameter values."""
        reg = GBDTRegressor()
        assert reg.n_estimators == 100
        assert reg.learning_rate == 0.3
        assert reg.max_depth == -1
        assert reg.n_leaves == 31
        assert reg.l2 == 1.0

    def test_get_set_params(self) -> None:
        """Test sklearn get_params/set_params interface."""
        reg = GBDTRegressor(n_estimators=50, max_depth=5)
        params = reg.get_params()

        assert params["n_estimators"] == 50
        assert params["max_depth"] == 5

        reg.set_params(learning_rate=0.1)
        assert reg.learning_rate == 0.1

    def test_fit_predict_basic(self) -> None:
        """Test basic fit and predict workflow."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1] + np.random.randn(100).astype(np.float32) * 0.1

        reg = GBDTRegressor(n_estimators=20, max_depth=3, verbose=0)
        reg.fit(X, y)

        assert hasattr(reg, "model_")
        assert reg.n_features_in_ == 5

        preds = reg.predict(X)
        assert preds.shape == (100,)

        # Should have reasonable correlation with truth
        corr = np.corrcoef(preds, y)[0, 1]
        assert corr > 0.8

    def test_fit_with_eval_set(self) -> None:
        """Test fit with validation set."""
        np.random.seed(42)
        X_train = np.random.randn(80, 5).astype(np.float32)
        y_train = X_train[:, 0] + np.random.randn(80).astype(np.float32) * 0.1
        X_val = np.random.randn(20, 5).astype(np.float32)
        y_val = X_val[:, 0] + np.random.randn(20).astype(np.float32) * 0.1

        reg = GBDTRegressor(n_estimators=50, verbose=0)
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        assert hasattr(reg, "model_")

    def test_feature_importances(self) -> None:
        """Test feature_importances_ property."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1]  # First two features are important

        reg = GBDTRegressor(n_estimators=20, max_depth=3, verbose=0)
        reg.fit(X, y.astype(np.float32))

        importance = reg.feature_importances_
        assert importance.shape == (5,)
        assert importance.sum() > 0

    def test_clone_and_refit(self) -> None:
        """Test that estimator can be cloned and refitted."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not available")

        from sklearn.base import clone

        np.random.seed(42)
        X = np.random.randn(50, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        reg = GBDTRegressor(n_estimators=10, verbose=0)
        reg.fit(X, y)

        reg2 = clone(reg)
        assert not hasattr(reg2, "model_")
        reg2.fit(X, y)
        assert hasattr(reg2, "model_")


# =============================================================================
# GBDTClassifier Tests
# =============================================================================


class TestGBDTClassifier:
    """Tests for GBDTClassifier."""

    def test_binary_classification(self) -> None:
        """Test binary classification."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        clf = GBDTClassifier(n_estimators=20, max_depth=3, verbose=0)
        clf.fit(X, y)

        assert clf.n_classes_ == 2
        assert np.array_equal(clf.classes_, np.array([0, 1]))

        preds = clf.predict(X)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1})

        # Accuracy should be reasonable
        accuracy = (preds == y).mean()
        assert accuracy > 0.7

    def test_predict_proba_binary(self) -> None:
        """Test predict_proba for binary classification."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        clf = GBDTClassifier(n_estimators=20, verbose=0)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_multiclass_classification(self) -> None:
        """Test multiclass classification."""
        np.random.seed(42)
        X = np.random.randn(150, 5).astype(np.float32)
        # Create 3 classes based on feature values
        y = np.zeros(150, dtype=int)
        y[X[:, 0] > 0.5] = 1
        y[X[:, 0] < -0.5] = 2

        clf = GBDTClassifier(n_estimators=30, max_depth=4, verbose=0)
        clf.fit(X, y)

        assert clf.n_classes_ == 3
        assert np.array_equal(clf.classes_, np.array([0, 1, 2]))

        preds = clf.predict(X)
        assert set(preds).issubset({0, 1, 2})

        proba = clf.predict_proba(X)
        assert proba.shape == (150, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_string_labels(self) -> None:
        """Test classification with string labels."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.where(X[:, 0] > 0, "positive", "negative")

        clf = GBDTClassifier(n_estimators=20, verbose=0)
        clf.fit(X, y)

        assert "positive" in clf.classes_
        assert "negative" in clf.classes_

        preds = clf.predict(X)
        assert all(p in ["positive", "negative"] for p in preds)


# =============================================================================
# GBLinearRegressor Tests
# =============================================================================


class TestGBLinearRegressor:
    """Tests for GBLinearRegressor."""

    def test_fit_predict_linear(self) -> None:
        """Test linear regression on linear data."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        true_weights = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        y = X @ true_weights + 1.0  # Linear with intercept

        reg = GBLinearRegressor(n_estimators=100, learning_rate=0.3, l2=0.01)
        reg.fit(X, y.astype(np.float32))

        assert hasattr(reg, "model_")
        assert reg.n_features_in_ == 3

        preds = reg.predict(X)
        corr = np.corrcoef(preds, y)[0, 1]
        assert corr > 0.95

    def test_coef_intercept(self) -> None:
        """Test coef_ and intercept_ properties."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0] + 2.0

        reg = GBLinearRegressor(n_estimators=50)
        reg.fit(X, y.astype(np.float32))

        assert reg.coef_.shape[0] == 3
        assert reg.intercept_.shape == (1,)


# =============================================================================
# GBLinearClassifier Tests
# =============================================================================


class TestGBLinearClassifier:
    """Tests for GBLinearClassifier."""

    def test_binary_classification(self) -> None:
        """Test binary logistic regression."""
        np.random.seed(42)
        # Linearly separable data
        X = np.random.randn(100, 3).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        clf = GBLinearClassifier(n_estimators=100, learning_rate=0.3)
        clf.fit(X, y)

        assert clf.n_classes_ == 2

        preds = clf.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy > 0.8

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_coef_intercept(self) -> None:
        """Test coef_ and intercept_ for classifier."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        clf = GBLinearClassifier(n_estimators=50)
        clf.fit(X, y)

        # Binary: single coefficient vector
        assert clf.coef_.shape[0] == 3
        assert clf.intercept_.shape == (1,)


# =============================================================================
# Cross-validation and Pipeline Tests
# =============================================================================


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not available")
class TestSklearnIntegration:
    """Test sklearn integration features."""

    def test_cross_val_score_regressor(self) -> None:
        """Test cross_val_score with GBDTRegressor."""
        from sklearn.model_selection import cross_val_score

        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1

        reg = GBDTRegressor(n_estimators=10, verbose=0)
        scores = cross_val_score(reg, X, y, cv=3, scoring="r2")

        assert len(scores) == 3
        assert all(s > 0 for s in scores)  # Better than random

    def test_cross_val_score_classifier(self) -> None:
        """Test cross_val_score with GBDTClassifier."""
        from sklearn.model_selection import cross_val_score

        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)

        clf = GBDTClassifier(n_estimators=10, verbose=0)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")

        assert len(scores) == 3
        assert all(s > 0.5 for s in scores)  # Better than random

    def test_pipeline(self) -> None:
        """Test in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32) * 100  # Large scale
        y = (X[:, 0] > 0).astype(int)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GBDTClassifier(n_estimators=10, verbose=0)),
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (100,)

    def test_grid_search(self) -> None:
        """Test GridSearchCV compatibility."""
        from sklearn.model_selection import GridSearchCV

        np.random.seed(42)
        X = np.random.randn(60, 3).astype(np.float32)
        y = X[:, 0].astype(np.float32)

        reg = GBDTRegressor(verbose=0)
        param_grid = {
            "n_estimators": [5, 10],
            "max_depth": [2, 3],
        }

        grid = GridSearchCV(reg, param_grid, cv=2, scoring="r2")
        grid.fit(X, y)

        assert grid.best_params_ is not None
        assert hasattr(grid.best_estimator_, "model_")
