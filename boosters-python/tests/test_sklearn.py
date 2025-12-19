"""Test sklearn-compatible wrappers."""

import numpy as np
import pytest

# Skip all tests if sklearn is not available
sklearn = pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boosters.sklearn import (
    GBDTClassifier,
    GBDTRegressor,
    GBLinearClassifier,
    GBLinearRegressor,
)


@pytest.fixture
def regression_data():
    """Generate regression data."""
    np.random.seed(42)
    n_samples, n_features = 200, 5
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)).astype(np.float32)
    return X, y


@pytest.fixture
def classification_data():
    """Generate binary classification data."""
    np.random.seed(42)
    n_samples, n_features = 200, 5
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate multiclass classification data."""
    np.random.seed(42)
    n_samples, n_features = 300, 5
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # 3 classes based on sum of first two features
    sums = X[:, 0] + X[:, 1]
    y = np.where(sums < -0.5, 0, np.where(sums > 0.5, 2, 1)).astype(int)
    return X, y


class TestGBDTRegressorSklearn:
    """Test GBDTRegressor sklearn compatibility."""

    def test_basic_fit_predict(self, regression_data):
        """Test basic fit and predict."""
        X, y = regression_data
        model = GBDTRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert model.n_features_in_ == 5

    def test_get_set_params(self, regression_data):
        """Test get_params and set_params."""
        model = GBDTRegressor(n_estimators=50, learning_rate=0.1)

        params = model.get_params()
        assert params['n_estimators'] == 50
        assert params['learning_rate'] == 0.1

        model.set_params(n_estimators=100)
        assert model.n_estimators == 100

    def test_clone(self, regression_data):
        """Test sklearn clone."""
        X, y = regression_data
        model = GBDTRegressor(n_estimators=10)
        model.fit(X, y)

        cloned = clone(model)
        # Clone should have same params but not be fitted
        assert cloned.n_estimators == 10
        assert not hasattr(cloned, 'booster_')

    def test_cross_val_score(self, regression_data):
        """Test sklearn cross_val_score."""
        X, y = regression_data
        model = GBDTRegressor(n_estimators=10, max_depth=3)

        scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
        assert len(scores) == 3
        assert all(s < 0 for s in scores)  # neg_mse is negative

    def test_pipeline(self, regression_data):
        """Test sklearn Pipeline."""
        X, y = regression_data

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GBDTRegressor(n_estimators=10)),
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)

    def test_grid_search(self, regression_data):
        """Test sklearn GridSearchCV."""
        X, y = regression_data
        model = GBDTRegressor()

        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [2, 3],
        }

        grid = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error')
        grid.fit(X, y)

        assert grid.best_params_ is not None
        assert 'n_estimators' in grid.best_params_

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ property."""
        X, y = regression_data
        model = GBDTRegressor(n_estimators=20)
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.shape == (5,)
        assert importances.sum() > 0


class TestGBDTClassifierSklearn:
    """Test GBDTClassifier sklearn compatibility."""

    def test_binary_classification(self, classification_data):
        """Test binary classification."""
        X, y = classification_data
        model = GBDTClassifier(n_estimators=20, max_depth=3)
        model.fit(X, y)

        # Check classes
        assert len(model.classes_) == 2

        # Check predictions
        preds = model.predict(X)
        assert set(preds).issubset(set(model.classes_))

        # Check probabilities
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_multiclass_classification(self, multiclass_data):
        """Test multiclass classification."""
        X, y = multiclass_data
        model = GBDTClassifier(n_estimators=30, max_depth=4)
        model.fit(X, y)

        assert model.n_classes_ == 3

        preds = model.predict(X)
        assert set(preds).issubset({0, 1, 2})

        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_cross_val_score(self, classification_data):
        """Test sklearn cross_val_score."""
        X, y = classification_data
        model = GBDTClassifier(n_estimators=10)

        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_pipeline(self, classification_data):
        """Test sklearn Pipeline."""
        X, y = classification_data

        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GBDTClassifier(n_estimators=10)),
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)
        proba = pipe.predict_proba(X)

        assert preds.shape == (len(X),)
        assert proba.shape == (len(X), 2)


class TestGBLinearSklearn:
    """Test GBLinear sklearn wrappers."""

    def test_regressor_fit_predict(self, regression_data):
        """Test GBLinearRegressor."""
        X, y = regression_data
        model = GBLinearRegressor(n_estimators=50)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),)

        # Check coef_ and intercept_
        assert model.coef_.shape == (5,)
        assert isinstance(model.intercept_, (float, np.floating))

    def test_classifier_fit_predict(self, classification_data):
        """Test GBLinearClassifier."""
        X, y = classification_data
        model = GBLinearClassifier(n_estimators=50)
        model.fit(X, y)

        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert set(preds).issubset(set(model.classes_))
        assert proba.shape == (len(X), 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
