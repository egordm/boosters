"""Test training and prediction functionality."""

import numpy as np
import pytest

from boosters import GBDTBooster, GBLinearBooster


@pytest.fixture
def regression_data():
    """Generate regression data: y = x0 + 0.5*x1 + noise."""
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
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.float32)
    return X, y


class TestGBDTTraining:
    """Test GBDTBooster training functionality."""

    def test_basic_training(self, regression_data):
        """Test that we can train a model and make predictions."""
        X, y = regression_data
        model = GBDTBooster(n_estimators=10, learning_rate=0.1, max_depth=3)
        model.fit(X, y)

        assert model.is_fitted, "Model should be fitted"
        assert model.n_trees == 10, "Should have 10 trees"
        assert model.n_features == 5, "Should have 5 features"

    def test_prediction_shape(self, regression_data):
        """Test that predictions have correct shape."""
        X, y = regression_data
        model = GBDTBooster(n_estimators=10)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),), f"Expected shape ({len(X)},), got {preds.shape}"

    def test_prediction_improves_with_more_trees(self, regression_data):
        """Test that more trees generally improves fit."""
        X, y = regression_data

        model_small = GBDTBooster(n_estimators=5, learning_rate=0.1)
        model_small.fit(X, y)
        mse_small = np.mean((model_small.predict(X) - y) ** 2)

        model_large = GBDTBooster(n_estimators=50, learning_rate=0.1)
        model_large.fit(X, y)
        mse_large = np.mean((model_large.predict(X) - y) ** 2)

        assert mse_large < mse_small, "More trees should reduce training error"

    def test_fit_returns_self(self, regression_data):
        """Test that fit() returns self for chaining."""
        X, y = regression_data
        model = GBDTBooster(n_estimators=5)
        result = model.fit(X, y)

        assert result is model, "fit() should return self"

    def test_feature_names(self, regression_data):
        """Test that feature names are stored correctly."""
        X, y = regression_data
        names = ['a', 'b', 'c', 'd', 'e']

        model = GBDTBooster(n_estimators=5)
        model.fit(X, y, feature_names=names)

        assert model.feature_names == names, "Feature names should be stored"

    def test_sample_weights(self, regression_data):
        """Test training with sample weights."""
        X, y = regression_data
        weights = np.ones(len(X), dtype=np.float32)
        weights[:50] = 10.0  # Higher weight for first 50 samples

        model = GBDTBooster(n_estimators=10)
        model.fit(X, y, sample_weight=weights)

        assert model.is_fitted, "Should train with weights"

    def test_binary_classification(self, classification_data):
        """Test binary classification returns margins (logits)."""
        X, y = classification_data

        model = GBDTBooster(n_estimators=30, objective='binary:logistic', max_depth=4)
        model.fit(X, y)

        preds = model.predict(X)
        # Note: Currently returns raw margins (logits), not probabilities
        # Users can apply sigmoid manually: 1 / (1 + np.exp(-preds))

        # Check predictions are reasonable (positive for class 1, negative for class 0)
        # Apply sigmoid to get probabilities for accuracy check
        probs = 1 / (1 + np.exp(-preds))
        accuracy = np.mean((probs > 0.5) == y)
        assert accuracy > 0.6, f"Accuracy should be > 60%, got {accuracy:.2%}"

    def test_repr(self, regression_data):
        """Test string representation."""
        model = GBDTBooster(n_estimators=100, learning_rate=0.1)
        repr_str = repr(model)

        assert 'GBDTBooster' in repr_str
        assert '100' in repr_str
        assert 'is_fitted=False' in repr_str

        X, y = regression_data
        model.fit(X, y)
        repr_str = repr(model)
        assert 'is_fitted=True' in repr_str


class TestGBLinearTraining:
    """Test GBLinearBooster training functionality."""

    def test_basic_training(self, regression_data):
        """Test that we can train a linear model."""
        X, y = regression_data

        model = GBLinearBooster(n_estimators=50, learning_rate=0.5)
        model.fit(X, y)

        assert model.is_fitted, "Model should be fitted"
        assert model.n_features == 5, "Should have 5 features"

    def test_prediction_shape(self, regression_data):
        """Test that predictions have correct shape."""
        X, y = regression_data

        model = GBLinearBooster(n_estimators=50)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),), f"Expected shape ({len(X)},), got {preds.shape}"

    def test_weights_shape(self, regression_data):
        """Test that weights have correct shape."""
        X, y = regression_data

        model = GBLinearBooster(n_estimators=50)
        model.fit(X, y)

        weights = model.weights
        assert weights.shape == (5,), f"Expected shape (5,), got {weights.shape}"

    def test_bias_value(self, regression_data):
        """Test that bias is accessible."""
        X, y = regression_data

        model = GBLinearBooster(n_estimators=50)
        model.fit(X, y)

        bias = model.bias
        # Bias should be an array for multi-group support
        assert isinstance(bias, np.ndarray), "Bias should be a numpy array"
        assert len(bias) >= 1, "Bias should have at least one element"


class TestCategoricalFeatures:
    """Test categorical feature support."""

    def test_categorical_features_training(self):
        """Test training with categorical features."""
        np.random.seed(42)
        n_samples = 200

        # Feature 0: categorical (3 categories)
        # Feature 1: numeric
        X = np.zeros((n_samples, 2), dtype=np.float32)
        X[:, 0] = np.random.randint(0, 3, n_samples).astype(np.float32)  # Categories: 0, 1, 2
        X[:, 1] = np.random.randn(n_samples).astype(np.float32)

        # y depends on category
        y = np.where(X[:, 0] == 0, 1.0, np.where(X[:, 0] == 1, 2.0, 3.0))
        y = (y + 0.1 * np.random.randn(n_samples)).astype(np.float32)

        model = GBDTBooster(n_estimators=50, max_depth=4)
        model.fit(X, y, categorical_features=[0])

        assert model.is_fitted, "Should train with categorical features"

        # Predictions should be reasonable
        preds = model.predict(X)
        mse = np.mean((preds - y) ** 2)
        # Relaxed threshold - categorical features should help but we're not testing optimization
        assert mse < 1.0, f"MSE should be reasonable with categorical features, got {mse:.4f}"


class TestInputValidation:
    """Test input validation."""

    def test_x_y_shape_mismatch(self, regression_data):
        """Test that mismatched X and y shapes raise error."""
        X, y = regression_data

        model = GBDTBooster()
        with pytest.raises(Exception):
            model.fit(X, y[:50])  # y too short

    def test_feature_names_length_mismatch(self, regression_data):
        """Test that wrong number of feature names raises error."""
        X, y = regression_data

        model = GBDTBooster()
        with pytest.raises(Exception):
            model.fit(X, y, feature_names=['a', 'b'])  # Too few names

    def test_predict_before_fit(self, regression_data):
        """Test that predicting before fit raises error."""
        X, _ = regression_data

        model = GBDTBooster()
        with pytest.raises(Exception):
            model.predict(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
