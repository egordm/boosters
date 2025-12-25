"""Integration tests demonstrating complete training workflows.

This module provides comprehensive tests showing:
1. Complete train/predict workflow for GBDTModel
2. Complete train/predict workflow for GBLinearModel
3. Early stopping and validation set usage
4. Various objectives (regression, classification)

These tests serve as both verification and documentation of the API.
"""

import numpy as np
import pytest

from boosters import (
    Dataset,
    EvalSet,
    GBDTConfig,
    GBDTModel,
    GBLinearConfig,
    GBLinearModel,
    LogisticLoss,
    Rmse,
    SquaredLoss,
    TreeConfig,
)

# =============================================================================
# Helper functions for generating test data
# =============================================================================


def make_regression_data(
    n_samples: int = 500,
    n_features: int = 10,
    seed: int = 42,
    noise: float = 0.1,
):
    """Generate synthetic regression data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # Non-linear relationship with some features
    y = (
        np.sin(X[:, 0] * 2) + X[:, 1] ** 2 + X[:, 2] * 0.5 + rng.standard_normal(n_samples) * noise
    ).astype(np.float32)
    return X, y


def make_binary_classification_data(
    n_samples: int = 500,
    n_features: int = 10,
    seed: int = 42,
):
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # Decision boundary based on first two features
    logits = X[:, 0] + X[:, 1] * 0.5 + rng.standard_normal(n_samples) * 0.3
    y = (logits > 0).astype(np.float32)
    return X, y


def make_linear_data(
    n_samples: int = 500,
    n_features: int = 10,
    seed: int = 42,
    noise: float = 0.1,
):
    """Generate synthetic data with linear relationship."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # True weights
    weights = rng.standard_normal(n_features).astype(np.float32)
    y = (X @ weights + rng.standard_normal(n_samples) * noise).astype(np.float32)
    return X, y


# =============================================================================
# GBDTModel Integration Tests
# =============================================================================


class TestGBDTModelIntegration:
    """Complete workflow tests for GBDTModel."""

    def test_basic_regression_workflow(self):
        """Test complete regression workflow: train → predict → evaluate."""
        # Generate data
        X_train, y_train = make_regression_data(500, 10, seed=42)
        X_test, y_test = make_regression_data(100, 10, seed=123)

        # Configure model
        config = GBDTConfig(
            n_estimators=50,
            learning_rate=0.1,
            tree=TreeConfig(max_depth=5),
            objective=SquaredLoss(),
        )

        # Train
        model = GBDTModel(config=config)
        model.fit(Dataset(X_train, y_train))

        # Verify model is fitted
        assert model.is_fitted
        assert model.n_trees > 0
        assert model.n_features == 10

        # Predict
        preds = model.predict(X_test)
        assert preds.shape == (100,)
        assert preds.dtype == np.float32

        # Basic quality check - predictions should correlate with targets
        correlation = np.corrcoef(y_test, preds)[0, 1]
        assert correlation > 0.5, f"Correlation too low: {correlation:.3f}"

    def test_regression_with_validation(self):
        """Test regression with validation set for monitoring."""
        X_train, y_train = make_regression_data(400, 10, seed=42)
        X_val, y_val = make_regression_data(100, 10, seed=123)

        config = GBDTConfig(
            n_estimators=100,
            learning_rate=0.1,
            tree=TreeConfig(max_depth=5),
            objective=SquaredLoss(),
            metric=Rmse(),
        )

        model = GBDTModel(config=config)
        model.fit(
            Dataset(X_train, y_train),
            valid=[EvalSet("validation", Dataset(X_val, y_val))],
        )

        assert model.is_fitted
        assert model.n_trees == 100

    def test_regression_with_early_stopping(self):
        """Test early stopping prevents overfitting."""
        X_train, y_train = make_regression_data(400, 10, seed=42)
        X_val, y_val = make_regression_data(100, 10, seed=123)

        config = GBDTConfig(
            n_estimators=1000,  # High limit
            early_stopping_rounds=10,
            learning_rate=0.1,
            tree=TreeConfig(max_depth=5),
            objective=SquaredLoss(),
            metric=Rmse(),
        )

        model = GBDTModel(config=config)
        model.fit(
            Dataset(X_train, y_train),
            valid=[EvalSet("validation", Dataset(X_val, y_val))],
        )

        assert model.is_fitted
        # Early stopping should trigger before 1000
        assert model.n_trees < 1000
        # Model should make reasonable predictions
        preds = model.predict(X_val)
        assert preds.shape == (100,)

    def test_binary_classification_workflow(self):
        """Test complete binary classification workflow."""
        X_train, y_train = make_binary_classification_data(500, 10, seed=42)
        X_test, y_test = make_binary_classification_data(100, 10, seed=123)

        config = GBDTConfig(
            n_estimators=50,
            learning_rate=0.1,
            tree=TreeConfig(max_depth=4),
            objective=LogisticLoss(),
        )

        model = GBDTModel(config=config)
        model.fit(Dataset(X_train, y_train))

        # Predict probabilities
        preds = model.predict(X_test)
        assert preds.shape == (100,)

        # Probabilities should be in [0, 1] range
        assert np.all(preds >= 0) and np.all(preds <= 1)

        # Basic quality check - should do better than random
        pred_classes = (preds > 0.5).astype(np.float32)
        accuracy = np.mean(pred_classes == y_test)
        assert accuracy > 0.6, f"Accuracy too low: {accuracy:.3f}"

    def test_raw_score_prediction(self):
        """Test raw score (margin) predictions."""
        X, y = make_binary_classification_data(500, 10, seed=42)

        config = GBDTConfig(
            n_estimators=30,
            objective=LogisticLoss(),
        )

        model = GBDTModel(config=config).fit(Dataset(X, y))

        # Normal predictions (probabilities)
        preds = model.predict(X[:10])
        # Raw predictions (logits)
        raw_preds = model.predict(X[:10], raw_score=True)

        # Raw scores should be unbounded
        assert preds.shape == raw_preds.shape
        # For logistic loss, prob = sigmoid(raw)
        # Just check they're different and raw can be negative/positive
        assert not np.allclose(preds, raw_preds)

    def test_feature_importance(self):
        """Test feature importance extraction."""
        X, y = make_regression_data(500, 10, seed=42)

        config = GBDTConfig(n_estimators=50, tree=TreeConfig(max_depth=5))
        model = GBDTModel(config=config).fit(Dataset(X, y))

        # Get feature importance
        importance = model.feature_importance()
        assert importance.shape == (10,)
        assert np.all(importance >= 0)
        # At least some features should be used
        assert np.sum(importance) > 0

    def test_predict_with_dataset_object(self):
        """Test that predict works with Dataset as well as arrays."""
        X, y = make_regression_data(500, 10, seed=42)

        config = GBDTConfig(n_estimators=30)
        model = GBDTModel(config=config).fit(Dataset(X, y))

        # Predict with array
        preds_array = model.predict(X[:10])
        # Predict with Dataset
        preds_dataset = model.predict(Dataset(X[:10], y[:10]))

        np.testing.assert_array_equal(preds_array, preds_dataset)


# =============================================================================
# GBLinearModel Integration Tests
# =============================================================================


class TestGBLinearModelIntegration:
    """Complete workflow tests for GBLinearModel."""

    def test_basic_linear_regression_workflow(self):
        """Test complete linear regression workflow."""
        # Use simpler data generation for more reliable linear fit
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((500, 10)).astype(np.float32)
        # Simple linear relationship with known weights
        true_weights = np.array([1, 0.5, -0.3, 0.2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        y_train = X_train @ true_weights + rng.standard_normal(500).astype(np.float32) * 0.1

        X_test = rng.standard_normal((100, 10)).astype(np.float32)
        y_test = X_test @ true_weights

        config = GBLinearConfig(
            n_estimators=100,
            learning_rate=0.5,
            l2=0.1,
        )

        model = GBLinearModel(config=config)
        model.fit(Dataset(X_train, y_train))

        assert model.is_fitted
        assert model.n_features_in_ == 10

        preds = model.predict(X_test)
        assert preds.shape == (100,)

        # Linear model should do reasonably well on clean linear data
        correlation = np.corrcoef(y_test, preds)[0, 1]
        assert correlation > 0.5, f"Correlation too low: {correlation:.3f}"

    def test_sklearn_compatible_properties(self):
        """Test sklearn-compatible coefficient properties."""
        X, y = make_linear_data(500, 10, seed=42)

        config = GBLinearConfig(n_estimators=50)
        model = GBLinearModel(config=config).fit(Dataset(X, y))

        # Check coef_ and intercept_
        coef = model.coef_
        intercept = model.intercept_

        assert coef.shape == (10,)
        assert intercept.shape == (1,)

    def test_regularization_effect(self):
        """Test that L2 regularization shrinks weights."""
        X, y = make_linear_data(500, 10, seed=42)

        # Low regularization
        model_low = GBLinearModel(config=GBLinearConfig(n_estimators=50, l2=0.01)).fit(
            Dataset(X, y)
        )

        # High regularization
        model_high = GBLinearModel(config=GBLinearConfig(n_estimators=50, l2=10.0)).fit(
            Dataset(X, y)
        )

        # High reg should have smaller weights
        norm_low = np.linalg.norm(model_low.coef_)
        norm_high = np.linalg.norm(model_high.coef_)

        assert norm_high < norm_low

    def test_with_validation_set(self):
        """Test training with validation set."""
        X_train, y_train = make_linear_data(400, 10, seed=42)
        X_val, y_val = make_linear_data(100, 10, seed=123)

        config = GBLinearConfig(
            n_estimators=100,
            metric=Rmse(),
        )

        model = GBLinearModel(config=config)
        model.fit(
            Dataset(X_train, y_train),
            eval_set=[EvalSet("validation", Dataset(X_val, y_val))],
        )

        assert model.is_fitted


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling and validation."""

    def test_predict_before_fit_raises(self):
        """Predict before fit should raise clear error."""
        model = GBDTModel()
        X = np.random.rand(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_wrong_feature_count_raises(self):
        """Predicting with wrong feature count should raise error."""
        X_train = np.random.rand(100, 10).astype(np.float32)
        y_train = np.random.rand(100).astype(np.float32)

        model = GBDTModel(config=GBDTConfig(n_estimators=10))
        model.fit(Dataset(X_train, y_train))

        # Predict with wrong number of features
        X_wrong = np.random.rand(10, 5).astype(np.float32)

        with pytest.raises(ValueError, match="features"):
            model.predict(X_wrong)

    def test_missing_labels_for_training_raises(self):
        """Training without labels should raise error."""
        X = np.random.rand(100, 10).astype(np.float32)

        model = GBDTModel(config=GBDTConfig(n_estimators=10))

        with pytest.raises(ValueError, match="labels"):
            model.fit(Dataset(X))  # No labels


# =============================================================================
# Method Chaining Tests
# =============================================================================


class TestMethodChaining:
    """Tests for fluent API (method chaining)."""

    def test_fit_returns_self(self):
        """fit() returns self for chaining."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.rand(100).astype(np.float32)

        # Train and predict in one line
        preds = GBDTModel(config=GBDTConfig(n_estimators=10)).fit(Dataset(X, y)).predict(X[:10])

        assert preds.shape == (10,)
