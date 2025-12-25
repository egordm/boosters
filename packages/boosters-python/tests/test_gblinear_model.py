"""Tests for GBLinearModel training and prediction."""

import numpy as np
import pytest

from boosters import (
    Dataset,
    EvalSet,
    GBLinearConfig,
    GBLinearModel,
    Rmse,
    SquaredLoss,
)


def _make_linear_data(n_samples: int = 200, n_features: int = 5, seed: int = 42):
    """Generate synthetic data with linear relationship for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    # True weights - generate for any n_features
    w = rng.standard_normal(n_features)
    y = X @ w + rng.standard_normal(n_samples) * 0.1
    return X.astype(np.float32), y.astype(np.float32)


class TestGBLinearModelCreation:
    """Tests for GBLinearModel construction."""

    def test_create_with_config(self):
        """GBLinearModel can be created with config."""
        config = GBLinearConfig(n_estimators=50)
        model = GBLinearModel(config=config)
        assert model is not None

    def test_create_without_config_uses_default(self):
        """GBLinearModel uses default config if none provided."""
        model = GBLinearModel()
        assert model is not None

    def test_repr_shows_not_fitted(self):
        """Repr indicates model is not fitted."""
        model = GBLinearModel()
        r = repr(model)
        assert "GBLinearModel" in r
        assert "fitted=False" in r or "not fitted" in r.lower()


class TestGBLinearModelFit:
    """Tests for GBLinearModel.fit()."""

    def test_fit_with_dataset(self):
        """Model trains successfully with Dataset."""
        X, y = _make_linear_data()
        config = GBLinearConfig(n_estimators=10)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X, y))
        assert model.is_fitted

    def test_fit_returns_self(self):
        """fit() returns self for chaining."""
        X, y = _make_linear_data()
        config = GBLinearConfig(n_estimators=5)
        model = GBLinearModel(config=config)
        result = model.fit(Dataset(X, y))
        assert result is model

    def test_fit_with_eval_set(self):
        """Model trains with validation data."""
        X, y = _make_linear_data(200)
        X_val, y_val = _make_linear_data(50, seed=123)
        config = GBLinearConfig(n_estimators=10, metric=Rmse())
        model = GBLinearModel(config=config)
        model.fit(
            Dataset(X, y),
            eval_set=[EvalSet("valid", Dataset(X_val, y_val))],
        )
        assert model.is_fitted

    def test_fit_with_early_stopping(self):
        """Model trains with early stopping configured."""
        X, y = _make_linear_data(200)
        X_val, y_val = _make_linear_data(50, seed=123)
        config = GBLinearConfig(
            n_estimators=100,
            early_stopping_rounds=5,
            metric=Rmse(),
        )
        model = GBLinearModel(config=config)
        model.fit(
            Dataset(X, y),
            eval_set=[EvalSet("valid", Dataset(X_val, y_val))],
        )
        # Model should be fitted
        assert model.is_fitted
        # Predictions should work
        preds = model.predict(X)
        assert preds.shape == (200,)

    def test_refit_replaces_model(self):
        """Fitting again replaces the model."""
        X, y = _make_linear_data()
        config = GBLinearConfig(n_estimators=5)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X, y))
        coef_first = model.coef_.copy()
        # Fit again with different data
        X2, y2 = _make_linear_data(seed=999)
        model.fit(Dataset(X2, y2))
        # Coefficients should be different
        assert not np.allclose(coef_first, model.coef_)


class TestGBLinearModelPredict:
    """Tests for GBLinearModel.predict()."""

    def test_predict_with_array(self):
        """predict() works with numpy array."""
        X, y = _make_linear_data()
        config = GBLinearConfig(n_estimators=20)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X, y))
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert preds.dtype == np.float32

    def test_predict_with_dataset(self):
        """predict() works with Dataset."""
        X, y = _make_linear_data()
        config = GBLinearConfig(n_estimators=20)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X, y))
        preds = model.predict(Dataset(X, y))
        assert preds.shape == (len(X),)

    def test_predict_on_new_data(self):
        """predict() works on unseen data."""
        X_train, y_train = _make_linear_data(200)
        X_test, _ = _make_linear_data(50, seed=123)
        config = GBLinearConfig(n_estimators=20)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X_train, y_train))
        preds = model.predict(X_test)
        assert preds.shape == (50,)

    def test_predict_not_fitted_raises(self):
        """predict() raises if model not fitted."""
        X, _ = _make_linear_data(10)
        model = GBLinearModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)


class TestGBLinearModelProperties:
    """Tests for GBLinearModel properties."""

    def test_coef_shape(self):
        """coef_ has shape (n_features,)."""
        X, y = _make_linear_data(n_features=5)
        config = GBLinearConfig(n_estimators=10)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X, y))
        assert model.coef_.shape == (5,)

    def test_intercept_is_scalar(self):
        """intercept_ is a scalar (or 1-element array)."""
        X, y = _make_linear_data()
        config = GBLinearConfig(n_estimators=10)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X, y))
        assert model.intercept_.shape == (1,) or isinstance(model.intercept_, float)

    def test_n_features_in(self):
        """n_features_in_ matches input features."""
        X, y = _make_linear_data(n_features=7)
        config = GBLinearConfig(n_estimators=10)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X, y))
        assert model.n_features_in_ == 7

    def test_coef_not_fitted_raises(self):
        """coef_ raises if model not fitted."""
        model = GBLinearModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.coef_

    def test_intercept_not_fitted_raises(self):
        """intercept_ raises if model not fitted."""
        model = GBLinearModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.intercept_


class TestGBLinearModelQuality:
    """Tests for GBLinearModel prediction quality."""

    def test_learns_linear_relationship(self):
        """Model learns a simple linear relationship."""
        X, y = _make_linear_data(n_samples=500, n_features=5)
        config = GBLinearConfig(n_estimators=100, learning_rate=0.5)
        model = GBLinearModel(config=config)
        model.fit(Dataset(X, y))
        preds = model.predict(X)
        # Check reasonable correlation
        corr = np.corrcoef(y, preds)[0, 1]
        assert corr > 0.9, f"Correlation too low: {corr}"

    def test_regularization_shrinks_weights(self):
        """L2 regularization shrinks weights toward zero."""
        X, y = _make_linear_data()
        # Low regularization
        config_low = GBLinearConfig(n_estimators=50, l2=0.01)
        model_low = GBLinearModel(config=config_low)
        model_low.fit(Dataset(X, y))
        # High regularization
        config_high = GBLinearConfig(n_estimators=50, l2=100.0)
        model_high = GBLinearModel(config=config_high)
        model_high.fit(Dataset(X, y))
        # High reg should have smaller weights
        assert np.linalg.norm(model_high.coef_) < np.linalg.norm(model_low.coef_)
