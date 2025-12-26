"""Tests for GBDTModel and GBLinearModel.

Focuses on:
- Core model behavior (fit, predict)
- Error handling
- Sklearn-compatible properties
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
    Metric,
    Objective,
)


def make_regression_data(n_samples: int = 200, n_features: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple regression data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)  # noqa: N806
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n_samples) * 0.1).astype(np.float32)
    return X, y


def make_binary_data(n_samples: int = 200, n_features: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)  # noqa: N806
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return X, y


class TestGBDTModelFitPredict:
    """Tests for GBDTModel training and prediction."""

    def test_regression_workflow(self) -> None:
        """Complete regression workflow works."""
        X, y = make_regression_data()  # noqa: N806
        model = GBDTModel(config=GBDTConfig(n_estimators=20))
        model.fit(Dataset(X, y))

        assert model.is_fitted
        assert model.n_trees == 20
        assert model.n_features == 5

        preds = model.predict(Dataset(X))
        assert preds.shape == (200, 1)
        assert preds.dtype == np.float32

    def test_binary_classification_workflow(self) -> None:
        """Binary classification workflow works."""
        X, y = make_binary_data()  # noqa: N806
        model = GBDTModel(config=GBDTConfig(n_estimators=20, objective=Objective.logistic()))
        model.fit(Dataset(X, y))

        preds = model.predict(Dataset(X))
        # Should be probabilities in [0, 1]
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_predict_vs_predict_raw(self) -> None:
        """predict returns transformed, predict_raw returns margins."""
        X, y = make_binary_data()  # noqa: N806
        model = GBDTModel(config=GBDTConfig(n_estimators=20, objective=Objective.logistic()))
        model.fit(Dataset(X, y))

        preds = model.predict(Dataset(X[:10]))
        raw = model.predict_raw(Dataset(X[:10]))

        # Transformed should be in [0, 1], raw can be any value
        assert np.all(preds >= 0) and np.all(preds <= 1)
        assert not np.allclose(preds, raw)

    def test_early_stopping(self) -> None:
        """Early stopping stops before max iterations."""
        X, y = make_regression_data(400)  # noqa: N806
        X_train, X_val = X[:300], X[300:]  # noqa: N806
        y_train, y_val = y[:300], y[300:]

        model = GBDTModel(
            config=GBDTConfig(
                n_estimators=1000,
                early_stopping_rounds=10,
                metric=Metric.rmse(),
            )
        )
        model.fit(
            Dataset(X_train, y_train),
            valid=[EvalSet(Dataset(X_val, y_val), "valid")],
        )

        assert model.n_trees < 1000

    def test_fit_returns_self(self) -> None:
        """fit returns self for method chaining."""
        X, y = make_regression_data()  # noqa: N806
        model = GBDTModel(config=GBDTConfig(n_estimators=5))
        result = model.fit(Dataset(X, y))
        assert result is model

    def test_feature_importance(self) -> None:
        """Feature importance has correct shape."""
        X, y = make_regression_data()  # noqa: N806
        model = GBDTModel(config=GBDTConfig(n_estimators=20))
        model.fit(Dataset(X, y))

        importance = model.feature_importance()
        assert importance.shape == (5,)
        assert np.all(importance >= 0)


class TestGBDTModelErrors:
    """Tests for GBDTModel error handling."""

    def test_predict_before_fit_raises(self) -> None:
        """predict before fit raises clear error."""
        model = GBDTModel()
        rng = np.random.default_rng(42)
        X = rng.random((10, 5)).astype(np.float32)  # noqa: N806
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(Dataset(X))

    def test_wrong_feature_count_raises(self) -> None:
        """Predicting with wrong feature count raises error."""
        X, y = make_regression_data()  # noqa: N806
        model = GBDTModel(config=GBDTConfig(n_estimators=10))
        model.fit(Dataset(X, y))

        rng = np.random.default_rng(42)
        X_wrong = rng.random((10, 3)).astype(np.float32)  # noqa: N806
        with pytest.raises((ValueError, RuntimeError), match="features"):
            model.predict(Dataset(X_wrong))

    def test_fit_without_labels_raises(self) -> None:
        """fit without labels raises error."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 5)).astype(np.float32)  # noqa: N806
        model = GBDTModel()
        with pytest.raises((ValueError, RuntimeError), match="labels"):
            model.fit(Dataset(X))


class TestGBLinearModelFitPredict:
    """Tests for GBLinearModel training and prediction."""

    def test_linear_regression_workflow(self) -> None:
        """Linear regression workflow works."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)  # noqa: N806
        true_weights = np.array([1, 0.5, -0.3, 0.2, 0], dtype=np.float32)
        y = X @ true_weights

        model = GBLinearModel(config=GBLinearConfig(n_estimators=100, learning_rate=0.5))
        model.fit(Dataset(X, y))

        assert model.is_fitted
        assert model.coef_.shape == (5,)
        assert model.intercept_.shape == (1,)

        preds = model.predict(Dataset(X))
        assert preds.shape == (200, 1)

    def test_regularization_shrinks_weights(self) -> None:
        """L2 regularization shrinks weights."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)  # noqa: N806
        y = (X[:, 0] + X[:, 1]).astype(np.float32)

        model_low_reg = GBLinearModel(config=GBLinearConfig(n_estimators=50, l2=0.01))
        model_low_reg.fit(Dataset(X, y))

        model_high_reg = GBLinearModel(config=GBLinearConfig(n_estimators=50, l2=100.0))
        model_high_reg.fit(Dataset(X, y))

        assert np.linalg.norm(model_high_reg.coef_) < np.linalg.norm(model_low_reg.coef_)
