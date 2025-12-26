"""Tests for GBDTModel type."""

import numpy as np
import pytest

import boosters as bst
from boosters import Dataset, EvalSet, GBDTConfig, GBDTModel


class TestGBDTModelConstruction:
    """Tests for GBDTModel construction."""

    def test_default_construction(self) -> None:
        """Test constructing with default config."""
        model = GBDTModel()
        assert not model.is_fitted
        assert model.config is not None

    def test_construction_with_config(self) -> None:
        """Test constructing with custom config."""
        config = GBDTConfig(n_estimators=50, learning_rate=0.1)
        model = GBDTModel(config=config)
        assert not model.is_fitted
        assert model.config.n_estimators == 50
        assert model.config.learning_rate == pytest.approx(0.1)


class TestGBDTModelProperties:
    """Tests for GBDTModel properties before fitting."""

    def test_n_trees_raises_before_fit(self) -> None:
        """Test that n_trees raises error before fit."""
        model = GBDTModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.n_trees

    def test_n_features_raises_before_fit(self) -> None:
        """Test that n_features raises error before fit."""
        model = GBDTModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.n_features

    def test_feature_importance_raises_before_fit(self) -> None:
        """Test that feature_importance raises error before fit."""
        model = GBDTModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.feature_importance()

    def test_best_iteration_none_before_fit(self) -> None:
        """Test that best_iteration is None before fit."""
        model = GBDTModel()
        assert model.best_iteration is None

    def test_best_score_none_before_fit(self) -> None:
        """Test that best_score is None before fit."""
        model = GBDTModel()
        assert model.best_score is None

    def test_eval_results_none_before_fit(self) -> None:
        """Test that eval_results is None before fit."""
        model = GBDTModel()
        assert model.eval_results is None


class TestGBDTModelRepr:
    """Tests for GBDTModel string representation."""

    def test_repr_unfitted(self) -> None:
        """Test repr for unfitted model."""
        model = GBDTModel()
        r = repr(model)
        assert "fitted=False" in r


class TestGBDTModelFit:
    """Tests for GBDTModel.fit() method."""

    @pytest.fixture
    def regression_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate simple regression data."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(100) * 0.1).astype(np.float32)
        return X, y

    @pytest.fixture
    def binary_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate simple binary classification data."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        return X, y

    def test_fit_regression(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic regression fit."""
        X, y = regression_data
        train = Dataset(X, y)
        model = GBDTModel(config=GBDTConfig(n_estimators=10))
        model.fit(train)

        assert model.is_fitted
        assert model.n_trees == 10
        assert model.n_features == 5

    def test_fit_binary_classification(self, binary_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test basic binary classification fit."""
        X, y = binary_data
        train = Dataset(X, y)
        model = GBDTModel(config=GBDTConfig(n_estimators=10, objective=bst.LogisticLoss()))
        model.fit(train)

        assert model.is_fitted
        assert model.n_trees == 10

    def test_fit_returns_self(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that fit returns self for method chaining."""
        X, y = regression_data
        train = Dataset(X, y)
        model = GBDTModel(config=GBDTConfig(n_estimators=5))
        result = model.fit(train)

        assert result is model

    def test_fit_with_eval_set(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test fit with evaluation set."""
        X, y = regression_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        train = Dataset(X_train, y_train)
        val_ds = Dataset(X_val, y_val)
        valid = EvalSet("valid", val_ds._inner)

        model = GBDTModel(config=GBDTConfig(n_estimators=10))
        model.fit(train._inner, valid=[valid])

        assert model.is_fitted
        # eval_results will be populated in story 4.6
        # For now, just ensure fit doesn't crash

    def test_fit_with_early_stopping(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test fit with early stopping configuration."""
        X, y = regression_data
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        train = Dataset(X_train, y_train)
        val_ds = Dataset(X_val, y_val)
        valid = EvalSet("valid", val_ds._inner)

        model = GBDTModel(config=GBDTConfig(n_estimators=100, early_stopping_rounds=5))
        model.fit(train._inner, valid=[valid])

        assert model.is_fitted
        # May have stopped early
        assert model.n_trees <= 100

    def test_fit_requires_labels(self, regression_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that fit raises error if dataset has no labels."""
        X, _ = regression_data
        train = Dataset(X)  # No labels
        model = GBDTModel()
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(train)


class TestGBDTModelPredict:
    """Tests for GBDTModel.predict() method."""

    @pytest.fixture
    def fitted_regression_model(self) -> tuple[GBDTModel, np.ndarray, np.ndarray]:
        """Return a fitted regression model with test data."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(100) * 0.1).astype(np.float32)
        train = Dataset(X, y)
        model = GBDTModel(config=GBDTConfig(n_estimators=10))
        model.fit(train)
        return model, X, y

    @pytest.fixture
    def fitted_classification_model(self) -> tuple[GBDTModel, np.ndarray, np.ndarray]:
        """Return a fitted classification model with test data."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        train = Dataset(X, y)
        model = GBDTModel(config=GBDTConfig(n_estimators=10, objective=bst.LogisticLoss()))
        model.fit(train)
        return model, X, y

    def test_predict_returns_correct_shape(
        self, fitted_regression_model: tuple[GBDTModel, np.ndarray, np.ndarray]
    ) -> None:
        """Test that predict returns correct shape for regression."""
        model, X, _ = fitted_regression_model
        predictions = model.predict(X)

        # Always returns 2D array (n_samples, n_outputs)
        assert predictions.shape == (100, 1)
        assert predictions.dtype == np.float32

    def test_predict_with_dataset(
        self, fitted_regression_model: tuple[GBDTModel, np.ndarray, np.ndarray]
    ) -> None:
        """Test that predict works with Dataset input."""
        model, X, _y = fitted_regression_model
        test_ds = Dataset(X)  # No labels needed for prediction
        predictions = model.predict(test_ds)

        assert predictions.shape == (100, 1)

    def test_predict_raw_score(
        self, fitted_classification_model: tuple[GBDTModel, np.ndarray, np.ndarray]
    ) -> None:
        """Test that predict_raw returns margins."""
        model, X, _ = fitted_classification_model
        normal_preds = model.predict(X)
        raw_preds = model.predict_raw(X)

        # Raw should be logits (can be any value), normal should be probabilities [0, 1]
        assert raw_preds.shape == (100, 1)
        # Transformed predictions should be in [0, 1] range
        assert np.all(normal_preds >= 0) and np.all(normal_preds <= 1)
        # Raw can be outside [0, 1]
        # Note: they might still be in [0,1] by chance, but relationship differs

    def test_predict_raises_if_not_fitted(self) -> None:
        """Test that predict raises error if model not fitted."""
        model = GBDTModel()
        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_predict_validates_feature_count(
        self, fitted_regression_model: tuple[GBDTModel, np.ndarray, np.ndarray]
    ) -> None:
        """Test that predict validates feature count."""
        model, _, _ = fitted_regression_model
        X_wrong = np.random.randn(10, 3).astype(np.float32)  # 3 features, trained on 5

        with pytest.raises((ValueError, RuntimeError)):
            model.predict(X_wrong)


class TestGBDTModelImports:
    """Test that GBDTModel is properly exported."""

    def test_import_from_main(self) -> None:
        """Test importing from main boosters module."""
        assert hasattr(bst, "GBDTModel")

    def test_import_from_model(self) -> None:
        """Test importing from boosters.model module."""
        from boosters.model import GBDTModel as M

        assert M is GBDTModel
