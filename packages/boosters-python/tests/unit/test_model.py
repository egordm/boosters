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
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n_samples) * 0.1).astype(np.float32)
    return X, y


def make_binary_data(n_samples: int = 200, n_features: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate simple binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return X, y


class TestGBDTModelFitPredict:
    """Tests for GBDTModel training and prediction."""

    def test_regression_workflow(self) -> None:
        """Complete regression workflow works."""
        X, y = make_regression_data()
        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=20))
        assert model.n_trees == 20
        assert model.n_features == 5

        preds = model.predict(Dataset(X))
        assert preds.shape == (200, 1)
        assert preds.dtype == np.float32

    def test_binary_classification_workflow(self) -> None:
        """Binary classification workflow works."""
        X, y = make_binary_data()
        model = GBDTModel.train(
            Dataset(X, y),
            config=GBDTConfig(n_estimators=20, objective=Objective.logistic()),
        )

        preds = model.predict(Dataset(X))
        # Should be probabilities in [0, 1]
        assert np.all(preds >= 0) and np.all(preds <= 1)

    def test_predict_vs_predict_raw(self) -> None:
        """predict returns transformed, predict_raw returns margins."""
        X, y = make_binary_data()
        model = GBDTModel.train(
            Dataset(X, y),
            config=GBDTConfig(n_estimators=20, objective=Objective.logistic()),
        )

        preds = model.predict(Dataset(X[:10]))
        raw = model.predict_raw(Dataset(X[:10]))

        # Transformed should be in [0, 1], raw can be any value
        assert np.all(preds >= 0) and np.all(preds <= 1)
        assert not np.allclose(preds, raw)

    def test_early_stopping(self) -> None:
        """Early stopping stops before max iterations."""
        X, y = make_regression_data(400)
        X_train, X_val = X[:300], X[300:]
        y_train, y_val = y[:300], y[300:]

        model = GBDTModel.train(
            Dataset(X_train, y_train),
            val_set=Dataset(X_val, y_val),
            config=GBDTConfig(
                n_estimators=1000,
                early_stopping_rounds=10,
                metric=Metric.rmse(),
            ),
        )

        assert model.n_trees < 1000

    def test_feature_importance(self) -> None:
        """Feature importance has correct shape."""
        X, y = make_regression_data()
        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=20))

        importance = model.feature_importance()
        assert importance.shape == (5,)
        assert np.all(importance >= 0)


class TestGBDTModelErrors:
    """Tests for GBDTModel error handling."""

    def test_predict_before_fit_raises(self) -> None:
        """Unfitted models are not constructible; training requires labels."""
        rng = np.random.default_rng(42)
        X = rng.random((10, 5)).astype(np.float32)

        with pytest.raises((ValueError, RuntimeError), match="labels"):
            GBDTModel.train(Dataset(X), config=GBDTConfig(n_estimators=3))

    def test_wrong_feature_count_raises(self) -> None:
        """Predicting with wrong feature count raises error."""
        X, y = make_regression_data()
        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=10))

        rng = np.random.default_rng(42)
        X_wrong = rng.random((10, 3)).astype(np.float32)
        with pytest.raises((ValueError, RuntimeError), match="features"):
            model.predict(Dataset(X_wrong))

    def test_fit_without_labels_raises(self) -> None:
        """train without labels raises error."""
        rng = np.random.default_rng(42)
        X = rng.random((100, 5)).astype(np.float32)
        with pytest.raises((ValueError, RuntimeError), match="labels"):
            GBDTModel.train(Dataset(X), config=GBDTConfig(n_estimators=3))


class TestGBLinearModelFitPredict:
    """Tests for GBLinearModel training and prediction."""

    def test_linear_regression_workflow(self) -> None:
        """Linear regression workflow works."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        true_weights = np.array([1, 0.5, -0.3, 0.2, 0], dtype=np.float32)
        y = X @ true_weights

        model = GBLinearModel.train(
            Dataset(X, y),
            config=GBLinearConfig(n_estimators=100, learning_rate=0.5),
        )
        assert model.coef_.shape == (5,)
        assert model.intercept_.shape == (1,)

        preds = model.predict(Dataset(X))
        assert preds.shape == (200, 1)

    def test_regularization_shrinks_weights(self) -> None:
        """L2 regularization shrinks weights."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] + X[:, 1]).astype(np.float32)

        model_low_reg = GBLinearModel.train(
            Dataset(X, y),
            config=GBLinearConfig(n_estimators=50, l2=0.01),
        )

        model_high_reg = GBLinearModel.train(
            Dataset(X, y),
            config=GBLinearConfig(n_estimators=50, l2=100.0),
        )

        assert np.linalg.norm(model_high_reg.coef_) < np.linalg.norm(model_low_reg.coef_)


class TestGBDTModelPersistence:
    """Tests for GBDTModel serialization round-trip."""

    def test_binary_roundtrip_preserves_predictions(self) -> None:
        """to_bytes → from_bytes preserves predictions exactly."""
        X, y = make_regression_data()
        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=20))

        original_preds = model.predict(Dataset(X))

        data = model.to_bytes()
        loaded = GBDTModel.from_bytes(data)
        loaded_preds = loaded.predict(Dataset(X))

        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_json_roundtrip_preserves_predictions(self) -> None:
        """to_json_bytes → from_json_bytes preserves predictions."""
        X, y = make_regression_data()
        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=20))

        original_preds = model.predict(Dataset(X))

        data = model.to_json_bytes()
        loaded = GBDTModel.from_json_bytes(data)
        loaded_preds = loaded.predict(Dataset(X))

        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_binary_classification_roundtrip(self) -> None:
        """Binary classification model roundtrips correctly."""
        X, y = make_binary_data()
        model = GBDTModel.train(
            Dataset(X, y),
            config=GBDTConfig(n_estimators=20, objective=Objective.logistic()),
        )

        original_preds = model.predict(Dataset(X))

        loaded = GBDTModel.from_bytes(model.to_bytes())
        loaded_preds = loaded.predict(Dataset(X))

        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_serialized_model_properties(self) -> None:
        """Loaded model has correct properties."""
        X, y = make_regression_data()
        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=20))

        loaded = GBDTModel.from_bytes(model.to_bytes())

        assert loaded.n_trees == model.n_trees
        assert loaded.n_features == model.n_features


class TestGBLinearModelValidation:
    def test_train_allows_nan_features(self) -> None:
        x, y = make_regression_data()
        x = x.copy()
        x[0, 0] = np.nan

        model = GBLinearModel.train(Dataset(x, y), config=GBLinearConfig(n_estimators=5))
        preds = model.predict(Dataset(x))

        assert preds.shape == (len(y), 1)
        assert np.isfinite(preds).all()

    def test_train_raises_on_nan_targets(self) -> None:
        x, y = make_regression_data()
        y = y.copy()
        y[0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            GBLinearModel.train(Dataset(x, y), config=GBLinearConfig(n_estimators=5))


class TestGBLinearModelPersistence:
    """Tests for GBLinearModel serialization round-trip."""

    def test_binary_roundtrip_preserves_predictions(self) -> None:
        """to_bytes → from_bytes preserves predictions exactly."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)

        model = GBLinearModel.train(Dataset(X, y), config=GBLinearConfig(n_estimators=50))

        original_preds = model.predict(Dataset(X))

        data = model.to_bytes()
        loaded = GBLinearModel.from_bytes(data)
        loaded_preds = loaded.predict(Dataset(X))

        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_json_roundtrip_preserves_predictions(self) -> None:
        """to_json_bytes → from_json_bytes preserves predictions."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)

        model = GBLinearModel.train(Dataset(X, y), config=GBLinearConfig(n_estimators=50))

        original_preds = model.predict(Dataset(X))

        data = model.to_json_bytes()
        loaded = GBLinearModel.from_json_bytes(data)
        loaded_preds = loaded.predict(Dataset(X))

        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_serialized_model_preserves_weights(self) -> None:
        """Loaded model has same weights."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)

        model = GBLinearModel.train(Dataset(X, y), config=GBLinearConfig(n_estimators=50))

        loaded = GBLinearModel.from_bytes(model.to_bytes())

        np.testing.assert_allclose(loaded.coef_, model.coef_, rtol=1e-6)
        np.testing.assert_allclose(loaded.intercept_, model.intercept_, rtol=1e-6)


class TestPolymorphicLoading:
    """Tests for boosters.Model polymorphic loading and inspection."""

    def test_loads_gbdt_binary(self) -> None:
        """Model.load_from_bytes returns GBDTModel from binary GBDT data."""
        import boosters

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)

        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=3))

        data = model.to_bytes()
        loaded = boosters.Model.load_from_bytes(data)

        assert isinstance(loaded, GBDTModel)
        assert loaded.n_trees == 3

    def test_loads_gbdt_json(self) -> None:
        """Model.load_from_bytes returns GBDTModel from JSON GBDT data."""
        import boosters

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)

        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=3))

        data = model.to_json_bytes()
        loaded = boosters.Model.load_from_bytes(data)

        assert isinstance(loaded, GBDTModel)
        assert loaded.n_trees == 3

    def test_loads_gblinear(self) -> None:
        """Model.load_from_bytes returns GBLinearModel from GBLinear data."""
        import boosters

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)

        model = GBLinearModel.train(Dataset(X, y), config=GBLinearConfig(n_estimators=5))

        data = model.to_bytes()
        loaded = boosters.Model.load_from_bytes(data)

        assert isinstance(loaded, GBLinearModel)
        assert loaded.n_features_in_ == 5

    def test_loads_preserves_predictions(self) -> None:
        """Model.load_from_bytes produces a model with identical predictions."""
        import boosters

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)
        ds = Dataset(X, y)

        model = GBDTModel.train(ds, config=GBDTConfig(n_estimators=5))
        original_preds = model.predict(ds)

        loaded = boosters.Model.load_from_bytes(model.to_bytes())
        loaded_preds = loaded.predict(ds)

        np.testing.assert_allclose(loaded_preds, original_preds, rtol=1e-6)

    def test_inspect_binary_gbdt(self) -> None:
        """Model.inspect_bytes returns correct ModelInfo for binary GBDT."""
        import boosters

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)

        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=3))

        data = model.to_bytes()
        info = boosters.Model.inspect_bytes(data)

        assert info.schema_version == 2
        assert info.model_type == "gbdt"
        assert info.format == "binary"
        assert isinstance(info.is_compressed, bool)

    def test_inspect_json_gbdt(self) -> None:
        """Model.inspect_bytes returns correct ModelInfo for JSON GBDT."""
        import boosters

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)

        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=3))

        data = model.to_json_bytes()
        info = boosters.Model.inspect_bytes(data)

        assert info.schema_version == 2
        assert info.model_type == "gbdt"
        assert info.format == "json"
        assert info.is_compressed is False

    def test_inspect_gblinear(self) -> None:
        """Model.inspect_bytes returns correct ModelInfo for GBLinear."""
        import boosters

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        y = rng.standard_normal(100).astype(np.float32)

        model = GBLinearModel.train(Dataset(X, y), config=GBLinearConfig(n_estimators=5))

        info = boosters.Model.inspect_bytes(model.to_bytes())

        assert info.model_type == "gblinear"
        assert info.format == "binary"

    def test_model_info_repr(self) -> None:
        """ModelInfo has readable repr."""
        import boosters

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3)).astype(np.float32)
        y = rng.standard_normal(50).astype(np.float32)

        model = GBDTModel.train(Dataset(X, y), config=GBDTConfig(n_estimators=2))

        info = boosters.Model.inspect_bytes(model.to_bytes())
        repr_str = repr(info)

        assert "ModelInfo" in repr_str
        assert "schema_version=" in repr_str
        assert "model_type=" in repr_str


class TestReadError:
    """Tests for boosters.ReadError exception handling."""

    def test_read_error_exists(self) -> None:
        """ReadError is exported from boosters package."""
        import boosters

        assert hasattr(boosters, "ReadError")
        assert issubclass(boosters.ReadError, ValueError)

    def test_loads_invalid_data_raises_read_error(self) -> None:
        """Model.load_from_bytes raises ReadError for invalid data."""
        import boosters

        with pytest.raises(boosters.ReadError):
            boosters.Model.load_from_bytes(b"not valid model data")

    def test_loads_truncated_data_raises_read_error(self) -> None:
        """Model.load_from_bytes raises ReadError for truncated binary data."""
        import boosters

        # BSTR magic followed by incomplete header
        with pytest.raises(boosters.ReadError):
            boosters.Model.load_from_bytes(b"BSTR\x00\x00")

    def test_inspect_invalid_data_raises_read_error(self) -> None:
        """Model.inspect_bytes raises ReadError for invalid data."""
        import boosters

        with pytest.raises(boosters.ReadError):
            boosters.Model.inspect_bytes(b"garbage")

    def test_from_bytes_invalid_raises_read_error(self) -> None:
        """GBDTModel.from_bytes raises ReadError for invalid data."""
        import boosters

        with pytest.raises(boosters.ReadError):
            GBDTModel.from_bytes(b"invalid")

    def test_from_json_bytes_invalid_raises_read_error(self) -> None:
        """GBDTModel.from_json_bytes raises ReadError for invalid JSON."""
        import boosters

        with pytest.raises(boosters.ReadError):
            GBDTModel.from_json_bytes(b"not json")

    def test_read_error_caught_as_value_error(self) -> None:
        """ReadError can be caught as ValueError."""
        import boosters

        with pytest.raises(ValueError, match=r".+"):
            boosters.Model.load_from_bytes(b"invalid")

    def test_read_error_has_message(self) -> None:
        """ReadError contains a descriptive message."""
        import boosters

        with pytest.raises(boosters.ReadError, match=r".+") as excinfo:
            boosters.Model.load_from_bytes(b"invalid model data")

        assert str(excinfo.value)
