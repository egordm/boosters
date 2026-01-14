"""Tests for XGBoost and LightGBM conversion utilities."""

# Allow uppercase variable names for X, y (ML convention)

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from boosters.convert import (
    lightgbm_to_json_bytes,
    lightgbm_to_schema,
    xgboost_to_json_bytes,
    xgboost_to_schema,
)
from boosters.persist.schema import GBDTModelSchema, GBLinearModelSchema, JsonEnvelope

# Test case directories
TEST_CASES_DIR = Path(__file__).parents[4] / "crates" / "boosters" / "tests" / "test-cases"
XGBOOST_GBTREE_DIR = TEST_CASES_DIR / "xgboost" / "gbtree" / "inference"
XGBOOST_GBLINEAR_DIR = TEST_CASES_DIR / "xgboost" / "gblinear" / "inference"
XGBOOST_DART_DIR = TEST_CASES_DIR / "xgboost" / "dart" / "inference"
LIGHTGBM_DIR = TEST_CASES_DIR / "lightgbm" / "inference"


class TestXGBoostConverter:
    """Test XGBoost model conversion."""

    def test_gbtree_regression_conversion(self) -> None:
        """Convert gbtree regression model to schema."""
        model_path = XGBOOST_GBTREE_DIR / "gbtree_regression.model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = xgboost_to_schema(model_path)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.output_transform == "identity"
        assert schema.meta.num_features == 5
        assert len(schema.forest.trees) > 0

    def test_gbtree_binary_conversion(self) -> None:
        """Convert gbtree binary classification model to schema."""
        model_path = XGBOOST_GBTREE_DIR / "gbtree_binary_logistic.model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = xgboost_to_schema(model_path)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.output_transform == "sigmoid"
        assert schema.forest.n_groups == 1

    def test_gbtree_multiclass_conversion(self) -> None:
        """Convert gbtree multiclass model to schema."""
        model_path = XGBOOST_GBTREE_DIR / "gbtree_multiclass.model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = xgboost_to_schema(model_path)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.output_transform == "softmax"
        assert schema.forest.n_groups == 3

    def test_gblinear_regression_conversion(self) -> None:
        """Convert gblinear regression model to schema."""
        model_path = XGBOOST_GBLINEAR_DIR / "gblinear_regression.model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = xgboost_to_schema(model_path)

        assert isinstance(schema, GBLinearModelSchema)
        assert schema.output_transform == "identity"
        assert schema.meta.num_features == 5
        assert schema.weights.num_groups == 1

    def test_gblinear_binary_conversion(self) -> None:
        """Convert gblinear binary classification model to schema."""
        model_path = XGBOOST_GBLINEAR_DIR / "gblinear_binary.model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = xgboost_to_schema(model_path)

        assert isinstance(schema, GBLinearModelSchema)
        assert schema.output_transform == "sigmoid"

    def test_dart_regression_conversion(self) -> None:
        """Convert DART regression model to schema."""
        model_path = XGBOOST_DART_DIR / "dart_regression.model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = xgboost_to_schema(model_path)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.output_transform == "identity"

    def test_xgboost_to_json_bytes(self) -> None:
        """Convert XGBoost model to JSON bytes."""
        model_path = XGBOOST_GBTREE_DIR / "gbtree_regression.model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        json_bytes = xgboost_to_json_bytes(model_path)

        assert isinstance(json_bytes, bytes)
        assert b'"bstr_version":' in json_bytes
        assert b'"model_type": "gbdt"' in json_bytes

    def test_xgboost_roundtrip_parse(self) -> None:
        """Converted JSON parses back to schema."""
        model_path = XGBOOST_GBTREE_DIR / "gbtree_regression.model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        json_bytes = xgboost_to_json_bytes(model_path)
        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_bytes)

        assert envelope.bstr_version == 2
        assert envelope.model_type == "gbdt"
        assert envelope.model.output_transform == "identity"


class TestXGBoostBoosterInput:
    """Test conversion from xgb.Booster objects."""

    @pytest.fixture
    def xgb_booster(self) -> Any:
        """Create an XGBoost Booster for testing."""
        pytest.importorskip("xgboost")
        import xgboost as xgb
        from sklearn.datasets import make_regression

        X, y, *_ = make_regression(n_samples=100, n_features=5, random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        dtrain = xgb.DMatrix(X, label=y)
        params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.1}
        return xgb.train(params, dtrain, num_boost_round=5)

    def test_booster_to_schema(self, xgb_booster: Any) -> None:
        """Convert Booster object to schema."""
        schema = xgboost_to_schema(xgb_booster)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.meta.num_features == 5
        assert len(schema.forest.trees) == 5

    def test_booster_to_json_bytes(self, xgb_booster: Any) -> None:
        """Convert Booster object to JSON bytes."""
        json_bytes = xgboost_to_json_bytes(xgb_booster)

        assert isinstance(json_bytes, bytes)
        assert b'"bstr_version":' in json_bytes


class TestLightGBMConverter:
    """Test LightGBM model conversion."""

    def test_regression_conversion(self) -> None:
        """Convert LightGBM regression model to schema."""
        model_dir = LIGHTGBM_DIR / "regression"
        model_path = model_dir / "model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = lightgbm_to_schema(model_path)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.output_transform == "identity"
        assert len(schema.forest.trees) > 0

    def test_binary_conversion(self) -> None:
        """Convert LightGBM binary classification model to schema."""
        model_dir = LIGHTGBM_DIR / "binary"
        model_path = model_dir / "model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = lightgbm_to_schema(model_path)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.output_transform == "sigmoid"

    def test_multiclass_conversion(self) -> None:
        """Convert LightGBM multiclass model to schema."""
        model_dir = LIGHTGBM_DIR / "multiclass"
        model_path = model_dir / "model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        schema = lightgbm_to_schema(model_path)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.output_transform == "softmax"
        assert schema.forest.n_groups == 3

    def test_lightgbm_to_json_bytes(self) -> None:
        """Convert LightGBM model to JSON bytes."""
        model_dir = LIGHTGBM_DIR / "regression"
        model_path = model_dir / "model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        json_bytes = lightgbm_to_json_bytes(model_path)

        assert isinstance(json_bytes, bytes)
        assert b'"bstr_version":' in json_bytes
        assert b'"model_type": "gbdt"' in json_bytes

    def test_lightgbm_roundtrip_parse(self) -> None:
        """Converted JSON parses back to schema."""
        model_dir = LIGHTGBM_DIR / "regression"
        model_path = model_dir / "model.json"
        if not model_path.exists():
            pytest.skip("Test case not found")

        json_bytes = lightgbm_to_json_bytes(model_path)
        envelope = JsonEnvelope[GBDTModelSchema].model_validate_json(json_bytes)

        assert envelope.bstr_version == 2
        assert envelope.model_type == "gbdt"


class TestLightGBMBoosterInput:
    """Test conversion from lgb.Booster objects."""

    @pytest.fixture
    def lgb_booster(self) -> Any:
        """Create a LightGBM Booster for testing."""
        pytest.importorskip("lightgbm")
        import lightgbm as lgb
        from sklearn.datasets import make_regression

        X, y, *_ = make_regression(n_samples=100, n_features=10, random_state=42)
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        params = {
            "objective": "regression",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "verbose": -1,
        }
        train_data = lgb.Dataset(X, label=y, params={"verbose": -1})
        return lgb.train(params, train_data, num_boost_round=5)

    def test_booster_to_schema(self, lgb_booster: Any) -> None:
        """Convert Booster object to schema."""
        schema = lightgbm_to_schema(lgb_booster)

        assert isinstance(schema, GBDTModelSchema)
        assert schema.meta.num_features == 10
        assert len(schema.forest.trees) == 5

    def test_booster_to_json_bytes(self, lgb_booster: Any) -> None:
        """Convert Booster object to JSON bytes."""
        json_bytes = lightgbm_to_json_bytes(lgb_booster)

        assert isinstance(json_bytes, bytes)
        assert b'"bstr_version":' in json_bytes


class TestPredictionCompatibility:
    """Test that converted models produce correct predictions."""

    def test_xgboost_conversion_predictions(self) -> None:
        """Converted XGBoost model predictions match original."""
        pytest.importorskip("xgboost")
        import xgboost as xgb

        # Create and train XGBoost model
        from sklearn.datasets import make_regression

        import boosters as bst

        X, y, *_ = make_regression(n_samples=100, n_features=5, random_state=42)
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        dtrain = xgb.DMatrix(X, label=y)
        params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.1}
        xgb_model = xgb.train(params, dtrain, num_boost_round=10)

        # Get XGBoost predictions
        xgb_preds = xgb_model.predict(xgb.DMatrix(X[:10]))

        # Convert to boosters format and load
        json_bytes = xgboost_to_json_bytes(xgb_model)
        model = bst.Model.load_from_bytes(json_bytes)

        # Get boosters predictions
        bst_preds = model.predict_raw(bst.Dataset(X[:10]))

        # Compare predictions
        np.testing.assert_allclose(bst_preds.flatten(), xgb_preds, rtol=1e-5, atol=1e-5)

    def test_lightgbm_conversion_predictions(self) -> None:
        """Converted LightGBM model predictions match original."""
        pytest.importorskip("lightgbm")
        import lightgbm as lgb

        # Create and train LightGBM model
        from sklearn.datasets import make_regression

        import boosters as bst

        X, y, *_ = make_regression(n_samples=100, n_features=10, random_state=42)
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        params = {
            "objective": "regression",
            "num_leaves": 31,
            "learning_rate": 0.1,
            "verbose": -1,
        }
        train_data = lgb.Dataset(X, label=y, params={"verbose": -1})
        lgb_model = lgb.train(params, train_data, num_boost_round=10)

        # Get LightGBM predictions
        lgb_preds = np.asarray(lgb_model.predict(X[:10], raw_score=True))

        # Convert to boosters format and load
        json_bytes = lightgbm_to_json_bytes(lgb_model)
        model = bst.Model.load_from_bytes(json_bytes)

        # Get boosters predictions
        X_f32 = X[:10].astype(np.float32)
        bst_preds = model.predict_raw(bst.Dataset(X_f32))

        # Compare predictions
        np.testing.assert_allclose(bst_preds.flatten(), lgb_preds, rtol=1e-5, atol=1e-5)
