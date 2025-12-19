"""Tests for XGBoost and LightGBM file format conversion utilities.

These tests verify the file format converters that parse XGBoost JSON
and LightGBM text formats to produce .bstr files. These converters do NOT
require xgboost or lightgbm libraries to be installed - they use the
Rust parsers directly.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Test data paths (relative to boosters-python directory)
TEST_DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "test-cases"


class TestXGBoostJsonConverter:
    """Tests for xgboost_json_to_bstr conversion."""

    def test_regression_model_conversion(self):
        """Test converting XGBoost regression model JSON to .bstr."""
        from boosters import GBDTBooster, xgboost_json_to_bstr

        json_path = TEST_DATA_DIR / "xgboost/gbtree/inference/gbtree_regression.model.json"
        if not json_path.exists():
            pytest.skip(f"Test data not found: {json_path}")

        with tempfile.NamedTemporaryFile(suffix=".bstr", delete=False) as f:
            bstr_path = f.name

        try:
            # Convert
            xgboost_json_to_bstr(str(json_path), bstr_path)

            # Load and verify
            loaded = GBDTBooster.load(bstr_path)
            assert loaded.n_trees > 0
            assert loaded.n_features > 0

            # Should be able to make predictions
            X = np.random.randn(10, loaded.n_features).astype(np.float32)
            preds = loaded.predict(X)
            assert preds.shape == (10,)
            assert not np.any(np.isnan(preds))

        finally:
            Path(bstr_path).unlink(missing_ok=True)

    def test_multiclass_model_conversion(self):
        """Test converting XGBoost multiclass model JSON to .bstr."""
        from boosters import GBDTBooster, xgboost_json_to_bstr

        json_path = TEST_DATA_DIR / "xgboost/gbtree/inference/gbtree_multiclass.model.json"
        if not json_path.exists():
            pytest.skip(f"Test data not found: {json_path}")

        with tempfile.NamedTemporaryFile(suffix=".bstr", delete=False) as f:
            bstr_path = f.name

        try:
            xgboost_json_to_bstr(str(json_path), bstr_path)
            loaded = GBDTBooster.load(bstr_path)
            assert loaded.n_trees > 0

        finally:
            Path(bstr_path).unlink(missing_ok=True)

    def test_categorical_model_conversion(self):
        """Test converting XGBoost model with categorical features."""
        from boosters import GBDTBooster, xgboost_json_to_bstr

        json_path = TEST_DATA_DIR / "xgboost/gbtree/inference/gbtree_categorical.model.json"
        if not json_path.exists():
            pytest.skip(f"Test data not found: {json_path}")

        with tempfile.NamedTemporaryFile(suffix=".bstr", delete=False) as f:
            bstr_path = f.name

        try:
            xgboost_json_to_bstr(str(json_path), bstr_path)
            loaded = GBDTBooster.load(bstr_path)
            assert loaded.n_trees > 0

        finally:
            Path(bstr_path).unlink(missing_ok=True)

    def test_invalid_json_path(self):
        """Test error handling for invalid input path."""
        from boosters import xgboost_json_to_bstr

        with pytest.raises(Exception):  # OSError or ValueError from Rust
            xgboost_json_to_bstr("/nonexistent/path.json", "/tmp/out.bstr")

    def test_invalid_json_content(self):
        """Test error handling for invalid JSON content."""
        from boosters import xgboost_json_to_bstr

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json {{{")
            json_path = f.name

        try:
            with pytest.raises(Exception):  # ValueError from Rust parser
                xgboost_json_to_bstr(json_path, "/tmp/out.bstr")
        finally:
            Path(json_path).unlink(missing_ok=True)


class TestLightGBMTxtConverter:
    """Tests for lightgbm_txt_to_bstr conversion."""

    def test_regression_model_conversion(self):
        """Test converting LightGBM regression model txt to .bstr."""
        from boosters import GBDTBooster, lightgbm_txt_to_bstr

        txt_path = TEST_DATA_DIR / "lightgbm/inference/regression/model.txt"
        if not txt_path.exists():
            pytest.skip(f"Test data not found: {txt_path}")

        with tempfile.NamedTemporaryFile(suffix=".bstr", delete=False) as f:
            bstr_path = f.name

        try:
            # Convert
            lightgbm_txt_to_bstr(str(txt_path), bstr_path)

            # Load and verify
            loaded = GBDTBooster.load(bstr_path)
            assert loaded.n_trees > 0
            assert loaded.n_features > 0

            # Should be able to make predictions
            X = np.random.randn(10, loaded.n_features).astype(np.float32)
            preds = loaded.predict(X)
            assert preds.shape == (10,)
            assert not np.any(np.isnan(preds))

        finally:
            Path(bstr_path).unlink(missing_ok=True)

    def test_multiclass_model_conversion(self):
        """Test converting LightGBM multiclass model txt to .bstr."""
        from boosters import GBDTBooster, lightgbm_txt_to_bstr

        txt_path = TEST_DATA_DIR / "lightgbm/inference/multiclass/model.txt"
        if not txt_path.exists():
            pytest.skip(f"Test data not found: {txt_path}")

        with tempfile.NamedTemporaryFile(suffix=".bstr", delete=False) as f:
            bstr_path = f.name

        try:
            lightgbm_txt_to_bstr(str(txt_path), bstr_path)
            loaded = GBDTBooster.load(bstr_path)
            assert loaded.n_trees > 0

        finally:
            Path(bstr_path).unlink(missing_ok=True)

    def test_binary_classification_conversion(self):
        """Test converting LightGBM binary classification model."""
        from boosters import GBDTBooster, lightgbm_txt_to_bstr

        txt_path = TEST_DATA_DIR / "lightgbm/inference/binary_classification/model.txt"
        if not txt_path.exists():
            pytest.skip(f"Test data not found: {txt_path}")

        with tempfile.NamedTemporaryFile(suffix=".bstr", delete=False) as f:
            bstr_path = f.name

        try:
            lightgbm_txt_to_bstr(str(txt_path), bstr_path)
            loaded = GBDTBooster.load(bstr_path)
            assert loaded.n_trees > 0

        finally:
            Path(bstr_path).unlink(missing_ok=True)

    def test_invalid_txt_path(self):
        """Test error handling for invalid input path."""
        from boosters import lightgbm_txt_to_bstr

        with pytest.raises(Exception):  # OSError or ValueError from Rust
            lightgbm_txt_to_bstr("/nonexistent/path.txt", "/tmp/out.bstr")

    def test_invalid_txt_content(self):
        """Test error handling for invalid txt content."""
        from boosters import lightgbm_txt_to_bstr

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("not a valid lightgbm model file")
            txt_path = f.name

        try:
            with pytest.raises(Exception):  # ValueError from Rust parser
                lightgbm_txt_to_bstr(txt_path, "/tmp/out.bstr")
        finally:
            Path(txt_path).unlink(missing_ok=True)


class TestConverterInferenceMatch:
    """Tests that verify converted models produce correct predictions.

    These tests compare predictions from converted models against reference
    expected values stored in the test-cases directory.
    """

    def test_xgboost_regression_predictions(self):
        """Test that converted XGBoost model produces expected predictions."""
        import json

        from boosters import GBDTBooster, xgboost_json_to_bstr

        model_path = TEST_DATA_DIR / "xgboost/gbtree/inference/gbtree_regression.model.json"
        input_path = TEST_DATA_DIR / "xgboost/gbtree/inference/gbtree_regression.input.json"
        expected_path = TEST_DATA_DIR / "xgboost/gbtree/inference/gbtree_regression.expected.json"

        if not all(p.exists() for p in [model_path, input_path, expected_path]):
            pytest.skip("Test data not found")

        with tempfile.NamedTemporaryFile(suffix=".bstr", delete=False) as f:
            bstr_path = f.name

        try:
            # Convert model
            xgboost_json_to_bstr(str(model_path), bstr_path)
            loaded = GBDTBooster.load(bstr_path)

            # Load test inputs and expected outputs
            with open(input_path) as f:
                inputs = json.load(f)
            with open(expected_path) as f:
                expected = json.load(f)

            # Handle both dict-with-features and plain array formats
            if isinstance(inputs, dict) and "features" in inputs:
                X = np.array(inputs["features"], dtype=np.float32)
            else:
                X = np.array(inputs, dtype=np.float32)

            if isinstance(expected, dict) and "predictions" in expected:
                expected_preds = np.array(expected["predictions"], dtype=np.float32)
            else:
                expected_preds = np.array(expected, dtype=np.float32)

            # Predict and compare
            actual_preds = loaded.predict(X)
            np.testing.assert_allclose(actual_preds, expected_preds, rtol=1e-5, atol=1e-5)

        finally:
            Path(bstr_path).unlink(missing_ok=True)

    def test_lightgbm_regression_predictions(self):
        """Test that converted LightGBM model produces expected predictions."""
        import json

        from boosters import GBDTBooster, lightgbm_txt_to_bstr

        model_path = TEST_DATA_DIR / "lightgbm/inference/regression/model.txt"
        input_path = TEST_DATA_DIR / "lightgbm/inference/regression/input.json"
        expected_path = TEST_DATA_DIR / "lightgbm/inference/regression/expected.json"

        if not all(p.exists() for p in [model_path, input_path, expected_path]):
            pytest.skip("Test data not found")

        with tempfile.NamedTemporaryFile(suffix=".bstr", delete=False) as f:
            bstr_path = f.name

        try:
            # Convert model
            lightgbm_txt_to_bstr(str(model_path), bstr_path)
            loaded = GBDTBooster.load(bstr_path)

            # Load test inputs and expected outputs
            with open(input_path) as f:
                inputs = json.load(f)
            with open(expected_path) as f:
                expected = json.load(f)

            # Handle various input formats: dict with "features"/"data" or plain array
            if isinstance(inputs, dict):
                if "features" in inputs:
                    X = np.array(inputs["features"], dtype=np.float32)
                elif "data" in inputs:
                    X = np.array(inputs["data"], dtype=np.float32)
                else:
                    raise ValueError(f"Unknown input format: {list(inputs.keys())}")
            else:
                X = np.array(inputs, dtype=np.float32)

            # Handle various expected formats
            if isinstance(expected, dict):
                if "predictions" in expected:
                    expected_preds = np.array(expected["predictions"], dtype=np.float32)
                elif "raw" in expected:
                    expected_preds = np.array(expected["raw"], dtype=np.float32)
                elif "data" in expected:
                    expected_preds = np.array(expected["data"], dtype=np.float32)
                else:
                    raise ValueError(f"Unknown expected format: {list(expected.keys())}")
            else:
                expected_preds = np.array(expected, dtype=np.float32)

            # Predict and compare
            actual_preds = loaded.predict(X)
            np.testing.assert_allclose(actual_preds, expected_preds, rtol=1e-5, atol=1e-5)

        finally:
            Path(bstr_path).unlink(missing_ok=True)
