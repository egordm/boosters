"""Test serialization: pickle, save/load, to_bytes/from_bytes."""

import numpy as np
import pickle
import tempfile
import os
import pytest

from boosters import GBDTBooster, GBLinearBooster


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(100) * 0.1).astype(np.float32)
    return X, y


@pytest.fixture
def trained_gbdt_model(sample_data):
    """Create and train a GBDTBooster."""
    X, y = sample_data
    model = GBDTBooster(n_estimators=10, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    return model, X


@pytest.fixture
def trained_gblinear_model(sample_data):
    """Create and train a GBLinearBooster."""
    X, y = sample_data
    model = GBLinearBooster(n_estimators=10, learning_rate=0.5)
    model.fit(X, y)
    return model, X


class TestGBDTSerialization:
    """Test GBDTBooster serialization methods."""

    def test_pickle_roundtrip(self, trained_gbdt_model):
        """Test that pickle preserves model predictions."""
        model, X = trained_gbdt_model
        original_preds = model.predict(X)
        
        # Pickle and unpickle
        pickled = pickle.dumps(model)
        loaded = pickle.loads(pickled)
        loaded_preds = loaded.predict(X)
        
        assert np.allclose(original_preds, loaded_preds), "Pickle predictions mismatch"

    def test_save_load_roundtrip(self, trained_gbdt_model):
        """Test that save/load preserves model predictions."""
        model, X = trained_gbdt_model
        original_preds = model.predict(X)
        
        with tempfile.NamedTemporaryFile(suffix='.bstr', delete=False) as f:
            path = f.name
        
        try:
            model.save(path)
            loaded = GBDTBooster.load(path)
            loaded_preds = loaded.predict(X)
            assert np.allclose(original_preds, loaded_preds), "Save/Load predictions mismatch"
        finally:
            os.unlink(path)

    def test_to_bytes_from_bytes_roundtrip(self, trained_gbdt_model):
        """Test that to_bytes/from_bytes preserves model predictions."""
        model, X = trained_gbdt_model
        original_preds = model.predict(X)
        
        model_bytes = model.to_bytes()
        loaded = GBDTBooster.from_bytes(model_bytes)
        loaded_preds = loaded.predict(X)
        
        assert np.allclose(original_preds, loaded_preds), "Bytes predictions mismatch"


class TestGBLinearSerialization:
    """Test GBLinearBooster serialization methods."""

    def test_pickle_roundtrip(self, trained_gblinear_model):
        """Test that pickle preserves model predictions."""
        model, X = trained_gblinear_model
        original_preds = model.predict(X)
        
        pickled = pickle.dumps(model)
        loaded = pickle.loads(pickled)
        loaded_preds = loaded.predict(X)
        
        assert np.allclose(original_preds, loaded_preds), "Pickle predictions mismatch"

    def test_save_load_roundtrip(self, trained_gblinear_model):
        """Test that save/load preserves model predictions."""
        model, X = trained_gblinear_model
        original_preds = model.predict(X)
        
        with tempfile.NamedTemporaryFile(suffix='.bstr', delete=False) as f:
            path = f.name
        
        try:
            model.save(path)
            loaded = GBLinearBooster.load(path)
            loaded_preds = loaded.predict(X)
            assert np.allclose(original_preds, loaded_preds), "Save/Load predictions mismatch"
        finally:
            os.unlink(path)

    def test_to_bytes_from_bytes_roundtrip(self, trained_gblinear_model):
        """Test that to_bytes/from_bytes preserves model predictions."""
        model, X = trained_gblinear_model
        original_preds = model.predict(X)
        
        model_bytes = model.to_bytes()
        loaded = GBLinearBooster.from_bytes(model_bytes)
        loaded_preds = loaded.predict(X)
        
        assert np.allclose(original_preds, loaded_preds), "Bytes predictions mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
