"""Test pickle serialization for boosters models."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from boosters import Dataset, GBDTConfig, GBDTModel, GBLinearConfig, GBLinearModel, Metric, Objective


@pytest.fixture
def simple_dataset() -> Dataset:
    """Create a simple synthetic dataset for testing."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5), dtype=np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    return Dataset(X, y)


def test_gbdt_pickle_roundtrip(simple_dataset: Dataset) -> None:
    """Test that GBDTModel can be pickled and unpickled."""
    # Train a simple model
    config = GBDTConfig(
        n_estimators=10,
        max_depth=3,
        learning_rate=0.1,
        objective=Objective.squared(),
        metric=Metric.rmse(),
    )
    model = GBDTModel.train(simple_dataset, config=config)

    # Get predictions before pickling
    predictions_before = model.predict(simple_dataset)

    # Pickle and unpickle
    pickled = pickle.dumps(model)
    restored = pickle.loads(pickled)  # noqa: S301 - Testing pickle serialization

    # Verify predictions match
    predictions_after = restored.predict(simple_dataset)
    np.testing.assert_array_almost_equal(predictions_before, predictions_after, decimal=6)

    # Verify model attributes
    assert restored.n_trees == model.n_trees
    assert restored.n_features == model.n_features


def test_gblinear_pickle_roundtrip(simple_dataset: Dataset) -> None:
    """Test that GBLinearModel can be pickled and unpickled."""
    # Train a simple model
    config = GBLinearConfig(
        n_estimators=20,
        learning_rate=0.3,
        objective=Objective.squared(),
        metric=Metric.rmse(),
    )
    model = GBLinearModel.train(simple_dataset, config=config)

    # Get predictions and coefficients before pickling
    predictions_before = model.predict(simple_dataset)
    coef_before = model.coef_.copy()
    intercept_before = model.intercept_.copy()

    # Pickle and unpickle
    pickled = pickle.dumps(model)
    restored = pickle.loads(pickled)  # noqa: S301 - Testing pickle serialization

    # Verify predictions match
    predictions_after = restored.predict(simple_dataset)
    np.testing.assert_array_almost_equal(predictions_before, predictions_after, decimal=6)

    # Verify coefficients match
    np.testing.assert_array_almost_equal(coef_before, restored.coef_, decimal=6)
    np.testing.assert_array_almost_equal(intercept_before, restored.intercept_, decimal=6)

    # Verify model attributes
    assert restored.n_features_in_ == model.n_features_in_


def test_pickle_protocol_versions(simple_dataset: Dataset) -> None:
    """Test that pickling works with different protocol versions."""
    config = GBDTConfig(n_estimators=5, max_depth=2)
    model = GBDTModel.train(simple_dataset, config=config)

    predictions_original = model.predict(simple_dataset)

    # Test with different pickle protocols
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        pickled = pickle.dumps(model, protocol=protocol)
        restored = pickle.loads(pickled)  # noqa: S301 - Testing pickle serialization
        predictions_restored = restored.predict(simple_dataset)
        np.testing.assert_array_almost_equal(predictions_original, predictions_restored, decimal=6)


def test_pickle_file_io(simple_dataset: Dataset, tmp_path: Path) -> None:
    """Test that models can be saved to and loaded from pickle files."""
    # Train model
    config = GBDTConfig(n_estimators=10, max_depth=3)
    model = GBDTModel.train(simple_dataset, config=config)
    predictions_before = model.predict(simple_dataset)

    # Save to file
    pickle_path = tmp_path / "model.pkl"
    pickle_path.write_bytes(pickle.dumps(model))

    # Load from file
    restored = pickle.loads(pickle_path.read_bytes())  # noqa: S301 - Testing pickle serialization

    # Verify
    predictions_after = restored.predict(simple_dataset)
    np.testing.assert_array_almost_equal(predictions_before, predictions_after, decimal=6)
