"""Shared pytest fixtures for boosters tests.

This module provides reusable fixtures for testing:
- Synthetic datasets (regression, classification)
- Pre-trained models for prediction tests
- NumPy arrays in various formats
"""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def small_regression_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Small regression dataset (100 samples, 10 features).

    Returns:
        Tuple of (X, y) where X is (100, 10) and y is (100,).
    """
    n_samples, n_features = 100, 10
    # Using X as variable name is sklearn convention for feature matrices
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # Simple linear combination plus noise
    weights = rng.standard_normal(n_features).astype(np.float32)
    y = X @ weights + rng.standard_normal(n_samples).astype(np.float32) * 0.1
    return X, y


@pytest.fixture
def small_binary_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Small binary classification dataset (100 samples, 10 features).

    Returns:
        Tuple of (X, y) where X is (100, 10) and y is (100,) with values in {0, 1}.
    """
    n_samples, n_features = 100, 10
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    # Labels based on first feature
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y


@pytest.fixture
def medium_regression_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Medium regression dataset (1000 samples, 50 features).

    Returns:
        Tuple of (X, y) where X is (1000, 50) and y is (1000,).
    """
    n_samples, n_features = 1000, 50
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    weights = rng.standard_normal(n_features).astype(np.float32)
    y = X @ weights + rng.standard_normal(n_samples).astype(np.float32) * 0.1
    return X, y
