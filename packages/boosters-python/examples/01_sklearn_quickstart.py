#!/usr/bin/env python
"""Sklearn API quickstart example.

This example demonstrates the sklearn-compatible estimators for regression
and classification.

Usage:
    python examples/01_sklearn_quickstart.py
"""

import numpy as np

from boosters.sklearn import GBDTClassifier, GBDTRegressor


def main() -> None:
    """Run sklearn quickstart examples."""
    print("=" * 60)
    print("Boosters sklearn API Quickstart")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y_reg = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1
    y_cls = (X[:, 0] > 0).astype(np.int32)

    # Regression
    print("\n--- Regression ---")
    reg = GBDTRegressor(max_depth=5, n_estimators=100, verbose=0)
    reg.fit(X, y_reg)
    preds = reg.predict(X)
    rmse = np.sqrt(np.mean((preds - y_reg) ** 2))
    print(f"Training RMSE: {rmse:.4f}")

    # Classification
    print("\n--- Binary Classification ---")
    clf = GBDTClassifier(n_estimators=50, verbose=0)
    clf.fit(X, y_cls)
    proba = clf.predict_proba(X)
    accuracy = np.mean(clf.predict(X) == y_cls)
    print(f"Training Accuracy: {accuracy:.2%}")
    print(f"Probability shape: {proba.shape}")

    print("\n✓ Examples completed successfully!")


if __name__ == "__main__":
    main()
