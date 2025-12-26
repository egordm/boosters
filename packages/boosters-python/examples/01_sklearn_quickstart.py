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
    rng = np.random.default_rng(42)
    features = rng.standard_normal((100, 5)).astype(np.float32)
    y_reg = features[:, 0] + rng.standard_normal(100).astype(np.float32) * 0.1
    y_cls = (features[:, 0] > 0).astype(np.int32)

    # Regression
    print("\n--- Regression ---")
    reg = GBDTRegressor(max_depth=5, n_estimators=100, verbose=0)
    reg.fit(features, y_reg)
    preds = reg.predict(features)
    rmse = np.sqrt(np.mean((preds - y_reg) ** 2))
    print(f"Training RMSE: {rmse:.4f}")

    # Classification
    print("\n--- Binary Classification ---")
    clf = GBDTClassifier(n_estimators=50, verbose=0)
    clf.fit(features, y_cls)
    proba = clf.predict_proba(features)
    accuracy = np.mean(clf.predict(features) == y_cls)
    print(f"Training Accuracy: {accuracy:.2%}")
    print(f"Probability shape: {proba.shape}")

    print("\n✓ Examples completed successfully!")


if __name__ == "__main__":
    main()
