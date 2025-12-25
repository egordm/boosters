#!/usr/bin/env python
"""Core API example with full control.

This example demonstrates the core API with nested config objects
for maximum control over model parameters.

Usage:
    python examples/02_core_api.py
"""

import numpy as np

import boosters as bst


def main() -> None:
    """Run core API examples."""
    print("=" * 60)
    print("Boosters Core API Example")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1

    # Create config with nested structure
    print("\n--- Creating Config ---")
    config = bst.GBDTConfig(
        n_estimators=100,
        learning_rate=0.1,
        objective=bst.SquaredLoss(),
        metric=bst.Rmse(),
        tree=bst.TreeConfig(max_depth=5, n_leaves=31),
        regularization=bst.RegularizationConfig(l2=1.0),
    )
    print(f"Config: n_estimators={config.n_estimators}, learning_rate={config.learning_rate}")
    print(f"Tree: max_depth={config.tree.max_depth}, n_leaves={config.tree.n_leaves}")

    # Create model and train
    print("\n--- Training Model ---")
    model = bst.GBDTModel(config=config)
    train_data = bst.Dataset(X, y)
    model.fit(train_data)
    print("Model trained successfully!")

    # Predict
    print("\n--- Predictions ---")
    predictions = model.predict(X)
    rmse = np.sqrt(np.mean((predictions - y) ** 2))
    print(f"Training RMSE: {rmse:.4f}")
    print(f"Predictions shape: {predictions.shape}")

    # Feature importance
    print("\n--- Feature Importance ---")
    importance = model.feature_importance()
    for i, imp in enumerate(importance):
        print(f"  Feature {i}: {imp:.4f}")

    print("\n✓ Core API example completed successfully!")


if __name__ == "__main__":
    main()
