"""Core API example with full control.

This example demonstrates the core API with flat config parameters
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
    rng = np.random.default_rng(42)
    features = rng.standard_normal((100, 5)).astype(np.float32)
    y = features[:, 0] + rng.standard_normal(100).astype(np.float32) * 0.1

    # Create config with flat structure
    print("\n--- Creating Config ---")
    config = bst.GBDTConfig(
        n_estimators=100,
        learning_rate=0.1,
        objective=bst.Objective.squared(),
        metric=bst.Metric.rmse(),
        max_depth=5,
        n_leaves=31,
        l2=1.0,
    )
    print(f"Config: n_estimators={config.n_estimators}, learning_rate={config.learning_rate}")
    print(f"Tree: max_depth={config.max_depth}, n_leaves={config.n_leaves}")

    # Create model and train
    print("\n--- Training Model ---")
    train_data = bst.Dataset(features, y)
    model = bst.GBDTModel.train(train_data, config=config)
    print("Model trained successfully!")

    # Predict
    print("\n--- Predictions ---")
    predictions = model.predict(bst.Dataset(features))
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
