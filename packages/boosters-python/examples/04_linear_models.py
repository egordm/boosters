"""Linear boosting models example.

This example demonstrates GBLinear models for both regression
and classification using the sklearn-compatible API.

Usage:
    python examples/04_linear_models.py
"""

import numpy as np

from boosters.sklearn import GBLinearClassifier, GBLinearRegressor


def main() -> None:
    """Run linear model examples."""
    print("=" * 60)
    print("Boosters Linear Models")
    print("=" * 60)

    # Generate sample data (linear relationship)
    rng = np.random.default_rng(42)
    features = rng.standard_normal((200, 5)).astype(np.float32)
    # Linear combination with noise
    noise = rng.standard_normal(200).astype(np.float32) * 0.1
    y_reg = 2.0 * features[:, 0] - 1.5 * features[:, 1] + 0.5 * features[:, 2] + noise
    y_cls = (y_reg > 0).astype(np.int32)

    # Linear Regression
    print("\n--- GBLinear Regression ---")
    reg = GBLinearRegressor(n_estimators=100, learning_rate=0.3)
    reg.fit(features, y_reg)
    preds = reg.predict(features)
    rmse = np.sqrt(np.mean((preds - y_reg) ** 2))
    print(f"Training RMSE: {rmse:.4f}")

    # Access coefficients
    print("\n--- Coefficients ---")
    coef = reg.coef_
    intercept = reg.intercept_
    print(f"Intercept: {intercept[0]:.4f}")
    print("Coefficients:")
    for i, c in enumerate(coef):
        print(f"  Feature {i}: {c:.4f}")

    # Linear Classification
    print("\n--- GBLinear Classification ---")
    clf = GBLinearClassifier(n_estimators=100, learning_rate=0.3)
    clf.fit(features, y_cls)
    accuracy = np.mean(clf.predict(features) == y_cls)
    print(f"Training Accuracy: {accuracy:.2%}")

    print("\n✓ Linear model examples completed successfully!")


if __name__ == "__main__":
    main()
