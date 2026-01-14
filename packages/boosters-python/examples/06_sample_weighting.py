"""Sample weighting example.

This example demonstrates how to use sample weights for:
1. Handling class imbalance in classification
2. Giving more importance to recent data
3. Downweighting outliers or noisy samples

Usage:
    python examples/06_sample_weighting.py
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from boosters import Dataset, GBDTConfig, GBDTModel, Objective
from boosters.sklearn import GBDTClassifier


def class_imbalance_example() -> None:
    """Handle class imbalance with sample weights."""
    print("\n--- Class Imbalance Example ---")

    # Create imbalanced dataset (95% class 0, 5% class 1)
    rng = np.random.default_rng(42)
    n_samples = 1000
    n_minority = 50

    X = rng.standard_normal((n_samples, 5)).astype(np.float32)

    # Create imbalanced labels
    y = np.zeros(n_samples, dtype=np.float32)
    y[:n_minority] = 1.0

    # Shuffle
    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"Class distribution - Test:  {np.bincount(y_test.astype(int))}")

    # Without weights - model biased toward majority class
    clf_no_weights = GBDTClassifier(n_estimators=50, verbose=0)
    clf_no_weights.fit(X_train, y_train)
    pred_no_weights = clf_no_weights.predict(X_test)

    print("\nWithout sample weights:")
    print(classification_report(y_test, pred_no_weights, zero_division=0))

    # With balanced weights
    weights = compute_sample_weight("balanced", y_train)

    clf_weighted = GBDTClassifier(n_estimators=50, verbose=0)
    clf_weighted.fit(X_train, y_train, sample_weight=weights)
    pred_weighted = clf_weighted.predict(X_test)

    print("With balanced sample weights:")
    print(classification_report(y_test, pred_weighted, zero_division=0))


def temporal_weighting_example() -> None:
    """Give more importance to recent data using sample weights."""
    print("\n--- Temporal Weighting Example (Recent Data More Important) ---")

    # Simulate temporal data where recent samples are more relevant
    rng = np.random.default_rng(42)
    n_samples = 500

    X = rng.standard_normal((n_samples, 3)).astype(np.float32)

    # Target has a concept drift - relationship changes over time
    # Early: y = X[:, 0]
    # Late: y = X[:, 1]
    time_factor = np.linspace(0, 1, n_samples)
    y = ((1 - time_factor) * X[:, 0] + time_factor * X[:, 1]).astype(np.float32)
    y += rng.standard_normal(n_samples).astype(np.float32) * 0.1

    # Split chronologically (not randomly!)
    train_size = 400
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Uniform weights
    from boosters.sklearn import GBDTRegressor

    reg_uniform = GBDTRegressor(n_estimators=50, verbose=0)
    reg_uniform.fit(X_train, y_train)
    pred_uniform = reg_uniform.predict(X_test)
    rmse_uniform = np.sqrt(np.mean((pred_uniform - y_test) ** 2))

    # Exponential decay weights (recent = higher weight)
    decay_rate = 3.0  # Higher = more emphasis on recent
    temporal_weights = np.exp(decay_rate * np.linspace(0, 1, train_size)).astype(np.float32)

    reg_weighted = GBDTRegressor(n_estimators=50, verbose=0)
    reg_weighted.fit(X_train, y_train, sample_weight=temporal_weights)
    pred_weighted = reg_weighted.predict(X_test)
    rmse_weighted = np.sqrt(np.mean((pred_weighted - y_test) ** 2))

    print(f"RMSE with uniform weights:   {rmse_uniform:.4f}")
    print(f"RMSE with temporal weights:  {rmse_weighted:.4f}")
    print(f"Improvement: {(rmse_uniform - rmse_weighted) / rmse_uniform * 100:.1f}%")


def core_api_example() -> None:
    """Use sample weights with the core API."""
    print("\n--- Core API with Sample Weights ---")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] * 0.5 + rng.standard_normal(200) * 0.1).astype(np.float32)

    # Create custom weights (e.g., downweight potential outliers)
    weights = np.ones(200, dtype=np.float32)
    weights[np.abs(y) > 2] = 0.1  # Downweight extreme values

    print(f"Samples with reduced weight: {np.sum(weights < 1)}")

    # Create dataset WITH weights
    train_data = Dataset(X, y, weights=weights)

    config = GBDTConfig(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        objective=Objective.squared(),
    )

    model = GBDTModel.train(train_data, config=config)
    predictions = model.predict(Dataset(X))

    rmse = np.sqrt(np.mean((predictions.flatten() - y) ** 2))
    print(f"Training RMSE: {rmse:.4f}")


def main() -> None:
    """Run all sample weighting examples."""
    print("=" * 60)
    print("Sample Weighting Examples")
    print("=" * 60)

    class_imbalance_example()
    temporal_weighting_example()
    core_api_example()

    print("\n✓ All examples completed successfully!")


if __name__ == "__main__":
    main()
