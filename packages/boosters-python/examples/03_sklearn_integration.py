#!/usr/bin/env python
"""sklearn integration example with Pipeline and GridSearchCV.

This example demonstrates how boosters works seamlessly with sklearn tools
like Pipeline, cross_val_score, and GridSearchCV.

Usage:
    python examples/03_sklearn_integration.py
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boosters.sklearn import GBDTRegressor


def main() -> None:
    """Run sklearn integration examples."""
    print("=" * 60)
    print("Boosters sklearn Integration")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(200, 5).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(200).astype(np.float32) * 0.1

    # Cross-validation
    print("\n--- Cross-Validation ---")
    reg = GBDTRegressor(n_estimators=50, verbose=0)
    scores = cross_val_score(reg, X, y, cv=5, scoring="neg_root_mean_squared_error")
    print(f"CV RMSE: {-scores.mean():.4f} (+/- {scores.std():.4f})")

    # Pipeline with preprocessing
    print("\n--- Pipeline ---")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GBDTRegressor(n_estimators=50, verbose=0)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    rmse = np.sqrt(np.mean((preds - y) ** 2))
    print(f"Pipeline training RMSE: {rmse:.4f}")

    # Grid search (smaller grid for speed)
    print("\n--- Grid Search ---")
    param_grid = {"n_estimators": [25, 50], "max_depth": [3, 5]}
    grid = GridSearchCV(
        GBDTRegressor(verbose=0),
        param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
    )
    grid.fit(X, y)
    print(f"Best params: {grid.best_params_}")
    print(f"Best score: {-grid.best_score_:.4f} RMSE")

    print("\n✓ sklearn integration examples completed successfully!")


if __name__ == "__main__":
    main()
