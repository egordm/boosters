"""Generate XGBoost test cases with models, input data, and expected predictions.

Each test case generates:
- {name}.model.json      - The XGBoost model
- {name}.input.json      - Test input features  
- {name}.expected.json   - Expected predictions (raw margin scores, no transform)

This allows Rust tests to load a model, predict on inputs, and compare to expected output.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.datasets import make_regression, make_classification
import xgboost as xgb

# Output to tests/test-cases/xgboost/
OUT_DIR = Path(__file__).parents[3] / "tests" / "test-cases" / "xgboost"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def nan_to_null(obj):
    """Convert NaN values to None for JSON serialization."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, list):
        return [nan_to_null(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: nan_to_null(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return nan_to_null(obj.tolist())
    return obj


def save_test_case(name: str, booster: xgb.Booster, X_test: np.ndarray, output_margin: bool = True):
    """Save a complete test case: model, inputs, and expected predictions."""
    # Save model
    model_path = OUT_DIR / f"{name}.model.json"
    booster.save_model(str(model_path))
    
    # Save test inputs (NaN becomes null in JSON)
    input_path = OUT_DIR / f"{name}.input.json"
    with open(input_path, "w") as f:
        json.dump({
            "features": nan_to_null(X_test),
            "num_rows": len(X_test),
            "num_features": X_test.shape[1],
        }, f, indent=2)
    
    # Get predictions (raw margin scores for tree sum, no sigmoid/softmax)
    dtest = xgb.DMatrix(X_test)
    predictions = booster.predict(dtest, output_margin=output_margin)
    
    expected_path = OUT_DIR / f"{name}.expected.json"
    with open(expected_path, "w") as f:
        json.dump({
            "predictions": predictions.tolist(),
            "output_margin": output_margin,
        }, f, indent=2)
    
    print(f"✓ {name}: model + {len(X_test)} test cases")


def generate_regression():
    """Simple regression: 5 features, 5 trees."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=42)
    X_test = X[:10]  # Use first 10 samples as test cases
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_test_case("regression", booster, X_test)


def generate_binary_logistic():
    """Binary classification with logistic objective."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, random_state=42
    )
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 2,
        'eta': 0.15,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    # output_margin=True gives raw scores (before sigmoid)
    save_test_case("binary_logistic", booster, X_test, output_margin=True)


def generate_multiclass():
    """Multiclass classification (3 classes)."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=1, random_state=42
    )
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 2,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=4)
    # For multiclass, output_margin gives raw scores per class
    save_test_case("multiclass", booster, X_test, output_margin=True)


def generate_with_missing():
    """Regression with missing values (NaN) in test data."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=4, noise=5.0, random_state=42)
    
    # Create test data with some NaN values
    X_test = X[:10].copy()
    X_test[0, 0] = np.nan  # First row, first feature
    X_test[2, 1] = np.nan  # Third row, second feature
    X_test[5, :] = np.nan  # Sixth row, all features missing
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_test_case("regression_missing", booster, X_test)


def generate_dart():
    """DART booster regression."""
    np.random.seed(42)
    X, y = make_regression(n_samples=90, n_features=3, noise=5.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'booster': 'dart',
        'max_depth': 2,
        'eta': 0.1,
        'rate_drop': 0.1,
        'skip_drop': 0.5,
    }
    booster = xgb.train(params, dtrain, num_boost_round=4)
    save_test_case("dart_regression", booster, X_test)


if __name__ == "__main__":
    print(f"Generating XGBoost test cases to {OUT_DIR}...\n")
    generate_regression()
    generate_binary_logistic()
    generate_multiclass()
    generate_with_missing()
    generate_dart()
    print(f"\nDone! Test cases saved to {OUT_DIR}")
