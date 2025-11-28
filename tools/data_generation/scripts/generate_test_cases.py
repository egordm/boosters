"""Generate XGBoost test cases with models, input data, and expected predictions.

Each test case generates:
- {name}.model.json      - The XGBoost model
- {name}.input.json      - Test input features  
- {name}.expected.json   - Expected predictions (raw margin + XGBoost transformed)

This allows Rust tests to load a model, predict on inputs, and compare to expected output.
We use XGBoost's own predictions for both raw and transformed to ensure correctness.
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


def save_test_case(
    name: str, 
    booster: xgb.Booster, 
    X_test: np.ndarray, 
    objective: str,
    num_class: int = 0,
):
    """Save a complete test case: model, inputs, and expected predictions.
    
    Uses XGBoost's own predict() for both raw and transformed outputs.
    """
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
    
    # Get predictions from XGBoost itself (not our own transforms!)
    dtest = xgb.DMatrix(X_test)
    raw_predictions = booster.predict(dtest, output_margin=True)
    transformed_predictions = booster.predict(dtest, output_margin=False)
    
    expected_path = OUT_DIR / f"{name}.expected.json"
    with open(expected_path, "w") as f:
        json.dump({
            "predictions": raw_predictions.tolist(),
            "predictions_transformed": transformed_predictions.tolist(),
            "objective": objective,
            "num_class": num_class,
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
    save_test_case("regression", booster, X_test, objective='reg:squarederror')


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
    save_test_case("binary_logistic", booster, X_test, objective='binary:logistic')


def generate_binary_logistic_missing():
    """Binary classification with missing values."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, random_state=42
    )
    
    # Create test data with some NaN values
    X_test = X[:10].copy()
    X_test[0, 0] = np.nan
    X_test[2, 1] = np.nan
    X_test[4, :2] = np.nan  # Multiple missing in one row
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 2,
        'eta': 0.15,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_test_case("binary_logistic_missing", booster, X_test, objective='binary:logistic')


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
        'objective': 'multi:softprob',  # Use softprob for probability output
        'num_class': 3,
        'max_depth': 2,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=4)
    save_test_case("multiclass", booster, X_test, objective='multi:softprob', num_class=3)


def generate_multiclass_missing():
    """Multiclass classification with missing values."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=1, random_state=42
    )
    
    # Create test data with some NaN values
    X_test = X[:10].copy()
    X_test[1, 0] = np.nan
    X_test[3, 2] = np.nan
    X_test[7, :] = np.nan  # All features missing
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 2,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=4)
    save_test_case("multiclass_missing", booster, X_test, objective='multi:softprob', num_class=3)


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
    save_test_case("regression_missing", booster, X_test, objective='reg:squarederror')


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
    save_test_case("dart_regression", booster, X_test, objective='reg:squarederror')


def generate_deep_trees():
    """Regression with deep trees (max_depth=6) for longer traversal paths."""
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=8, noise=5.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_test_case("deep_trees", booster, X_test, objective='reg:squarederror')


def generate_single_tree():
    """Single tree regression - edge case."""
    np.random.seed(42)
    X, y = make_regression(n_samples=50, n_features=3, noise=5.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 1.0,  # High learning rate since only 1 tree
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=1)
    save_test_case("single_tree", booster, X_test, objective='reg:squarederror')


def generate_many_trees():
    """Many trees (50) to test accumulation precision."""
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 2,
        'eta': 0.05,  # Low learning rate for many trees
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=50)
    save_test_case("many_trees", booster, X_test, objective='reg:squarederror')


def generate_wide_features():
    """Wide data (100 features) to test feature indexing."""
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=100, n_informative=20, noise=10.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.1,
        'booster': 'gbtree',
        'colsample_bytree': 0.5,  # Use subset of features per tree
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_test_case("wide_features", booster, X_test, objective='reg:squarederror')


def generate_categorical():
    """Regression with categorical features.
    
    Creates a model with both numeric and categorical features to test
    categorical split handling.
    """
    np.random.seed(42)
    n_samples = 200
    
    # Create numeric features
    X_numeric = np.random.randn(n_samples, 3)
    
    # Create categorical features (3 categories: 0, 1, 2)
    X_cat1 = np.random.randint(0, 3, size=(n_samples, 1)).astype(np.float32)
    # Create another categorical feature (5 categories: 0-4)
    X_cat2 = np.random.randint(0, 5, size=(n_samples, 1)).astype(np.float32)
    
    # Combine all features: [num0, num1, num2, cat1, cat2]
    X = np.hstack([X_numeric, X_cat1, X_cat2])
    
    # Target depends on categorical features
    y = (
        X[:, 0] * 2  # numeric
        + np.where(X[:, 3] == 0, 10, np.where(X[:, 3] == 1, -5, 0))  # cat1 effect
        + np.where(X[:, 4] >= 3, 8, -4)  # cat2 effect
        + np.random.randn(n_samples) * 2  # noise
    )
    
    # Test data covers various category values
    X_test = X[:10].copy()
    # Ensure we have different category values in test data
    X_test[0, 3] = 0  # cat1 = 0
    X_test[1, 3] = 1  # cat1 = 1
    X_test[2, 3] = 2  # cat1 = 2
    X_test[3, 4] = 0  # cat2 = 0
    X_test[4, 4] = 2  # cat2 = 2
    X_test[5, 4] = 4  # cat2 = 4
    
    # Create DMatrix with categorical feature specification
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    dtrain.set_info(feature_types=['q', 'q', 'q', 'c', 'c'])  # q=quantitative, c=categorical
    
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    dtest.set_info(feature_types=['q', 'q', 'q', 'c', 'c'])
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.3,
        'booster': 'gbtree',
        'tree_method': 'hist',  # Required for categorical
        'max_cat_to_onehot': 1,  # Force partition-based splits
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    
    # Save model
    model_path = OUT_DIR / "categorical.model.json"
    booster.save_model(str(model_path))
    
    # Save test inputs
    input_path = OUT_DIR / "categorical.input.json"
    with open(input_path, "w") as f:
        json.dump({
            "features": nan_to_null(X_test),
            "num_rows": len(X_test),
            "num_features": X_test.shape[1],
            "feature_types": ['q', 'q', 'q', 'c', 'c'],
        }, f, indent=2)
    
    # Get predictions from XGBoost
    raw_predictions = booster.predict(dtest, output_margin=True)
    transformed_predictions = booster.predict(dtest, output_margin=False)
    
    expected_path = OUT_DIR / "categorical.expected.json"
    with open(expected_path, "w") as f:
        json.dump({
            "predictions": raw_predictions.tolist(),
            "predictions_transformed": transformed_predictions.tolist(),
            "objective": 'reg:squarederror',
            "num_class": 0,
        }, f, indent=2)
    
    print(f"✓ categorical: model + {len(X_test)} test cases")


def generate_categorical_binary():
    """Binary classification with categorical features."""
    np.random.seed(42)
    n_samples = 200
    
    # Create features: 2 numeric + 1 categorical (4 categories)
    X_numeric = np.random.randn(n_samples, 2)
    X_cat = np.random.randint(0, 4, size=(n_samples, 1)).astype(np.float32)
    X = np.hstack([X_numeric, X_cat])
    
    # Target: categories 0,2 -> class 1, categories 1,3 -> class 0
    base_prob = 1 / (1 + np.exp(-X[:, 0]))  # sigmoid of first numeric
    cat_effect = np.where((X[:, 2] == 0) | (X[:, 2] == 2), 0.3, -0.3)
    prob = np.clip(base_prob + cat_effect, 0.05, 0.95)
    y = (np.random.rand(n_samples) < prob).astype(np.float32)
    
    X_test = X[:10].copy()
    # Ensure test data has all category values
    X_test[0, 2] = 0
    X_test[1, 2] = 1
    X_test[2, 2] = 2
    X_test[3, 2] = 3
    
    # Create DMatrix with categorical feature specification
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    dtrain.set_info(feature_types=['q', 'q', 'c'])
    
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    dtest.set_info(feature_types=['q', 'q', 'c'])
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'eta': 0.3,
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_cat_to_onehot': 1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    
    # Save model
    model_path = OUT_DIR / "categorical_binary.model.json"
    booster.save_model(str(model_path))
    
    # Save test inputs
    input_path = OUT_DIR / "categorical_binary.input.json"
    with open(input_path, "w") as f:
        json.dump({
            "features": nan_to_null(X_test),
            "num_rows": len(X_test),
            "num_features": X_test.shape[1],
            "feature_types": ['q', 'q', 'c'],
        }, f, indent=2)
    
    # Get predictions from XGBoost
    raw_predictions = booster.predict(dtest, output_margin=True)
    transformed_predictions = booster.predict(dtest, output_margin=False)
    
    expected_path = OUT_DIR / "categorical_binary.expected.json"
    with open(expected_path, "w") as f:
        json.dump({
            "predictions": raw_predictions.tolist(),
            "predictions_transformed": transformed_predictions.tolist(),
            "objective": 'binary:logistic',
            "num_class": 0,
        }, f, indent=2)
    
    print(f"✓ categorical_binary: model + {len(X_test)} test cases")


if __name__ == "__main__":
    print(f"Generating XGBoost test cases to {OUT_DIR}...\n")
    
    # Core cases
    generate_regression()
    generate_binary_logistic()
    generate_multiclass()
    generate_with_missing()
    generate_dart()
    
    # Additional coverage
    generate_binary_logistic_missing()
    generate_multiclass_missing()
    generate_deep_trees()
    generate_single_tree()
    generate_many_trees()
    generate_wide_features()
    
    # Categorical features
    generate_categorical()
    generate_categorical_binary()
    
    print(f"\nDone! Test cases saved to {OUT_DIR}")
