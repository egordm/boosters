"""Generate XGBoost test cases with models, input data, and expected predictions.

Each test case generates:
- {name}.model.json      - The XGBoost model
- {name}.input.json      - Test input features  
- {name}.expected.json   - Expected predictions (raw margin + transformed)

Test cases are organized by booster type:
- gbtree/   - Standard tree ensemble models
- gblinear/ - Linear booster models  
- dart/     - DART booster models
- training/ - Training test cases (for training validation)

This allows Rust tests to load a model, predict on inputs, and compare to expected output.
We use XGBoost's own predictions for both raw and transformed to ensure correctness.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.datasets import make_regression, make_classification
import xgboost as xgb

from .common import nan_to_null, save_test_input, save_test_expected, print_success

# =============================================================================
# Output Directories
# =============================================================================
# Structure:
#   xgboost/
#   ├── gbtree/
#   │   ├── inference/   # model + input + expected predictions
#   │   └── training/    # train_data + labels + config + xgb_predictions
#   ├── gblinear/
#   │   ├── inference/
#   │   └── training/
#   └── dart/
#       └── inference/

BASE_DIR = Path(__file__).parents[3] / "tests" / "test-cases" / "xgboost"

# Inference directories (model + input + expected)
GBTREE_INFERENCE_DIR = BASE_DIR / "gbtree" / "inference"
GBLINEAR_INFERENCE_DIR = BASE_DIR / "gblinear" / "inference"
DART_INFERENCE_DIR = BASE_DIR / "dart" / "inference"

# Training directories (train data + labels + config + xgb predictions)
GBTREE_TRAINING_DIR = BASE_DIR / "gbtree" / "training"
GBLINEAR_TRAINING_DIR = BASE_DIR / "gblinear" / "training"

# Benchmark models
BENCH_DIR = Path(__file__).parents[3] / "tests" / "test-cases" / "benchmark"

# Legacy aliases for backward compatibility
GBTREE_DIR = GBTREE_INFERENCE_DIR
GBLINEAR_DIR = GBLINEAR_INFERENCE_DIR
DART_DIR = DART_INFERENCE_DIR
TRAINING_DIR = GBLINEAR_TRAINING_DIR  # Legacy: gblinear training

# Ensure directories exist
for d in [
    GBTREE_INFERENCE_DIR, GBLINEAR_INFERENCE_DIR, DART_INFERENCE_DIR,
    GBTREE_TRAINING_DIR, GBLINEAR_TRAINING_DIR, BENCH_DIR
]:
    d.mkdir(parents=True, exist_ok=True)


def save_xgb_test_case(
    name: str,
    booster: xgb.Booster,
    X_test: np.ndarray,
    objective: str,
    num_class: int = 0,
    output_dir: Path = GBTREE_DIR,
    dtest: xgb.DMatrix | None = None,
    feature_types: list[str] | None = None,
):
    """Save a complete XGBoost test case: model, inputs, and expected predictions.
    
    Args:
        name: Test case name (without extension)
        booster: Trained XGBoost booster
        X_test: Test features
        objective: Objective function name
        num_class: Number of classes (0 for regression/binary)
        output_dir: Directory to save files
        dtest: Optional pre-built DMatrix for categorical features
        feature_types: Optional feature type list for categorical features
    """
    # Save model
    model_path = output_dir / f"{name}.model.json"
    booster.save_model(str(model_path))
    
    # Save test inputs
    input_path = output_dir / f"{name}.input.json"
    save_test_input(input_path, X_test, feature_types)
    
    # Get predictions from XGBoost itself
    if dtest is None:
        dtest = xgb.DMatrix(X_test)
    raw_predictions = booster.predict(dtest, output_margin=True)
    transformed_predictions = booster.predict(dtest, output_margin=False)
    
    # Save expected predictions
    expected_path = output_dir / f"{name}.expected.json"
    save_test_expected(expected_path, raw_predictions, transformed_predictions, objective, num_class)
    
    print_success(name, len(X_test))


# =============================================================================
# GBTree Test Cases
# =============================================================================

def generate_regression():
    """Simple regression: 5 features, 5 trees."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case("gbtree_regression", booster, X_test, objective='reg:squarederror')


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
    save_xgb_test_case("gbtree_binary_logistic", booster, X_test, objective='binary:logistic')


def generate_binary_logistic_missing():
    """Binary classification with missing values."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, random_state=42
    )
    
    X_test = X[:10].copy()
    X_test[0, 0] = np.nan
    X_test[2, 1] = np.nan
    X_test[4, :2] = np.nan
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 2,
        'eta': 0.15,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case("gbtree_binary_logistic_missing", booster, X_test, objective='binary:logistic')


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
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 2,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=4)
    save_xgb_test_case("gbtree_multiclass", booster, X_test, objective='multi:softprob', num_class=3)


def generate_multiclass_missing():
    """Multiclass classification with missing values."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=1, random_state=42
    )
    
    X_test = X[:10].copy()
    X_test[1, 0] = np.nan
    X_test[3, 2] = np.nan
    X_test[7, :] = np.nan
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 2,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=4)
    save_xgb_test_case("gbtree_multiclass_missing", booster, X_test, objective='multi:softprob', num_class=3)


def generate_regression_missing():
    """Regression with missing values (NaN) in test data."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=4, noise=5.0, random_state=42)
    
    X_test = X[:10].copy()
    X_test[0, 0] = np.nan
    X_test[2, 1] = np.nan
    X_test[5, :] = np.nan
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case("gbtree_regression_missing", booster, X_test, objective='reg:squarederror')


def generate_deep_trees():
    """Regression with deep trees (max_depth=6)."""
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
    save_xgb_test_case("gbtree_deep_trees", booster, X_test, objective='reg:squarederror')


def generate_single_tree():
    """Single tree regression - edge case."""
    np.random.seed(42)
    X, y = make_regression(n_samples=50, n_features=3, noise=5.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 1.0,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=1)
    save_xgb_test_case("gbtree_single_tree", booster, X_test, objective='reg:squarederror')


def generate_many_trees():
    """Many trees (50) to test accumulation precision."""
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 2,
        'eta': 0.05,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=50)
    save_xgb_test_case("gbtree_many_trees", booster, X_test, objective='reg:squarederror')


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
        'colsample_bytree': 0.5,
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case("gbtree_wide_features", booster, X_test, objective='reg:squarederror')


def generate_categorical():
    """Regression with categorical features."""
    np.random.seed(42)
    n_samples = 200
    
    X_numeric = np.random.randn(n_samples, 3)
    X_cat1 = np.random.randint(0, 3, size=(n_samples, 1)).astype(np.float32)
    X_cat2 = np.random.randint(0, 5, size=(n_samples, 1)).astype(np.float32)
    X = np.hstack([X_numeric, X_cat1, X_cat2])
    
    y = (
        X[:, 0] * 2
        + np.where(X[:, 3] == 0, 10, np.where(X[:, 3] == 1, -5, 0))
        + np.where(X[:, 4] >= 3, 8, -4)
        + np.random.randn(n_samples) * 2
    )
    
    X_test = X[:10].copy()
    X_test[0, 3] = 0
    X_test[1, 3] = 1
    X_test[2, 3] = 2
    X_test[3, 4] = 0
    X_test[4, 4] = 2
    X_test[5, 4] = 4
    
    feature_types = ['q', 'q', 'q', 'c', 'c']
    
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    dtrain.set_info(feature_types=feature_types)
    
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    dtest.set_info(feature_types=feature_types)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.3,
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_cat_to_onehot': 1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case(
        "gbtree_categorical", booster, X_test, objective='reg:squarederror',
        dtest=dtest, feature_types=feature_types
    )


def generate_categorical_binary():
    """Binary classification with categorical features."""
    np.random.seed(42)
    n_samples = 200
    
    X_numeric = np.random.randn(n_samples, 2)
    X_cat = np.random.randint(0, 4, size=(n_samples, 1)).astype(np.float32)
    X = np.hstack([X_numeric, X_cat])
    
    base_prob = 1 / (1 + np.exp(-X[:, 0]))
    cat_effect = np.where((X[:, 2] == 0) | (X[:, 2] == 2), 0.3, -0.3)
    prob = np.clip(base_prob + cat_effect, 0.05, 0.95)
    y = (np.random.rand(n_samples) < prob).astype(np.float32)
    
    X_test = X[:10].copy()
    X_test[0, 2] = 0
    X_test[1, 2] = 1
    X_test[2, 2] = 2
    X_test[3, 2] = 3
    
    feature_types = ['q', 'q', 'c']
    
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    dtrain.set_info(feature_types=feature_types)
    
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    dtest.set_info(feature_types=feature_types)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'eta': 0.3,
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_cat_to_onehot': 1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case(
        "gbtree_categorical_binary", booster, X_test, objective='binary:logistic',
        dtest=dtest, feature_types=feature_types
    )


# =============================================================================
# DART Test Cases
# =============================================================================

def generate_dart_regression():
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
    save_xgb_test_case(
        "dart_regression", booster, X_test, 
        objective='reg:squarederror', output_dir=DART_DIR
    )


# =============================================================================
# GBLinear Test Cases
# =============================================================================

def generate_gblinear_regression():
    """GBLinear regression model."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=5.0, random_state=42)
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'booster': 'gblinear',
        'eta': 0.1,
        'lambda': 0.1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=50)
    save_xgb_test_case(
        "gblinear_regression", booster, X_test,
        objective='reg:squarederror', output_dir=GBLINEAR_DIR
    )


def generate_gblinear_binary():
    """GBLinear binary classification."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, random_state=42
    )
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'booster': 'gblinear',
        'eta': 0.1,
        'lambda': 0.1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=50)
    save_xgb_test_case(
        "gblinear_binary", booster, X_test,
        objective='binary:logistic', output_dir=GBLINEAR_DIR
    )


def generate_gblinear_multiclass():
    """GBLinear multiclass classification."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=1, random_state=42
    )
    X_test = X[:10]
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'booster': 'gblinear',
        'eta': 0.1,
        'lambda': 0.1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=50)
    save_xgb_test_case(
        "gblinear_multiclass", booster, X_test,
        objective='multi:softprob', num_class=3, output_dir=GBLINEAR_DIR
    )


# =============================================================================
# Benchmark Models
# =============================================================================

def save_benchmark_model(name: str, booster: xgb.Booster, num_features: int):
    """Save a benchmark model (model only, no test data needed)."""
    model_path = BENCH_DIR / f"{name}.model.json"
    booster.save_model(str(model_path))
    size_kb = model_path.stat().st_size / 1024
    print(f"✓ {name}: {size_kb:.1f} KB")


def generate_bench_small():
    """Small benchmark model: 10 trees, 5 features, max_depth=3."""
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=5, noise=10.0, random_state=42)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=10)
    save_benchmark_model("bench_small", booster, num_features=5)


def generate_bench_medium():
    """Medium benchmark model: 100 trees, 50 features, max_depth=4."""
    np.random.seed(42)
    X, y = make_regression(n_samples=2000, n_features=50, n_informative=30, noise=10.0, random_state=42)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'eta': 0.05,
        'booster': 'gbtree',
        'colsample_bytree': 0.8,
    }
    booster = xgb.train(params, dtrain, num_boost_round=100)
    save_benchmark_model("bench_medium", booster, num_features=50)


def generate_bench_large():
    """Large benchmark model: 500 trees, 100 features, max_depth=5."""
    np.random.seed(42)
    X, y = make_regression(n_samples=5000, n_features=100, n_informative=50, noise=10.0, random_state=42)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'eta': 0.02,
        'booster': 'gbtree',
        'colsample_bytree': 0.7,
        'subsample': 0.8,
    }
    booster = xgb.train(params, dtrain, num_boost_round=500)
    save_benchmark_model("bench_large", booster, num_features=100)


# =============================================================================
# Main Entry Points
# =============================================================================

def generate_all_gbtree():
    """Generate all GBTree test cases."""
    print("Generating GBTree test cases...")
    generate_regression()
    generate_binary_logistic()
    generate_multiclass()
    generate_regression_missing()
    generate_binary_logistic_missing()
    generate_multiclass_missing()
    generate_deep_trees()
    generate_single_tree()
    generate_many_trees()
    generate_wide_features()
    generate_categorical()
    generate_categorical_binary()


def generate_all_dart():
    """Generate all DART test cases."""
    print("\nGenerating DART test cases...")
    generate_dart_regression()


def generate_all_gblinear():
    """Generate all GBLinear test cases."""
    print("\nGenerating GBLinear test cases...")
    generate_gblinear_regression()
    generate_gblinear_binary()
    generate_gblinear_multiclass()


def generate_all_benchmarks():
    """Generate all benchmark models."""
    print("\nGenerating benchmark models...")
    generate_bench_small()
    generate_bench_medium()
    generate_bench_large()


# =============================================================================
# Training Test Cases (for training validation)
# =============================================================================

def save_training_case(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    booster: xgb.Booster,
    config: dict,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    output_dir: Path = TRAINING_DIR,
):
    """Save a complete training test case.
    
    Args:
        name: Test case name
        X_train: Training features
        y_train: Training labels
        booster: Trained XGBoost booster
        config: Training configuration
        X_test: Optional held-out test features
        y_test: Optional held-out test labels
        output_dir: Directory to save files
    """
    # Save training data
    train_data_path = output_dir / f"{name}.train_data.json"
    with open(train_data_path, 'w') as f:
        json.dump({
            "num_rows": int(X_train.shape[0]),
            "num_features": int(X_train.shape[1]),
            "data": nan_to_null(X_train.flatten().tolist()),
        }, f, indent=2)
    
    # Save labels
    labels_path = output_dir / f"{name}.train_labels.json"
    with open(labels_path, 'w') as f:
        json.dump({
            "labels": y_train.tolist(),
        }, f, indent=2)
    
    # Extract weights from XGBoost model
    model_str = booster.save_raw(raw_format='json').decode('utf-8')
    model_json = json.loads(model_str)
    
    # GBLinear weights are in learner.gradient_booster.model.weights
    gblinear_model = model_json['learner']['gradient_booster']['model']
    weights = gblinear_model['weights']
    
    weights_path = output_dir / f"{name}.xgb_weights.json"
    with open(weights_path, 'w') as f:
        json.dump({
            "weights": weights,
            "num_features": int(X_train.shape[1]),
            "num_groups": config.get("num_class", 1),
        }, f, indent=2)
    
    # Save config
    config_path = output_dir / f"{name}.config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save held-out test data and predictions if provided
    if X_test is not None and y_test is not None:
        test_data_path = output_dir / f"{name}.test_data.json"
        with open(test_data_path, 'w') as f:
            json.dump({
                "num_rows": int(X_test.shape[0]),
                "num_features": int(X_test.shape[1]),
                "data": nan_to_null(X_test.flatten().tolist()),
            }, f, indent=2)
        
        test_labels_path = output_dir / f"{name}.test_labels.json"
        with open(test_labels_path, 'w') as f:
            json.dump({
                "labels": y_test.tolist(),
            }, f, indent=2)
        
        # Get XGBoost predictions on test data
        dtest = xgb.DMatrix(X_test)
        xgb_predictions = booster.predict(dtest, output_margin=True)
        
        predictions_path = output_dir / f"{name}.xgb_predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump({
                "predictions": xgb_predictions.tolist(),
            }, f, indent=2)
    
    print_success(f"training/{name}", len(X_train))


def generate_training_regression_simple():
    """Simple linear regression - y = 2*x + 1."""
    np.random.seed(42)
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0], dtype=np.float32)  # y = 2x + 1
    
    dtrain = xgb.DMatrix(X, label=y)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gblinear',
        'eta': 0.5,
        'lambda': 0.0,
        'alpha': 0.0,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'base_score': 0.0,
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("regression_simple", X, y, booster, config)


def generate_training_regression_multifeature():
    """Multi-feature linear regression - y = x0 + 2*x1 + 3."""
    np.random.seed(42)
    X = np.array([
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [2.0, 3.0],
    ], dtype=np.float32)
    y = X[:, 0] + 2 * X[:, 1] + 3
    
    dtrain = xgb.DMatrix(X, label=y)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gblinear',
        'eta': 0.3,
        'lambda': 0.0,
        'alpha': 0.0,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'base_score': 0.0,
        'num_boost_round': 200,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("regression_multifeature", X, y, booster, config)


def generate_training_regression_l2():
    """Regression with L2 regularization and held-out test set."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=5.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Split into train/test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gblinear',
        'eta': 0.5,
        'lambda': 1.0,
        'alpha': 0.0,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("regression_l2", X_train, y_train, booster, config, X_test, y_test)


def generate_training_regression_elastic_net():
    """Regression with elastic net (L1 + L2) regularization and held-out test set."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=10, noise=5.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Split into train/test
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gblinear',
        'eta': 0.3,
        'lambda': 1.0,
        'alpha': 0.5,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("regression_elastic_net", X_train, y_train, booster, config, X_test, y_test)


def generate_training_binary():
    """Binary logistic classification with held-out test set."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=150, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, random_state=42
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Split into train/test
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'binary:logistic',
        'booster': 'gblinear',
        'eta': 0.3,
        'lambda': 0.1,
        'alpha': 0.0,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("binary_classification", X_train, y_train, booster, config, X_test, y_test)


def generate_training_multiclass():
    """Multiclass softmax classification with held-out test set."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=180, n_features=4, n_informative=3,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=1, random_state=42
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Split into train/test
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'booster': 'gblinear',
        'eta': 0.3,
        'lambda': 0.1,
        'alpha': 0.0,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("multiclass_classification", X_train, y_train, booster, config, X_test, y_test)


def generate_training_quantile():
    """Quantile regression (median, α=0.5) with held-out test set."""
    np.random.seed(42)
    
    # Generate data with heteroscedastic noise (variance depends on X)
    # This makes quantile regression more interesting
    X, y = make_regression(n_samples=150, n_features=5, noise=0.0, random_state=42)
    X = X.astype(np.float32)
    
    # Add heteroscedastic noise: larger variance for larger X values
    noise_scale = 1 + 0.5 * np.abs(X[:, 0])
    y = y + np.random.normal(0, noise_scale * 5)
    y = y.astype(np.float32)
    
    # Split into train/test
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': 0.5,  # Median regression
        'booster': 'gblinear',
        'eta': 0.5,
        'lambda': 1.0,
        'alpha': 0.0,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("quantile_regression", X_train, y_train, booster, config, X_test, y_test)


def generate_training_quantile_low():
    """Quantile regression (10th percentile, α=0.1)."""
    np.random.seed(42)
    X, y = make_regression(n_samples=150, n_features=5, noise=10.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': 0.1,  # 10th percentile
        'booster': 'gblinear',
        'eta': 0.5,
        'lambda': 1.0,
        'alpha': 0.0,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("quantile_low", X_train, y_train, booster, config, X_test, y_test)


def generate_training_quantile_high():
    """Quantile regression (90th percentile, α=0.9)."""
    np.random.seed(42)
    X, y = make_regression(n_samples=150, n_features=5, noise=10.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': 0.9,  # 90th percentile
        'booster': 'gblinear',
        'eta': 0.5,
        'lambda': 1.0,
        'alpha': 0.0,
        'updater': 'coord_descent',
        'feature_selector': 'cyclic',
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_training_case("quantile_high", X_train, y_train, booster, config, X_test, y_test)


def generate_training_multi_quantile():
    """Multi-quantile regression (10th, 50th, 90th percentiles) with held-out test set.
    
    This tests XGBoost's multi-output quantile regression feature, where
    multiple quantiles are predicted simultaneously with a single model.
    
    Uses 'reg:quantileerror' with 'quantile_alpha' as a list.
    
    NOTE: XGBoost's gblinear doesn't support multi-quantile, but gbtree does.
    We train a gbtree model here for reference, then train 3 separate gblinear
    models to compare.
    """
    np.random.seed(42)
    
    # Generate data with heteroscedastic noise for interesting quantile behavior
    X, y = make_regression(n_samples=200, n_features=5, noise=0.0, random_state=42)
    X = X.astype(np.float32)
    
    # Add heteroscedastic noise
    noise_scale = 1 + 0.5 * np.abs(X[:, 0])
    y = y + np.random.normal(0, noise_scale * 10)
    y = y.astype(np.float32)
    
    # Split into train/test
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    
    # Multi-quantile: predict 10th, 50th, 90th percentiles simultaneously
    quantile_alphas = [0.1, 0.5, 0.9]
    
    # Train 3 separate gblinear models (one per quantile) since XGBoost gblinear
    # doesn't support multi-target output
    all_predictions = []
    all_weights = []
    
    for i, alpha in enumerate(quantile_alphas):
        params = {
            'objective': 'reg:quantileerror',
            'quantile_alpha': alpha,
            'booster': 'gblinear',
            'eta': 0.5,
            'lambda': 1.0,
            'alpha': 0.0,
            'updater': 'coord_descent',
            'feature_selector': 'cyclic',
        }
        booster = xgb.train(params, dtrain, num_boost_round=100)
        
        # Get predictions
        preds = booster.predict(dtest, output_margin=True)
        all_predictions.append(preds.tolist())
        
        # Extract weights
        model_str = booster.save_raw(raw_format='json').decode('utf-8')
        model_json = json.loads(model_str)
        gblinear_model = model_json['learner']['gradient_booster']['model']
        weights = gblinear_model['weights']
        all_weights.append(weights)
    
    # Save training data
    train_data_path = TRAINING_DIR / "multi_quantile.train_data.json"
    with open(train_data_path, 'w') as f:
        json.dump({
            "num_rows": int(X_train.shape[0]),
            "num_features": int(X_train.shape[1]),
            "data": nan_to_null(X_train.flatten().tolist()),
        }, f, indent=2)
    
    # Save labels
    labels_path = TRAINING_DIR / "multi_quantile.train_labels.json"
    with open(labels_path, 'w') as f:
        json.dump({
            "labels": y_train.tolist(),
        }, f, indent=2)
    
    # Save config
    config = {
        'objective': 'reg:quantileerror',
        'quantile_alpha': quantile_alphas,
        'booster': 'gblinear',
        'eta': 0.5,
        'lambda': 1.0,
        'alpha': 0.0,
        'num_boost_round': 100,
        'num_quantiles': len(quantile_alphas),
    }
    config_path = TRAINING_DIR / "multi_quantile.config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save weights from all 3 models (for comparison)
    weights_path = TRAINING_DIR / "multi_quantile.xgb_weights.json"
    with open(weights_path, 'w') as f:
        json.dump({
            "quantile_weights": all_weights,  # [[weights for q0], [weights for q1], ...]
            "num_features": int(X_train.shape[1]),
            "num_quantiles": len(quantile_alphas),
            "quantile_alphas": quantile_alphas,
        }, f, indent=2)
    
    # Save test data
    test_data_path = TRAINING_DIR / "multi_quantile.test_data.json"
    with open(test_data_path, 'w') as f:
        json.dump({
            "num_rows": int(X_test.shape[0]),
            "num_features": int(X_test.shape[1]),
            "data": nan_to_null(X_test.flatten().tolist()),
        }, f, indent=2)
    
    test_labels_path = TRAINING_DIR / "multi_quantile.test_labels.json"
    with open(test_labels_path, 'w') as f:
        json.dump({
            "labels": y_test.tolist(),
        }, f, indent=2)
    
    # Save predictions from each quantile model
    predictions_path = TRAINING_DIR / "multi_quantile.xgb_predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump({
            "predictions": all_predictions,  # [[preds for q0], [preds for q1], ...]
            "quantile_alphas": quantile_alphas,
        }, f, indent=2)
    
    print_success("training/multi_quantile", len(X_train))


# =============================================================================
# GBTree Training Test Cases
# =============================================================================


def save_gbtree_training_case(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    booster: xgb.Booster,
    config: dict,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    train_predictions: np.ndarray | None = None,
):
    """Save a complete GBTree training test case.
    
    Args:
        name: Test case name
        X_train: Training features
        y_train: Training labels
        booster: Trained XGBoost booster
        config: Training configuration
        X_test: Optional held-out test features
        y_test: Optional held-out test labels
        train_predictions: Optional training set predictions
    """
    output_dir = GBTREE_TRAINING_DIR
    
    # Save training data
    train_data_path = output_dir / f"{name}.train_data.json"
    with open(train_data_path, 'w') as f:
        json.dump({
            "num_rows": int(X_train.shape[0]),
            "num_features": int(X_train.shape[1]),
            "data": nan_to_null(X_train.flatten().tolist()),
        }, f, indent=2)
    
    # Save labels
    labels_path = output_dir / f"{name}.train_labels.json"
    with open(labels_path, 'w') as f:
        json.dump({
            "labels": y_train.tolist(),
        }, f, indent=2)
    
    # Save model
    model_path = output_dir / f"{name}.model.json"
    booster.save_model(str(model_path))
    
    # Save config
    config_path = output_dir / f"{name}.config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Get training predictions
    dtrain = xgb.DMatrix(X_train, label=y_train)
    train_preds = booster.predict(dtrain, output_margin=True)
    
    train_preds_path = output_dir / f"{name}.train_predictions.json"
    with open(train_preds_path, 'w') as f:
        json.dump({
            "predictions": train_preds.tolist(),
        }, f, indent=2)
    
    # Save held-out test data and predictions if provided
    if X_test is not None and y_test is not None:
        test_data_path = output_dir / f"{name}.test_data.json"
        with open(test_data_path, 'w') as f:
            json.dump({
                "num_rows": int(X_test.shape[0]),
                "num_features": int(X_test.shape[1]),
                "data": nan_to_null(X_test.flatten().tolist()),
            }, f, indent=2)
        
        test_labels_path = output_dir / f"{name}.test_labels.json"
        with open(test_labels_path, 'w') as f:
            json.dump({
                "labels": y_test.tolist(),
            }, f, indent=2)
        
        # Get XGBoost predictions on test data
        dtest = xgb.DMatrix(X_test)
        xgb_predictions = booster.predict(dtest, output_margin=True)
        
        predictions_path = output_dir / f"{name}.test_predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump({
                "predictions": xgb_predictions.tolist(),
            }, f, indent=2)
        
        # Compute test metrics (handle multiclass case)
        metrics = {"num_trees": int(booster.num_boosted_rounds())}
        
        if xgb_predictions.ndim == 1:
            # Regression or binary: compute RMSE
            test_rmse = np.sqrt(np.mean((xgb_predictions - y_test) ** 2))
            metrics["test_rmse"] = float(test_rmse)
        else:
            # Multiclass: compute accuracy
            pred_classes = np.argmax(xgb_predictions, axis=1)
            accuracy = np.mean(pred_classes == y_test)
            metrics["test_accuracy"] = float(accuracy)
        
        metrics_path = output_dir / f"{name}.metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print_success(f"training/gbtree/{name}", len(X_train))


def generate_gbtree_training_regression_simple():
    """Simple regression: 100 samples, 5 features, 20 trees, depth 3."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=5, noise=5.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_depth': 3,
        'eta': 0.3,
        'lambda': 1.0,
        'alpha': 0.0,
        'min_child_weight': 1.0,
        'num_boost_round': 20,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_gbtree_training_case("regression_simple", X_train, y_train, booster, config, X_test, y_test)


def generate_gbtree_training_regression_deep():
    """Deeper trees: max_depth=6, 50 trees."""
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=10, noise=10.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_depth': 6,
        'eta': 0.1,
        'lambda': 1.0,
        'alpha': 0.0,
        'min_child_weight': 1.0,
        'num_boost_round': 50,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_gbtree_training_case("regression_deep", X_train, y_train, booster, config, X_test, y_test)


def generate_gbtree_training_regression_regularized():
    """Regression with L1 + L2 regularization."""
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=20, noise=10.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_depth': 4,
        'eta': 0.2,
        'lambda': 5.0,  # Strong L2
        'alpha': 2.0,   # Some L1
        'min_child_weight': 5.0,
        'gamma': 1.0,   # Min split gain
        'num_boost_round': 50,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_gbtree_training_case("regression_regularized", X_train, y_train, booster, config, X_test, y_test)


def generate_gbtree_training_binary():
    """Binary logistic classification with trees."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=8,
        n_redundant=0, n_repeated=0, random_state=42
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_depth': 4,
        'eta': 0.2,
        'lambda': 1.0,
        'alpha': 0.0,
        'min_child_weight': 1.0,
        'num_boost_round': 30,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_gbtree_training_case("binary_classification", X_train, y_train, booster, config, X_test, y_test)


def generate_gbtree_training_multiclass():
    """Multiclass softmax classification."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300, n_features=10, n_informative=8,
        n_redundant=0, n_repeated=0, n_classes=3,
        n_clusters_per_class=1, random_state=42
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_depth': 4,
        'eta': 0.2,
        'lambda': 1.0,
        'alpha': 0.0,
        'min_child_weight': 1.0,
        'num_boost_round': 30,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_gbtree_training_case("multiclass", X_train, y_train, booster, config, X_test, y_test)


def generate_gbtree_training_leaf_wise():
    """Leaf-wise tree growth (grow_policy='lossguide')."""
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=10, noise=10.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:160], X[160:]
    y_train, y_test = y[:160], y[160:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'grow_policy': 'lossguide',  # Leaf-wise
        'max_leaves': 16,
        'max_depth': 0,  # No depth limit when using leaf-wise
        'eta': 0.2,
        'lambda': 1.0,
        'alpha': 0.0,
        'min_child_weight': 1.0,
        'num_boost_round': 30,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_gbtree_training_case("leaf_wise", X_train, y_train, booster, config, X_test, y_test)


def generate_gbtree_training_large():
    """Larger dataset for performance validation."""
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=20, noise=10.0, random_state=42)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    config = {
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'max_depth': 5,
        'eta': 0.1,
        'lambda': 1.0,
        'alpha': 0.0,
        'min_child_weight': 1.0,
        'num_boost_round': 100,
    }
    
    params = {k: v for k, v in config.items() if k != 'num_boost_round'}
    booster = xgb.train(params, dtrain, num_boost_round=config['num_boost_round'])
    
    save_gbtree_training_case("large", X_train, y_train, booster, config, X_test, y_test)


def generate_all_gbtree_training():
    """Generate all GBTree training test cases."""
    print("\nGenerating GBTree training test cases...")
    generate_gbtree_training_regression_simple()
    generate_gbtree_training_regression_deep()
    generate_gbtree_training_regression_regularized()
    generate_gbtree_training_binary()
    generate_gbtree_training_multiclass()
    generate_gbtree_training_leaf_wise()
    generate_gbtree_training_large()


def generate_all_training():
    """Generate all training test cases."""
    print("\nGenerating training test cases...")
    # GBLinear training cases
    generate_training_regression_simple()
    generate_training_regression_multifeature()
    generate_training_regression_l2()
    generate_training_regression_elastic_net()
    generate_training_binary()
    generate_training_multiclass()
    generate_training_quantile()
    generate_training_quantile_low()
    generate_training_quantile_high()
    generate_training_multi_quantile()
    # GBTree training cases
    generate_all_gbtree_training()


# =============================================================================
# Main Entry Points
# =============================================================================

def generate_all():
    """Generate all XGBoost test cases and benchmarks."""
    generate_all_gbtree()
    generate_all_dart()
    generate_all_gblinear()
    generate_all_benchmarks()
    generate_all_training()
    print("\nDone!")


if __name__ == "__main__":
    generate_all()
