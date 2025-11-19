"""Generate XGBoost models with different objectives, boosters, and feature types.
Uses the native XGBoost API (xgb.train) instead of sklearn wrappers.
"""
from pathlib import Path
import numpy as np
from sklearn.datasets import make_regression, make_classification, make_blobs
import xgboost as xgb

out_dir = Path(__file__).parents[3] / "tests" / "models"
out_dir.mkdir(parents=True, exist_ok=True)


def generate_gbtree_regression():
    """Generate a gbtree regression model with reg:squarederror objective."""
    X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=42)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    booster.save_model(str(out_dir / "gbtree_regression.json"))
    print("✓ Saved gbtree regression model")


def generate_gblinear_binary():
    """Generate a gblinear binary classification model."""
    X, y = make_classification(n_samples=80, n_features=3, n_informative=2, n_redundant=0, n_repeated=0, random_state=42)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'booster': 'gblinear',
        'eta': 0.1,
        'lambda': 0.1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=3)
    booster.save_model(str(out_dir / "gblinear_binary.json"))
    print("✓ Saved gblinear binary model")


def generate_gbtree_multiclass_softmax():
    """Generate a gbtree multiclass classifier using softmax."""
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1, random_state=42
    )
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'max_depth': 2,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=4)
    booster.save_model(str(out_dir / "gbtree_multiclass_softmax.json"))
    print("✓ Saved gbtree multiclass softmax model")


def generate_gbtree_multiclass_softprob():
    """Generate a gbtree multiclass classifier using softprob (probabilities)."""
    X, y = make_classification(
        n_samples=120, n_features=4, n_informative=3, n_redundant=0, n_repeated=0, n_classes=4, n_clusters_per_class=1, random_state=42
    )
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'max_depth': 3,
        'eta': 0.2,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=6)
    booster.save_model(str(out_dir / "gbtree_multiclass_softprob.json"))
    print("✓ Saved gbtree multiclass softprob model")


def generate_dart_regression():
    """Generate a DART (Dropouts meet Multiple Additive Regression Trees) regression model."""
    X, y = make_regression(n_samples=90, n_features=3, noise=5.0, random_state=42)
    
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
    booster.save_model(str(out_dir / "dart_regression.json"))
    print("✓ Saved DART regression model")


def generate_gblinear_multiclass():
    """Generate a gblinear multiclass classifier."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=4, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1, random_state=42
    )
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'booster': 'gblinear',
        'eta': 0.1,
        'lambda': 0.1,
    }
    booster = xgb.train(params, dtrain, num_boost_round=2)
    booster.save_model(str(out_dir / "gblinear_multiclass.json"))
    print("✓ Saved gblinear multiclass model")


def generate_binary_logistic():
    """Generate a binary logistic regression model."""
    X, y = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=0, n_repeated=0, random_state=42)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 2,
        'eta': 0.15,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    booster.save_model(str(out_dir / "gbtree_binary_logistic.json"))
    print("✓ Saved gbtree binary logistic model")


def generate_multiclass_with_blobs():
    """Generate a multiclass classifier using make_blobs."""
    X, y = make_blobs(n_samples=150, n_features=3, centers=5, random_state=42)
    
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'objective': 'multi:softmax',
        'num_class': 5,
        'max_depth': 3,
        'eta': 0.1,
        'booster': 'gbtree',
    }
    booster = xgb.train(params, dtrain, num_boost_round=5)
    booster.save_model(str(out_dir / "gbtree_multiclass_blobs.json"))
    print("✓ Saved gbtree multiclass model from blobs")


if __name__ == "__main__":
    print("Generating XGBoost models...")
    generate_gbtree_regression()
    generate_gblinear_binary()
    generate_gbtree_multiclass_softmax()
    generate_gbtree_multiclass_softprob()
    generate_dart_regression()
    generate_gblinear_multiclass()
    generate_binary_logistic()
    generate_multiclass_with_blobs()
    print(f"\nAll models saved to {out_dir}")
