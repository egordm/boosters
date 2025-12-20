"""LightGBM test case generation."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification, make_regression

from boosters_datagen.utils import print_success, save_json

# Output directory
BASE_DIR = Path(__file__).parents[4] / "tests" / "test-cases" / "lightgbm"
INFERENCE_DIR = BASE_DIR / "inference"


def ensure_dirs() -> None:
    """Create output directories."""
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)


def save_lgb_test_case(
    name: str,
    model: lgb.Booster,
    x_test: np.ndarray,
    y_test: np.ndarray,
    expected_raw: np.ndarray,
    expected_proba: np.ndarray | None = None,
) -> None:
    """Save a LightGBM test case."""
    output_dir = INFERENCE_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model.save_model(str(output_dir / "model.txt"))
    with open(output_dir / "model.json", "w") as f:
        json.dump(model.dump_model(), f, indent=2)

    # Input
    save_json(
        output_dir / "input.json",
        {
            "num_samples": x_test.shape[0],
            "num_features": x_test.shape[1],
            "data": x_test.tolist(),
            "labels": y_test.tolist() if y_test is not None else None,
        },
    )

    # Expected
    expected = {
        "raw": expected_raw.tolist()
        if expected_raw.ndim == 1
        else [row.tolist() for row in expected_raw]
    }
    if expected_proba is not None:
        expected["proba"] = (
            expected_proba.tolist()
            if expected_proba.ndim == 1
            else [row.tolist() for row in expected_proba]
        )
    save_json(output_dir / "expected.json", expected)

    print_success(f"lightgbm/{name}", x_test.shape[0])


def gen_regression() -> None:
    """Regression test case."""
    np.random.seed(42)
    x, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    x, y = x.astype(np.float64), y.astype(np.float64)

    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(x, label=y, params={"verbose": -1})
    model = lgb.train(params, train_data, num_boost_round=10)

    x_test = x[:10]
    y_test = y[:10]
    expected_raw = model.predict(x_test, raw_score=True)
    save_lgb_test_case("regression", model, x_test, y_test, expected_raw)


def gen_binary() -> None:
    """Binary classification test case."""
    np.random.seed(42)
    x, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
    x, y = x.astype(np.float64), y.astype(np.float64)

    params = {
        "objective": "binary",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(x, label=y, params={"verbose": -1})
    model = lgb.train(params, train_data, num_boost_round=10)

    x_test = x[:10]
    y_test = y[:10]
    expected_raw = model.predict(x_test, raw_score=True)
    expected_proba = model.predict(x_test)
    save_lgb_test_case("binary", model, x_test, y_test, expected_raw, expected_proba)


def gen_multiclass() -> None:
    """Multiclass classification test case."""
    np.random.seed(42)
    x, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    x, y = x.astype(np.float64), y.astype(np.float64)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "num_leaves": 31,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(x, label=y, params={"verbose": -1})
    model = lgb.train(params, train_data, num_boost_round=10)

    x_test = x[:10]
    y_test = y[:10]
    expected_raw = model.predict(x_test, raw_score=True)
    expected_proba = model.predict(x_test)
    save_lgb_test_case("multiclass", model, x_test, y_test, expected_raw, expected_proba)


def generate_all() -> None:
    """Generate all LightGBM test cases."""
    ensure_dirs()
    print("\n=== LightGBM Inference ===")
    gen_regression()
    gen_binary()
    gen_multiclass()
    print("\n✓ All LightGBM test cases generated")
