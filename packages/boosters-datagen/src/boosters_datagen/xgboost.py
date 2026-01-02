"""XGBoost test case generation.

Generates test cases for XGBoost inference and training validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

from boosters_datagen.utils import (
    nan_to_null,
    print_success,
    save_json,
    save_test_expected,
    save_test_input,
)

# Output directories
BASE_DIR = Path(__file__).parents[4] / "tests" / "test-cases" / "xgboost"

GBTREE_INFERENCE_DIR = BASE_DIR / "gbtree" / "inference"
GBLINEAR_INFERENCE_DIR = BASE_DIR / "gblinear" / "inference"
DART_INFERENCE_DIR = BASE_DIR / "dart" / "inference"
GBTREE_TRAINING_DIR = BASE_DIR / "gbtree" / "training"
GBLINEAR_TRAINING_DIR = BASE_DIR / "gblinear" / "training"
WEIGHTED_TRAINING_DIR = BASE_DIR / "gbtree" / "training" / "weighted"


def ensure_dirs() -> None:
    """Create output directories."""
    for d in [
        GBTREE_INFERENCE_DIR,
        GBLINEAR_INFERENCE_DIR,
        DART_INFERENCE_DIR,
        GBTREE_TRAINING_DIR,
        GBLINEAR_TRAINING_DIR,
        WEIGHTED_TRAINING_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def save_xgb_test_case(
    name: str,
    booster: xgb.Booster,
    x_test: np.ndarray,
    objective: str,
    num_class: int = 0,
    output_dir: Path = GBTREE_INFERENCE_DIR,
    dtest: xgb.DMatrix | None = None,
    feature_types: list[str] | None = None,
) -> None:
    """Save a complete XGBoost test case."""
    booster.save_model(str(output_dir / f"{name}.model.json"))
    save_test_input(output_dir / f"{name}.input.json", x_test, feature_types)

    if dtest is None:
        dtest = xgb.DMatrix(x_test)
    raw = booster.predict(dtest, output_margin=True)
    transformed = booster.predict(dtest, output_margin=False)
    save_test_expected(output_dir / f"{name}.expected.json", raw, transformed, objective, num_class)
    print_success(name, len(x_test))


def save_training_case(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    booster: xgb.Booster,
    config: dict,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    output_dir: Path = GBTREE_TRAINING_DIR,
) -> None:
    """Save a complete training test case."""
    # Training data
    save_json(
        output_dir / f"{name}.train_data.json",
        {
            "num_rows": x_train.shape[0],
            "num_features": x_train.shape[1],
            "data": nan_to_null(x_train.flatten().tolist()),
        },
    )
    save_json(output_dir / f"{name}.train_labels.json", {"labels": y_train.tolist()})

    # Model
    booster.save_model(str(output_dir / f"{name}.model.json"))

    # Config
    save_json(output_dir / f"{name}.config.json", config)

    # Training predictions
    dtrain = xgb.DMatrix(x_train, label=y_train)
    train_preds = booster.predict(dtrain, output_margin=True)
    save_json(output_dir / f"{name}.train_predictions.json", {"predictions": train_preds.tolist()})

    # Test data if provided
    if x_test is not None and y_test is not None:
        save_json(
            output_dir / f"{name}.test_data.json",
            {
                "num_rows": x_test.shape[0],
                "num_features": x_test.shape[1],
                "data": nan_to_null(x_test.flatten().tolist()),
            },
        )
        save_json(output_dir / f"{name}.test_labels.json", {"labels": y_test.tolist()})

        dtest = xgb.DMatrix(x_test)
        test_preds = booster.predict(dtest, output_margin=True)
        save_json(output_dir / f"{name}.test_predictions.json", {"predictions": test_preds.tolist()})

        # Metrics
        metrics: dict[str, int | float] = {"num_trees": int(booster.num_boosted_rounds())}
        if test_preds.ndim == 1:
            rmse = float(np.sqrt(np.mean((test_preds - y_test) ** 2)))
            metrics["test_rmse"] = rmse
        else:
            pred_classes = np.argmax(test_preds, axis=1)
            accuracy = float(np.mean(pred_classes == y_test))
            metrics["test_accuracy"] = accuracy
        save_json(output_dir / f"{name}.metrics.json", metrics)

    print_success(f"training/{name}", len(x_train))


# =============================================================================
# GBTree Inference Cases
# =============================================================================


def gen_regression() -> None:
    """Simple regression."""
    x, y = make_regression(  # pyright: ignore[reportAssignmentType]
        n_samples=100, n_features=5, noise=10.0, random_state=42
    )
    x_test = x[:10]

    dtrain = xgb.DMatrix(x, label=y)
    params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.1, "booster": "gbtree"}
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case("gbtree_regression", booster, x_test, "reg:squarederror")


def gen_binary() -> None:
    """Binary classification."""
    x, y = make_classification(n_samples=100, n_features=4, n_informative=3, random_state=42)
    x_test = x[:10]

    dtrain = xgb.DMatrix(x, label=y)
    params = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.15, "booster": "gbtree"}
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case("gbtree_binary_logistic", booster, x_test, "binary:logistic")


def gen_binary_missing() -> None:
    """Binary classification with missing values."""
    x, y = make_classification(n_samples=100, n_features=4, n_informative=3, random_state=42)
    x_test = x[:10].copy()
    x_test[0, 0] = np.nan
    x_test[2, 1] = np.nan
    x_test[4, :2] = np.nan

    dtrain = xgb.DMatrix(x, label=y)
    params = {"objective": "binary:logistic", "max_depth": 2, "eta": 0.15, "booster": "gbtree"}
    booster = xgb.train(params, dtrain, num_boost_round=5)
    save_xgb_test_case("gbtree_binary_logistic_missing", booster, x_test, "binary:logistic")


def gen_multiclass() -> None:
    """Multiclass classification."""
    x, y = make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    x_test = x[:10]

    dtrain = xgb.DMatrix(x, label=y)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 3,
        "eta": 0.1,
        "booster": "gbtree",
    }
    booster = xgb.train(params, dtrain, num_boost_round=10)
    save_xgb_test_case("gbtree_multiclass", booster, x_test, "multi:softprob", num_class=3)


# =============================================================================
# GBLinear Inference Cases
# =============================================================================


def gen_gblinear_regression() -> None:
    """GBLinear regression."""
    x, y = make_regression(  # pyright: ignore[reportAssignmentType]
        n_samples=100, n_features=5, noise=10.0, random_state=42
    )
    x_test = x[:10]

    dtrain = xgb.DMatrix(x, label=y)
    params = {"objective": "reg:squarederror", "booster": "gblinear", "eta": 0.5}
    booster = xgb.train(params, dtrain, num_boost_round=20)
    save_xgb_test_case(
        "gblinear_regression",
        booster,
        x_test,
        "reg:squarederror",
        output_dir=GBLINEAR_INFERENCE_DIR,
    )


def gen_gblinear_binary() -> None:
    """GBLinear binary classification."""
    x, y = make_classification(n_samples=100, n_features=4, n_informative=3, random_state=42)
    x_test = x[:10]

    dtrain = xgb.DMatrix(x, label=y)
    params = {"objective": "binary:logistic", "booster": "gblinear", "eta": 0.5}
    booster = xgb.train(params, dtrain, num_boost_round=20)
    save_xgb_test_case("gblinear_binary", booster, x_test, "binary:logistic", output_dir=GBLINEAR_INFERENCE_DIR)


# =============================================================================
# Training Cases
# =============================================================================


def gen_training_regression() -> None:
    """Regression training case."""
    x, y = make_regression(  # pyright: ignore[reportAssignmentType]
        n_samples=200, n_features=10, noise=10.0, random_state=42
    )
    x, y = x.astype(np.float32), y.astype(np.float32)
    x_train, x_test = x[:160], x[160:]
    y_train, y_test = y[:160], y[160:]

    dtrain = xgb.DMatrix(x_train, label=y_train)
    config = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.1,
        "lambda": 1.0,
        "alpha": 0.0,
        "min_child_weight": 1.0,
        "num_boost_round": 30,
    }
    params = {k: v for k, v in config.items() if k != "num_boost_round"}
    booster = xgb.train(params, dtrain, num_boost_round=config["num_boost_round"])
    save_training_case("regression", x_train, y_train, booster, config, x_test, y_test)


def gen_training_binary() -> None:
    """Binary classification training case."""
    x, y = make_classification(n_samples=200, n_features=10, n_informative=6, random_state=42)
    x, y = x.astype(np.float32), y.astype(np.float32)
    x_train, x_test = x[:160], x[160:]
    y_train, y_test = y[:160], y[160:]

    dtrain = xgb.DMatrix(x_train, label=y_train)
    config = {
        "objective": "binary:logistic",
        "max_depth": 4,
        "eta": 0.1,
        "lambda": 1.0,
        "alpha": 0.0,
        "min_child_weight": 1.0,
        "num_boost_round": 30,
    }
    params = {k: v for k, v in config.items() if k != "num_boost_round"}
    booster = xgb.train(params, dtrain, num_boost_round=config["num_boost_round"])
    save_training_case("binary", x_train, y_train, booster, config, x_test, y_test)


# =============================================================================
# Main
# =============================================================================


def generate_all() -> None:
    """Generate all XGBoost test cases."""
    from boosters_datagen.utils import console

    ensure_dirs()
    console.print("\n[bold]=== GBTree Inference ===[/bold]")
    gen_regression()
    gen_binary()
    gen_binary_missing()
    gen_multiclass()

    console.print("\n[bold]=== GBLinear Inference ===[/bold]")
    gen_gblinear_regression()
    gen_gblinear_binary()

    console.print("\n[bold]=== Training ===[/bold]")
    gen_training_regression()
    gen_training_binary()

    console.print("\n[green]✓ All XGBoost test cases generated[/green]")
