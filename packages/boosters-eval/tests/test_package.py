"""Tests for package structure and imports."""

from __future__ import annotations


def test_package_imports() -> None:
    """Test that the main package imports work."""
    from boosters_eval import (
        DATASETS,
        FULL_SUITE,
        QUICK_SUITE,
        BenchmarkConfig,
        BoosterType,
        DatasetConfig,
        ResultCollection,
        Task,
        TrainingConfig,
        compare,
        run_suite,
    )

    # Verify core types exist
    assert Task.REGRESSION is not None
    assert Task.BINARY is not None
    assert Task.MULTICLASS is not None

    # Verify suite configs exist
    assert QUICK_SUITE.name == "quick"
    assert FULL_SUITE.name == "full"

    # Verify DATASETS registry
    assert "california" in DATASETS
    assert "breast_cancer" in DATASETS

    # Verify callables
    assert callable(compare)
    assert callable(run_suite)


def test_config_dataclasses() -> None:
    """Test that config Pydantic models work correctly."""
    from boosters_eval import (
        BoosterType,
        DatasetConfig,
        Task,
        TrainingConfig,
    )

    # TrainingConfig with defaults
    tc = TrainingConfig()
    assert tc.n_estimators == 100
    assert tc.max_depth == 6
    assert tc.learning_rate == 0.1

    # TrainingConfig with custom values
    tc2 = TrainingConfig(n_estimators=50, max_depth=8)
    assert tc2.n_estimators == 50
    assert tc2.max_depth == 8


def test_metrics_imports() -> None:
    """Test that metrics functions are available."""
    from boosters_eval import (
        compute_metrics,
        is_lower_better,
        primary_metric,
    )

    from boosters_eval.config import Task

    # Check primary metric mapping
    assert primary_metric(Task.REGRESSION) == "rmse"
    assert primary_metric(Task.BINARY) == "logloss"
    assert primary_metric(Task.MULTICLASS) == "mlogloss"

    # Check lower is better
    assert is_lower_better("rmse") is True
    assert is_lower_better("accuracy") is False


def test_runner_discovery() -> None:
    """Test that runner discovery works."""
    from boosters_eval import get_available_runners

    runners = get_available_runners()
    assert isinstance(runners, list)
    # At minimum boosters should be available in this env
    # Don't assert specific runners since xgboost/lightgbm may not be installed


def test_result_collection() -> None:
    """Test ResultCollection basic operations."""
    from boosters_eval import BenchmarkResult, ResultCollection

    collection = ResultCollection()
    assert len(collection) == 0
    assert collection.results == []
    assert collection.errors == []

    # Add a result
    result = BenchmarkResult(
        config_name="test",
        library="boosters",
        seed=42,
        task="regression",
        booster_type="gbdt",
        dataset_name="california",
        metrics={"rmse": 0.5, "mae": 0.3},
        train_time_s=1.0,
        predict_time_s=0.1,
    )
    collection.add_result(result)
    assert len(collection) == 1


def test_cli_app_exists() -> None:
    """Test that CLI app can be imported."""
    from boosters_eval.cli import app, main

    assert app is not None
    assert callable(main)
