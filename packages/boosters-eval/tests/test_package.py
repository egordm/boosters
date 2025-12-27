"""Tests for package structure and imports."""

from __future__ import annotations


def test_package_imports() -> None:
    """Test that the main public API imports work."""
    from boosters_eval import (
        DATASETS,
        FULL_SUITE,
        QUICK_SUITE,
        get_available_runners,
    )

    # Verify suite configs exist
    assert QUICK_SUITE.name == "quick"
    assert FULL_SUITE.name == "full"

    # Verify DATASETS registry has entries
    assert len(DATASETS) >= 9

    # Verify all runners available (xgboost/lightgbm are mandatory now)
    runners = get_available_runners()
    assert "boosters" in runners
    assert "xgboost" in runners
    assert "lightgbm" in runners


def test_result_collection() -> None:
    """Test ResultCollection basic operations."""
    from boosters_eval import BenchmarkResult, ResultCollection

    collection = ResultCollection()
    assert len(collection) == 0

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

    # Test filtering
    filtered = collection.filter(library="boosters")
    assert len(filtered) == 1

    filtered_empty = collection.filter(library="nonexistent")
    assert len(filtered_empty) == 0
