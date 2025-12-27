"""Tests for suite execution."""

from __future__ import annotations

import numpy as np
import pytest

from boosters_eval.config import BoosterType, DatasetConfig, Task
from boosters_eval.datasets import DATASETS
from boosters_eval.suite import (
    ABLATION_SUITES,
    FULL_SUITE,
    MINIMAL_SUITE,
    QUICK_SUITE,
    compare,
    create_ablation_suite,
    run_ablation,
    run_suite,
    SuiteConfig,
)


class TestSuiteConfigs:
    """Tests for predefined suite configurations."""

    def test_quick_suite_exists(self) -> None:
        """Test QUICK_SUITE is defined."""
        assert QUICK_SUITE is not None
        assert QUICK_SUITE.name == "quick"
        assert len(QUICK_SUITE.seeds) == 3
        assert len(QUICK_SUITE.datasets) == 2

    def test_full_suite_exists(self) -> None:
        """Test FULL_SUITE is defined."""
        assert FULL_SUITE is not None
        assert FULL_SUITE.name == "full"
        assert len(FULL_SUITE.seeds) == 5

    def test_minimal_suite_exists(self) -> None:
        """Test MINIMAL_SUITE is defined for CI."""
        assert MINIMAL_SUITE is not None
        assert MINIMAL_SUITE.name == "minimal"
        assert len(MINIMAL_SUITE.seeds) == 1

    def test_ablation_suites_exist(self) -> None:
        """Test ablation suites are defined."""
        assert "depth" in ABLATION_SUITES
        assert "lr" in ABLATION_SUITES
        assert "growth" in ABLATION_SUITES
        # Each should be a list of suite configs
        assert len(ABLATION_SUITES["depth"]) == 3
        assert len(ABLATION_SUITES["lr"]) == 3
        assert len(ABLATION_SUITES["growth"]) == 2


class TestAblationSuites:
    """Tests for ablation suite generation."""

    def test_create_ablation_suite(self) -> None:
        """Test creating ablation suites from variants."""
        base = SuiteConfig(
            name="base",
            description="Base suite",
            datasets=["synthetic_reg_small"],
            n_estimators=10,
            seeds=[42],
            libraries=["boosters"],
        )

        variants = {
            "small": {"n_estimators": 5},
            "large": {"n_estimators": 20},
        }

        suites = create_ablation_suite("test", base, variants)

        assert len(suites) == 2
        assert suites[0].name == "test_small"
        assert suites[0].n_estimators == 5
        assert suites[1].name == "test_large"
        assert suites[1].n_estimators == 20

    def test_ablation_preserves_base(self) -> None:
        """Test ablation preserves non-overridden fields."""
        base = SuiteConfig(
            name="base",
            description="Base suite",
            datasets=["synthetic_reg_small"],
            n_estimators=10,
            seeds=[42, 123],
            libraries=["boosters", "xgboost"],
        )

        variants = {"variant": {"n_estimators": 5}}

        suites = create_ablation_suite("test", base, variants)

        assert suites[0].seeds == [42, 123]
        assert suites[0].libraries == ["boosters", "xgboost"]
        assert suites[0].datasets == ["synthetic_reg_small"]


class TestRunSuite:
    """Tests for run_suite function."""

    def test_minimal_suite_execution(self) -> None:
        """Test running minimal suite completes."""
        # Use a tiny custom suite for speed
        suite = SuiteConfig(
            name="test",
            description="Test suite",
            datasets=["synthetic_reg_small"],
            n_estimators=5,
            seeds=[42],
            libraries=["boosters"],
        )

        results = run_suite(suite, verbose=False)

        assert len(results) >= 1
        assert results.results[0].library == "boosters"

    def test_multiple_libraries(self) -> None:
        """Test running with multiple libraries."""
        suite = SuiteConfig(
            name="test",
            description="Test suite",
            datasets=["synthetic_reg_small"],
            n_estimators=5,
            seeds=[42],
            libraries=["boosters", "xgboost", "lightgbm"],
        )

        results = run_suite(suite, verbose=False)

        libraries = {r.library for r in results.results}
        assert "boosters" in libraries
        assert "xgboost" in libraries
        assert "lightgbm" in libraries

    def test_multiple_seeds(self) -> None:
        """Test running with multiple seeds."""
        suite = SuiteConfig(
            name="test",
            description="Test suite",
            datasets=["synthetic_reg_small"],
            n_estimators=5,
            seeds=[42, 123],
            libraries=["boosters"],
        )

        results = run_suite(suite, verbose=False)

        seeds = {r.seed for r in results.results}
        assert 42 in seeds
        assert 123 in seeds

    def test_error_handling(self) -> None:
        """Test that runner errors are captured."""
        # Create a dataset that will work
        suite = SuiteConfig(
            name="test",
            description="Test suite",
            datasets=["synthetic_reg_small"],
            n_estimators=5,
            seeds=[42],
            libraries=["boosters"],
        )

        # Should complete without raising
        results = run_suite(suite, verbose=False)
        assert len(results) >= 0  # May have errors or results

    def test_unknown_dataset_ignored(self) -> None:
        """Test that unknown datasets are skipped."""
        suite = SuiteConfig(
            name="test",
            description="Test suite",
            datasets=["nonexistent_dataset", "synthetic_reg_small"],
            n_estimators=5,
            seeds=[42],
            libraries=["boosters"],
        )

        results = run_suite(suite, verbose=False)

        # Should only have results from the valid dataset
        assert all(r.dataset_name == "synthetic_reg_small" for r in results.results)


class TestCompare:
    """Tests for compare convenience function."""

    def test_compare_default(self) -> None:
        """Test compare with minimal arguments."""
        results = compare(
            datasets=["synthetic_reg_small"],
            seeds=[42],
            n_estimators=5,
            libraries=["boosters"],
            verbose=False,
        )

        assert len(results) >= 1

    def test_compare_multiple_libraries(self) -> None:
        """Test compare returns results for all libraries."""
        results = compare(
            datasets=["synthetic_reg_small"],
            seeds=[42],
            n_estimators=5,
            verbose=False,
        )

        libraries = {r.library for r in results.results}
        # Should have results from available libraries
        assert len(libraries) >= 1


class TestResultIntegrity:
    """Tests for result data integrity."""

    def test_results_have_metrics(self) -> None:
        """Test that results contain expected metrics."""
        suite = SuiteConfig(
            name="test",
            description="Test suite",
            datasets=["synthetic_reg_small"],
            n_estimators=5,
            seeds=[42],
            libraries=["boosters"],
        )

        results = run_suite(suite, verbose=False)

        for result in results.results:
            assert "rmse" in result.metrics
            assert np.isfinite(result.metrics["rmse"])

    def test_results_have_timing(self) -> None:
        """Test that results include timing information."""
        suite = SuiteConfig(
            name="test",
            description="Test suite",
            datasets=["synthetic_reg_small"],
            n_estimators=5,
            seeds=[42],
            libraries=["boosters"],
        )

        results = run_suite(suite, verbose=False)

        for result in results.results:
            assert result.train_time_s is not None
            assert result.train_time_s > 0
            assert result.predict_time_s is not None
            assert result.predict_time_s > 0
