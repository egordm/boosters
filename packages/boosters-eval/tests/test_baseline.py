"""Tests for baseline and regression detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from boosters_eval.baseline import (
    SCHEMA_VERSION,
    Baseline,
    BaselineResult,
    MetricStats,
    check_baseline,
    is_regression,
    load_baseline,
    record_baseline,
)
from boosters_eval.results import BenchmarkResult, ResultCollection


def make_result(
    library: str = "boosters",
    config: str = "california/gbdt",
    dataset: str = "california",
    seed: int = 42,
    rmse: float = 0.5,
) -> BenchmarkResult:
    """Create a test result."""
    return BenchmarkResult(
        config_name=config,
        library=library,
        seed=seed,
        task="regression",
        booster_type="gbdt",
        dataset_name=dataset,
        metrics={"rmse": rmse, "mae": rmse * 0.8},
    )


class TestMetricStats:
    """Tests for MetricStats model."""

    def test_create_stats(self) -> None:
        """Test creating metric stats."""
        stats = MetricStats(mean=0.5, std=0.1, n=5)
        assert stats.mean == 0.5
        assert stats.std == 0.1
        assert stats.n == 5

    def test_frozen(self) -> None:
        """Test stats are immutable."""
        stats = MetricStats(mean=0.5, std=0.1, n=5)
        with pytest.raises(ValidationError):
            stats.mean = 0.6  # type: ignore[misc]


class TestBaseline:
    """Tests for Baseline model."""

    def test_create_baseline(self) -> None:
        """Test creating a baseline."""
        result = BaselineResult(
            config_name="test/gbdt",
            library="boosters",
            task="regression",
            booster_type="gbdt",
            dataset_name="california",
            metrics={"rmse": MetricStats(mean=0.5, std=0.1, n=3)},
        )
        baseline = Baseline(
            schema_version=SCHEMA_VERSION,
            created_at="2024-01-01T00:00:00Z",
            results=[result],
        )
        assert baseline.schema_version == SCHEMA_VERSION
        assert len(baseline.results) == 1

    def test_future_schema_rejected(self) -> None:
        """Test that future schema version is rejected."""
        with pytest.raises(ValidationError):
            Baseline(
                schema_version=99,
                created_at="2024-01-01T00:00:00Z",
                results=[],
            )

    def test_valid_baseline_loads(self) -> None:
        """Test loading valid baseline JSON."""
        data = {
            "schema_version": SCHEMA_VERSION,
            "created_at": "2024-01-01T00:00:00Z",
            "results": [
                {
                    "config_name": "test/gbdt",
                    "library": "boosters",
                    "task": "regression",
                    "booster_type": "gbdt",
                    "dataset_name": "california",
                    "metrics": {"rmse": {"mean": 0.5, "std": 0.1, "n": 3}},
                }
            ],
        }
        baseline = Baseline(**data)
        assert baseline.schema_version == SCHEMA_VERSION


class TestRecordBaseline:
    """Tests for record_baseline function."""

    def test_record_baseline(self) -> None:
        """Test recording baseline from results."""
        collection = ResultCollection()
        collection.add_result(make_result(library="boosters", seed=42, rmse=0.5))
        collection.add_result(make_result(library="boosters", seed=43, rmse=0.6))
        collection.add_result(make_result(library="boosters", seed=44, rmse=0.55))

        baseline = record_baseline(collection)

        assert baseline.schema_version == SCHEMA_VERSION
        assert len(baseline.results) == 1
        assert baseline.results[0].library == "boosters"

        rmse_stats = baseline.results[0].metrics["rmse"]
        assert rmse_stats.n == 3
        assert 0.5 < rmse_stats.mean < 0.6

    def test_record_multiple_libraries(self) -> None:
        """Test recording with multiple libraries."""
        collection = ResultCollection()
        collection.add_result(make_result(library="boosters", rmse=0.5))
        collection.add_result(make_result(library="xgboost", rmse=0.6))

        baseline = record_baseline(collection)

        assert len(baseline.results) == 2
        libraries = {r.library for r in baseline.results}
        assert libraries == {"boosters", "xgboost"}

    def test_save_baseline_file(self) -> None:
        """Test saving baseline to file."""
        collection = ResultCollection()
        collection.add_result(make_result())

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            baseline = record_baseline(collection, output_path=path)
            assert path.exists()

            # Load and verify
            loaded = load_baseline(path)
            assert loaded.schema_version == baseline.schema_version
            assert len(loaded.results) == 1
        finally:
            path.unlink()


class TestIsRegression:
    """Tests for is_regression function."""

    def test_lower_better_regression(self) -> None:
        """Test regression for lower-is-better metric (RMSE increased)."""
        # 3% increase with 2% tolerance = regression
        assert is_regression(current=0.515, baseline=0.5, metric="rmse", tolerance=0.02)

    def test_lower_better_no_regression(self) -> None:
        """Test no regression when within tolerance."""
        # 1% increase with 2% tolerance = no regression
        assert not is_regression(current=0.505, baseline=0.5, metric="rmse", tolerance=0.02)

    def test_higher_better_regression(self) -> None:
        """Test regression for higher-is-better metric (accuracy decreased)."""
        # 3% decrease with 2% tolerance = regression
        assert is_regression(current=0.97, baseline=1.0, metric="accuracy", tolerance=0.02)

    def test_higher_better_no_regression(self) -> None:
        """Test no regression when within tolerance."""
        # 1% decrease with 2% tolerance = no regression
        assert not is_regression(current=0.99, baseline=1.0, metric="accuracy", tolerance=0.02)


class TestCheckBaseline:
    """Tests for check_baseline function."""

    def test_regression_detected(self) -> None:
        """Test that regression is detected."""
        # Create baseline with RMSE = 0.5
        baseline = Baseline(
            schema_version=SCHEMA_VERSION,
            created_at="2024-01-01T00:00:00Z",
            results=[
                BaselineResult(
                    config_name="california/gbdt",
                    library="boosters",
                    task="regression",
                    booster_type="gbdt",
                    dataset_name="california",
                    metrics={"rmse": MetricStats(mean=0.5, std=0.02, n=5)},
                )
            ],
        )

        # Current results with RMSE = 0.55 (10% worse)
        collection = ResultCollection()
        collection.add_result(make_result(rmse=0.55))

        report = check_baseline(collection, baseline, tolerance=0.02)

        assert report.has_regressions
        assert len(report.regressions) == 1
        assert report.regressions[0]["metric"] == "rmse"

    def test_no_regression(self) -> None:
        """Test no regression when within tolerance."""
        baseline = Baseline(
            schema_version=SCHEMA_VERSION,
            created_at="2024-01-01T00:00:00Z",
            results=[
                BaselineResult(
                    config_name="california/gbdt",
                    library="boosters",
                    task="regression",
                    booster_type="gbdt",
                    dataset_name="california",
                    metrics={"rmse": MetricStats(mean=0.5, std=0.02, n=5)},
                )
            ],
        )

        # Current results with RMSE = 0.505 (1% worse, within 2% tolerance)
        collection = ResultCollection()
        collection.add_result(make_result(rmse=0.505))

        report = check_baseline(collection, baseline, tolerance=0.02)

        assert not report.has_regressions
        assert len(report.regressions) == 0

    def test_missing_config_handled(self) -> None:
        """Test that new configs not in baseline are skipped."""
        baseline = Baseline(
            schema_version=SCHEMA_VERSION,
            created_at="2024-01-01T00:00:00Z",
            results=[],  # Empty baseline
        )

        collection = ResultCollection()
        collection.add_result(make_result())

        # Should not crash
        report = check_baseline(collection, baseline)
        assert not report.has_regressions
