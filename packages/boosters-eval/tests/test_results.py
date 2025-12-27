"""Tests for result collection system."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from boosters_eval.results import (
    BenchmarkError,
    BenchmarkResult,
    ResultCollection,
    derive_seed,
)


def make_result(
    library: str = "boosters",
    dataset: str = "california",
    seed: int = 42,
    rmse: float = 0.5,
    train_time: float = 1.0,
) -> BenchmarkResult:
    """Helper to create test results."""
    return BenchmarkResult(
        config_name=f"{dataset}/gbdt",
        library=library,
        seed=seed,
        task="regression",
        booster_type="gbdt",
        dataset_name=dataset,
        metrics={"rmse": rmse, "mae": rmse * 0.8},
        train_time_s=train_time,
        predict_time_s=0.1,
    )


def make_error(
    library: str = "xgboost",
    dataset: str = "california",
    seed: int = 42,
) -> BenchmarkError:
    """Helper to create test errors."""
    return BenchmarkError(
        config_name=f"{dataset}/gbdt",
        library=library,
        seed=seed,
        error_type="ImportError",
        error_message="Module not found",
        dataset_name=dataset,
    )


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a basic result."""
        result = make_result()
        assert result.library == "boosters"
        assert result.dataset_name == "california"
        assert result.metrics["rmse"] == 0.5

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        result = make_result()
        d = result.to_dict()

        assert d["library"] == "boosters"
        assert d["dataset"] == "california"
        assert d["rmse"] == 0.5
        assert d["mae"] == 0.4
        assert d["train_time_s"] == 1.0

    def test_frozen(self) -> None:
        """Test result is immutable."""
        result = make_result()
        with pytest.raises(Exception):
            result.library = "xgboost"  # type: ignore[misc]


class TestBenchmarkError:
    """Tests for BenchmarkError dataclass."""

    def test_create_error(self) -> None:
        """Test creating an error."""
        error = make_error()
        assert error.error_type == "ImportError"
        assert error.library == "xgboost"

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        error = make_error()
        d = error.to_dict()

        assert d["error_type"] == "ImportError"
        assert d["error_message"] == "Module not found"


class TestDeriveSeed:
    """Tests for derive_seed function."""

    def test_deterministic(self) -> None:
        """Test that same inputs produce same output."""
        seed1 = derive_seed(42, "california/gbdt", "boosters")
        seed2 = derive_seed(42, "california/gbdt", "boosters")
        assert seed1 == seed2

    def test_different_base_seeds(self) -> None:
        """Test different base seeds produce different results."""
        seed1 = derive_seed(42, "california/gbdt", "boosters")
        seed2 = derive_seed(43, "california/gbdt", "boosters")
        assert seed1 != seed2

    def test_different_configs(self) -> None:
        """Test different configs produce different results."""
        seed1 = derive_seed(42, "california/gbdt", "boosters")
        seed2 = derive_seed(42, "iris/gbdt", "boosters")
        assert seed1 != seed2

    def test_different_libraries(self) -> None:
        """Test different libraries produce different results."""
        seed1 = derive_seed(42, "california/gbdt", "boosters")
        seed2 = derive_seed(42, "california/gbdt", "xgboost")
        assert seed1 != seed2

    def test_within_32_bit_range(self) -> None:
        """Test that result is within 32-bit unsigned int range."""
        seed = derive_seed(42, "california/gbdt", "boosters")
        assert 0 <= seed < 2**32


class TestResultCollection:
    """Tests for ResultCollection class."""

    def test_empty_collection(self) -> None:
        """Test empty collection."""
        collection = ResultCollection()
        assert len(collection) == 0
        assert collection.results == []
        assert collection.errors == []

    def test_add_result(self) -> None:
        """Test adding results."""
        collection = ResultCollection()
        collection.add_result(make_result())
        assert len(collection) == 1

    def test_add_error(self) -> None:
        """Test adding errors."""
        collection = ResultCollection()
        collection.add_error(make_error())
        assert len(collection.errors) == 1

    def test_to_dataframe(self) -> None:
        """Test converting to DataFrame."""
        collection = ResultCollection()
        collection.add_result(make_result(library="boosters", rmse=0.5))
        collection.add_result(make_result(library="xgboost", rmse=0.6))

        df = collection.to_dataframe()
        assert len(df) == 2
        assert "library" in df.columns
        assert "rmse" in df.columns

    def test_to_dataframe_empty(self) -> None:
        """Test DataFrame from empty collection."""
        collection = ResultCollection()
        df = collection.to_dataframe()
        assert df.empty

    def test_filter_by_library(self) -> None:
        """Test filtering by library."""
        collection = ResultCollection()
        collection.add_result(make_result(library="boosters"))
        collection.add_result(make_result(library="xgboost"))
        collection.add_result(make_result(library="lightgbm"))

        filtered = collection.filter(library="boosters")
        assert len(filtered) == 1
        assert filtered.results[0].library == "boosters"

    def test_filter_by_library_list(self) -> None:
        """Test filtering by multiple libraries."""
        collection = ResultCollection()
        collection.add_result(make_result(library="boosters"))
        collection.add_result(make_result(library="xgboost"))
        collection.add_result(make_result(library="lightgbm"))

        filtered = collection.filter(library=["boosters", "xgboost"])
        assert len(filtered) == 2

    def test_filter_by_dataset(self) -> None:
        """Test filtering by dataset."""
        collection = ResultCollection()
        collection.add_result(make_result(dataset="california"))
        collection.add_result(make_result(dataset="iris"))

        filtered = collection.filter(dataset="california")
        assert len(filtered) == 1

    def test_filter_chaining(self) -> None:
        """Test chaining multiple filters."""
        collection = ResultCollection()
        collection.add_result(make_result(library="boosters", dataset="california"))
        collection.add_result(make_result(library="boosters", dataset="iris"))
        collection.add_result(make_result(library="xgboost", dataset="california"))

        filtered = collection.filter(library="boosters").filter(dataset="california")
        assert len(filtered) == 1

    def test_summary(self) -> None:
        """Test summary aggregation."""
        collection = ResultCollection()
        # Add multiple seeds for same config
        collection.add_result(make_result(library="boosters", seed=42, rmse=0.5))
        collection.add_result(make_result(library="boosters", seed=43, rmse=0.6))
        collection.add_result(make_result(library="boosters", seed=44, rmse=0.55))

        summary = collection.summary()
        assert len(summary) == 1
        assert "rmse_mean" in summary.columns
        assert "rmse_std" in summary.columns

    def test_to_markdown(self) -> None:
        """Test markdown output."""
        collection = ResultCollection()
        collection.add_result(make_result(library="boosters", rmse=0.5))

        markdown = collection.to_markdown()
        assert isinstance(markdown, str)
        assert "boosters" in markdown or "library" in markdown.lower()

    def test_to_markdown_empty(self) -> None:
        """Test markdown from empty collection."""
        collection = ResultCollection()
        markdown = collection.to_markdown()
        assert "No results" in markdown

    def test_to_json(self) -> None:
        """Test JSON export."""
        collection = ResultCollection()
        collection.add_result(make_result())
        collection.add_error(make_error())

        json_str = collection.to_json()
        data = json.loads(json_str)

        assert "results" in data
        assert "errors" in data
        assert len(data["results"]) == 1
        assert len(data["errors"]) == 1

    def test_to_csv(self) -> None:
        """Test CSV export."""
        collection = ResultCollection()
        collection.add_result(make_result())

        csv = collection.to_csv()
        assert "library" in csv
        assert "boosters" in csv

    def test_from_json_roundtrip(self) -> None:
        """Test JSON roundtrip."""
        original = ResultCollection()
        original.add_result(make_result(library="boosters", rmse=0.5))
        original.add_result(make_result(library="xgboost", rmse=0.6))
        original.add_error(make_error())

        json_str = original.to_json()
        restored = ResultCollection.from_json(json_str)

        assert len(restored) == 2
        assert len(restored.errors) == 1
        assert restored.results[0].library == original.results[0].library


class TestSummaryByTask:
    """Tests for task-grouped summary functionality."""

    def make_regression_result(
        self,
        library: str = "boosters",
        dataset: str = "california",
        seed: int = 42,
        rmse: float = 0.5,
    ) -> BenchmarkResult:
        """Create a regression benchmark result."""
        return BenchmarkResult(
            config_name=f"{dataset}/gbdt",
            library=library,
            seed=seed,
            task="regression",
            booster_type="gbdt",
            dataset_name=dataset,
            metrics={"rmse": rmse, "mae": rmse * 0.8, "r2": 0.9 - rmse},
            train_time_s=1.0,
            predict_time_s=0.1,
        )

    def make_binary_result(
        self,
        library: str = "boosters",
        dataset: str = "breast_cancer",
        seed: int = 42,
        logloss: float = 0.1,
    ) -> BenchmarkResult:
        """Create a binary classification benchmark result."""
        return BenchmarkResult(
            config_name=f"{dataset}/gbdt",
            library=library,
            seed=seed,
            task="binary",
            booster_type="gbdt",
            dataset_name=dataset,
            metrics={"logloss": logloss, "accuracy": 0.95},
            train_time_s=0.5,
            predict_time_s=0.05,
        )

    def test_summary_by_task_separates_tasks(self) -> None:
        """Test that summary_by_task separates results by task type."""
        from boosters_eval.config import Task

        collection = ResultCollection()
        collection.add_result(self.make_regression_result())
        collection.add_result(self.make_binary_result())

        summaries = collection.summary_by_task()

        assert Task.REGRESSION in summaries
        assert Task.BINARY in summaries
        assert Task.MULTICLASS not in summaries

    def test_summary_by_task_only_relevant_metrics(self) -> None:
        """Test that each task summary only includes relevant metrics."""
        from boosters_eval.config import Task

        collection = ResultCollection()
        collection.add_result(self.make_regression_result())
        collection.add_result(self.make_binary_result())

        summaries = collection.summary_by_task()

        reg_df = summaries[Task.REGRESSION]
        assert "rmse_mean" in reg_df.columns
        assert "logloss_mean" not in reg_df.columns

        bin_df = summaries[Task.BINARY]
        assert "logloss_mean" in bin_df.columns
        assert "rmse_mean" not in bin_df.columns

    def test_format_summary_table_highlights_best(self) -> None:
        """Test that format_summary_table highlights best values."""
        from boosters_eval.config import Task

        collection = ResultCollection()
        # Add results with different values
        collection.add_result(self.make_regression_result(library="boosters", rmse=0.4))
        collection.add_result(self.make_regression_result(library="xgboost", rmse=0.5))
        collection.add_result(self.make_regression_result(library="lightgbm", rmse=0.6))

        table = collection.format_summary_table(Task.REGRESSION, highlight_best=True)

        # Best rmse (0.4) from boosters should be bold
        assert "**0.4000**" in table
        # xgboost rmse (0.5) should not be bold
        lines = table.split("\n")
        xgboost_line = [l for l in lines if "xgboost" in l][0]
        # Check that 0.5000 appears without bold markers in xgboost line
        assert "| 0.5000 |" in xgboost_line or "| 0.5000±" in xgboost_line

    def test_to_markdown_groups_by_task(self) -> None:
        """Test that to_markdown creates sections per task."""
        collection = ResultCollection()
        collection.add_result(self.make_regression_result())
        collection.add_result(self.make_binary_result())

        markdown = collection.to_markdown()

        assert "### Regression" in markdown
        assert "### Binary Classification" in markdown
        assert "### Multiclass Classification" not in markdown
