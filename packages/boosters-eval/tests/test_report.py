"""Tests for report generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from boosters_eval.report import (
    LibraryVersions,
    MachineInfo,
    ReportMetadata,
    generate_report,
    get_library_versions,
    get_machine_info,
    is_significant,
    render_report,
)
from boosters_eval.results import BenchmarkResult, ResultCollection


def make_result(library: str = "boosters", rmse: float = 0.5, seed: int = 42) -> BenchmarkResult:
    """Create a test result."""
    return BenchmarkResult(
        config_name="test/gbdt",
        library=library,
        seed=seed,
        task="regression",
        booster_type="gbdt",
        dataset_name="california",
        metrics={"rmse": rmse, "mae": rmse * 0.8},
    )


class TestMachineInfo:
    """Tests for machine info collection."""

    def test_get_machine_info(self) -> None:
        """Test collecting machine info."""
        info = get_machine_info()

        assert isinstance(info, MachineInfo)
        assert info.cores >= 1
        assert info.memory_gb > 0

    def test_machine_info_fields(self) -> None:
        """Test all fields are populated."""
        info = get_machine_info()

        # CPU might be "Unknown" on some systems but should be a string
        assert isinstance(info.cpu, str)
        assert isinstance(info.os, str)

    def test_machine_info_frozen(self) -> None:
        """Test machine info is immutable."""
        info = get_machine_info()
        with pytest.raises(Exception):
            info.cores = 99  # type: ignore[misc]


class TestLibraryVersions:
    """Tests for library version collection."""

    def test_get_library_versions(self) -> None:
        """Test collecting library versions."""
        versions = get_library_versions()

        assert isinstance(versions, LibraryVersions)
        assert versions.python  # Python version should always be present

    def test_library_versions_fields(self) -> None:
        """Test library version fields."""
        versions = get_library_versions()

        # At least numpy should be installed
        assert versions.numpy is not None


class TestIsSignificant:
    """Tests for statistical significance testing."""

    def test_significant_difference(self) -> None:
        """Test detecting significant difference."""
        # Very different distributions
        values1 = [0.5, 0.51, 0.49, 0.5, 0.52]
        values2 = [0.9, 0.91, 0.89, 0.9, 0.92]

        assert is_significant(values1, values2)

    def test_non_significant_difference(self) -> None:
        """Test non-significant difference (similar values)."""
        values1 = [0.5, 0.51, 0.49, 0.52, 0.48]
        values2 = [0.51, 0.50, 0.52, 0.49, 0.50]

        # Similar distributions - not significant
        assert not is_significant(values1, values2)

    def test_single_sample_not_significant(self) -> None:
        """Test that single samples return False (can't compute t-test)."""
        assert not is_significant([0.5], [0.9])

    def test_identical_values(self) -> None:
        """Test identical values within group."""
        # Different between groups but no variance within
        assert is_significant([0.5, 0.5, 0.5], [0.9, 0.9, 0.9])

        # Same between and within
        assert not is_significant([0.5, 0.5], [0.5, 0.5])


class TestRenderReport:
    """Tests for report rendering."""

    def test_render_report_structure(self) -> None:
        """Test report has expected sections."""
        collection = ResultCollection()
        collection.add_result(make_result())

        machine = MachineInfo(
            cpu="Test CPU",
            cores=4,
            memory_gb=16.0,
            os="Test OS",
        )
        library_versions = LibraryVersions(
            python="3.12.0",
            boosters="0.1.0",
        )

        metadata = ReportMetadata(
            title="Test Report",
            created_at="2024-01-01T00:00:00",
            git_sha="abc123",
            machine=machine,
            library_versions=library_versions,
            suite_name="quick",
            n_seeds=3,
        )

        report = render_report(collection, metadata)

        # Check sections exist
        assert "# Test Report" in report
        assert "## Environment" in report
        assert "## Configuration" in report
        assert "## Results" in report
        assert "## Reproducing" in report

        # Check metadata
        assert "Test CPU" in report
        assert "abc123" in report
        assert "0.1.0" in report

    def test_no_placeholders(self) -> None:
        """Test report has no unfilled placeholders."""
        collection = ResultCollection()
        collection.add_result(make_result())

        machine = get_machine_info()
        library_versions = get_library_versions()
        metadata = ReportMetadata(
            title="Test",
            created_at="2024-01-01T00:00:00",
            machine=machine,
            library_versions=library_versions,
            suite_name="quick",
            n_seeds=1,
        )

        report = render_report(collection, metadata)

        # No placeholder tokens
        assert "{" not in report or "```" in report  # Allow code blocks


class TestGenerateReport:
    """Tests for report generation."""

    def test_generate_report_returns_markdown(self) -> None:
        """Test generate_report returns markdown string."""
        collection = ResultCollection()
        collection.add_result(make_result())

        report = generate_report(collection, suite_name="quick")

        assert isinstance(report, str)
        assert "# Benchmark Report" in report

    def test_generate_report_saves_file(self) -> None:
        """Test generate_report saves to file."""
        collection = ResultCollection()
        collection.add_result(make_result())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"

            report = generate_report(collection, suite_name="quick", output_path=path)

            assert path.exists()
            assert path.read_text() == report

    def test_generate_report_custom_title(self) -> None:
        """Test generate_report with custom title."""
        collection = ResultCollection()
        collection.add_result(make_result())

        report = generate_report(
            collection, suite_name="full", title="My Custom Report"
        )

        assert "# My Custom Report" in report
