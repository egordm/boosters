"""Tests for CLI commands."""

from __future__ import annotations

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from boosters_eval.cli import app

runner = CliRunner(env={"NO_COLOR": "1"})


class TestCliHelp:
    """Tests for CLI help output."""

    def test_main_help(self) -> None:
        """Test main help displays."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "boosters-eval" in result.stdout or "Evaluate" in result.stdout

    def test_quick_help(self) -> None:
        """Test quick command help."""
        result = runner.invoke(app, ["quick", "--help"])
        assert result.exit_code == 0
        assert "quick" in result.stdout.lower()

    def test_compare_help(self) -> None:
        """Test compare command help."""
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.stdout.lower()

    def test_baseline_help(self) -> None:
        """Test baseline subcommand help."""
        result = runner.invoke(app, ["baseline", "--help"])
        assert result.exit_code == 0
        assert "record" in result.stdout
        assert "check" in result.stdout


class TestListCommands:
    """Tests for list commands."""

    def test_list_datasets(self) -> None:
        """Test list-datasets command."""
        result = runner.invoke(app, ["list-datasets"])
        assert result.exit_code == 0
        assert "california" in result.stdout
        assert "regression" in result.stdout.lower()

    def test_list_libraries(self) -> None:
        """Test list-libraries command."""
        result = runner.invoke(app, ["list-libraries"])
        assert result.exit_code == 0
        assert "boosters" in result.stdout
        assert "xgboost" in result.stdout
        assert "lightgbm" in result.stdout

    def test_list_tasks(self) -> None:
        """Test list-tasks command."""
        result = runner.invoke(app, ["list-tasks"])
        assert result.exit_code == 0
        assert "regression" in result.stdout.lower()
        assert "binary" in result.stdout.lower()


class TestCompareCommand:
    """Tests for compare command."""

    def test_compare_minimal(self) -> None:
        """Test compare with minimal options."""
        result = runner.invoke(
            app,
            [
                "compare",
                "-d",
                "synthetic_reg_small",
                "-l",
                "boosters",
                "--trees",
                "5",
                "--seeds",
                "1",
            ],
        )
        # Should complete without error
        assert result.exit_code == 0

    def test_compare_invalid_dataset(self) -> None:
        """Test compare with invalid dataset."""
        result = runner.invoke(
            app,
            [
                "compare",
                "-d",
                "nonexistent",
                "-l",
                "boosters",
            ],
        )
        # Should exit with error
        assert result.exit_code == 1

    def test_compare_invalid_booster(self) -> None:
        """Test compare with invalid booster type."""
        result = runner.invoke(
            app,
            [
                "compare",
                "-d",
                "synthetic_reg_small",
                "--booster",
                "invalid_type",
            ],
        )
        assert result.exit_code == 1


class TestBaselineCommands:
    """Tests for baseline record and check commands."""

    def test_baseline_record_help(self) -> None:
        """Test baseline record help."""
        result = runner.invoke(app, ["baseline", "record", "--help"])
        assert result.exit_code == 0
        assert "output" in result.stdout.lower()

    def test_baseline_check_help(self) -> None:
        """Test baseline check help."""
        result = runner.invoke(app, ["baseline", "check", "--help"])
        assert result.exit_code == 0
        assert "tolerance" in result.stdout.lower()

    def test_baseline_check_missing_file(self) -> None:
        """Test baseline check with missing file."""
        result = runner.invoke(app, ["baseline", "check", "/nonexistent/baseline.json"])
        assert result.exit_code == 2
        assert "not found" in result.stdout.lower()

    def test_baseline_record_invalid_suite(self) -> None:
        """Test baseline record with invalid suite."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            result = runner.invoke(app, ["baseline", "record", "-o", str(path), "-s", "invalid"])
            assert result.exit_code == 1
        finally:
            if path.exists():
                path.unlink()


class TestReportCommand:
    """Tests for report command."""

    def test_report_invalid_suite(self) -> None:
        """Test report with invalid suite."""
        result = runner.invoke(app, ["report", "-s", "invalid"])
        assert result.exit_code == 1
