"""Tests for Metric (evaluation) validation.

Tests focus on validation behavior—PyO3 enum mechanics are not tested
since that's the library's responsibility, not ours.
"""

import pytest

from boosters import Metric


class TestMetricValidation:
    """Tests for parameter validation in Metric constructors."""

    def test_ndcg_rejects_zero_at(self) -> None:
        """Metric.ndcg rejects at < 1."""
        with pytest.raises((ValueError, Exception)):
            Metric.ndcg(at=0)


class TestMetricDefaults:
    """Tests for default parameter values."""

    def test_ndcg_default_at(self) -> None:
        """Metric.ndcg has default at=10."""
        metric = Metric.ndcg()
        assert metric.at == 10  # type: ignore[attr-defined]


class TestMetricEquality:
    """Tests for Metric equality (needed for config comparison)."""

    def test_parameterless_variants_equal(self) -> None:
        """Same parameterless variants are equal."""
        assert Metric.rmse() == Metric.rmse()
        assert Metric.rmse() != Metric.mae()

    def test_ndcg_equal_same_at(self) -> None:
        """Ndcg with same at are equal."""
        assert Metric.ndcg(at=5) == Metric.ndcg(at=5)

    def test_ndcg_different_at(self) -> None:
        """Ndcg with different at are not equal."""
        assert Metric.ndcg(at=5) != Metric.ndcg(at=10)
