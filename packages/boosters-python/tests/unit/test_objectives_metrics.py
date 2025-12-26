"""Tests for Objective and Metric validation.

Focuses on validation behavior that protects users from errors.
PyO3 enum mechanics are not tested—that's the library's responsibility.
"""

import pytest

from boosters import Metric, Objective


class TestObjectiveValidation:
    """Tests for Objective parameter validation."""

    def test_huber_rejects_invalid_delta(self) -> None:
        """Huber delta must be positive."""
        with pytest.raises((ValueError, Exception)):
            Objective.huber(delta=0.0)
        with pytest.raises((ValueError, Exception)):
            Objective.huber(delta=-1.0)

    def test_pinball_requires_alpha(self) -> None:
        """Pinball requires alpha parameter."""
        with pytest.raises(TypeError):
            Objective.pinball()  # type: ignore[call-arg]

    def test_pinball_alpha_must_be_in_0_1(self) -> None:
        """Pinball alpha must be in (0, 1)."""
        with pytest.raises((ValueError, Exception)):
            Objective.pinball(alpha=[0.0])
        with pytest.raises((ValueError, Exception)):
            Objective.pinball(alpha=[1.0])

    def test_softmax_requires_n_classes(self) -> None:
        """Softmax requires n_classes parameter."""
        with pytest.raises(TypeError):
            Objective.softmax()  # type: ignore[call-arg]

    def test_softmax_n_classes_must_be_at_least_2(self) -> None:
        """Softmax n_classes must be >= 2."""
        with pytest.raises((ValueError, Exception)):
            Objective.softmax(n_classes=1)

    def test_lambdarank_ndcg_at_must_be_positive(self) -> None:
        """LambdaRank ndcg_at must be >= 1."""
        with pytest.raises((ValueError, Exception)):
            Objective.lambdarank(ndcg_at=0)


class TestObjectiveEquality:
    """Tests for Objective equality (needed for config comparison)."""

    def test_same_objectives_equal(self) -> None:
        """Same objectives are equal."""
        assert Objective.squared() == Objective.squared()
        assert Objective.huber(delta=1.5) == Objective.huber(delta=1.5)
        assert Objective.softmax(n_classes=10) == Objective.softmax(n_classes=10)

    def test_different_objectives_not_equal(self) -> None:
        """Different objectives are not equal."""
        assert Objective.squared() != Objective.absolute()
        assert Objective.huber(delta=1.0) != Objective.huber(delta=2.0)
        assert Objective.softmax(n_classes=5) != Objective.softmax(n_classes=10)


class TestMetricValidation:
    """Tests for Metric parameter validation."""

    def test_ndcg_at_must_be_positive(self) -> None:
        """NDCG at must be >= 1."""
        with pytest.raises((ValueError, Exception)):
            Metric.ndcg(at=0)


class TestMetricEquality:
    """Tests for Metric equality."""

    def test_same_metrics_equal(self) -> None:
        """Same metrics are equal."""
        assert Metric.rmse() == Metric.rmse()
        assert Metric.ndcg(at=5) == Metric.ndcg(at=5)

    def test_different_metrics_not_equal(self) -> None:
        """Different metrics are not equal."""
        assert Metric.rmse() != Metric.mae()
        assert Metric.ndcg(at=5) != Metric.ndcg(at=10)
