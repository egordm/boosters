"""Tests for Objective (loss function) validation.

Tests focus on validation behavior—PyO3 enum mechanics are not tested
since that's the library's responsibility, not ours.
"""

import pytest

from boosters import Objective


class TestObjectiveValidation:
    """Tests for parameter validation in Objective constructors."""

    def test_huber_rejects_zero_delta(self) -> None:
        """Objective.huber rejects delta <= 0."""
        with pytest.raises((ValueError, Exception)):
            Objective.huber(delta=0.0)

    def test_huber_rejects_negative_delta(self) -> None:
        """Objective.huber rejects negative delta."""
        with pytest.raises((ValueError, Exception)):
            Objective.huber(delta=-1.0)

    def test_pinball_requires_alpha(self) -> None:
        """Objective.pinball requires alpha parameter."""
        with pytest.raises(TypeError):
            Objective.pinball()  # type: ignore[call-arg]

    def test_pinball_rejects_out_of_range_alpha(self) -> None:
        """Objective.pinball rejects alpha outside (0, 1)."""
        with pytest.raises((ValueError, Exception)):
            Objective.pinball(alpha=[0.0])
        with pytest.raises((ValueError, Exception)):
            Objective.pinball(alpha=[1.0])
        with pytest.raises((ValueError, Exception)):
            Objective.pinball(alpha=[-0.5])
        with pytest.raises((ValueError, Exception)):
            Objective.pinball(alpha=[0.1, 1.5, 0.9])

    def test_softmax_requires_n_classes(self) -> None:
        """Objective.softmax requires n_classes parameter."""
        with pytest.raises(TypeError):
            Objective.softmax()  # type: ignore[call-arg]

    def test_softmax_rejects_invalid_n_classes(self) -> None:
        """Objective.softmax rejects n_classes < 2."""
        with pytest.raises((ValueError, Exception)):
            Objective.softmax(n_classes=1)
        with pytest.raises((ValueError, Exception)):
            Objective.softmax(n_classes=0)

    def test_lambdarank_rejects_zero_ndcg_at(self) -> None:
        """Objective.lambdarank rejects ndcg_at < 1."""
        with pytest.raises((ValueError, Exception)):
            Objective.lambdarank(ndcg_at=0)


class TestObjectiveDefaults:
    """Tests for default parameter values."""

    def test_huber_default_delta(self) -> None:
        """Objective.huber has default delta=1.0."""
        obj = Objective.huber()
        assert obj.delta == 1.0  # type: ignore[attr-defined]

    def test_lambdarank_default_ndcg_at(self) -> None:
        """Objective.lambdarank has default ndcg_at=10."""
        obj = Objective.lambdarank()
        assert obj.ndcg_at == 10  # type: ignore[attr-defined]


class TestObjectiveEquality:
    """Tests for Objective equality (needed for config comparison)."""

    def test_parameterless_variants_equal(self) -> None:
        """Same parameterless variants are equal."""
        assert Objective.squared() == Objective.squared()
        assert Objective.squared() != Objective.absolute()

    def test_parameterized_variants_equal_same_params(self) -> None:
        """Parameterized variants with same values are equal."""
        assert Objective.huber(delta=1.5) == Objective.huber(delta=1.5)
        assert Objective.softmax(n_classes=10) == Objective.softmax(n_classes=10)

    def test_parameterized_variants_different_params(self) -> None:
        """Parameterized variants with different values are not equal."""
        assert Objective.huber(delta=1.0) != Objective.huber(delta=2.0)
        assert Objective.softmax(n_classes=5) != Objective.softmax(n_classes=10)
