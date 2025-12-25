"""Tests for objective (loss function) types."""

import pytest

import boosters as bst


class TestParameterlessObjectives:
    """Tests for objectives without parameters."""

    def test_squared_loss_creation(self) -> None:
        """SquaredLoss can be instantiated."""
        obj = bst.SquaredLoss()
        assert obj is not None

    def test_squared_loss_repr(self) -> None:
        """SquaredLoss has meaningful repr."""
        obj = bst.SquaredLoss()
        assert "SquaredLoss" in repr(obj)

    def test_absolute_loss_creation(self) -> None:
        """AbsoluteLoss can be instantiated."""
        obj = bst.AbsoluteLoss()
        assert obj is not None

    def test_absolute_loss_repr(self) -> None:
        """AbsoluteLoss has meaningful repr."""
        obj = bst.AbsoluteLoss()
        assert "AbsoluteLoss" in repr(obj)

    def test_poisson_loss_creation(self) -> None:
        """PoissonLoss can be instantiated."""
        obj = bst.PoissonLoss()
        assert obj is not None

    def test_poisson_loss_repr(self) -> None:
        """PoissonLoss has meaningful repr."""
        obj = bst.PoissonLoss()
        assert "PoissonLoss" in repr(obj)

    def test_logistic_loss_creation(self) -> None:
        """LogisticLoss can be instantiated."""
        obj = bst.LogisticLoss()
        assert obj is not None

    def test_logistic_loss_repr(self) -> None:
        """LogisticLoss has meaningful repr."""
        obj = bst.LogisticLoss()
        assert "LogisticLoss" in repr(obj)

    def test_hinge_loss_creation(self) -> None:
        """HingeLoss can be instantiated."""
        obj = bst.HingeLoss()
        assert obj is not None

    def test_hinge_loss_repr(self) -> None:
        """HingeLoss has meaningful repr."""
        obj = bst.HingeLoss()
        assert "HingeLoss" in repr(obj)


class TestHuberLoss:
    """Tests for HuberLoss with delta parameter."""

    def test_default_delta(self) -> None:
        """HuberLoss has default delta=1.0."""
        obj = bst.HuberLoss()
        assert obj.delta == 1.0

    def test_custom_delta(self) -> None:
        """HuberLoss accepts custom delta."""
        obj = bst.HuberLoss(delta=2.5)
        assert obj.delta == 2.5

    def test_repr_includes_delta(self) -> None:
        """HuberLoss repr includes delta value."""
        obj = bst.HuberLoss(delta=1.5)
        r = repr(obj)
        assert "HuberLoss" in r
        assert "1.5" in r

    def test_invalid_delta_zero(self) -> None:
        """HuberLoss rejects delta <= 0."""
        with pytest.raises((ValueError, Exception)):
            bst.HuberLoss(delta=0.0)

    def test_invalid_delta_negative(self) -> None:
        """HuberLoss rejects negative delta."""
        with pytest.raises((ValueError, Exception)):
            bst.HuberLoss(delta=-1.0)


class TestPinballLoss:
    """Tests for PinballLoss (quantile regression)."""

    def test_default_alpha(self) -> None:
        """PinballLoss has default alpha=[0.5] (median)."""
        obj = bst.PinballLoss()
        # Alpha is always stored as a list for consistency
        assert obj.alpha == [0.5]

    def test_custom_alpha_float(self) -> None:
        """PinballLoss accepts single quantile, stores as list."""
        obj = bst.PinballLoss(alpha=0.25)
        # Single float is converted to list
        assert obj.alpha == [0.25]

    def test_custom_alpha_list(self) -> None:
        """PinballLoss accepts list of quantiles."""
        obj = bst.PinballLoss(alpha=[0.1, 0.5, 0.9])
        assert obj.alpha == [0.1, 0.5, 0.9]

    def test_repr_includes_alpha(self) -> None:
        """PinballLoss repr includes alpha value."""
        obj = bst.PinballLoss(alpha=0.75)
        r = repr(obj)
        assert "PinballLoss" in r
        assert "0.75" in r

    def test_invalid_alpha_zero(self) -> None:
        """PinballLoss rejects alpha <= 0."""
        with pytest.raises((ValueError, Exception)):
            bst.PinballLoss(alpha=0.0)

    def test_invalid_alpha_one(self) -> None:
        """PinballLoss rejects alpha >= 1."""
        with pytest.raises((ValueError, Exception)):
            bst.PinballLoss(alpha=1.0)

    def test_invalid_alpha_negative(self) -> None:
        """PinballLoss rejects negative alpha."""
        with pytest.raises((ValueError, Exception)):
            bst.PinballLoss(alpha=-0.5)

    def test_invalid_alpha_list_element(self) -> None:
        """PinballLoss rejects invalid list elements."""
        with pytest.raises((ValueError, Exception)):
            bst.PinballLoss(alpha=[0.1, 1.5, 0.9])


class TestArctanLoss:
    """Tests for ArctanLoss with alpha parameter.

    Note: ArctanLoss alpha must be in (0, 1), not unbounded like HuberLoss delta.
    """

    def test_default_alpha(self) -> None:
        """ArctanLoss has default alpha=0.5."""
        obj = bst.ArctanLoss()
        assert obj.alpha == 0.5

    def test_custom_alpha(self) -> None:
        """ArctanLoss accepts custom alpha in (0, 1)."""
        obj = bst.ArctanLoss(alpha=0.3)
        assert obj.alpha == 0.3

    def test_repr_includes_alpha(self) -> None:
        """ArctanLoss repr includes alpha value."""
        obj = bst.ArctanLoss(alpha=0.5)
        r = repr(obj)
        assert "ArctanLoss" in r
        assert "0.5" in r

    def test_invalid_alpha_zero(self) -> None:
        """ArctanLoss rejects alpha <= 0."""
        with pytest.raises((ValueError, Exception)):
            bst.ArctanLoss(alpha=0.0)

    def test_invalid_alpha_negative(self) -> None:
        """ArctanLoss rejects negative alpha."""
        with pytest.raises((ValueError, Exception)):
            bst.ArctanLoss(alpha=-0.5)

    def test_invalid_alpha_one(self) -> None:
        """ArctanLoss rejects alpha >= 1."""
        with pytest.raises((ValueError, Exception)):
            bst.ArctanLoss(alpha=1.0)

    def test_invalid_alpha_greater_than_one(self) -> None:
        """ArctanLoss rejects alpha > 1."""
        with pytest.raises((ValueError, Exception)):
            bst.ArctanLoss(alpha=2.0)


class TestSoftmaxLoss:
    """Tests for SoftmaxLoss (multiclass classification).

    Note: n_classes is required (no default), unlike other parameterized objectives.
    """

    def test_requires_n_classes(self) -> None:
        """SoftmaxLoss requires n_classes parameter."""
        with pytest.raises(TypeError):
            bst.SoftmaxLoss()  # type: ignore[call-arg]

    def test_custom_n_classes(self) -> None:
        """SoftmaxLoss accepts custom n_classes."""
        obj = bst.SoftmaxLoss(n_classes=10)
        assert obj.n_classes == 10

    def test_binary_n_classes(self) -> None:
        """SoftmaxLoss accepts n_classes=2 for binary."""
        obj = bst.SoftmaxLoss(n_classes=2)
        assert obj.n_classes == 2

    def test_repr_includes_n_classes(self) -> None:
        """SoftmaxLoss repr includes n_classes."""
        obj = bst.SoftmaxLoss(n_classes=5)
        r = repr(obj)
        assert "SoftmaxLoss" in r
        assert "5" in r

    def test_invalid_n_classes_one(self) -> None:
        """SoftmaxLoss rejects n_classes < 2."""
        with pytest.raises((ValueError, Exception)):
            bst.SoftmaxLoss(n_classes=1)

    def test_invalid_n_classes_zero(self) -> None:
        """SoftmaxLoss rejects n_classes = 0."""
        with pytest.raises((ValueError, Exception)):
            bst.SoftmaxLoss(n_classes=0)


class TestLambdaRankLoss:
    """Tests for LambdaRankLoss (learning-to-rank)."""

    def test_default_ndcg_at(self) -> None:
        """LambdaRankLoss has default ndcg_at=10."""
        obj = bst.LambdaRankLoss()
        assert obj.ndcg_at == 10

    def test_custom_ndcg_at(self) -> None:
        """LambdaRankLoss accepts custom ndcg_at."""
        obj = bst.LambdaRankLoss(ndcg_at=5)
        assert obj.ndcg_at == 5

    def test_repr_includes_ndcg_at(self) -> None:
        """LambdaRankLoss repr includes ndcg_at."""
        obj = bst.LambdaRankLoss(ndcg_at=20)
        r = repr(obj)
        assert "LambdaRankLoss" in r
        assert "20" in r

    def test_invalid_ndcg_at_zero(self) -> None:
        """LambdaRankLoss rejects ndcg_at < 1."""
        with pytest.raises((ValueError, Exception)):
            bst.LambdaRankLoss(ndcg_at=0)


class TestObjectiveTypeAlias:
    """Tests for Objective type alias."""

    def test_squared_loss_is_objective(self) -> None:
        """SquaredLoss is a valid Objective."""
        obj: bst.Objective = bst.SquaredLoss()
        assert obj is not None

    def test_huber_loss_is_objective(self) -> None:
        """HuberLoss is a valid Objective."""
        obj: bst.Objective = bst.HuberLoss(delta=2.0)
        assert obj is not None

    def test_softmax_loss_is_objective(self) -> None:
        """SoftmaxLoss is a valid Objective."""
        obj: bst.Objective = bst.SoftmaxLoss(n_classes=5)
        assert obj is not None
