"""Tests for GBDTModel type."""

import numpy as np
import pytest

import boosters as bst
from boosters import GBDTConfig, GBDTModel


class TestGBDTModelConstruction:
    """Tests for GBDTModel construction."""

    def test_default_construction(self) -> None:
        """Test constructing with default config."""
        model = GBDTModel()
        assert not model.is_fitted
        assert model.config is not None

    def test_construction_with_config(self) -> None:
        """Test constructing with custom config."""
        config = GBDTConfig(n_estimators=50, learning_rate=0.1)
        model = GBDTModel(config=config)
        assert not model.is_fitted
        assert model.config.n_estimators == 50
        assert model.config.learning_rate == pytest.approx(0.1)


class TestGBDTModelProperties:
    """Tests for GBDTModel properties before fitting."""

    def test_n_trees_raises_before_fit(self) -> None:
        """Test that n_trees raises error before fit."""
        model = GBDTModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.n_trees

    def test_n_features_raises_before_fit(self) -> None:
        """Test that n_features raises error before fit."""
        model = GBDTModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.n_features

    def test_feature_importance_raises_before_fit(self) -> None:
        """Test that feature_importance raises error before fit."""
        model = GBDTModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.feature_importance()

    def test_best_iteration_none_before_fit(self) -> None:
        """Test that best_iteration is None before fit."""
        model = GBDTModel()
        assert model.best_iteration is None

    def test_best_score_none_before_fit(self) -> None:
        """Test that best_score is None before fit."""
        model = GBDTModel()
        assert model.best_score is None

    def test_eval_results_none_before_fit(self) -> None:
        """Test that eval_results is None before fit."""
        model = GBDTModel()
        assert model.eval_results is None


class TestGBDTModelRepr:
    """Tests for GBDTModel string representation."""

    def test_repr_unfitted(self) -> None:
        """Test repr for unfitted model."""
        model = GBDTModel()
        r = repr(model)
        assert "fitted=False" in r


class TestGBDTModelImports:
    """Test that GBDTModel is properly exported."""

    def test_import_from_main(self) -> None:
        """Test importing from main boosters module."""
        assert hasattr(bst, "GBDTModel")

    def test_import_from_model(self) -> None:
        """Test importing from boosters.model module."""
        from boosters.model import GBDTModel as M

        assert M is GBDTModel
