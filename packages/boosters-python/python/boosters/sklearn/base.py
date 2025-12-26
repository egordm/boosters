"""Base classes and utilities for sklearn integration.

This module provides shared logic for sklearn-compatible estimators.
Requires scikit-learn to be installed.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator as _SklearnBaseEstimator
from sklearn.base import ClassifierMixin as _SklearnClassifierMixin
from sklearn.base import RegressorMixin as _SklearnRegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Re-export sklearn classes and functions
BaseEstimator = _SklearnBaseEstimator
ClassifierMixin = _SklearnClassifierMixin
RegressorMixin = _SklearnRegressorMixin

__all__ = [
    "BaseEstimator",
    "ClassifierMixin",
    "RegressorMixin",
    "check_X_y",
    "check_array",
    "check_is_fitted",
]
