"""Sklearn-compatible gradient boosting estimators.

This module provides scikit-learn compatible wrappers for boosters models,
allowing them to be used with sklearn pipelines, cross-validation, and
hyperparameter tuning.

Classes
-------
GBDTRegressor
    Gradient Boosted Decision Tree regressor.
GBDTClassifier
    Gradient Boosted Decision Tree classifier.
GBLinearRegressor
    Gradient Boosted Linear regressor.
GBLinearClassifier
    Gradient Boosted Linear classifier.
"""

from .gbdt import GBDTClassifier, GBDTRegressor
from .gblinear import GBLinearClassifier, GBLinearRegressor

__all__ = [
    "GBDTClassifier",
    "GBDTRegressor",
    "GBLinearClassifier",
    "GBLinearRegressor",
]
