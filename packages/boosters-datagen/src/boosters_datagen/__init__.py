"""Boosters test data generation.

Generates test cases for XGBoost and LightGBM model inference and training validation.

Usage:
    boosters-datagen all       # Generate all test cases
    boosters-datagen xgboost   # Generate XGBoost only
    boosters-datagen lightgbm  # Generate LightGBM only
"""

from boosters_datagen.cli import app, main

__all__ = ["app", "main"]
__version__ = "0.1.0"
