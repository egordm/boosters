"""Benchmark runner implementations.

This package contains one module per backend library.
"""

from __future__ import annotations

from boosters_eval.runners.base import RunData, Runner
from boosters_eval.runners.boosters import BoostersRunner
from boosters_eval.runners.lightgbm import LightGBMRunner
from boosters_eval.runners.registry import get_available_runners, get_runner
from boosters_eval.runners.xgboost import XGBoostRunner

__all__ = [
    "BoostersRunner",
    "LightGBMRunner",
    "RunData",
    "Runner",
    "XGBoostRunner",
    "get_available_runners",
    "get_runner",
]
