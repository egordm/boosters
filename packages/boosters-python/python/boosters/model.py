"""Model types for gradient boosting.

This module provides the main model classes for training and prediction.
All types are Rust-owned (#[pyclass]) with generated type stubs.

Types:
    - GBDTModel: Gradient Boosted Decision Trees model
    - GBLinearModel: Gradient Boosted Linear model
"""

from boosters._boosters_rs import GBDTModel, GBLinearModel

__all__: list[str] = [
    "GBDTModel",
    "GBLinearModel",
]
