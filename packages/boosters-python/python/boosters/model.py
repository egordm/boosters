"""Model types for gradient boosting.

This module provides the main model classes for training and prediction.
All types are Rust-owned (#[pyclass]) with generated type stubs.

Types:
    - GBDTModel: Gradient Boosted Decision Trees model
    - GBLinearModel: Gradient Boosted Linear model (coming in Story 4.5)
"""

from boosters._boosters_rs import GBDTModel

# GBLinearModel will be added in Story 4.5

__all__: list[str] = [
    "GBDTModel",
    # "GBLinearModel",  # Coming in Story 4.5
]
