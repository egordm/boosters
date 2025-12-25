"""Data types for gradient boosting.

This module provides dataset wrappers for training and evaluation.
All types are Rust-owned (#[pyclass]) with generated type stubs.

Types:
    - Dataset: Training/prediction dataset with features, labels, weights
    - EvalSet: Named evaluation set for validation during training
"""

# Re-exports will be added as Epic 3 is implemented
# from boosters._boosters_rs import (
#     Dataset,
#     EvalSet,
# )

__all__: list[str] = [
    # "Dataset",
    # "EvalSet",
]
