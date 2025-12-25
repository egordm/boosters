"""Callback classes for training control.

This module provides callbacks for controlling training behavior.
All types are Rust-owned (#[pyclass]) with generated type stubs.

Types:
    - EarlyStopping: Stop training when validation metric stops improving
    - LogEvaluation: Log evaluation metrics periodically
"""

# Re-exports will be added as Epic 4 is implemented
# from boosters._boosters_rs import (
#     EarlyStopping,
#     LogEvaluation,
# )

__all__: list[str] = [
    # "EarlyStopping",
    # "LogEvaluation",
]
