"""Configuration types for boosters.

This module provides configuration dataclasses for GBDT and GBLinear models.
All types are Rust-owned (#[pyclass]) with generated type stubs.

Types:
    - GBDTConfig: Top-level GBDT configuration (flat structure)
    - GBLinearConfig: Top-level GBLinear configuration (flat structure)
"""

from boosters._boosters_rs import (
    GBDTConfig,
    GBLinearConfig,
)

__all__: list[str] = [
    "GBDTConfig",
    "GBLinearConfig",
]
