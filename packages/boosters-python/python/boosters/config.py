"""Configuration types for boosters.

This module provides configuration dataclasses for GBDT and GBLinear models.
All types are Rust-owned (#[pyclass]) with generated type stubs.

Types:
    - TreeConfig: Tree structure configuration
    - RegularizationConfig: L1/L2 regularization
    - SamplingConfig: Row/column sampling
    - CategoricalConfig: Categorical feature handling
    - EFBConfig: Exclusive Feature Bundling
    - LinearLeavesConfig: Linear models in leaves
    - GBDTConfig: Top-level GBDT configuration
    - GBLinearConfig: Top-level GBLinear configuration
"""

from boosters._boosters_rs import (
    CategoricalConfig,
    EFBConfig,
    GBDTConfig,
    GBLinearConfig,
    LinearLeavesConfig,
    RegularizationConfig,
    SamplingConfig,
    TreeConfig,
)

__all__: list[str] = [
    "CategoricalConfig",
    "EFBConfig",
    "GBDTConfig",
    "GBLinearConfig",
    "LinearLeavesConfig",
    "RegularizationConfig",
    "SamplingConfig",
    "TreeConfig",
]
