"""Boosters - Fast gradient boosting library with native Rust core.

This package provides gradient boosted decision trees (GBDT) and linear models
with a clean Pythonic API and scikit-learn compatibility.

Example:
    >>> import boosters as bst
    >>> import numpy as np
    >>> X = np.random.rand(100, 10).astype(np.float32)
    >>> y = np.random.rand(100).astype(np.float32)
    >>> # More examples after model implementation

See Also:
    - RFC-0014 for API design rationale
    - https://github.com/your-org/boosters for documentation
"""

from boosters._boosters_rs import __version__

# Config types
from boosters.config import (
    CategoricalConfig,
    EFBConfig,
    GBDTConfig,
    GBLinearConfig,
    LinearLeavesConfig,
    RegularizationConfig,
    SamplingConfig,
    TreeConfig,
)

__all__ = [
    # Config types (Epic 2 - Story 2.1)
    "CategoricalConfig",
    "EFBConfig",
    "GBDTConfig",
    "GBLinearConfig",
    "LinearLeavesConfig",
    "RegularizationConfig",
    "SamplingConfig",
    "TreeConfig",
    # Version
    "__version__",
    # Objective types (Epic 2 - Story 2.2)
    # Metric types (Epic 2 - Story 2.3)
    # Data types (Epic 3)
    # Model types (Epic 4)
]
