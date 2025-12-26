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

from boosters._boosters_rs import __version__  # pyright: ignore[reportAttributeAccessIssue]

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

# Data types
from boosters.data import Dataset, EvalSet

# Metric enum
from boosters.metrics import Metric

# Model types
from boosters.model import GBDTModel, GBLinearModel

# Objective enum
from boosters.objectives import Objective

# Type aliases
from boosters.types import GrowthStrategy

__all__ = [
    "CategoricalConfig",
    "Dataset",
    "EFBConfig",
    "EvalSet",
    "GBDTConfig",
    "GBDTModel",
    "GBLinearConfig",
    "GBLinearModel",
    "GrowthStrategy",
    "LinearLeavesConfig",
    "Metric",
    "Objective",
    "RegularizationConfig",
    "SamplingConfig",
    "TreeConfig",
    "__version__",
]
