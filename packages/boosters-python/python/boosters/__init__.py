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

# Explainability types
from boosters._boosters_rs import (
    GBLinearUpdateStrategy,
    ImportanceType,
    Verbosity,
    __version__,  # pyright: ignore[reportAttributeAccessIssue]
)

# Config types
from boosters.config import GBDTConfig, GBLinearConfig

# Data types
from boosters.data import Dataset

# Metric enum
from boosters.metrics import Metric

# Model types
from boosters.model import GBDTModel, GBLinearModel

# Objective enum
from boosters.objectives import Objective

# Type aliases
from boosters.types import GrowthStrategy

__all__ = [
    "Dataset",
    "GBDTConfig",
    "GBDTModel",
    "GBLinearConfig",
    "GBLinearModel",
    "GBLinearUpdateStrategy",
    "GrowthStrategy",
    "ImportanceType",
    "Metric",
    "Objective",
    "Verbosity",
    "__version__",
]
