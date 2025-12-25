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

# Metric types
from boosters.metrics import (
    Accuracy,
    Auc,
    LogLoss,
    Mae,
    Mape,
    Metric,
    Ndcg,
    Rmse,
)

# Objective types
from boosters.objectives import (
    AbsoluteLoss,
    ArctanLoss,
    HingeLoss,
    HuberLoss,
    LambdaRankLoss,
    LogisticLoss,
    Objective,
    PinballLoss,
    PoissonLoss,
    SoftmaxLoss,
    SquaredLoss,
)

__all__ = [
    "AbsoluteLoss",
    "Accuracy",
    "ArctanLoss",
    "Auc",
    "CategoricalConfig",
    "EFBConfig",
    "GBDTConfig",
    "GBLinearConfig",
    "HingeLoss",
    "HuberLoss",
    "LambdaRankLoss",
    "LinearLeavesConfig",
    "LogLoss",
    "LogisticLoss",
    "Mae",
    "Mape",
    "Metric",
    "Ndcg",
    "Objective",
    "PinballLoss",
    "PoissonLoss",
    "RegularizationConfig",
    "Rmse",
    "SamplingConfig",
    "SoftmaxLoss",
    "SquaredLoss",
    "TreeConfig",
    "__version__",
]
