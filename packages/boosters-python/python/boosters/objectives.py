"""Objective (loss) functions for gradient boosting.

This module provides objective classes for training GBDT and GBLinear models.
All types are Rust-owned (#[pyclass]) with generated type stubs.

Regression:
    - SquaredLoss: Mean squared error (L2)
    - AbsoluteLoss: Mean absolute error (L1)
    - HuberLoss: Pseudo-Huber loss (robust)
    - PinballLoss: Quantile regression (single or multi)
    - PoissonLoss: Poisson deviance for count data

Classification:
    - LogisticLoss: Binary cross-entropy
    - HingeLoss: SVM-style hinge loss
    - SoftmaxLoss: Multiclass cross-entropy

Ranking:
    - LambdaRankLoss: LambdaMART for NDCG optimization

Type Aliases:
    - Objective: Union of all objective types
"""

from typing import TypeAlias

# Re-exports will be added as Epic 2 is implemented
# from boosters._boosters_rs import (
#     SquaredLoss,
#     AbsoluteLoss,
#     HuberLoss,
#     PinballLoss,
#     PoissonLoss,
#     LogisticLoss,
#     HingeLoss,
#     SoftmaxLoss,
#     LambdaRankLoss,
# )

# Type alias for all objectives (populated after implementation)
# Objective: TypeAlias = (
#     SquaredLoss | AbsoluteLoss | HuberLoss | PinballLoss | PoissonLoss |
#     LogisticLoss | HingeLoss | SoftmaxLoss | LambdaRankLoss
# )

__all__: list[str] = [
    # "SquaredLoss",
    # "AbsoluteLoss",
    # "HuberLoss",
    # "PinballLoss",
    # "PoissonLoss",
    # "LogisticLoss",
    # "HingeLoss",
    # "SoftmaxLoss",
    # "LambdaRankLoss",
    # "Objective",
]
