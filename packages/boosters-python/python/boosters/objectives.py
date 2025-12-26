"""Objective (loss) functions for gradient boosting.

This module provides the Objective enum for training GBDT and GBLinear models.
The Objective class is a Rust-backed PyO3 complex enum with variants for each loss type.

Usage:
    >>> from boosters import Objective
    >>> obj = Objective.squared()  # L2 regression
    >>> obj = Objective.logistic()  # Binary classification
    >>> obj = Objective.pinball([0.1, 0.5, 0.9])  # Quantile regression
    >>> obj = Objective.softmax(10)  # Multiclass classification

Regression:
    - Objective.Squared(): Mean squared error (L2)
    - Objective.Absolute(): Mean absolute error (L1)
    - Objective.Huber(delta): Pseudo-Huber loss (robust)
    - Objective.Pinball(alpha): Quantile regression
    - Objective.Poisson(): Poisson deviance for count data

Classification:
    - Objective.Logistic(): Binary cross-entropy
    - Objective.Hinge(): SVM-style hinge loss
    - Objective.Softmax(n_classes): Multiclass cross-entropy

Ranking:
    - Objective.LambdaRank(ndcg_at): LambdaMART for NDCG optimization

Pattern Matching:
    >>> match obj:
    ...     case Objective.Squared():
    ...         print("L2 loss")
    ...     case Objective.Pinball(alpha=a):
    ...         print(f"Quantile: {a}")
"""

from boosters._boosters_rs import Objective

__all__: list[str] = ["Objective"]
