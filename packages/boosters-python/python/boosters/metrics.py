"""Evaluation metrics for gradient boosting.

This module provides metric classes for model evaluation during training.
All types are Rust-owned (#[pyclass]) with generated type stubs.

Regression:
    - Rmse: Root mean squared error
    - Mae: Mean absolute error
    - Mape: Mean absolute percentage error

Classification:
    - LogLoss: Binary log loss
    - Auc: Area under ROC curve
    - Accuracy: Classification accuracy

Ranking:
    - Ndcg: Normalized discounted cumulative gain

Type Aliases:
    - Metric: Union of all metric types
"""

from boosters._boosters_rs import (
    Accuracy,
    Auc,
    LogLoss,
    Mae,
    Mape,
    Ndcg,
    Rmse,
)

# Type alias for all metrics
type Metric = Rmse | Mae | Mape | LogLoss | Auc | Accuracy | Ndcg

__all__: list[str] = [
    "Accuracy",
    "Auc",
    "LogLoss",
    "Mae",
    "Mape",
    "Metric",
    "Ndcg",
    "Rmse",
]
