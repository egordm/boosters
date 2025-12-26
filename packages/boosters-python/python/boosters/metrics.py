"""Evaluation metrics for gradient boosting.

This module provides the Metric enum for model evaluation during training.
The Metric class is a Rust-backed PyO3 complex enum with variants for each metric type.

Usage:
    >>> from boosters import Metric
    >>> metric = Metric.rmse()  # Root mean squared error
    >>> metric = Metric.logloss()  # Binary log loss
    >>> metric = Metric.ndcg(at=10)  # NDCG@10 for ranking

Regression:
    - Metric.Rmse(): Root mean squared error
    - Metric.Mae(): Mean absolute error
    - Metric.Mape(): Mean absolute percentage error

Classification:
    - Metric.LogLoss(): Binary log loss
    - Metric.Auc(): Area under ROC curve
    - Metric.Accuracy(): Classification accuracy

Ranking:
    - Metric.Ndcg(at): Normalized discounted cumulative gain

Pattern Matching::

    match metric:
        case Metric.Rmse():
            print("RMSE")
        case Metric.Ndcg(at=k):
            print(f"NDCG@{k}")
"""

from boosters._boosters_rs import Metric

__all__: list[str] = ["Metric"]
