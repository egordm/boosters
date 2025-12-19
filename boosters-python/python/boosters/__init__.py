"""Boosters - Fast gradient boosting library.

This library provides sklearn-compatible gradient boosting models with high performance.

Classes:
    GBDTBooster: Gradient Boosted Decision Trees model
    GBLinearBooster: Gradient Boosted Linear model

Example:
    >>> import numpy as np
    >>> from boosters import GBDTBooster
    >>> 
    >>> X = np.random.randn(100, 10).astype(np.float32)
    >>> y = np.random.randn(100).astype(np.float32)
    >>> 
    >>> model = GBDTBooster(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
"""

from boosters._boosters_python import (
    __version__,
    GBDTBooster,
    GBLinearBooster,
)

__all__ = [
    "__version__",
    "GBDTBooster",
    "GBLinearBooster",
]
