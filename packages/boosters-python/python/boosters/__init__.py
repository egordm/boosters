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
    - https://github.com/egordm/booste-rs for documentation
"""

# Explainability types
import boosters._boosters_rs as _rs
from boosters._boosters_rs import (
    GBLinearUpdateStrategy,
    ImportanceType,
    Model,
    ModelInfo,
    ReadError,
    Verbosity,
)
from boosters.config import GBDTConfig, GBLinearConfig
from boosters.convert import (
    lightgbm_to_json_bytes,
    lightgbm_to_schema,
    xgboost_to_json_bytes,
    xgboost_to_schema,
)
from boosters.dataset import Dataset, DatasetBuilder, Feature
from boosters.metrics import Metric
from boosters.model import GBDTModel, GBLinearModel
from boosters.objectives import Objective
from boosters.plotting import plot_tree, tree_to_dataframe
from boosters.types import GrowthStrategy

__version__: str = getattr(_rs, "__version__", "0.0.0")

__all__ = [
    "Dataset",
    "DatasetBuilder",
    "Feature",
    "GBDTConfig",
    "GBDTModel",
    "GBLinearConfig",
    "GBLinearModel",
    "GBLinearUpdateStrategy",
    "GrowthStrategy",
    "ImportanceType",
    "Metric",
    "Model",
    "ModelInfo",
    "Objective",
    "ReadError",
    "Verbosity",
    "__version__",
    "lightgbm_to_json_bytes",
    "lightgbm_to_schema",
    "plot_tree",
    "tree_to_dataframe",
    "xgboost_to_json_bytes",
    "xgboost_to_schema",
]
