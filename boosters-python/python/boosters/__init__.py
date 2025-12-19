"""Boosters - Fast gradient boosting library.

This library provides sklearn-compatible gradient boosting models with high performance.

Classes:
    GBDTBooster: Gradient Boosted Decision Trees model
    GBLinearBooster: Gradient Boosted Linear model

Conversion Utilities (file format parsers - no XGBoost/LightGBM required):
    xgboost_json_to_bstr: Convert XGBoost JSON file to .bstr format
    lightgbm_txt_to_bstr: Convert LightGBM text file to .bstr format

Conversion Utilities (model object converters - requires libraries):
    xgboost_to_bstr: Convert XGBoost model object to .bstr format
    lightgbm_to_bstr: Convert LightGBM model object to .bstr format

Example:
    >>> import numpy as np
    >>> from boosters import GBDTBooster

    >>> X = np.random.randn(100, 10).astype(np.float32)
    >>> y = np.random.randn(100).astype(np.float32)

    >>> model = GBDTBooster(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from boosters._boosters_python import (
    GBDTBooster,
    GBLinearBooster,
    __version__,
)

if TYPE_CHECKING:
    import lightgbm as lgb
    import xgboost as xgb


def xgboost_json_to_bstr(
    json_path: str | Path,
    output_path: str | Path,
) -> None:
    """Convert XGBoost JSON model file to boosters .bstr format.

    This function parses the XGBoost JSON format directly without requiring
    XGBoost to be installed. Useful for deployment scenarios where you have
    pre-trained model files.

    Args:
        json_path: Path to the XGBoost JSON model file
        output_path: Path for the output .bstr file

    Raises:
        OSError: If the input file cannot be read
        ValueError: If the JSON is invalid or unsupported

    Example:
        >>> from boosters import xgboost_json_to_bstr, GBDTBooster
        >>>
        >>> # Convert (one-time, can be done during build)
        >>> xgboost_json_to_bstr("model.json", "model.bstr")
        >>>
        >>> # Load and use (fast, for inference)
        >>> model = GBDTBooster.load("model.bstr")
        >>> predictions = model.predict(X)
    """
    model = GBDTBooster.load_from_xgboost_json(str(json_path))
    model.save(str(output_path))


def lightgbm_txt_to_bstr(
    txt_path: str | Path,
    output_path: str | Path,
) -> None:
    """Convert LightGBM text model file to boosters .bstr format.

    This function parses the LightGBM text format directly without requiring
    LightGBM to be installed. Useful for deployment scenarios where you have
    pre-trained model files.

    Args:
        txt_path: Path to the LightGBM text model file (.txt)
        output_path: Path for the output .bstr file

    Raises:
        OSError: If the input file cannot be read
        ValueError: If the text file is invalid or unsupported

    Example:
        >>> from boosters import lightgbm_txt_to_bstr, GBDTBooster
        >>>
        >>> # Convert (one-time, can be done during build)
        >>> lightgbm_txt_to_bstr("model.txt", "model.bstr")
        >>>
        >>> # Load and use (fast, for inference)
        >>> model = GBDTBooster.load("model.bstr")
        >>> predictions = model.predict(X)
    """
    model = GBDTBooster.load_from_lightgbm_txt(str(txt_path))
    model.save(str(output_path))


# Lazy import converters that require xgboost/lightgbm libraries
def xgboost_to_bstr(
    model: xgb.Booster | xgb.XGBModel,
    path: str | Path,
    *,
    include_feature_names: bool = True,
) -> None:
    """Convert XGBoost model object to boosters .bstr format.

    Requires XGBoost to be installed. Uses XGBoost's introspection APIs.
    See boosters.convert.xgboost_to_bstr for full documentation.
    """
    from boosters.convert import xgboost_to_bstr as _xgboost_to_bstr

    _xgboost_to_bstr(model, path, include_feature_names=include_feature_names)


def lightgbm_to_bstr(
    model: lgb.Booster | lgb.LGBMModel,
    path: str | Path,
    *,
    include_feature_names: bool = True,
) -> None:
    """Convert LightGBM model object to boosters .bstr format.

    Requires LightGBM to be installed. Uses LightGBM's introspection APIs.
    See boosters.convert.lightgbm_to_bstr for full documentation.
    """
    from boosters.convert import lightgbm_to_bstr as _lightgbm_to_bstr

    _lightgbm_to_bstr(model, path, include_feature_names=include_feature_names)


__all__ = [
    "GBDTBooster",
    "GBLinearBooster",
    "__version__",
    "lightgbm_to_bstr",
    "lightgbm_txt_to_bstr",
    "xgboost_json_to_bstr",
    "xgboost_to_bstr",
]
