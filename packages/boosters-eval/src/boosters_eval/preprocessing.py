"""Data preprocessing to create RunData for runners.

The suite runner should stay orchestration-only; all dataset manipulation (subsampling,
feature selection, scaling, dtype casting) lives here.

Design goals:
- A single implementation used consistently for train/validation.
- Prefer sklearn transformers where practical.
- Perform dtype casting (float32) as the last step.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from boosters_eval.config import BenchmarkConfig, BoosterType, LoadedDataset, Task
from boosters_eval.runners.base import RunData


@runtime_checkable
class _SklearnTransformer(Protocol):
    def fit(self, X: Any, y: Any | None = None) -> Any:  # noqa: N803
        ...

    def transform(self, X: Any) -> Any:  # noqa: N803
        ...


class AsFloat32(BaseEstimator, TransformerMixin):
    """Cast an array-like input to a contiguous float32 NumPy array."""

    def fit(self, _X: Any, _y: Any | None = None) -> AsFloat32:  # noqa: N803
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803
        """Transform X to a contiguous float32 NumPy array."""
        arr = np.asarray(X)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return np.ascontiguousarray(arr)


def _build_feature_transform(
    *,
    booster_type: BoosterType,
    categorical_features: list[int],
    feature_names: list[str] | None,
    n_features: int,
) -> tuple[_SklearnTransformer, list[int], list[str] | None]:
    """Return (transformer, new_categorical_features, new_feature_names)."""
    if booster_type != BoosterType.GBLINEAR or not categorical_features:
        return FunctionTransformer(lambda x: x, validate=False), categorical_features, feature_names

    cat = sorted(set(categorical_features))
    keep_cols = [i for i in range(n_features) if i not in cat]

    # Keep columns with sklearn so we can reuse the same fitted transform on valid.
    transformer = ColumnTransformer(
        [("keep", "passthrough", keep_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    new_feature_names = feature_names
    if feature_names is not None:
        new_feature_names = [feature_names[i] for i in keep_cols]

    return transformer, [], new_feature_names


def prepare_run_data(
    *,
    config: BenchmarkConfig,
    loaded: LoadedDataset,
    seed: int,
) -> RunData:
    """Finalize preprocessing into the `RunData` format.

    Handles:
    - target scaling (regression + quantile regression; metrics on normalized scale)
    - final float32 casting for all arrays
    """
    x_train, x_valid, y_train, y_valid, w_train, w_valid = config.dataset.splitter(loaded, seed)

    categorical_features = list(loaded.categorical_features)
    feature_names = loaded.feature_names

    feature_transform, categorical_features, feature_names = _build_feature_transform(
        booster_type=config.booster_type,
        categorical_features=categorical_features,
        feature_names=feature_names,
        n_features=int(np.asarray(x_train).shape[1]),
    )

    feature_transform.fit(x_train)
    x_train = np.asarray(feature_transform.transform(x_train))
    x_valid = np.asarray(feature_transform.transform(x_valid))

    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)

    if config.dataset.task in {Task.REGRESSION, Task.QUANTILE_REGRESSION}:
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
        y_valid = scaler.transform(y_valid.reshape(-1, 1)).reshape(-1)

    x_cast = AsFloat32()
    y_cast = AsFloat32()

    x_train = x_cast.transform(x_train)
    x_valid = x_cast.transform(x_valid)
    y_train = y_cast.transform(y_train)
    y_valid = y_cast.transform(y_valid)

    if w_train is not None:
        w_train = y_cast.transform(w_train)
    if w_valid is not None:
        w_valid = y_cast.transform(w_valid)

    return RunData(
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
        categorical_features=categorical_features,
        feature_names=feature_names,
        sample_weight_train=w_train,
        sample_weight_valid=w_valid,
    )
