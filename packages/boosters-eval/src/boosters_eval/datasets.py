"""Dataset configurations for benchmarks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import (
    fetch_california_housing,
    fetch_covtype,
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)

from boosters_eval.config import BoosterType, DatasetConfig, LoadedDataset, Task


@lru_cache(maxsize=1)
def california_housing() -> LoadedDataset:
    """California housing regression dataset."""
    data = cast(Any, fetch_california_housing())
    return LoadedDataset(
        x=data.data.astype(np.float32),
        y=data.target.astype(np.float32),  # pyright: ignore[reportAttributeAccessIssue]
        feature_names=[f"f{i}" for i in range(data.data.shape[1])],
    )


@lru_cache(maxsize=1)
def breast_cancer() -> LoadedDataset:
    """Breast cancer binary classification dataset."""
    data = cast(Any, load_breast_cancer())
    return LoadedDataset(
        x=data.data.astype(np.float32),
        y=data.target.astype(np.float32),  # pyright: ignore[reportAttributeAccessIssue]
        feature_names=[f"f{i}" for i in range(data.data.shape[1])],
    )


@lru_cache(maxsize=1)
def iris() -> LoadedDataset:
    """Iris multiclass classification dataset."""
    data = cast(Any, load_iris())
    return LoadedDataset(
        x=data.data.astype(np.float32),
        y=data.target.astype(np.float32),  # pyright: ignore[reportAttributeAccessIssue]
        feature_names=[f"f{i}" for i in range(data.data.shape[1])],
    )


@lru_cache(maxsize=1)
def covertype() -> LoadedDataset:
    """Covertype multiclass classification dataset (subsampled)."""
    data = cast(Any, fetch_covtype())
    x = data.data.astype(np.float32)
    y = (data.target - 1).astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]
    return LoadedDataset(x=x, y=y, feature_names=[f"f{i}" for i in range(x.shape[1])])


def _synthetic_regression(
    n_samples: int = 10000,
    n_features: int = 50,
    noise: float = 0.1,
) -> LoadedDataset:
    """Generate synthetic regression dataset."""
    x, y = make_regression(  # pyright: ignore[reportAssignmentType]
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=42,
    )
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return LoadedDataset(x=x, y=y, feature_names=[f"f{i}" for i in range(n_features)])


def _synthetic_classification(
    n_samples: int = 10000,
    n_features: int = 50,
    n_classes: int = 2,
) -> LoadedDataset:
    """Generate synthetic classification dataset."""
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=min(n_features // 2, n_classes * 2),
        n_redundant=n_features // 4,
        random_state=42,
    )
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return LoadedDataset(x=x, y=y, feature_names=[f"f{i}" for i in range(n_features)])


# Dataset loader factories
@lru_cache(maxsize=1)
def synthetic_regression_small() -> LoadedDataset:
    """Small synthetic regression dataset."""
    return _synthetic_regression(2000, 50)


@lru_cache(maxsize=1)
def synthetic_regression_medium() -> LoadedDataset:
    """Medium synthetic regression dataset."""
    return _synthetic_regression(10000, 100)


@lru_cache(maxsize=1)
def synthetic_binary_small() -> LoadedDataset:
    """Small synthetic binary classification dataset."""
    return _synthetic_classification(2000, 50, 2)


@lru_cache(maxsize=1)
def synthetic_binary_medium() -> LoadedDataset:
    """Medium synthetic binary classification dataset."""
    return _synthetic_classification(10000, 100, 2)


@lru_cache(maxsize=1)
def synthetic_multi_small() -> LoadedDataset:
    """Small synthetic multiclass classification dataset."""
    return _synthetic_classification(2000, 50, 5)


def _latest_available_at_per_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    if "available_at" in out.columns:
        out["available_at"] = pd.to_datetime(out["available_at"], utc=True)
        out = out.sort_values(["timestamp", "available_at"]).drop_duplicates("timestamp", keep="last")
    else:
        out = out.sort_values(["timestamp"]).drop_duplicates("timestamp", keep="last")

    return out


def _detect_target_column(load_df: pd.DataFrame) -> str:
    candidates = [
        "value",
        "load",
        "measurement",
        "y",
        "target",
    ]
    for name in candidates:
        if name in load_df.columns:
            return name

    excluded = {"timestamp", "available_at"}
    numeric_cols = [c for c in load_df.columns if c not in excluded and pd.api.types.is_numeric_dtype(load_df[c])]
    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        "Could not infer regression target column from load_measurements parquet. "
        f"Tried {candidates}. Numeric candidates: {numeric_cols}."
    )


@lru_cache(maxsize=1)
def liander_energy_prices_os_gorredijk(
    *,
    repo_id: str = "OpenSTEF/liander2024-energy-forecasting-benchmark",
    target: str = "mv_feeder/OS Gorredijk",
    cache_dir: str | None = None,
) -> LoadedDataset:
    """Liander energy forecasting benchmark (subset) as a regression dataset.

    Data source: HuggingFace dataset OpenSTEF/liander2024-energy-forecasting-benchmark.

    Notes:
    - Each parquet contains (timestamp, available_at, ...). We take the latest available_at per timestamp.
    - We join load measurements (target) with weather forecasts, EPEX and profiles on timestamp.
    - We add seasonality features: day-of-week and quarter-hour-of-day as categorical features.
    """
    try:
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for the Liander benchmark. "
            "Install with: uv add -p packages/boosters-eval huggingface_hub"
        ) from e

    local_dir = Path(cache_dir) if cache_dir else Path(".cache") / "liander_dataset"
    local_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [
        f"load_measurements/{target}.parquet",
        f"weather_forecasts_versioned/{target}.parquet",
        "EPEX.parquet",
        "profiles.parquet",
    ]

    resolved_paths: list[Path] = []
    for filename in files_to_download:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(local_dir),
        )
        resolved_paths.append(Path(path))

    load_path, weather_path, epex_path, profiles_path = resolved_paths

    try:
        load_df = pd.read_parquet(load_path)
        weather_df = pd.read_parquet(weather_path)
        epex_df = pd.read_parquet(epex_path)
        profiles_df = pd.read_parquet(profiles_path)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Reading parquet requires a parquet engine (pyarrow recommended). "
            "Install with: uv add -p packages/boosters-eval pyarrow"
        ) from e

    load_df = _latest_available_at_per_timestamp(load_df)
    weather_df = _latest_available_at_per_timestamp(weather_df)
    epex_df = _latest_available_at_per_timestamp(epex_df)
    profiles_df = _latest_available_at_per_timestamp(profiles_df)

    target_col = _detect_target_column(load_df)

    # Join (load as base to define the target)
    df = load_df.merge(weather_df, on="timestamp", how="inner", suffixes=("", "_weather"))
    df = df.merge(epex_df, on="timestamp", how="left", suffixes=("", "_epex"))
    df = df.merge(profiles_df, on="timestamp", how="left", suffixes=("", "_profiles"))

    # Sort chronologically (so the split can be done via slicing)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(_ts=ts).sort_values("_ts").drop(columns=["_ts"]).reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    # The dataset contains some noisy/weird values early in January.
    # Drop the first ~half month to stabilize the benchmark.
    cutoff = ts.iloc[0] + pd.Timedelta(days=15)
    keep = ts >= cutoff
    df = df.loc[keep].reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)

    # Precompute split indices: train = first 4 months, valid = next 2 months.
    start = ts.iloc[0]
    train_end_ts = start + pd.DateOffset(months=4)
    valid_end_ts = train_end_ts + pd.DateOffset(months=2)
    train_end = int(np.searchsorted(ts.to_numpy(), train_end_ts.to_datetime64(), side="left"))
    valid_end = int(np.searchsorted(ts.to_numpy(), valid_end_ts.to_datetime64(), side="left"))
    df["dow"] = ts.dt.dayofweek.astype("int16")
    df["qod"] = (ts.dt.hour * 4 + (ts.dt.minute // 15)).astype("int16")

    # Build X/y
    exclude_cols = {
        "timestamp",
        "available_at",
        target_col,
    }
    exclude_cols.update({c for c in df.columns if c.endswith(("_x", "_y"))})
    # Keep only numeric features
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Ensure deterministic column order (and keep dow/qod at the end)
    feature_cols = [c for c in feature_cols if c not in ("dow", "qod")] + ["dow", "qod"]

    x = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df[target_col].to_numpy(dtype=np.float32, copy=True)

    categorical_features = [feature_cols.index("dow"), feature_cols.index("qod")]

    return LoadedDataset(
        x=x,
        y=y,
        feature_names=feature_cols,
        categorical_features=categorical_features,
        train_end=train_end,
        valid_end=valid_end,
    )


def _liander_time_split_4m_train_2m_valid(
    data: LoadedDataset,
    _seed: int,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
]:
    if data.train_end is None or data.valid_end is None:
        raise ValueError("Liander split requires precomputed train_end/valid_end")

    train_end = data.train_end
    valid_end = data.valid_end
    if train_end <= 0 or valid_end <= train_end or valid_end > len(data.y):
        raise ValueError("Liander split indices are invalid; check time span after joins")

    x_train = data.x[:train_end]
    y_train = data.y[:train_end]
    x_valid = data.x[train_end:valid_end]
    y_valid = data.y[train_end:valid_end]
    return x_train, x_valid, y_train, y_valid


# Pre-defined dataset configurations
DATASETS: dict[str, DatasetConfig] = {
    # Regression
    "california": DatasetConfig(
        name="california",
        task=Task.REGRESSION,
        loader=california_housing,
    ),
    "synthetic_reg_small": DatasetConfig(
        name="synthetic_reg_small",
        task=Task.REGRESSION,
        loader=synthetic_regression_small,
    ),
    "synthetic_reg_medium": DatasetConfig(
        name="synthetic_reg_medium",
        task=Task.REGRESSION,
        loader=synthetic_regression_medium,
    ),
    # Binary classification
    "breast_cancer": DatasetConfig(
        name="breast_cancer",
        task=Task.BINARY,
        loader=breast_cancer,
    ),
    "synthetic_bin_small": DatasetConfig(
        name="synthetic_bin_small",
        task=Task.BINARY,
        loader=synthetic_binary_small,
    ),
    "synthetic_bin_medium": DatasetConfig(
        name="synthetic_bin_medium",
        task=Task.BINARY,
        loader=synthetic_binary_medium,
    ),
    # Multiclass classification
    "iris": DatasetConfig(
        name="iris",
        task=Task.MULTICLASS,
        loader=iris,
        n_classes=3,
    ),
    "covertype": DatasetConfig(
        name="covertype",
        task=Task.MULTICLASS,
        loader=covertype,
        n_classes=7,
        subsample=50000,
    ),
    "synthetic_multi_small": DatasetConfig(
        name="synthetic_multi_small",
        task=Task.MULTICLASS,
        loader=synthetic_multi_small,
        n_classes=5,
    ),
    # Real-world regression (time-series forecasting)
    "liander_energy_os_gorredijk": DatasetConfig(
        name="liander_energy_os_gorredijk",
        task=Task.REGRESSION,
        loader=liander_energy_prices_os_gorredijk,
        splitter=_liander_time_split_4m_train_2m_valid,
        primary_metric="rmae",
        supported_booster_types=[BoosterType.GBDT, BoosterType.GBLINEAR, BoosterType.LINEAR_TREES],
    ),
}


def get_datasets_by_task(task: Task) -> dict[str, DatasetConfig]:
    """Get all datasets for a specific task."""
    return {k: v for k, v in DATASETS.items() if v.task == task}
