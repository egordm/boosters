"""Dataset configurations for benchmarks."""

from __future__ import annotations

from functools import lru_cache, partial
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
def california_housing_with_nans(
    *,
    nan_fraction: float = 0.1,
    seed: int = 42,
) -> LoadedDataset:
    """California housing regression with NaNs in feature columns.

    Missingness is partially correlated with the target to make the NaN pattern
    informative for models that support missing values.
    """
    if nan_fraction < 0.0 or nan_fraction >= 1.0:
        raise ValueError("nan_fraction must be in [0.0, 1.0)")

    data = cast(Any, fetch_california_housing())
    x = data.data.astype(np.float32, copy=True)
    y = data.target.astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]
    feature_names = [f"f{i}" for i in range(x.shape[1])]

    rng = np.random.default_rng(seed)

    # Target-correlated missingness: hide f0 for ~half of higher-target samples.
    median = float(np.median(y))
    hi = np.flatnonzero(y > median)
    if len(hi) > 0:
        chosen = rng.choice(hi, size=len(hi) // 2, replace=False)
        x[chosen, 0] = np.nan

    # Random missingness across the rest of the matrix.
    if nan_fraction > 0.0:
        mask = rng.random(x.shape) < nan_fraction
        x[mask] = np.nan

    return LoadedDataset(
        x=x,
        y=y,
        feature_names=feature_names,
    )


@lru_cache(maxsize=1)
def california_housing_weighted(
    *,
    seed: int = 42,
) -> LoadedDataset:
    """California housing regression with sample weights.

    Weights emphasize higher-target examples to exercise weighted training.
    """
    data = cast(Any, fetch_california_housing())
    x = data.data.astype(np.float32)
    y = data.target.astype(np.float32)  # pyright: ignore[reportAttributeAccessIssue]
    feature_names = [f"f{i}" for i in range(x.shape[1])]

    rng = np.random.default_rng(seed)

    # Emphasize upper quintile; add small noise so weights are not all identical.
    q80 = float(np.quantile(y, 0.8))
    w = np.ones(len(y), dtype=np.float32)
    w[y >= q80] = 5.0
    w *= 1.0 + rng.normal(loc=0.0, scale=0.01, size=len(y)).astype(np.float32)
    w = np.clip(w, 0.01, None)

    return LoadedDataset(
        x=x,
        y=y,
        sample_weight=w,
        feature_names=feature_names,
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


@lru_cache(maxsize=1)
def liander_energy_forecasting(
    *,
    repo_id: str = "OpenSTEF/liander2024-energy-forecasting-benchmark",
    target: str = "mv_feeder/OS Gorredijk",
    cache_dir: str | None = None,
) -> LoadedDataset:
    """Liander energy forecasting benchmark (subset) as a regression dataset."""
    try:
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required for the Liander benchmark. "
            "Install with: uv add -p packages/boosters-eval huggingface_hub"
        ) from e

    def preprocess_part(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in df.columns:
            raise ValueError("Expected a 'timestamp' column")

        out = df.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"])

        if "available_at" in out.columns:
            out["available_at"] = pd.to_datetime(out["available_at"], utc=True, errors="coerce")
            out = out.sort_values(["timestamp", "available_at"], na_position="first").drop_duplicates(
                "timestamp", keep="last"
            )
            out = out.drop(columns=["available_at"], errors="ignore")
        else:
            out = out.sort_values(["timestamp"]).drop_duplicates("timestamp", keep="last")

        return out.set_index("timestamp", drop=True).sort_index()

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

    load_df = load_df.pipe(preprocess_part)
    weather_df = weather_df.pipe(preprocess_part)
    epex_df = epex_df.pipe(preprocess_part)
    profiles_df = profiles_df.pipe(preprocess_part)

    target_col = "load"
    if target_col not in load_df.columns:
        raise ValueError("Expected 'load' column in load_measurements parquet")

    # Join (load as base to define the target)
    df = load_df.join(weather_df, how="inner").join(epex_df, how="left").join(profiles_df, how="left")
    df = df.sort_index()

    # Add categorical seasonality features
    # Use Series.dt accessors for better type checking support.
    ts = pd.Series(pd.to_datetime(df.index), index=df.index)
    df["day_of_week"] = ts.dt.dayofweek.astype(np.int16).to_numpy()
    df["quarter_of_day"] = ((ts.dt.hour * 60 + ts.dt.minute) // 15).astype(np.int16).to_numpy()
    df["lag_7d"] = df[target_col].shift(7 * 24 * 4)  # 7 days ago lag
    df["lag_14d"] = df[target_col].shift(14 * 24 * 4)  # 14 days ago lag

    # Keep a fixed 4-month window after skipping the first 2 months.
    # 15-minute resolution => 96 samples/day. We use 30-day months for simple iloc slicing.
    samples_per_month = 30 * 24 * 4
    start = 2 * samples_per_month
    end = start + (4 * samples_per_month)
    if len(df) < end:
        raise ValueError(
            f"Liander dataset is shorter than expected after joins; need at least {end} samples, got {len(df)}."
        )
    df = df.iloc[start:end]

    # Build X/y
    exclude_cols = {target_col}
    exclude_cols.update({c for c in df.columns if c.endswith(("_x", "_y"))})

    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = sorted(feature_cols)
    categorical_feature_names = ["day_of_week", "quarter_of_day"]
    categorical_features = [feature_cols.index(name) for name in categorical_feature_names]

    x = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df[target_col].to_numpy(dtype=np.float32, copy=True)

    return LoadedDataset(
        x=x,
        y=y,
        feature_names=feature_cols,
        categorical_features=categorical_features,
    )


def rolling_origin_splitter(
    data: LoadedDataset,
    seed: int,
    *,
    train_window: int,
    valid_window: int,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]] | None,
    NDArray[np.floating[Any]] | None,
]:
    """Time-series split with a rolling cut point.

    Picks a cut point (end of train) based on seed, then returns:
      train = [cut - train_window : cut]
      valid = [cut : cut + valid_window]

    This matches forecasting evaluation where the validation horizon directly
    follows the training window.
    """
    if train_window <= 0:
        raise ValueError("train_window must be positive")
    if valid_window <= 0:
        raise ValueError("valid_window must be positive")

    n = len(data.y)
    if n < train_window + valid_window:
        raise ValueError(
            "Time-series split requires enough samples for both windows; "
            f"need at least {train_window + valid_window}, got {n}."
        )

    rng = np.random.default_rng(seed)
    cut = int(rng.integers(low=train_window, high=(n - valid_window + 1)))

    train_start = cut - train_window
    train_end = cut
    valid_end = cut + valid_window

    return (
        data.x[train_start:train_end],
        data.x[train_end:valid_end],
        data.y[train_start:train_end],
        data.y[train_end:valid_end],
        (data.sample_weight[train_start:train_end] if data.sample_weight is not None else None),
        (data.sample_weight[train_end:valid_end] if data.sample_weight is not None else None),
    )


def split_at_index_splitter(
    data: LoadedDataset,
    _seed: int,
    *,
    train_end: int,
    valid_end: int | None = None,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]] | None,
    NDArray[np.floating[Any]] | None,
]:
    """Create a deterministic split function based on sample indices."""
    end = valid_end if valid_end is not None else len(data.y)
    if train_end <= 0 or end <= train_end or end > len(data.y):
        raise ValueError("Split indices are invalid")
    return (
        data.x[:train_end],
        data.x[train_end:end],
        data.y[:train_end],
        data.y[train_end:end],
        (data.sample_weight[:train_end] if data.sample_weight is not None else None),
        (data.sample_weight[train_end:end] if data.sample_weight is not None else None),
    )


def split_by_precomputed_indices(
    data: LoadedDataset,
    _seed: int,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]] | None,
    NDArray[np.floating[Any]] | None,
]:
    """Split using LoadedDataset.train_end/valid_end."""
    if data.train_end is None or data.valid_end is None:
        raise ValueError("Dataset split requires precomputed train_end/valid_end")

    return split_at_index_splitter(data, _seed, train_end=data.train_end, valid_end=data.valid_end)


# Pre-defined dataset configurations
DATASETS: dict[str, DatasetConfig] = {
    # Regression
    "california": DatasetConfig(
        name="california",
        task=Task.REGRESSION,
        loader=california_housing,
    ),
    "california_nan": DatasetConfig(
        name="california_nan",
        task=Task.REGRESSION,
        loader=california_housing_with_nans,
    ),
    "california_weighted": DatasetConfig(
        name="california_weighted",
        task=Task.REGRESSION,
        loader=california_housing_weighted,
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
    "liander_energy_forecasting": DatasetConfig(
        name="liander_energy_forecasting",
        task=Task.QUANTILE_REGRESSION,
        quantiles=[0.95, 0.8, 0.7, 0.5, 0.3, 0.2, 0.05],
        loader=liander_energy_forecasting,
        splitter=partial(
            rolling_origin_splitter,
            train_window=60 * 24 * 4,  # 60 days
            valid_window=7 * 24 * 4,  # 7 days
        ),
        allow_random_subsample=False,
        primary_metric="rcrps",
        supported_booster_types=[BoosterType.GBDT, BoosterType.GBLINEAR, BoosterType.LINEAR_TREES],
    ),
}


def get_datasets_by_task(task: Task) -> dict[str, DatasetConfig]:
    """Get all datasets for a specific task."""
    return {k: v for k, v in DATASETS.items() if v.task == task}
