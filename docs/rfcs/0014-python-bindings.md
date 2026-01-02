# RFC-0014: Python Bindings

**Status**: Implemented  
**Created**: 2025-12-01  
**Updated**: 2026-01-02  
**Scope**: PyO3 bindings for boosters

## Summary

Python bindings via PyO3 expose `GBDTModel`, `GBLinearModel`, `Dataset`, and
configuration types. The API is Pythonic with scikit-learn compatibility.

## Why Python Bindings?

| Aspect | Benefit |
| ------ | ------- |
| Adoption | ML ecosystem is Python-dominated |
| Integration | Works with pandas, numpy, scikit-learn |
| Iteration | Fast prototyping, then production |
| Comparison | Benchmark against XGBoost/LightGBM |

## Package Structure

```
packages/boosters-python/
├── src/                    # Rust PyO3 bindings
│   ├── lib.rs              # Module definition
│   ├── config.rs           # PyGBDTConfig, PyGBLinearConfig
│   ├── data/               # Dataset bindings
│   ├── model/              # Model bindings
│   └── ...
└── python/boosters/        # Python wrapper layer
    ├── __init__.py         # Public exports
    ├── dataset/            # Dataset + DatasetBuilder
    ├── model.py            # Model re-exports
    ├── sklearn/            # scikit-learn estimators
    └── ...
```

Two-layer design:
- **Rust layer** (`_boosters_rs`): Raw bindings, efficient memory handling
- **Python layer** (`boosters`): Pythonic API, input validation, convenience

## Core Types

### Dataset

The `Dataset` constructor accepts numpy arrays, pandas DataFrames, or scipy
sparse matrices. Internally, it delegates to `DatasetBuilder`:

```python
import boosters as bst
import numpy as np

# From numpy array (most common)
dataset = bst.Dataset(X, y, weights=None)

# From pandas DataFrame
dataset = bst.Dataset(df.drop("target", axis=1), df["target"])

# From scipy sparse matrix
dataset = bst.Dataset(sparse_X, y)

# With optional metadata
dataset = bst.Dataset(
    X, y,
    feature_names=["age", "income", "category"],
    categorical_features=[2],  # index of categorical column
)
```

### DatasetBuilder

For advanced use cases with mixed feature types:

```python
from boosters.dataset import DatasetBuilder, Feature

# Builder pattern for complex datasets
dataset = (
    DatasetBuilder()
    .add_feature(Feature.from_dense(age_arr, name="age"))
    .add_feature(Feature.from_sparse(indices, values, n_samples, name="category", categorical=True))
    .labels(y)
    .weights(sample_weights)
    .build()
)

# Or add features in bulk
builder = DatasetBuilder()
builder.add_features_from_array(X_dense, feature_names=["f0", "f1", "f2"])
builder.add_features_from_dataframe(df)
builder.add_features_from_sparse(csc_matrix)
builder.labels(y)
dataset = builder.build()
```

### Feature

Individual feature columns with type metadata:

```python
from boosters.dataset import Feature

# Dense feature from array-like
f1 = Feature.from_dense(np.array([1.0, 2.0, 3.0]), name="age")

# Sparse feature from indices + values
f2 = Feature.from_sparse(
    indices=np.array([0, 2]),      # Non-zero sample indices
    values=np.array([1.0, 3.0]),   # Non-zero values
    n_samples=3,
    name="sparse_col",
    categorical=True,
)
```

### Configuration

```python
config = bst.GBDTConfig(
    n_trees=100,
    learning_rate=0.1,
    max_depth=6,
    reg_lambda=1.0,
    objective=bst.Objective.squared_error(),
    metric=bst.Metric.rmse(),
)

linear_config = bst.GBLinearConfig(
    n_rounds=100,
    learning_rate=0.5,
    alpha=0.0,
    lambda_=1.0,
)
```

### Models

```python
# Create model with config
model = bst.GBDTModel(config)

# Train
model.fit(train_data, val_set=valid_data, n_threads=4)

# Predict (returns transformed predictions, e.g., probabilities)
predictions = model.predict(test_data, n_threads=4)

# Raw predictions (untransformed margin scores)
raw_scores = model.predict_raw(test_data)

# Feature importance
importances = model.feature_importance(bst.ImportanceType.Gain)

# SHAP values (shape: [n_samples, n_features+1, n_outputs])
shap_values = model.shap_values(test_data)
```

Note: There is no `model.predict_proba()` on the core model. The `predict()`
method already applies the appropriate transformation (sigmoid for binary,
softmax for multiclass). Use sklearn wrappers for `predict_proba()`.

## scikit-learn API

```python
from boosters.sklearn import GBDTRegressor, GBDTClassifier

# Standard sklearn interface
model = GBDTRegressor(n_trees=100, max_depth=6)
model.fit(X, y)
preds = model.predict(X_test)

# Classifiers have predict_proba
clf = GBDTClassifier(n_trees=100)
clf.fit(X, y)
proba = clf.predict_proba(X_test)  # [n_samples, n_classes]

# Compatible with pipelines, cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
```

## Memory Model

### Zero-Copy Where Possible

```python
# NumPy array: borrowed if C-contiguous float32, else copied
dataset = bst.Dataset(X, y)  # Check dataset.was_converted

# Predictions: returns new numpy array (owned by Python)
preds = model.predict(dataset)  # np.ndarray
```

### Explicit Copies

```python
# Sparse: data is copied into internal CSC format
dataset = bst.Dataset(sparse_matrix, y)

# DataFrame: column data extracted and stored
dataset = bst.Dataset(df, y)
```

## Threading

```python
# Training parallelism
model.fit(train, n_threads=8)

# Prediction parallelism
preds = model.predict(test, n_threads=8)

# Default: use all available cores
preds = model.predict(test)  # n_threads=0 = auto
```

Rayon thread pool is managed internally. `n_threads=1` forces sequential.

**GIL handling**: The GIL is released during training and prediction. Python
threads can run concurrently with Rust computation.

## Error Handling

```python
try:
    model.fit(invalid_data)
except bst.BoostersError as e:
    print(f"Training failed: {e}")
```

Rust errors are converted to Python exceptions with context.

## Stub Generation

Type stubs (`.pyi`) are auto-generated via `pyo3-stub-gen`:

```python
# IDE gets full type info
def fit(
    self,
    train: Dataset,
    val_set: Dataset | None = None,
    n_threads: int = 0,
) -> GBDTModel: ...
```

## Files

| Path | Contents |
| ---- | -------- |
| `src/lib.rs` | PyO3 module registration |
| `src/config.rs` | Config bindings |
| `src/data/` | Dataset, DatasetBuilder, Feature bindings |
| `src/model/` | GBDTModel, GBLinearModel bindings |
| `python/boosters/__init__.py` | Public API |
| `python/boosters/dataset/` | Dataset, DatasetBuilder, Feature |
| `python/boosters/sklearn/` | scikit-learn estimators |

## Design Decisions

**DD-1: Two-layer design.** Rust layer is minimal, Python layer adds
ergonomics. Easier to iterate on Python API without recompiling.

**DD-2: Dataset constructor delegates to builder.** The `Dataset(X, y)`
constructor internally uses `DatasetBuilder` for flexibility, but provides
a simple interface for common cases.

**DD-3: Config objects, not kwargs.** Explicit config types provide IDE
support and validation. Matches XGBoost's approach.

**DD-4: Thread pool managed internally.** Users specify `n_threads`, Rayon
handles pool. No global state leakage between calls.

**DD-5: sklearn wrappers for predict_proba.** The core `GBDTModel.predict()`
already applies transformations. The sklearn `GBDTClassifier.predict_proba()`
is provided separately for sklearn compatibility.

**DD-6: Feature-major layout.** Datasets store features in column-major order
for cache-friendly binning and histogram construction.

## Installation

```bash
# From source
cd packages/boosters-python
maturin develop --release
```

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Type conversion | numpy ↔ Rust array roundtrip |
| Memory safety | No leaks under repeated calls |
| sklearn compliance | sklearn `check_estimator` passes |
| Error handling | Rust errors → Python exceptions |
| Threading | No deadlocks, correct results |

Tests in `packages/boosters-python/tests/`.
