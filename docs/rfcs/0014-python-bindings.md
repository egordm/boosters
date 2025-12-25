# RFC-0014: Python Bindings

- **Status**: Accepted
- **Created**: 2025-12-25
- **Depends on**: RFC-0001 through RFC-0013
- **Scope**: Python package for boosters library

## Summary

Python bindings for the boosters library, exposing all major functionality to Python users.
Built with PyO3/Maturin, providing native performance with a Pythonic API. Supports common
data formats (NumPy, pandas, PyArrow) and integrates with scikit-learn.

## Motivation

Python is the primary language for ML practitioners. A Python interface will:

1. **Enable adoption**: Most ML practitioners work in Python
2. **Integrate with ecosystem**: Work with pandas, NumPy, scikit-learn workflows
3. **Maintain performance**: Native Rust core, zero-copy where possible
4. **Expose all features**: GBDT, GBLinear, linear leaves, EFB, categoricals, SHAP

### Goals

- Performance competitive with LightGBM Python bindings
- Clean, Pythonic API (not just a thin wrapper)
- Support for pandas, NumPy, PyArrow, scipy.sparse
- scikit-learn compatible estimators
- Feature parity with native Rust API

### Non-Goals

- Dask/distributed support (future work)
- Custom objective/metric functions in Python (future work)
- Streaming/incremental training (future work)

---

## Python Version and Distribution Policy

### Supported Python Versions

| Python | Support Status |
| ------ | -------------- |
| 3.12 | Supported (minimum) |
| 3.13+ | Added when stable |

**Policy**: Target Python 3.12+ to leverage modern type hints (`type` statements,
`|` unions, `Self`, generics without `typing` imports). Drop versions ~6 months
after upstream end-of-life.

### Wheel Distribution

Pre-built wheels for:

| Platform | Architecture | Notes |
| -------- | ------------ | ----- |
| Linux (manylinux2014) | x86_64 | Primary (glibc) |
| Linux (manylinux2014) | aarch64 | ARM64 servers |
| Linux (musllinux_1_2) | x86_64 | Alpine Linux |
| Linux (musllinux_1_2) | aarch64 | Alpine ARM64 |
| macOS | x86_64 | Intel Macs |
| macOS | arm64 | Apple Silicon |
| Windows | x86_64 | |

Built via GitHub Actions + maturin. Source distribution also available for
unsupported platforms (requires Rust toolchain).

---

## Design Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Python User API                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ boosters.   │  │ boosters.   │  │ boosters.sklearn        │  │
│  │ train()     │  │ Dataset     │  │ BoostersRegressor       │  │
│  │ Booster     │  │             │  │ BoostersClassifier      │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PyO3 Wrapper Layer                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Python ↔ Rust type conversions                              ││
│  │ - NumPy arrays ↔ ndarray views                              ││
│  │ - pandas DataFrame → FeaturesView                           ││
│  │ - PyArrow Table → zero-copy views                           ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    boosters Rust Core                            │
│  GBDTModel, GBLinearModel, Dataset, BinnedDataset               │
│  Training, Inference, Explainability                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```text
packages/boosters-python/
├── Cargo.toml              # PyO3 crate
├── pyproject.toml          # Python package metadata
├── src/
│   ├── lib.rs              # PyO3 module registration
│   ├── dataset.rs          # Dataset wrapper
│   ├── booster.rs          # Booster (model) wrapper
│   ├── config.rs           # Configuration types
│   ├── convert.rs          # Data conversion utilities
│   └── error.rs            # Error handling
└── python/
    └── boosters/
        ├── __init__.py     # Public API exports
        ├── basic.py        # Core types (Dataset, Booster)
        ├── engine.py       # train(), cv() functions
        ├── sklearn.py      # scikit-learn estimators
        ├── compat.py       # Optional dependency handling
        └── plotting.py     # Visualization (optional)
```

---

## Core API Design

### Design Principles

1. **Strict typing**: All public APIs use Python 3.12+ type hints
2. **Dataclass configs**: Configuration via typed dataclasses, not dicts
3. **Pythonic naming**: `n_samples`, `n_features`, `labels` (not `num_data`, `get_label`)
4. **Model separation**: `GBDTModel` and `GBLinearModel` as distinct classes (like Rust)
5. **Google-style docstrings**: Brief, typed, no redundant parameter descriptions
6. **sklearn-like fit/predict**: Config in `__init__`, data in `fit`/`predict`

### Dataset

```python
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
import numpy as np

@dataclass
class Dataset:
    """Training dataset with features, labels, and optional metadata."""
    
    features: NDArray[np.floating] | pd.DataFrame
    labels: NDArray[np.floating] | None = None
    weights: NDArray[np.floating] | None = None
    groups: NDArray[np.integer] | None = None  # For ranking
    feature_names: list[str] | None = None
    categorical_features: list[int | str] | None = None
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
    
    @property
    def n_features(self) -> int:
        """Number of features."""
    
    def __len__(self) -> int:
        return self.n_samples
```

### Configuration Dataclasses

```python
from dataclasses import dataclass, field
from enum import Enum

class Objective(Enum):
    """Loss function for training."""
    SQUARED = "squared"
    ABSOLUTE = "absolute"
    HUBER = "huber"
    QUANTILE = "quantile"
    POISSON = "poisson"
    LOGISTIC = "logistic"
    HINGE = "hinge"
    SOFTMAX = "softmax"
    LAMBDARANK = "lambdarank"

class Metric(Enum):
    """Evaluation metric."""
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    LOGLOSS = "logloss"
    AUC = "auc"
    ACCURACY = "accuracy"
    NDCG = "ndcg"

@dataclass
class TreeConfig:
    """Tree structure configuration."""
    max_depth: int = -1  # -1 = unlimited
    n_leaves: int = 31
    min_samples_leaf: int = 20
    min_gain_to_split: float = 0.0

@dataclass  
class RegularizationConfig:
    """Regularization parameters."""
    l1: float = 0.0
    l2: float = 0.0
    
@dataclass
class SamplingConfig:
    """Row and column sampling."""
    subsample: float = 1.0
    colsample: float = 1.0
    
@dataclass
class GBDTConfig:
    """Configuration for GBDT training."""
    
    # Core
    n_estimators: int = 100
    learning_rate: float = 0.1
    objective: Objective = Objective.SQUARED
    metric: Metric | list[Metric] | None = None
    
    # Tree structure
    tree: TreeConfig = field(default_factory=TreeConfig)
    
    # Regularization
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    
    # Sampling
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    
    # Features
    linear_trees: bool = False
    enable_bundling: bool = True  # EFB
    
    # Runtime
    n_threads: int = 0  # 0 = auto
    seed: int | None = None
    verbose: int = 1

@dataclass
class GBLinearConfig:
    """Configuration for GBLinear training."""
    
    n_estimators: int = 100
    learning_rate: float = 0.1
    objective: Objective = Objective.SQUARED
    metric: Metric | list[Metric] | None = None
    
    # Regularization (critical for linear)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    
    # Runtime
    n_threads: int = 0
    seed: int | None = None
    verbose: int = 1
```

### GBDT Model

```python
class GBDTModel:
    """Gradient Boosted Decision Trees model."""
    
    def __init__(self, config: GBDTConfig | None = None) -> None:
        """Initialize with configuration.
        
        Args:
            config: Training configuration. Uses defaults if None.
        """
    
    def fit(
        self,
        train: Dataset,
        *,
        valid: Dataset | list[Dataset] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> Self:
        """Train the model.
        
        Args:
            train: Training dataset.
            valid: Validation dataset(s) for early stopping.
            callbacks: Training callbacks (early stopping, logging).
        
        Returns:
            Self for method chaining.
        """
    
    def predict(
        self,
        features: NDArray[np.floating] | pd.DataFrame,
        *,
        n_iterations: int | None = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
    ) -> NDArray[np.floating]:
        """Make predictions.
        
        Args:
            features: Feature matrix (n_samples, n_features).
            n_iterations: Limit trees used. None = best_iteration or all.
            raw_score: Return raw margins (no sigmoid/softmax).
            pred_leaf: Return leaf indices instead of predictions.
            pred_contrib: Return SHAP values.
        
        Returns:
            Predictions array.
        """
    
    # Model inspection
    @property
    def n_features(self) -> int: ...
    
    @property
    def n_trees(self) -> int: ...
    
    @property
    def feature_names(self) -> list[str] | None: ...
    
    @property
    def best_iteration(self) -> int | None: ...
    
    @property
    def best_score(self) -> float | None: ...
    
    @property
    def eval_results(self) -> dict[str, list[float]] | None: ...
    
    def feature_importance(
        self,
        importance_type: Literal["gain", "split", "cover"] = "gain",
    ) -> NDArray[np.floating]: ...
    
    def trees_to_dataframe(self) -> pd.DataFrame: ...

class GBLinearModel:
    """Gradient Boosted Linear model."""
    
    def __init__(self, config: GBLinearConfig | None = None) -> None:
        """Initialize with configuration."""
    
    def fit(
        self,
        train: Dataset,
        *,
        valid: Dataset | list[Dataset] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> Self:
        """Train the model."""
    
    def predict(
        self,
        features: NDArray[np.floating] | pd.DataFrame,
        *,
        n_iterations: int | None = None,
        raw_score: bool = False,
    ) -> NDArray[np.floating]:
        """Make predictions."""
    
    @property
    def weights(self) -> NDArray[np.floating]:
        """Model weights (n_features,) or (n_features, n_outputs)."""
    
    @property
    def bias(self) -> float | NDArray[np.floating]:
        """Model bias term."""
```

### Training Function (Alternative API)

For users who prefer functional style:

```python
def train_gbdt(
    config: GBDTConfig,
    train: Dataset,
    *,
    valid: Dataset | list[Dataset] | None = None,
    callbacks: list[Callback] | None = None,
) -> GBDTModel:
    """Train a GBDT model.
    
    Args:
        config: Training configuration.
        train: Training dataset.
        valid: Validation dataset(s).
        callbacks: Training callbacks.
    
    Returns:
        Trained model.
    """

def train_gblinear(
    config: GBLinearConfig,
    train: Dataset,
    *,
    valid: Dataset | list[Dataset] | None = None,
    callbacks: list[Callback] | None = None,
) -> GBLinearModel:
    """Train a GBLinear model."""
```

### Callbacks

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

class Callback(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def __call__(self, iteration: int, eval_results: dict[str, float]) -> bool:
        """Called after each iteration.
        
        Args:
            iteration: Current iteration number.
            eval_results: Metric values for this iteration.
        
        Returns:
            True to stop training, False to continue.
        """

@dataclass
class EarlyStopping(Callback):
    """Stop training when validation metric stops improving."""
    
    patience: int = 50
    min_delta: float = 0.0
    
    def __call__(self, iteration: int, eval_results: dict[str, float]) -> bool: ...

@dataclass
class LogEvaluation(Callback):
    """Log evaluation metrics periodically."""
    
    period: int = 10
    
    def __call__(self, iteration: int, eval_results: dict[str, float]) -> bool: ...
```

---

## Objective and Metric Mapping

### Objective Mapping

| Python `objective` | Rust Type | Notes |
| ------------------ | --------- | ----- |
| `'regression'`, `'mse'`, `'l2'` | `SquaredLoss` | Default for regression |
| `'mae'`, `'l1'` | `AbsoluteLoss` | Robust to outliers |
| `'quantile'` | `PinballLoss` | Requires `alpha` param (default 0.5) |
| `'huber'` | `PseudoHuberLoss` | Requires `delta` param |
| `'poisson'` | `PoissonLoss` | For count data |
| `'binary'`, `'binary_logloss'` | `LogisticLoss` | Binary classification |
| `'hinge'` | `HingeLoss` | SVM-style classification |
| `'multiclass'`, `'softmax'` | `SoftmaxLoss` | Requires `num_class` param |
| `'lambdarank'` | `LambdaRankLoss` | Learning to rank |

### Metric Mapping

| Python `metric` | Rust Type | Task |
| --------------- | --------- | ---- |
| `'rmse'` | `Rmse` | Regression |
| `'mae'` | `Mae` | Regression |
| `'mape'` | `Mape` | Regression |
| `'logloss'`, `'binary_logloss'` | `LogLoss` | Classification |
| `'auc'` | `Auc` | Binary classification |
| `'accuracy'` | `Accuracy` | Classification |
| `'multi_logloss'` | `MultiLogLoss` | Multiclass |
| `'ndcg'` | `Ndcg` | Ranking |

---

## scikit-learn Integration

```python
from boosters.sklearn import GBDTRegressor, GBDTClassifier, GBDTRanker

# Works like any sklearn estimator
model = GBDTRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    n_threads=0,  # 0 = auto
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# With validation for early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[EarlyStopping(patience=50)],
)

# Cross-validation works
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

# Pipeline integration
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GBDTClassifier()),
])
```

### scikit-learn Estimator Classes

```python
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from typing import Self

class GBDTRegressor(BaseEstimator, RegressorMixin):
    """GBDT regressor with sklearn interface.
    
    All constructor parameters become sklearn-compatible attributes.
    Configuration is flattened (no nested dataclasses) for sklearn compatibility.
    """
    
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        n_leaves: int = 31,
        min_samples_leaf: int = 20,
        l1: float = 0.0,
        l2: float = 0.0,
        subsample: float = 1.0,
        colsample: float = 1.0,
        linear_trees: bool = False,
        n_threads: int = 0,
        seed: int | None = None,
        verbose: int = 1,
    ) -> None: ...
    
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
        eval_set: list[tuple[ArrayLike, ArrayLike]] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> Self: ...
    
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
    
    # sklearn-standard attributes (set after fit)
    n_features_in_: int
    feature_names_in_: NDArray[np.str_] | None
    feature_importances_: NDArray[np.floating]
    best_iteration_: int | None

class GBDTClassifier(BaseEstimator, ClassifierMixin):
    """GBDT classifier with sklearn interface."""
    
    def __init__(self, *, **kwargs) -> None: ...  # Same as GBDTRegressor
    
    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray[np.integer]: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]: ...
    
    classes_: NDArray  # Set after fit

class GBDTRanker(BaseEstimator):
    """GBDT ranker for learning-to-rank."""
    
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        group: ArrayLike | None = None,
        **kwargs,
    ) -> Self: ...

class GBLinearRegressor(BaseEstimator, RegressorMixin):
    """GBLinear regressor with sklearn interface."""
    
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        l1: float = 0.0,
        l2: float = 1.0,  # Usually need regularization for linear
        n_threads: int = 0,
        seed: int | None = None,
        verbose: int = 1,
    ) -> None: ...

class GBLinearClassifier(BaseEstimator, ClassifierMixin):
    """GBLinear classifier with sklearn interface."""
```

---

## Data Format Support

### Recommended Input Format

**Best performance**: `pandas.DataFrame` or F-contiguous NumPy array.

```python
import pandas as pd
import numpy as np

# Option 1: pandas DataFrame (recommended)
# - Automatic categorical detection
# - Feature names preserved
# - Optimal memory layout extracted automatically
df = pd.DataFrame(data, columns=feature_names)
model.fit(df, y)

# Option 2: F-contiguous NumPy array (zero-copy)
X = np.asfortranarray(data, dtype=np.float32)
model.fit(X, y)

# Option 3: C-contiguous array (will be transposed internally)
X = np.ascontiguousarray(data, dtype=np.float32)
model.fit(X, y)  # One-time transpose cost
```

### Input Formats

| Format | Support | Conversion Cost |
| ------ | ------- | --------------- |
| pandas DataFrame | ✓ | Optimal (extracts F-contiguous) |
| NumPy F-contiguous f32 | ✓ | Zero copy |
| NumPy C-contiguous f32 | ✓ | Transpose O(n×m) |
| NumPy f64 (any order) | ✓ | Cast + possible transpose |
| PyArrow Table | ✓ | Usually copy (unless f32 chunked) |
| scipy.sparse CSR/CSC | ✓ | Sparse path (efficient) |

### Memory Layout Strategy

**Rust core** uses feature-major layout `[n_features][n_samples]` for cache-efficient
histogram building. The Python bindings handle conversion:

```python
def _convert_features(data: ArrayLike) -> FeatureMatrix:
    """Convert Python data to Rust-compatible format.
    
    Priority:
    1. If pandas DataFrame: extract underlying arrays optimally
    2. If F-contiguous f32: zero-copy view
    3. If C-contiguous: transpose to F-contiguous
    4. Otherwise: copy to F-contiguous f32
    """
    if isinstance(data, pd.DataFrame):
        # pandas stores columns contiguously - optimal for feature-major!
        return _convert_dataframe(data)
    
    arr = np.asarray(data)
    
    if arr.flags['F_CONTIGUOUS'] and arr.dtype == np.float32:
        # Zero-copy path
        return FeatureMatrix.from_f_contiguous(arr)
    
    # Convert to optimal layout
    return FeatureMatrix.from_array(
        np.asfortranarray(arr, dtype=np.float32)
    )
```

**Key insight**: pandas DataFrames naturally store columns contiguously, which
aligns perfectly with our feature-major layout. Recommend DataFrame input in docs.

### Type Handling

```python
# Automatic dtype handling
# f32/f64 → kept as-is (f32 preferred for performance)
# int → converted to f32
# categorical → encoded to int, tracked for proper split handling

# Automatic categorical detection for pandas
df = pd.DataFrame({
    'age': [25, 30, 35],
    'city': pd.Categorical(['NYC', 'LA', 'SF']),  # Auto-detected
    'score': [0.5, 0.8, 0.6],
})
ds = Dataset(df, label=y)  # 'city' automatically categorical
```

---

## PyO3 Implementation

### Architecture Overview

**Separate model types** (matching Rust): `PyGBDTModel` and `PyGBLinearModel` as
distinct `#[pyclass]` types, not a unified `Booster` enum.

```rust
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use boosters::{GBDTModel, GBLinearModel, GBDTConfig, GBLinearConfig};

/// GBDT model wrapper.
#[pyclass(name = "GBDTModel")]
pub struct PyGBDTModel {
    inner: Option<GBDTModel>,
    config: GBDTConfig,
    feature_names: Option<Vec<String>>,
    best_iteration: Option<usize>,
}

#[pymethods]
impl PyGBDTModel {
    #[new]
    fn new(config: Option<PyGBDTConfig>) -> Self {
        Self {
            inner: None,
            config: config.map(Into::into).unwrap_or_default(),
            feature_names: None,
            best_iteration: None,
        }
    }
    
    fn fit(
        &mut self,
        py: Python<'_>,
        train: &PyDataset,
        valid: Option<&PyDataset>,
        callbacks: Option<Vec<PyObject>>,
    ) -> PyResult<()> {
        // Threading is handled here at the bindings level
        let n_threads = self.config.n_threads;
        
        py.allow_threads(|| {
            boosters::run_with_threads(n_threads, |_| {
                // Train model - Rust core does NOT manage threads
                self.inner = Some(GBDTModel::train(&self.config, train.inner())?);
                Ok(())
            })
        })
    }
    
    fn predict<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        n_iterations: Option<usize>,
        raw_score: bool,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = self.inner.as_ref().ok_or_else(|| {
            PyValueError::new_err("Model not trained. Call fit() first.")
        })?;
        
        let n_threads = self.config.n_threads;
        
        py.allow_threads(|| {
            boosters::run_with_threads(n_threads, |_| {
                model.predict(&features.as_array(), n_iterations)
            })
        })
        .map(|arr| PyArray1::from_vec_bound(py, arr))
    }
    
    #[getter]
    fn n_trees(&self) -> PyResult<usize> {
        self.inner.as_ref()
            .map(|m| m.forest().n_trees())
            .ok_or_else(|| PyValueError::new_err("Model not trained"))
    }
}

/// GBLinear model wrapper.
#[pyclass(name = "GBLinearModel")]
pub struct PyGBLinearModel {
    inner: Option<GBLinearModel>,
    config: GBLinearConfig,
    // ... similar structure
}
```

### Threading Strategy

**Key decision**: Threading is managed at the Python bindings level, not in Rust core.

```rust
// Python bindings handle thread pool setup
impl PyGBDTModel {
    fn fit(&mut self, py: Python<'_>, ...) -> PyResult<()> {
        let n_threads = convert_n_threads(self.config.n_threads);
        
        py.allow_threads(|| {
            boosters::run_with_threads(n_threads, |_| {
                // Rust core assumes thread pool is already set up
                // No internal run_with_threads calls in GBDTModel::train
                GBDTModel::train(...)
            })
        })
    }
}

/// Convert Python n_threads convention to Rust.
/// 
/// Python: 0 = auto, -1 = auto, 1 = sequential, n > 1 = n threads
/// Rust:   0 = auto, 1 = sequential, n > 1 = n threads
fn convert_n_threads(n: i32) -> usize {
    match n {
        0 | -1 => 0,  // Auto
        1 => 1,       // Sequential
        n if n > 1 => n as usize,
        _ => 0,       // Negative → auto
    }
}
```

**Implication for Rust core**: The `GBDTModel::train()` and `GBLinearModel::train()`
methods should NOT call `run_with_threads` internally. They should assume the
caller has already set up the thread pool. This allows:

1. Python bindings to manage threads with GIL release
2. Rust CLI/library users to manage threads at their level  
3. No nested thread pool issues

### Dataset Architecture

```rust
#[pyclass]
struct Dataset {
    // Store the raw Python object to prevent garbage collection
    raw_data: Option<PyObject>,
    // Rust representation (built lazily or on construct())
    inner: Option<Arc<boosters::Dataset>>,
    // Metadata
    label: Option<PyObject>,
    weight: Option<PyObject>,
    feature_names: Option<Vec<String>>,
    categorical_features: Vec<usize>,
    // Whether to free raw_data after building inner
    free_raw_data: bool,
}

impl Dataset {
    fn construct(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.inner.is_some() { return Ok(()); }
        
        // Convert raw_data to Rust Dataset
        let features = convert_features(py, &self.raw_data)?;
        let dataset = boosters::Dataset::new(...);
        self.inner = Some(Arc::new(dataset));
        
        // Free raw data if configured
        if self.free_raw_data {
            self.raw_data = None;
        }
        Ok(())
    }
}
```

**Key insight**: Like LightGBM, we store the raw Python object until construction,
then optionally free it. This allows lazy construction while preventing GC issues.

### GIL Management

```rust
// Release GIL during expensive operations
impl PyGBDTModel {
    fn predict(&self, py: Python<'_>, features: PyReadonlyArray2<f32>) -> PyResult<...> {
        let model = self.inner.as_ref().ok_or(...)?;
        let n_threads = self.config.n_threads;
        
        // Release GIL, set up thread pool, run prediction
        py.allow_threads(|| {
            boosters::run_with_threads(n_threads, |_| {
                model.predict(&features.as_array())
            })
        })
    }
}
```

### Error Handling

```rust
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError};

// Map Rust errors to Python exceptions
impl From<boosters::Error> for PyErr {
    fn from(err: boosters::Error) -> PyErr {
        match err {
            boosters::Error::InvalidParameter(msg) => PyValueError::new_err(msg),
            boosters::Error::ShapeMismatch(msg) => PyValueError::new_err(msg),
            boosters::Error::NotFitted => PyRuntimeError::new_err("Model not fitted"),
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}
```

---

## Design Decisions

### DD-1: PyO3 + Maturin

**Decision**: Use PyO3 for bindings, Maturin for build.

**Rationale**: Single build step, excellent numpy integration, active ecosystem.

### DD-2: Separate Model Types

**Decision**: `GBDTModel` and `GBLinearModel` as separate classes (not unified `Booster`).

**Rationale**: Matches Rust API, clearer type safety, avoid enum dispatch overhead.

### DD-3: Dataclass Configuration

**Decision**: Use `@dataclass` for configuration, not dict-based params.

**Rationale**:

- Full type safety and IDE autocomplete
- Validation at construction time
- Nested configs for organization (TreeConfig, RegularizationConfig)
- Modern Python 3.12+ style

### DD-4: Pythonic Naming

**Decision**: Use Python conventions (`n_samples`, `n_features`, `labels`).

**Rationale**:

- Consistent with sklearn and numpy
- `n_` prefix standard for counts
- Plural for collections (`labels`, `weights`, `feature_names`)

### DD-5: pandas DataFrame Recommended

**Decision**: Recommend DataFrame input, document as optimal path.

**Rationale**:

- pandas stores columns contiguously (matches feature-major layout)
- Automatic categorical detection
- Feature names preserved
- Most users already have DataFrames

### DD-6: Threading at Bindings Level

**Decision**: Python bindings manage thread pool, Rust core is thread-agnostic.

**Rationale**:

- GIL must be released in Python layer anyway
- Avoids nested thread pool issues
- Rust library users can manage threads their way
- Clear separation of concerns

### DD-7: Strict Typing + Google Docstrings

**Decision**: Full Python 3.12+ type hints, Google-style docstrings.

**Rationale**:

- Type hints enable IDE autocomplete and static analysis
- Google style is concise (no duplicate type info in docstrings)
- Python 3.12+ unlocks cleaner syntax (`type X = ...`, `Self`)

### DD-8: sklearn Estimators Use Flat Config

**Decision**: sklearn estimators take flat kwargs (not nested dataclasses).

**Rationale**:

- sklearn's `get_params()`/`set_params()` expect flat attributes
- GridSearchCV and similar tools require flat parameter space
- Core API uses dataclasses, sklearn API flattens them

---

## Explainability API

### SHAP Values

```python
# Via prediction API (like LightGBM)
shap_values = model.predict(X, pred_contrib=True)
# Shape: (n_samples, n_features + 1) for regression
# Shape: (n_samples, n_features + 1, n_classes) for multiclass
# Last feature column is the base value (expected value)

# Or via explicit API
from boosters import TreeExplainer

explainer = TreeExplainer(model)
shap_values = explainer.shap_values(X)
base_value = explainer.expected_value
```

---

## Multi-Output Support

### Classification

```python
from boosters import GBDTModel, GBDTConfig, Objective

# Binary classification
config = GBDTConfig(objective=Objective.LOGISTIC)
model = GBDTModel(config)
model.fit(train)
probs = model.predict(X)  # (n_samples,) - probabilities

# Multiclass classification
config = GBDTConfig(
    objective=Objective.SOFTMAX,
    n_classes=3,
)
model = GBDTModel(config)
model.fit(train)
probs = model.predict(X)              # (n_samples, n_classes)
logits = model.predict(X, raw_score=True)  # Raw logits
```

### Multi-Output Regression

```python
# Native multi-output (one tree per output per round)
config = GBDTConfig(
    objective=Objective.SQUARED,
    n_outputs=2,
)
model = GBDTModel(config)
model.fit(train)
preds = model.predict(X)  # (n_samples, n_outputs)
```

---

## Model Inspection

### Feature Importance

```python
# Multiple importance types
importance_split = model.feature_importance(importance_type='split')
importance_gain = model.feature_importance(importance_type='gain')

# As DataFrame with feature names
import pandas as pd
importance_df = pd.DataFrame({
    'feature': model.feature_names,
    'importance': model.feature_importance('gain'),
}).sort_values('importance', ascending=False)
```

### Tree Structure Export

```python
# Export trees to DataFrame
trees_df = model.trees_to_dataframe()
# Columns: tree_index, node_depth, node_index, left_child, right_child,
#          parent_index, split_feature, split_gain, threshold, 
#          decision_type, missing_direction, n_samples, value

# Filter to specific tree
tree_0 = trees_df[trees_df['tree_index'] == 0]
```

### Model Attributes

```python
# Core model API attributes (after fit):
model.n_features       # int: Number of features
model.n_trees          # int: Number of trees trained
model.feature_names    # list[str] | None: Feature names
model.best_iteration   # int | None: Best iteration (early stopping)
model.best_score       # float | None: Best validation score
model.eval_results     # dict[str, list[float]] | None: Training history

# sklearn estimator attributes (after fit):
# These follow sklearn naming conventions with trailing underscore
estimator.n_features_in_      # int
estimator.feature_names_in_   # ndarray | None
estimator.feature_importances_  # ndarray
estimator.classes_            # ndarray (classifier only)
estimator.best_iteration_     # int | None
```

---

## Categorical Feature Handling

### Auto-Detection

```python
# Pandas categoricals are auto-detected
df = pd.DataFrame({
    'numeric': [1.0, 2.0, 3.0],
    'category': pd.Categorical(['a', 'b', 'c']),
})
dataset = Dataset(features=df, labels=y)
# dataset.categorical_features == [1]  # index of 'category' column

# Or specify manually
dataset = Dataset(features=X, labels=y, categorical_features=[0, 5])
```

### Category Value Preservation

```python
# Original categories are preserved for interpretability
dataset = Dataset(features=df, labels=y)
model = GBDTModel()
model.fit(dataset)

# Access category mappings
print(model.category_encodings)
# {'category': ['a', 'b', 'c']}  # Original category order

# Tree export shows category names in thresholds
trees_df = model.trees_to_dataframe()
```

### Integer-Encoded Categoricals

```python
# If data is already integer-encoded, specify feature indices
dataset = Dataset(features=X, labels=y, categorical_features=[2, 5])
# Values are treated as category indices (0, 1, 2, ...)
```

### Unseen Categories

```python
# Categories not seen during training are treated as missing
train_df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'c'])})
test_df = pd.DataFrame({'cat': pd.Categorical(['a', 'd'])})  # 'd' is unseen

train_ds = Dataset(features=train_df, labels=y)
model = GBDTModel()
model.fit(train_ds)

pred = model.predict(test_df)  # 'd' uses missing-value path
```

---

## Parameter Validation and Edge Cases

### Parameter Validation

With dataclass configuration, validation happens at construction time:

```python
from boosters import GBDTConfig, TreeConfig

# Type errors caught by dataclass
config = GBDTConfig(learning_rate='fast')
# TypeError: learning_rate must be float, got str

# Value validation in __post_init__
config = GBDTConfig(n_estimators=-1)
# ValueError: n_estimators must be positive, got -1

# Nested config validation
config = GBDTConfig(tree=TreeConfig(n_leaves=0))
# ValueError: n_leaves must be positive, got 0
```

### Edge Cases

| Input | Behavior |
| ----- | -------- |
| Empty array (0 samples) | ValueError |
| Single sample | Works (min_samples_leaf=1 required) |
| NaN in features | Uses missing value handling |
| Inf in features | ValueError (must be finite) |
| NaN in labels | ValueError |
| All-constant feature | Skipped during split finding |
| Mismatched shapes | ValueError with clear message |
| Wrong n_features at predict | ValueError: expected N features, got M |

---

## Serialization

**Note:** Serialization support depends on RFC for storage formats (not yet implemented
in Rust core). This section documents planned API when available.

### Planned API (Future)

```python
# File-based serialization
model.save('model.boosters')     # Native binary format
model.save('model.json')         # JSON format (human-readable)

loaded = GBDTModel.load('model.boosters')

# Pickle support (requires serialization to bytes)
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

**Timeline:** Blocked on Rust serialization format RFC (see backlog/05-storage-format.md).

---

## API Compatibility Matrix

| Feature | boosters | LightGBM | Notes |
| ------- | -------- | -------- | ----- |
| Basic GBDT | ✓ | ✓ | Parity |
| GBLinear | ✓ | ✗ | boosters-only |
| Linear leaves | ✓ | ✓ | Parity |
| Native categorical | ✓ | ✓ | Parity |
| Feature bundling | ✓ | ✓ | Parity |
| SHAP values | ✓ | ✓ | Via `pred_contrib` |
| Feature importance | ✓ | ✓ | Parity |
| Early stopping | ✓ | ✓ | Via callbacks |
| Custom objectives | Future | ✓ | Planned |
| Dask integration | Future | ✓ | Planned |

---

## Performance Considerations

### Thread Management

Threading is managed at the Python bindings level (see DD-6). The `n_threads`
parameter uses sklearn-compatible semantics:

```python
n_threads = 0   # Auto (all available cores)
n_threads = -1  # Same as 0 (sklearn convention)
n_threads = 1   # Sequential (no parallelism)
n_threads = 4   # Exactly 4 threads
```

The conversion happens in Python before calling Rust:

```python
def _to_rust_threads(n_threads: int) -> int:
    """Convert sklearn n_threads to Rust convention."""
    if n_threads <= 0:
        return 0  # Auto in Rust
    return n_threads
```

### Data Conversion Performance

| Input Type | Conversion Cost | Notes |
| ---------- | --------------- | ----- |
| pandas DataFrame | Optimal | Columns already contiguous |
| F-contiguous f32 array | Zero copy | Optimal path |
| C-contiguous f32 array | Transpose O(n×m) | Cache-unfriendly |
| F-contiguous f64 array | Cast O(n×m) | One pass |
| C-contiguous f64 array | Cast + transpose | Two passes |
| pandas DataFrame | Extract + possible transpose | Categorical detection overhead |
| PyArrow Table | Usually copy | Zero-copy only if f32, no nulls, contiguous |
| scipy.sparse CSR | View (sparse path) | Best for >90% sparse |
| PyArrow Table | Zero copy possible | If Arrow layout matches |
| scipy.sparse CSR | Direct use | Sparse path |

### Memory Guidelines

```python
# Estimate memory before training
n_samples, n_features = 1_000_000, 100
n_bins = 256

# Raw data: float32
raw_mb = n_samples * n_features * 4 / 1e6

# Binned data: uint8 per bin (typically)
binned_mb = n_samples * n_features * 1 / 1e6

# Histograms per leaf: 2 * float64 per bin per feature
hist_mb = n_features * n_bins * 16 / 1e6

print(f"Raw data: {raw_mb:.1f} MB")
print(f"Binned data: {binned_mb:.1f} MB") 
print(f"Histograms/leaf: {hist_mb:.3f} MB")
```

---

## Future Work

### v1.1 (Near-term)

- [ ] Pickle serialization support
- [ ] Plotting utilities (tree visualization, feature importance)
- [ ] SHAP summary/dependency plots

### v1.2 (Medium-term)

- [ ] Custom objective/metric functions in Python
- [ ] Additional callbacks (checkpointing, LR scheduling)
- [ ] Model comparison utilities

### v2.0 (Long-term)

- [ ] Dask distributed training
- [ ] GPU support (CUDA)
- [ ] Streaming/incremental training
- [ ] Programmatic tree construction

---

## Testing Strategy

### Test Categories

1. **Unit Tests** (pytest, fast):
   - Data conversion: NumPy, pandas, PyArrow, scipy.sparse
   - Parameter parsing and validation
   - Basic train/predict flow

2. **Integration Tests** (pytest, medium):
   - sklearn estimator compliance (`check_estimator()`)
   - Cross-validation compatibility
   - Pipeline integration
   - Early stopping callbacks

3. **Numerical Parity Tests** (pytest, slow):
   - Python predictions match Rust predictions exactly
   - Compare with LightGBM on standard datasets
   - SHAP value consistency

4. **Round-Trip Tests**:
   - Train in Python → Save → Load in Rust → Predict
   - Train in Rust → Save → Load in Python → Predict
   - pandas categorical preservation

5. **Memory/Performance Tests**:
   - Memory leak detection (memray for Python)
   - GIL release verification
   - Large dataset handling

6. **Thread Safety Tests**:
   - Concurrent `predict()` calls from multiple Python threads
   - Concurrent training (should block or error, not corrupt)
   - Dataset shared across threads (read-only safe)

### CI Matrix

| Python | OS | Test Scope |
| ------ | -- | ---------- |
| 3.12 | Linux, macOS, Windows | Full |
| 3.13 | Linux | Full |

### sklearn Compliance

```python
from sklearn.utils.estimator_checks import check_estimator

def test_sklearn_compliance():
    # Run standard sklearn estimator checks
    for estimator in [GBDTRegressor(), GBDTClassifier()]:
        for check in check_estimator(estimator, generate_only=True):
            check(estimator)
```

---

## Changelog

- 2025-12-25: Initial draft
- 2025-12-25: Rounds 1-6 - Design review (memory, threading, testing, etc.)
- 2025-12-25: Major revision based on feedback:
  - Python 3.12+ only (leverage modern type hints)
  - Alpine/musl support added
  - Separate GBDTModel/GBLinearModel classes (match Rust)
  - Dataclass-based configuration instead of dicts
  - Pythonic naming (n_samples, labels, feature_names)
  - pandas DataFrame recommended as optimal input
  - Threading managed at bindings level (Rust core thread-agnostic)
  - Strict typing with Google-style docstrings
  - Serialization deferred until Rust storage format RFC
