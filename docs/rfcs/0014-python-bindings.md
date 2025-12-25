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
| 3.9 | Supported (minimum) |
| 3.10 | Supported |
| 3.11 | Supported (primary) |
| 3.12 | Supported |
| 3.13+ | Added when stable |

**Policy**: Support Python versions with security updates. Drop versions ~6 months
after upstream end-of-life.

### Wheel Distribution

Pre-built wheels for:

| Platform | Architecture | Notes |
| -------- | ------------ | ----- |
| Linux (manylinux2014) | x86_64 | Primary |
| Linux (manylinux2014) | aarch64 | ARM64 servers |
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

### Dataset

Wrapper around Rust `Dataset` that accepts various input formats.

```python
class Dataset:
    """Dataset for boosters training.
    
    Parameters
    ----------
    data : array-like, pandas DataFrame, or PyArrow Table
        Feature matrix. Shape (n_samples, n_features).
    label : array-like, optional
        Target values. Shape (n_samples,) or (n_samples, n_outputs).
    weight : array-like, optional
        Sample weights.
    categorical_feature : list of str or int, optional
        Indices or names of categorical features.
    feature_name : list of str, optional
        Feature names.
    """
    
    def __init__(
        self,
        data,
        label=None,
        weight=None,
        categorical_feature=None,
        feature_name='auto',
    ): ...
    
    @property
    def num_data(self) -> int:
        """Number of samples."""
    
    @property
    def num_feature(self) -> int:
        """Number of features."""
    
    def get_label(self) -> np.ndarray: ...
    def get_weight(self) -> np.ndarray | None: ...
    def get_feature_name(self) -> list[str]: ...
```

### Booster

Core model class wrapping `GBDTModel` or `GBLinearModel`.

```python
class Booster:
    """Boosters model.
    
    Can be created via train() or loaded from file.
    """
    
    @classmethod
    def from_file(cls, filename: str) -> 'Booster':
        """Load model from file."""
    
    def save_model(self, filename: str) -> None:
        """Save model to file."""
    
    def predict(
        self,
        data,
        num_iteration: int | None = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Make predictions.
        
        Parameters
        ----------
        data : array-like, pandas DataFrame, or PyArrow Table
            Features to predict on.
        num_iteration : int, optional
            Limit iterations used. None = use best_iteration or all.
        raw_score : bool
            Return raw margin scores (no transformation).
        pred_leaf : bool
            Return leaf indices instead of predictions.
        pred_contrib : bool
            Return SHAP values for each prediction.
        n_jobs : int
            Number of threads. -1 = auto.
        """
    
    def feature_importance(
        self,
        importance_type: str = 'gain',
    ) -> np.ndarray:
        """Get feature importances.
        
        Parameters
        ----------
        importance_type : str
            'gain', 'split', 'cover', 'average_gain', 'average_cover'
        """
    
    @property
    def feature_name(self) -> list[str]: ...
    
    @property
    def num_trees(self) -> int: ...
    
    @property
    def best_iteration(self) -> int | None: ...
```

### Training Function

```python
def train(
    params: dict,
    train_set: Dataset,
    num_boost_round: int = 100,
    valid_sets: list[Dataset] | None = None,
    valid_names: list[str] | None = None,
    callbacks: list[Callable] | None = None,
    init_model: str | Booster | None = None,
) -> Booster:
    """Train a boosters model.
    
    Parameters
    ----------
    params : dict
        Training parameters. See Parameters documentation.
    train_set : Dataset
        Training data.
    num_boost_round : int
        Number of boosting iterations.
    valid_sets : list of Dataset, optional
        Validation sets for early stopping.
    valid_names : list of str, optional
        Names for validation sets.
    callbacks : list of callable, optional
        Callbacks for logging, early stopping, etc.
    init_model : str or Booster, optional
        Continue training from existing model.
    """
```

### Parameters

Configuration via dictionary (like LightGBM):

```python
params = {
    # Core
    'objective': 'regression',        # See objective mapping below
    'metric': 'rmse',                 # See metric mapping below
    'num_leaves': 31,
    'max_depth': -1,                  # -1 = unlimited
    'learning_rate': 0.1,
    'n_estimators': 100,
    
    # Regularization
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'min_gain_to_split': 0.0,
    'min_data_in_leaf': 20,
    
    # Sampling
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    
    # Categorical
    'categorical_feature': 'auto',
    
    # Advanced
    'booster': 'gbdt',               # 'gbdt' or 'gblinear'
    'linear_tree': False,            # Enable linear leaves
    'enable_bundle': True,           # EFB for sparse data
    
    # Threading
    'n_jobs': -1,                    # -1 = auto
    'verbose': 1,
}
```

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
from boosters.sklearn import BoostersRegressor, BoostersClassifier

# Works like any sklearn estimator
model = BoostersRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    n_jobs=-1,
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
predictions = model.predict(X_test)

# Cross-validation works
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

# Pipeline integration
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', BoostersClassifier()),
])
```

### scikit-learn Estimator Classes

```python
class BoostersModel(BaseEstimator):
    """Base class for scikit-learn estimators."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        n_jobs: int = -1,
        random_state: int | None = None,
        **kwargs,
    ): ...
    
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        callbacks=None,
    ) -> 'BoostersModel': ...
    
    def predict(self, X) -> np.ndarray: ...
    
    @property
    def feature_importances_(self) -> np.ndarray: ...
    
    @property
    def n_features_in_(self) -> int: ...
    
    @property
    def feature_names_in_(self) -> np.ndarray: ...

class BoostersRegressor(BoostersModel, RegressorMixin):
    """Boosters regressor."""

class BoostersClassifier(BoostersModel, ClassifierMixin):
    """Boosters classifier."""
    
    def predict_proba(self, X) -> np.ndarray: ...
    
    @property
    def classes_(self) -> np.ndarray: ...

class BoostersRanker(BoostersModel):
    """Boosters ranker for learning-to-rank tasks."""
    
    def fit(self, X, y, group=None, ...): ...
```

---

## Data Format Support

### Input Formats

| Format | Support | Notes |
| ------ | ------- | ----- |
| NumPy ndarray | ✓ | Accepts both C and F contiguous |
| pandas DataFrame | ✓ | Auto-detect categorical columns |
| PyArrow Table | ✓ | Zero-copy via Arrow C Data Interface |
| scipy.sparse CSR | ✓ | Sparse matrix support |
| scipy.sparse CSC | ✓ | Converted to CSR internally |
| Python lists | ✓ | Converted to NumPy |

### Memory Layout Strategy

**Problem**: Python uses row-major `(n_samples, n_features)`, Rust core uses
feature-major `[n_features, n_samples]`.

**Solution**: Accept both layouts, detect at runtime:

```python
# Detection in conversion layer
if array.flags['F_CONTIGUOUS']:
    # F-contiguous (column-major) - can be viewed as feature-major
    layout = 'feature_major'
    data = array.ravel(order='F')
elif array.flags['C_CONTIGUOUS']:
    # C-contiguous (row-major) - needs transpose or row-major path
    layout = 'sample_major'
    data = array.ravel(order='C')
else:
    # Non-contiguous - copy to C order
    array = np.ascontiguousarray(array)
    layout = 'sample_major'
    data = array.ravel(order='C')
```

For training, we transpose C-order data to F-order (one-time cost amortized over
many iterations). For prediction, we support both layouts via the inference layer.

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

### Booster Architecture

```rust
use pyo3::prelude::*;
use std::sync::Arc;
use boosters::{GBDTModel, GBLinearModel, ModelMeta};

/// Unified booster type supporting both GBDT and GBLinear.
#[pyclass]
pub struct Booster {
    inner: BoosterInner,
    // Python-side metadata
    feature_names: Option<Vec<String>>,
    pandas_categorical: Option<Vec<Vec<PyObject>>>,
    best_iteration: Option<usize>,
}

enum BoosterInner {
    GBDT(Arc<GBDTModel>),
    GBLinear(Arc<GBLinearModel>),
}

#[pymethods]
impl Booster {
    /// Predict on new data.
    fn predict(
        &self,
        py: Python<'_>,
        data: PyObject,
        num_iteration: Option<usize>,
        raw_score: bool,
        pred_leaf: bool,
        pred_contrib: bool,
        n_jobs: i32,
    ) -> PyResult<PyObject> {
        // Release GIL during prediction
        py.allow_threads(|| {
            match &self.inner {
                BoosterInner::GBDT(model) => {
                    // ... predict logic
                }
                BoosterInner::GBLinear(model) => {
                    // ... predict logic  
                }
            }
        })
    }
    
    /// Get feature importance.
    fn feature_importance(&self, importance_type: &str) -> PyResult<Vec<f64>> {
        match &self.inner {
            BoosterInner::GBDT(model) => {
                let imp_type = parse_importance_type(importance_type)?;
                Ok(model.feature_importance(imp_type).values().to_vec())
            }
            BoosterInner::GBLinear(model) => {
                // Weight magnitudes for linear model
                Ok(model.weight_importance())
            }
        }
    }
    
    #[getter]
    fn num_trees(&self) -> usize {
        match &self.inner {
            BoosterInner::GBDT(m) => m.forest().n_trees(),
            BoosterInner::GBLinear(m) => m.n_rounds(),
        }
    }
}
```

### Dataset Architecture

**Dataset Lifetime Strategy**:

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
impl PyBooster {
    fn predict(&self, py: Python<'_>, data: PyObject) -> PyResult<PyObject> {
        let features = convert_to_features(py, data)?;
        
        // Release GIL during prediction
        let result = py.allow_threads(|| {
            self.inner.predict(&features)
        });
        
        // Convert back to NumPy
        numpy_from_array2(py, result)
    }
}
```

### Error Handling

```rust
// Map Rust errors to Python exceptions
#[pyclass]
struct BoostersError;

impl From<boosters::DatasetError> for PyErr {
    fn from(err: DatasetError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
```

---

## Design Decisions

### DD-1: PyO3 vs cffi

**Decision**: Use PyO3 (Rust-native Python bindings).

**Rationale**:

- Single build step (Maturin)
- Better error handling integration
- No need for C API layer
- Active ecosystem and community

### DD-2: API Style

**Decision**: Mirror LightGBM's Python API where possible.

**Rationale**:

- Familiar to existing GBDT users
- Easy migration from LightGBM
- Proven API design
- dict-based params for flexibility

### DD-3: scikit-learn First-Class

**Decision**: Provide native sklearn estimators (not just wrappers).

**Rationale**:

- Full sklearn compatibility (pipelines, CV, etc.)
- Property-based API (n_estimators vs num_trees)
- Type hints and IDE support

### DD-4: Zero-Copy Where Possible

**Decision**: Use Arrow C Data Interface for zero-copy with PyArrow.

**Rationale**:

- Avoid unnecessary memory copies
- Critical for large datasets
- Arrow becoming standard for data interchange

### DD-5: Package Naming

**Decision**: Package name `boosters` (import as `import boosters`).

**Rationale**:

- Clean, memorable name
- Consistent with crate name
- Available on PyPI (need to verify)

### DD-6: Memory Layout Handling

**Decision**: Accept both C and F contiguous arrays, transpose C-order for training.

**Rationale**:

- Users expect `(n_samples, n_features)` convention
- One-time transpose cost acceptable for training
- Prediction can use row-major path for compatibility

### DD-7: Verbose/Logging

**Decision**: Match LightGBM's `verbose` parameter semantics.

```python
verbose = -1  # Silent
verbose = 0   # Warning only
verbose = 1   # Info (default)
verbose > 1   # Debug
```

### DD-8: Callbacks for Early Stopping

**Decision**: Provide `early_stopping()` callback (like LightGBM).

```python
from boosters.callback import early_stopping

model = train(
    params,
    train_set,
    valid_sets=[valid_set],
    callbacks=[early_stopping(stopping_rounds=50)],
)
```

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
shap_values = explainer.shap_values(X)  # Returns ShapValues object
base_value = explainer.expected_value
```

### Feature Importance

```python
# Multiple importance types
gain_importance = model.feature_importance(importance_type='gain')
split_importance = model.feature_importance(importance_type='split')
cover_importance = model.feature_importance(importance_type='cover')

# As DataFrame with feature names
import pandas as pd
df = pd.DataFrame({
    'feature': model.feature_name,
    'importance': model.feature_importance('gain'),
}).sort_values('importance', ascending=False)
```

---

## Multi-Output Support

### Classification

```python
# Binary classification
params = {'objective': 'binary', 'metric': 'auc'}

# Multiclass classification
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
}

# Prediction shapes
binary_pred = model.predict(X)      # (n_samples,) - probabilities
multi_pred = model.predict(X)       # (n_samples, n_classes) - probabilities
multi_raw = model.predict(X, raw_score=True)  # Raw logits
```

### Multi-Output Regression

```python
# Native multi-output (one tree per output per round)
params = {
    'objective': 'regression',
    'num_output': 2,  # New parameter
}

# Prediction: (n_samples, n_outputs)
```

---

## Model Inspection

### Feature Importance Methods

```python
# Multiple importance types
importance_split = model.feature_importance(importance_type='split')  # Number of splits
importance_gain = model.feature_importance(importance_type='gain')    # Total gain

# As DataFrame with feature names
import pandas as pd
importance_df = pd.DataFrame({
    'feature': model.feature_name_,
    'importance': model.feature_importance('gain'),
}).sort_values('importance', ascending=False)
```

### Tree Structure Export

```python
# Export trees to DataFrame (like LightGBM)
trees_df = model.trees_to_dataframe()
# Columns: tree_index, node_depth, node_index, left_child, right_child,
#          parent_index, split_feature, split_gain, threshold, 
#          decision_type, missing_direction, n_samples, value

# Filter to specific tree
tree_0 = trees_df[trees_df['tree_index'] == 0]
```

### Model Attributes

```python
# After fit():
model.n_features_in_       # int: Number of features seen during fit
model.feature_names_in_    # ndarray: Feature names if pandas input
model.feature_name_        # list: Feature names (same as feature_names_in_)
model.n_classes_           # int: Number of classes (classification only)
model.classes_             # ndarray: Class labels (classification only)
model.best_iteration_      # int: Best iteration if early stopping
model.best_score_          # float: Best eval score if early stopping
model.n_estimators_        # int: Actual number of trees trained
model.evals_result_        # dict: Training history if record_evaluation used
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
ds = Dataset(df, label=y)
# ds.categorical_feature_ == ['category']

# Or specify manually
ds = Dataset(X, label=y, categorical_feature=['col_0', 'col_5'])
ds = Dataset(X, label=y, categorical_feature=[0, 5])  # By index
```

### Category Value Preservation

```python
# Original categories are preserved for interpretability
model.fit(df[['category']], y)
print(model.pandas_categorical_)
# [['a', 'b', 'c']]  # Original category order

# When inspecting trees:
trees_df = model.trees_to_dataframe()
# threshold contains original category names for categorical splits
```

### Integer-Encoded Categoricals

```python
# If data is already integer-encoded, specify feature indices
ds = Dataset(X, label=y, categorical_feature=[2, 5])
# Values are treated as category indices (0, 1, 2, ...)
```

### Unseen Categories

```python
# Categories not seen during training are treated as missing
# This matches LightGBM behavior
train_df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'c'])})
test_df = pd.DataFrame({'cat': pd.Categorical(['a', 'd'])})  # 'd' is unseen

model.fit(train_df, y)
pred = model.predict(test_df)  # 'd' uses missing-value path
```

---

## Parameter Validation and Edge Cases

### Parameter Validation

```python
# Unknown parameters: warn and ignore (like LightGBM)
params = {'max_depth': 6, 'typo_param': 5}  
# UserWarning: Unknown parameter 'typo_param' will be ignored

# Invalid parameter values: raise immediately
params = {'max_depth': -1}
# ValueError: max_depth must be positive, got -1

# Type errors: raise immediately
params = {'learning_rate': 'fast'}
# TypeError: learning_rate must be float, got str
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

---

## Serialization

### File-Based (v1.0)

```python
# Save/load model
model.save_model('model.bin')  # Binary format (fast, compact)
model.save_model('model.json')  # JSON format (human-readable)

loaded = Booster(model_file='model.bin')

# Or using class method
loaded = Booster.load('model.bin')
```

### Pickle Support (v1.1+)

```python
import pickle

# Future: full pickle support
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Implementation: serialize Rust model to bytes, store in Python object
def __reduce__(self):
    return (Booster._from_bytes, (self._to_bytes(),))
```

**Note:** v1.0 will support file-based serialization. Pickle support is planned for v1.1.

---

## Callbacks

```python
from boosters.callback import early_stopping, log_evaluation, record_evaluation

# Early stopping
model = train(
    params,
    train_set,
    valid_sets=[valid_set],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=10),  # Print every 10 iterations
    ],
)

# Record training history
evals_result = {}
model = train(
    params,
    train_set,
    valid_sets=[train_set, valid_set],
    valid_names=['train', 'valid'],
    callbacks=[record_evaluation(evals_result)],
)
# evals_result = {'train': {'rmse': [...]}, 'valid': {'rmse': [...]}}
```

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

```rust
// Python n_jobs convention:
// -1 = auto (all cores)
//  0 = auto (same as -1 in Python convention)
//  1 = sequential
//  n > 1 = exactly n threads

fn convert_n_jobs(n_jobs: i32) -> usize {
    match n_jobs {
        -1 | 0 => 0,  // 0 means "auto" in run_with_threads
        1 => 1,       // Sequential
        n if n > 1 => n as usize,
        _ => 0,       // Negative (other than -1) → auto
    }
}

// In PyO3, release GIL during Rust computation
impl Booster {
    fn predict(&self, py: Python<'_>, ...) -> PyResult<...> {
        let n_threads = convert_n_jobs(n_jobs);
        
        py.allow_threads(|| {
            boosters::run_with_threads(n_threads, |_| {
                self.inner.predict(...)
            })
        })
    }
}
```

**Key principles:**

1. Always release GIL during Rust computation
2. Use `run_with_threads` to manage Rayon thread pool
3. Avoid nested parallelism conflicts with sklearn's `n_jobs`

### Data Conversion Performance

| Input Type | Conversion Cost | Notes |
| ---------- | --------------- | ----- |
| F-contiguous f32 array | Zero copy | Optimal path |
| C-contiguous f32 array | Transpose O(n×m) | Training only, cache-unfriendly |
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
| 3.9 | Linux | Full |
| 3.10 | Linux | Full |
| 3.11 | Linux, macOS, Windows | Full |
| 3.12 | Linux | Full |

### sklearn Compliance

```python
from sklearn.utils.estimator_checks import check_estimator

def test_sklearn_compliance():
    # Run standard sklearn estimator checks
    for estimator in [BoostersRegressor(), BoostersClassifier()]:
        for check in check_estimator(estimator, generate_only=True):
            check(estimator)
```

---

## Changelog

- 2025-12-25: Initial draft
- 2025-12-25: Round 1 - Added memory layout strategy, objective/metric mapping
- 2025-12-25: Round 2 - Added Dataset lifetime strategy, PyArrow integration details
- 2025-12-25: Round 3 - Added testing strategy, PyO3 Booster architecture
- 2025-12-25: Round 4 - Added threading model, error handling, edge cases
- 2025-12-25: Round 5 - Added model inspection, categorical handling, tree export
- 2025-12-25: Round 6 - Added Python version policy, wheel distribution, unseen categories
