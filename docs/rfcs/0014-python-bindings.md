# RFC-0014: Python Bindings

- **Status**: Implemented
- **Created**: 2025-12-25
- **Updated**: 2025-12-27
- **Depends on**: RFC-0001 through RFC-0013
- **Scope**: Python package for boosters library

## Summary

Python bindings for the boosters library, exposing all major functionality to Python users.
Built with PyO3/Maturin, providing native performance with a Pythonic API. Supports common
data formats (NumPy, pandas, Polars) and integrates with scikit-learn.

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
│   ├── data.rs             # Dataset/EvalSet Rust bindings
│   ├── config.rs           # GBDTConfig/GBLinearConfig bindings
│   ├── objectives.rs       # Objective enum bindings
│   ├── metrics.rs          # Metric enum bindings
│   ├── types.rs            # GrowthStrategy, ImportanceType, Verbosity
│   ├── validation.rs       # Input validation utilities
│   ├── error.rs            # Error handling
│   └── model/
│       ├── mod.rs          # Model module
│       ├── gbdt.rs         # GBDTModel bindings
│       └── gblinear.rs     # GBLinearModel bindings
└── python/
    └── boosters/
        ├── __init__.py     # Public API re-exports
        ├── _boosters_rs.pyi # Type stubs for Rust extension (generated)
        ├── config.py       # Re-exports GBDTConfig, GBLinearConfig
        ├── data.py         # Dataset (extends Rust), EvalSet
        ├── objectives.py   # Re-exports Objective enum
        ├── metrics.py      # Re-exports Metric enum
        ├── model.py        # Re-exports GBDTModel, GBLinearModel
        ├── types.py        # Re-exports GrowthStrategy
        └── sklearn/
            ├── __init__.py # sklearn estimator exports
            ├── gbdt.py     # GBDTRegressor, GBDTClassifier
            └── gblinear.py # GBLinearRegressor, GBLinearClassifier
```

### Module Exports

```python
# boosters/__init__.py
from boosters._boosters_rs import (
    ImportanceType,
    Verbosity,
    __version__,
)

# Config types (Rust-owned)
from boosters.config import GBDTConfig, GBLinearConfig

# Data types
from boosters.data import Dataset, EvalSet

# Metric enum
from boosters.metrics import Metric

# Model types
from boosters.model import GBDTModel, GBLinearModel

# Objective enum  
from boosters.objectives import Objective

# Type aliases
from boosters.types import GrowthStrategy

__all__ = [
    "Dataset",
    "EvalSet",
    "GBDTConfig",
    "GBDTModel",
    "GBLinearConfig",
    "GBLinearModel",
    "GrowthStrategy",
    "ImportanceType",
    "Metric",
    "Objective",
    "Verbosity",
    "__version__",
]
```

---

## Core API Design

### Design Principles

1. **Strict typing**: All public APIs use Python 3.12+ type hints
2. **Flat configs**: Configuration via flat Rust-owned classes, not nested Python dataclasses
3. **Pythonic naming**: `n_samples`, `n_features`, `labels` (not `num_data`, `get_label`)
4. **Model separation**: `GBDTModel` and `GBLinearModel` as distinct classes (like Rust)
5. **Factory-method enums**: `Objective.squared()`, `Metric.rmse()` with validation
6. **sklearn-like fit/predict**: Config in `__init__`, data in `fit`/`predict`

### Quick Start (Core API)

This is the **core API**, not the sklearn wrapper. The core API uses Rust-owned
configuration classes and `Dataset` objects for maximum type safety and control.

```python
import boosters as bst
import numpy as np

# Create dataset
X = np.random.rand(1000, 10).astype(np.float32)
y = np.random.rand(1000).astype(np.float32)
train = bst.Dataset(X[:800], y[:800])
valid = bst.Dataset(X[800:], y[800:])

# Configure with flat config (all parameters in one class)
config = bst.GBDTConfig(
    n_estimators=100,
    learning_rate=0.1,
    objective=bst.Objective.squared(),
    metric=bst.Metric.rmse(),
    max_depth=6,
    l2=1.0,
)

# Train
model = bst.GBDTModel(config)
model.fit(train, valid=[bst.EvalSet(valid, "valid")])

# Predict
predictions = model.predict(bst.Dataset(X[800:]))
```

**Multi-output quantile regression example:**

```python
# Multi-quantile prediction (3 output columns)
config = bst.GBDTConfig(
    objective=bst.Objective.pinball([0.1, 0.5, 0.9]),
)
model = bst.GBDTModel(config)
model.fit(train)
quantiles = model.predict(bst.Dataset(X_test))  # Shape: (n_samples, 3)
```

For sklearn-style usage, see the [scikit-learn Integration](#scikit-learn-integration) section.

### Package Version

```python
import boosters
print(boosters.__version__)  # "0.1.0"
```

### Dtype Recommendations

- **`float32`**: Recommended for performance (2× faster, half memory)
- **`float64`**: Use when precision matters (financial, scientific)
- Arrays must be C-contiguous; F-order arrays are copied internally

### Dataset

```python
class Dataset:
    """Training dataset with features, labels, and optional metadata.
    
    Dataset extends a Rust base class with Python-friendly constructors.
    Data is converted to C-contiguous float32 arrays for efficient processing.
    
    Supports multiple input types:
    - NumPy arrays (preferred, zero-copy when possible)
    - Pandas DataFrames (auto-detects feature names and categorical columns)
    - Polars DataFrames (auto-detects feature names and categorical columns)
    - Any array-like that can be converted to NumPy
    """
    
    def __new__(
        cls,
        features: NDArray | pd.DataFrame | pl.DataFrame,
        labels: NDArray | None = None,
        weights: NDArray | None = None,
        groups: NDArray | None = None,
        feature_names: list[str] | None = None,
        categorical_features: list[int] | None = None,
    ) -> Dataset:
        """Create a new Dataset instance."""
    
    @property
    def n_samples(self) -> int:
        """Number of samples."""
    
    @property
    def n_features(self) -> int:
        """Number of features."""
    
    @property
    def has_labels(self) -> bool:
        """Whether labels are present."""
    
    @property
    def has_weights(self) -> bool:
        """Whether weights are present."""
    
    @property
    def feature_names(self) -> list[str] | None:
        """Feature names if provided."""
    
    @property
    def categorical_features(self) -> list[int]:
        """Indices of categorical features."""
    
    @property
    def shape(self) -> tuple[int, int]:
        """Shape as (n_samples, n_features)."""
    
    def __len__(self) -> int:
        return self.n_samples
```

### EvalSet (Named Evaluation Dataset)

```python
class EvalSet:
    """Named evaluation set for validation during training.
    
    An EvalSet wraps a Dataset with a name, which is used to identify
    the evaluation set in training logs and `eval_results`.
    """
    
    def __new__(cls, dataset: Dataset, name: str) -> EvalSet:
        """Create a new named evaluation set.
        
        Args:
            dataset: Dataset containing features and labels.
            name: Name for this evaluation set (e.g., "validation", "test")
        """
    
    @property
    def name(self) -> str:
        """Name of this evaluation set."""
    
    @property
    def dataset(self) -> Dataset:
        """The underlying dataset."""
```

### Configuration Classes

Configuration is done via flat Rust-owned classes with factory constructors.
All parameters are specified in the main config class rather than nested sub-configs.

```python
class GBDTConfig:
    """Main configuration for GBDT model.
    
    All parameters are flat (no nested config objects) matching the core Rust API.
    """
    
    def __new__(
        cls,
        n_estimators: int = 100,
        learning_rate: float = 0.3,
        objective: Objective | None = None,  # Default: Objective.Squared
        metric: Metric | None = None,
        growth_strategy: GrowthStrategy = GrowthStrategy.Depthwise,
        max_depth: int = 6,
        n_leaves: int = 31,
        max_onehot_cats: int = 4,
        l1: float = 0.0,
        l2: float = 1.0,
        min_gain_to_split: float = 0.0,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        colsample_bylevel: float = 1.0,
        linear_leaves: bool = False,
        linear_l2: float = 0.01,
        linear_l1: float = 0.0,
        linear_max_iterations: int = 10,
        linear_tolerance: float = 1e-6,
        linear_min_samples: int = 50,
        linear_coefficient_threshold: float = 1e-6,
        linear_max_features: int = 10,
        max_bins: int = 256,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        verbosity: Verbosity = Verbosity.Silent,
    ) -> GBDTConfig: ...

class GBLinearConfig:
    """Main configuration for GBLinear model.
    
    GBLinear uses gradient boosting to train a linear model via coordinate descent.
    """
    
    def __new__(
        cls,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        objective: Objective | None = None,
        metric: Metric | None = None,
        l1: float = 0.0,
        l2: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        verbosity: Verbosity = Verbosity.Silent,
    ) -> GBLinearConfig: ...

class GrowthStrategy(enum.Enum):
    """Tree growth strategy for building decision trees."""
    Depthwise = ...  # Level-by-level (like XGBoost)
    Leafwise = ...   # Best-first split (like LightGBM)

class Verbosity(enum.Enum):
    """Verbosity level for training output."""
    Silent = ...   # No output
    Warning = ...  # Errors and warnings only
    Info = ...     # Progress and important information
    Debug = ...    # Detailed debugging information
```

### Objective Enum (Factory-Method Pattern)

Objectives use a Rust-backed enum with static factory methods for validation:

```python
class Objective(enum.Enum):
    """Objective (loss) functions for gradient boosting.
    
    Use static factory methods to create instances with validation.
    Supports Python pattern matching for type inspection.
    """
    
    # Enum variants
    Squared = ...    # Mean squared error (L2)
    Absolute = ...   # Mean absolute error (L1)
    Huber = ...      # Pseudo-Huber loss (delta parameter)
    Pinball = ...    # Quantile regression (alpha parameter)
    Poisson = ...    # Poisson deviance for count data
    Logistic = ...   # Binary cross-entropy
    Hinge = ...      # SVM-style hinge loss
    Softmax = ...    # Multiclass cross-entropy (n_classes parameter)
    LambdaRank = ... # LambdaMART for ranking (ndcg_at parameter)
    
    # Factory methods with validation
    @staticmethod
    def squared() -> Objective:
        """Create squared error loss (L2)."""
    
    @staticmethod
    def absolute() -> Objective:
        """Create absolute error loss (L1)."""
    
    @staticmethod
    def huber(delta: float = 1.0) -> Objective:
        """Create Huber loss with validation (delta > 0)."""
    
    @staticmethod
    def pinball(alpha: Sequence[float]) -> Objective:
        """Create pinball loss with validation (all alpha in (0, 1))."""
    
    @staticmethod
    def poisson() -> Objective:
        """Create Poisson loss."""
    
    @staticmethod
    def logistic() -> Objective:
        """Create logistic loss for binary classification."""
    
    @staticmethod
    def hinge() -> Objective:
        """Create hinge loss for binary classification."""
    
    @staticmethod
    def softmax(n_classes: int) -> Objective:
        """Create softmax loss with validation (n_classes >= 2)."""
    
    @staticmethod
    def lambdarank(ndcg_at: int = 10) -> Objective:
        """Create LambdaRank loss with validation (ndcg_at > 0)."""
```

**Pattern matching example:**

```python
match obj:
    case Objective.Squared():
        print("L2 loss")
    case Objective.Pinball(alpha=a):
        print(f"Quantile: {a}")
    case Objective.Softmax(n_classes=k):
        print(f"Multiclass with {k} classes")
```

### Metric Enum (Factory-Method Pattern)

```python
class Metric(enum.Enum):
    """Evaluation metrics for gradient boosting.
    
    Use static factory methods to create instances with validation.
    """
    
    # Enum variants
    Rmse = ...      # Root Mean Squared Error
    Mae = ...       # Mean Absolute Error
    Mape = ...      # Mean Absolute Percentage Error
    LogLoss = ...   # Binary cross-entropy
    Auc = ...       # Area Under ROC Curve
    Accuracy = ...  # Classification accuracy
    Ndcg = ...      # Normalized Discounted Cumulative Gain (at parameter)
    
    # Factory methods
    @staticmethod
    def rmse() -> Metric: ...
    
    @staticmethod
    def mae() -> Metric: ...
    
    @staticmethod
    def mape() -> Metric: ...
    
    @staticmethod
    def logloss() -> Metric: ...
    
    @staticmethod
    def auc() -> Metric: ...
    
    @staticmethod
    def accuracy() -> Metric: ...
    
    @staticmethod
    def ndcg(at: int = 10) -> Metric:
        """Create NDCG@k metric with validation (at > 0)."""
```

### Output Shape by Objective

| Objective | Output Shape | Notes |
| --------- | ------------ | ----- |
| `Objective.squared()` | `(n_samples, 1)` | Scalar regression |
| `Objective.pinball([0.5])` | `(n_samples, 1)` | Single quantile |
| `Objective.pinball([0.1, 0.5, 0.9])` | `(n_samples, 3)` | Columns in alpha order |
| `Objective.logistic()` | `(n_samples, 1)` | Probability [0, 1] |
| `Objective.softmax(k)` | `(n_samples, k)` | Probabilities per class |

**Note on multi-quantile**: Output columns are ordered by the `alpha` list. With `alpha=[0.1, 0.5, 0.9]`,
column 0 is the 10th percentile, column 1 is median, column 2 is 90th percentile.

### GBDT Model

```python
class GBDTModel:
    """Gradient Boosted Decision Trees model."""
    
    def __new__(cls, config: GBDTConfig | None = None) -> GBDTModel:
        """Create a new GBDT model.
        
        Args:
            config: Training configuration. Uses defaults if None.
        """
    
    def fit(
        self,
        train: Dataset,
        valid: Sequence[EvalSet] | None = None,
        n_threads: int = 0,
    ) -> GBDTModel:
        """Train the model.
        
        Args:
            train: Training dataset with features and labels.
            valid: Named validation set(s) for early stopping and metrics.
            n_threads: Number of threads (0 = auto).
        
        Returns:
            Self for method chaining.
        """
    
    def predict(
        self,
        data: Dataset,
        n_threads: int = 0,
    ) -> NDArray[np.float32]:
        """Make predictions.
        
        Returns transformed predictions (e.g., probabilities for classification).
        Output shape is (n_samples, n_outputs) - sklearn convention.
        
        Args:
            data: Dataset containing features for prediction.
            n_threads: Number of threads (0 = auto).
        
        Returns:
            Predictions array with shape (n_samples, n_outputs).
        """
    
    def predict_raw(
        self,
        data: Dataset,
        n_threads: int = 0,
    ) -> NDArray[np.float32]:
        """Make raw (untransformed) predictions.
        
        Returns raw margin scores without transformation.
        """
    
    # Model inspection
    @property
    def is_fitted(self) -> bool: ...
    
    @property
    def n_features(self) -> int: ...
    
    @property
    def n_trees(self) -> int: ...
    
    @property
    def best_iteration(self) -> int | None: ...
    
    @property
    def best_score(self) -> float | None: ...
    
    @property
    def config(self) -> GBDTConfig: ...
    
    @property
    def eval_results(self) -> dict[str, dict[str, list[float]]] | None:
        """Evaluation results from training.
        
        Returns:
            Nested dict: {dataset_name: {metric_name: [values_per_iteration]}}
            Example: {"valid": {"rmse": [0.5, 0.4, 0.35, ...]}}
        """
    
    def feature_importance(
        self,
        importance_type: ImportanceType = ImportanceType.Split,
    ) -> NDArray[np.float32]: ...
    
    def shap_values(self, data: Dataset) -> NDArray[np.float32]:
        """Compute SHAP values for feature contribution analysis.
        
        Returns:
            Array with shape (n_samples, n_features + 1, n_outputs).
        """

class GBLinearModel:
    """Gradient Boosted Linear model."""
    
    def __new__(cls, config: GBLinearConfig | None = None) -> GBLinearModel:
        """Create a new GBLinear model."""
    
    def fit(
        self,
        train: Dataset,
        eval_set: Sequence[EvalSet] | None = None,
        n_threads: int = 0,
    ) -> GBLinearModel:
        """Train the model."""
    
    def predict(
        self,
        data: Dataset,
        n_threads: int = 0,
    ) -> NDArray[np.float32]:
        """Make predictions."""
    
    def predict_raw(
        self,
        data: Dataset,
        n_threads: int = 0,
    ) -> NDArray[np.float32]:
        """Make raw (untransformed) predictions."""
    
    @property
    def coef_(self) -> NDArray[np.float32]:
        """Model coefficients (n_features,) or (n_features, n_outputs)."""
    
    @property
    def intercept_(self) -> NDArray[np.float32]:
        """Model intercept (bias) term."""
    
    @property
    def is_fitted(self) -> bool: ...
    
    @property
    def n_features_in_(self) -> int: ...
    
    @property
    def best_iteration(self) -> int | None: ...
    
    @property
    def best_score(self) -> float | None: ...
    
    @property
    def eval_results(self) -> dict[str, dict[str, list[float]]] | None: ...
    
    @property
    def config(self) -> GBLinearConfig: ...
```

---

## Objective and Metric Mapping

### Objective Factory Methods

| Python Factory | Parameters | Notes |
| -------------- | ---------- | ----- |
| `Objective.squared()` | — | Default for regression |
| `Objective.absolute()` | — | Robust to outliers (L1) |
| `Objective.huber(delta=1.0)` | `delta: float` | Smooth L1/L2 blend |
| `Objective.pinball([0.5])` | `alpha: Sequence[float]` | Multi-quantile supported |
| `Objective.poisson()` | — | For count data |
| `Objective.logistic()` | — | Binary classification |
| `Objective.hinge()` | — | SVM-style classification |
| `Objective.softmax(n_classes)` | `n_classes: int` | Multiclass (required) |
| `Objective.lambdarank(ndcg_at=10)` | `ndcg_at: int` | Learning to rank |

### Metric Factory Methods

| Python Factory | Parameters | Task |
| ------------- | ---------- | ---- |
| `Metric.rmse()` | — | Regression |
| `Metric.mae()` | — | Regression |
| `Metric.mape()` | — | Regression |
| `Metric.logloss()` | — | Classification |
| `Metric.auc()` | — | Binary classification |
| `Metric.accuracy()` | — | Classification |
| `Metric.ndcg(at=10)` | `at: int` | Ranking |

---

## scikit-learn Integration

```python
from boosters.sklearn import GBDTRegressor, GBDTClassifier
from boosters.sklearn import GBLinearRegressor, GBLinearClassifier

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
)

# Cross-validation works
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)

# Pipeline integration
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    
    All constructor parameters are flat (no nested config) for sklearn compatibility.
    """
    
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: float = 1.0,
        max_leaves: int = 31,
        grow_strategy: GrowthStrategy = GrowthStrategy.Depthwise,
        colsample_bytree: float = 1.0,
        subsample: float = 1.0,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        n_threads: int = 0,
        verbose: int = 1,
        objective: Objective | None = None,  # Default: Objective.squared()
        metric: Metric | None = None,  # Default: Metric.rmse()
    ) -> None: ...
    
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        eval_set: list[tuple[ArrayLike, ArrayLike]] | list[EvalSet] | None = None,
        sample_weight: ArrayLike | None = None,
    ) -> Self: ...
    
    def predict(self, X: ArrayLike) -> NDArray[np.float32]: ...
    
    # sklearn-standard attributes (set after fit)
    n_features_in_: int
    feature_importances_: NDArray[np.float32]
    model_: GBDTModel  # Underlying core model

class GBDTClassifier(BaseEstimator, ClassifierMixin):
    """GBDT classifier with sklearn interface.
    
    Auto-infers binary vs multiclass from labels:
    - 2 classes: Uses Objective.logistic()
    - >2 classes: Uses Objective.softmax(n_classes)
    """
    
    def __init__(self, **kwargs) -> None: ...  # Same params as GBDTRegressor
    
    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> Self: ...
    def predict(self, X: ArrayLike) -> NDArray: ...
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float32]: ...
    
    classes_: NDArray  # Set after fit
    n_classes_: int

class GBLinearRegressor(BaseEstimator, RegressorMixin):
    """GBLinear regressor with sklearn interface."""
    
    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        l1: float = 0.0,
        l2: float = 1.0,
        early_stopping_rounds: int | None = None,
        seed: int = 42,
        verbose: int = 1,
        objective: Objective | None = None,
        metric: Metric | None = None,
    ) -> None: ...
    
    coef_: NDArray[np.float32]
    intercept_: NDArray[np.float32]

class GBLinearClassifier(BaseEstimator, ClassifierMixin):
    """GBLinear classifier with sklearn interface."""
```

### sklearn Parameter Mapping

sklearn flat parameters map to GBDTConfig:

| sklearn param | GBDTConfig param |
| ------------- | ---------------- |
| `n_estimators` | `n_estimators` |
| `learning_rate` | `learning_rate` |
| `max_depth` | `max_depth` |
| `max_leaves` | `n_leaves` |
| `min_child_weight` | `min_child_weight` |
| `gamma` | `min_gain_to_split` |
| `reg_alpha` | `l1` |
| `reg_lambda` | `l2` |
| `subsample` | `subsample` |
| `colsample_bytree` | `colsample_bytree` |
| `grow_strategy` | `growth_strategy` |

### sklearn Conversions

```python
# eval_set conversion (sklearn → core API)
eval_set = [(X_val1, y_val1), (X_val2, y_val2)]
# Becomes:
valid = [
    EvalSet(Dataset(X_val1, y_val1), "valid_0"),
    EvalSet(Dataset(X_val2, y_val2), "valid_1"),
]

# Classifier objective inference
y = [0, 1, 0, 1]  # Binary
# Infers: Objective.logistic()

y = [0, 1, 2, 0]  # Multiclass
# Infers: Objective.softmax(n_classes=3)
```

---

## Data Format Support

### Recommended Input Format

**Best performance**: C-contiguous NumPy float32 array.

```python
import pandas as pd
import numpy as np
from boosters import Dataset

# Option 1: NumPy array (preferred)
X = np.random.rand(1000, 10).astype(np.float32)
y = np.random.rand(1000).astype(np.float32)
dataset = Dataset(X, y)

# Option 2: pandas DataFrame
# - Automatic categorical detection
# - Feature names preserved
df = pd.DataFrame(data, columns=feature_names)
dataset = Dataset(df, y)

# Option 3: Polars DataFrame
import polars as pl
df = pl.DataFrame(data)
dataset = Dataset(df, y)
```

### Input Formats

| Format | Support | Conversion Cost |
| ------ | ------- | --------------- |
| NumPy C-contiguous f32 | ✓ | Optimal |
| NumPy C-contiguous f64 | ✓ | Cast O(n×m) |
| NumPy F-contiguous | ✓ | Copy + transpose |
| pandas DataFrame | ✓ | Extract + convert |
| Polars DataFrame | ✓ | Extract + convert |
| scipy.sparse | ✗ | Not yet supported |

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

# Polars categorical support
import polars as pl
df = pl.DataFrame({
    'city': pl.Series(['NYC', 'LA', 'SF']).cast(pl.Categorical),
})
ds = Dataset(df, label=y)  # 'city' automatically categorical
```

---

## PyO3 Implementation

### Configuration Types: Rust-Owned with Generated Stubs

**Decision**: Configuration types are defined in Rust with `#[pyclass]`,
type stubs are auto-generated via `pyo3-stub-gen`.

| Type | Implementation | Rationale |
| ---- | -------------- | --------- |
| `GBDTConfig`, `GBLinearConfig` | Rust `#[pyclass]` | Single source of truth |
| `Objective`, `Metric` | Rust complex enums | Pattern matching support |
| `GrowthStrategy`, `Verbosity`, `ImportanceType` | Rust simple enums | StrEnum-like behavior |
| `Dataset` | Rust base + Python subclass | Data validation in Python |
| `EvalSet` | Rust `#[pyclass]` | Type-safe validation set wrapper |
| `GBDTModel`, `GBLinearModel` | Rust `#[pyclass]` | Training/prediction |

### Objective Implementation (Complex Enum)

Objectives are implemented as a Rust enum with data, exposed to Python with
factory methods for validation:

```rust
use pyo3::prelude::*;

/// Objective function enum exposed to Python.
#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum Objective {
    #[pyo3(constructor = ())]
    Squared,
    #[pyo3(constructor = ())]
    Absolute,
    #[pyo3(constructor = ())]
    Poisson,
    #[pyo3(constructor = ())]
    Logistic,
    #[pyo3(constructor = ())]
    Hinge,
    #[pyo3(constructor = (delta))]
    Huber { delta: f64 },
    #[pyo3(constructor = (alpha))]
    Pinball { alpha: Vec<f64> },
    #[pyo3(constructor = (n_classes))]
    Softmax { n_classes: u32 },
    #[pyo3(constructor = (ndcg_at))]
    LambdaRank { ndcg_at: u32 },
}

#[pymethods]
impl Objective {
    /// Create squared error loss (L2).
    #[staticmethod]
    fn squared() -> Self {
        Self::Squared
    }
    
    /// Create pinball loss with validation.
    #[staticmethod]
    fn pinball(alpha: Vec<f64>) -> PyResult<Self> {
        if alpha.is_empty() {
            return Err(PyValueError::new_err("alpha must be non-empty"));
        }
        for &a in &alpha {
            if a <= 0.0 || a >= 1.0 {
                return Err(PyValueError::new_err(
                    format!("alpha values must be in (0, 1), got {}", a)
                ));
            }
        }
        Ok(Self::Pinball { alpha })
    }
    
    /// Create softmax loss with validation.
    #[staticmethod]
    fn softmax(n_classes: u32) -> PyResult<Self> {
        if n_classes < 2 {
            return Err(PyValueError::new_err(
                format!("n_classes must be >= 2, got {}", n_classes)
            ));
        }
        Ok(Self::Softmax { n_classes })
    }
}
```

### Type Stub Generation

**Tool**: `pyo3-stub-gen` crate for automatic stub generation.

```toml
# Cargo.toml
[dependencies]
pyo3-stub-gen = "0.17"
```

**Generated stub example** (`_boosters_rs.pyi`):

```python
class Objective(enum.Enum):
    Squared = ...
    Absolute = ...
    Huber = ...  # Has delta parameter
    Pinball = ...  # Has alpha parameter
    Softmax = ...  # Has n_classes parameter
    
    @staticmethod
    def squared() -> Objective: ...
    
    @staticmethod
    def pinball(alpha: Sequence[float]) -> Objective: ...
    
    @staticmethod
    def softmax(n_classes: int) -> Objective: ...
```

### Dataset Architecture

```rust
#[pyclass(subclass)]
pub struct Dataset {
    features: Arc<Array2<f32>>,
    labels: Option<Arc<Array2<f32>>>,
    weights: Option<Arc<Array1<f32>>>,
    feature_names: Option<Vec<String>>,
    categorical_features: Vec<usize>,
}

#[pymethods]
impl Dataset {
    #[new]
    #[pyo3(signature = (features, labels=None, weights=None, feature_names=None, categorical_features=None))]
    fn new(
        features: PyReadonlyArray2<f32>,
        labels: Option<PyReadonlyArray2<f32>>,
        weights: Option<PyReadonlyArray1<f32>>,
        feature_names: Option<Vec<String>>,
        categorical_features: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        // Validation and construction
        Ok(Self { ... })
    }
    
    #[getter]
    fn n_samples(&self) -> usize {
        self.features.nrows()
    }
    
    #[getter]
    fn n_features(&self) -> usize {
        self.features.ncols()
    }
}
```

The Python `Dataset` class extends this base with user-friendly constructors:

```python
# python/boosters/data.py
class Dataset(_RustDataset):
    """Extended Dataset with DataFrame support."""
    
    def __new__(cls, features, labels=None, ...):
        # Handle pandas/polars DataFrames
        # Convert to numpy arrays
        # Detect categorical columns
        # Call Rust base constructor
        return _RustDataset.__new__(cls, features_arr, ...)
```

### GIL Management

```rust
impl GBDTModel {
    fn fit(&mut self, py: Python<'_>, ...) -> PyResult<Self> {
        let n_threads = self.config.n_threads();
        
        // Release GIL during training
        py.allow_threads(|| {
            boosters::run_with_threads(n_threads, |_| {
                // Training happens here with GIL released
                self.train_internal(...)
            })
        })
    }
}
```

### Error Handling

```rust
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError};

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

### DD-3: Rust-Owned Config with Flat Structure

**Decision**: Configuration types (`GBDTConfig`, `GBLinearConfig`) are defined in Rust
with flat parameter structure (no nested sub-configs like `TreeConfig`).

**Rationale**:

- **Simpler API**: Users don't need to construct nested objects
- **sklearn compatibility**: Flat parameters map directly to sklearn estimator kwargs
- **Single source of truth**: Rust struct IS the definition, no sync risk
- **IDE support via stubs**: Generated `.pyi` files provide autocomplete

### DD-4: Factory-Method Enums for Objectives/Metrics

**Decision**: Objectives and Metrics use Rust complex enums with static factory methods
(`Objective.squared()`, `Metric.rmse()`) rather than separate classes.

**Rationale**:

- **Validation in factory**: Factory methods validate parameters before construction
- **Pattern matching**: Python can pattern match on enum variants
- **Exhaustive handling**: Rust compiler ensures all variants are handled
- **Single type**: `Objective` is one type, not a union of many classes

### DD-5: Pythonic Naming

**Decision**: Use Python conventions (`n_samples`, `n_features`, `labels`).

**Rationale**:

- Consistent with sklearn and numpy
- `n_` prefix standard for counts
- Plural for collections (`labels`, `weights`, `feature_names`)

### DD-6: Dataset with Python Subclass

**Decision**: `Dataset` has a Rust base class extended by Python for DataFrame support.

**Rationale**:

- **Rust base**: Efficient storage and access
- **Python extension**: User-friendly constructors with pandas/polars support
- **Validation in Python**: Better error messages for data conversion issues

### DD-7: Threading at Bindings Level

**Decision**: Python bindings manage thread pool, Rust core is thread-agnostic.

**Rationale**:

- GIL must be released in Python layer anyway
- Avoids nested thread pool issues
- Rust library users can manage threads their way
- Clear separation of concerns

### DD-8: sklearn Estimators Use Flat Config

**Decision**: sklearn estimators take flat kwargs that map to `GBDTConfig` parameters.

**Rationale**:

- sklearn's `get_params()`/`set_params()` expect flat attributes
- GridSearchCV and similar tools require flat parameter space
- Direct mapping to core API config

### DD-9: EvalSet Wraps Dataset with Name

**Decision**: Validation sets use `EvalSet(dataset, name)` wrapper.

**Rationale**:

- **Named results**: `eval_results["valid"]["rmse"]` vs `eval_results[0]["rmse"]`
- **Rust parity**: Matches Rust training API's named eval sets
- **Multiple datasets**: Train metrics, validation metrics clearly distinguished

### DD-10: Generated Type Stubs

**Decision**: Use `pyo3-stub-gen` to generate `.pyi` stub files automatically.

**Rationale**:

- **IDE support**: Full autocomplete and type checking
- **CI verification**: Stubs regenerated and verified in CI
- **Single source**: Stubs derived from Rust, not manually maintained

---

## Explainability API

### SHAP Values

SHAP (SHapley Additive exPlanations) values decompose predictions into per-feature
contributions. The model provides a direct `shap_values()` method:

```python
import boosters as bst

# Train model
model = bst.GBDTModel(bst.GBDTConfig())
model.fit(train)

# Compute SHAP values
test_data = bst.Dataset(X_test)
shap_values = model.shap_values(test_data)
# Shape: (n_samples, n_features + 1, n_outputs)
# Last feature column is the base value (expected value)

# For single-output models, squeeze the last dimension
if shap_values.shape[2] == 1:
    shap_values = shap_values[:, :, 0]  # (n_samples, n_features + 1)
```

---

## Multi-Output Support

### Classification

```python
import boosters as bst

# Binary classification
config = bst.GBDTConfig(objective=bst.Objective.logistic())
model = bst.GBDTModel(config)
model.fit(train)
probs = model.predict(test_data)  # (n_samples, 1) - probabilities
logits = model.predict_raw(test_data)  # Raw logits

# Multiclass classification
config = bst.GBDTConfig(objective=bst.Objective.softmax(n_classes=3))
model = bst.GBDTModel(config)
model.fit(train)
probs = model.predict(test_data)         # (n_samples, n_classes)
logits = model.predict_raw(test_data)    # Raw logits
```

### Multi-Output Regression

```python
import boosters as bst

# Quantile regression with multiple quantiles
config = bst.GBDTConfig(
    objective=bst.Objective.pinball([0.1, 0.5, 0.9]),  # 10th, 50th, 90th percentiles
)
model = bst.GBDTModel(config)
model.fit(train)
quantiles = model.predict(test_data)  # (n_samples, 3) - one column per quantile
```

---

## Model Inspection

### Feature Importance

```python
from boosters import ImportanceType

# Multiple importance types using enum
importance_split = model.feature_importance(ImportanceType.Split)
importance_gain = model.feature_importance(ImportanceType.Gain)

# As DataFrame with feature names (if set)
import pandas as pd
importance_df = pd.DataFrame({
    'feature': range(model.n_features),  # or use feature_names if available
    'importance': model.feature_importance(ImportanceType.Gain),
}).sort_values('importance', ascending=False)
```

### Tree Structure Export (Future)

Tree structure export is planned for a future release:

```python
# PLANNED: Export trees to DataFrame
# trees_df = model.trees_to_dataframe()
```

### Model Attributes

```python
# Core model API attributes (after fit):
model.n_features       # int: Number of features
model.n_trees          # int: Number of trees trained
model.feature_names    # list[str] | None: Feature names
model.best_iteration   # int | None: Best iteration (early stopping)
model.best_score       # float | None: Best validation score
model.eval_results     # dict[str, dict[str, list[float]]] | None
                       # Nested dict: {dataset_name: {metric_name: [values]}}

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
# Categorical features auto-detected from pandas Categorical dtype
```

### Integer-Encoded Categoricals

```python
# If data is already integer-encoded, specify feature indices
dataset = Dataset(features=X, labels=y, categorical_features=[2, 5])
# Values are treated as category indices (0, 1, 2, ...)
```

### Unseen Categories

```python
import boosters as bst

# Categories not seen during training are treated as missing
train_df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'c'])})
test_df = pd.DataFrame({'cat': pd.Categorical(['a', 'd'])})  # 'd' is unseen

train_ds = bst.Dataset(features=train_df, labels=y)
model = bst.GBDTModel(bst.GBDTConfig())
model.fit(train_ds)

test_ds = bst.Dataset(features=test_df)
pred = model.predict(test_ds)  # 'd' uses missing-value path
```

---

## Parameter Validation and Edge Cases

### Validation Timing

With Rust-owned config types, validation is split across two stages:

| Validation Stage | What's Checked | Examples |
| ---------------- | -------------- | -------- |
| **Constructor** | Type correctness, range validity | `alpha ∈ (0,1)`, `n_classes ≥ 2`, `n_estimators > 0` |
| **Fit** | Cross-field consistency, data compatibility | `max_depth` vs `n_leaves` feasibility, feature count matches |

**Constructor-time validation** (Rust `#[new]` methods):

```python
import boosters as bst

# Type errors caught immediately
config = bst.GBDTConfig(learning_rate='fast')
# TypeError: 'str' cannot be converted to 'float'

# Value validation in Rust constructor
config = bst.GBDTConfig(n_estimators=0)
# ValueError: n_estimators must be positive, got 0

# Parameterized objective validation
obj = bst.Objective.pinball([1.5])  # alpha must be in (0, 1)
# ValueError: alpha must be in (0, 1), got 1.5
```

**Fit-time validation** (cross-field and data checks):

```python
import boosters as bst

# Data compatibility
model.fit(train)  # train has 100 features
model.predict(test_data)  # test_data has 50 features
# ValueError: expected 100 features, got 50
```

### Flat Configuration

All configuration uses flat parameters in `GBDTConfig` and `GBLinearConfig`:

| Config | Key Fields | Purpose |
| ------ | ---------- | ------- |
| `GBDTConfig` | `n_estimators`, `learning_rate`, `max_depth`, `n_leaves`, `l1`, `l2`, `subsample`, `colsample` | All GBDT hyperparameters |
| `GBLinearConfig` | `n_estimators`, `learning_rate`, `l1`, `l2` | Linear booster hyperparameters |

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
| SHAP values | ✓ | ✓ | Via `model.shap_values(data)` |
| Feature importance | ✓ | ✓ | Parity |
| Early stopping | ✓ | ✓ | Via `early_stopping_rounds` |
| Custom objectives | Future | ✓ | Planned |
| Dask integration | Future | ✓ | Planned |

---

## Performance Considerations

### Thread Management

Threading is managed at the Python bindings level. The `n_threads`
parameter uses sklearn-compatible semantics:

```python
n_threads = 0   # Auto (all available cores)
n_threads = -1  # Same as 0 (sklearn convention)
n_threads = 1   # Sequential (no parallelism)
n_threads = 4   # Exactly 4 threads
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
- [ ] Tweedie/Gamma deviance objectives
- [ ] Learning rate schedules
- [ ] Custom objectives in Python

---

## Known Limitations (v1)

These features are explicitly out of scope for the initial release:

| Feature | Status | Notes |
| ------- | ------ | ----- |
| Custom objectives (Python-defined) | Not supported | Requires Python→Rust callbacks; use built-in objectives |
| Custom metrics (Python-defined) | Not supported | Use built-in metrics; add new ones via GitHub issue |
| Distributed training | Not supported | Single-machine only |
| GPU training | Not supported | CPU only |
| Model warm-start (continue training) | Not supported | Train from scratch |
| Learning rate schedules | Not supported | Use constant learning rate |

**Workarounds:**

- **Custom objective needed?** Open a GitHub issue describing your use case. Common objectives
  will be added to the built-in set.
- **Distributed training?** Pre-aggregate data; use Dask/Spark for data prep only.
- **GPU acceleration?** This library focuses on CPU; consider CatBoost/XGBoost for GPU.

---

## Migration Guide

### From XGBoost

| XGBoost Parameter | boosters Equivalent |
| ----------------- | ------------------- |
| `n_estimators` | `n_estimators` |
| `max_depth` | `max_depth` |
| `learning_rate` / `eta` | `learning_rate` |
| `reg_lambda` | `l2` |
| `reg_alpha` | `l1` |
| `subsample` | `subsample` |
| `colsample_bytree` | `colsample` |
| `min_child_weight` | `min_samples_leaf` (approximate) |
| `gamma` | `min_gain_to_split` |
| `objective="reg:squarederror"` | `objective=Objective.squared()` |
| `objective="binary:logistic"` | `objective=Objective.logistic()` |
| `objective="multi:softmax"` | `objective=Objective.softmax(n_classes=k)` |

### From LightGBM

| LightGBM Parameter | boosters Equivalent |
| ------------------ | ------------------- |
| `n_estimators` / `num_iterations` | `n_estimators` |
| `num_leaves` | `n_leaves` |
| `max_depth` | `max_depth` |
| `learning_rate` | `learning_rate` |
| `lambda_l1` | `l1` |
| `lambda_l2` | `l2` |
| `bagging_fraction` | `subsample` |
| `feature_fraction` | `colsample` |
| `min_data_in_leaf` | `min_samples_leaf` |
| `min_gain_to_split` | `min_gain_to_split` |
| `objective="regression"` | `objective=Objective.squared()` |
| `objective="binary"` | `objective=Objective.logistic()` |
| `objective="multiclass"` | `objective=Objective.softmax(n_classes=k)` |

### sklearn Wrapper vs Core API

| sklearn Wrapper | Core API Equivalent |
| --------------- | ------------------- |
| `GBDTRegressor(max_depth=5)` | `GBDTModel(GBDTConfig(max_depth=5))` |
| `clf.fit(X, y, eval_set=[(X_val, y_val)])` | `model.fit(train, valid=[EvalSet(valid, "valid")])` |
| `clf.fit(..., early_stopping_rounds=10)` | `GBDTConfig(early_stopping_rounds=10)` |

---

## Testing Strategy

### Test Categories

1. **Unit Tests** (pytest, fast):
   - Data conversion: NumPy, pandas, PyArrow, scipy.sparse
   - Parameter parsing and validation
   - Config construction and validation
   - Basic train/predict flow

2. **Integration Tests** (pytest, medium):
   - sklearn estimator compliance (`check_estimator()`)
   - Cross-validation compatibility
   - Pipeline integration
   - Early stopping
   - EvalSet handling

3. **Numerical Parity Tests** (pytest, slow):
   - Python predictions match Rust predictions exactly
   - Compare with LightGBM on standard datasets
   - SHAP value consistency

4. **Round-Trip Tests** (future):
   - Train in Python → Save → Load in Rust → Predict
   - Train in Rust → Save → Load in Python → Predict
   - pandas categorical preservation

5. **Memory/Performance Tests**:
   - Memory leak detection (memray for Python)
   - GIL release verification
   - Large dataset handling
   - Zero-copy verification

6. **Thread Safety Tests**:
   - Concurrent `predict()` calls from multiple Python threads
   - Concurrent training (should block or error, not corrupt)
   - Dataset shared across threads (read-only safe)

### Objective/Metric Test Matrix

For each objective type, verify:

| Objective | Output Range | Shape | Loss Decreases | Reference Comparison |
| --------- | ------------ | ----- | -------------- | -------------------- |
| `Objective.squared()` | (-∞, ∞) | (n, 1) | ✓ RMSE | XGBoost correlation > 0.99 |
| `Objective.logistic()` | [0, 1] | (n, 1) | ✓ LogLoss | XGBoost AUC within 1% |
| `Objective.softmax(k)` | [0, 1], sum=1 | (n, k) | ✓ LogLoss | XGBoost accuracy within 2% |
| `Objective.pinball([α])` | (-∞, ∞) | (n, len(α)) | ✓ Pinball | Calibration check |

### Edge Case Tests

```python
import boosters as bst

# Zero samples
def test_predict_empty_input():
    empty_data = bst.Dataset(np.zeros((0, 10), dtype=np.float32))
    preds = model.predict(empty_data)
    assert preds.shape == (0, 1)

# All same labels
def test_all_same_labels():
    y = np.ones(100, dtype=np.float32)
    model.fit(bst.Dataset(X, y))
    preds = model.predict(bst.Dataset(X))
    assert np.allclose(preds, 1.0, atol=0.1)

# Duplicate EvalSet names
def test_duplicate_evalset_names():
    with pytest.raises(ValueError, match="Duplicate EvalSet name"):
        model.fit(train, valid=[
            bst.EvalSet(ds1, "valid"),
            bst.EvalSet(ds2, "valid"),  # Duplicate!
        ])
```

### CI Matrix

| Python | OS | Test Scope |
| ------ | -- | ---------- |
| 3.12 | Linux, macOS, Windows | Full |
| 3.13 | Linux | Full |

### Doctest

All code examples in docstrings must be runnable:

```bash
# Run doctest as part of CI
pytest --doctest-modules boosters/
```

### sklearn Compliance

```python
from sklearn.utils.estimator_checks import check_estimator
from boosters.sklearn import GBDTRegressor, GBDTClassifier

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
- 2025-12-25: Rounds 7-10 - Final design review:
  - Parameterized objectives (DD-9) with validation
  - Named evaluation sets EvalSet (DD-10)
  - Rust-Python API sync strategy (DD-11)
  - Default metric inference (DD-12)
  - Enhanced prediction shape documentation
  - Comprehensive testing strategy
  - Pythonic naming (n_samples, labels, feature_names)
  - pandas DataFrame recommended as optimal input
  - Threading managed at bindings level (Rust core thread-agnostic)
  - Strict typing with Google-style docstrings
  - Serialization deferred until Rust storage format RFC
- 2025-12-25: Rounds 11-14 - Implementation details review:
  - Renamed QuantileLoss → PinballLoss (consistency with ArctanLoss)
  - Added ArctanLoss for robust quantile regression
  - Clarified Quick Start as core API (not sklearn)
  - Reorganized package structure with logical submodules
  - Added `__all__` exports documentation
  - sklearn: Added `early_stopping_rounds` shortcut
  - sklearn: Added parameter mapping table
  - sklearn: Documented objective inference for classifiers
  - sklearn: Added eval_set conversion documentation
- 2025-12-25: Rounds 15-18 - Config ownership and final review:
  - DD-3 rewritten: Rust-owned configs with generated stubs (was: hybrid Python dataclass)
  - All config types (`GBDTConfig`, `TreeConfig`, `PinballLoss`, etc.) now `#[pyclass]`
  - Added pyo3-stub-gen workflow and CI verification
  - Documented Rust constructor validation patterns
  - Updated PyO3 implementation section with full examples
  - DD-13: Nested configs (core API) vs flat kwargs (sklearn)
  - DD-14: Objective union type safety with `PyObjective` enum
  - Validation timing: Constructor (type/range) vs Fit (cross-field)
  - Complete sub-config reference table
  - Quick Start updated with full Rust-owned type examples
  - Migration guide: XGBoost and LightGBM parameter mapping
  - Known Limitations section for v1 scope
- 2025-12-27: **Post-implementation revision** - Updated RFC to match actual implementation:
  - Status changed from Accepted to Implemented
  - Flat config pattern: `GBDTConfig` has all parameters directly (no nested TreeConfig, RegularizationConfig)
  - Factory-method enums: `Objective.squared()`, `Metric.rmse()` instead of separate classes
  - EvalSet signature: `EvalSet(dataset, name)` order
  - SHAP API: `model.shap_values(data)` instead of `pred_contrib` parameter
  - Removed `TreeExplainer` class (not implemented)
  - Removed `trees_to_dataframe()` (future work)
  - Updated all code examples to use correct import patterns
  - Simplified design decisions to 10 key decisions matching implementation

