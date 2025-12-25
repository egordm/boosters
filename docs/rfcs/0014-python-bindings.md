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
│   ├── dataset.rs          # Dataset/features Rust conversion
│   ├── model.rs            # Model training/prediction core
│   ├── convert.rs          # Type conversion utilities
│   └── error.rs            # Error handling
└── python/
    └── boosters/
        ├── __init__.py     # Public API re-exports
        ├── _boosters_rs.pyi # Type stubs for Rust extension
        ├── dataset.py      # Dataset, EvalSet
        ├── config.py       # GBDTConfig, TreeConfig, etc.
        ├── objectives.py   # Loss classes (SquaredLoss, PinballLoss, ...)
        ├── metrics.py      # Metric classes (Rmse, Mae, ...)
        ├── model.py        # GBDTModel, GBLinearModel
        ├── callbacks.py    # Callback, EarlyStopping, LogEvaluation
        ├── sklearn.py      # scikit-learn estimators
        ├── compat.py       # Optional dependency handling
        └── plotting.py     # Visualization (optional)
```

### Module Exports

```python
# boosters/__init__.py
from boosters._version import __version__

# Data
from boosters.dataset import Dataset, EvalSet

# Configuration  
from boosters.config import (
    GBDTConfig, GBLinearConfig, TreeConfig,
    RegularizationConfig, SamplingConfig,
)

# Objectives
from boosters.objectives import (
    SquaredLoss, AbsoluteLoss, HuberLoss, PinballLoss, ArctanLoss,
    PoissonLoss, LogisticLoss, HingeLoss, SoftmaxLoss, LambdaRankLoss,
    Objective,  # Union type
)

# Metrics
from boosters.metrics import (
    Rmse, Mae, Mape, LogLoss, Auc, Accuracy, Ndcg,
    Metric,  # Union type
)

# Models
from boosters.model import GBDTModel, GBLinearModel

# Callbacks
from boosters.callbacks import Callback, EarlyStopping, LogEvaluation

__all__ = [
    "__version__",
    "Dataset", "EvalSet",
    "GBDTConfig", "GBLinearConfig", "TreeConfig",
    "RegularizationConfig", "SamplingConfig",
    "SquaredLoss", "AbsoluteLoss", "HuberLoss", "PinballLoss", "ArctanLoss",
    "PoissonLoss", "LogisticLoss", "HingeLoss", "SoftmaxLoss", "LambdaRankLoss",
    "Objective",
    "Rmse", "Mae", "Mape", "LogLoss", "Auc", "Accuracy", "Ndcg", "Metric",
    "GBDTModel", "GBLinearModel",
    "Callback", "EarlyStopping", "LogEvaluation",
]
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

### Quick Start (Core API)

This is the **core API**, not the sklearn wrapper. The core API uses explicit
dataclass configuration and `Dataset` objects for maximum type safety and control.

```python
import boosters as bst
import numpy as np

# Create dataset
X, y = np.random.rand(1000, 10), np.random.rand(1000)
train = bst.Dataset(features=X[:800], labels=y[:800])
valid = bst.Dataset(features=X[800:], labels=y[800:])

# Configure with explicit nested configs (Rust-owned types)
config = bst.GBDTConfig(
    n_estimators=100,
    learning_rate=0.1,
    objective=bst.SquaredLoss(),
    tree=bst.TreeConfig(max_depth=6, n_leaves=31),
    regularization=bst.RegularizationConfig(l2=0.1),
)

# Train
model = bst.GBDTModel(config)
model.fit(train, valid=[bst.EvalSet("valid", valid)])

# Predict
predictions = model.predict(X[800:])
```

**Multi-output quantile regression example:**

```python
# Multi-quantile prediction (3 output columns)
config = bst.GBDTConfig(
    objective=bst.PinballLoss(alpha=[0.1, 0.5, 0.9]),
)
model = bst.GBDTModel(config)
model.fit(train)
quantiles = model.predict(X_test)  # Shape: (n_samples, 3)
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
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
import numpy as np

@dataclass
class Dataset:
    """Training dataset with features, labels, and optional metadata.
    
    Dataset is a pure Python dataclass that holds references to numpy arrays.
    When passed to model.fit(), arrays are converted to zero-copy Rust views
    (if C/F-contiguous float32) or copied once (if conversion needed).
    """
    
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

# --- Objectives (parameterized, type-safe) ---

@dataclass(frozen=True)
class SquaredLoss:
    """Mean squared error loss."""
    pass

@dataclass(frozen=True)
class AbsoluteLoss:
    """Mean absolute error loss."""
    pass

@dataclass(frozen=True)
class HuberLoss:
    """Pseudo-Huber loss, smooth approximation to MAE."""
    delta: float = 1.0

@dataclass(frozen=True)
class PinballLoss:
    """Pinball loss for quantile regression.
    
    Standard piecewise-linear quantile loss. Prefer for most quantile
    regression tasks. For outlier-robust quantile regression, see ArctanLoss.
    """
    alpha: float | list[float] = 0.5  # Single or multiple quantiles

@dataclass(frozen=True)
class ArctanLoss:
    """Arctan loss for robust quantile regression.
    
    Smooth approximation to quantile loss, more robust to outliers than
    PinballLoss. Use when data has extreme outliers that shouldn't dominate.
    """
    alpha: float = 0.5

@dataclass(frozen=True)
class PoissonLoss:
    """Poisson deviance loss for count data."""
    pass

@dataclass(frozen=True)
class LogisticLoss:
    """Binary cross-entropy loss."""
    pass

@dataclass(frozen=True)
class HingeLoss:
    """SVM-style hinge loss."""
    pass

@dataclass(frozen=True)
class SoftmaxLoss:
    """Multi-class cross-entropy loss."""
    n_classes: int

@dataclass(frozen=True)
class LambdaRankLoss:
    """LambdaRank loss for learning to rank."""
    ndcg_at: int = 10

# Union type for all objectives
type Objective = (
    SquaredLoss | AbsoluteLoss | HuberLoss | PinballLoss | ArctanLoss |
    PoissonLoss | LogisticLoss | HingeLoss | SoftmaxLoss | LambdaRankLoss
)

# --- Metrics (parameterized where needed) ---

@dataclass(frozen=True)
class Rmse:
    """Root mean squared error."""
    pass

@dataclass(frozen=True)
class Mae:
    """Mean absolute error."""
    pass

@dataclass(frozen=True)
class Mape:
    """Mean absolute percentage error."""
    pass

@dataclass(frozen=True)
class LogLoss:
    """Binary log loss."""
    pass

@dataclass(frozen=True)
class Auc:
    """Area under ROC curve."""
    pass

@dataclass(frozen=True)
class Accuracy:
    """Classification accuracy."""
    pass

@dataclass(frozen=True)
class Ndcg:
    """Normalized discounted cumulative gain."""
    at: int = 10  # NDCG@k

type Metric = Rmse | Mae | Mape | LogLoss | Auc | Accuracy | Ndcg
```

### Objective Validation

Objectives with parameters validate at construction time via `__post_init__`:

```python
@dataclass(frozen=True)
class PinballLoss:
    alpha: float | list[float] = 0.5
    
    def __post_init__(self):
        alphas = [self.alpha] if isinstance(self.alpha, (int, float)) else self.alpha
        if len(alphas) == 0:
            raise ValueError("alpha must be non-empty")
        for a in alphas:
            if not 0 < a < 1:
                raise ValueError(f"alpha values must be in (0, 1), got {a}")

@dataclass(frozen=True)
class SoftmaxLoss:
    n_classes: int
    
    def __post_init__(self):
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {self.n_classes}")
```

### Output Shape by Objective

| Objective | Output Shape | Notes |
| --------- | ------------ | ----- |
| `SquaredLoss()` | `(n_samples,)` | Scalar regression |
| `PinballLoss(alpha=0.5)` | `(n_samples,)` | Single quantile |
| `PinballLoss(alpha=[0.1, 0.5, 0.9])` | `(n_samples, 3)` | Columns in alpha order |
| `ArctanLoss(alpha=0.5)` | `(n_samples,)` | Robust quantile |
| `LogisticLoss()` | `(n_samples,)` | Probability [0, 1] |
| `SoftmaxLoss(n_classes=k)` | `(n_samples, k)` | Probabilities per class |

**Note on multi-quantile**: Output columns are ordered by the `alpha` list. With `alpha=[0.1, 0.5, 0.9]`,
column 0 is the 10th percentile, column 1 is median, column 2 is 90th percentile. Quantile crossing
(where a higher quantile predicts lower than a lower quantile) can occur; users should post-process
if monotonicity is required.

### Evaluation Set (Named Datasets)

```python
@dataclass
class EvalSet:
    """Named evaluation dataset for validation during training."""
    name: str
    dataset: Dataset

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
    objective: Objective = field(default_factory=SquaredLoss)
    metrics: list[Metric] | None = None
    
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
    objective: Objective = field(default_factory=SquaredLoss)
    metrics: list[Metric] | None = None
    
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
        valid: EvalSet | list[EvalSet] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> Self:
        """Train the model.
        
        Args:
            train: Training dataset.
            valid: Named validation set(s) for early stopping and metrics.
                   Each EvalSet has a name used in eval_results.
            callbacks: Training callbacks (early stopping, logging).
        
        Returns:
            Self for method chaining.
        
        Example:
            >>> model.fit(
            ...     train_ds,
            ...     valid=[
            ...         EvalSet("train", train_ds),
            ...         EvalSet("valid", valid_ds),
            ...     ],
            ... )
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
                      Must have same n_features as training.
                      NaN values use default split direction.
            n_iterations: Limit trees used. None = best_iteration or all.
            raw_score: Return raw margins (no sigmoid/softmax).
            pred_leaf: Return leaf indices instead of predictions.
                       Cannot be combined with pred_contrib.
            pred_contrib: Return SHAP values. Implies raw output.
                          Cannot be combined with pred_leaf.
        
        Returns:
            Predictions array. Shape depends on objective and flags:
            
            Standard prediction:
            - Regression: (n_samples,)
            - Binary classification: (n_samples,) probabilities
            - Multi-class (k classes): (n_samples, k) probabilities
            - Quantile regression (q quantiles): (n_samples, q)
            
            With pred_leaf=True:
            - (n_samples, n_trees) leaf indices
            
            With pred_contrib=True (SHAP):
            - Regression/binary: (n_samples, n_features + 1)
            - Multi-class: (n_samples, n_features + 1, k)
            Last column is base value (expected value).
        
        Raises:
            ValueError: If both pred_leaf and pred_contrib are True.
            ValueError: If features has wrong number of columns.
        
        Note:
            For large datasets with pred_contrib, memory usage is
            O(n_samples × n_features × n_outputs). Consider batching
            manually for datasets >100k rows.
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
    def eval_results(self) -> dict[str, dict[str, list[float]]] | None:
        """Evaluation results from training.
        
        Returns:
            Nested dict: {dataset_name: {metric_name: [values_per_iteration]}}
            Example: {"valid": {"rmse": [0.5, 0.4, 0.35, ...]}}
        """
        ...
    
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
        valid: EvalSet | list[EvalSet] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> Self:
        """Train the model.
        
        Args:
            train: Training dataset.
            valid: Named validation set(s) for early stopping and metrics.
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
    ) -> NDArray[np.floating]:
        """Make predictions.
        
        Returns:
            Predictions array. Shape matches objective output dimensions.
        """
    
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
    valid: EvalSet | list[EvalSet] | None = None,
    callbacks: list[Callback] | None = None,
) -> GBDTModel:
    """Train a GBDT model.
    
    Args:
        config: Training configuration.
        train: Training dataset.
        valid: Named validation set(s).
        callbacks: Training callbacks.
    
    Returns:
        Trained model.
    """

def train_gblinear(
    config: GBLinearConfig,
    train: Dataset,
    *,
    valid: EvalSet | list[EvalSet] | None = None,
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
    def __call__(
        self,
        iteration: int,
        eval_results: dict[str, dict[str, float]],
    ) -> bool:
        """Called after each iteration.
        
        Args:
            iteration: Current iteration number.
            eval_results: Metric values for this iteration.
                         Format: {dataset_name: {metric_name: value}}
        
        Returns:
            True to stop training, False to continue.
        """

@dataclass
class EarlyStopping(Callback):
    """Stop training when validation metric stops improving."""
    
    patience: int = 50
    min_delta: float = 0.0
    monitor_set: str = "valid"  # Which EvalSet to monitor
    monitor_metric: str | None = None  # None = use first metric
    
    def __call__(
        self,
        iteration: int,
        eval_results: dict[str, dict[str, float]],
    ) -> bool: ...

@dataclass
class LogEvaluation(Callback):
    """Log evaluation metrics periodically."""
    
    period: int = 10
    
    def __call__(
        self,
        iteration: int,
        eval_results: dict[str, dict[str, float]],
    ) -> bool: ...
```

---

## Objective and Metric Mapping

### Objective Classes

| Python Objective | Rust Type | Parameters | Notes |
| ---------------- | --------- | ---------- | ----- |
| `SquaredLoss()` | `SquaredLoss` | — | Default for regression |
| `AbsoluteLoss()` | `AbsoluteLoss` | — | Robust to outliers (L1) |
| `HuberLoss(delta=1.0)` | `PseudoHuberLoss` | `delta: float` | Smooth L1/L2 blend |
| `PinballLoss(alpha=0.5)` | `PinballLoss` | `alpha: float \| list[float]` | Single or multi-quantile |
| `ArctanLoss(alpha=0.5)` | `ArctanLoss` | `alpha: float` | Robust quantile |
| `PoissonLoss()` | `PoissonLoss` | — | For count data |
| `LogisticLoss()` | `LogisticLoss` | — | Binary classification |
| `HingeLoss()` | `HingeLoss` | — | SVM-style classification |
| `SoftmaxLoss(n_classes=3)` | `SoftmaxLoss` | `n_classes: int` | Multiclass (required) |
| `LambdaRankLoss(ndcg_at=10)` | `LambdaRankLoss` | `ndcg_at: int` | Learning to rank |

### Metric Classes

| Python Metric | Rust Type | Parameters | Task |
| ------------- | --------- | ---------- | ---- |
| `Rmse()` | `Rmse` | — | Regression |
| `Mae()` | `Mae` | — | Regression |
| `Mape()` | `Mape` | — | Regression |
| `LogLoss()` | `LogLoss` | — | Classification |
| `Auc()` | `Auc` | — | Binary classification |
| `Accuracy()` | `Accuracy` | — | Classification |
| `Ndcg(at=10)` | `Ndcg` | `at: int` | Ranking |

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

# With validation for early stopping (sklearn-style shortcut)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,  # Shortcut for EarlyStopping callback
)

# Or with explicit callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[EarlyStopping(patience=50, min_delta=0.001)],
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
        early_stopping_rounds: int | None = None,
        callbacks: list[Callback] | None = None,
    ) -> Self: ...
    
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
    
    # sklearn-standard attributes (set after fit)
    n_features_in_: int
    feature_names_in_: NDArray[np.str_] | None
    feature_importances_: NDArray[np.floating]
    best_iteration_: int | None

class GBDTClassifier(BaseEstimator, ClassifierMixin):
    """GBDT classifier with sklearn interface.
    
    Automatically infers binary vs multiclass from labels:
    - 2 classes: LogisticLoss()
    - >2 classes: SoftmaxLoss(n_classes=k)
    """
    
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

### sklearn Parameter Mapping

sklearn flat parameters map to nested config:

| sklearn param | Core API config path |
| ------------- | -------------------- |
| `n_estimators` | `GBDTConfig.n_estimators` |
| `learning_rate` | `GBDTConfig.learning_rate` |
| `max_depth` | `GBDTConfig.tree.max_depth` |
| `n_leaves` | `GBDTConfig.tree.n_leaves` |
| `min_samples_leaf` | `GBDTConfig.tree.min_samples_leaf` |
| `l1` | `GBDTConfig.regularization.l1` |
| `l2` | `GBDTConfig.regularization.l2` |
| `subsample` | `GBDTConfig.sampling.subsample` |
| `colsample` | `GBDTConfig.sampling.colsample` |
| `linear_trees` | `GBDTConfig.linear_trees` |

### sklearn Conversions

```python
# eval_set conversion (sklearn → core API)
eval_set = [(X_val1, y_val1), (X_val2, y_val2)]
# Becomes:
valid = [
    EvalSet("valid_0", Dataset(X_val1, y_val1)),
    EvalSet("valid_1", Dataset(X_val2, y_val2)),
]

# early_stopping_rounds shortcut
early_stopping_rounds = 50
# Becomes:
callbacks = [EarlyStopping(patience=50, monitor_set="valid_0")]

# Classifier objective inference
y = [0, 1, 0, 1]  # Binary
# Infers: LogisticLoss()

y = [0, 1, 2, 0]  # Multiclass
# Infers: SoftmaxLoss(n_classes=3)
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

### Configuration Types: Rust-Owned with Generated Stubs

**Key design question**: Where do config types like `GBDTConfig`, `PinballLoss`, `EvalSet` live?

**Decision**: **Rust-owned** — config types are defined in Rust with `#[pyclass]`,
type stubs are auto-generated, Python uses Rust types directly.

| Type | Implementation | Rationale |
| ---- | -------------- | --------- |
| `GBDTConfig`, `TreeConfig`, etc. | Rust `#[pyclass]` | Single source of truth |
| `PinballLoss`, `SoftmaxLoss`, etc. | Rust `#[pyclass]` | Compiler-enforced completeness |
| `EvalSet` | Rust `#[pyclass]` | Type-safe validation set wrapper |
| `Dataset` | Rust `#[pyclass]` | Holds data references, lazy conversion |
| `GBDTModel`, `GBLinearModel` | Rust `#[pyclass]` | Training/prediction |

**Why Rust-owned configs?**

1. **Single source of truth**: Rust struct IS the definition
2. **Compiler-enforced**: Adding a Rust field requires updating PyO3 bindings
3. **No parser drift**: Direct field access, not dict parsing
4. **CI catches sync**: Stub generation + pyright catches Python code drift
5. **IDE support via stubs**: Generated `.pyi` files provide autocomplete

**Risk with Python-owned configs** (rejected approach):

```text
Rust adds field → Python dataclass not updated → Parser silently ignores → Bug
```

**Safe failure mode with Rust-owned configs**:

```text
Rust adds field → Stub regenerated → Python code using old API fails pyright → Caught in CI
```

### Rust Config Implementation

```rust
use pyo3::prelude::*;

/// Tree structure configuration.
#[pyclass(get_all, set_all)]
#[derive(Clone, Default)]
pub struct TreeConfig {
    /// Maximum tree depth. -1 for unlimited.
    pub max_depth: i32,
    /// Maximum number of leaves.
    pub n_leaves: u32,
    /// Minimum samples per leaf.
    pub min_samples_leaf: u32,
    /// Minimum gain to make a split.
    pub min_gain_to_split: f64,
}

#[pymethods]
impl TreeConfig {
    #[new]
    #[pyo3(signature = (max_depth=-1, n_leaves=31, min_samples_leaf=20, min_gain_to_split=0.0))]
    fn new(max_depth: i32, n_leaves: u32, min_samples_leaf: u32, min_gain_to_split: f64) -> Self {
        Self { max_depth, n_leaves, min_samples_leaf, min_gain_to_split }
    }
}

/// Regularization parameters.
#[pyclass(get_all, set_all)]
#[derive(Clone, Default)]
pub struct RegularizationConfig {
    pub l1: f64,
    pub l2: f64,
}

#[pymethods]
impl RegularizationConfig {
    #[new]
    #[pyo3(signature = (l1=0.0, l2=0.0))]
    fn new(l1: f64, l2: f64) -> Self {
        Self { l1, l2 }
    }
}

/// GBDT training configuration.
#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct GBDTConfig {
    pub n_estimators: u32,
    pub learning_rate: f64,
    pub objective: PyObject,  // Union type handled via PyObject
    pub metrics: Option<Vec<PyObject>>,
    pub tree: Py<TreeConfig>,
    pub regularization: Py<RegularizationConfig>,
    // ...
}

#[pymethods]
impl GBDTConfig {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        learning_rate=0.1,
        objective=None,
        metrics=None,
        tree=None,
        regularization=None,
        // ...
    ))]
    fn new(
        py: Python<'_>,
        n_estimators: u32,
        learning_rate: f64,
        objective: Option<PyObject>,
        metrics: Option<Vec<PyObject>>,
        tree: Option<Py<TreeConfig>>,
        regularization: Option<Py<RegularizationConfig>>,
    ) -> PyResult<Self> {
        Ok(Self {
            n_estimators,
            learning_rate,
            objective: objective.unwrap_or_else(|| {
                Py::new(py, SquaredLoss {}).unwrap().into_py(py)
            }),
            metrics,
            tree: tree.unwrap_or_else(|| Py::new(py, TreeConfig::default()).unwrap()),
            regularization: regularization.unwrap_or_else(|| {
                Py::new(py, RegularizationConfig::default()).unwrap()
            }),
        })
    }
}
```

### Objective Types (Enum Variants as Separate Classes)

Each objective is a separate `#[pyclass]`:

```rust
/// Mean squared error loss.
#[pyclass]
#[derive(Clone)]
pub struct SquaredLoss;

#[pymethods]
impl SquaredLoss {
    #[new]
    fn new() -> Self { Self }
}

/// Pinball loss for quantile regression.
#[pyclass(get_all, set_all)]
#[derive(Clone)]
pub struct PinballLoss {
    /// Quantile(s) to predict. Single value or list.
    pub alpha: PyObject,  // f64 | Vec<f64>
}

#[pymethods]
impl PinballLoss {
    #[new]
    #[pyo3(signature = (alpha=0.5))]
    fn new(py: Python<'_>, alpha: PyObject) -> PyResult<Self> {
        // Validate alpha in (0, 1)
        validate_alpha(py, &alpha)?;
        Ok(Self { alpha: alpha.into_py(py) })
    }
}

/// Multi-class cross-entropy loss.
#[pyclass(get_all)]
#[derive(Clone)]
pub struct SoftmaxLoss {
    pub n_classes: u32,
}

#[pymethods]
impl SoftmaxLoss {
    #[new]
    fn new(n_classes: u32) -> PyResult<Self> {
        if n_classes < 2 {
            return Err(PyValueError::new_err("n_classes must be >= 2"));
        }
        Ok(Self { n_classes })
    }
}
```

Python sees these as separate classes, and we define the union in the stub:

```python
# _boosters_rs.pyi (generated)
class SquaredLoss:
    def __init__(self) -> None: ...

class PinballLoss:
    alpha: float | list[float]
    def __init__(self, alpha: float | list[float] = 0.5) -> None: ...

class SoftmaxLoss:
    n_classes: int
    def __init__(self, n_classes: int) -> None: ...

# Union defined in Python wrapper or stub
Objective = SquaredLoss | AbsoluteLoss | HuberLoss | PinballLoss | ...
```

### Objective Union Handling in Rust

To maintain type safety, use a Rust enum with `FromPyObject`:

```rust
use pyo3::prelude::*;

/// Rust-side objective enum for type-safe extraction.
#[derive(Clone, FromPyObject)]
pub enum PyObjective {
    Squared(SquaredLoss),
    Absolute(AbsoluteLoss),
    Huber(HuberLoss),
    Pinball(PinballLoss),
    Arctan(ArctanLoss),
    Poisson(PoissonLoss),
    Logistic(LogisticLoss),
    Hinge(HingeLoss),
    Softmax(SoftmaxLoss),
    LambdaRank(LambdaRankLoss),
}

// Used in GBDTConfig
#[pyclass(get_all, set_all)]
pub struct GBDTConfig {
    pub n_estimators: u32,
    pub learning_rate: f64,
    pub objective: PyObject,  // Stored as PyObject for Python access
    pub tree: Py<TreeConfig>,
    // ...
}

#[pymethods]
impl GBDTConfig {
    /// Extract objective as Rust enum (for internal use).
    fn objective_kind(&self, py: Python<'_>) -> PyResult<PyObjective> {
        self.objective.extract(py)
    }
}
```

**Error messages** for invalid objectives:

```python
# Wrong type
config = GBDTConfig(objective="squared")
# TypeError: 'str' object cannot be converted to 'SquaredLoss | AbsoluteLoss | ...'

# Valid objective, invalid params
config = GBDTConfig(objective=PinballLoss(alpha=1.5))
# ValueError: alpha must be in (0, 1), got 1.5

# Custom class (not supported)
class MyLoss: pass
config = GBDTConfig(objective=MyLoss())
# TypeError: expected one of [SquaredLoss, AbsoluteLoss, ...], got 'MyLoss'
```

### Type Stub Generation

**Tool**: `pyo3-stub-gen` or Maturin's built-in stub generation.

```toml
# pyproject.toml
[tool.maturin]
python-source = "python"
module-name = "boosters._boosters_rs"

[tool.maturin.generate-stubs]
enabled = true
```

**Generated stub example** (`_boosters_rs.pyi`):

```python
from numpy.typing import NDArray
import numpy as np

class TreeConfig:
    max_depth: int
    n_leaves: int
    min_samples_leaf: int
    min_gain_to_split: float
    
    def __init__(
        self,
        max_depth: int = -1,
        n_leaves: int = 31,
        min_samples_leaf: int = 20,
        min_gain_to_split: float = 0.0,
    ) -> None: ...

class GBDTConfig:
    n_estimators: int
    learning_rate: float
    objective: Objective
    tree: TreeConfig
    regularization: RegularizationConfig
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        objective: Objective | None = None,
        tree: TreeConfig | None = None,
        regularization: RegularizationConfig | None = None,
        # ...
    ) -> None: ...

class GBDTModel:
    def __init__(self, config: GBDTConfig | None = None) -> None: ...
    def fit(
        self,
        train: Dataset,
        valid: EvalSet | list[EvalSet] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> GBDTModel: ...
    def predict(
        self,
        features: NDArray[np.floating],
        n_iterations: int | None = None,
        raw_score: bool = False,
    ) -> NDArray[np.floating]: ...
    
    @property
    def n_trees(self) -> int: ...
    @property
    def n_features(self) -> int: ...
```

### CI Stub Verification

```yaml
# .github/workflows/ci.yml
- name: Generate stubs
  run: maturin develop --generate-stubs

- name: Type check
  run: pyright boosters/

- name: Verify stubs match
  run: |
    # Ensure generated stubs are committed
    git diff --exit-code python/boosters/_boosters_rs.pyi
```

If a developer adds a Rust field but forgets to regenerate stubs:

1. CI regenerates stubs
2. `git diff --exit-code` fails
3. PR blocked until stubs committed

### Python Re-exports (Thin Wrapper)

The `boosters` package re-exports Rust types with Pythonic naming:

```python
# boosters/__init__.py
from boosters._boosters_rs import (
    # Config
    GBDTConfig, GBLinearConfig, TreeConfig,
    RegularizationConfig, SamplingConfig,
    # Objectives
    SquaredLoss, AbsoluteLoss, HuberLoss, PinballLoss, ArctanLoss,
    PoissonLoss, LogisticLoss, HingeLoss, SoftmaxLoss, LambdaRankLoss,
    # Metrics  
    Rmse, Mae, Mape, LogLoss, Auc, Accuracy, Ndcg,
    # Data
    Dataset, EvalSet,
    # Models
    GBDTModel, GBLinearModel,
    # Callbacks
    EarlyStopping, LogEvaluation,
)

# Type aliases for documentation
from typing import TypeAlias
Objective: TypeAlias = (
    SquaredLoss | AbsoluteLoss | HuberLoss | PinballLoss | ArctanLoss |
    PoissonLoss | LogisticLoss | HingeLoss | SoftmaxLoss | LambdaRankLoss
)
Metric: TypeAlias = Rmse | Mae | Mape | LogLoss | Auc | Accuracy | Ndcg
```

### Validation in Rust

With Rust-owned types, validation happens in Rust constructors:

```rust
#[pymethods]
impl PinballLoss {
    #[new]
    #[pyo3(signature = (alpha=0.5))]
    fn new(py: Python<'_>, alpha: PyObject) -> PyResult<Self> {
        // Handle both f64 and Vec<f64>
        if let Ok(single) = alpha.extract::<f64>(py) {
            if single <= 0.0 || single >= 1.0 {
                return Err(PyValueError::new_err(
                    format!("alpha must be in (0, 1), got {}", single)
                ));
            }
        } else if let Ok(multi) = alpha.extract::<Vec<f64>>(py) {
            if multi.is_empty() {
                return Err(PyValueError::new_err("alpha must be non-empty"));
            }
            for a in &multi {
                if *a <= 0.0 || *a >= 1.0 {
                    return Err(PyValueError::new_err(
                        format!("alpha values must be in (0, 1), got {}", a)
                    ));
                }
            }
        } else {
            return Err(PyTypeError::new_err(
                "alpha must be float or list of floats"
            ));
        }
        Ok(Self { alpha: alpha.into_py(py) })
    }
}
```

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

### Python-Side Validation

Before calling Rust, validate in Python for better error messages:

```python
def _validate_predict_flags(
    pred_leaf: bool,
    pred_contrib: bool,
) -> None:
    """Validate mutually exclusive prediction flags."""
    if pred_leaf and pred_contrib:
        raise ValueError(
            "pred_leaf and pred_contrib cannot both be True. "
            "pred_leaf returns leaf indices, pred_contrib returns SHAP values."
        )

def _validate_feature_count(
    features: NDArray,
    expected: int,
) -> None:
    """Validate feature count matches model."""
    actual = features.shape[1] if features.ndim == 2 else 1
    if actual != expected:
        raise ValueError(
            f"Feature count mismatch: model expects {expected} features, "
            f"got {actual}. Ensure predict input has same features as training."
        )
```

### Common Error Scenarios

| Scenario | Exception | Message |
| -------- | --------- | ------- |
| Model not fitted | `RuntimeError` | "Model not fitted. Call fit() first." |
| Wrong feature count | `ValueError` | "Feature count mismatch: model expects N, got M." |
| Invalid objective param | `ValueError` | "alpha must be in (0, 1), got 2.0" |
| Duplicate EvalSet name | `ValueError` | "Duplicate EvalSet name: 'valid'" |
| pred_leaf + pred_contrib | `ValueError` | "pred_leaf and pred_contrib cannot both be True." |
| Empty features array | Returns | Empty array of correct shape (not error) |

---

## Design Decisions

### DD-1: PyO3 + Maturin

**Decision**: Use PyO3 for bindings, Maturin for build.

**Rationale**: Single build step, excellent numpy integration, active ecosystem.

### DD-2: Separate Model Types

**Decision**: `GBDTModel` and `GBLinearModel` as separate classes (not unified `Booster`).

**Rationale**: Matches Rust API, clearer type safety, avoid enum dispatch overhead.

### DD-3: Rust-Owned Config Types with Generated Stubs

**Decision**: Configuration types (`GBDTConfig`, `PinballLoss`, `EvalSet`) are defined in
Rust using `#[pyclass]`. Type stubs are auto-generated via `pyo3-stub-gen` or Maturin.
Python uses the Rust types directly (thin re-export wrapper only).

**Rationale**:

- **Single source of truth**: Rust struct IS the definition, no sync risk
- **Compiler-enforced**: Adding a Rust field requires updating PyO3 `#[new]`
- **No parser drift**: Direct field access, not dict parsing that can miss fields
- **CI catches sync**: `git diff --exit-code` on stubs blocks stale PRs
- **IDE support via stubs**: Generated `.pyi` files provide autocomplete

**Rejected alternative**: Pure Python dataclasses with parser conversion.

- Risk: Parser silently ignores missing fields → runtime bugs
- Problem: No compile-time guarantee between Python dataclass and Rust struct

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

### DD-9: Parameterized Objectives (Struct Initialization Pattern)

**Decision**: Objectives and metrics are parameterized dataclasses, not simple enums.

**Rationale**:

- **Type safety**: `QuantileLoss(alpha=0.5)` vs `(Objective.QUANTILE, {"alpha": 0.5})`
- **IDE support**: Autocomplete shows required parameters
- **Validation**: Dataclass construction validates parameters
- **Rust parity**: Mirrors Rust's struct initialization style
- **Self-documenting**: Parameters are explicit in the type

Example mapping Rust → Python:

```rust
// Rust API
let objective = Objective::Quantile { alpha: 0.5 };
let objective = Objective::Softmax { n_classes: 3 };
```

```python
# Python API (struct initialization style)
objective = QuantileLoss(alpha=0.5)
objective = SoftmaxLoss(n_classes=3)
```

### DD-10: Named Evaluation Sets (EvalSet Pattern)

**Decision**: Validation sets use `EvalSet(name, dataset)` wrapper.

**Rationale**:

- **Named results**: `eval_results["valid"]["rmse"]` vs `eval_results[0]["rmse"]`
- **Rust parity**: Matches Rust training API's named eval sets
- **Multiple datasets**: Train metrics, validation metrics clearly distinguished
- **Early stopping clarity**: `early_stopping_set="valid"` references by name

### DD-11: Rust-Python API Synchronization Strategy

**Decision**: Python API mirrors Rust API structure; both use struct/dataclass initialization.

**Strategy**:

1. **Mirror types**: Each Rust struct gets a Python dataclass equivalent
2. **Same naming**: `GBDTConfig` in Rust → `GBDTConfig` in Python
3. **Same structure**: Nested configs in both (TreeConfig, RegularizationConfig)
4. **Same defaults**: Default values match between Rust and Python
5. **Generated checks**: CI runs a check that Python types cover all Rust fields

**Naming alignment**: Where Rust uses technical names (e.g., `PinballLoss`), Python uses
user-friendly names (e.g., `QuantileLoss`). The Rust API should align to user-friendly
names where practical, as most users interact via Python.

| Rust Name | Python Name | User-Friendly? |
| --------- | ----------- | -------------- |
| `SquaredLoss` | `SquaredLoss` | ✓ Same |
| `PinballLoss` | `PinballLoss` | ✓ Same |
| `ArctanLoss` | `ArctanLoss` | ✓ Same |
| `PseudoHuberLoss` | `HuberLoss` | Python preferred |

**Synchronization checkpoints**:

- When adding a Rust config field → add to Python dataclass
- When adding a Rust objective variant → add Python objective class
- When renaming Rust types → rename Python types

**Automated verification** (future):

```python
# Generated from Rust types via proc-macro or build script
def verify_api_sync():
    """Ensure Python API covers all Rust API surface."""
    rust_fields = get_rust_config_fields()  # From cbindgen or similar
    python_fields = get_dataclass_fields(GBDTConfig)
    assert rust_fields == python_fields
```

### DD-12: Default Metric and Early Stopping Behavior

**Decisions**:

1. **Default metric**: If `config.metrics=None`, infer from objective:
   - Regression objectives → `[Rmse()]`
   - `LogisticLoss` → `[LogLoss(), Auc()]`
   - `SoftmaxLoss` → `[LogLoss(), Accuracy()]`
   - `LambdaRankLoss` → `[Ndcg(at=config.objective.ndcg_at)]`

2. **Early stopping monitor**: `EarlyStopping.monitor_metric` defaults to first metric in list.

3. **Duplicate EvalSet names**: Raise `ValueError` at fit time if two EvalSets have the same name.

4. **Reserved names**: `"train"` is reserved for training metrics (if `eval_train=True`).

**Rationale**: Explicit defaults reduce user confusion while allowing full customization.

### DD-13: Nested Configs (Core API) vs Flat Kwargs (sklearn)

**Decision**: Core API uses **nested configs only**. Sklearn wrapper provides **flat kwargs**.

**Core API pattern** (explicit):

```python
# User must construct sub-configs explicitly
config = GBDTConfig(
    tree=TreeConfig(max_depth=5),
    regularization=RegularizationConfig(l2=0.1),
)
model = GBDTModel(config)
```

**sklearn pattern** (flat):

```python
# Flat kwargs routed to appropriate sub-config
model = GBDTRegressor(max_depth=5, reg_lambda=0.1)
```

**Rationale**:

- **Core API clarity**: Explicit nesting shows which config owns each parameter
- **IDE navigation**: Jump to `TreeConfig` definition to see all tree params
- **No ambiguity**: `l1` could be tree regularization or linear leaf regularization
- **sklearn compatibility**: Users expect flat kwargs from existing XGBoost/LightGBM usage
- **Separation of concerns**: Core API prioritizes correctness, sklearn prioritizes convenience

**Not supported in core API** (no flat kwargs routing):

```python
# This does NOT work in core API
config = GBDTConfig(max_depth=5)  # Error: unexpected kwarg 'max_depth'
```

### DD-14: Objective Union Type Safety

**Decision**: Objectives use a Rust enum (`PyObjective`) with `FromPyObject` for type-safe extraction.
Config stores `PyObject` for Python accessibility, but extracts to enum for Rust usage.

**Implementation**:

```rust
#[derive(Clone, FromPyObject)]
pub enum PyObjective {
    Squared(SquaredLoss),
    Pinball(PinballLoss),
    // ... all variants
}
```

**Error message strategy**:

| Error Type | Message Pattern |
| ---------- | --------------- |
| Wrong Python type | `TypeError: 'str' cannot be converted to 'SquaredLoss \| AbsoluteLoss \| ...'` |
| Invalid params | `ValueError: PinballLoss.alpha must be in (0, 1), got 1.5` |
| Unknown class | `TypeError: expected one of [SquaredLoss, ...], got 'MyCustomLoss'` |

**Rationale**:

- **Type safety**: Rust compiler ensures all objective variants are handled
- **Clear errors**: Users get actionable error messages
- **Extensibility**: Adding new objectives requires adding enum variant (compiler-enforced)

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
from boosters import GBDTModel, GBDTConfig, LogisticLoss, SoftmaxLoss

# Binary classification
config = GBDTConfig(objective=LogisticLoss())
model = GBDTModel(config)
model.fit(train)
probs = model.predict(X)  # (n_samples,) - probabilities

# Multiclass classification
config = GBDTConfig(objective=SoftmaxLoss(n_classes=3))
model = GBDTModel(config)
model.fit(train)
probs = model.predict(X)              # (n_samples, n_classes)
logits = model.predict(X, raw_score=True)  # Raw logits
```

### Multi-Output Regression

```python
from boosters import SquaredLoss, QuantileLoss

# Native multi-output (one tree per output per round)
config = GBDTConfig(
    objective=SquaredLoss(),
    n_outputs=2,
)
model = GBDTModel(config)
model.fit(train)
preds = model.predict(X)  # (n_samples, n_outputs)

# Quantile regression with multiple quantiles
config = GBDTConfig(
    objective=PinballLoss(alpha=[0.1, 0.5, 0.9]),  # 10th, 50th, 90th percentiles
)
model = GBDTModel(config)
model.fit(train)
preds = model.predict(X)  # (n_samples, 3) - one column per quantile
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

### Validation Timing

With Rust-owned config types, validation is split across two stages:

| Validation Stage | What's Checked | Examples |
| ---------------- | -------------- | -------- |
| **Constructor** | Type correctness, range validity | `alpha ∈ (0,1)`, `n_classes ≥ 2`, `n_estimators > 0` |
| **Fit** | Cross-field consistency, data compatibility | `max_depth` vs `n_leaves` feasibility, feature count matches |

**Constructor-time validation** (Rust `#[new]` methods):

```python
from boosters import GBDTConfig, TreeConfig, PinballLoss

# Type errors caught immediately
config = GBDTConfig(learning_rate='fast')
# TypeError: 'str' cannot be converted to 'float'

# Value validation in Rust constructor
config = GBDTConfig(n_estimators=0)
# ValueError: n_estimators must be positive, got 0

# Nested config validation
config = GBDTConfig(tree=TreeConfig(n_leaves=0))
# ValueError: n_leaves must be positive, got 0

# Parameterized objective validation
obj = PinballLoss(alpha=1.5)
# ValueError: alpha must be in (0, 1), got 1.5
```

**Fit-time validation** (cross-field and data checks):

```python
# Cross-field consistency
config = GBDTConfig(tree=TreeConfig(max_depth=2, n_leaves=100))
model = GBDTModel(config)
model.fit(train)  
# ValueError: max_depth=2 cannot produce n_leaves=100 (max 4 leaves at depth 2)

# Data compatibility
model.fit(train)  # train has 100 features
model.predict(X)  # X has 50 features
# ValueError: expected 100 features, got 50
```

### Complete Sub-Config Reference

All configuration types exposed via PyO3:

| Config | Fields | Purpose |
| ------ | ------ | ------- |
| `TreeConfig` | `max_depth`, `n_leaves`, `min_samples_leaf`, `min_gain_to_split` | Tree structure |
| `RegularizationConfig` | `l1`, `l2` | L1/L2 penalties |
| `SamplingConfig` | `subsample`, `colsample`, `goss_alpha`, `goss_beta` | Row/column sampling |
| `EFBConfig` | `enable`, `max_conflict_rate` | Exclusive Feature Bundling |
| `CategoricalConfig` | `max_categories`, `min_category_count` | Category handling |
| `LinearLeavesConfig` | `enable`, `l2_regularization` | Linear models in leaves |

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
| `max_depth` | `tree.max_depth` (or flat `max_depth` in sklearn) |
| `learning_rate` / `eta` | `learning_rate` |
| `reg_lambda` | `regularization.l2` (or flat `reg_lambda` in sklearn) |
| `reg_alpha` | `regularization.l1` (or flat `reg_alpha` in sklearn) |
| `subsample` | `sampling.subsample` |
| `colsample_bytree` | `sampling.colsample` |
| `min_child_weight` | `tree.min_samples_leaf` (approximate) |
| `gamma` | `tree.min_gain_to_split` |
| `objective="reg:squarederror"` | `objective=SquaredLoss()` |
| `objective="binary:logistic"` | `objective=LogisticLoss()` |
| `objective="multi:softmax"` | `objective=SoftmaxLoss(n_classes=k)` |

### From LightGBM

| LightGBM Parameter | boosters Equivalent |
| ------------------ | ------------------- |
| `n_estimators` / `num_iterations` | `n_estimators` |
| `num_leaves` | `tree.n_leaves` |
| `max_depth` | `tree.max_depth` |
| `learning_rate` | `learning_rate` |
| `lambda_l1` | `regularization.l1` |
| `lambda_l2` | `regularization.l2` |
| `bagging_fraction` | `sampling.subsample` |
| `feature_fraction` | `sampling.colsample` |
| `min_data_in_leaf` | `tree.min_samples_leaf` |
| `min_gain_to_split` | `tree.min_gain_to_split` |
| `objective="regression"` | `objective=SquaredLoss()` |
| `objective="binary"` | `objective=LogisticLoss()` |
| `objective="multiclass"` | `objective=SoftmaxLoss(n_classes=k)` |

### sklearn Wrapper vs Core API

| sklearn Wrapper | Core API Equivalent |
| --------------- | ------------------- |
| `GBDTRegressor(max_depth=5)` | `GBDTModel(GBDTConfig(tree=TreeConfig(max_depth=5)))` |
| `clf.fit(X, y, eval_set=[(X_val, y_val)])` | `model.fit(train, valid=[EvalSet("valid", valid)])` |
| `clf.fit(..., early_stopping_rounds=10)` | `model.fit(..., callbacks=[EarlyStopping(patience=10)])` |

---

## Testing Strategy

### Test Categories

1. **Unit Tests** (pytest, fast):
   - Data conversion: NumPy, pandas, PyArrow, scipy.sparse
   - Parameter parsing and validation
   - Dataclass construction and validation
   - Basic train/predict flow

2. **Integration Tests** (pytest, medium):
   - sklearn estimator compliance (`check_estimator()`)
   - Cross-validation compatibility
   - Pipeline integration
   - Early stopping callbacks
   - EvalSet handling

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
   - Zero-copy verification

6. **Thread Safety Tests**:
   - Concurrent `predict()` calls from multiple Python threads
   - Concurrent training (should block or error, not corrupt)
   - Dataset shared across threads (read-only safe)

### Objective/Metric Test Matrix

For each objective type, verify:

| Objective | Output Range | Shape | Loss Decreases | Reference Comparison |
| --------- | ------------ | ----- | -------------- | -------------------- |
| `SquaredLoss()` | (-∞, ∞) | (n,) | ✓ RMSE | XGBoost correlation > 0.99 |
| `LogisticLoss()` | [0, 1] | (n,) | ✓ LogLoss | XGBoost AUC within 1% |
| `SoftmaxLoss(k)` | [0, 1], sum=1 | (n, k) | ✓ LogLoss | XGBoost accuracy within 2% |
| `PinballLoss([α])` | (-∞, ∞) | (n, len(α)) | ✓ Pinball | Calibration check |

### Edge Case Tests

```python
# Zero samples
def test_predict_empty_input():
    preds = model.predict(np.zeros((0, 10)))
    assert preds.shape == (0,)

# All same labels
def test_all_same_labels():
    y = np.ones(100)
    model.fit(Dataset(X, y))
    preds = model.predict(X)
    assert np.allclose(preds, 1.0, atol=0.1)

# Category not seen in training
def test_unseen_category():
    # Training has categories [0, 1, 2]
    # Predict has category [3]
    # Should use default split direction (not crash)

# Duplicate EvalSet names
def test_duplicate_evalset_names():
    with pytest.raises(ValueError, match="Duplicate EvalSet name"):
        model.fit(train, valid=[
            EvalSet("valid", ds1),
            EvalSet("valid", ds2),  # Duplicate!
        ])
```

### API Synchronization Tests

```python
def test_rust_python_api_sync():
    """Verify Python API covers all Rust config fields."""
    from boosters._internal import get_rust_config_schema
    
    rust_fields = get_rust_config_schema("GBDTConfig")
    python_fields = set(f.name for f in fields(GBDTConfig))
    
    missing = rust_fields - python_fields
    assert not missing, f"Python missing Rust fields: {missing}"
    
    extra = python_fields - rust_fields
    assert not extra, f"Python has extra fields not in Rust: {extra}"
```

### Zero-Copy Verification

```python
def test_numpy_zero_copy():
    """Verify large arrays aren't copied when passed to Rust."""
    arr = np.random.rand(100_000, 10).astype(np.float32)
    original_ptr = arr.ctypes.data
    
    # Internal helper to check data pointer in Rust
    rust_ptr = model._get_data_pointer(arr)
    
    assert original_ptr == rust_ptr, "Array was copied!"
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
