# Boosters Python

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast gradient boosting library with native Rust core. Provides both a core API with
nested configs and sklearn-compatible estimators with flat kwargs.

## Features

- **High Performance**: Native Rust core with Python bindings via PyO3
- **Dual API**: Core API (nested configs) and sklearn API (flat kwargs)
- **sklearn Compatible**: Works with `Pipeline`, `cross_val_score`, `GridSearchCV`
- **Multiple Objectives**: Regression, classification, ranking, quantile regression
- **GBDT & Linear**: Both tree-based and linear boosting models

## Installation

```bash
# Development install from workspace root
cd /path/to/booste-rs
uv run maturin develop -m packages/boosters-python/Cargo.toml
```

## Quick Start

### sklearn API (Recommended)

The sklearn-compatible estimators provide familiar flat kwargs:

```python
from boosters.sklearn import GBDTRegressor, GBDTClassifier
import numpy as np

# Regression
X = np.random.randn(100, 5).astype(np.float32)
y = X[:, 0] + np.random.randn(100).astype(np.float32) * 0.1

reg = GBDTRegressor(max_depth=5, n_estimators=100)
reg.fit(X, y)
predictions = reg.predict(X)

# Classification
y_cls = (X[:, 0] > 0).astype(int)
clf = GBDTClassifier(n_estimators=50)
clf.fit(X, y_cls)
proba = clf.predict_proba(X)
```

### Core API

The core API provides full control with nested config objects:

```python
import boosters as bst
import numpy as np

# Create config with nested structure
config = bst.GBDTConfig(
    n_estimators=100,
    learning_rate=0.1,
    objective=bst.SquaredLoss(),
    metric=bst.Rmse(),
    tree=bst.TreeConfig(max_depth=5, n_leaves=31),
    regularization=bst.RegularizationConfig(l2=1.0),
)

# Create model and train
model = bst.GBDTModel(config=config)
train_data = bst.Dataset(X, y)
model.fit(train_data)

# Predict
predictions = model.predict(X)
```

### sklearn Integration

Works seamlessly with sklearn tools:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from boosters.sklearn import GBDTRegressor

# Cross-validation
scores = cross_val_score(GBDTRegressor(), X, y, cv=5)

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GBDTRegressor(n_estimators=50)),
])
pipe.fit(X, y)

# Grid search
param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
grid = GridSearchCV(GBDTRegressor(), param_grid, cv=3)
grid.fit(X, y)
```

## API Reference

### sklearn Estimators

| Class | Description |
|-------|-------------|
| `GBDTRegressor` | Gradient boosted trees for regression |
| `GBDTClassifier` | Gradient boosted trees for classification |
| `GBLinearRegressor` | Gradient boosted linear model for regression |
| `GBLinearClassifier` | Gradient boosted linear model for classification |

### Core Types

| Class | Description |
|-------|-------------|
| `GBDTModel` | Tree-based gradient boosting model |
| `GBLinearModel` | Linear gradient boosting model |
| `GBDTConfig` | Configuration for GBDT models |
| `GBLinearConfig` | Configuration for linear models |
| `Dataset` | Data wrapper for features and labels |
| `EvalSet` | Named evaluation set for validation |

### Objectives

| Class | Description |
|-------|-------------|
| `SquaredLoss` | L2 regression |
| `AbsoluteLoss` | L1 regression |
| `HuberLoss` | Huber loss (delta parameter) |
| `LogisticLoss` | Binary classification |
| `SoftmaxLoss` | Multiclass classification |
| `PoissonLoss` | Poisson regression |
| `PinballLoss` | Quantile regression |
| `LambdaRankLoss` | Learning to rank |

### Metrics

| Class | Description |
|-------|-------------|
| `Rmse` | Root mean squared error |
| `Mae` | Mean absolute error |
| `LogLoss` | Log loss / cross-entropy |
| `Auc` | Area under ROC curve |
| `Accuracy` | Classification accuracy |
| `Ndcg` | Normalized discounted cumulative gain |

## Status

This package is feature-complete for v0.1.0 MVP. See [RFC-0014](../../docs/rfcs/0014-python-bindings.md) for design details.

### Completed

- ✅ Package structure and build (Epic 1)
- ✅ Type stub generation
- ✅ Python tooling (ruff, pyright, pytest)
- ✅ CI pipeline
- ✅ Configuration types (Epic 2)
- ✅ Dataset handling (Epic 3)
- ✅ Model training/prediction (Epic 4)
- ✅ scikit-learn integration (Epic 5)

### Planned

- 📋 API documentation site
- 📋 Example notebooks
- 📋 Migration guide from XGBoost/LightGBM
- 📋 PyPI release

## License

MIT
