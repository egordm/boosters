# Boosters Python

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://egordm.github.io/booste-rs/)

Fast gradient boosting library with native Rust core. Provides both a core API
and sklearn-compatible estimators.

📚 **[Full Documentation](https://egordm.github.io/booste-rs/)** — See the main documentation for tutorials, API reference, and guides.

## Features

- **High Performance**: Native Rust core with Python bindings via PyO3
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

# Binary classification
y_cls = (X[:, 0] > 0).astype(int)
clf = GBDTClassifier(n_estimators=50)
clf.fit(X, y_cls)
proba = clf.predict_proba(X)

# Multiclass classification (explicit objective required)
from boosters import Objective
y_multi = np.random.randint(0, 3, size=100)
clf_multi = GBDTClassifier(
    n_estimators=50,
    objective=Objective.softmax(n_classes=3)
)
clf_multi.fit(X, y_multi)
```

### Core API

The core API provides full control with flat config parameters:

```python
import boosters as bst
import numpy as np

# Create config
config = bst.GBDTConfig(
    n_estimators=100,
    learning_rate=0.1,
    objective=bst.Objective.squared(),
    metric=bst.Metric.rmse(),
    max_depth=5,
    l2=1.0,
)

# Create model and train
X = np.random.randn(100, 10).astype(np.float32)
y = np.random.randn(100).astype(np.float32)

model = bst.GBDTModel(config=config)
train_data = bst.Dataset(X, y)
model.fit(train_data)

# Predict
predictions = model.predict(bst.Dataset(X))
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

### Objectives

| Method | Description |
|--------|-------------|
| `Objective.squared()` | L2 regression |
| `Objective.absolute()` | L1 regression |
| `Objective.huber(delta)` | Huber loss |
| `Objective.logistic()` | Binary classification |
| `Objective.softmax(n_classes)` | Multiclass classification |
| `Objective.poisson()` | Poisson regression |
| `Objective.pinball(alpha)` | Quantile regression |
| `Objective.lambdarank(ndcg_at)` | Learning to rank |

### Metrics

| Method | Description |
|--------|-------------|
| `Metric.rmse()` | Root mean squared error |
| `Metric.mae()` | Mean absolute error |
| `Metric.logloss()` | Log loss / cross-entropy |
| `Metric.auc()` | Area under ROC curve |
| `Metric.Accuracy()` | Classification accuracy |
| `Metric.ndcg(at)` | Normalized discounted cumulative gain |

## Examples

See the [examples/](examples/) directory:

- [01_sklearn_quickstart.py](examples/01_sklearn_quickstart.py) - Basic sklearn API usage
- [02_core_api.py](examples/02_core_api.py) - Core API with full control
- [03_sklearn_integration.py](examples/03_sklearn_integration.py) - Pipeline, cross-validation, grid search
- [04_linear_models.py](examples/04_linear_models.py) - GBLinear models

## License

MIT
