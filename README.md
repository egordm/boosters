# 🚀 boosters

A fast, pure-Rust gradient boosting library for training and inference.

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://egordm.github.io/booste-rs/)
[![PyPI](https://img.shields.io/pypi/v/boosters)](https://pypi.org/project/boosters/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Work in Progress**: This library is under active development but already functional for training and inference with XGBoost/LightGBM compatibility.

## What is boosters?

boosters is a gradient boosting implementation written from scratch in Rust, designed to be:

- **Fast** — Matches or beats LightGBM training speed, significantly outperforms XGBoost
- **Pure Rust** — No C/C++ dependencies for core functionality
- **Compatible** — Load and use XGBoost and LightGBM models
- **Accurate** — Achieves equal or better model quality on standard benchmarks

## Quick Start

### Python

```python
import boosters
import numpy as np

# Create dataset
X = np.random.randn(1000, 10).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

# Train a GBDT model
model = boosters.GBDTModel.train(
    boosters.Dataset(X, y),
    config=boosters.GBDTConfig(
        n_trees=100,
        objective=boosters.Objective.logistic(),
    ),
)

# Predict
predictions = model.predict(X)
```

Or use the sklearn-compatible API:

```python
from boosters.sklearn import GBDTClassifier
from sklearn.model_selection import cross_val_score

model = GBDTClassifier(n_trees=100)
scores = cross_val_score(model, X, y, cv=5)
```

### Rust

```rust
use boosters::{GBDTModel, GBDTConfig, Objective, Metric};
use boosters::data::Dataset;
use ndarray::array;

// Create dataset with feature-major data [n_features, n_samples]
let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let targets = array![[0.5, 1.5, 2.5]];
let dataset = Dataset::new(features.view(), Some(targets.view()), None);

// Configure and train
let config = GBDTConfig::builder()
    .n_trees(100)
    .objective(Objective::squared())
    .metric(Metric::rmse())
    .build()?;

let model = GBDTModel::train(&dataset, &[], config, 0)?;

// Predict
let predictions = model.predict(&dataset, 0);
```

### Loading Models

```rust
use boosters::persist::Model;
use boosters::data::Dataset;

// Load from native .bstr.json format
let model = Model::load_json("model.bstr.json")?;

// Or load from native binary format (.bstr)
let model = Model::load("model.bstr")?;

// Predict using the loaded model
let gbdt = model.into_gbdt().expect("expected GBDT model");
let predictions = gbdt.predict(&dataset);
```

To convert XGBoost or LightGBM models, use the Python utilities:

```python
from boosters.convert import xgboost_to_json_bytes

# Convert from XGBoost JSON to native .bstr.json
json_bytes = xgboost_to_json_bytes("xgb_model.json")
with open("model.bstr.json", "wb") as f:
    f.write(json_bytes)
```

## Performance

GBDT benchmarks with equivalent parameters (histogram-based, 50 trees, depth 6, 256 bins).  
See [full benchmark reports](docs/benchmarks/) for details.

### Training Speed

| Dataset | boosters | XGBoost | LightGBM |
|---------|----------|---------|----------|
| Small (5k×100) | 314ms | 553ms | **245ms** |
| Medium (50k×100) | **1.39s** | 2.13s | 1.49s |

On medium datasets, **boosters is 1.5x faster than XGBoost** and matches LightGBM performance.

### Prediction Speed (batch 1K rows)

| Model | boosters | LightGBM |
|-------|----------|----------|
| Medium (50 trees, 100 features) | **0.88ms** | 4.14ms |
| Large (200 trees, 100 features) | **5.66ms** | 29.27ms |

**boosters is 4-5x faster** than LightGBM for batch prediction.

### Model Quality

Equal or better across regression, binary, and multiclass tasks — with boosters achieving **10-25% better logloss** on multiclass classification.

## Features

### Implemented

- **Tree-based Boosting (GBDT)**
  - Histogram-based training with 256 bins
  - Depth-wise and leaf-wise tree growth
  - GOSS sampling (gradient-based one-side sampling)
  - Row and column subsampling
  - Native categorical feature support
  
- **Linear Boosting (GBLinear)**
  - Coordinate descent training
  - Multiple feature selectors (cyclic, shuffle, greedy, thrifty)
  - L1/L2 regularization

- **Objectives**
  - Regression: squared loss, absolute loss, Huber, quantile, Poisson
  - Binary: logistic, hinge
  - Multiclass: softmax
  - Ranking: pairwise, LambdaRank
  
- **Metrics**
  - Regression: RMSE, MAE, MAPE
  - Classification: accuracy, log-loss, AUC
  - Ranking: NDCG, MAP

- **Model Serialization**
  - Native binary format (.bstr) with compression
  - Native JSON format (.bstr.json)
  - Python utilities to convert from XGBoost/LightGBM
  - DART booster support
  - Sample weights

- **Data I/O**
  - Arrow IPC and Parquet loading (feature-gated)

## Documentation

📚 **[Full Documentation](https://egordm.github.io/booste-rs/)** — Tutorials, API reference, and guides

- [Getting Started](https://egordm.github.io/booste-rs/getting-started/) — Installation and quickstart
- [Tutorials](https://egordm.github.io/booste-rs/tutorials/) — Step-by-step Jupyter notebooks
- [API Reference](https://egordm.github.io/booste-rs/api/) — Python and Rust API documentation
- [Explanations](https://egordm.github.io/booste-rs/explanations/) — Theory and hyperparameter guides

### For Rust Users

See the [Rust API documentation](https://egordm.github.io/booste-rs/rustdoc/boosters/) for detailed Rust API reference.

### Design Documents

- [Roadmap](docs/ROADMAP.md) — Current status and future plans
- [RFCs](docs/rfcs/) — Design documents for all major components
- [Benchmarks](docs/benchmarks/) — Performance and quality reports
- [Research](docs/research/) — Implementation analysis of XGBoost/LightGBM internals

## Project Status

boosters is functional for both training and inference with Python bindings available:

- API may change without notice (pre-1.0)
- Some advanced features (monotonic constraints) are not yet implemented
- Python bindings available via `pip install boosters` (coming soon)

See the [roadmap](docs/ROADMAP.md) for planned features.

## Development Workflow

- Run the full workspace checks (required before committing): `uv run poe all --check`
- Enable the repo hooks to enforce this automatically:
  - `git config core.hooksPath .githooks`

## License

MIT
