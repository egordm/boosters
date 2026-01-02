# 🚀 boosters

A fast, pure-Rust gradient boosting library for training and inference.

> **Work in Progress**: This library is under active development but already functional for training and inference with XGBoost/LightGBM compatibility.

## What is boosters?

boosters is a gradient boosting implementation written from scratch in Rust, designed to be:

- **Fast** — Matches or beats LightGBM training speed, significantly outperforms XGBoost
- **Pure Rust** — No C/C++ dependencies for core functionality
- **Compatible** — Load and use XGBoost and LightGBM models
- **Accurate** — Achieves equal or better model quality on standard benchmarks

## Quick Start

### Training a Model

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

### Loading XGBoost Models

```rust
use boosters::compat::xgboost::XgbModel;
use boosters::inference::gbdt::{Predictor, StandardTraversal};
use boosters::data::Dataset;

// Load and convert
let model = XgbModel::from_file("model.json")?;
let forest = model.to_forest()?;

// Predict
let predictor = Predictor::<StandardTraversal>::new(&forest);
let predictions = predictor.predict(&dataset);
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

- **Model Compatibility**
  - Load XGBoost JSON models
  - Load LightGBM text models
  - DART booster support
  - Sample weights

- **Data I/O**
  - Arrow IPC and Parquet loading (feature-gated)

## Documentation

- [Roadmap](docs/ROADMAP.md) — Current status and future plans
- [RFCs](docs/rfcs/) — Design documents for all major components
- [Benchmarks](docs/benchmarks/) — Performance and quality reports
- [Research](docs/research/) — Implementation analysis of XGBoost/LightGBM internals

## Project Status

boosters is functional for both training and inference but not yet production-ready:

- API may change without notice
- Some advanced features (monotonic constraints, SHAP) are not yet implemented
- Python bindings are planned but not yet available

## Development Workflow

- Run the full workspace checks (required before committing): `uv run poe all --check`
- Enable the repo hooks to enforce this automatically:
  - `git config core.hooksPath .githooks`

## License

MIT
