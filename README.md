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
use boosters::model::{GBDTModel, gbdt::GBDTConfig};
use boosters::data::{Dataset, RowMatrix};
use boosters::training::{Objective, Metric};

// Prepare data
let features = RowMatrix::from_vec(data, n_samples, n_features);
let labels: Vec<f32> = /* your targets */;

// Create dataset with binned features for histogram-based training
let dataset = Dataset::from_parts(features, labels.clone(), None);

// Configure and train
let config = GBDTConfig::builder()
    .objective(Objective::squared_error())
    .metric(Metric::rmse())
    .n_trees(100)
    .learning_rate(0.1)
    .build()?;

let model = GBDTModel::train(&dataset, config)?;

// Predict
let test_features = RowMatrix::from_vec(test_data, n_test, n_features);
let predictions = model.predict(&test_features);  // ColMatrix<f32>
```

### Loading XGBoost Models

```rust
use boosters::compat::xgboost::XgbModel;
use boosters::inference::gbdt::Predictor;
use boosters::data::RowMatrix;

// Load XGBoost JSON model
let model = XgbModel::from_file("model.json")?;
let forest = model.to_forest()?;

// Use Predictor for efficient batch prediction
let features = RowMatrix::from_vec(data, n_rows, n_features);
let predictor = Predictor::new(&forest);
let predictions = predictor.predict(&features);  // Batch prediction
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

## License

MIT
