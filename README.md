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
use boosters::data::{Dataset, BinningConfig, BinnedDatasetBuilder, TargetsView, WeightsView};
use boosters::{GBDTConfig, GBDTModel, Metric, Objective, Parallelism, TreeParams};
use ndarray::Array2;

// Prepare feature-major data [n_features, n_samples]
let features = Array2::<f32>::from_shape_vec((n_features, n_samples), data).unwrap();
let labels = Array1::<f32>::from_vec(targets);

// Create binned dataset for histogram-based training
let dataset = Dataset::new(features.view(), None, None);
let binned = BinnedDatasetBuilder::new(BinningConfig::builder().max_bins(256).build())
    .add_features(dataset.features(), Parallelism::Parallel)
    .build()?;

// Configure and train
let config = GBDTConfig::builder()
    .n_trees(100)
    .learning_rate(0.1)
    .tree(TreeParams::depth_wise(6))
    .objective(Objective::squared())
    .metric(Metric::rmse())
    .build()?;

let targets = TargetsView::new(labels.view().insert_axis(ndarray::Axis(0)));
let model = GBDTModel::train_binned(&binned, targets, WeightsView::None, &[], config, 1)?;

// Predict
let predictions = model.predict(&dataset, 1);
```

### Loading XGBoost Models

```rust
use boosters::compat::xgboost::XgbModel;
use boosters::inference::gbdt::{Predictor, StandardTraversal};
use boosters::data::Dataset;
use ndarray::Array2;

// Load XGBoost JSON model
let model = XgbModel::from_file("model.json")?;
let forest = model.to_forest()?;

// Create predictor for efficient batch prediction
let predictor = Predictor::<StandardTraversal>::new(&forest);

// Predict on feature-major data [n_features, n_samples]
let features = Array2::<f32>::from_shape_vec((n_features, n_samples), data).unwrap();
let dataset = Dataset::new(features.view(), None, None);
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

## License

MIT
