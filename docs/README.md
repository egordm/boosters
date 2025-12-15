# booste-rs Documentation

Pure Rust gradient boosting library with XGBoost compatibility.

## Contents

| Section | Description |
|---------|-------------|
| [PERFORMANCE_COMPARISON.md](./PERFORMANCE_COMPARISON.md) | Benchmark results vs LightGBM and XGBoost |
| [ROADMAP.md](./ROADMAP.md) | Feature status and future plans |
| [design/](./design/) | Architecture, RFCs, and design decisions |
| [benchmarks/](./benchmarks/) | Historical performance benchmark data |

## Performance Summary

**December 2025** on Apple Silicon (single-threaded, 100 trees):

### Training Performance

| Comparison | Result |
|------------|--------|
| vs LightGBM (medium dataset) | ~1% difference (parity) |
| vs XGBoost (warm start) | **13% faster** |
| vs XGBoost (cold start) | **28% faster** |

### Inference Performance

| Metric | booste-rs | XGBoost C++ | Speedup |
|--------|-----------|-------------|---------|
| Single-row | 1.24 µs | 11.6 µs | **9.4x** |
| 10K batch (8 threads) | 1.58 ms | 5.00 ms | **3.2x** |

## Features

### Implemented

- **Training**: GBDT (histogram-based), GBLinear
- **Inference**: Tree and linear model prediction
- **Objectives**: Binary/multiclass classification, regression, ranking, quantile
- **Metrics**: AUC, RMSE, MAE, log-loss, and more
- **Sampling**: GOSS, row/column sampling, class weights
- **Compatibility**: Load XGBoost JSON models

### Future Work

- Sparse data support
- Python bindings
- LightGBM model loading
