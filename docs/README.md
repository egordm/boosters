# booste-rs Documentation

Pure Rust gradient boosting library with XGBoost compatibility.

## Contents

| Section | Description |
|---------|-------------|
| [benchmarks/2025-12-14-benchmark-report.md](./benchmarks/2025-12-14-benchmark-report.md) | Latest consolidated benchmarks (training + inference + quality) |
| [ROADMAP.md](./ROADMAP.md) | Feature status and future plans |
| [design/](./design/) | Architecture, RFCs, and design decisions |
| [benchmarks/](./benchmarks/) | Historical performance benchmark data |

## Performance + Quality

See [benchmarks/2025-12-14-benchmark-report.md](./benchmarks/2025-12-14-benchmark-report.md) for the current performance comparisons and quality tables.

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
