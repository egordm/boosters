# boosters Documentation

Pure Rust gradient boosting library with XGBoost compatibility.

## Contents

| Section | Description |
|---------|-------------|
| [ROADMAP.md](./ROADMAP.md) | Feature status and future plans |
| [rfcs/](./rfcs/) | Design documents (14 RFCs) |
| [research/](./research/) | Implementation research |
| [benchmarks/](./benchmarks/) | Performance and quality reports |

## Features

### Implemented

- **Training**: GBDT (histogram-based), GBLinear
- **Inference**: Tree and linear model prediction
- **Objectives**: Binary/multiclass, regression, ranking, quantile
- **Metrics**: AUC, RMSE, MAE, log-loss, NDCG, and more
- **Sampling**: GOSS, row/column sampling
- **Sample Weights**: Weighted training, class imbalance
- **Categorical**: Native categorical feature support
- **Compatibility**: Load XGBoost JSON models, LightGBM inference
- **I/O**: Arrow and Parquet data loading

### Future Work

- Monotonic/interaction constraints
- Sparse data support
- Python bindings
- Exclusive feature bundling
- LightGBM model loading
