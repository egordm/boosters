# boosters Documentation

Pure Rust gradient boosting library with XGBoost compatibility.

## Contents

| Section | Description |
|---------|-------------|
| [ROADMAP.md](./ROADMAP.md) | Feature status and future plans |
| [rfcs/](./rfcs/) | Design documents (15 RFCs) |
| [research/](./research/) | Implementation research |
| [benchmarks/](./benchmarks/) | Performance and quality reports |

## Features

### Implemented

- **Training**: GBDT (histogram-based), GBLinear, linear leaves
- **Inference**: Tree and linear model prediction
- **Objectives**: Binary/multiclass, regression, ranking, quantile
- **Metrics**: AUC, RMSE, MAE, log-loss, NDCG, and more
- **Sampling**: GOSS, row/column sampling
- **Sample Weights**: Weighted training, class imbalance
- **Categorical**: Native categorical feature support
- **Compatibility**: Load XGBoost JSON models, LightGBM text models
- **I/O**: Arrow and Parquet data loading
- **Feature Bundling (EFB)**: Exclusive feature bundling for one-hot data
- **Python Bindings**: PyO3 bindings with sklearn estimators
- **Explainability**: Feature importance, TreeSHAP, Linear SHAP

### Future Work

- Model serialization (stable on-disk format)
- Monotonic/interaction constraints
- GPU acceleration
