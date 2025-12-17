# booste-rs Roadmap

## Current Status

booste-rs has achieved **performance parity with LightGBM** and is **13-28%% faster than XGBoost** while maintaining full quality compatibility.

### Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| **Tree Inference** | ✅ Complete | 3-9x faster than XGBoost C++ |
| **Tree Training** | ✅ Complete | Histogram-based, depth/leaf-wise |
| **Linear Booster** | ✅ Complete | GBLinear training and inference |
| **XGBoost Compat** | ✅ Complete | Load JSON models, prediction parity |
| **LightGBM Compat** | ✅ Complete | Inference from loaded models |
| **Objectives** | ✅ Complete | Binary, multiclass, regression, ranking, quantile |
| **Metrics** | ✅ Complete | AUC, RMSE, MAE, log-loss, NDCG, etc. |
| **Sampling** | ✅ Complete | GOSS, row/column sampling |
| **Sample Weights** | ✅ Complete | Weighted training, class imbalance |
| **Categorical** | ✅ Complete | Native categorical feature support |
| **Arrow/Parquet** | ✅ Complete | Data loading from Arrow/Parquet |

### Future Work

| Feature | Priority | Notes |
|---------|----------|-------|
| **Monotonic Constraints** | High | Enforce monotonic relationships |
| **Interaction Constraints** | Medium | Limit feature interactions |
| **Sparse Data** | Medium | CSR/CSC matrix support |
| **Feature Bundling (EFB)** | Medium | Bundle mutually exclusive features |
| **Python Bindings** | High | PyO3-based bindings |
| **LightGBM Model Loading** | Low | Load LightGBM model files |

## Design Documents

See [rfcs/](./rfcs/) for 14 RFCs covering all major components.

## Development Philosophy

- **Slice-wise implementation**: Build working features end-to-end, then expand
- **Test against reference**: Every feature validated against XGBoost/LightGBM
- **Document as you go**: RFCs updated when implementation diverges
