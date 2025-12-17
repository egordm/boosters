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
| **LightGBM Compat** | ✅ Complete | Load text models, inference |
| **Objectives** | ✅ Complete | Binary, multiclass, regression, ranking, quantile |
| **Metrics** | ✅ Complete | AUC, RMSE, MAE, log-loss, NDCG, etc. |
| **Sampling** | ✅ Complete | GOSS, row/column sampling |
| **Sample Weights** | ✅ Complete | Weighted training, class imbalance |
| **Categorical** | ✅ Complete | Native categorical feature support |
| **Arrow/Parquet** | ✅ Complete | Data loading (may deprecate after Python bindings) |

### Future Work

| Feature | Priority | Description |
|---------|----------|-------------|
| **Python Bindings** | High | PyO3 bindings with NumPy/Pandas zero-copy |
| **Monotonic Constraints** | High | Enforce monotonic feature relationships |
| **Interaction Constraints** | Medium | Limit which features can interact |
| **Sparse Data** | Medium | CSR/CSC matrix support |
| **Linear Trees** | Medium | LightGBM-style linear models in leaves |
| **Explainability** | Medium | SHAP values, feature importance |
| **Feature Bundling (EFB)** | Medium | Bundle mutually exclusive features |
| **Per-Feature Binning** | Low | Custom max_bins per feature |
| **GPU Acceleration** | Low | CUDA/Metal/WebGPU support |
| **SIMD Inference** | Low | Vectorized tree traversal |

## Design Documents

See [rfcs/](./rfcs/) for 14 RFCs covering all major components.

## Known Improvements

These are opportunities noted during development:

1. **Batch prediction in training**: Use optimized inference pipeline for evaluation instead of per-row prediction
2. **Thread pool control**: Add `num_threads` parameter to `par_predict`
3. **Max bins parameter**: Allow users to specify `max_bins` globally and per-feature

## Development Philosophy

- **Slice-wise implementation**: Build working features end-to-end, then expand
- **Test against reference**: Every feature validated against XGBoost/LightGBM
- **Document as you go**: RFCs updated when implementation diverges
