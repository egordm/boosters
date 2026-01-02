# boosters Roadmap

## Current Status

boosters has achieved **performance parity with LightGBM** and is **13-28% faster than XGBoost** while maintaining full quality compatibility.

### Implemented Features

| Feature | Notes |
| ------- | ----- |
| **Tree Inference** | 3-9x faster than XGBoost C++ |
| **Tree Training** | Histogram-based, depth/leaf-wise growth |
| **Linear Booster** | GBLinear training and inference |
| **XGBoost Compat** | Load JSON models, prediction parity |
| **LightGBM Compat** | Load text models, inference |
| **Objectives** | Binary, multiclass, regression, ranking, quantile |
| **Metrics** | AUC, RMSE, MAE, log-loss, NDCG, etc. |
| **Sampling** | GOSS, row/column sampling |
| **Sample Weights** | Weighted training, class imbalance |
| **Categorical** | Native categorical feature support |
| **Feature Bundling (EFB)** | 84-98% memory reduction for one-hot data |
| **Per-Feature Binning** | Custom max_bins per feature via BinningConfig |
| **Linear Trees** | Linear models at tree leaves |
| **Python Bindings** | PyO3 bindings with NumPy/Pandas zero-copy, sklearn estimators |
| **Explainability** | Feature importance (gain/split count), TreeSHAP, Linear SHAP |
| **Evaluation Framework** | `boosters-eval` for quality benchmarks and regression testing |

---

## Planned Features

Ordered by priority. Not yet assigned to releases.

### High Priority

- **Model Serialization**: Stable on-disk format for boosters models (save/load without XGBoost/LightGBM)
- **Monotonic Constraints**: Enforce monotonic feature relationships during training
- **GPU Acceleration**: CUDA/Metal histogram building

### Medium Priority

- **Interaction Constraints**: Limit which features can interact in splits
- **Training Diagnostics**: Optional verbose logging: per-tree metrics, gradient stats

### Low Priority

- **Natural Gradient Boosting**: NGBoost-style probabilistic predictions
- **Quantized Histograms**: Integer histogram accumulation for memory-bound hardware

---

## Small Open Tasks

- [ ] Python examples for explainability (SHAP values, feature importance)
- [ ] Python bindings to inspect tree structure (nodes, thresholds, splits)
- [ ] Training diagnostics logging (optional verbose output: per-tree metrics, gradient stats)
- [ ] Batch prediction in training (use optimized inference for evaluation)

---

## Design Documents

See [rfcs/](./rfcs/) for 15 RFCs covering all major components.

## Development Philosophy

- **Slice-wise implementation**: Build working features end-to-end, then expand
- **Test against reference**: Every feature validated against XGBoost/LightGBM
- **Document as you go**: RFCs updated when implementation diverges
