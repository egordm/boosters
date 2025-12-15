# booste-rs Roadmap

## Current Status

booste-rs has achieved **performance parity with LightGBM** and is **13-28% faster than XGBoost** while maintaining full quality compatibility.

### Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| **Tree Inference** | ✅ Complete | 3-9x faster than XGBoost C++ |
| **Tree Training** | ✅ Complete | Histogram-based, on par with LightGBM |
| **Linear Booster** | ✅ Complete | GBLinear training and inference |
| **XGBoost Compat** | ✅ Complete | Load JSON models, prediction parity |
| **Objectives** | ✅ Complete | Binary, multiclass, regression, ranking, quantile |
| **Metrics** | ✅ Complete | AUC, RMSE, MAE, log-loss, NDCG, etc. |
| **Sampling** | ✅ Complete | GOSS, row/column sampling |
| **Sample Weights** | ✅ Complete | Weighted training, class imbalance |
| **Categorical** | ✅ Complete | Native categorical feature support |
| **LightGBM Compat** | 🚧 Partial | Inference complete, model loading planned |
| **Constraints** | 📋 Planned | Monotonic/interaction constraints |
| **Sparse Data** | 📋 Planned | Sparse matrix support |
| **Python Bindings** | 📋 Planned | PyO3 bindings |

## Design Documents

See [design/rfcs/](./design/rfcs/) for detailed specifications of each component.

### Core RFCs (Implemented)

| RFC | Topic |
|-----|-------|
| 0001 | Forest data structures |
| 0002 | Tree data structures |
| 0003 | Visitor and traversal |
| 0004 | DMatrix input |
| 0007 | XGBoost serialization |
| 0008 | GBLinear inference |
| 0009 | GBLinear training |
| 0010 | Matrix layouts |
| 0011 | Quantization and binning |
| 0012 | Histogram building |
| 0013 | Split finding |
| 0014 | Row partitioning |
| 0015 | Tree growing |
| 0016 | Categorical training |
| 0017 | Sampling strategies |
| 0026 | Sample weighting |
| 0028 | Prediction outputs |

### Future RFCs (Draft/Planned)

| RFC | Topic | Status |
|-----|-------|--------|
| 0023 | Monotonic constraints | Delayed |
| 0025 | Row-parallel histograms | Draft |
| 0027 | Gradient quantization | Draft |
| 0029 | Arrow datasets | Draft |

## Development Philosophy

- **Slice-wise implementation**: Build working features end-to-end, then expand
- **Test against reference**: Every feature validated against XGBoost/LightGBM
- **Don't over-engineer early**: Get it working, then optimize
- **Document as you go**: RFCs updated when implementation diverges

## Test Data Generation

Reference predictions from Python XGBoost:

```bash
cd tools/data_generation && uv run python scripts/generate_test_cases.py
```
