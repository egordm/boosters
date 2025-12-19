# boosters Roadmap

## Current Status

boosters has achieved **performance parity with LightGBM** and is **13-28%% faster than XGBoost** while maintaining full quality compatibility.

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
| **Feature Bundling (EFB)** | ✅ Complete | 84-98% memory reduction for one-hot data |
| **Per-Feature Binning** | ✅ Complete | Custom max_bins per feature via BinningConfig |
| **Arrow/Parquet** | ✅ Complete | Data loading (may deprecate after Python bindings) |

---

## Release 1.0.0 Roadmap

**Target**: Production-ready release with Python ecosystem integration.
**Estimated Timeline**: ~8-10 weeks

### Release Strategy

The team has agreed on the following approach after discussion:

1. **Code Audit First**: Stabilize internals before exposing Python API
2. **Python Bindings**: Core value delivery - unlocks user adoption
3. **Explainability**: Feature importance (gain/split count) + basic SHAP
4. **GPU Deferred**: Too risky for 1.0 - target for 1.1 or later

### 1.0.0 Required Features

| Feature | Priority | Status | Effort | Description |
|---------|----------|--------|--------|-------------|
| **Code Audit & Cleanup** | 🔴 P0 | Not Started | 2 weeks | Architecture review, API stabilization, test audit |
| **Python Bindings** | 🔴 P0 | Not Started | 3-4 weeks | PyO3 bindings with NumPy/Pandas zero-copy |
| **Explainability** | 🟡 P1 | Not Started | 2-3 weeks | Feature importance, TreeSHAP values |

### 1.0.0 Code Audit Scope

Before 1.0.0, a thorough audit is required:

**Architecture Review**:
- [ ] Module boundaries analysis (`src/data/`, `src/training/`, `src/inference/`)
- [ ] Public API surface audit (minimize `pub`, use `pub(crate)`)
- [ ] Coupling analysis between components
- [ ] Identify deprecated or dead code paths

**Test Coverage Audit**:
- [ ] GOSS sampling edge cases
- [ ] Multiclass objective coverage
- [ ] Quantile regression tests
- [ ] Integration test expansion
- [ ] Remove redundant tests

**API Consistency**:
- [ ] Naming conventions review
- [ ] Error handling patterns (no silent failures)
- [ ] Documentation coverage (all public items have examples)
- [ ] Panic safety audit (intentional vs accidental unwraps)

**Performance Review**:
- [ ] Identify unnecessary allocations
- [ ] Profile hot paths
- [ ] Validate benchmark results are reproducible

### 1.1.0+ Planned Features

| Feature | Priority | Description |
|---------|----------|-------------|
| **GPU Acceleration** | High | CUDA/Metal histogram building (deferred from 1.0) |
| **Monotonic Constraints** | High | Enforce monotonic feature relationships |
| **Interaction Constraints** | Medium | Limit which features can interact |
| **Sparse Data** | Medium | CSR/CSC matrix support |
| **Linear Trees** | Medium | LightGBM-style linear models in leaves |
| **SIMD Inference** | Medium | Vectorized tree traversal |
| **Natural Gradient Boosting** | Low | NGBoost-style probabilistic boosting |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Python binding complexity | Medium | High | Use PyO3 best practices, test on CI early |
| SHAP implementation bugs | Medium | Medium | Validate against reference SHAP library |
| API churn after 1.0 | High | High | Complete code audit before Python release |
| GPU delays 1.0 | High | High | **Deferred to 1.1** |

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
