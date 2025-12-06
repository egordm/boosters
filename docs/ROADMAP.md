# booste-rs Roadmap

## Philosophy

**Slice-wise implementation**: Build thin vertical slices that work end-to-end, then expand.

**Guiding principle**: At each milestone, we should be able to load a real model and produce correct predictions.

---

## Current Focus

```text
┌─────────────────────────────────────────────────────────────────┐
│  GBTree Inference                       ✅ COMPLETE             │
│  ══════════════════════════════════════════════════             │
│  Load XGBoost JSON models, predict with 3x+ speedup vs C++      │
│                                                                  │
│  GBLinear Support                       ⏸️  PAUSED               │
│  ════════════════════════════════════════════════               │
│  Core training complete, feature parity stories pending         │
│                                                                  │
│  GBTree Training (Phase 1)              ✅ COMPLETE              │
│  ════════════════════════════════════════════════               │
│  Histogram-based tree training: 1.05x-1.57x of XGBoost          │
│  See: docs/benchmarks/2024-11-30-gbtree-vs-xgboost.md           │
│                                                                  │
│  LightGBM Compatibility                 ✅ INFERENCE COMPLETE    │
│  ════════════════════════════════════════════════               │
│  Model loading, inference validation (Stories 1-5 done)         │
│  Training baselines pending (Stories 6-7)                       │
│                                                                  │
│  GBTree Training (Phase 2)              📋 PLANNED              │
│  ════════════════════════════════════════════════               │
│  Optimizations: histogram subtraction, SIMD, parallelism        │
│  (Requires new RFCs)                                            │
│                                                                  │
│  Future (backlog)                                                │
│  ════════════════                                                │
│  - Sparse data, Python bindings                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Epics

| Epic | Status | Summary |
|------|--------|---------|
| [GBTree Inference](backlog/01-gbtree-inference.md) | ✅ Complete | Tree inference, 3x faster than XGBoost C++ |
| [GBLinear](backlog/02-gblinear.md) | ⏸️ Paused | Linear booster, core training complete |
| [GBTree Training Ph1](backlog/03-gbtree-training.md) | ✅ Complete | Histogram-based training, 1.05-1.57x of XGBoost |
| [LightGBM Compat](backlog/04-lightgbm-compat.md) | ✅ Inference | Model loading complete, training baselines pending |
| [GBTree Training Ph2](backlog/03-gbtree-training.md) | 📋 Planned | Optimizations (needs RFCs) |
| [Future](backlog/99-future.md) | 📋 Backlog | Sparse data, Python bindings, etc. |

---

## Performance Summary

### Inference (vs XGBoost C++)

| Metric | booste-rs | XGBoost C++ | Speedup |
|--------|-----------|-------------|---------|
| Single-row latency | 1.24µs | 11.6µs | **9.4x** |
| 10K batch (8 threads) | 1.58ms | 5.0ms | **3.2x** |

### Training (vs XGBoost C++, single-threaded)

| Dataset | booste-rs | XGBoost | Ratio |
|---------|-----------|---------|-------|
| small (1k×20, 100 trees) | 322ms | 308ms | 1.05x |
| medium (5k×50, 100 trees) | 1.32s | 1.11s | 1.19x |
| large (20k×100, 100 trees) | 5.12s | 3.27s | 1.57x |

See [benchmarks](benchmarks/) for details.

---

## RFCs

| RFC | Status | Topic |
|-----|--------|-------|
| [0001](design/rfcs/0001-forest-data-structures.md) | Implemented | Forest structures |
| [0002](design/rfcs/0002-tree-data-structures.md) | Implemented | Tree structures |
| [0003](design/rfcs/0003-visitor-and-traversal.md) | Implemented | Traversal & prediction |
| [0004](design/rfcs/0004-dmatrix.md) | Implemented | Data input |
| [0007](design/rfcs/0007-serialization.md) | Implemented | XGBoost loading |
| [0008](design/rfcs/0008-gblinear-inference.md) | Implemented | Linear inference |
| [0009](design/rfcs/0009-gblinear-training.md) | Implemented | Linear training |
| [0010](design/rfcs/0010-matrix-layouts.md) | Implemented | Matrix layouts |
| [0011](design/rfcs/0011-quantization-binning.md) | Implemented | Quantization & binning |
| [0012](design/rfcs/0012-histogram-building.md) | Implemented | Histogram building |
| [0013](design/rfcs/0013-split-finding.md) | Implemented | Split finding |
| [0014](design/rfcs/0014-row-partitioning.md) | Implemented | Row partitioning |
| [0015](design/rfcs/0015-tree-growing.md) | Implemented | Tree growing strategies |

---

## Test Data

Reference predictions generated from Python XGBoost:

```bash
cd tools/data_generation && uv run python scripts/generate_test_cases.py
```

---

## Notes

- **Don't over-engineer early**: Get something working, then refactor
- **Test against Python**: Every story should have validation tests
- **Feature flags**: Keep optional stuff behind features
- **Document as you go**: Update RFCs if implementation diverges
