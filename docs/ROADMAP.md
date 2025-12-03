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
│  GBLinear Support                       ◄── ACTIVE              │
│  ════════════════════════════════════════════════               │
│  Core training complete, working on feature parity:             │
│  Story 7: Fix Multiclass Training       [ ] HIGH                │
│  Story 8: Quantile Regression           [ ] MEDIUM              │
│  Story 9: Additional Loss Functions     [ ] LOW                 │
│  Story 10: Additional Feature Selectors [ ] LOW                 │
│                                                                  │
│  Future (backlog)                                                │
│  ════════════════                                                │
│  - Sparse data, LightGBM, GBTree training, Python bindings      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Epics

| Epic | Status | Summary |
|------|--------|---------|
| [GBTree Inference](backlog/01-gbtree-inference.md) | ✅ Complete | Tree inference, 3x faster than XGBoost C++ |
| [GBLinear](backlog/02-gblinear.md) | 🔄 Active | Linear booster, working on feature parity |
| [Future](backlog/99-future.md) | 📋 Backlog | Sparse data, LightGBM, bindings, etc. |

---

## Performance Summary

Current benchmark results (Apple M1 Pro, vs XGBoost C++):

| Metric | booste-rs | XGBoost C++ | Speedup |
|--------|-----------|-------------|---------|
| Single-row latency | 1.24µs | 11.6µs | **9.4x** |
| 10K batch (8 threads) | 1.58ms | 5.0ms | **3.2x** |

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
