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
│  GBLinear Support                       ✅ COMPLETE             │
│  ════════════════════════════════════════════════               │
│  Story 1: GBLinear Inference            ✓                       │
│  Story 2: Training Infrastructure       ✓                       │
│  Story 3: GBLinear Training             ✓                       │
│  Story 4: Matrix Layout Refactor        ✓                       │
│  Story 5: Training Validation           ✓                       │
│  Story 6: Benchmarks & Optimization     ✓                       │
│                                                                  │
│  Future (backlog)                       ◄── NEXT                │
│  ════════════════                                                │
│  - Sparse data, LightGBM, GBTree training, Python bindings      │
│  - SIMD/BLAS optimizations for linear training                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Epics

| Epic | Status | Summary |
|------|--------|---------|
| [GBTree Inference](backlog/01-gbtree-inference.md) | ✅ Complete | Tree inference, 3x faster than XGBoost C++ |
| [GBLinear](backlog/02-gblinear.md) | ✅ Complete | Linear booster inference + training |
| [Future](backlog/99-future.md) | 📋 Backlog | Sparse data, LightGBM, bindings, etc. |

---

## Performance Summary

Current benchmark results (Apple M1 Pro, vs XGBoost C++):

| Metric | booste-rs | XGBoost C++ | Speedup |
|--------|-----------|-------------|---------|
| Single-row latency | 1.24µs | 11.6µs | **9.4x** |
| 10K batch (8 threads) | 1.58ms | 5.0ms | **3.2x** |

See [benchmarks](../benchmarks/) for details.

---

## RFCs

| RFC | Status | Topic |
|-----|--------|-------|
| [0001](rfcs/0001-forest-data-structures.md) | Implemented | Forest structures |
| [0002](rfcs/0002-tree-data-structures.md) | Implemented | Tree structures |
| [0003](rfcs/0003-visitor-and-traversal.md) | Implemented | Traversal & prediction |
| [0004](rfcs/0004-dmatrix.md) | Implemented | Data input |
| [0007](rfcs/0007-serialization.md) | Implemented | XGBoost loading |
| [0008](rfcs/0008-gblinear-inference.md) | Implemented | Linear inference |
| [0009](rfcs/0009-gblinear-training.md) | Implemented | Linear training |
| [0010](rfcs/0010-matrix-layouts.md) | Implemented | Matrix layouts |

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
