# Data Structures in Gradient Boosting

This documentation covers the key data structures used in gradient boosting implementations, synthesizing approaches from XGBoost and LightGBM.

## Overview

Gradient boosting frameworks rely on specialized data structures optimized for different phases:

| Phase | Key Challenge | Primary Structures |
|-------|---------------|-------------------|
| **Data Preparation** | Efficient quantization | Histogram Cuts, BinMapper |
| **Training** | Fast histogram building | GHistIndexMatrix, ColumnMatrix |
| **Inference** | Low-latency prediction | SoA Tree Layout, Packed Forests |

### The Core Insight

Modern gradient boosting doesn't operate on raw floating-point features. Instead:

1. **Quantize** continuous values into discrete bins (256 bins typical)
2. **Store** bin indices instead of floats (1 byte vs 4 bytes)
3. **Compare** integer bin indices instead of float thresholds

This enables:
- 4× memory reduction
- Cache-friendly sequential access
- SIMD-friendly integer operations
- Deterministic behavior across platforms

## Table of Contents

### [Histogram Cuts](histogram-cuts.md)
How continuous features become discrete bins:
- Quantile sketch algorithms
- Cut point storage and lookup
- Per-feature bin boundaries

### [Quantized Matrix](quantized-matrix.md)
Storing quantized feature data:
- Dense vs sparse layouts
- CPU (GHistIndexMatrix) vs GPU (EllpackPage) formats
- Type-adaptive storage (u8/u16/u32)

### [Tree Storage](tree-storage.md)
How decision trees are stored in memory:
- Training trees: AoS for mutability
- Inference trees: SoA for cache efficiency
- Multi-output leaf storage

## Quick Reference

### XGBoost vs LightGBM Terminology

| Concept | XGBoost | LightGBM |
|---------|---------|----------|
| Bin boundaries | `HistogramCuts` | `BinMapper` |
| Quantized data | `GHistIndexMatrix` | `Dataset` / `Bin` |
| Tree node | `RegTree::Node` | `Tree` (SoA arrays) |
| GPU format | `EllpackPage` | CUDA Dataset |

### Memory Layout Patterns

```
AoS (Array of Structures):
┌──────────────────────┐
│ Node { feat, thresh, │ ← One node, all fields together
│        left, right } │
└──────────────────────┘

SoA (Structure of Arrays):
┌────────────┐ ┌────────────┐ ┌────────────┐
│ feat[0..n] │ │thresh[0..n]│ │ left[0..n] │ ← Each field contiguous
└────────────┘ └────────────┘ └────────────┘
```

- **AoS**: Better for training (node-at-a-time access, easy mutation)
- **SoA**: Better for inference (field-at-a-time access, SIMD-friendly)

## Design Philosophy

These data structures share common design principles:

1. **Minimize memory bandwidth** — The limiting factor on modern CPUs/GPUs
2. **Maximize cache utilization** — Sequential access patterns where possible
3. **Enable vectorization** — Contiguous data for SIMD operations
4. **Support hardware diversity** — Different layouts for CPU vs GPU
