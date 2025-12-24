# Baseline Performance Report: Pre-Migration (4154b94)

**Date**: 2025-12-24
**Commit**: 4154b94
**Context**: Baseline before Epic 3 (Algorithm Integration) - new Dataset module exists but not yet integrated

## Purpose

Capture baseline performance before migrating GBDT and GBLinear to use the new `dataset::Dataset` type. This allows us to verify that the migration does not introduce performance regressions.

## System

- macOS (M-series)
- Rust 1.87.0
- Release build

## Prediction Benchmarks

| Benchmark | Samples | Time | Throughput |
|-----------|---------|------|------------|
| model_size/small/1000 | 1,000 | ~0.24 ms | ~4.2 Melem/s |
| model_size/medium/1000 | 1,000 | **1.18 ms** | **845 Kelem/s** |
| model_size/large/1000 | 1,000 | **5.90 ms** | **170 Kelem/s** |
| single_row/medium | 1 | 8.5 µs | - |
| traversal/standard/10000 | 10,000 | **36.9 ms** | **271 Kelem/s** |
| traversal/unrolled6/10000 | 10,000 | **12.0 ms** | **833 Kelem/s** |

## Training Benchmarks

| Benchmark | Samples × Features | Time | Throughput |
|-----------|-------------------|------|------------|
| thread_scaling/1 thread | 5,000 × 100 | ~0.7s | ~0.7 Melem/s |
| thread_scaling/8 threads | 5,000 × 100 | **440 ms** | **11.4 Melem/s** |
| growth/depthwise | 50,000 × 100 | **1.21 s** | **4.14 Melem/s** |
| growth/leafwise | 50,000 × 100 | **1.24 s** | **4.02 Melem/s** |

## Key Metrics for Comparison

After migration, verify:

1. **Prediction overhead**: Block buffering should add ≤10% overhead (target 5%)
2. **Training throughput**: Should remain within ±5% of baseline
3. **Single-row prediction**: Should remain unchanged

## Notes

- Prediction benchmarks currently use row-major data directly
- After migration, prediction will use feature-major Dataset with block transpose
- Training already uses feature-major binned data internally
