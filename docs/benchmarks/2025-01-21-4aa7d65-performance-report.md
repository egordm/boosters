# Performance Benchmark Report

**Commit**: 4aa7d65  
**Date**: 2025-01-21  
**Focus**: API refactoring - WeightsView and Dataset::weights()

## Summary

This benchmark compares booste-rs against XGBoost and LightGBM after the WeightsView API refactoring. The changes include:

- `Dataset::weights()` now returns `WeightsView<'_>` directly instead of `Option<ArrayView1>`
- `WeightsView::from_optional()` removed - no longer needed
- `TreeView` moved to separate module

## Training Performance

Benchmarks run with 100 trees, depth 6, single-threaded.

### Small Dataset (5,000 rows × 50 features)

| Library | Time | Throughput | Speedup vs XGBoost |
|---------|------|------------|-------------------|
| **booste-rs** | 239 ms | 1.04 M samples/s | **2.2x** |
| LightGBM | 220 ms | 1.13 M samples/s | 2.4x |
| XGBoost | 526 ms | 0.47 M samples/s | 1.0x |

### Medium Dataset (50,000 rows × 100 features)

| Library | Time | Throughput | Speedup vs XGBoost |
|---------|------|------------|-------------------|
| **booste-rs** | 1.65 s | 3.02 M samples/s | **1.4x** |
| LightGBM | 1.60 s | 3.12 M samples/s | 1.4x |
| XGBoost | 2.29 s | 2.19 M samples/s | 1.0x |

## Prediction Performance

### Batch Prediction (10,000 samples, 100 trees, depth 6)

| Library | Time | Throughput | Speedup vs XGBoost |
|---------|------|------------|-------------------|
| **booste-rs** | 448 µs | **22.3 M samples/s** | **20x** |
| XGBoost | 9.1 ms | 1.1 M samples/s | 1.0x |
| LightGBM | 41.1 ms | 0.24 M samples/s | 0.2x |

### Batch Size Scaling

| Batch Size | booste-rs | XGBoost | LightGBM |
|------------|-----------|---------|----------|
| 100 | 65 µs (1.5 M/s) | 1.4 ms (71 K/s) | 376 µs (266 K/s) |
| 1,000 | 79 µs (**12.6 M/s**) | 2.1 ms (485 K/s) | 4.8 ms (210 K/s) |
| 10,000 | 448 µs (**22.3 M/s**) | 9.1 ms (1.1 M/s) | 41.1 ms (243 K/s) |

### Single Row Prediction

| Library | Time | Notes |
|---------|------|-------|
| LightGBM | 3.6 µs | **Fastest for single rows** |
| booste-rs | 59 µs | Optimized for batches |
| XGBoost | 1.3 ms | Includes DMatrix overhead |

## Key Findings

1. **Training**: booste-rs is competitive with LightGBM and ~1.4-2.2x faster than XGBoost
2. **Batch Prediction**: booste-rs is **20x faster** than XGBoost and **92x faster** than LightGBM
3. **Single Row**: LightGBM is fastest due to minimal overhead; booste-rs optimizes for batch workloads
4. **Scaling**: booste-rs throughput increases dramatically with batch size (1.5M → 22.3M samples/s)

## Configuration

- **Hardware**: Apple Silicon (M-series)
- **Rust**: 1.87 nightly
- **Features**: bench-compare (includes xgboost, lightgbm bindings)
- **Model**: 100 trees, max depth 6, regression objective
- **Threads**: Single-threaded for fair comparison
