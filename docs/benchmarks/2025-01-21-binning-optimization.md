# Binning Optimization and Fair Benchmark Comparison

**Date**: 2025-01-21  
**Commit**: 14750ea + refactor  
**Machine**: Apple Silicon (M-series)

## Summary

This report documents the investigation into training performance and the resulting optimizations to make benchmarks fair and transparent.

## Background

During performance investigation, we discovered that:

1. Quantile binning (introduced for 5% quality improvement) adds overhead to preprocessing
2. The benchmark was not using consistent threading settings between binning and training
3. Data layout conversions were happening inside the benchmark loop for boosters but not for LightGBM

## Changes Made

### 1. Simplified BinnedDatasetBuilder API

The builder now has two constructor patterns:
- **Simple**: `from_matrix(&data, max_bins)` - uses all threads and default config
- **Full control**: `from_matrix_with_options(&data, config, parallelism)` - explicit threading and config

```rust
use boosters::{Parallelism, data::{BinnedDatasetBuilder, BinningConfig}};

// Simple: parallel with defaults
let binned = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build()?;

// Full control: single-threaded for benchmarks
let config = BinningConfig::new(256)
    .with_sample_cnt(200_000);  // LightGBM-style sampling
let binned = BinnedDatasetBuilder::from_matrix_with_options(
    &col_matrix,
    config,
    Parallelism::Sequential,
).build()?;
```

When `Parallelism::Sequential`, we use pure iteration without touching rayon,
avoiding any thread pool overhead.

### 2. Configurable Sample Count

The `bin_construct_sample_cnt` is now configurable in `BinningConfig`:

```rust
// Default: 200K samples (matches LightGBM)
let config = BinningConfig::new(256);

// Custom sample count
let config = BinningConfig::new(256)
    .with_sample_cnt(100_000);  // Lower for faster binning
```

### 3. Parallelism Moved to Utils

`Parallelism` enum is now in `crate::utils` and re-exported at crate root.
Components receive `Parallelism` and self-correct based on workload size.

## Training Performance Results

**Configuration**: 50 trees, depth 6, 256 bins, single-threaded

| Dataset | boosters | LightGBM | Ratio |
|---------|----------|----------|-------|
| Small (5K×50) | **237ms** | 221ms | 1.07x slower |
| Medium (50K×100) | **1.63s** | 1.63s | **parity** |

**Analysis**: boosters achieves parity with LightGBM on the medium dataset. The small dataset overhead is likely fixed-cost overhead that amortizes on larger data.

## Quality Results

**Configuration**: 50 trees, depth 6, 3 seeds averaged

### Regression (RMSE, lower is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|----------|---------|----------|
| Small (2K×50) | **1.409** | 1.417 | 1.417 |
| Medium (10K×100) | **2.276** | 2.296 | 2.301 |

### Binary Classification (Accuracy, higher is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|----------|---------|----------|
| Small (2K×50) | 0.801 | **0.803** | 0.801 |
| Medium (10K×100) | **0.804** | 0.795 | 0.792 |

### Multiclass Classification (Accuracy, higher is better)

| Dataset | boosters | XGBoost | LightGBM |
|---------|----------|---------|----------|
| Small (2K×50) | **0.624** | 0.585 | 0.609 |
| Medium (10K×100) | **0.650** | 0.608 | 0.626 |

## Key Findings

1. **Performance parity**: boosters matches LightGBM within measurement noise on medium datasets
2. **Quality advantage**: boosters achieves better or equal quality on regression and multiclass tasks
3. **Quantile binning trade-off**: Adds ~5-10% preprocessing cost but improves quality ~5%
4. **Fair comparison**: With proper benchmark configuration, libraries are within expected variance

## Files Changed

- [crates/boosters/src/data/binned/builder.rs](../../crates/boosters/src/data/binned/builder.rs):
  - Added `bin_construct_sample_cnt` to `BinningConfig` with default 200K
  - Simplified to two constructors: `from_matrix()` and `from_matrix_with_options()`
  - Accepts `Parallelism` instead of raw `n_threads`
  - When `Parallelism::Sequential`, uses pure iteration (no rayon)
- [crates/boosters/src/utils.rs](../../crates/boosters/src/utils.rs): Added `Parallelism` enum
- [crates/boosters/src/training/gbdt/parallelism.rs](../../crates/boosters/src/training/gbdt/parallelism.rs): Re-exports `Parallelism` from utils
- [crates/boosters/src/data/binned/mod.rs](../../crates/boosters/src/data/binned/mod.rs): Export `DEFAULT_BIN_CONSTRUCT_SAMPLE_CNT`
- [crates/boosters/benches/suites/compare/gbdt_training.rs](../../crates/boosters/benches/suites/compare/gbdt_training.rs): Use new `from_matrix_with_options()` API

## Recommendations

1. Use quantile binning (default) for best quality
2. For single-threaded benchmarks, use `Parallelism::Sequential` for binning
3. For production, use `Parallelism::from_threads(0)` for parallel binning
4. Thread pool management is at model level; components receive `Parallelism` hints
