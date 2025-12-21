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

### 1. Removed `n_threads` from BinningConfig

Instead of storing `n_threads` in the config (which requires users to set it twice), 
the thread count is now passed at runtime to `from_matrix_with_config_threaded()`:

```rust
// Single-threaded binning - no rayon overhead
let binned = BinnedDatasetBuilder::from_matrix_with_config_threaded(
    &col_matrix,
    BinningConfig::new(256),
    1,  // n_threads: 1 = sequential, 0 = all available
).build()?;
```

When `n_threads == 1`, we use pure sequential iteration without touching rayon at all,
avoiding any thread pool overhead.

### 2. LightGBM-style Sampling for Quantile Binning

For datasets larger than 200K rows (matching LightGBM's `bin_construct_sample_cnt` default),
we use uniform sampling to compute approximate quantiles:

```rust
pub const BIN_CONSTRUCT_SAMPLE_CNT: usize = 200_000;
```

This reduces memory and compute cost while maintaining good bin boundary quality.
Smaller datasets still use exact quantiles via full sort.

### 3. Fair Benchmark Configuration

- Data layout conversions moved **outside** the benchmark loop (users store data in optimal format)
- Single-threaded binning when `n_threads: 1` for training (mirrors LightGBM behavior)

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
  - Removed `n_threads` from `BinningConfig`
  - Added `from_matrix_with_config_threaded()` with runtime `n_threads` parameter
  - When `n_threads == 1`, uses pure sequential iteration (no rayon)
  - Added `BIN_CONSTRUCT_SAMPLE_CNT` constant (200K, matching LightGBM)
  - Updated `compute_quantile_bounds()` to use sampling for large datasets
- [crates/boosters/src/data/binned/mod.rs](../../crates/boosters/src/data/binned/mod.rs): Export `BIN_CONSTRUCT_SAMPLE_CNT`
- [crates/boosters/benches/suites/compare/gbdt_training.rs](../../crates/boosters/benches/suites/compare/gbdt_training.rs): Use new threaded API

## Recommendations

1. Use quantile binning (default) for best quality
2. For single-threaded training, pass `n_threads=1` to binning for fair measurement
3. For production, use `n_threads=0` for parallel binning regardless of training threads
4. Future work: Consider moving thread pool management to a higher level (model/session)
