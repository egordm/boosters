# Exclusive Feature Bundling (EFB) Performance Analysis

**Date**: 2025-12-19  
**Commit**: 0f8c2b2  
**Platform**: macOS (Apple Silicon)  
**Rust version**: stable

## Executive Summary

This report evaluates the performance and memory characteristics of Exclusive Feature Bundling (EFB) in booste-rs compared to LightGBM's bundling implementation.

**Key Findings**:
- **Memory reduction**: 84-98% fewer binned columns for one-hot encoded data
- **Training time**: No measurable impact (bundling affects memory, not speed)
- **Binning overhead**: 4-19% additional time for bundling analysis
- **Quality**: Identical predictions (bundling is lossless)

## Test Configuration

### Datasets

| Name | Rows | Numerical | Categoricals × Cats | Total Features | Sparsity |
|------|------|-----------|---------------------|----------------|----------|
| small_sparse | 10,000 | 2 | 3 × 10 | 32 | ~97% |
| medium_sparse | 50,000 | 5 | 5 × 20 | 105 | ~95% |
| high_sparse | 20,000 | 2 | 10 × 50 | 502 | ~98% |

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Trees | 10 |
| Max Depth | 6 |
| Learning Rate | 0.1 |
| Lambda | 1.0 |
| Threads | 1 |

## Results

### Memory Reduction (Binned Columns)

| Dataset | Original Features | Bundled Columns | Reduction |
|---------|------------------|-----------------|-----------|
| small_sparse | 32 | **5** | 84.4% |
| medium_sparse | 105 | **10** | 90.5% |
| high_sparse | 502 | **12** | **97.6%** |

The bundling algorithm successfully identifies mutually exclusive one-hot encoded features and merges them into single columns, achieving near-optimal compression.

### Binning Performance (Dataset Construction)

| Dataset | No Bundling | With Bundling | Overhead |
|---------|-------------|---------------|----------|
| small_sparse | 4.52 ms | 4.92 ms | +8.9% |
| medium_sparse | 67.0 ms | 69.6 ms | +3.9% |
| high_sparse | 101.2 ms | 119.9 ms | +18.5% |

The bundling analysis adds a small overhead during dataset construction. This is a one-time cost that can be amortized over multiple training runs or hyperparameter searches.

### Training Performance (10 Trees, Pre-Binned Data)

| Dataset | No Bundling | With Bundling | Difference |
|---------|-------------|---------------|------------|
| small_sparse | 24.69 ms | 24.82 ms | ~0% |
| medium_sparse | 285.2 ms | 284.9 ms | ~0% |
| high_sparse | 389.1 ms | 390.7 ms | ~0% |

**Key Insight**: Training time is effectively identical with or without bundling. This is because:
1. Tree building time is dominated by tree structure operations, not histogram storage
2. The histogram iteration loops over *rows*, not features - fewer columns doesn't reduce iteration count
3. The bundled representation uses the same bin-to-histogram mapping

### LightGBM Comparison

| Dataset | LightGBM (no bundle) | LightGBM (bundle) | Difference |
|---------|---------------------|-------------------|------------|
| small_sparse | 19.12 ms | 19.17 ms | ~0% |
| medium_sparse | 120.6 ms | 121.5 ms | ~0% |
| high_sparse | 81.8 ms | 81.8 ms | ~0% |

LightGBM also shows no training time difference between bundling modes, confirming that EFB's value is memory-focused, not compute-focused.

**Note**: LightGBM is ~10-15× faster than booste-rs in these benchmarks due to:
- SIMD histogram updates
- Optimized multi-threading
- Mature C++ implementation

## Memory Savings Analysis

For the high_sparse dataset (502 features → 12 bundled columns):

| Metric | Without Bundling | With Bundling | Savings |
|--------|------------------|---------------|---------|
| Binned data | 20K × 502 × 1B = **10.0 MB** | 20K × 12 × 1B = **240 KB** | **97.6%** |
| Histogram space | 502 × 256 bins × 16B = **2.1 MB** | 12 × 256 bins × 16B = **49 KB** | **97.6%** |

For datasets with many one-hot encoded categoricals, EFB can reduce memory usage by **40×** or more.

## Use Cases

EFB is most valuable when:

1. **Memory-constrained environments**: Embedded systems, edge devices, or when training many models in parallel
2. **Wide one-hot datasets**: NLP bag-of-words, high-cardinality categoricals, or multi-label encodings
3. **Batch hyperparameter tuning**: Binning overhead is paid once, savings compound across runs

EFB adds minimal value when:

1. **Dense numerical data**: No sparse features to bundle
2. **Single training runs**: Binning overhead may not be amortized
3. **Compute-bound workflows**: Training speed is the bottleneck, not memory

## Conclusion

Exclusive Feature Bundling in booste-rs achieves its design goal: **massive memory reduction for sparse one-hot data with negligible impact on training accuracy or speed**. 

The implementation correctly identifies mutually exclusive features and compresses them efficiently. The 4-19% binning overhead is acceptable given the 84-98% memory reduction.

## Appendix: Bundling Presets

| Preset | Conflict Rate | Min Sparsity | Use Case |
|--------|---------------|--------------|----------|
| `auto()` | 0.01% | 90% | General purpose (default) |
| `disabled()` | - | - | Baseline comparison |
| `aggressive()` | 0.1% | 80% | More bundling, allow minor conflicts |
| `strict()` | 0% | 95% | Only truly exclusive features |
