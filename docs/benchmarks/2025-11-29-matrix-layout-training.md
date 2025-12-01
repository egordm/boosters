# Matrix Layout Training Benchmarks

**Date**: 2025-11-29  
**Commit**: Story 6 - Matrix Layout Benchmarks  
**Focus**: Compare matrix layouts for GBLinear training performance

## Summary

Benchmarks to evaluate the performance impact of different matrix formats
(ColMajor, CSC) on linear model training. The trainer now accepts any type
implementing `ColumnAccess`, supporting both dense and sparse matrices directly.

## Key Findings

### 1. Training Performance by Format

These benchmarks test actual training time with no conversion overhead.

| Dataset Size | ColMajor | CSC | CSC Advantage |
|--------------|----------|-----|---------------|
| 1,000 × 100 | 2.88 ms (34.7 Melem/s) | 2.52 ms (39.7 Melem/s) | **14% faster** |
| 10,000 × 100 | 28.9 ms (34.6 Melem/s) | 25.6 ms (39.0 Melem/s) | **13% faster** |
| 50,000 × 100 | 145 ms (34.4 Melem/s) | 125 ms (39.9 Melem/s) | **16% faster** |

**Surprising Finding**: CSC outperforms ColMajor for dense data! Despite CSC using
2× memory (storing both values and row_indices), it consistently runs ~13% faster.

**Investigation**: The core loop benchmark (`gradient_column_sum`) shows identical
performance for both formats in isolation. The training difference likely comes from:

- LLVM optimizing the CSC code path differently during monomorphization
- Better prefetching patterns due to the explicit row_indices array
- Memory alignment effects in the larger training context

This remains an open question, but the practical implication is clear: **use CSC
for training, even with dense data**.

### 2. Sequential vs Parallel Updaters

| Mode | Time (10K×100) | Throughput | Speedup |
|------|----------------|------------|---------|
| Sequential (coordinate descent) | 18.1 ms | 55.2 Melem/s | 1.0× |
| Parallel (shotgun) | 10.7 ms | 93.4 Melem/s | **1.69×** |

**Conclusion**: Parallel (shotgun) training provides ~70% speedup.

### 3. Core CD Operation Performance (gradient_column_sum)

This benchmark tests the critical inner loop of coordinate descent: iterating
over a column and accumulating gradient × value products.

| Method | Time (50K×100) | Throughput | Notes |
|--------|----------------|------------|-------|
| col_slice_direct | 4.74 ms | 1.055 Gelem/s | Direct `col_slice()` access |
| col_trait_dense | 4.74 ms | 1.054 Gelem/s | Via `ColumnAccess` trait |
| csc_trait | 4.76 ms | 1.049 Gelem/s | CSC via `ColumnAccess` trait |

**Key Finding**: The `ColumnAccess` trait has **zero overhead** - trait-based
iteration matches direct slice access within noise margin.

### 4. Conversion Overhead

| Conversion | 1K rows | 10K rows | 50K rows |
|------------|---------|----------|----------|
| RowMajor → ColMajor | 68 µs | 697 µs | 8.4 ms |
| RowMajor → CSC | 85 µs | 836 µs | 6.5 ms |

## Architecture

The training pipeline now accepts any `ColumnAccess` type directly:

```text
Input (ColMatrix or CSCMatrix)
    ↓
LinearTrainer::train() - no conversion needed
    ↓
UpdaterKind::Parallel or Sequential
    ↓
compute_weight_update() via ColumnAccess::column()
```

The `ColumnAccess` trait provides unified column iteration for:

- `ColMatrix<f32>` - dense column-major (contiguous columns)
- `CSCMatrix<f32>` - sparse column storage

## Recommendations

1. **For Dense Data**: Either `ColMatrix` or `CSCMatrix` work well. CSC shows
   ~13% faster in benchmarks, but this appears to be a compiler optimization
   artifact rather than a fundamental advantage. Both are valid choices.

2. **For Sparse Data**: Use `CSCMatrix` directly - it will skip zeros and be
   much more efficient for high sparsity.

3. **Parallel Training**: Use `UpdaterKind::Parallel` (default) for best
   performance. Sequential is only needed for exact gradient reproducibility.

4. **Conversion**: Convert from RowMajor once at the start. Conversion overhead
   is small (~5-7% of training time for 50K samples).

## Open Questions

- **CSC vs ColMajor**: CSC shows ~13% faster in full training, but micro-benchmarks
  show identical performance. Likely a compiler optimization artifact.

## Environment

- **Machine**: Apple M3 Pro
- **Rust**: 2024 edition
- **Profile**: Release (optimized)

## Raw Data

```text
training_format/col_major/1000:  2.88 ms (34.7 Melem/s)
training_format/csc/1000:        2.52 ms (39.7 Melem/s)
training_format/col_major/10000: 28.9 ms (34.6 Melem/s)
training_format/csc/10000:       25.6 ms (39.0 Melem/s)
training_format/col_major/50000: 145 ms (34.4 Melem/s)
training_format/csc/50000:       125 ms (39.9 Melem/s)

updater/sequential/10000: 18.1 ms (55.2 Melem/s)
updater/parallel/10000:   10.7 ms (93.4 Melem/s)

cd_core_operation/col_slice_direct: 4.74 ms (1.055 Gelem/s)
cd_core_operation/col_trait_dense:  4.74 ms (1.054 Gelem/s)
cd_core_operation/csc_trait:        4.76 ms (1.049 Gelem/s)

conversion_overhead/row_to_col/1000:  68 µs
conversion_overhead/row_to_csc/1000:  85 µs
conversion_overhead/row_to_col/10000: 697 µs
conversion_overhead/row_to_csc/10000: 836 µs
```
