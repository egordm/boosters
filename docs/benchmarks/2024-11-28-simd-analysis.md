# SIMD Analysis and XGBoost Comparison

**Date**: 2024-11-28  
**Milestone**: M3.6 SIMD Investigation  
**Author**: Benchmark analysis

## Executive Summary

Our SIMD implementation using the `wide` crate is **11% slower** than the UnrolledTraversal,
not faster as expected. Investigation reveals this is **not a bug** but a fundamental
limitation of the row-parallel SIMD approach with row-major data layout.

**Key finding**: XGBoost C++ also does **NOT use SIMD** for tree prediction. They rely
on the same optimizations we have: array tree layout + block processing + thread parallelism.

**Final comparison**: With single-threaded XGBoost (`nthread=1`, `predictor=cpu_predictor`):

| Batch Size | booste-rs | XGBoost | Result |
|------------|-----------|---------|--------|
| 100 rows | 72.0 µs | 94.1 µs | **booste-rs 31% faster** |
| 1,000 rows | 706 µs | 754 µs | **booste-rs 7% faster** |
| 10,000 rows | 7.06 ms | 7.43 ms | **booste-rs 5% faster** |
| Single row | 5.88 µs | 20.5 µs | **booste-rs 3.5x faster** |

booste-rs now **outperforms XGBoost C++** on single-threaded inference!

## Benchmark Results

### All Strategy/Blocking Combinations (100 trees, 50 features, depth 6)

| Strategy | 1K No-Block | 1K Block64 | 10K No-Block | 10K Block64 |
|----------|-------------|------------|--------------|-------------|
| Standard | 2.13ms | 2.14ms | 21.1ms | 21.4ms |
| Unrolled | 793µs | **752µs** | 10.2ms | **7.59ms** |
| SIMD | 845µs | 842µs | 8.72ms | 8.41ms |

### Performance Analysis

| Metric | Standard | Unrolled | SIMD |
|--------|----------|----------|------|
| Best time (10K) | 21.1ms | 7.59ms | 8.41ms |
| vs Standard | baseline | **2.78x faster** | 2.51x faster |
| vs Unrolled | 2.78x slower | baseline | **11% slower** |

### Key Observations

1. **Blocking only helps Unrolled**: Standard sees -1.6% (overhead), SIMD sees +3.7%
2. **Unrolled+Block64 wins**: 7.59ms is our best result
3. **SIMD adds overhead**: The scalar gather loops negate SIMD comparison benefits

## Why SIMD Doesn't Help

### The Fundamental Problem: Gather Overhead

Our SIMD implementation processes 8 rows in parallel, but at each tree level we must:

```rust
// This loop runs 8 times per level - kills SIMD benefit
for lane in 0..SIMD_WIDTH {
    let row_idx = row_start + lane;
    let feat_idx = split_indices[lane] as usize;  // Different feature per row!
    let row_offset = row_idx * num_features;
    fvalues[lane] = features.get(row_offset + feat_idx).copied().unwrap_or(f32::NAN);
}
```

**Problem**: Each of the 8 rows may need a **different feature index**. With row-major
layout `features[row][feature]`, gathering these 8 scattered values requires 8 separate
memory accesses.

### What Would Fix This

**Hardware gather instructions** like AVX2's `vpgatherdd` can load 8 non-contiguous
floats in parallel. But:

1. `wide` crate doesn't expose gather intrinsics
2. Rust's `std::simd` (nightly) has gather but requires nightly compiler
3. Even with gather, the random access pattern has poor cache behavior

### Alternative SIMD Strategies (Future Work)

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Column-major features** | Store `features[feature][row]` | Vectorized loads | Requires data transpose |
| **Tree-parallel SIMD** | Same row through 8 trees | Same feature access | Complex accumulation |
| **Leaf accumulation SIMD** | Vectorize output accumulation | Simple | Small win |
| **Native AVX2 intrinsics** | Use `_mm256_i32gather_ps` | Hardware gather | Unsafe, arch-specific |

## XGBoost C++ Analysis

### Key Finding: XGBoost Does NOT Use SIMD for Prediction

After analyzing XGBoost's source code, we found:

| Technique | XGBoost | booste-rs |
|-----------|---------|-----------|
| Explicit SIMD (AVX/SSE) | ❌ No | ❌ No (ineffective) |
| Array tree layout | ✅ Yes (6 levels) | ✅ Yes (configurable) |
| Block processing | ✅ Yes (64 rows) | ✅ Yes (64 rows) |
| OpenMP threading | ✅ Yes | ❌ Not yet |
| Template specialization | ✅ Yes | ✅ Yes (const generics) |

### XGBoost's Optimization Stack

1. **Array Tree Layout** (`array_tree_layout.h`)
   - Unrolls top 6 levels into contiguous arrays
   - Eliminates pointer-chasing
   - Cache-friendly sequential access

2. **Block Processing** (`cpu_predictor.cc`)
   - 64-row blocks for cache efficiency
   - Thread-local feature vectors

3. **OpenMP Parallelism**
   - Parallel block processing across threads
   - This is their main scalability win

### Why XGBoost Doesn't Use SIMD Either

From the XGBoost codebase analysis:

> "Gather bottleneck: SIMD requires gathering feature values across rows (same feature
> index, different rows). Row-major layout: Features are stored per-row, but SIMD needs
> column values. Irregular tree access: Different rows may need different features at
> each level."

**This is exactly the same problem we hit.**

## Recommendations

### Short-term: Remove SIMD Overhead

Keep the `simd` feature flag but document it as experimental. The current implementation
adds overhead without benefit.

### Medium-term: Add Rayon Parallelism

XGBoost's main advantage is OpenMP threading. We should add rayon-based parallelism:

```rust
// Future: parallel block processing
features.par_chunks(block_size).for_each(|block| {
    // Process block
});
```

### Long-term: Alternative SIMD Approaches

If we want effective SIMD, consider:

1. **Column-major DMatrix** - Allow transposed storage for SIMD-friendly access
2. **Tree-parallel SIMD** - Process same row through multiple trees simultaneously
3. **Nightly std::simd** - Use portable gather when available

## Conclusion

Our SIMD implementation is **correct but ineffective** due to the fundamental mismatch
between row-parallel SIMD and row-major data layout. This is a known limitation that
even XGBoost doesn't attempt to solve with SIMD.

**Best current strategy**: `UnrolledTraversal` with block size 64, achieving 2.78x
speedup over standard traversal.

**Next optimization target**: Thread parallelism (like XGBoost's OpenMP) would provide
the largest gains for multi-core systems.

## References

- XGBoost `src/predictor/cpu_predictor.cc` - Block-based prediction
- XGBoost `include/xgboost/tree_model.h` - Array tree layout
- [wide crate documentation](https://docs.rs/wide/)
- [Rust std::simd RFC](https://github.com/rust-lang/rust/issues/86656)

---

## Appendix: SIMD Crate Comparison

### What `wide` Crate Provides

| Feature | Available | Notes |
|---------|-----------|-------|
| f32x8 vector type | ✅ Yes | Basic 8-wide float |
| Arithmetic (+, -, *, /) | ✅ Yes | Vectorized |
| Comparisons (lt, gt, eq) | ✅ Yes | Returns mask |
| Blend/select | ✅ Yes | Mask-based selection |
| Gather (indexed load) | ❌ No | **Critical missing feature** |
| Scatter (indexed store) | ❌ No | Not available |
| Horizontal operations | ✅ Partial | Sum, min, max |

### What `std::simd` (Nightly) Provides

| Feature | Available | Notes |
|---------|-----------|-------|
| f32x8 vector type | ✅ Yes | `Simd<f32, 8>` |
| Arithmetic | ✅ Yes | Full support |
| Comparisons | ✅ Yes | Better mask handling |
| Gather | ✅ Yes | `Simd::gather_or` |
| Scatter | ✅ Yes | `Simd::scatter` |
| Portable across targets | ✅ Yes | Falls back gracefully |

### What We Actually Need for SIMD Tree Traversal

```rust
// IDEAL: What we want to write
let feature_indices: i32x8 = [...];  // 8 different feature indices
let row_offsets: i32x8 = [...];      // 8 different row offsets
let addresses = row_offsets * num_features + feature_indices;

// Hardware gather: load 8 scattered floats in ONE instruction
let fvalues: f32x8 = features.gather(addresses);  // ← NOT AVAILABLE IN wide

// Compare all 8 against thresholds
let go_left = fvalues.cmp_lt(thresholds);
```

### What We Actually Have to Write

```rust
// ACTUAL: Scalar loop defeats SIMD purpose
let mut fvalues = [0.0f32; 8];
for lane in 0..8 {
    let row_idx = row_start + lane;
    let feat_idx = split_indices[lane] as usize;
    let row_offset = row_idx * num_features;
    fvalues[lane] = features[row_offset + feat_idx];  // ← 8 separate loads!
}
let fvalues_simd = f32x8::from(fvalues);
let go_left = fvalues_simd.cmp_lt(thresholds_simd);  // ← This part IS SIMD
```

### AVX2 Gather Intrinsic (What We'd Need)

```c
// C intrinsic for hardware gather
__m256 _mm256_i32gather_ps(float const* base, __m256i vindex, int scale);

// Loads 8 floats from: base[vindex[0]], base[vindex[1]], ..., base[vindex[7]]
// ALL IN ONE INSTRUCTION (~7 cycles vs ~40 for 8 scalar loads)
```

### Recommendations for Effective SIMD

1. **Short-term**: Don't use SIMD for row-parallel traversal
2. **Medium-term**: Focus on thread parallelism (rayon) instead of SIMD
3. **Long-term**: Alternative approaches that don't need gather:
   - **Tree-parallel**: Same row through 8 trees (same feature indices!)
   - **Column-major data**: Transpose for contiguous access
   - **Quantized features**: Pack 8 features per SIMD lane

---

## Appendix B: Nightly std::simd Gather Experiment

### Hypothesis

We hypothesized that `std::simd`'s `Simd::gather_or` would provide hardware gather
instructions (AVX2 `vpgatherdd`) and improve SIMD performance.

### Experiment

Implemented `NightlySimdTraversal` using `std::simd` portable SIMD with:
- `Simd::gather_or` for feature value gathering
- Full SIMD comparison and position updates
- Requires `#![feature(portable_simd)]` on nightly Rust

### Results

| Strategy | 10K rows (no AVX2) | 10K rows (AVX2 native) |
|----------|-------------------|------------------------|
| Unrolled+Block64 | 7.56ms | 7.57ms |
| wide SIMD | 8.33ms | ~8.3ms |
| **Nightly SIMD (gather)** | **27.5ms** | **14.5ms** |

### Analysis

1. **Without AVX2**: `std::simd` falls back to scalar gather emulation (3.6x slower)
2. **With AVX2** (`-C target-cpu=native`): Uses hardware gather but still **1.9x slower**

**Key finding**: Even with hardware gather, row-parallel SIMD doesn't help because:

- AVX2 gather (`vpgatherdd`) has ~10-20 cycle latency per 8 elements
- Random memory access patterns defeat cache prefetching
- The overhead of SIMD setup/teardown per level exceeds scalar loop cost
- Tree traversal is **latency-bound**, not **throughput-bound**

### Conclusion

**Hardware gather doesn't help**. The fundamental problem is that tree traversal
requires random memory access based on data-dependent decisions at each level.
This is an inherently serial workload that SIMD cannot effectively parallelize.

**Recommendation**: Remove `simd-nightly` feature and focus on thread parallelism
for scaling. The `wide` crate SIMD is retained as experimental baseline.
