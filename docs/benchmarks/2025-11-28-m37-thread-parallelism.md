# Benchmark Results: M3.7 Thread Parallelism

**Date**: 2024-11-28  
**Milestone**: M3.7 Thread Parallelism (Rayon)  
**Platform**: Apple M1 Pro (10-core, 8P+2E), 32GB RAM, macOS 15.1  
**Rust**: 1.83.0

## Goal

Add and validate thread parallelism via Rayon for multi-core scaling.
Hypothesis: Parallel prediction should scale near-linearly with thread count.
Also: compare our thread scaling against XGBoost's OpenMP parallelism.

## Summary

M3.7 adds thread parallelism via Rayon for multi-core prediction scaling. Key findings:

1. **Excellent thread scaling** — 6.8x speedup with 8 threads on 10-core M1 Pro
2. **Better than XGBoost** — 3.18x faster at 8 threads (1.59ms vs 5.06ms)
3. **Proper thread pool management** — Use `build() + install()`, not `build_global()`
4. **Simplified parallel path** — Removed branching, always use block-optimized traversal

## Thread Scaling Performance

Using `bench_medium` model (100 trees, 50 features), 10K rows:

| Threads | Time | Throughput | Speedup | Efficiency |
|---------|------|------------|---------|------------|
| Sequential | 10.95ms | 913K elem/s | 1.00x | — |
| 1 thread | 11.23ms | 890K elem/s | 0.98x | 98% |
| 2 threads | 5.85ms | 1.71M elem/s | 1.87x | 94% |
| 4 threads | 3.00ms | 3.33M elem/s | 3.65x | 91% |
| 8 threads | 1.61ms | 6.23M elem/s | **6.80x** | 85% |

**Parallel efficiency**: 85% at 8 threads is excellent, considering:

- M1 Pro has 8 performance cores + 2 efficiency cores
- Memory bandwidth limits prevent perfect linear scaling
- Theoretical max speedup ~8x, achieved 6.8x (85% efficiency)

## Comparison vs XGBoost

Both tested with controlled thread counts (10K rows):

| Threads | booste-rs | XGBoost | Speedup | Relative |
|---------|-----------|---------|---------|----------|
| 1 | 10.92ms | 13.81ms | 1.26x | +26% faster |
| 2 | 5.62ms | 8.70ms | 1.55x | +55% faster |
| 4 | 2.90ms | 6.15ms | 2.12x | +112% faster |
| 8 | 1.59ms | 5.06ms | **3.18x** | +218% faster |

**Key observations**:

- booste-rs is faster at **all** thread counts
- Gap **widens** with more threads (XGBoost scales poorly on M1 Pro)
- XGBoost achieves only ~2.7x speedup at 8 threads vs our 6.8x

### XGBoost Scaling Analysis

XGBoost's poor scaling on M1 Pro (2.7x at 8 threads) likely due to:

1. OpenMP may not optimize well for Apple Silicon's hybrid architecture
2. C++ implementation may have thread synchronization overhead
3. DMatrix creation overhead in benchmarks (though we tried to minimize this)

## Implementation Details

### Thread Pool Management

**Initial approach (broken)**:

```rust
rayon::ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .build_global()  // Can only be called once!
    .ok();           // Silently ignores errors

predictor.par_predict(matrix)  // Uses default pool
```

**Fixed approach**:

```rust
let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .build()
    .unwrap();

pool.install(|| predictor.par_predict(matrix))
```

**Lesson**: Global rayon pool can only be initialized once per process. Subsequent `build_global()` calls fail silently, causing all benchmarks to use the same thread count.

### Code Simplification

Removed branching in `process_block_parallel`:

**Before**: Two paths based on `USES_BLOCK_OPTIMIZATION`

- Block-optimized path for `UnrolledTraversal` (calls `traverse_block`)
- Per-row path for `StandardTraversal` (calls `traverse_tree`)

**After**: Single path always using block-optimized traversal

- `UnrolledTraversal`: optimized batch processing via `traverse_block`
- `StandardTraversal`: falls back to default `traverse_block` (per-row loop)

**Rationale**: Parallel prediction is for large batches where `UnrolledTraversal` excels. Users should use sequential `predict()` with `StandardTraversal` for small batches anyway.

**Result**: ~30 lines removed, simpler maintenance, no performance impact.

## Platform Notes

These benchmarks were run on **macOS (Apple M1 Pro)**. Previous benchmarks were on **Linux (AMD Ryzen 9 9950X)**.

Key differences:

- M1 Pro: 10 cores (8P+2E), unified memory, 3.2GHz base
- Ryzen 9950X: 16 cores (Zen 5), 192KB L1 cache, up to 5.7GHz boost

Expect different absolute numbers but similar relative scaling patterns.

## Architecture Changes

### Modified Files

- `src/predict/predictor.rs`:
  - Added `par_predict()` and `par_predict_weighted()` methods
  - Added `par_predict_internal()` helper
  - Simplified `process_block_parallel()` to remove branching
  - Added 4 comprehensive tests

- `benches/prediction.rs`:
  - Added `bench_thread_scaling()` for booste-rs only
  - Added `bench_thread_scaling_xgboost()` for fair comparison
  - Fixed thread pool usage with `build() + install()`

- `src/predict/traversal.rs`:
  - Added `Send + Sync` bound to `TreeState`

- `src/trees/unrolled_layout.rs`:
  - Added `Send + Sync` to `NodeArray` and `ExitArray` types
  - Added `Send + Sync` to generic `T` parameters

## Conclusions

1. **Thread parallelism works excellently** — 6.8x speedup with 8 threads
2. **Outperforms XGBoost** — 3x faster at 8 threads, gap widens with scale
3. **Production-ready** — Clean API, well-tested, documented edge cases
4. **Simple to use** — `par_predict()` drop-in replacement for `predict()`

## Next Steps

M3.8 Performance Validation will:

- Re-run comprehensive benchmarks with all optimizations
- Test on Linux (Ryzen 9950X) for higher core counts
- Profile to identify any remaining bottlenecks
- Update README with final performance numbers
