# Benchmark Results: M3.8 Performance Validation

**Date**: 2024-11-29  
**Milestone**: M3.8 Performance Validation  
**Platform**: Apple M1 Pro (10-core: 8P+2E), 32GB RAM, macOS 15.1  
**Rust**: 1.83.0  
**XGBoost**: 2.1.3 (via `xgb` crate)

## Goal

Comprehensive performance validation after completing all M3.x optimizations.
Answer: Does booste-rs meet production performance requirements? How does it
compare to XGBoost C++ across all scenarios?

## Summary

booste-rs achieves **significant performance advantages** over XGBoost C++ across all benchmarked scenarios:

| Scenario | booste-rs | XGBoost | Speedup |
|----------|-----------|---------|---------|
| Single row prediction | 1.24µs | 11.6µs | **9.4x faster** |
| 100 rows batch | 108µs | 150µs | **1.4x faster** |
| 1,000 rows batch | 1.07ms | 1.38ms | **1.29x faster** |
| 10,000 rows batch | 10.7ms | 13.8ms | **1.29x faster** |
| 8-thread parallel (10K) | 1.58ms | 5.0ms | **3.17x faster** |

**Key findings**:

1. **Single-row latency**: 9.4x faster than XGBoost C++ (1.24µs vs 11.6µs)
2. **Batch prediction**: 29% faster with UnrolledTraversal
3. **Thread scaling**: 6.9x speedup with 8 threads, 3.17x faster than XGBoost at same thread count
4. **UnrolledTraversal**: 3.3x speedup over StandardTraversal for large batches

## Detailed Results

### 1. Batch Size Scaling

Using `bench_medium` model (100 trees, depth 6, 50 features):

| Batch Size | Time | Throughput | Notes |
|------------|------|------------|-------|
| 1 | 1.23µs | 812K elem/s | Single-row optimized |
| 10 | 12.96µs | 771K elem/s | Small batch |
| 100 | 335µs | 298K elem/s | Medium batch |
| 1,000 | 3.61ms | 277K elem/s | Standard batch |
| 10,000 | 36.3ms | 276K elem/s | Large batch |

**Note**: `Model::predict()` uses StandardTraversal by default for compatibility.
Use `Predictor::<UnrolledTraversal6>` directly for 3x better batch performance.

### 2. Model Size Comparison

All with 1,000 rows:

| Model | Trees | Features | Time | Throughput |
|-------|-------|----------|------|------------|
| Small | 10 | 5 | 221µs | 4.5M elem/s |
| Medium | 100 | 50 | 3.6ms | 277K elem/s |
| Large | 500 | 100 | 23.0ms | 43K elem/s |

**Scaling**: Near-linear with tree count. Large model (500 trees) is ~6.4x slower than
medium (100 trees), consistent with 5x more trees.

### 3. Traversal Strategy Comparison

| Strategy | Blocking | 1K rows | 10K rows | vs Standard |
|----------|----------|---------|----------|-------------|
| Standard | None | 3.59ms | 35.2ms | baseline |
| Standard | Block-64 | 3.60ms | 36.1ms | 0.97x (slight overhead) |
| Unrolled | None | 1.61ms | 13.1ms | 2.7x faster |
| Unrolled | Block-64 | **1.08ms** | **10.8ms** | **3.3x faster** |

**Key insights**:

- Blocking alone doesn't help StandardTraversal (slight overhead)
- UnrolledTraversal + Block-64 is optimal: **3.3x faster** than Standard
- Level-by-level processing keeps tree levels in cache

### 4. Thread Scaling (Parallel Prediction)

Using UnrolledTraversal with 10K rows:

| Threads | Time | Speedup | Efficiency |
|---------|------|---------|------------|
| Sequential | 10.94ms | 1.00x | — |
| 1 | 10.99ms | 1.00x | 100% |
| 2 | 5.68ms | 1.93x | 96% |
| 4 | 2.94ms | 3.72x | 93% |
| 8 | 1.60ms | **6.88x** | 86% |

**Parallel efficiency**: 86% at 8 threads is excellent, considering M1 Pro has
8 performance + 2 efficiency cores. Memory bandwidth limits prevent perfect scaling.

### 5. XGBoost Comparison

#### 5.1 Single-Row Prediction

| Implementation | Time | Relative |
|----------------|------|----------|
| booste-rs (StandardTraversal) | 1.24µs | baseline |
| XGBoost C++ | 11.6µs | 9.4x slower |

**Analysis**: XGBoost's single-row overhead comes from:

- DMatrix creation (~8µs overhead)
- Safety checks and padding
- Less optimized for latency

#### 5.2 Batch Prediction Comparison

| Batch Size | booste-rs | XGBoost | Speedup |
|------------|-----------|---------|---------|
| 100 | 108µs | 150µs | 1.39x |
| 1,000 | 1.07ms | 1.38ms | 1.29x |
| 10,000 | 10.73ms | 13.80ms | 1.29x |

**Note**: Both using default settings (XGBoost with all threads, booste-rs with
UnrolledTraversal sequential). booste-rs maintains consistent 29% advantage.

#### 5.3 Thread Scaling Comparison

With controlled thread counts (10K rows):

| Threads | booste-rs | XGBoost | Relative |
|---------|-----------|---------|----------|
| 1 | 10.91ms | 13.77ms | 1.26x faster |
| 2 | 5.66ms | 8.71ms | 1.54x faster |
| 4 | 2.94ms | 6.14ms | 2.09x faster |
| 8 | 1.58ms | 5.00ms | **3.17x faster** |

**Analysis**: XGBoost scales poorly on M1 Pro (only 2.75x at 8 threads vs our 6.9x).
Likely causes:

- OpenMP may not optimize for Apple Silicon's hybrid architecture
- Higher thread synchronization overhead
- Different cache behavior

### 6. Summary Table

| Metric | booste-rs | XGBoost | Winner |
|--------|-----------|---------|--------|
| Single-row latency | 1.24µs | 11.6µs | booste-rs **9.4x** |
| 100-row batch | 108µs | 150µs | booste-rs **1.4x** |
| 1K-row batch | 1.07ms | 1.38ms | booste-rs **1.3x** |
| 10K-row batch | 10.7ms | 13.8ms | booste-rs **1.3x** |
| 8-thread parallel | 1.58ms | 5.00ms | booste-rs **3.2x** |
| Thread scaling (8T) | 6.9x | 2.75x | booste-rs **2.5x better** |

## Recommendations

### For Users

1. **Single-row prediction**: Use `StandardTraversal` (lowest latency)
2. **Small batches (<100)**: Use `StandardTraversal` (setup cost matters)
3. **Large batches (100+)**: Use `UnrolledTraversal6` for 3x speedup
4. **Multi-core systems**: Use `par_predict()` for additional 4-7x scaling

### Configuration Examples

```rust
use booste_rs::inference::{Predictor, StandardTraversal, UnrolledTraversal6};

// Single-row / small batch (lowest latency)
let predictor = Predictor::<StandardTraversal>::new(&forest);

// Large batch (highest throughput)
let predictor = Predictor::<UnrolledTraversal6>::new(&forest);

// Large batch + multi-core
let output = predictor.par_predict(&matrix);
```

## Platform Notes

These benchmarks were run on **Apple M1 Pro (macOS)**. Expect different absolute
numbers on other platforms:

- **x86-64 (Intel/AMD)**: Similar relative performance, may see better SIMD gains
- **Linux**: Typically slightly faster due to lower kernel overhead
- **High-core-count CPUs**: Better thread scaling (tested up to 8 threads here)

Previous benchmarks on **AMD Ryzen 9 9950X (16-core, Linux)** showed similar
relative performance with better absolute throughput due to higher clock speeds.

## Conclusions

1. ✅ **Production-ready performance**: Consistently faster than XGBoost C++
2. ✅ **Excellent single-row latency**: 9.4x faster, ideal for real-time inference
3. ✅ **Good batch scaling**: 3.3x speedup with UnrolledTraversal
4. ✅ **Excellent thread scaling**: 6.9x with 8 threads, 86% efficiency
5. ✅ **Cross-platform**: Works on both Apple Silicon and x86-64

## Next Steps

- [ ] Test on high-core-count systems (16+ cores)
- [ ] Profile for remaining optimization opportunities
- [ ] Investigate SIMD with column-major layout (future work)
- [ ] Add sparse matrix support (future milestone)
