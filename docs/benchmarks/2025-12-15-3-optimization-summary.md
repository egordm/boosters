# Optimization Summary: December 15, 2025

## Implemented Optimizations

### 1. Sequential Histogram Builder Fast Path (commit `87fe637`)

**Impact:** 5.6-8.3% training speedup on medium datasets

When a leaf's row indices are strictly sequential (e.g., `[start, start+1, ..., start+len-1]`),
we skip streaming the indices array entirely and directly access gradients/bins by offset.

This benefits:

- Root nodes (always sequential at training start)
- Early tree nodes before much splitting has occurred
- Sampled subsets that happen to be contiguous

### 2. Single-Pass Stable Partition (commit `79a24f1`)

**Impact:** ~2.7-2.8% improvement on thread scaling benchmarks

Replaced the 2-pass count-then-write partition with a single-pass approach:

- Write left indices forward from `begin`
- Write right indices backward from `end`
- Reverse right portion to restore stable order

This eliminates one full pass over the indices array per split.

### 3. Root Gradient Sum from Histogram (commit `6eb9e4c`)

**Impact:** Small (within noise), but cleaner algorithm

Instead of scanning all samples to compute gradient sums before building
the root histogram, build the histogram first and derive the sums by
summing histogram bins.

Changes O(n_samples) to O(n_bins) for computing root gradient sums.

### 4. Skip Backward Split Scan When No Missing Values

**Impact:** large when the dataset has no missing values (common for our synthetic benches)

LightGBM only runs the “missing goes left” (backward) split scan when the feature actually has missing values.
When there are no missing values, the backward scan can’t produce a different best split than the forward scan,
but it still costs a full extra histogram sweep per feature.

We restored this behavior in our numerical split finder by guarding the backward scan behind `has_missing`.

**Benchmark evidence (Criterion `training_gbdt`, Apple Silicon):**

- Guarded (`has_missing`): `train_binary` was ~2.06s
- Forced backward scan always-on: `train_binary` was ~2.47s (≈ +20%)

Similar regressions showed up across regression/multiclass and thread-scaling benchmarks (≈ +9% to +37%),
confirming that skipping the backward scan when it cannot matter is a meaningful win.

## Notes: Histogram Layout Microbenchmark

We previously carried an isolated microbenchmark (`histogram_layout`) comparing gradient/hessian layouts.
It was useful for a quick signal, but it’s easy to overfit to it. We now rely on end-to-end `training_gbdt`
and `e2e_train_predict_gbdt` as primary sources of truth.

**Findings (before removal):** interleaving `(grad, hess)` as `(f32, f32)` was consistently faster than
two separate `grad: &[f32]` / `hess: &[f32]` arrays in that microbenchmark:

- 1k rows: ~802ns (separate) vs ~636ns (interleaved f32)  (≈ 21% faster)
- 10k rows: ~7.46µs (separate) vs ~5.39µs (interleaved f32) (≈ 28% faster)
- 50k rows: ~36.95µs (separate) vs ~28.86µs (interleaved f32) (≈ 22% faster)

We keep these notes as context, but prioritize measuring any real implementation changes using the training
benches above.

## Current Performance vs LightGBM

| Size | boosters | LightGBM | Ratio |
|------|----------|----------|-------|
| Small (50k×10) | 460ms | 239ms | 1.93× slower |
| Medium (50k×100) | 2.18s | 1.47s | 1.48× slower |

## Remaining Performance Gap Analysis

Based on research into LightGBM and XGBoost source code:

### LightGBM Advantages

1. **4-bit bin packing**: For features with ≤16 bins, LightGBM packs two bins per byte,
   reducing memory bandwidth by 50% for those features.

2. **Exclusive Feature Bundling (EFB)**: Mutually exclusive sparse features are bundled
   together, reducing effective feature count.

3. **ParallelPartitionRunner**: Two-phase parallel partitioning that scales better on
   many cores.

4. **Template-based branch elimination**: Uses C++ templates to eliminate runtime
   branches for common configurations (hessian constant, missing value handling).

5. **SIMD histogram accumulation**: Uses vectorized instructions for histogram building.

### XGBoost Patterns We Could Adopt

1. **Multi-target shared partitioning**: For multiclass problems, compute row indices
   once and reuse across all K class trees.

2. **Compressed bin index storage**: Use u8 when bins ≤256, with offset subtraction
   for low memory access.

3. **Cache-adaptive row/column iteration**: Switch between row-wise and column-wise
   histogram building based on L2 cache size.

## Recommended Future Optimizations

### High Priority (Large Expected Impact)

1. **4-bit bin packing** (~5-10%)
   - For features with ≤16 bins, pack two per byte
   - Reduces memory bandwidth significantly

2. **Template/const-generic dispatch** (~5-8%)
   - Eliminate runtime branches for common cases:
     - Has missing values: yes/no
     - Default left: yes/no
     - Hessian is constant: yes/no

3. **Parallel partition** (~10-15% on many cores)
   - Two-phase approach: count left/right per thread, then parallel write
   - Requires careful synchronization

### Medium Priority

1. **Software prefetching for non-contiguous access**
   - Only for random access patterns (deep tree nodes)
   - Prefetch bin data 10-32 elements ahead

2. **SIMD histogram accumulation**
   - Use NEON on ARM, AVX2 on x86
   - Requires careful alignment handling

### Low Priority (Specialized Cases)

1. **Exclusive Feature Bundling**
   - Only helps for sparse datasets with mutually exclusive features

2. **Multi-target shared partitioning**
   - Only helps for multiclass (K > 2) problems

## Notes on ARM64 (Apple M1 Pro, 10 cores)

The current optimizations were benchmarked on Apple M1 Pro, 10 cores. Some observations:

- ARM's hardware prefetcher is very effective for sequential access
- Software prefetching showed minimal benefit in our tests
- Memory bandwidth is less of a bottleneck compared to x86
- L1/L2 cache sizes are generous (192KB L1, shared L2)

For best cross-platform performance, focus on algorithmic improvements over
micro-optimizations.
