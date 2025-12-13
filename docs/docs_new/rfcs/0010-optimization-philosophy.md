```markdown
# RFC-0010: Optimization Philosophy

- **Status**: Draft
- **Created**: 2024-12-05
- **Updated**: 2024-12-05
- **Depends on**: RFC-0008 (Feature Quantization), RFC-0009 (Histogram Building), RFC-0012 (Gradient Quantization), RFC-0007 (GBTree Training)
- **Scope**: Framework for optimization selection, user control, and platform adaptation

## Summary

This RFC establishes the **optimization philosophy** for boosters: which optimizations are always-on, which are auto-selected transparently, and which require user choice. The goal is maximum performance with minimal user burden, while never silently affecting model quality.

## Core Principle

**User controls model quality. We control execution speed.**

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     User-Controlled                                 │
│  Affects model quality or training semantics                        │
│  → User must explicitly enable                                      │
│  → We warn if choice is suboptimal                                  │
├─────────────────────────────────────────────────────────────────────┤
│                     Auto-Selected                                   │
│  Affects only execution speed, not results                          │
│  → We pick the best strategy automatically                          │
│  → Transparent to user                                              │
├─────────────────────────────────────────────────────────────────────┤
│                     Always-On                                       │
│  No downside, always beneficial                                     │
│  → Hardcoded, no configuration needed                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Optimization Categories

### Always-On (Hardcoded)

These optimizations have no downside and are always enabled:

| Optimization | RFC | Why Always-On |
|--------------|-----|---------------|
| Histogram subtraction trick | 0009 | 50% less work, exact same results |
| LRU histogram pooling | 0009 | Avoids allocation, no overhead |
| Col-major data layout | 0001 | Optimal for feature iteration |
| SoA histogram layout | 0009 | Better SIMD, same results |
| Dedicated missing bin | 0008 | Clean handling, negligible memory |
| Quantile binning | 0008 | Always better than uniform |

### Auto-Selected (Transparent)

These optimizations affect only execution speed. We detect data characteristics and select the best strategy:

| Optimization | RFC | Selection Criterion |
|--------------|-----|---------------------|
| Histogram iteration order | 0009 | Row-wise if histogram fits L2, else feature-wise |
| Histogram parallelism | 0009 | Feature-parallel or row-parallel based on shape |
| Software prefetching | 0009 | Always enabled, distance tuned per platform |
| Parallel partitioning | 0007 | Node size > 10k rows |
| Bin storage width (4/8/16-bit) | 0008 | Based on n_bins per feature |
| EFB bundling | 0011 | Sparsity > 50%, significant reduction |

**User can query** what was auto-selected via `TrainingInfo` after training, but cannot override (it would only hurt performance).

### User-Controlled (Explicit Choice)

These optimizations affect model quality or training semantics. User must enable explicitly:

| Optimization | RFC | Why User Choice | Default |
|--------------|-----|-----------------|---------|
| Row subsampling | 0007 | Regularization effect | None (1.0) |
| GOSS sampling | 0007 | Changes gradient distribution | None |
| Column subsampling | 0007 | Regularization effect | None (1.0) |
| Gradient quantization (lossy) | 0012 | ~0.1-0.5% accuracy impact | F32 |

**We validate and warn** if user's choice seems suboptimal (e.g., GOSS on <50k rows, subsampling on tiny datasets). Warnings are logged but user choice is respected.

## L2 Cache and Histogram Strategy

The **key auto-selections** for histogram building are based on L2 cache fit and problem shape.

### Two Orthogonal Choices

1. **Iteration order**: Row-wise vs Feature-wise (determined by L2 cache fit)
2. **Parallelism**: Serial, Feature-parallel, or Row-parallel (determined by problem shape)

### The Cache Problem

Histogram building has two possible iteration orders:

```text
Row-wise (histogram fits L2):       Feature-wise (histogram > L2):
  for row in rows:                    for feat in features:
    for feat in features:               for row in rows:
      hist[feat][bin] += grad             hist[feat][bin] += grad
```

This has two access patterns:

1. **Row-wise**: For each row, update all features. Histogram must fit in cache.
2. **Feature-wise**: For each feature, iterate all rows. Only one feature histogram in cache.

### When to Use Which

```text
histogram_bytes = n_features × max_bins × 12  // grad(4) + hess(4) + count(4)

if histogram_bytes ≤ L2_cache:
    use RowWise     // Histogram stays hot, best locality
else:
    use FeatureParallel or RowParallel
```

### Platform Impact

| Platform | L2 Cache | histogram_bytes Threshold | Typical Features |
|----------|----------|--------------------------|------------------|
| x86_64 | 256KB-1MB | ~20-80 features @ 256 bins | Often exceeds |
| Apple Silicon | 4-16MB | ~300-1300 features @ 256 bins | Usually fits |

**Implication**: Apple Silicon's larger L2 means row-wise accumulation is optimal for most datasets. On x86, we more often fall back to feature-parallel.

## Data Characteristics Detection

At training start, we detect data properties to guide auto-selection:

```rust
/// Detected at quantization time, used for auto-selection.
pub struct DataCharacteristics {
    pub n_rows: usize,
    pub n_features: usize,
    pub n_outputs: usize,
    pub max_bins: u16,
    pub sparsity: f32,              // Fraction of missing values
    pub features_with_few_bins: usize, // Features with ≤15 bins
}

impl DataCharacteristics {
    /// Histogram memory for all features (bytes).
    pub fn histogram_bytes(&self) -> usize {
        self.n_features * self.max_bins as usize * 12
    }
    
    /// Does histogram fit in L2 cache?
    pub fn histogram_fits_l2(&self) -> bool {
        self.histogram_bytes() <= platform::l2_cache_size()
    }
}
```

## Platform-Specific Optimizations

Different CPU architectures benefit from different strategies:

```rust
/// Platform-specific tuning, detected at compile time.
pub mod platform {
    #[cfg(target_arch = "x86_64")]
    pub mod x86_64 {
        pub const L2_CACHE_SIZE: usize = 512 * 1024;  // Conservative estimate
        pub const PREFETCH_DISTANCE: usize = 8;
        pub const CACHE_LINE: usize = 64;
        
        #[inline]
        pub fn prefetch<T>(ptr: *const T) {
            unsafe { std::arch::x86_64::_mm_prefetch(ptr as *const i8, _MM_HINT_T0) }
        }
        
        // AVX2 available via is_x86_feature_detected!("avx2")
    }
    
    #[cfg(target_arch = "aarch64")]
    pub mod aarch64 {
        pub const L2_CACHE_SIZE: usize = 4 * 1024 * 1024;  // Apple Silicon M1+
        pub const PREFETCH_DISTANCE: usize = 12;  // Wider pipelines
        pub const CACHE_LINE: usize = 128;        // Apple Silicon
        
        #[inline]
        pub fn prefetch<T>(ptr: *const T) {
            unsafe { core::arch::aarch64::__prefetch(ptr as *const i8, 0, 3, 1) }
        }
        
        // NEON always available on aarch64
    }
}
```

### Platform Differences

| Aspect | x86_64 | Apple Silicon (aarch64) |
|--------|--------|------------------------|
| L2 cache | 256KB - 1MB typical | 4-16MB per cluster |
| Cache line | 64 bytes | 128 bytes |
| Prefetch distance | 8 elements | 12 elements |
| SIMD | AVX2/AVX-512 (detect) | NEON (always) |
| Memory bandwidth | Lower | Higher (unified memory) |

**Implication**: Apple Silicon's larger caches and higher bandwidth mean:

- Row-wise histogram more often beneficial (histogram fits cache)
- Gradient quantization less beneficial (bandwidth not bottleneck)
- Prefetch distance should be larger

## Const Generics for Zero-Cost Abstraction

We use two complementary techniques for zero-cost dispatch:

### 1. Traits for Gradient Types

The `Gradients` trait (RFC-0012) enables generic accumulation:

```rust
/// Generic over gradient storage - monomorphized at compile time.
fn accumulate<G: Gradients>(
    histogram: &mut NodeHistogram,
    quantized: &QuantizedMatrix,
    grads: &G,
    rows: &[u32],
);
```

Calling `grads.grad(idx)` is inlined. For `GradientsInt16`, dequantization (multiply by scale) is fused into the call.

### 2. Const Generics for Algorithm Variants

Prefetch and bin width are compile-time parameters:

```rust
fn accumulate_inner<G: Gradients, const PREFETCH: bool>(
    hist: &mut FeatureHistogram,
    bins: BinColumn<'_>,
    grads: &G,
    rows: &[u32],
);
```

**Dispatch once** on `GradientStorage` variant (runtime), then call monomorphized generic. No per-element dispatch.

## Training Info (Observability)

Users can inspect what optimizations were applied:

```rust
/// Returned alongside the trained model.
pub struct TrainingInfo {
    pub data: DataCharacteristics,
    pub auto_selections: AutoSelections,
    pub warnings: Vec<String>,
}

pub struct AutoSelections {
    pub histogram_algorithm: HistogramAlgorithm,
    pub accumulation_strategy: AccumulationStrategy,
    pub bin_packing: BinPackingStats,
    pub efb_enabled: bool,
    pub efb_bundle_count: Option<usize>,
}

pub struct BinPackingStats {
    pub features_u4: usize,
    pub features_u8: usize,
    pub features_u16: usize,
    pub memory_saved_bytes: usize,
}
```

## Gradient Quantization: Optional and Explicit

Gradient quantization is **user-controlled** because it's lossy. See RFC-0012 for storage details.

```rust
pub struct GBTreeParams {
    // ... other params ...
    
    /// Gradient precision. Default: F32 (no quantization).
    /// Int16 recommended only for >1M rows where bandwidth is bottleneck.
    pub gradient_precision: GradientPrecision,
}

#[derive(Default)]
pub enum GradientPrecision {
    #[default]
    F32,      // Full precision
    Int16,    // ~0.1% accuracy loss, 4× bandwidth reduction
}
```

**Not auto-selected** because the accuracy tradeoff is domain-specific. A 0.1% loss might be unacceptable for medical models but fine for click prediction.

## Design Decisions

### DD-1: Never Silently Affect Model Quality

**Context**: Some optimizations (GOSS, quantization) speed up training but change results.

**Decision**: These require explicit user opt-in.

**Rationale**: Users trust that the same parameters produce the same model. Silent optimization that changes results breaks reproducibility and trust.

### DD-2: Warn Instead of Override

**Context**: User sets `subsample=0.5` on 5k rows where it's pure overhead.

**Decision**: Warn but respect user choice.

**Rationale**: User might be testing sampling behavior, benchmarking, or have domain knowledge we don't. Our job is to inform, not dictate.

### DD-3: Auto-Selection is Not User-Configurable

**Context**: Should users be able to force row-wise histogram?

**Decision**: No. Auto-selected optimizations are internal.

**Rationale**: There's no scenario where forcing a suboptimal strategy helps. Exposing these adds cognitive load without benefit. Advanced users can fork.

### DD-4: Platform Detection at Compile Time

**Context**: Detect CPU features at runtime or compile time?

**Decision**: Architecture at compile time (`#[cfg]`), specific features at runtime (`is_x86_feature_detected!`).

**Rationale**: Architecture (x86 vs ARM) is known at compile time. Specific features (AVX-512) vary and need runtime detection.

## Summary Table

| Category | User Configures | We Select | Examples |
|----------|-----------------|-----------|----------|
| Model Quality | ✓ | | GOSS, subsampling, gradient quantization |
| Execution Speed | | ✓ | Prefetch, histogram algo, bin packing |
| Always Best | | (hardcoded) | Subtraction trick, pooling, SoA |

## References

- [XGBoost Tree Methods](https://xgboost.readthedocs.io/en/latest/treemethod.html)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
```
