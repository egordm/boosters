# RFC-0012: Histogram Building

- **Status**: Draft
- **Created**: 2024-11-30
- **Updated**: 2024-11-30
- **Depends on**: RFC-0011 (Quantization & Binning)
- **Scope**: Gradient histogram construction and storage for split finding

## Summary

This RFC defines how gradient/hessian histograms are built from quantized features. It covers:

1. **Histogram storage**: Per-node, per-feature gradient aggregates
2. **Build algorithms**: Row iteration, SIMD accumulation
3. **Histogram subtraction**: Computing sibling from parent
4. **Multi-output support**: Vector-valued gradients

## Motivation

Histogram building is the core compute kernel of gradient boosting training. For each tree node:

1. Iterate over rows belonging to that node
2. For each feature, accumulate gradients into bins
3. Use aggregated histograms to find best split

The histogram builder is called O(trees × nodes_per_tree × features) times, making it
performance-critical. Key optimizations:

- **Histogram subtraction**: Build histogram for smaller child, derive larger from parent
- **SIMD accumulation**: Vectorize gradient summation across multiple bins
- **Cache blocking**: Process rows in cache-friendly chunks

## Design

### Overview

```
Per-node histogram building:
━━━━━━━━━━━━━━━━━━━━━━━━━━━

GHistIndexMatrix          GradientBuffer           NodeHistogram
┌───────────────┐        ┌──────────────┐        ┌──────────────────┐
│ bin[row, feat]│   +    │ grad[row]    │   =    │ sum_grad[bin]    │
│    (u8)       │        │ hess[row]    │        │ sum_hess[bin]    │
└───────────────┘        └──────────────┘        │ count[bin]       │
                                                 └──────────────────┘
```

### Histogram Storage

```rust
/// Gradient histogram for a single feature
/// 
/// Each bin stores sum of gradients, hessians, and sample count.
/// SoA layout for cache-friendly subtraction and SIMD operations.
pub struct FeatureHistogram {
    sum_grad: Box<[f32]>,
    sum_hess: Box<[f32]>,
    count: Box<[u32]>,
    num_bins: u16,  // u16 to match max bin index type
}

impl FeatureHistogram {
    pub fn new(num_bins: u16) -> Self;
    pub fn add(&mut self, bin: usize, grad: f32, hess: f32);
    pub fn subtract_from(&mut self, parent: &Self);
    pub fn reset(&mut self);
    pub fn bin_stats(&self, bin: usize) -> (f32, f32, u32);
}

/// Histograms for all features at a single node
pub struct NodeHistogram {
    features: Box<[FeatureHistogram]>,
    total_grad: f32,
    total_hess: f32,
    total_count: u32,
}
```

Key operations:
- `add(bin, grad, hess)`: Accumulate sample into bin
- `subtract_from(parent)`: Derive sibling histogram (parent - self)
- `reset()`: Zero out for reuse
```

### Histogram Builder

```rust
/// Builds histograms from quantized features and gradients
pub struct HistogramBuilder;

impl HistogramBuilder {
    /// Build histogram for a node from its row indices
    /// 
    /// Core algorithm:
    /// 1. Reset histogram to zero
    /// 2. For each row in node:
    ///    - For each feature, add (grad, hess) to bin
    /// 3. Accumulate totals
    pub fn build<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        rows: &[u32],
    );
    
    /// Build with per-feature parallelism (Rayon)
    /// Each feature histogram is independent → embarrassingly parallel
    pub fn build_parallel<B: BinIndex>(
        &self,
        hist: &mut NodeHistogram,
        index: &QuantizedMatrix<B>,
        grads: &GradientBuffer,
        rows: &[u32],
    );
}
```

Optimization strategies (implementation details, not in RFC):
- SIMD accumulation for contiguous bin ranges
- Sorted-row accumulation for better locality
- Prefetching next row's bins
```

### Histogram Subtraction

The key optimization: when splitting a node, build histogram for smaller child,
derive larger child via `parent - smaller = larger`. Nearly halves build cost.

```rust
pub struct HistogramSubtractor;

impl HistogramSubtractor {
    /// Given parent and one child, compute the other child
    pub fn compute_sibling(
        parent: &NodeHistogram,
        child: &NodeHistogram,
        sibling: &mut NodeHistogram,
    );
    
    /// Decide which child to build (prefer smaller)
    pub fn select_build_child(left_count: u32, right_count: u32) -> ChildSide;
}
```
```

### Multi-Output Histogram Support

For multi-class training, histograms store vector gradients:

```rust
/// Histogram for multi-output (multi-class) training
/// Each bin stores gradient/hessian vectors, one element per class.
pub struct MultiOutputFeatureHistogram {
    /// Flat storage: [bin * num_classes + class]
    sum_grad: Box<[f32]>,
    sum_hess: Box<[f32]>,
    count: Box<[u32]>,
    num_bins: u16,
    num_classes: u16,
}
```

The const-generic variant `<const N: usize>` can be added for small, fixed class counts
where stack allocation and loop unrolling are beneficial.
```

### Histogram Cache (Optional)

A pool of pre-allocated histograms can reduce allocation overhead:
- Borrow histogram from pool for node
- Return to pool after node is expanded
- Useful when histogram allocation is measurable overhead
```

### Memory Layout

```
FeatureHistogram layout (256 bins):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

sum_grad: [f32; 256] = 1024 bytes
sum_hess: [f32; 256] = 1024 bytes
count:    [u32; 256] = 1024 bytes
                     ─────────────
                Total: 3 KB per feature

NodeHistogram for 100 features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

100 features × 3 KB = 300 KB per node

With histogram subtraction:
- Build 1 child = 300 KB writes
- Subtract for sibling = 300 KB reads + writes
- Total: ~900 KB memory traffic per node split

Without subtraction:
- Build 2 children = 600 KB writes
- Total: ~600 KB memory traffic per node split

But subtraction processes HALF the rows → net win for large nodes.
```

## Design Decisions

### DD-1: Separate Grad/Hess/Count Arrays vs Interleaved

**Context**: How to lay out histogram bin data in memory.

**Options considered**:

1. **Interleaved (AoS)**: `[(grad, hess, count), (grad, hess, count), ...]`
2. **Separate (SoA)**: `[grad0, grad1, ...], [hess0, hess1, ...], [cnt0, cnt1, ...]`

**Decision**: Separate arrays (SoA layout).

**Rationale**:

- Subtraction operates on grad, hess, count independently
- SIMD can process 8 grad values at once without gather
- Count might be omitted in some algorithms (just grad/hess)
- Cache line utilization: when only reading grads, don't pollute with hess/count

### DD-2: Histogram Subtraction Strategy

**Context**: When to use subtraction vs direct build.

**Options considered**:

1. **Always subtract**: Build smaller child, derive larger
2. **Never subtract**: Build both children directly  
3. **Threshold-based**: Subtract only if size difference > threshold

**Decision**: Always build smaller child, derive larger via subtraction.

**Rationale**:

- Subtraction is O(n_bins), build is O(n_rows × n_features)
- For nodes with thousands of rows, subtraction is always faster
- XGBoost and LightGBM both use this strategy universally
- Threshold-based adds complexity with minimal benefit

### DD-3: Multi-Output Histogram Design

**Context**: Support multi-class classification with vector gradients.

**Options considered**:

1. **Separate histograms per class**: `Vec<FeatureHistogram>` per class
2. **Interleaved in single struct**: `sum_grad[bin][class]`
3. **Const generic**: `FeatureHistogram<const NUM_CLASSES: usize>`

**Decision**: Provide both const-generic (for known small class counts) and dynamic versions.

**Rationale**:

- Const generic enables stack allocation and loop unrolling for common cases (2-10 classes)
- Dynamic needed for many-class problems (100+ classes)
- Split gain computation naturally iterates classes, so `[bin][class]` layout is good

### DD-4: Parallelization Strategy

**Context**: How to parallelize histogram building.

**Options considered**:

1. **Per-feature parallelism**: Each thread builds histograms for subset of features
2. **Per-row parallelism**: Each thread processes subset of rows, merge at end
3. **Block parallelism**: Divide rows into blocks, each thread builds partial histogram, then reduce

**Decision**: Start with per-feature parallelism (simplest), explore row parallelism later.

**Rationale**:

- Per-feature: No synchronization needed, each feature histogram independent
- Per-row: Requires atomic adds or per-thread scratch + reduction
- For many features (100+), per-feature parallelism is sufficient
- Per-row might be faster for few features, many rows (future optimization)

## Integration

| Component | Integration Point | Notes |
|-----------|-------------------|-------|
| RFC-0011 (Quantization) | `QuantizedMatrix`, `BinCuts` | Input to histogram builder |
| RFC-0013 (Split Finding) | `NodeHistogram` | Output consumed by split finder |
| RFC-0014 (Row Partitioning) | Row indices per node | Determines which rows to iterate |
| RFC-0018 (Multi-output) | `MultiOutputFeatureHistogram` | Vector gradient support |

### Integration with Existing Code

- **`src/training/buffer.rs`**: `GradientBuffer` provides gradients/hessians
- **`src/training/mod.rs`**: Export histogram types alongside existing training infrastructure
- **New module**: `src/training/histogram.rs` for `FeatureHistogram`, `NodeHistogram`, `HistogramBuilder`

## Open Questions

1. **Gradient quantization**: **Yes** — LightGBM uses 16-bit packed gradients. Worth supporting as a P2 optimization.

2. **Sparse histograms**: LightGBM uses sparse histograms only for bundled multi-value features, not for high-cardinality categoricals directly. High-cardinality categoricals are handled via feature bundling (EFB). **Lower priority** — defer to Feature Bundling RFC.

3. **GPU histogram building**: Separate RFC as noted.

## Future Work

- [ ] SIMD-optimized accumulation kernels
- [ ] Per-row parallel build with reduction
- [ ] Gradient quantization (16-bit packed)
- [ ] Sparse histogram variant
- [ ] GPU histogram building (separate RFC)

## References

- [XGBoost hist_util.h](https://github.com/dmlc/xgboost/blob/master/src/common/hist_util.h)
- [LightGBM histogram.cpp](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp)
- [Feature Overview](../FEATURE_OVERVIEW.md) - Priority and design context

## Changelog

- 2024-11-30: Initial draft
