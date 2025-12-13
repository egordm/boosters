# RFC-0009: Histogram Building

- **Status**: Draft
- **Created**: 2024-12-05
- **Updated**: 2024-12-05
- **Depends on**: RFC-0008 (Feature Quantization), RFC-0012 (Gradient Quantization)
- **Scope**: Gradient histogram accumulation, pooling, and split finding

## Summary

Histogram building is the core computation in tree training. For each node, we:

1. **Build**: Accumulate gradient statistics into per-feature histograms
2. **Find split**: Scan histograms to find the best split point

This RFC covers histogram structures, accumulation strategies, histogram pooling, and split finding. Tree growing orchestration is in RFC-0007.

## Overview

### Role in Tree Growing

This RFC defines the **building blocks**. RFC-0007 orchestrates them:

```text
RFC-0007 (GBTree Training)          RFC-0009 (This RFC)
─────────────────────────           ───────────────────
Tree growing algorithm        uses→  HistogramBuilder
Expansion strategy            uses→  HistogramPool
Subtraction trick logic       uses→  NodeHistogram arithmetic
Split decision                uses→  SplitFinder
```

### Why Histograms?

| Approach | Split Finding Complexity |
|----------|-------------------------|
| Sort per feature | O(n × d × log n) |
| Histogram scan | O(n × d + bins × d) |

With bins=256 and n=1M, histogram approach is ~20× faster.

## Histogram Structures

### NodeHistogram

All histograms have **fixed size**: `n_features × max_bins`. This enables pooling.

```rust
pub struct NodeHistogram {
    features: Box<[FeatureHistogram]>,
}

impl NodeHistogram {
    pub fn new(n_features: usize, max_bins: u16) -> Self;
    pub fn reset(&mut self);
    pub fn feature(&self, f: usize) -> &FeatureHistogram;
    pub fn feature_mut(&mut self, f: usize) -> &mut FeatureHistogram;
    pub fn n_features(&self) -> usize;
}
```

### FeatureHistogram

Per-feature histogram in **SoA layout**:

```rust
pub struct FeatureHistogram {
    grads: Box<[f32]>,
    hess: Box<[f32]>,
    counts: Box<[u32]>,
    n_bins: u16,
}

impl FeatureHistogram {
    pub fn add(&mut self, bin: u16, grad: f32, hess: f32);
    pub fn grads(&self) -> &[f32];
    pub fn hess(&self) -> &[f32];
    pub fn counts(&self) -> &[u32];
}
```

**Why SoA?** Split enumeration scans `grads` sequentially. Contiguous layout enables SIMD.

### Histogram Arithmetic

Used by the subtraction trick (orchestrated in RFC-0007):

```rust
impl NodeHistogram {
    /// self = self - other
    pub fn sub_assign(&mut self, other: &NodeHistogram);
    
    /// self = self + other
    pub fn add_assign(&mut self, other: &NodeHistogram);
    
    /// self = other
    pub fn copy_from(&mut self, other: &NodeHistogram);
}
```

Pseudocode for `sub_assign`:

```text
sub_assign(self, other):
    for f in 0..n_features:
        for b in 0..n_bins:
            self.features[f].grads[b] -= other.features[f].grads[b]
            self.features[f].hess[b] -= other.features[f].hess[b]
            self.features[f].counts[b] -= other.features[f].counts[b]
```

## Histogram Pool

### Why Pooling?

During tree growth, multiple histograms exist simultaneously:

- **DepthWise**: Up to 2^depth nodes at current level
- **LeafWise**: Up to max_leaves candidate nodes

Since all histograms have identical size, we reuse them via a pool.

### HistogramPool

```rust
pub struct HistogramPool { /* ... */ }

impl HistogramPool {
    pub fn new(n_features: usize, max_bins: u16, capacity: usize) -> Self;
    pub fn reset(&mut self);
    pub fn acquire(&mut self, node_id: u32) -> &mut NodeHistogram;
    pub fn get(&self, node_id: u32) -> Option<&NodeHistogram>;
    pub fn get_mut(&mut self, node_id: u32) -> Option<&mut NodeHistogram>;
    pub fn release(&mut self, node_id: u32);
}
```

## Histogram Building

### Two Orthogonal Concerns

Histogram building has **two independent choices**:

1. **Iteration order**: Row-wise vs Feature-wise (determined by L2 cache fit)
2. **Parallelism**: Serial, Feature-parallel, or Row-parallel

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Iteration Order                                  │
│         (based on whether histogram fits in L2 cache)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RowWise (histogram fits L2)     FeatureWise (histogram > L2)       │
│  ────────────────────────────    ─────────────────────────────      │
│  for row in rows:                for feat in features:              │
│    for feat in features:           for row in rows:                 │
│      hist[feat][bin] += g            hist[feat][bin] += g           │
│                                                                     │
│  ✓ Best when histogram hot       ✓ One feature histogram hot        │
│  ✗ Thrashes if histogram > L2    ✗ More memory traffic              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Parallelism Strategy                             │
│              (based on n_features, n_rows, n_threads)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Serial               FeatureParallel         RowParallel           │
│  ──────               ───────────────         ───────────           │
│  Single thread        Split features          Split rows            │
│                       across threads          across threads        │
│                       No sync needed          Merge at end          │
│                                                                     │
│  Small problems       Many features           Few features,         │
│                       (≥4× n_threads)         many rows (>10k)      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### HistogramBuilder

```rust
pub struct HistogramBuilder {
    iteration: IterationOrder,
    parallelism: Parallelism,
}

pub enum IterationOrder {
    RowWise,      // Histogram fits L2: iterate rows, scatter to features
    FeatureWise,  // Histogram > L2: iterate features, each stays hot
}

pub enum Parallelism {
    Serial,
    FeatureParallel,  // Partition features across threads
    RowParallel,      // Partition rows, merge histograms
}

impl HistogramBuilder {
    pub fn new(iteration: IterationOrder, parallelism: Parallelism) -> Self;
    
    pub fn build<G: Gradients>(
        &self,
        histogram: &mut NodeHistogram,
        quantized: &QuantizedFeatures,
        grads: &G,
        rows: &[u32],
    );
}
```

### Strategy Selection

```text
select_strategies(n_features, n_rows, max_bins, n_threads):
    histogram_bytes = n_features * max_bins * 12  // grad + hess + count
    l2_cache = platform::l2_cache_size()
    
    // Iteration order: based on L2 cache fit
    if histogram_bytes <= l2_cache:
        iteration = RowWise
    else:
        iteration = FeatureWise
    
    // Parallelism: based on problem shape
    if n_threads == 1 or (n_rows < 1000 and n_features < 100):
        parallelism = Serial
    else if n_features >= 4 * n_threads:
        parallelism = FeatureParallel
    else if n_rows > 10_000:
        parallelism = RowParallel
    else:
        parallelism = Serial
    
    return (iteration, parallelism)
```

### Build Algorithms

**Row-wise iteration** (histogram fits L2):

```text
build_row_wise(histogram, quantized, grads, rows):
    for row in rows:
        g = grads.grad(row)
        h = grads.hess(row)
        
        for feat in 0..n_features:
            bin = quantized.bin(feat, row)
            histogram.feature(feat).add(bin, g, h)
```

**Feature-wise iteration** (histogram > L2):

```text
build_feature_wise(histogram, quantized, grads, rows):
    for feat in 0..n_features:
        feat_hist = histogram.feature(feat)
        
        for row in rows:
            bin = quantized.bin(feat, row)
            feat_hist.add(bin, grads.grad(row), grads.hess(row))
```

**Feature-parallel** (works with either iteration order):

```text
build_feature_parallel(histogram, quantized, grads, rows):
    // Partition features across threads
    parallel_for feat_chunk in partition(0..n_features, n_threads):
        for feat in feat_chunk:
            feat_hist = histogram.feature(feat)  // Disjoint, no sync
            
            for row in rows:
                bin = quantized.bin(feat, row)
                feat_hist.add(bin, grads.grad(row), grads.hess(row))
```

**Row-parallel** (works with either iteration order):

```text
build_row_parallel(histogram, quantized, grads, rows):
    // Each thread builds local histogram
    local_hists = parallel_map(partition(rows, n_threads), |row_chunk|:
        local = NodeHistogram::new()
        
        for row in row_chunk:
            g = grads.grad(row)
            h = grads.hess(row)
            for feat in 0..n_features:
                bin = quantized.bin(feat, row)
                local.feature(feat).add(bin, g, h)
        
        return local
    )
    
    // Merge thread-local histograms
    for local in local_hists:
        histogram.add_assign(local)
```

## Split Finding

### SplitFinder

Uses the **greedy split enumeration** algorithm from XGBoost: try all possible split points and pick maximum gain.

```rust
pub struct SplitFinder {
    params: SplitParams,
}

pub struct SplitParams {
    pub lambda: f32,           // L2 regularization
    pub gamma: f32,            // Minimum split gain
    pub min_child_weight: f32, // Minimum hessian sum per child
}

impl SplitFinder {
    pub fn new(params: SplitParams) -> Self;
    
    pub fn find_best(
        &self,
        histogram: &NodeHistogram,
        parent_grad: f32,
        parent_hess: f32,
        allowed_features: &[u32],
    ) -> SplitInfo;
}
```

### SplitInfo

```rust
pub struct SplitInfo {
    pub feature: u32,
    pub bin: u16,
    pub gain: f32,
    pub left_grad: f32,
    pub left_hess: f32,
    pub right_grad: f32,
    pub right_hess: f32,
    pub default_left: bool,
}

impl SplitInfo {
    pub fn none() -> Self;        // No valid split
    pub fn is_valid(&self) -> bool;
}
```

### Gain Formula

XGBoost gain formula:

```text
gain = 0.5 × [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G_P²/(H_P + λ)] - γ
```

Where: G = gradient sum, H = hessian sum, λ = L2 reg, γ = min gain.

### Greedy Split Enumeration

Scan bins with cumulative sum. Check both directions for missing value handling:

```text
find_best(histogram, parent_grad, parent_hess, allowed_features):
    best = SplitInfo::none()
    
    for feat in allowed_features:
        candidate = find_best_for_feature(histogram.feature(feat), feat, 
                                          parent_grad, parent_hess)
        if candidate.gain > best.gain:
            best = candidate
    
    return best

find_best_for_feature(feat_hist, feature, parent_grad, parent_hess):
    best = SplitInfo::none()
    n_bins = feat_hist.n_bins
    missing_bin = n_bins - 1
    
    // Forward scan: missing values go RIGHT
    left_g, left_h = 0, 0
    for bin in 0..missing_bin:
        left_g += feat_hist.grads[bin]
        left_h += feat_hist.hess[bin]
        right_g = parent_grad - left_g
        right_h = parent_hess - left_h
        
        if is_valid_split(left_h, right_h):
            gain = compute_gain(left_g, left_h, right_g, right_h, parent_grad, parent_hess)
            if gain > best.gain:
                best = SplitInfo { feature, bin, gain, left_g, left_h, right_g, right_h, 
                                   default_left: false }
    
    // Backward scan: missing values go LEFT  
    right_g, right_h = 0, 0
    for bin in (1..=missing_bin).reverse():
        right_g += feat_hist.grads[bin]
        right_h += feat_hist.hess[bin]
        left_g = parent_grad - right_g
        left_h = parent_hess - right_h
        
        if is_valid_split(left_h, right_h):
            gain = compute_gain(left_g, left_h, right_g, right_h, parent_grad, parent_hess)
            if gain > best.gain:
                best = SplitInfo { feature, bin: bin - 1, gain, left_g, left_h, 
                                   right_g, right_h, default_left: true }
    
    return best

is_valid_split(left_h, right_h):
    return left_h >= min_child_weight and right_h >= min_child_weight

compute_gain(left_g, left_h, right_g, right_h, parent_g, parent_h):
    left_score = left_g² / (left_h + λ)
    right_score = right_g² / (right_h + λ)
    parent_score = parent_g² / (parent_h + λ)
    return 0.5 * (left_score + right_score - parent_score) - γ
```

## Design Decisions

### DD-1: Separate Build and Find

**Decision**: `HistogramBuilder::build()` and `SplitFinder::find_best()` are separate operations.

**Rationale**:

- Clearer control flow in tree growing (RFC-0007)
- Pool acquire/release is explicit
- Subtraction trick orchestration is visible

### DD-2: Fixed-Size Histograms

**Decision**: All histograms use `n_features × max_bins` size.

**Rationale**:

- Enables pooling (any histogram fits any node)
- Unused bins stay zero (acceptable waste)

### DD-3: SoA Histogram Layout

**Decision**: Store per-bin data as separate arrays.

**Rationale**:

- Split enumeration scans grads sequentially
- Enables SIMD for histogram arithmetic

### DD-4: Orthogonal Iteration and Parallelism

**Decision**: Iteration order (row-wise/feature-wise) and parallelism (serial/feature-parallel/row-parallel) are independent choices.

**Rationale**:

- L2 cache fit determines iteration order
- Problem shape determines parallelism
- All combinations are valid

### DD-5: L2 Cache Determines Iteration Order

**Decision**: Use row-wise when histogram fits L2, feature-wise otherwise.

**Rationale**:

- Row-wise has best locality when histogram stays hot
- Feature-wise avoids thrashing for large histograms
- Apple Silicon's larger L2 means row-wise works more often

### DD-6: Greedy Split Finding

**Decision**: Use greedy enumeration (XGBoost "exact" algorithm on bins).

**Rationale**:

- O(bins) per feature, not O(n)
- Optimal within bin granularity
- Simple and fast for ≤256 bins

## References

- [XGBoost: Exact Greedy Algorithm](https://arxiv.org/abs/1603.02754)
- [LightGBM Histogram](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp)
