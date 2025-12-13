```markdown
# RFC-0009: Histogram Building

- **Status**: Draft
- **Created**: 2024-12-05
- **Updated**: 2024-12-05
- **Depends on**: RFC-0008 (Quantization)
- **Scope**: Gradient histogram accumulation, pooling, and split finding

## Summary

Histogram building is the core computation in gradient boosted tree training. For each tree node, we accumulate gradient statistics into per-feature histograms, then scan those histograms to find the best split. This RFC covers histogram structures, accumulation strategies, the subtraction trick, and histogram pooling.

## Overview

```text
HistogramBuilder
│
├── pool: HistogramPool              ← LRU cache of fixed-size histograms
│   └── histograms: Vec<NodeHistogram>
│
├── strategy: AccumulationStrategy   ← Serial, FeatureParallel, RowParallel
│
└── split_params: SplitParams        ← lambda, gamma, min_child_weight
```

### Why Histograms?

Without histograms, finding the best split requires sorting all values per feature: O(n × d × log n).

With histograms:

1. **Quantize** features to discrete bins (done once, RFC-0008)
2. **Accumulate** gradients into bins: O(n × d)
3. **Scan** bins to find best split: O(bins × d)

Total: O(n × d) instead of O(n × d × log n). With bins=256 and n=1M, this is ~20× faster.

## Histogram Structure

### NodeHistogram

All histograms have the **same fixed size**: `n_features × max_bins`. This uniformity enables pooling—any histogram can be reused for any node.

```rust
pub struct NodeHistogram {
    features: Box<[FeatureHistogram]>,
}

impl NodeHistogram {
    pub fn new(cuts: &BinCuts) -> Self;
    pub fn reset(&mut self);  // Zero all bins, ready for reuse
    pub fn feature(&self, f: usize) -> &FeatureHistogram;
    pub fn feature_mut(&mut self, f: usize) -> &mut FeatureHistogram;
}
```

### FeatureHistogram

Per-feature histogram in **SoA layout** (Struct of Arrays):

```rust
pub struct FeatureHistogram {
    grads: Box<[f32]>,   // Gradient sum per bin
    hess: Box<[f32]>,    // Hessian sum per bin
    counts: Box<[u32]>,  // Sample count per bin
    n_bins: u16,
}

impl FeatureHistogram {
    pub fn add(&mut self, bin: u16, grad: f32, hess: f32);
    pub fn grads(&self) -> &[f32];
    pub fn hess(&self) -> &[f32];
}
```

**Why SoA?** Split enumeration scans gradients sequentially to compute cumulative sums. Contiguous `grads` array enables SIMD vectorization.

### Histogram Arithmetic

```rust
impl SubAssign<&NodeHistogram> for NodeHistogram {
    // Subtract element-wise: self[i] -= other[i]
}

impl AddAssign<&NodeHistogram> for NodeHistogram {
    // Add element-wise: self[i] += other[i]
}
```

**Why `SubAssign` instead of `Sub`?** Avoids allocating a new histogram. The target histogram is already acquired from the pool; we modify it in place.

## Histogram Pool

### Why Pooling?

During tree growth, we need histograms for multiple nodes simultaneously:

- **DepthWise**: Up to 2^depth nodes at current level
- **LeafWise**: Up to max_leaves candidate nodes

Allocating/deallocating is expensive. Since **all histograms have identical size** (`n_features × max_bins`), we can reuse them via a pool.

### Pool Structure

The pool uses **LRU eviction**: least-recently-used histograms are evicted first when capacity is exceeded.

```rust
pub struct HistogramPool {
    histograms: Vec<NodeHistogram>,  // Pre-allocated, all same size
    node_to_idx: HashMap<u32, usize>, // Active: node_id → pool index
    free_list: Vec<usize>,            // Available indices
    lru_order: VecDeque<u32>,         // For eviction: oldest at front
    cuts: BinCuts,
}

impl HistogramPool {
    pub fn new(cuts: &BinCuts, capacity: usize) -> Self;
    
    /// Acquire histogram for node. Reuses from free_list or evicts LRU.
    pub fn acquire(&mut self, node_id: u32) -> &mut NodeHistogram;
    
    /// Release histogram back to pool. Clears it and adds to free_list.
    pub fn release(&mut self, node_id: u32);
    
    /// Get histogram without modifying LRU (for subtraction source).
    pub fn get(&self, node_id: u32) -> Option<&NodeHistogram>;
}
```

### Pool Lifecycle During Tree Growth

```text
1. acquire(ROOT) → build → find split
2. Split ROOT into LEFT, RIGHT:
   - acquire(LEFT), acquire(RIGHT)
   - Build smaller child, derive larger via subtraction
   - release(ROOT)  ← Parent no longer needed
3. For each child that splits:
   - acquire(grandchildren)
   - release(child) after grandchildren built
4. At tree end: release all remaining nodes
```

**Key insight**: A node's histogram is needed until both its children are built. Then it can be released. The subtraction trick (see below) uses the parent histogram as the source for `target -= parent`.

### Pool Sizing

```text
DepthWise: Need histograms for current level + building next level
  ≈ 2^max_depth + 2^(max_depth-1) ≈ 1.5 × 2^max_depth

LeafWise: Need histograms for all candidate leaves
  ≈ max_leaves

Capacity = max(1.5 × 2^max_depth, max_leaves) + buffer
```

## Subtraction Trick

### The Insight

A node's rows = left child rows + right child rows. Therefore:

```text
parent_histogram = left_histogram + right_histogram
```

Rearranging:

```text
larger_child_histogram = parent_histogram - smaller_child_histogram
```

Build the smaller child explicitly, derive the larger by subtraction. **Saves ~50% of histogram building work.**

### Algorithm

```text
build_children(parent_id, left_id, right_id, left_rows, right_rows, ...):
  parent_hist = pool.get(parent_id)  // Don't modify LRU
  
  // Determine smaller child
  if left_rows.len() <= right_rows.len():
    small_id, small_rows = left_id, left_rows
    large_id = right_id
  else:
    small_id, small_rows = right_id, right_rows
    large_id = left_id
  
  // Build smaller child explicitly
  small_hist = pool.acquire(small_id)
  accumulate(small_hist, small_rows, ...)
  
  // Derive larger child: large = parent - small
  large_hist = pool.acquire(large_id)
  large_hist.copy_from(parent_hist)  // Copy parent into large
  large_hist -= small_hist           // Subtract in place (no allocation)
  
  // Release parent
  pool.release(parent_id)
  
  return (left_split, right_split)
```

## Accumulation

### Iteration Order: Features-First vs Row-First

The optimal iteration order depends on whether the histogram fits in L2 cache:

**Col-wise (Features-First)**: Default when histogram exceeds L2 cache.

```text
accumulate_colwise(histogram, quantized, grads, hess, rows):
  for feat in 0..n_features:
    bins = quantized.col(feat)        // Contiguous column slice
    hist = histogram.feature_mut(feat)
    for row in rows:
      bin = bins[row]
      hist.add(bin, grads[row], hess[row])
```

**Row-wise (Rows-First)**: Better when histogram fits in L2 cache.

```text
accumulate_rowwise(histogram, quantized, grads, hess, rows):
  for row in rows:
    grad = grads[row]
    hes = hess[row]
    for feat in 0..n_features:
      bin = quantized.get_bin(feat, row)
      histogram.feature_mut(feat).add(bin, grad, hes)
```

### Algorithm Selection (Auto)

```text
select_histogram_algorithm(data: DataCharacteristics) -> HistogramAlgorithm:
  hist_bytes = data.n_features * data.max_bins * 12  // 3 × f32 per bin
  
  if hist_bytes <= data.l2_cache_bytes:
    return RowWise   // Histogram in cache → grad/hess locality wins
  else:
    return ColWise   // Large histogram → feature column locality wins
```

**Why this matters**:

| Algorithm | Histogram Access | Gradient Access | Best When |
|-----------|-----------------|-----------------|-----------|
| Col-wise | Sequential per feature | Random (by row index) | Large histogram |
| Row-wise | Random (all features) | Sequential | Small histogram (fits L2) |

### Software Prefetching

Prefetching hints tell the CPU to load data before it's needed, hiding memory latency:

```text
accumulate_with_prefetch<const PREFETCH_DISTANCE: usize>(
    histogram, quantized, grads, hess, rows
):
  for feat in 0..n_features:
    bins = quantized.col(feat)
    hist = histogram.feature_mut(feat)
    
    for i in 0..rows.len():
      row = rows[i]
      
      // Prefetch data for future iteration
      if i + PREFETCH_DISTANCE < rows.len():
        future_row = rows[i + PREFETCH_DISTANCE]
        prefetch_read(bins.as_ptr().add(future_row))
        prefetch_read(grads.as_ptr().add(future_row))
        prefetch_read(hess.as_ptr().add(future_row))
      
      bin = bins[row]
      hist.add(bin, grads[row], hess[row])
```

**Prefetch distance tuning**:

```text
optimal_prefetch_distance(element_size: usize) -> usize:
  // Memory latency ~100 cycles, cache line 64 bytes
  // Process ~10 elements per latency period
  cache_line_elements = 64 / element_size
  return max(8, cache_line_elements * 2)  // Typically 8-16
```

**Implementation via intrinsics**:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_prefetch;

#[inline]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}
```

### Const Generic Kernels

Different combinations are specialized at compile time:

```rust
fn accumulate_inner<const ALGORITHM: HistAlgo, const PREFETCH: bool, const BITS: BinBits>(
    histogram: &mut NodeHistogram,
    quantized: &QuantizedMatrix,
    grads: &[f32],
    hess: &[f32],
    rows: &[u32],
) {
    match ALGORITHM {
        HistAlgo::ColWise => {
            for feat in 0..quantized.n_features() {
                let bins = quantized.col(feat);
                let hist = histogram.feature_mut(feat);
                
                // PREFETCH and BITS branches eliminated at compile time
                accumulate_feature::<PREFETCH, BITS>(bins, grads, hess, rows, hist);
            }
        }
        HistAlgo::RowWise => {
            for (i, &row) in rows.iter().enumerate() {
                if PREFETCH && i + 8 < rows.len() {
                    // Prefetch for row-wise
                }
                let grad = grads[row as usize];
                let hes = hess[row as usize];
                for feat in 0..quantized.n_features() {
                    let bin = quantized.get_bin(feat, row as usize);
                    histogram.feature_mut(feat).add(bin, grad, hes);
                }
            }
        }
    }
}
```

### Accumulation Strategy

```rust
pub enum AccumulationStrategy {
    Serial,
    FeatureParallel,
    RowParallel,
}
```

| Strategy | Iteration | Parallelism | Memory Overhead | Best When |
|----------|-----------|-------------|-----------------|-----------|
| Serial | Features-first | None | None | Small nodes |
| FeatureParallel | Features-first | Over features | None | Many features |
| RowParallel | Features-first per thread | Over rows | n_threads × histogram | Many rows, few features |

### FeatureParallel

Partition features across threads. Each thread handles disjoint features, no synchronization needed:

```text
parallel_for feat_chunk in partition(0..n_features, n_threads):
  for feat in feat_chunk:
    bins = quantized.col(feat)
    hist = histogram.feature_mut(feat)  // Disjoint, no conflict
    for row in rows:
      hist.add(bins[row], grads[row], hess[row])
```

### RowParallel

Partition rows across threads. Each thread accumulates into its own histogram, then merge:

```text
thread_histograms = [NodeHistogram::new() for _ in n_threads]

parallel_for (thread_id, row_chunk) in partition(rows, n_threads):
  local_hist = thread_histograms[thread_id]
  for feat in 0..n_features:           // Still features-first within thread
    bins = quantized.col(feat)
    for row in row_chunk:
      local_hist[feat].add(bins[row], grads[row], hess[row])

// Merge: histogram += each thread histogram
for local_hist in thread_histograms:
  histogram += local_hist
```

**Memory cost**: `n_threads × n_features × max_bins × 12 bytes`

## Split Finding

### Gain Formula (XGBoost)

```text
gain = 0.5 × [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G_P²/(H_P + λ)] - γ
```

Where: G = gradient sum, H = hessian sum, λ = L2 reg, γ = min split gain.

### Split Enumeration

Scan bins with cumulative sum. Check both directions for optimal missing value assignment:

```text
find_best_split(histogram, parent_grad, parent_hess, allowed_features, params):
  best = SplitInfo::invalid()
  
  for feat in allowed_features:
    bins = histogram.feature(feat)
    
    // Forward scan: missing values go RIGHT
    left_g, left_h = 0, 0
    for bin in 0..n_bins-1:
      left_g += bins.grads[bin]
      left_h += bins.hess[bin]
      right_g = parent_grad - left_g
      right_h = parent_hess - left_h
      
      if is_valid(left_h, right_h, params):
        gain = compute_gain(left_g, left_h, right_g, right_h, params)
        if gain > best.gain:
          best = SplitInfo { feat, bin, gain, default_left: false, ... }
    
    // Backward scan: missing values go LEFT
    right_g, right_h = 0, 0
    for bin in (1..n_bins).rev():
      right_g += bins.grads[bin]
      right_h += bins.hess[bin]
      left_g = parent_grad - right_g
      left_h = parent_hess - right_h
      
      if is_valid(left_h, right_h, params):
        gain = compute_gain(left_g, left_h, right_g, right_h, params)
        if gain > best.gain:
          best = SplitInfo { feat, bin-1, gain, default_left: true, ... }
  
  return best
```

(Origin: XGBoost's sparsity-aware split finding)

### Split Validity

```text
valid_split(left_h, right_h, params):
  return left_h >= params.min_child_weight 
     and right_h >= params.min_child_weight
```

## HistogramBuilder Interface

Combines pooling, accumulation, and split finding:

```rust
pub struct HistogramBuilder {
    pool: HistogramPool,
    strategy: AccumulationStrategy,
    params: SplitParams,
}

pub struct SplitParams {
    pub lambda: f32,
    pub gamma: f32,
    pub min_child_weight: f32,
}

impl HistogramBuilder {
    pub fn new(cuts: &BinCuts, strategy: AccumulationStrategy, params: SplitParams) -> Self;
    
    /// Build histogram and find best split.
    pub fn build_and_split(
        &mut self,
        node_id: u32,
        quantized: &QuantizedMatrix,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
        allowed_features: &[u32],
    ) -> SplitInfo;
    
    /// Build children using subtraction trick, return their splits.
    pub fn build_children(
        &mut self,
        parent_id: u32,
        left_id: u32, right_id: u32,
        left_rows: &[u32], right_rows: &[u32],
        quantized: &QuantizedMatrix,
        grads: &[f32], hess: &[f32],
        allowed_features: &[u32],
    ) -> (SplitInfo, SplitInfo);
    
    /// Release histogram for completed node.
    pub fn release(&mut self, node_id: u32);
}
```

## Design Decisions

### DD-1: Fixed-Size Histograms for Pooling

**Context**: Nodes have different numbers of rows, but do they need different histogram sizes?

**Decision**: All histograms use `n_features × max_bins`, regardless of node.

**Rationale**:
- Histogram size depends on feature count and max bins, not row count
- Fixed size enables pooling without per-node allocation
- Unused bins (features with fewer unique values) just stay zero
- Tradeoff: slight memory waste vs. allocation overhead

### DD-2: SoA Histogram Layout

**Context**: Store bins as AoS `[{grad, hess, count}, ...]` or SoA `{grads: [...], hess: [...], counts: [...]}`?

**Decision**: SoA layout.

**Rationale**: (XGBoost approach)
- Split enumeration scans `grads` array sequentially for cumulative sum
- Contiguous `grads` enables SIMD vectorization
- `hess` and `counts` accessed separately when needed
- Merge (histogram += other) can use SIMD on contiguous arrays

### DD-3: SubAssign for Subtraction Trick

**Context**: How to implement `larger_child = parent - smaller_child`?

**Decision**: Use `SubAssign` (`target -= source`) instead of `Sub` (`result = a - b`).

**Rationale**:
- Avoids allocating a new histogram for the result
- Target is already acquired from pool
- Copy parent into target, then subtract smaller child in place
- Same pattern for AddAssign in RowParallel merge

### DD-4: LRU Pool Eviction

**Context**: What to do when pool is full?

**Decision**: LRU eviction—remove least-recently-used histogram.

**Rationale**:
- In tree growth, recently-used histograms are likely still needed (parent/sibling)
- Old histograms (from earlier tree levels) are safe to evict
- Alternative (FIFO) might evict still-needed histograms
- LRU tracks access via `acquire()` calls

### DD-5: Features-First Iteration

**Context**: Iterate rows-first or features-first during accumulation?

**Decision**: Features-first (even in RowParallel, each thread iterates features-first within its row chunk).

**Rationale**:
- Col-major QuantizedMatrix: `quantized.col(feat)` is contiguous
- Each feature's histogram bins updated together → better cache locality
- Gradients (`grads[row]`) accessed multiple times per row, likely in cache
- XGBoost uses rows-first with row-major; we use features-first with col-major

### DD-6: Accumulation Strategy as Enum

**Context**: How to select between Serial, FeatureParallel, RowParallel?

**Decision**: Enum with runtime selection based on data shape.

**Rationale**:
- Strategy depends on n_rows, n_features, n_bins at runtime
- Enum allows exhaustive matching
- Auto-selection heuristic:
  - RowParallel: n_features ≤ 64, n_rows > 10k
  - FeatureParallel: histogram > L2 cache, n_features ≥ 4 × n_threads
  - Serial: otherwise

## Future Optimizations

The design supports these optimizations for future implementation:

### Cache-Sized Blocking (Multi-Node)

For DepthWise expansion, process multiple nodes' histograms in cache-friendly blocks:

```text
// Instead of: for each node, for each row
// Do: for each (node, row_block) pair sized for L2 cache

block_size = 2048  // Tuned for L2

for (node_idx, row_start, row_end) in blocked_iteration(nodes, rows, block_size):
  partial_accumulate(histograms[node_idx], rows[row_start..row_end], ...)
```

**Design support**:

- `HistogramPool::acquire_batch(node_ids)` can provide multiple histograms
- Accumulation loop can be restructured without changing HistogramBuilder interface
- SplitParams and pool management remain unchanged

### SIMD Split Enumeration

SoA layout enables SIMD for cumulative sum:

```text
// Instead of scalar loop:
for bin in 0..n_bins:
  left_g += grads[bin]

// SIMD prefix sum on grads array
cumsum = simd_prefix_sum(grads)
```

**Design support**: `FeatureHistogram::grads()` returns contiguous slice, ready for SIMD.

### Quantized Gradient Histograms

When using gradient quantization (RFC-0008), histograms can use integer accumulators:

```rust
/// Integer histogram for quantized gradients.
pub struct IntHistogram {
    /// Packed gradient sums (i32 or i64 depending on expected count).
    grad_sums: Box<[i64]>,
    /// Packed hessian sums.
    hess_sums: Box<[i64]>,
    counts: Box<[u32]>,
}

impl IntHistogram {
    /// Accumulate quantized gradient (no floating point).
    #[inline]
    pub fn add_quantized(&mut self, bin: u16, packed_grad_hess: i16) {
        let grad = (packed_grad_hess & 0xFF) as i8 as i64;
        let hess = ((packed_grad_hess >> 8) & 0xFF) as i64;
        self.grad_sums[bin as usize] += grad;
        self.hess_sums[bin as usize] += hess;
        self.counts[bin as usize] += 1;
    }
    
    /// Convert to f32 for split finding.
    pub fn dequantize(&self, grad_scale: f32, hess_scale: f32) -> FeatureHistogram;
}
```

This avoids all floating point in the accumulation inner loop.

## References

- [XGBoost Histogram](https://github.com/dmlc/xgboost/blob/master/src/tree/hist/histogram.h)
- [LightGBM Histogram](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp)
```
