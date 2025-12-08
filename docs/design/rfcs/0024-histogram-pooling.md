# RFC-0024: Histogram Pooling

- **Status**: Superseded by RFC-0025
- **Created**: 2024-12-02
- **Updated**: 2024-12-02
- **Depends on**: RFC-0012 (Histogram Building)
- **Superseded by**: RFC-0025 (Row-Parallel Histogram Building with Advanced Pooling)
- **Scope**: Memory reuse optimization for histogram allocation during tree growing

> **Note**: This RFC has been superseded by RFC-0025, which combines histogram
> pooling with row-parallel building for better performance. The research in
> the Appendix informed RFC-0025's design.

## Summary

This RFC proposes a histogram pool that pre-allocates and reuses `NodeHistogram` instances
during tree growing. Instead of allocating new histograms for each split, the grower borrows
from a pool and returns them when done. This eliminates allocation overhead on the hot path.

## Motivation

### Current Behavior

The tree grower currently allocates a new `NodeHistogram` for every node:

```rust
fn build_histogram<B: BinIndex>(&self, rows: &[u32], ...) -> NodeHistogram {
    let mut hist = NodeHistogram::new(self.cuts);  // ← Allocation on hot path
    HistogramBuilder.build(&mut hist, ...);
    hist
}
```

For a tree with `L` leaves, we create `2L - 1` nodes, meaning `2L - 1` histogram allocations.
With 100 features × 256 bins, each histogram is ~300 KB. For `max_leaves = 128`:

- **255 allocations** × ~300 KB = **76.5 MB allocated per tree**
- This happens every boosting round (potentially 1000+ times)

### Allocation Cost

While Rust's allocator is fast, the cost isn't zero:
- System call overhead for large allocations
- Memory fragmentation over many rounds
- Cache pollution from touching cold memory

More importantly, many histograms have **predictable lifetimes**:
- Parent histogram is only needed until both children are built
- After a node is expanded, its histogram is never used again

### The Opportunity

Since histograms are fixed-size (determined by bin cuts) and short-lived, we can:
1. Pre-allocate a pool of histograms at tree start
2. Borrow histograms instead of allocating
3. Return histograms to pool after use
4. Reset and reuse for the next split

## Design

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│  HistogramPool                                               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ Histogram 0 │ Histogram 1 │ Histogram 2 │     ...     │  │
│  │  (in use)   │  (in use)   │ (available) │ (available) │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
│                                                              │
│  available: [2, 3, 4, 5, ...]   (stack of free indices)     │
└─────────────────────────────────────────────────────────────┘

TreeGrower workflow:
━━━━━━━━━━━━━━━━━━━━

1. pool.acquire() → HistogramHandle(idx: 0)
2. Build histogram in pool.get_mut(handle)
3. Use histogram for split finding
4. When node children are done: pool.release(handle)
5. Histogram 0 is now available for next acquire()
```

### Core Types

```rust
/// Handle to a pooled histogram.
/// 
/// Zero-cost wrapper around pool index.
/// The handle does NOT auto-release — caller must explicitly release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistogramHandle(u32);

/// Pool of pre-allocated histograms for reuse.
/// 
/// # Lifecycle
/// 
/// ```ignore
/// let mut pool = HistogramPool::new(&cuts, capacity);
/// 
/// let h1 = pool.acquire().expect("pool exhausted");
/// let hist1 = pool.get_mut(h1);
/// // ... build and use hist1 ...
/// pool.release(h1);  // Now available for reuse
/// ```
pub struct HistogramPool {
    /// All pre-allocated histograms
    histograms: Vec<NodeHistogram>,
    
    /// Stack of available histogram indices
    available: Vec<u32>,
    
    /// Track which handles are currently in use (debug builds)
    #[cfg(debug_assertions)]
    in_use: std::collections::HashSet<u32>,
}

impl HistogramPool {
    /// Create a pool with `capacity` pre-allocated histograms.
    /// 
    /// Each histogram is sized according to `cuts`.
    pub fn new(cuts: &BinCuts, capacity: usize) -> Self;
    
    /// Acquire a histogram from the pool.
    /// 
    /// Returns `None` if pool is exhausted.
    /// The histogram is reset before returning.
    pub fn acquire(&mut self) -> Option<HistogramHandle>;
    
    /// Get immutable reference to a histogram.
    pub fn get(&self, handle: HistogramHandle) -> &NodeHistogram;
    
    /// Get mutable reference to a histogram.
    pub fn get_mut(&mut self, handle: HistogramHandle) -> &mut NodeHistogram;
    
    /// Release a histogram back to the pool.
    /// 
    /// # Panics
    /// 
    /// Panics in debug builds if handle is invalid or already released.
    pub fn release(&mut self, handle: HistogramHandle);
    
    /// Number of histograms currently in use.
    pub fn in_use(&self) -> usize;
    
    /// Number of histograms available.
    pub fn available(&self) -> usize;
    
    /// Total pool capacity.
    pub fn capacity(&self) -> usize;
}
```

### Pool Sizing

The required pool size depends on the growth strategy:

**Depth-wise (level-by-level):**
- At depth `d`, there are at most `2^d` nodes
- But we process one level at a time, so we need histograms for:
  - Current level nodes (being split)
  - Child candidates (just built)
- Maximum: `2^(max_depth + 1)` histograms

**Leaf-wise (best-first):**
- We maintain a priority queue of candidate nodes
- Each candidate needs its histogram until selected for splitting
- Maximum: `max_leaves` histograms (at most one per potential leaf)

**With histogram subtraction:**
- Parent histogram needed while building children
- Reduces by roughly half with subtraction optimization

**Recommended sizing:**
```rust
fn recommended_pool_size(max_depth: u32, max_leaves: u32, growth_mode: GrowthMode) -> usize {
    match growth_mode {
        GrowthMode::DepthWise => {
            // Current level + next level + some margin
            2usize.pow(max_depth.min(8)) * 3
        }
        GrowthMode::LeafWise => {
            // Active candidates + margin for subtraction
            (max_leaves as usize) * 2
        }
    }
}
```

### Integration with TreeGrower

```rust
impl TreeGrower {
    /// Build histograms for both children of a split.
    /// 
    /// Uses histogram subtraction: builds smaller child, derives larger.
    fn build_child_histograms(
        &mut self,
        pool: &mut HistogramPool,
        parent_handle: HistogramHandle,
        left_rows: &[u32],
        right_rows: &[u32],
        quantized: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
    ) -> (HistogramHandle, HistogramHandle) {
        let parent_hist = pool.get(parent_handle);
        
        // Acquire handles for children
        let left_handle = pool.acquire().expect("histogram pool exhausted");
        let right_handle = pool.acquire().expect("histogram pool exhausted");
        
        if left_rows.len() <= right_rows.len() {
            // Build left (smaller), derive right via subtraction
            let left_hist = pool.get_mut(left_handle);
            self.histogram_builder.build(left_hist, quantized, grads, hess, left_rows);
            
            // Derive right: right = parent - left
            let right_hist = pool.get_mut(right_handle);
            right_hist.copy_from(parent_hist);
            *right_hist -= pool.get(left_handle);
        } else {
            // Build right (smaller), derive left via subtraction
            let right_hist = pool.get_mut(right_handle);
            self.histogram_builder.build(right_hist, quantized, grads, hess, right_rows);
            
            // Derive left: left = parent - right
            let left_hist = pool.get_mut(left_handle);
            left_hist.copy_from(parent_hist);
            *left_hist -= pool.get(right_handle);
        }
        
        // Parent histogram no longer needed — release it
        pool.release(parent_handle);
        
        (left_handle, right_handle)
    }
}
```

### Memory Overhead

The pool trades allocation overhead for memory overhead:

| Scenario | Without Pool | With Pool |
|----------|--------------|-----------|
| Memory used | Peak during tree | Fixed upfront |
| Allocations per tree | O(nodes) | 0 |
| Memory layout | Scattered | Contiguous |
| Cache behavior | Cold allocations | Warm reuse |

For depth-8 tree (~256 leaves, ~511 nodes):
- Without pool: 511 × 300 KB = 153 MB allocated (then freed)
- With pool (size=64): 64 × 300 KB = 19.2 MB fixed

The pool is smaller because histograms are released as we go.

## Design Decisions

### DD-1: Explicit vs RAII Release

**Context**: Should histogram handles auto-release when dropped?

**Options considered**:

1. **RAII guard**: `HistogramGuard` that releases on drop
2. **Explicit release**: Caller must call `pool.release(handle)`

**Decision**: Explicit release.

**Rationale**:

- Pool requires `&mut self` for release, but we often hold multiple handles
- RAII would require `RefCell` or similar for interior mutability
- Explicit is clearer about when memory becomes available
- Debug assertions catch forgotten releases

### DD-2: Handle Validity Checking

**Context**: How to handle invalid/double-release of handles.

**Options considered**:

1. **Unchecked**: Undefined behavior on misuse
2. **Debug-only checks**: Assertions in debug builds
3. **Always checked**: Runtime validation with Result types

**Decision**: Debug-only checks with assertions.

**Rationale**:

- Correct code never double-releases or uses invalid handles
- Debug assertions catch bugs during development
- Zero overhead in release builds
- Result types would complicate API for no benefit in correct code

### DD-3: Pool Growth Policy

**Context**: What happens when pool is exhausted?

**Options considered**:

1. **Fixed size**: Return `None`, caller handles fallback
2. **Auto-grow**: Allocate more histograms on demand
3. **Panic**: Pool exhaustion is a bug

**Decision**: Fixed size with `Option` return.

**Rationale**:

- Pool size is predictable from tree parameters
- Auto-grow defeats the purpose (back to allocation on hot path)
- Caller can fall back to direct allocation if needed
- Clear signal when sizing is wrong

### DD-4: Thread Safety

**Context**: Should the pool support concurrent access?

**Options considered**:

1. **Single-threaded**: `&mut self` for all operations
2. **Thread-safe**: `Mutex` or atomic operations
3. **Per-thread pools**: Each thread has own pool

**Decision**: Single-threaded for now.

**Rationale**:

- Current tree growing processes one node at a time (sequential)
- **Per-feature parallelism** (RFC-0012 DD-4) parallelizes *within* a single histogram
  build, not across histogram allocations — this doesn't need pool synchronization
- The pool is used at node boundaries (acquire/release), which happen sequentially
- **Per-row parallelism** (RFC-0012 future work) would need per-thread scratch
  histograms, not a shared pool — each thread builds partial histogram, then merge
- If we add **depth-wise node parallelism** (build multiple nodes at same level
  concurrently), per-thread pools would be more efficient than locking

**Compatibility with existing parallelism:**

| Parallelism Type | Status | Pool Impact |
|------------------|--------|-------------|
| Per-feature (build_parallel) | ✅ Implemented | None — single histogram, features partitioned |
| Per-row (future) | ❌ Not implemented | Per-thread scratch, not shared pool |
| Depth-wise node parallel | ❌ Not implemented | Would need per-thread pools |

## Integration

| Component | Integration Point | Notes |
|-----------|-------------------|-------|
| RFC-0012 (Histogram Building) | `NodeHistogram` | Pooled type |
| RFC-0015 (Tree Growing) | `TreeGrower` | Uses pool for child histograms |
| Training loop | Per-tree setup | Create pool at tree start |

### Migration Path

1. Add `HistogramPool` as new type
2. Add pool parameter to `TreeGrower::build_child_histograms`
3. Update `GBTreeTrainer` to create pool per tree
4. Benchmark to confirm improvement

The change is internal to training — no public API impact.

## Open Questions

1. **Pool per tree vs per round?**
   - Per tree: Simpler, clear lifecycle
   - Per round: Amortize across trees in ensemble
   - Recommend: Start with per-tree, measure allocation overhead

2. **Histogram recycling priority?**
   - LIFO (stack): Most recently freed is warmest in cache
   - FIFO (queue): Fair distribution of cache coldness
   - Recommend: LIFO for cache locality

3. **Pool for parent histograms too?**
   - Current design pools child histograms
   - Parent histograms could also be pooled
   - Recommend: Pool everything, release parent when children done

## Future Work

- [ ] Per-thread pool variant for depth-wise node parallelism (if implemented)
- [ ] Per-round pool sharing across trees
- [ ] Adaptive pool sizing based on actual usage
- [ ] Pool statistics/metrics for tuning

**Note**: Per-row parallelism (RFC-0012 future work) doesn't need pool changes —
it uses per-thread scratch histograms with merge, orthogonal to node-level pooling.

## References

- [XGBoost HistogramPool](https://github.com/dmlc/xgboost/blob/master/src/common/hist_util.h)
- [LightGBM histogram memory management](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp)
- RFC-0012: Histogram Building

---

## Appendix: Research on XGBoost and LightGBM

This section documents how XGBoost and LightGBM handle histogram pooling and
parallelism, to inform our design decisions.

### LightGBM's HistogramPool

LightGBM implements a sophisticated histogram pool in `feature_histogram.hpp`:

```cpp
class HistogramPool {
  std::vector<std::unique_ptr<FeatureHistogram[]>> pool_;
  std::vector<std::vector<hist_t, AlignmentAllocator>> data_;
  int cache_size_;      // Max histograms to keep in memory
  int total_size_;      // Total nodes in tree (max_leaves)
  bool is_enough_;      // cache_size >= total_size
  std::vector<int> mapper_;          // node_id -> pool_slot
  std::vector<int> inverse_mapper_;  // pool_slot -> node_id  
  std::vector<int> last_used_time_;  // LRU eviction
};
```

**Key design points:**

1. **LRU eviction**: When cache is smaller than tree size, uses LRU to evict
   least-recently-used histograms. Controlled by `histogram_pool_size` parameter.

2. **Dynamic sizing**: Pool is sized based on `histogram_pool_size` (MB) or `num_leaves`:

   ```cpp
   if (config_->histogram_pool_size <= 0) {
       max_cache_size = config_->num_leaves;
   } else {
       max_cache_size = histogram_pool_size_mb / total_histogram_size;
   }
   max_cache_size = max(2, min(max_cache_size, num_leaves));
   ```

3. **Move semantics**: Supports `Move(src_idx, dst_idx)` to reassign histogram ownership
   without copying data.

4. **Pre-allocated backing store**: All histogram data is allocated upfront in
   contiguous aligned vectors (`data_`).

### XGBoost's BoundedHistCollection

XGBoost uses a simpler bounded cache in `hist_cache.h`:

```cpp
class BoundedHistCollection {
  std::map<bst_node_t, std::size_t> node_map_;  // node -> offset
  std::unique_ptr<Vec> data_;                    // contiguous storage
  bst_bin_t n_total_bins_;
  std::size_t max_cached_nodes_;
  bool has_exceeded_;  // tree grew beyond cache
};
```

**Key design points:**

1. **Contiguous allocation**: All histograms in one buffer for efficient allreduce
   in distributed mode.

2. **Batch allocation**: `AllocateHistograms(nodes_to_build, nodes_to_sub)` allocates
   for multiple nodes at once.

3. **Cache overflow handling**: When `has_exceeded_`, caller must rearrange nodes
   before allocating (can't rely on parent histograms existing).

4. **No eviction**: Unlike LightGBM, XGBoost clears entire cache when exceeded rather
   than LRU eviction.

### Parallelism Strategies

Both frameworks use **row-parallel histogram building** with thread-local scratch
and reduction.

#### XGBoost's ParallelGHistBuilder

```cpp
class ParallelGHistBuilder {
  HistCollection hist_buffer_;              // Thread-local scratch histograms
  std::vector<bool> threads_to_nids_map_;   // Which threads work on which nodes
  std::vector<GHistRow> targeted_hists_;    // Final output histograms
  std::map<pair<tid, nid>, int> tid_nid_to_hist_;  // Thread+node -> scratch slot
};
```

**How it works:**

1. Rows are partitioned across threads
2. Each thread builds partial histogram in thread-local scratch
3. `ReduceHist(nid, begin, end)` merges thread-local histograms into final

**Pool interaction**: The `ParallelGHistBuilder` is separate from `BoundedHistCollection`.
The cache stores final histograms; the builder uses its own scratch space.

#### LightGBM's Approach

LightGBM's `ConstructHistograms` operates differently:

1. **Column-wise or row-wise**: Decided at init based on data shape
2. **Per-feature parallelism**: `#pragma omp parallel for` over features for subtraction
3. **Row-wise builds**: Use ordered gradients with prefetching

**Pool interaction**: The `HistogramPool` stores node histograms. Parallel building
happens on the histogram data directly (no separate scratch).

### Histogram Subtraction in Both Frameworks

Both use the subtraction optimization but handle memory differently:

**LightGBM** (in-place subtraction):

```cpp
// larger_leaf = parent - smaller_leaf (modifies larger_leaf in place)
larger_leaf_histogram_array_[feature_index].Subtract(
    smaller_leaf_histogram_array_[feature_index]);
```

**XGBoost** (separate buffers, then copy):

```cpp
// SubtractionHist(dst, src1, src2) computes dst = src1 - src2
common::SubtractionHist(hist_[child_nid], hist_[parent_nid], 
                        hist_[sibling_nid], begin, end);
```

### Implications for Our Design

| Aspect | LightGBM | XGBoost | Our RFC |
|--------|----------|---------|---------|
| Pool sizing | MB-based or num_leaves | Bounded by max_cached_nodes | Fixed capacity |
| Eviction | LRU | Clear all | None (fixed size) |
| Thread-local scratch | No | Yes (ParallelGHistBuilder) | Not needed for per-feature |
| Contiguous storage | Yes (aligned) | Yes | Yes |
| Subtraction | In-place | Into separate buffer | Consuming Sub |

**Key insights:**

1. **Per-feature parallelism doesn't need pool changes**: Both frameworks parallelize
   feature iteration, not histogram allocation. Our DD-4 is correct.

2. **Row-parallel does need scratch**: XGBoost's `ParallelGHistBuilder` shows that
   row-parallel building requires per-thread scratch histograms + merge. This is
   separate from the node-level cache.

3. **LRU may be overkill**: XGBoost's simpler "clear all when exceeded" approach
   works well. We can start with fixed-size pool.

4. **Contiguous storage matters**: Both use contiguous backing storage. Important
   for cache locality and (in distributed mode) efficient allreduce.

### Recommendation Updates

Based on this research:

1. **Keep DD-4 (single-threaded)**: Per-feature parallelism works with our design.
   Row-parallel would need separate scratch space, not pool changes.

2. **Consider LRU for large trees**: If we support very deep trees where pool
   exhaustion is common, LRU eviction (like LightGBM) is better than failing.
   But for typical use cases, fixed size with good initial sizing suffices.

3. **Contiguous backing store**: Ensure our pool uses contiguous allocation for
   all histograms, not individual `Vec`s per histogram.

## Changelog

- 2024-12-02: Initial draft
- 2024-12-02: DD-4 clarified — per-feature parallelism is orthogonal to pool design
- 2024-12-02: Added Appendix with XGBoost/LightGBM histogram pooling research
- 2024-12-02: Superseded by RFC-0025 (row-parallel with advanced pooling)
