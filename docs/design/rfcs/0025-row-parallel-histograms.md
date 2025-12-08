# RFC-0025: Row-Parallel Histogram Building with Advanced Pooling

- **Status**: Approved
- **Created**: 2024-12-02
- **Updated**: 2024-12-03
- **Depends on**: RFC-0012 (Histogram Building), RFC-0024 (Histogram Pooling)
- **Scope**: Parallel histogram construction with memory pooling for high-performance training

## Summary

This RFC designs a unified system for parallel histogram building that combines:

1. **Row-parallel histogram construction** — partition rows across threads, merge partial histograms
2. **Advanced histogram pooling** — LRU eviction, contiguous storage, node-indexed access
3. **Gradient reordering** — cache-friendly gradient access patterns
4. **Multi-node batch building** — build histograms for multiple nodes simultaneously

The goal is to match or exceed XGBoost and LightGBM performance while maintaining
Rust's safety guarantees.

## Motivation

### Current State

RFC-0012 implemented per-feature parallelism for histogram building:

```rust
// Current: parallelize over features within one histogram
rayon::scope(|s| {
    for feature_chunk in features.chunks(chunk_size) {
        s.spawn(|_| build_feature_histograms(feature_chunk, rows, grads));
    }
});
```

This works well for wide datasets (many features) but leaves performance on the
table for tall datasets (many rows, fewer features).

### The Opportunity

Both XGBoost and LightGBM use row-parallel histogram building:

- **XGBoost**: `ParallelGHistBuilder` with per-thread scratch + `ReduceHist()` merge
- **LightGBM**: Ordered gradients with per-feature-group parallelism

For a node with 100K rows and 100 features:

| Strategy | Work Distribution | Memory Traffic |
|----------|-------------------|----------------|
| Per-feature | 100 parallel tasks, each reads 100K rows | 100 × 100K reads |
| Per-row | N threads, each reads 100K/N rows | 100K reads total |

Row-parallel reduces total memory traffic by factor of `num_features`.

### Combined Design Goals

1. **Row-parallel building**: Partition rows, not features
2. **Efficient pooling**: Minimize allocation, maximize cache reuse
3. **LRU eviction**: Handle deep trees without exhausting memory
4. **Contiguous storage**: Enable efficient SIMD and future allreduce
5. **Batch operations**: Build multiple node histograms in one pass

## Design

### Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HistogramSystem                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    ContiguousHistogramPool                             │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Backing Store: (Box<[f32]>, Box<[f32]>, Box<[u32]>)            │  │  │
│  │  │  grads: [hist0_bins][hist1_bins][hist2_bins]...                 │  │  │
│  │  │  hess:  [hist0_bins][hist1_bins][hist2_bins]...                 │  │  │
│  │  │  count: [hist0_bins][hist1_bins][hist2_bins]...                 │  │  │
│  │  │            ↑           ↑           ↑                            │  │  │
│  │  │          node0       node2       free                           │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                        │  │
│  │  Node Map: HashMap<NodeId, SlotId>     LRU: VecDeque<SlotId>          │  │
│  │  Free List: Vec<SlotId>                Stats: PoolMetrics             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    RowParallelScratch                                  │  │
│  │  Thread 0: [scratch_hist]                                             │  │
│  │  Thread 1: [scratch_hist]                                             │  │
│  │  Thread 2: [scratch_hist]                                             │  │
│  │  ...                                                                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    OrderedGradientBuffer                               │  │
│  │  Pre-sorted gradients for cache-friendly access                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

Build Flow:
━━━━━━━━━━━

1. Acquire target histogram slot from pool (may evict via LRU)
2. Partition rows across threads
3. Each thread builds into RowParallelScratch
4. Reduce: merge thread-local histograms into target
5. Mark slot as recently used (update LRU)
```

### Core Types

#### ContiguousHistogramPool

```rust
/// Pool of pre-allocated histograms in contiguous memory.
///
/// Uses SoA (Structure-of-Arrays) layout matching our existing `FeatureHistogram`:
/// separate contiguous arrays for gradients, hessians, and counts.
///
/// Features:
/// - Single allocation for all histograms (cache-friendly)
/// - LRU eviction when pool is exhausted
/// - O(1) access by node ID via HashMap
/// - Metrics tracking for tuning
pub struct ContiguousHistogramPool {
    /// Contiguous backing store for gradient sums.
    /// Layout: [hist0_bins][hist1_bins][hist2_bins]...
    sum_grads: Box<[f32]>,
    
    /// Contiguous backing store for hessian sums.
    sum_hess: Box<[f32]>,
    
    /// Contiguous backing store for sample counts.
    counts: Box<[u32]>,
    
    /// Number of bins per histogram (same for all).
    bins_per_hist: usize,
    
    /// Total number of histogram slots.
    capacity: usize,
    
    /// Maps node ID to slot index.
    node_to_slot: HashMap<NodeId, SlotId>,
    
    /// Maps slot index back to node ID (for eviction).
    slot_to_node: Vec<Option<NodeId>>,
    
    /// Free slots (LIFO for cache warmth).
    free_slots: Vec<SlotId>,
    
    /// LRU tracking: most recently used at back.
    /// Only contains slots that are in use.
    lru_order: VecDeque<SlotId>,
    
    /// Position in LRU queue for O(1) removal.
    slot_to_lru_pos: Vec<Option<usize>>,
    
    /// Usage statistics.
    metrics: PoolMetrics,
}

/// View into a single histogram slot (SoA layout).
///
/// Provides access to the three arrays for one histogram.
pub struct HistogramSlot<'a> {
    pub sum_grad: &'a [f32],
    pub sum_hess: &'a [f32],
    pub count: &'a [u32],
}

/// Mutable view into a single histogram slot.
pub struct HistogramSlotMut<'a> {
    pub sum_grad: &'a mut [f32],
    pub sum_hess: &'a mut [f32],
    pub count: &'a mut [u32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotId(u32);

#[derive(Debug, Default)]
pub struct PoolMetrics {
    pub hits: u64,           // Accessed existing histogram
    pub misses: u64,         // Needed to allocate new slot
    pub evictions: u64,      // LRU evictions
    pub peak_usage: usize,   // Maximum slots in use at once
}

impl ContiguousHistogramPool {
    /// Create pool with given capacity and histogram size.
    ///
    /// Total memory: `capacity * bins_per_hist * (2*sizeof(f32) + sizeof(u32))`
    ///             = `capacity * bins_per_hist * 12` bytes
    pub fn new(capacity: usize, bins_per_hist: usize) -> Self;
    
    /// Get or allocate histogram for a node.
    ///
    /// If node already has a histogram, returns it (cache hit).
    /// If pool has free slots, allocates one.
    /// If pool is full, evicts LRU node and reuses its slot.
    ///
    /// Returns mutable view to histogram arrays.
    pub fn get_or_allocate(&mut self, node: NodeId) -> HistogramSlotMut<'_>;
    
    /// Get existing histogram for node (if present).
    ///
    /// Does NOT allocate. Returns None if node not in pool.
    /// Updates LRU position (marks as recently used).
    pub fn get(&mut self, node: NodeId) -> Option<HistogramSlot<'_>>;
    
    /// Get mutable reference without LRU update.
    ///
    /// Used during histogram building (already "in use").
    pub fn get_mut_no_lru(&mut self, node: NodeId) -> Option<HistogramSlotMut<'_>>;
    
    /// Check if node has histogram in pool.
    pub fn contains(&self, node: NodeId) -> bool;
    
    /// Explicitly release a node's histogram.
    ///
    /// Use when node is fully expanded and histogram definitely not needed.
    /// More efficient than waiting for LRU eviction.
    pub fn release(&mut self, node: NodeId);
    
    /// Release multiple nodes at once (e.g., when tree level complete).
    pub fn release_batch(&mut self, nodes: &[NodeId]);
    
    /// Reset pool for new tree (keeps allocations).
    pub fn reset(&mut self);
    
    /// Get pool statistics.
    pub fn metrics(&self) -> &PoolMetrics;
    
    /// Number of histograms currently in use.
    pub fn in_use(&self) -> usize;
    
    /// Number of free slots.
    pub fn available(&self) -> usize;
}
```

#### RowParallelScratch

```rust
/// Per-thread scratch space for row-parallel histogram building.
///
/// Each thread accumulates gradients into its local scratch, then
/// results are merged into the target histogram.
///
/// Uses SoA layout matching our `FeatureHistogram`.
pub struct RowParallelScratch {
    /// Per-thread gradient sums: [thread][bins]
    sum_grads: Vec<Box<[f32]>>,
    
    /// Per-thread hessian sums: [thread][bins]
    sum_hess: Vec<Box<[f32]>>,
    
    /// Per-thread counts: [thread][bins]
    counts: Vec<Box<[u32]>>,
    
    /// Number of bins per histogram.
    bins_per_hist: usize,
    
    /// Number of threads.
    num_threads: usize,
}

/// View into a thread's scratch buffer.
pub struct ScratchSlotMut<'a> {
    pub sum_grad: &'a mut [f32],
    pub sum_hess: &'a mut [f32],
    pub count: &'a mut [u32],
}

impl RowParallelScratch {
    /// Create scratch space for `num_threads` threads.
    pub fn new(num_threads: usize, bins_per_hist: usize) -> Self;
    
    /// Get scratch buffer for current thread.
    ///
    /// # Safety
    /// Must only be called from within a parallel region where
    /// each thread uses a different index.
    pub fn get_buffer(&mut self, thread_id: usize) -> ScratchSlotMut<'_>;
    
    /// Reset all buffers to zero.
    pub fn reset_all(&mut self);
    
    /// Reset specific buffer to zero.
    pub fn reset(&mut self, thread_id: usize);
    
    /// Get all thread buffers for reduction.
    pub fn all_buffers(&self) -> impl Iterator<Item = (&[f32], &[f32], &[u32])>;
}
```

#### OrderedGradientBuffer

```rust
/// Gradient buffer with support for reordering.
///
/// LightGBM's key insight: when building histograms, reading gradients
/// in row order of the node (not original data order) is more cache-friendly.
pub struct OrderedGradientBuffer {
    /// Original gradients (indexed by original row ID).
    gradients: Box<[f32]>,
    hessians: Box<[f32]>,
    
    /// Reordered gradients for current node.
    /// `ordered_grad[i]` = gradient for i-th row in node's row list.
    ordered_gradients: Box<[f32]>,
    ordered_hessians: Box<[f32]>,
    
    /// Whether ordered buffers are valid.
    ordered_valid: bool,
}

impl OrderedGradientBuffer {
    /// Reorder gradients according to row indices.
    ///
    /// After this call, `ordered_gradients[i] = gradients[rows[i]]`.
    /// This is parallelized internally.
    pub fn reorder(&mut self, rows: &[u32]);
    
    /// Get ordered gradient slice (only valid after reorder).
    pub fn ordered_grads(&self) -> &[f32];
    pub fn ordered_hess(&self) -> &[f32];
    
    /// Get original gradients (always valid).
    pub fn original_grads(&self) -> &[f32];
    pub fn original_hess(&self) -> &[f32];
}
```

#### ParallelHistogramBuilder

```rust
/// Builds histograms using row-parallel strategy with thread-local scratch.
///
/// # Algorithm
///
/// 1. Partition rows across threads
/// 2. Each thread iterates its rows, accumulating into scratch histogram
/// 3. Reduce: sum all scratch histograms into target
///
/// This reduces memory traffic compared to per-feature parallelism because
/// each row's gradients are read only once (by one thread), not once per feature.
pub struct ParallelHistogramBuilder {
    scratch: RowParallelScratch,
    num_threads: usize,
}

impl ParallelHistogramBuilder {
    pub fn new(num_threads: usize, bins_per_hist: usize) -> Self;
    
    /// Build histogram for a single node.
    ///
    /// Rows are partitioned across threads. Each thread builds partial
    /// histogram in scratch space, then results are merged.
    pub fn build<B: BinIndex>(
        &mut self,
        target: HistogramSlotMut<'_>,
        quantized: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    );
    
    /// Build histogram using pre-ordered gradients (LightGBM style).
    ///
    /// Assumes `ordered_grads` and `ordered_hess` are already reordered
    /// to match `rows` order. More cache-friendly.
    pub fn build_ordered<B: BinIndex>(
        &mut self,
        target: HistogramSlotMut<'_>,
        quantized: &QuantizedMatrix<B>,
        ordered_grads: &[f32],
        ordered_hess: &[f32],
        num_rows: usize,
    );
    
    /// Build histograms for multiple nodes in single pass (XGBoost style).
    ///
    /// For depth-wise growth, can build all nodes at a level simultaneously.
    /// Uses 2D parallelism: (node, row_range).
    pub fn build_batch<B: BinIndex>(
        &mut self,
        pool: &mut ContiguousHistogramPool,
        nodes: &[(NodeId, &[u32])],  // (node_id, row_indices)
        quantized: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
    );
}
```

### Reduction Strategy

The critical operation is merging thread-local histograms (SoA layout):

```rust
/// Merge thread-local histograms into target.
///
/// SIMD-optimized: processes 8 f32s at a time for each array.
fn reduce_histograms(
    target_grad: &mut [f32],
    target_hess: &mut [f32],
    target_count: &mut [u32],
    thread_grads: &[&[f32]],
    thread_hess: &[&[f32]],
    thread_counts: &[&[u32]],
) {
    // Zero targets
    target_grad.fill(0.0);
    target_hess.fill(0.0);
    target_count.fill(0);
    
    // Sum all thread contributions (SoA: process each array separately)
    // This enables better SIMD since each array is homogeneous type
    for tg in thread_grads {
        for (t, s) in target_grad.iter_mut().zip(tg.iter()) {
            *t += *s;
        }
    }
    for th in thread_hess {
        for (t, s) in target_hess.iter_mut().zip(th.iter()) {
            *t += *s;
        }
    }
    for tc in thread_counts {
        for (t, s) in target_count.iter_mut().zip(tc.iter()) {
            *t += *s;
        }
    }
}
```

For many threads (>8), consider hierarchical reduction:

```text
Thread histograms:     [T0] [T1] [T2] [T3] [T4] [T5] [T6] [T7]
                         \  /     \  /     \  /     \  /
Level 1:                 [P0]     [P1]     [P2]     [P3]
                           \       /         \       /
Level 2:                    [Q0]               [Q1]
                              \                 /
Level 3 (target):              [Final Result]
```

### Pool Sizing Strategy

Combining insights from XGBoost and LightGBM:

```rust
/// Calculate recommended pool capacity.
///
/// # Strategy
///
/// - Base: 2 × max_leaves (need parent + both children during split)
/// - Buffer: +25% for histogram subtraction working space
/// - Minimum: 8 (even for very shallow trees)
/// - Cap: available memory / histogram_size
pub fn recommended_pool_capacity(
    max_leaves: usize,
    max_depth: Option<usize>,
    histogram_size_bytes: usize,
    available_memory_bytes: usize,
) -> usize {
    let base = if let Some(depth) = max_depth {
        // Depth-wise: need nodes at current + next level
        2usize.pow(depth as u32 + 1)
    } else {
        // Leaf-wise: need all candidate leaves + working space
        max_leaves * 2
    };
    
    let with_buffer = (base as f64 * 1.25) as usize;
    let memory_cap = available_memory_bytes / histogram_size_bytes;
    
    with_buffer.min(memory_cap).max(8)
}
```

### LRU Implementation Details

The LRU implementation must be O(1) for access updates:

```rust
impl ContiguousHistogramPool {
    /// Update LRU position when slot is accessed.
    fn touch_lru(&mut self, slot: SlotId) {
        if let Some(pos) = self.slot_to_lru_pos[slot.0 as usize] {
            // Remove from current position
            self.lru_order.remove(pos);
            // Update positions for slots that shifted
            for (i, &s) in self.lru_order.iter().enumerate().skip(pos) {
                self.slot_to_lru_pos[s.0 as usize] = Some(i);
            }
        }
        // Add to back (most recent)
        self.slot_to_lru_pos[slot.0 as usize] = Some(self.lru_order.len());
        self.lru_order.push_back(slot);
    }
    
    /// Evict least recently used slot.
    fn evict_lru(&mut self) -> SlotId {
        let slot = self.lru_order.pop_front().expect("LRU empty but pool full");
        let node = self.slot_to_node[slot.0 as usize].take().expect("slot not mapped");
        self.node_to_slot.remove(&node);
        self.slot_to_lru_pos[slot.0 as usize] = None;
        self.metrics.evictions += 1;
        slot
    }
}
```

**Note**: For very high throughput, consider using `intrusive_collections` crate
for O(1) doubly-linked list operations instead of `VecDeque`.

### Integration with Tree Grower

```rust
impl TreeGrower {
    /// Build child histograms with row-parallel + pooling.
    fn build_child_histograms_parallel(
        &mut self,
        parent_node: NodeId,
        left_node: NodeId,
        right_node: NodeId,
        left_rows: &[u32],
        right_rows: &[u32],
        quantized: &QuantizedMatrix<impl BinIndex>,
    ) {
        // Determine which child to build (smaller one)
        let (build_node, build_rows, sub_node) = if left_rows.len() <= right_rows.len() {
            (left_node, left_rows, right_node)
        } else {
            (right_node, right_rows, left_node)
        };
        
        // Reorder gradients for cache-friendly access
        self.grad_buffer.reorder(build_rows);
        
        // Build smaller child histogram
        let target = self.pool.get_or_allocate(build_node);
        self.builder.build_ordered(
            target,
            quantized,
            self.grad_buffer.ordered_grads(),
            self.grad_buffer.ordered_hess(),
            build_rows.len(),
        );
        
        // Derive larger child via subtraction
        let parent_hist = self.pool.get(parent_node)
            .expect("parent histogram not in pool");
        let build_hist = self.pool.get(build_node).unwrap();
        
        let sub_target = self.pool.get_or_allocate(sub_node);
        subtract_histograms(sub_target, parent_hist, build_hist);
        
        // Release parent (no longer needed)
        self.pool.release(parent_node);
    }
}
```

## Design Decisions

### DD-1: LRU vs Fixed-Size Pool

**Context**: RFC-0024 proposed fixed-size pool with explicit release.
XGBoost uses "clear all when exceeded", LightGBM uses LRU.

**Options considered**:

1. **Fixed-size with explicit release** (RFC-0024): Simple, requires careful lifecycle management
2. **Clear all when exceeded** (XGBoost): Simple, but wasteful for deep trees
3. **LRU eviction** (LightGBM): Automatic, handles deep trees gracefully

**Decision**: LRU eviction with explicit release as optimization.

**Rationale**:

- LRU handles unexpected tree shapes without failing
- Explicit release still useful when lifecycle is known (e.g., parent after children built)
- LRU overhead is minimal (~1 pointer update per access)
- For memory-constrained scenarios, LRU enables training deeper trees
- `histogram_pool_size` parameter can control memory budget

### DD-2: Contiguous vs Separate Allocations

**Context**: How to allocate histogram memory.

**Options considered**:

1. **Separate Vec per histogram**: Simple, flexible sizing
2. **Single contiguous buffer**: Better cache locality, enables SIMD

**Decision**: Single contiguous buffer (like XGBoost and LightGBM).

**Rationale**:

- Reduction iterates over all bins — contiguous is faster
- Single allocation instead of many small ones
- Required for efficient allreduce in future distributed mode
- Slight inflexibility (all histograms same size) acceptable

### DD-3: Row-Parallel with Thread-Local Scratch

**Context**: How to parallelize histogram building.

**Options considered**:

1. **Per-feature parallelism** (current): Good for wide data, poor for tall
2. **Atomic accumulation**: No scratch needed, but contention on bins
3. **Thread-local scratch + reduce**: More memory, but no contention

**Decision**: Thread-local scratch with merge (like XGBoost's `ParallelGHistBuilder`).

**Rationale**:

- Atomic updates cause severe contention on hot bins
- Extra memory (threads × histogram_size) is acceptable
- Reduction is fast (O(bins), not O(rows))
- Matches proven XGBoost approach

### DD-4: Gradient Reordering

**Context**: Should we reorder gradients before histogram building?

**Options considered**:

1. **Original order**: Random access to gradient buffer
2. **Reordered (LightGBM style)**: Sequential access to ordered buffer

**Decision**: Support both, default to reordered for large nodes.

**Rationale**:

- Reordering cost: O(rows) copies
- Benefit: Sequential access pattern in hot loop
- Threshold: ~1000 rows makes reordering worthwhile
- LightGBM's benchmarks show significant wins

### DD-5: Batch vs Single-Node Building

**Context**: Build one node at a time or multiple?

**Options considered**:

1. **Single node**: Simpler, good for leaf-wise growth
2. **Batch (XGBoost style)**: Build all nodes at level, better parallelism

**Decision**: Support both, with batch for depth-wise growth.

**Rationale**:

- Depth-wise growth naturally has multiple independent nodes
- Batch building amortizes thread synchronization overhead
- `ParallelFor2d` over (node, row_range) maximizes parallelism
- Single-node mode still useful for leaf-wise/best-first

### DD-6: Histogram Data Layout

**Context**: How to store gradient/hessian/count in histograms.

**Options considered**:

1. **SoA** (separate arrays): `[g0,g1,g2,...], [h0,h1,h2,...], [c0,c1,c2,...]`
2. **AoS** (interleaved): `[(g0,h0,c0), (g1,h1,c1), ...]`
3. **Packed** (LightGBM quantized): 16-bit integers

**Decision**: SoA (matching existing `FeatureHistogram` and `GradientBuffer`).

**Rationale**:

- **Consistency**: Matches our existing `FeatureHistogram` (separate `sum_grad`, `sum_hess`, `count` arrays)
- **Consistency**: Matches our `GradientBuffer` SoA design (column-major for outputs)
- **SIMD-friendly**: Reduction can process homogeneous f32 arrays with full SIMD width
- **Flexible access**: Split finding can iterate just gradients without touching hessians
- **Cache efficiency**: When only summing gradients, no cache pollution from hessians
- Packed (16-bit) is a P2 optimization, orthogonal to layout choice

## Performance Expectations

Based on XGBoost/LightGBM benchmarks and our analysis:

| Dataset Shape | Current (per-feature) | Row-Parallel | Expected Speedup |
|---------------|----------------------|--------------|------------------|
| 100K rows × 100 features | Baseline | Lower traffic | 2-4× |
| 1M rows × 50 features | Baseline | Much lower traffic | 3-5× |
| 10K rows × 1000 features | Baseline | Similar | 1-1.5× |

**Memory overhead**: `num_threads × histogram_size` for scratch buffers.
For 8 threads, 100 features, 256 bins per feature:

- Per histogram: 256 bins × (4 + 4 + 4) bytes = 3KB per feature × 100 features = 300KB
- Total scratch: 8 threads × 300KB = 2.4MB

## Integration

| Component | Integration Point | Notes |
|-----------|-------------------|-------|
| RFC-0012 | `NodeHistogram`, `HistogramBuilder` | Enhanced with row-parallel |
| RFC-0024 | `HistogramPool` | Replaced by `ContiguousHistogramPool` |
| RFC-0015 | `TreeGrower` | Uses new parallel builder |
| Training loop | Per-tree setup | Configure pool size |

### Migration from RFC-0024

RFC-0024's `HistogramPool` is superseded by `ContiguousHistogramPool`:

| RFC-0024 | RFC-0025 |
|----------|----------|
| `HistogramPool::new(cuts, capacity)` | `ContiguousHistogramPool::new(capacity, bins)` |
| `pool.acquire() -> Option<Handle>` | `pool.get_or_allocate(node) -> &mut [...]` |
| `pool.get(handle)` | `pool.get(node)` |
| `pool.release(handle)` | `pool.release(node)` |
| Fixed capacity, no eviction | LRU eviction when full |

## Open Questions

1. **Hierarchical reduction threshold?**
   - For N threads, when is tree reduction faster than linear sum?
   - Likely N > 8-16, need benchmarks

2. **Adaptive strategy selection?**
   - Auto-switch between per-feature and row-parallel based on data shape?
   - Threshold: `rows / features > X`?

3. **NUMA awareness?**
   - For multi-socket systems, pin threads + use local memory?
   - Future work, but design should not preclude it

4. **Gradient quantization integration?**
   - LightGBM's 16-bit packed gradients
   - Orthogonal to this RFC, but layout should support it

## Strategy Selection and Code Architecture

### Where Strategy Switching Occurs

The strategy selection happens at **two levels**:

1. **Per-feature vs Row-parallel** (macro-level): Selected in `TreeGrower.build_histogram()`
2. **Sequential vs Parallel within strategy** (micro-level): Selected in builder methods

```text
┌─────────────────────────────────────────────────────────────────┐
│                        TreeGrower                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  build_histogram(node, rows) -> NodeHistogram             │  │
│  │    ├─ if rows.len() < MIN_ROWS_FOR_PARALLEL:              │  │
│  │    │     sequential_build()                               │  │
│  │    ├─ else if features > FEATURE_PARALLEL_THRESHOLD:      │  │
│  │    │     per_feature_parallel()    // Current approach    │  │
│  │    └─ else:                                               │  │
│  │          row_parallel()            // This RFC            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Heuristic for Strategy Selection

```rust
impl TreeGrower {
    fn select_histogram_strategy(&self, num_rows: usize) -> HistogramStrategy {
        // Small node: sequential is fastest
        if num_rows < self.params.min_rows_for_parallel {
            return HistogramStrategy::Sequential;
        }
        
        let num_features = self.cuts.num_features();
        let num_threads = rayon::current_num_threads();
        
        // Heuristic: row-parallel wins when work per feature is low
        // but total work (rows × features) is high enough to parallelize
        let rows_per_feature = num_rows / num_features;
        
        if num_features >= num_threads && rows_per_feature < 1000 {
            // Many features, few rows per feature: feature-parallel
            HistogramStrategy::FeatureParallel
        } else {
            // Few features or many rows: row-parallel wins
            HistogramStrategy::RowParallel
        }
    }
}

enum HistogramStrategy {
    Sequential,
    FeatureParallel,  // Current HistogramBuilder::build_parallel
    RowParallel,      // This RFC: ParallelHistogramBuilder
}
```

### TreeBuildParams Integration

```rust
pub struct TreeBuildParams {
    // ... existing fields ...
    
    /// Minimum rows in a node to use parallel histogram building.
    /// Below this threshold, sequential building is used.
    /// Default: 1024
    pub min_rows_for_parallel: usize,
    
    /// Strategy for parallelizing histogram building.
    /// Default: Auto (uses heuristic based on data shape)
    pub histogram_parallel_strategy: ParallelStrategy,
}

pub enum ParallelStrategy {
    /// Automatically select based on data shape
    Auto,
    /// Parallelize across features (current approach)
    FeatureParallel,
    /// Parallelize across rows (this RFC)
    RowParallel,
    /// Always use sequential (for benchmarking/debugging)
    Sequential,
}
```

### Code Architecture to Avoid Duplication

The core histogram accumulation logic is the same regardless of parallelism strategy:

```text
for each row in row_range:
    for each feature:
        bin = quantized[row, feature]
        hist[feature].sum_grads[bin] += grads[row]
        hist[feature].sum_hess[bin] += hess[row]
        hist[feature].counts[bin] += 1
```

We factor this into a **common build kernel** that both strategies use:

```rust
/// Trait for histogram accumulation - shared between strategies
trait HistogramAccumulator {
    /// Accumulate gradients from a row range into the histogram.
    /// This is the inner kernel used by all strategies.
    fn accumulate_range<B: BinIndex>(
        &mut self,
        quantized: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    );
}

// Both NodeHistogram and thread-local scratch implement this
impl HistogramAccumulator for NodeHistogram { ... }
impl HistogramAccumulator for RowParallelScratch { ... }
```

**Architecture for avoiding duplication**:

```text
┌──────────────────────────────────────────────────────────────────┐
│                     Shared Core (src/training/gbtree/histogram/) │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  accumulate_kernel.rs                                      │  │
│  │  - fn accumulate_row_range(hist, quantized, grads, hess)   │  │
│  │  - #[inline(always)] for loop unrolling                    │  │
│  │  - Same code path for all strategies                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  builder.rs (existing HistogramBuilder)                    │  │
│  │  - build_sequential(): sequential, uses accumulate_kernel  │  │
│  │  - build_feature_parallel(): feature-parallel              │  │
│  │    - splits features across threads                        │  │
│  │    - each thread calls accumulate_kernel for its features  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  parallel_builder.rs (new - this RFC)                      │  │
│  │  - build_row_parallel(): row-parallel                      │  │
│  │    - splits rows across threads                            │  │
│  │    - each thread has scratch histogram                     │  │
│  │    - uses SAME accumulate_kernel                           │  │
│  │    - merge step: reduce scratch → final                    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**Key insight**: The only difference between strategies is:

- **Feature-parallel**: One thread owns all rows for a subset of features
- **Row-parallel**: One thread owns all features for a subset of rows

The inner loop is identical — just the outer parallelization differs.

### Concrete Module Structure

```text
src/training/gbtree/histogram/
├── mod.rs               # Re-exports, HistogramBuilder facade
├── types.rs             # FeatureHistogram, NodeHistogram, etc.
├── kernel.rs            # NEW: Core accumulate_row_range() function
├── builder.rs           # Sequential + per-feature parallel (uses kernel)
├── parallel_builder.rs  # NEW: Row-parallel builder (uses same kernel)
├── pool.rs              # NEW: ContiguousHistogramPool
└── scratch.rs           # NEW: RowParallelScratch, reduction
```

### Unified Builder Interface

```rust
/// Unified histogram builder that selects strategy automatically
pub struct HistogramBuilder {
    cuts: Arc<BinCuts>,
    params: HistogramBuildParams,
    
    // Strategy-specific state (lazily initialized)
    scratch: Option<RowParallelScratch>,
}

impl HistogramBuilder {
    /// Build histogram, automatically selecting best strategy
    pub fn build<B: BinIndex>(
        &mut self,
        hist: &mut NodeHistogram,
        quantized: &QuantizedMatrix<B>,
        grads: &[f32],
        hess: &[f32],
        rows: &[u32],
    ) {
        match self.select_strategy(rows.len()) {
            Strategy::Sequential => self.build_sequential(hist, quantized, grads, hess, rows),
            Strategy::FeatureParallel => self.build_feature_parallel(hist, quantized, grads, hess, rows),
            Strategy::RowParallel => self.build_row_parallel(hist, quantized, grads, hess, rows),
        }
    }
    
    // Private methods share kernel::accumulate_row_range
    fn build_sequential(...) { 
        kernel::accumulate_row_range(hist, quantized, grads, hess, rows);
    }
    
    fn build_feature_parallel(...) {
        hist.features.par_iter_mut().for_each(|feat_hist| {
            kernel::accumulate_feature(feat_hist, quantized, grads, hess, rows, feat_idx);
        });
    }
    
    fn build_row_parallel(...) {
        let scratch = self.get_or_init_scratch();
        rows.par_chunks(ROWS_PER_THREAD).enumerate().for_each(|(tid, chunk)| {
            kernel::accumulate_row_range(&mut scratch[tid], quantized, grads, hess, chunk);
        });
        scratch.reduce_into(hist);
    }
}
```

## Future Work

- [ ] SIMD-optimized reduction kernel
- [ ] Hierarchical reduction for many threads
- [ ] Adaptive per-feature vs row-parallel selection
- [ ] 16-bit packed gradient support
- [ ] NUMA-aware thread/memory placement
- [ ] GPU histogram building (separate RFC)

## References

- [XGBoost `ParallelGHistBuilder`](https://github.com/dmlc/xgboost/blob/master/src/common/hist_util.h)
- [XGBoost `HistogramBuilder`](https://github.com/dmlc/xgboost/blob/master/src/tree/hist/histogram.h)
- [XGBoost `BoundedHistCollection`](https://github.com/dmlc/xgboost/blob/master/src/tree/hist/hist_cache.h)
- [LightGBM `HistogramPool`](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp)
- [LightGBM ordered gradients](https://github.com/microsoft/LightGBM/blob/master/src/io/dataset.cpp)
- RFC-0012: Histogram Building
- RFC-0024: Histogram Pooling

## Appendix: XGBoost vs LightGBM Detailed Comparison

### Memory Management

| Aspect | XGBoost | LightGBM | This RFC |
|--------|---------|----------|----------|
| Pool structure | `BoundedHistCollection` with map | `HistogramPool` with LRU | Contiguous + LRU |
| Eviction | Clear all when exceeded | LRU eviction | LRU eviction |
| Size control | `max_cached_hist_nodes` | `histogram_pool_size` (MB) | Capacity count |
| Contiguous | Yes (for allreduce) | Yes (aligned vectors) | Yes |

### Parallelism

| Aspect | XGBoost | LightGBM | This RFC |
|--------|---------|----------|----------|
| Row-parallel | `ParallelGHistBuilder` | Per-feature-group in `Dataset` | Thread-local scratch |
| Thread-local | Per-(thread, node) scratch | Implicit in feature groups | Per-thread scratch |
| Reduction | `ReduceHist(node, begin, end)` | Per-feature-group accumulate | Explicit reduce step |
| Batch building | `ParallelFor2d(node, row_range)` | Single node at a time | Both supported |

### Gradient Handling

| Aspect | XGBoost | LightGBM | This RFC |
|--------|---------|----------|----------|
| Reordering | No explicit reorder | `ordered_gradients_` buffer | `OrderedGradientBuffer` |
| Quantization | Not in CPU hist | 16-bit packed | Future work |
| Access pattern | Random (original order) | Sequential (reordered) | Both supported |

### Key Insights Adopted

From **XGBoost**:

1. Per-thread scratch histograms with explicit merge
2. Batch building for depth-wise (multiple nodes at once)
3. Contiguous allocation for efficient allreduce
4. `ParallelFor2d` over (node, row_range) dimensions

From **LightGBM**:

1. LRU eviction for memory-bounded training
2. Gradient reordering for cache-friendly access
3. Per-feature-group parallelism (we generalize to per-feature)
4. `histogram_pool_size` parameter for memory control

**Novel in this RFC**:

1. Unified contiguous pool with integrated LRU
2. Explicit `release()` for known lifecycle optimization
3. Rust's type system for memory safety guarantees
4. Pluggable parallelism strategy (feature vs row)

## Changelog

- 2024-12-02: Initial draft, combining RFC-0024 pooling with row-parallel building
- 2024-12-02: Updated to use SoA layout (separate grad/hess/count arrays) matching existing `FeatureHistogram` and `GradientBuffer`
- 2024-12-02: Added "Strategy Selection and Code Architecture" section with detailed guidance on strategy switching and code duplication avoidance
- 2024-12-02: Fixed naming conventions: `PerFeature` → `FeatureParallel`, `build_per_feature` → `build_feature_parallel` for consistency
- 2024-12-02: Renamed `ThreadLocalScratch` → `RowParallelScratch` for clarity (thread-local buffers that merge into NodeHistogram)
- 2024-12-03: Status → Approved
