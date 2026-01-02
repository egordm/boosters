# RFC-0007: Histogram Building and Optimization

- **Status**: Implemented
- **Created**: 2025-12-27
- **Updated**: 2026-01-02
- **Depends on**: RFC-0003 (Binning)
- **Scope**: Gradient histogram construction, memory management, and performance optimization

## Summary

Histogram building is typically the dominant cost in histogram-based gradient boosted tree training. This RFC documents the histogram architecture used in boosters, including ordered gradients, the subtraction trick, LRU caching, and the parallelization strategy.

## Motivation

### Problem Statement

Training a gradient boosted tree requires finding optimal splits at each node. The naive approach examines every unique value for every feature at every node—O(samples × features × nodes). For production datasets with millions of samples, this is prohibitive.

Histogram-based splitting discretizes features into bins (typically ≤256), reducing split search to O(bins × features × nodes). However, building histograms—accumulating gradient/hessian sums per bin—becomes the new bottleneck.

### Why Histograms Dominate Training Time

In practice, histogram building dominates because it is performed for (many) nodes and touches every active feature for every row in the node. Reducing the amount of work in this hot path (better memory access, fewer histogram builds, less synchronization) tends to have the highest leverage on end-to-end training time.

This RFC intentionally avoids embedding point-in-time benchmark numbers or library “scorecards”. Performance results live in benchmark reports (see `docs/benchmarks/`) and Criterion suites; the RFC focuses on the architecture and the reasoning behind decisions.

## Design

### Histogram Layout

Each histogram bin stores gradient and hessian sums:

```rust
/// Single histogram bin: (gradient_sum, hessian_sum)
pub type HistogramBin = (f64, f64);

/// Offset and size for a feature's histogram region
pub struct HistogramLayout {
    pub offset: u32,   // Start position in flat array
    pub n_bins: u32,   // Number of bins for this feature
}
```

All feature histograms are stored contiguously in a flat array:

```text
[feat0: bin0..bin255][feat1: bin0..bin63][feat2: bin0..bin128]...
 ↑ offset=0           ↑ offset=256        ↑ offset=320
```

This layout enables cache-efficient sequential access during split finding and simple parallel builds with disjoint write regions.

### Histogram Builder

```rust
pub struct HistogramBuilder {
    parallelism: Parallelism,
}

impl HistogramBuilder {
    /// Build histogram for active features in a node
    pub fn build_histogram(
        &self,
        feature_views: &EffectiveViews<'_>,
        ordered_indices: &[u32],
        ordered_grad_hess: &[GradsTuple],
        histogram: &mut [HistogramBin],
        feature_indices: Option<&[u32]>,  // Column sampling
    );
}
```

The builder supports two modes:

- **Sequential**: Single-threaded, processes features one at a time
- **Parallel**: Multi-threaded, partitions features across threads

### Ordered Gradients

Before histogram building, gradients are reordered to match the sample partition:

```text
Original gradients:  [g0, g1, g2, g3, g4, g5, g6, g7]
Node partition:      [0, 2, 5, 7]  (samples in this node)

Ordered gradients:   [g0, g2, g5, g7]  (sequential access)
```

Benefits:

1. **Sequential memory access**: Inner loop reads gradients in order
2. **Cache efficiency**: Prefetcher works effectively
3. **Vectorization**: Compiler can auto-vectorize sequential loops

Implementation:

```rust
/// Pre-gather gradients for a node's samples
fn gather_gradients(
    gradients: &Gradients,
    indices: &[u32],
    ordered: &mut [GradsTuple],
) {
    for (i, &idx) in indices.iter().enumerate() {
        ordered[i] = gradients.get(idx as usize);
    }
}
```

### Histogram Building Kernels

The inner loop accumulates gradients into bins:

```rust
fn build_u8_gathered(
    bins: &[u8],
    ordered_grad_hess: &[GradsTuple],
    histogram: &mut [HistogramBin],
    indices: &[u32],
) {
    for i in 0..indices.len() {
        let row = indices[i] as usize;
        let bin = bins[row] as usize;
        let gh = ordered_grad_hess[i];
        histogram[bin].0 += gh.grad as f64;
        histogram[bin].1 += gh.hess as f64;
    }
}
```

Different kernels handle storage variants:

- `build_u8_*` / `build_u16_*`: Dense bins
- `build_sparse_*`: sparse (CSC-like) binned features (contiguous-row fast path)

Notes on sparse support:

- For **contiguous** row ranges, sparse features are supported by iterating the feature’s `sample_indices` and checking whether each non-default entry falls inside `[start_row, start_row + n_rows)`.
- For **gathered** row sets (arbitrary row indices), sparse histogram building is currently not implemented in the gathered kernels; use bundling (EFB) for very sparse one-hot style data (bundles are encoded as dense `u16` columns), or ensure the training hot path stays on dense views.

### Subtraction Trick

When splitting a node, we need histograms for both children. The subtraction trick computes the larger child by difference:

```text
    Parent (retained from previous level)
    ├── Smaller child → Build explicitly
    └── Larger child = Parent - Smaller
```

Implementation:

```rust
/// Compute sibling histogram by subtraction
fn subtract_histograms(
    parent: &[HistogramBin],
    smaller: &[HistogramBin],
    larger: &mut [HistogramBin],
) {
    for i in 0..parent.len() {
        larger[i].0 = parent[i].0 - smaller[i].0;
        larger[i].1 = parent[i].1 - smaller[i].1;
    }
}
```

This reduces work because only one child histogram is built explicitly; the other is derived from already-available data.
It is especially effective when splits are skewed, since the smaller child is cheaper to build.

### Histogram Pooling and LRU Cache

Storing all histograms for deep trees exceeds memory. The implementation uses a fixed-size **histogram pool** with an LRU eviction policy.

Key points:

- The pool allocates `cache_size` *slots* up-front.
- Each slot stores a full per-node histogram: `total_bins = sum(feature_bins)`.
- Training nodes are identified by the grower/partitioner’s **training node id** (leaf id during growth), not the final inference node id.

```rust
pub struct HistogramPool {
    // data: Box<[HistogramBin]> where bins are laid out as:
    // [slot0_bin0..slot0_binN][slot1_bin0..slot1_binN]...
    // plus mapping + timestamps for eviction.
}

impl HistogramPool {
    /// Acquire a slot for a training node.
    /// Returns Hit(slot) if cached, or Miss(slot) if it must be rebuilt.
    pub fn acquire(&mut self, node_id: NodeId) -> AcquireResult;

    /// Get a cached histogram view for a node (if present).
    pub fn get(&self, node_id: NodeId) -> Option<HistogramSlot<'_>>;

    /// Move a cached histogram mapping (used by subtraction trick).
    pub fn move_mapping(&mut self, from_node: NodeId, to_node: NodeId);
}
```

#### Pooling details (what’s actually implemented)

##### Slot allocation and clearing

- On a cache **miss**, the caller clears the slot and rebuilds the histogram.
- On a cache **hit**, the existing bins are reused as-is.

##### Eviction policy

- The LRU policy is **timestamp-based** (per-slot `last_used_time` and a global counter), not a linked list.
- Eviction chooses the minimum timestamp among unpinned slots.
- This makes eviction O(cache_size), which is fine because `cache_size` is small (typically on the order of active leaves).

##### Pinning (subtraction trick correctness)

- The grower temporarily **pins** the parent’s slot while performing `move_mapping + build_small + subtract`.
- Pinning prevents the parent histogram (now owned by the large child) from being chosen as an eviction victim mid-operation.

##### Identity mode (cache_size >= total_nodes)

When `cache_size >= total_nodes`, the pool switches to an optimized “identity mapping” mode:

- `node_id == slot_id` (no mapping arrays and no LRU timestamps).
- Slot validity is tracked via a per-tree **epoch** counter and a per-slot epoch marker.
- `reset_mappings()` bumps the epoch, so the next `acquire(node)` returns `Miss` unless that slot has already been built in the current epoch.

This avoids resetting large mapping arrays when the cache can hold all nodes.

#### What happens on eviction

If a parent histogram needed for the subtraction trick is not in the pool when splitting, training falls back to building both child histograms directly.
Increasing `cache_size` reduces the likelihood of this fallback at the cost of memory.

### Parallelization Strategy

```text
                     ┌─────────────┐
                     │  TreeGrower │
                     └──────┬──────┘
                            │
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
┌─────────────────┐                    ┌─────────────────┐
│ Feature-parallel│                    │ Level-parallel  │
│ (single node)   │                    │ (multiple nodes)│
└─────────────────┘                    └─────────────────┘
        ↓                                       ↓
  Many features                           Many nodes
  Large histograms                        Shallow trees
```

**Feature-parallel is default**: Each feature's histogram region is disjoint, enabling safe parallel writes without locks.

```rust
active_features.par_iter().for_each(|feature| {
    // Each feature writes to histogram[layout[feature].offset..]
    // No synchronization needed
});
```

Thresholds for parallel dispatch:

- `MIN_FEATURES_PARALLEL = 4`: Don't parallelize fewer features
- `MIN_WORK_PER_THREAD = 4096`: Minimum rows × features per thread

### Gradient Storage

```rust
/// Packed gradient/hessian pair
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct GradsTuple {
    pub grad: f32,
    pub hess: f32,
}
```

8-byte alignment enables single 64-bit load. The compiler can vectorize load operations.

## Design Decisions

### DD-1: f64 Histogram Bins

**Context**: Gradients are stored as f32, but histograms accumulate many values. What precision for bins?

**Options considered**:

1. **f32 bins**: Matches gradient precision, half the memory
2. **f64 bins**: Higher precision for accumulation
3. **Quantized i16/i32**: LightGBM approach, complex packing

**Decision**: Use f64 for histogram bins.

**Consequences**:

- Prevents numerical drift when summing millions of f32 values
- Split gain computation is precision-sensitive (difference of large sums)
- Memory overhead is acceptable—histograms are small (256 bins × features)
- Simpler implementation than quantized approach

### DD-2: Ordered Gradients

**Context**: During histogram building, we access gradients for samples in the current node. Access can be random (using indices) or sequential (pre-gathered).

**Options considered**:

1. **Random access**: `gradients[sample_index]` directly
2. **Pre-ordered**: Gather gradients into partition order before building

**Decision**: Pre-order gradients per node before histogram building.

**Consequences**:

- Adds O(n) gather step per node
- Enables sequential memory access in hot loop
- Enables compiler auto-vectorization

### DD-3: LRU Cache vs Full Storage

**Context**: Subtraction trick requires parent histograms. Deep trees have many nodes.

**Options considered**:

1. **Full storage**: Keep all histograms in memory
2. **Depth-limited**: Keep recent levels only
3. **LRU cache**: Evict least-recently-used when full

**Decision**: LRU cache with configurable size (default 8 slots).

**Consequences**:

- Memory-efficient for deep trees
- Works with any growth strategy
- Parent eviction before child build causes rebuild (rare in practice)
- 8 slots sufficient for typical depth-first growth

### DD-4: Feature-Parallel over Row-Parallel

**Context**: Histogram building can parallelize over rows (each thread builds partial histogram, then merge) or features (each thread builds complete histogram for subset of features).

**Options considered**:

1. **Row-parallel**: Each thread processes sample subset, merge at end
2. **Feature-parallel**: Each thread processes feature subset, no merge needed
3. **Hybrid**: Row-parallel for few features, feature-parallel otherwise

**Decision**: Feature-parallel only.

**Consequences**:

- No merge overhead (disjoint write regions)
- Scales with feature count (typical: 10-1000 features)
- Less effective for very few features (but rare in practice)

### DD-5: No Software Prefetching

**Context**: Bin access is pseudo-random (determined by sample ordering). Software prefetch could hide latency.

**Options considered**:

1. **No prefetching**: Rely on hardware prefetcher
2. **Software prefetch**: Explicit prefetch instructions with lookahead (64 elements)

**Decision**: No software prefetching.

**Consequences**:

- Hardware prefetcher handles our access pattern effectively
- In our experiments, explicit software prefetching was not beneficial
- Results may vary across CPUs; revisit only if profiling shows a clear latency bottleneck
- Simpler, more portable code

### DD-6: No SIMD Vectorization

**Context**: Inner loop accumulates two f64 values. SIMD could parallelize.

**Options considered**:

1. **Scalar loop**: Let compiler auto-vectorize
2. **Manual SIMD**: Use intrinsics (AVX2, NEON)
3. **pulp crate**: Portable SIMD abstraction

**Decision**: Scalar loops only.

**Consequences**:

- LLVM auto-vectorizes the simple scalar loop effectively
- Manual SIMD added overhead on ARM (pulp crate)
- Minimal benefit on x86 over auto-vectorization
- Simpler, more portable code

### DD-7: No Quantized Histograms

**Context**: LightGBM uses quantized i16/i32 histogram bins for reduced memory bandwidth.

**Options considered**:

1. **f64 bins**: Current approach
2. **Quantized i16 + scale**: Pack grad/hess in 32 bits, dequantize for split finding
3. **Quantized i32**: More precision, less packing benefit

**Decision**: Keep f64 bins, do not implement quantization.

**Consequences**:

- Simpler implementation
- No accuracy loss risk from quantization
- A high-performance quantized design would likely require accumulating in integer space and dequantizing only for split evaluation
- This remains a possible future direction if profiling shows we are memory-bandwidth bound

## Benchmarks and Regression Testing

Benchmark suites live alongside code (Criterion) and human-readable reports live under `docs/benchmarks/`.

- Use `cargo bench --bench training_gbdt` for end-to-end training benchmarks.
- Use the component benches under `crates/boosters/benches/` for focused histogram work.

Keep benchmark numbers out of the RFC so the design stays stable while performance data evolves.

## Integration

| Component | Integration Point | Notes |
| --------- | ----------------- | ----- |
| RFC-0003 (Binning) | BinnedDataset | Provides bin storage and views |
| RFC-0008 (GBDT Training) | SplitFinder | Iterates histogram bins |
| RFC-0008 (GBDT Training) | TreeGrower | Orchestrates build + split |
| RFC-0008 (GBDT Training) | GBDTConfig | `cache_size` setting |
| RFC-0006 (Sampling) | GOSS/subsampling | Sampling happens before histogram building; histograms receive pre-sampled gradients |

## Usage

### Configuration

Histogram cache size is the primary user-tunable parameter:

```rust
let config = GBDTConfig {
    cache_size: 8,  // Default, sufficient for typical depth-first growth
    ..Default::default()
};
```

**When to increase cache size**:

- Breadth-first tree growth strategy
- Very deep trees (max_depth > 15)
- Seeing rebuild warnings in logs

**Typical settings**:

| Growth Strategy | Recommended Cache Size |
| --------------- | ---------------------- |
| Depth-first | 8 (default) |
| Breadth-first | max_depth × 2 |
| Best-first (leaf-wise) | 16-32 |

### Observability

Histogram operations are not directly observable, but training logs include:

- Per-tree training time (histogram building is ~60-70%)
- Memory usage includes histogram cache

Users experiencing slow training with high `max_depth` should increase `cache_size` as a first step.

### Troubleshooting

| Symptom | Likely Cause | Solution |
| ------- | ------------ | -------- |
| Slow training with deep trees | Histogram cache eviction | Increase `cache_size` |
| Slow single-threaded mode | Expected—compute-bound | Use multi-threaded mode |
| High memory usage | Large histogram cache | Decrease `cache_size` or reduce `max_bins` |

## Testing Strategy

### Unit Tests

| Category | Tests |
| -------- | ----- |
| Histogram accumulation | Sum correctness, numerical precision |
| Subtraction trick | parent - child = sibling identity |
| LRU eviction | Correct eviction order, slot reuse |
| Parallel consistency | Same result as sequential |

### Concurrency Tests

| Test | Verification |
| ---- | ------------ |
| Data race detection | Run under ThreadSanitizer (TSAN) |
| Determinism | Parallel build == sequential build (bit-identical) |
| Thread scaling | 1, 2, 4, 8 threads produce same histograms |

Feature-parallel builds should be deterministic since each feature writes to disjoint memory regions. Tests verify this property.

**Determinism guarantee**: Given identical inputs and `ordered_indices`, histogram builds produce bit-identical results regardless of thread count. This is because:

1. Each feature accumulates independently (no cross-feature interaction)
2. Within a feature, samples are processed in `ordered_indices` order
3. Floating-point addition order is fixed by the iteration order

### Edge Cases

| Case | Test Approach |
| ---- | ------------- |
| Empty node (0 samples) | Histogram should be zero |
| Single sample | Histogram should equal that gradient |
| All same bin | Single bin should have full sum |
| Max bins (256/65536) | Boundary bins accumulate correctly |
| Overflow potential | Large gradients × many samples |
| Numerical precision | f64 vs f32 sum comparison |
| Subnormal gradients | Very small hessians (near 0) accumulate correctly |

**Subnormal handling**: f64 bins correctly accumulate subnormal f32 gradients. The test verifies that summing millions of tiny values (e.g., hessians ≈ 1e-38) doesn't collapse to zero and maintains reasonable precision.

### Performance Regression Testing

Performance is tracked via Criterion benchmarks in `benches/suites/component/`:

```bash
cargo bench --bench training_gbdt
```

The histogram building benchmark measures:

- Time per histogram build (microseconds)
- Throughput (samples × features / second)


CI can optionally flag regressions against stored baselines (thresholds are project-policy, not part of this RFC).


### Numerical Validation

```rust
#[test]
fn histogram_sum_equals_node_sum() {
    // Sum of all histogram bins should equal sum of node gradients
    let hist_sum: (f64, f64) = histogram.iter()
        .fold((0.0, 0.0), |acc, b| (acc.0 + b.0, acc.1 + b.1));
    let grad_sum: (f64, f64) = ordered_grads.iter()
        .fold((0.0, 0.0), |acc, g| (acc.0 + g.grad as f64, acc.1 + g.hess as f64));
    assert_relative_eq!(hist_sum.0, grad_sum.0, epsilon = 1e-6);
    assert_relative_eq!(hist_sum.1, grad_sum.1, epsilon = 1e-6);
}
```

## Files

| Path | Contents | Visibility |
| ---- | -------- | ---------- |
| `training/gbdt/histograms/mod.rs` | Module organization | Internal |
| `training/gbdt/histograms/ops.rs` | `HistogramBuilder`, build kernels | Internal |
| `training/gbdt/histograms/pool.rs` | `HistogramPool`, LRU cache | Internal |
| `training/gbdt/histograms/slices.rs` | Safe iteration over disjoint feature regions | Internal |
| `model/gbdt/config.rs` | `GBDTConfig::cache_size` | Public |
| `training/gradients.rs` | `GradsTuple`, `Gradients` buffer | Internal |

**Note**: Users interact with histograms only through `GBDTConfig::cache_size`. The histogram building machinery is internal and may change between versions.

## Concrete proposal: extracting the LRU policy

If we want to “split off” the LRU algorithm without changing behavior or performance, the lowest-risk refactor is to extract a small, array-based helper that owns only the eviction bookkeeping (timestamps + pins), while leaving node/slot mapping and histogram storage in `HistogramPool`.

Proposed internal API (no public surface):

- `training/gbdt/histograms/lru.rs` with `struct TimestampLru { last_used_time: Box<[u64]>, pinned: Box<[bool]>, current_time: u64 }`
- `TimestampLru` methods:
- `touch(slot_id)` increments time + records timestamp
- `find_victim()` scans for the min timestamp among unpinned slots
- `pin(slot_id) / unpin(slot_id)`

Migration plan:

1. Keep `HistogramPool`’s existing mapping arrays as-is.
2. Move only `last_used_time/current_time/pinned/find_lru_slot()` into `TimestampLru`.
3. Keep “identity mode” inside `HistogramPool` (it’s not LRU; it’s validity-by-epoch).

Why this is feasible:

- The current eviction is already O(cache_size) scanning over a dense array; the helper preserves that exact behavior.
- No heap allocations on the hot path; all arrays remain preallocated.
- Pin/unpin stays a slot-local boolean.

Why we might *not* want to do it:

- `HistogramPool` is performance-critical and currently self-contained; splitting files may not improve readability meaningfully.
- The tricky part is not LRU itself, it’s the interaction with `move_mapping()` and identity-mode semantics.

## Open Questions

1. **Quantized mode for memory-bound hardware**: If profiling shows histogram building becomes memory-bandwidth bound on some hardware, should we add a quantized histogram mode?

2. **Adaptive cache sizing**: Could we dynamically size the histogram cache based on tree depth and available memory?

## References

- [LightGBM feature_histogram.hpp](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp) - Histogram operations
- [XGBoost histogram.cuh](https://github.com/dmlc/xgboost/blob/master/src/tree/gpu_hist/histogram.cuh) - GPU histogram building
- Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017) - Section 2.2 on histogram-based algorithms

## Changelog

- 2026-01-02: Consolidated and clarified histogram design and tests
- 2025-12-27: Initial draft
