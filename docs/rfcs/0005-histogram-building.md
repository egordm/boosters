# RFC-0005: Histogram Building

- **Status**: Implemented
- **Created**: 2024-11-01
- **Updated**: 2025-01-21
- **Depends on**: RFC-0004
- **Scope**: Gradient/hessian histogram accumulation for split finding

## Summary

Histogram building accumulates gradient and hessian sums per bin for each feature, enabling efficient split finding. The implementation uses `(f64, f64)` bins, feature-parallel accumulation, LRU-cached histogram storage, and the subtraction trick for sibling computation.

## Design

### Histogram Structure

Each histogram bin stores a `(gradient_sum, hessian_sum)` tuple as `(f64, f64)`:

```rust
pub type HistogramBin = (f64, f64);
```

**Why f64?** Despite gradients being stored as `f32`, histograms use `f64` for accumulation. Gain computation involves differences of large sums that lose precision in f32. Memory overhead is acceptable since histograms are small (typically 256 bins × features).

Histograms are laid out contiguously per slot with features packed sequentially:
```
Slot 0: [feat0_bin0, feat0_bin1, ..., feat1_bin0, feat1_bin1, ...]
```

`HistogramLayout` tracks each feature's offset and bin count for indexing:
```rust
pub struct HistogramLayout {
    pub offset: u32,
    pub n_bins: u32,
}
```

### Memory Management

`HistogramPool` provides LRU-cached histogram storage mapping logical node IDs to physical slots:

- **Fixed-size cache**: Allocates `cache_size` slots, each holding `total_bins` bins
- **LRU eviction**: When cache is full, evicts least-recently-used slot (timestamp-based)
- **Identity mode**: When `cache_size >= total_nodes`, uses direct `node_id == slot_id` mapping
- **Pinning**: Slots can be pinned to prevent eviction during active use

Key operations:
- `acquire(node_id) -> Hit | Miss` — Get or allocate a slot
- `move_mapping(from, to)` — Transfer histogram ownership between nodes
- `subtract(target, source)` — In-place `target -= source`
- `reset_mappings()` — Invalidate all slots (new tree)

### Building Strategy

`HistogramBuilder` accumulates gradients into histograms using per-feature kernels:

**Ordered gradients**: Gradients are pre-gathered into partition order before histogram building. This enables sequential memory access instead of random lookups.

**Two build paths**:
1. `build_contiguous` — Fast path for contiguous row ranges
2. `build_gathered` — For non-contiguous indices with pre-gathered gradients

**Per-feature kernels** handle different bin storage formats:
- `U8`/`U16` — Dense bins (stride 1)
- `U8`/`U16` strided — Column-major layouts
- `SparseU8`/`SparseU16` — CSC-style sparse features

Each kernel iterates rows, looks up the bin index, and accumulates `grad`/`hess`:
```rust
for i in 0..ordered_grad_hess.len() {
    let bin = bins[row] as usize;
    let slot = &mut histogram[bin];
    slot.0 += gh.grad as f64;
    slot.1 += gh.hess as f64;
}
```

### Subtraction Trick

The subtraction trick avoids redundant histogram computation for sibling nodes:

```
sibling_histogram = parent_histogram - child_histogram
```

When a node splits:
1. Build histogram for the **smaller** child (fewer rows)
2. Compute larger child via `pool.subtract(larger, smaller)` using parent's histogram

This provides **10-44x speedup** for larger siblings and is the primary optimization.

Implementation uses `move_mapping` to transfer parent histogram ownership to larger child, then subtracts the smaller child's histogram in-place.

## Performance

### Parallelization

**Feature-parallel only**: Each feature's histogram region is disjoint, enabling safe parallel writes without locks or merging.

`HistogramFeatureIter` encapsulates unsafe pointer arithmetic to yield disjoint `&mut [HistogramBin]` slices per feature:
```rust
iter.par_for_each_mut(&features, |f, hist_slice| {
    build_feature(hist_slice, gradients, &bin_views[f]);
});
```

**Thresholds** (auto-corrected per-build):
- `MIN_FEATURES_PARALLEL = 4` — Minimum features to parallelize
- `MIN_WORK_PER_THREAD = 4096` — Minimum rows × features per thread

### Design Decisions

Benchmarks showed:
- LLVM auto-vectorizes scalar loops effectively
- Manual SIMD (pulp) added overhead on ARM, minimal benefit on x86
- Quantization to int8/16 added unpacking overhead exceeding bandwidth savings
- Prefetching was 2x slower than hardware prefetching
- Row-parallel was 2.8x slower due to merge overhead

The simple scalar feature-parallel approach with subtraction trick outperforms complex alternatives.

## Changelog

- 2025-01-23: Renamed `FeatureMeta` to `HistogramLayout` to clarify purpose and avoid confusion with other meta types.
- 2025-01-21: Updated terminology to match refactored implementation; standardized header format
