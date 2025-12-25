# RFC-0004: Binning and Histograms

- **Status**: Implemented
- **Created**: 2024-11-01
- **Updated**: 2025-01-21
- **Depends on**: RFC-0001 (Data Types)
- **Scope**: Feature discretization and gradient accumulation for histogram-based GBDT

## Summary

This RFC covers the quantization (binning) of continuous features and histogram building for gradient accumulation. Binning discretizes features into bins enabling O(bins) instead of O(samples) split finding. Histograms accumulate gradient/hessian sums per bin for efficient gain computation.

## 1. Binning

### Motivation

Histogram-based GBDT algorithms (LightGBM, XGBoost "hist") require binning for:
- **Speed**: Build histograms over ~256 bins instead of sorting ~millions samples
- **Memory**: Store bins as u8/u16 instead of f64 (8x reduction for u8)
- **Cache efficiency**: Small histograms fit in L1/L2 cache

### Bin Computation

`BinMapper` stores bin boundaries and handles value-to-bin mapping:

```rust
pub struct BinMapper {
    bin_upper_bounds: Box<[f64]>,  // Upper bound for each bin
    n_bins: u32,                    // Total bins (including missing bin)
    missing_type: MissingType,      // None | Zero | NaN
    default_bin: u32,               // Bin for missing/default values
    most_freq_bin: u32,             // For histogram subtraction optimization
    feature_type: FeatureType,      // Numerical | Categorical
}
```

For numerical features, values are mapped via binary search:
- Value v → first bin where `v <= bin_upper_bounds[bin]`
- Last bound is typically `f64::MAX` to catch all values

For categorical features, a `HashMap<i32, u32>` maps category → bin.

### Quantized Storage

`BinnedDataset` organizes quantized features into groups with flexible layouts:

```
BinnedDataset
├── FeatureGroup 0 (dense, column-major, u8)
│   ├── Feature 0
│   └── Feature 1
├── FeatureGroup 1 (dense, column-major, u16)
│   └── Feature 2 (wide: >256 bins)
└── FeatureGroup 2 (sparse, column-major, u8)
    └── Feature 3
```

**Storage types** (`BinStorage`):
- `DenseU8` / `DenseU16`: Contiguous bin arrays
- `SparseU8` / `SparseU16`: CSR-like (row_indices, bin_values) for >90% sparse features

**Layouts** (`GroupLayout`):
- `ColumnMajor`: `[f0_row0, f0_row1, ..., f0_rowN, f1_row0, ...]` — contiguous per-feature access for histogram building (default)
- `RowMajor`: `[row0_f0, row0_f1, ..., row0_fK, row1_f0, ...]` — sequential row access

Column-major is preferred for training (13% speedup in benchmarks) as each feature's bins are contiguous, enabling efficient histogram accumulation.

### Missing Values

`MissingType` enum specifies handling:
- `None`: No missing values
- `NaN`: NaN values get a dedicated bin (typically last bin: `n_bins - 1`)
- `Zero`: Zeros treated as missing (for sparse data)

The `default_bin` field specifies which bin receives missing values during binning.

### Binning Types Summary

| Type | Purpose |
|------|---------|
| `BinMapper` | Computes and stores bin boundaries; maps values ↔ bins |
| `BinnedDataset` | Main quantized dataset with feature groups |
| `FeatureGroup` | Storage for related features (shared layout/bin type) |
| `BinnedFeatureInfo` | Per-feature metadata (mapper, group_index, index_in_group) |
| `BinStorage` | Enum of storage formats (DenseU8/U16, SparseU8/U16) |
| `FeatureView` | Zero-cost slice view for accessing feature bins |

### Implementation Notes

- **Max bins**: 256 for u8 (typical), 65536 for u16 (wide features)
- **Auto-grouping**: `GroupStrategy::Auto` separates features by bin count and sparsity
- **Global bin offsets**: Pre-computed for flat histogram indexing across all features
- **Bin type selection**: `BinType::for_max_bins(n)` selects smallest type that fits

---

## 2. Histogram Building

Histograms accumulate gradient and hessian sums per bin for each feature, enabling efficient split finding.

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

### Performance Decisions

Benchmarks showed:
- LLVM auto-vectorizes scalar loops effectively
- Manual SIMD (pulp) added overhead on ARM, minimal benefit on x86
- Quantization to int8/16 added unpacking overhead exceeding bandwidth savings
- Prefetching was 2x slower than hardware prefetching
- Row-parallel was 2.8x slower due to merge overhead

The simple scalar feature-parallel approach with subtraction trick outperforms complex alternatives.

## Integration

| Component | Integration Point |
| --------- | ----------------- |
| RFC-0001 (Data) | `Dataset` converted to `BinnedDataset` |
| RFC-0005 (Growing) | Split finder iterates histogram bins |
| RFC-0006 (Training) | Trainer builds histograms per-node |

## Changelog

- 2025-01-24: Merged RFC-0004 (Quantization) and RFC-0005 (Histograms) into unified binning RFC.
- 2025-01-23: Renamed `FeatureMeta` to `BinnedFeatureInfo` and `HistogramLayout`.
- 2025-01-21: Updated terminology to match refactored implementation.
