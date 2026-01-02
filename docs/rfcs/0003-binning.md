# RFC-0003: Binning and Quantized Data Storage

**Status**: Implemented  
**Created**: 2025-12-15  
**Updated**: 2026-01-02  
**Scope**: Feature quantization, binned storage, and histogram infrastructure  
**Depends on**: RFC-0001 (Dataset)

## Summary

This RFC defines the binning and quantized data layer for histogram-based GBDT training:

1. **BinMapper**: Learns bin thresholds via weighted quantiles; maps float → bin index
2. **BinnedDataset**: Column-major storage of bin indices (u8/u16); dense or sparse
3. **FeatureGroup**: Homogeneous groups of features with same storage type (no per-element dispatch)
4. **HistogramBin**: `(f64, f64)` gradient/hessian accumulators for numerical precision

Column-major layout provides **13% speedup** over row-major in benchmarks.

## Motivation

### Problem Statement

GBDT training is dominated by split finding—for each node, evaluate all possible splits across all features. Naive implementation iterates raw floats, computing gain for each unique value:

```text
For each feature:
  For each sample in node:
    accumulate gradient/hessian into value bucket
  For each unique value:
    compute gain = f(left_sum, right_sum)
```

With 1M samples × 100 features, this is prohibitively slow. Unique values can number in the millions, and random float access has poor cache behavior.

### Solution: Histogram-Based Training

LightGBM and XGBoost "hist" mode discretize continuous features into a fixed number of bins (default: 256):

1. **Memory reduction**: 4-byte float → 1-byte bin index = **4× smaller**
2. **Speed**: Integer bin lookup; fixed histogram size enables cache-friendly access
3. **Subtraction trick**: `sibling_hist = parent_hist - child_hist` saves **10-44×** work for larger siblings (see Design Decisions)

### Scope

This RFC covers:
- Bin threshold learning (weighted quantiles)
- Binned data storage and grouping
- Feature views for histogram building
- Histogram bin representation

Histogram building algorithms are covered in RFC-0007 (Histograms).

## Design

### BinMapper

Maps raw feature values to bin indices:

```rust
pub struct BinMapper {
    /// Bin upper bounds (exclusive). len = n_bins - 1
    /// Values are sorted. bin(x) = upper_bounds.partition_point(|&t| t <= x)
    bin_upper_bounds: Box<[f32]>,
    /// Number of bins (including bin for missing values if present)
    n_bins: u32,
    /// How missing values (NaN) are handled
    missing_type: MissingType,
    /// Which bin missing values map to (if MissingType::AsBin)
    default_bin: u32,
}

pub enum MissingType {
    /// No missing values in training data; NaN at inference uses default_bin
    None,
    /// Missing values assigned to their own bin (typically bin 0 or n_bins-1)
    AsBin,
    /// Missing values go to zero bin (for sparse features)
    AsZero,
}

impl BinMapper {
    /// Map a single value to its bin index.
    #[inline]
    pub fn bin(&self, value: f32) -> u32 {
        if value.is_nan() {
            return self.default_bin;
        }
        self.bin_upper_bounds
            .partition_point(|&threshold| threshold <= value) as u32
    }
    
    /// Map values in batch (for binning entire feature columns).
    pub fn bin_column(&self, values: &[f32], out: &mut [u8]);
}
```

**Threshold learning**: Weighted quantiles create equal-frequency bins. For categoricals, each unique category gets its own bin (identity mapping: category 0 → bin 0, category 1 → bin 1, etc.).

**Note**: `BinMapper` is used only during training. Inference-time tree traversal uses floating split thresholds (`split_value: f32`), not bin indices.

### BinnedDataset

Column-major storage of bin indices:

```rust
pub struct BinnedDataset {
    /// Number of samples
    n_samples: usize,
    /// Feature groups with homogeneous storage
    groups: Vec<FeatureGroup>,
    /// Per-feature metadata (bin mapper, location)
    features: Box<[BinnedFeatureInfo]>,
    /// Global bin offsets for histogram allocation
    global_bin_offsets: Box<[u32]>,
}

pub struct BinnedFeatureInfo {
    pub bin_mapper: BinMapper,
    pub location: FeatureLocation,
}

pub enum FeatureLocation {
    /// Feature in a regular group
    Direct { group_idx: u32, idx_in_group: u32 },
    /// Feature bundled via EFB (see RFC-0004)
    Bundled { bundle_group_idx: u32, position_in_bundle: u32 },
    /// Feature skipped (trivial, constant)
    Skipped,
}
```

### FeatureGroup

Groups features with the same storage type for efficient iteration:

```rust
pub struct FeatureGroup {
    /// Global feature indices in this group
    feature_indices: Box<[u32]>,
    /// Bin storage (all features in group share same type)
    storage: GroupStorage,
    /// Per-feature bin counts
    bin_counts: Box<[u32]>,
    /// Cumulative bin offsets for histogram indexing
    bin_offsets: Box<[u32]>,
}

pub enum GroupStorage {
    /// Dense bins, u8 (≤256 bins per feature)
    DenseU8(Box<[u8]>),   // [n_features_in_group × n_samples], column-major
    /// Dense bins, u16 (>256 bins)
    DenseU16(Box<[u16]>),
    /// Sparse bins (CSC-like)
    SparseU8 {
        sample_indices: Box<[u32]>,  // Sorted sample indices with non-default values
        bin_values: Box<[u8]>,       // Bin at each index
        indptr: Box<[u32]>,          // Column pointers (per feature)
        default_bin: u8,
    },
    SparseU16 { /* ... */ },
    /// EFB bundle (multiple sparse features encoded together)
    Bundle {
        encoded_bins: Box<[u16]>,
        bin_offsets: Box<[u32]>,      // Per-feature offset in encoded space
        feature_n_bins: Box<[u32]>,
    },
}
```

### FeatureView

Training accesses bins through views that hide storage details:

```rust
/// Zero-copy view into feature bins for histogram building.
pub enum FeatureView<'a> {
    /// Dense bins, contiguous per feature (column-major)
    DenseU8(&'a [u8]),
    DenseU16(&'a [u16]),
    /// Sparse bins with sample indices
    SparseU8 {
        sample_indices: &'a [u32],
        bin_values: &'a [u8],
        default_bin: u8,
        n_samples: usize,
    },
    SparseU16 { /* ... */ },
}
```

**Key property**: No stride field. Column-major layout means feature values are contiguous—no per-element multiplication needed.

### HistogramBin

Gradient/hessian accumulator for split finding:

```rust
/// Histogram bin: (gradient_sum, hessian_sum)
/// Uses f64 for numerical precision in gain computation.
pub type HistogramBin = (f64, f64);

/// A histogram for one feature's bins
pub type Histogram = Box<[HistogramBin]>;
```

### Memory Layout

```text
Column-Major Layout (feature values contiguous):
┌─────────────────────────────────────────┐
│ Feature 0: [s0, s1, s2, ..., sN]        │  ← contiguous
│ Feature 1: [s0, s1, s2, ..., sN]        │  ← contiguous
│ Feature 2: [s0, s1, s2, ..., sN]        │  ← contiguous
└─────────────────────────────────────────┘

Row-Major Layout (sample features contiguous):
┌─────────────────────────────────────────┐
│ Sample 0: [f0, f1, f2, ..., fM]         │
│ Sample 1: [f0, f1, f2, ..., fM]         │
│ Sample 2: [f0, f1, f2, ..., fM]         │
└─────────────────────────────────────────┘
```

Column-major is optimal for histogram building: each feature's samples are accessed sequentially.

## Design Decisions

### DD-1: Column-Major Layout

**Context**: Histogram building iterates samples per feature. Memory layout affects cache performance.

**Options considered**:
1. Row-major (sample-major): Each sample's features contiguous
2. Column-major (feature-major): Each feature's samples contiguous

**Decision**: Column-major layout.

**Consequences**:
- **+13% speedup** in histogram building benchmarks (measured on synthetic 100K × 100 dataset)
- Feature iteration is sequential memory access (cache-friendly)
- Prediction needs transposition (handled by SampleBlocks buffer)
- Matches LightGBM's internal layout

### DD-2: u8 Default, u16 for >256 Bins

**Context**: Bin indices must be stored efficiently. More bins = finer splits but larger memory.

**Options considered**:
1. Always u8 (max 256 bins)
2. Always u16 (max 65536 bins)
3. Hybrid: u8 default, u16 when needed

**Decision**: Hybrid approach. 256 bins (u8) is the sweet spot for most problems.

**Consequences**:
- 256 bins is sufficient for most continuous features (XGBoost/LightGBM default)
- 4× memory savings vs always-u16
- High-cardinality categoricals can use u16 (>256 categories)
- Slight code complexity for two storage types

### DD-3: Homogeneous Feature Groups

**Context**: Different features may need different storage (u8/u16, dense/sparse). How to organize?

**Options considered**:
1. Single storage array with per-feature type tags
2. Homogeneous groups where all features share same storage type

**Decision**: Homogeneous groups.

**Consequences**:
- Single bounds check for entire group
- No per-element type dispatch in inner loops
- Slightly more complex group assignment logic
- Match statement at group level, not sample level

### DD-4: f64 Histogram Accumulators

**Context**: Gradients are f32; gain computation involves sums and differences. Precision matters.

**Options considered**:
1. f32 accumulators (8 bytes per bin for grad+hess)
2. f64 accumulators (16 bytes per bin)

**Decision**: f64 accumulators.

**Consequences**:
- Gain computation: `gain = (sum_grad)² / (sum_hess + λ)` involves large sums
- f32 loses precision when summing millions of gradients; differences of large sums amplify error
- Memory acceptable: 256 bins × 16 bytes = 4KB per feature histogram
- Matches XGBoost/LightGBM approach

### DD-5: No Stride in FeatureView

**Context**: Originally had `stride` field in FeatureView to support row-major layout.

**Options considered**:
1. Keep stride field for flexibility
2. Remove stride, enforce column-major

**Decision**: Remove stride field. Everything is column-major.

**Consequences**:
- Eliminates dead code (row-major histogram kernels were never used)
- No per-sample multiplication: `bins[sample]` instead of `bins[sample * stride]`
- Simpler FeatureView enum (4 variants instead of 6)
- ~60 lines of dead code removed

### DD-6: Weighted Quantile Binning

**Context**: How to determine bin thresholds?

**Options considered**:
1. Equal-width bins: `threshold[i] = min + i * (max - min) / n_bins`
2. Equal-frequency bins (quantiles): Each bin has same sample count
3. Weighted quantiles: Bins weighted by sample importance

**Decision**: Weighted quantile binning.

**Consequences**:
- Bins adapt to data distribution (more bins where data is dense)
- Sample weights respected during binning
- More expensive to compute than equal-width
- Matches XGBoost's approach

### DD-7: Subtraction Trick for Sibling Histograms

**Context**: When splitting a node, we need histograms for both children. Building both is redundant.

**Options considered**:
1. Build histograms for both children independently
2. Build one child, derive the other via subtraction from parent

**Decision**: Subtraction trick—build only the smaller child's histogram.

**Math**: Let `parent_hist` be the parent node's histogram. After splitting:
- `small_child_hist` = built by iterating over smaller child's samples
- `large_child_hist` = `parent_hist` - `small_child_hist`

For a balanced 50/50 split: 2× speedup (build 1 instead of 2).
For a 90/10 split: 10× speedup for the larger sibling.
For a 97.7/2.3 split (common with sparse data): **44× speedup**.

**Consequences**:
- Parent histograms must be retained until both children are processed
- HistogramPool manages parent histogram lifecycle
- Requires careful ordering of node processing (depth-first or breadth-first)

### DD-8: Ordered Gradients (Pre-Gathered)

**Context**: During histogram building, we access gradients by sample index. But sample indices within a node partition are scattered.

**Options considered**:
1. Random access: `grads[sample_idx]` for each sample in partition
2. Pre-gather: Copy gradients into contiguous buffer in partition order before histogram building

**Decision**: Pre-gather gradients into partition order.

**Rationale**: Histogram building accesses `(bin, gradient)` pairs in a tight loop. If gradients are scattered, each access is a cache miss. Pre-gathering makes gradient access sequential.

**Code pattern**:
```rust
// Before histogram building:
for (i, &sample_idx) in partition.iter().enumerate() {
    ordered_grads[i] = grads[sample_idx];
}

// During histogram building:
for i in 0..partition.len() {
    let bin = bins[partition[i]];
    let gh = ordered_grads[i];  // Sequential access
    histogram[bin] += gh;
}
```

**Consequences**:
- O(n) copy before each histogram build
- But: sequential gradient access → better cache utilization
- Net speedup in practice (measured in archived RFC-0004)

### DD-9: No Software Prefetching

**Context**: Histogram building has predictable access patterns. Should we add explicit prefetch hints?

**Options considered**:
1. Software prefetching via `_mm_prefetch` intrinsics
2. Rely on hardware prefetching

**Decision**: Rely on hardware prefetching only.

**Rationale**: Benchmarks on archived RFC-0004 showed explicit prefetching was **2× slower** than hardware-only. Modern CPUs have sophisticated prefetchers that detect sequential and strided patterns. Adding software prefetches can interfere with these, causing cache pollution.

**Consequences**:
- Simpler code (no intrinsics, no `#[cfg]` for architectures)
- Hardware prefetching handles column-major sequential access well
- May revisit if profiling shows specific hot spots

## Usage

### Configuration

Users configure binning through `GBDTConfig`:

```rust
let config = GBDTConfig::builder()
    .max_bins(256)           // Default, fits in u8
    .min_samples_bin(5)      // Minimum samples per bin for stability
    .build()?;

let model = GBDTModel::train(&dataset, None, config, 42)?;
```

### Bin Count Guidance

| Scenario | Recommended `max_bins` |
| -------- | ---------------------- |
| Default / most datasets | 256 (default) |
| Small datasets (<10K rows) | 64-128 (reduce overfitting risk) |
| High-cardinality categoricals | 512-1024 (u16 storage) |
| Memory-constrained | 64-128 |

**Note**: Features with fewer unique values than `max_bins` automatically use fewer bins. Requesting 256 bins on a feature with 10 unique values creates 10 bins.

### Per-Feature Bin Counts

For advanced use cases, per-feature bin counts can be specified via `FeatureMetadata`:

```rust
let metadata = FeatureMetadata::default()
    .max_bins_for(0, 64)     // Feature 0: 64 bins
    .max_bins_for(5, 1024);  // Feature 5: 1024 bins (u16 storage)
```

This is useful when you know certain features need finer or coarser binning.

## Benchmarks

**Hardware**: Apple M1 Pro, 16GB RAM, Rust 1.75 release mode.

### Layout Comparison

Dataset: Synthetic 100K samples × 100 features

| Layout | Histogram Build Time | Relative |
| ------ | -------------------- | -------- |
| Column-major | 12.3 ms | 1.00× |
| Row-major | 13.9 ms | 1.13× |

**Result**: Column-major is **13% faster**.

### Storage Type Impact

Dataset: Covertype (581K samples × 54 features)

| Storage | Memory | Build Time |
| ------- | ------ | ---------- |
| All u8 | 31.4 MB | 45 ms |
| All u16 | 62.8 MB | 52 ms |
| Hybrid | 31.4 MB | 45 ms |

**Result**: u8 storage is both smaller and slightly faster due to cache effects.

### Memory Analysis

Histogram memory per feature: 256 bins × 16 bytes = 4 KB

| Features | Histogram Memory | Cache Fit |
| -------- | ---------------- | --------- |
| 100 | 400 KB | L2 cache (256 KB-1 MB) |
| 1,000 | 4 MB | L3 cache (8-32 MB) |
| 10,000 | 40 MB | RAM (may spill L3) |

### Rejected Approaches (from archived RFC-0004)

| Approach | Result | Why Rejected |
| -------- | ------ | ------------ |
| Row-parallel histogram | 2.8× slower | Lock contention on histogram bins |
| Manual SIMD (ARM NEON) | Slight overhead | Compiler auto-vectorization sufficient |
| Software prefetching | 2× slower | Interferes with hardware prefetcher |

## Integration

### With RFC-0001 (Dataset)

`BinnedDataset` is created from `Dataset`:

```rust
let binned = BinnedDataset::from_dataset(&dataset, &binning_config)?;
```

`Dataset` retains raw values for prediction and linear models. `BinnedDataset` is used only for histogram-based training.

### With RFC-0004 (EFB)

EFB bundles sparse features into single encoded columns:

```rust
// Before bundling: 20 sparse one-hot features
// After bundling: 1 bundled feature with encoded bins

let views = binned.feature_views();  // May include Bundle variants
```

### With RFC-0007 (Histograms)

Histogram building uses `FeatureView` for bin access:

```rust
fn build_histogram(
    histogram: &mut [HistogramBin],
    view: &FeatureView,
    ordered_grads: &[GradsTuple],
    indices: &[u32],
) {
    match view {
        FeatureView::DenseU8(bins) => {
            // Direct array access, no stride
            for i in 0..indices.len() {
                let sample = indices[i] as usize;
                let bin = bins[sample] as usize;
                let gh = ordered_grads[i];
                histogram[bin].0 += gh.grad as f64;
                histogram[bin].1 += gh.hess as f64;
            }
        }
        // ... other variants
    }
}
```

## Files

| Path | Contents |
| ---- | -------- |
| `data/binned/dataset.rs` | `BinnedDataset`, creation, access |
| `data/binned/storage.rs` | `GroupStorage`, `FeatureGroup` |
| `data/binned/mapper.rs` | `BinMapper`, threshold learning |
| `data/binned/view.rs` | `FeatureView` |
| `data/binned/bundling.rs` | EFB bundle storage |

**Note**: Binning happens automatically during training. When users call `Model::train()`, the trainer internally creates `BinnedDataset` via `BinnedDataset::from_dataset()`. Users don't interact with binning directly unless using low-level APIs.

## Open Questions

1. **SIMD binning**: Could bin_column() use SIMD for threshold comparisons? Not prioritized—binning is one-time cost.

2. **Adaptive binning**: Should we support different bin counts per feature based on cardinality? Currently global `max_bins`.

## Testing Strategy

### Edge Cases

| Test Case | Expected Behavior |
| --------- | ----------------- |
| Empty feature (0 samples) | Return empty BinMapper |
| All-NaN feature | Single bin, all samples map to default_bin |
| Single unique value | Single bin |
| Exactly 256 unique values | 256 bins, u8 storage |
| 257 unique values | 257 bins, u16 storage |
| Sparse feature (99.9% zeros) | SparseU8 storage, zeros map to default_bin |
| Categorical with >256 categories | u16 storage |
| Weights with zeros | Zero-weight samples don't affect quantiles |
| `f32::INFINITY` | Maps to highest bin |
| `f32::NEG_INFINITY` | Maps to lowest bin |
| `f32::MAX` / `f32::MIN` | Handled without overflow |
| Subnormal floats | Handled correctly (no special treatment needed) |

### Correctness Verification

1. **Bin boundary test**: For each unique value, verify `bin(value)` returns correct bin index
2. **Monotonicity**: `bin(x) <= bin(y)` for `x <= y`
3. **Round-trip**: Values in same bin should be "close" (within bin width)

### Performance Regression Tests

| Benchmark | Threshold |
| --------- | --------- |
| BinMapper creation (100K samples) | <50ms |
| Histogram build (100 features) | No regression >5% |

## References

- LightGBM histogram implementation
- XGBoost "hist" tree method
- RFC-0004 (archived): Original binning RFC with benchmarks
- RFC-0018 (archived): BinnedDataset redesign

## Changelog

- 2026-01-02: Consolidated binning + storage RFC; expanded design decisions and tests
- 2025-12-15: Initial draft
