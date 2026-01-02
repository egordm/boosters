# RFC-0020: EFB Architecture Cleanup

## Status

Draft

## Summary

Refactor the EFB (Exclusive Feature Bundling) histogram building architecture to:

1. Move `BundleDecoder` to the bundling module
2. Replace complex tuple types with proper structs
3. Eliminate duplicate histogram building functions by integrating into FeatureView
4. Improve performance by adopting LightGBM's approach of building on bundled columns

## Motivation

Current issues:

- `BundleDecoder` lives in `ops.rs` but is bundling-specific
- `bundles: Vec<(u32, Vec<(usize, u32, u32)>)>` is hard to read and maintain
- Four separate histogram building functions: `build_contiguous`, `build_gathered`, `build_unbundled_contiguous`, `build_unbundled_gathered`
- Grower must know about bundling and dispatch accordingly
- Current unbundling approach is O(n_samples × n_features_in_bundles), negating performance benefits

## LightGBM's Approach (for reference)

LightGBM builds histograms directly on bundled columns:

1. **Histogram building**: O(n_samples × n_bundled_columns) - no decoding
2. **Split finding**: Iterate bundled histogram bins, use `bin_offsets` to identify which sub-feature owns each bin
3. **Partitioning**: Decode bundled bin to original feature only for prediction

This is faster because histogram building doesn't decode, and split finding just iterates bins once.

## Design

### Option A: OriginalFeatureView Abstraction (Simpler, chosen)

Create a new abstraction that provides access to **original feature bins** whether or not bundling is active.

```rust
/// Access to original feature bins, abstracting over bundling.
pub enum OriginalFeatureAccess<'a> {
    /// Direct access to a physical column (standalone feature).
    Direct(FeatureView<'a>),
    
    /// Decoded access from a bundled column.
    Decoded {
        view: FeatureView<'a>,
        offset: u32,
        n_bins: u32,
    },
}

impl<'a> OriginalFeatureAccess<'a> {
    /// Get the original feature's bin for a row.
    /// For bundled features, decodes the bundle on-the-fly.
    #[inline]
    pub fn get_bin(&self, row: usize) -> u32 {
        match self {
            Self::Direct(view) => view.get_bin(row).unwrap_or(0),
            Self::Decoded { view, offset, n_bins } => {
                let encoded = view.get_bin(row).unwrap_or(0);
                if encoded == 0 {
                    0 // all defaults
                } else if encoded >= *offset && encoded < offset + n_bins {
                    encoded - offset
                } else {
                    0 // not this feature
                }
            }
        }
    }
}
```

**Pros:**

- Simple mental model: iterate original features, always get original bins
- Histogram building code unchanged (just different views)
- Easy to implement

**Cons:**

- Still O(n_samples × n_original_features) for bundled columns
- Each row must decode for EACH feature in the bundle

### Option B: LightGBM-style (Optimal performance, more complex)

Build histograms on bundled columns, decode during split finding.

```rust
// During histogram building: just iterate bundled columns
for (col_idx, view) in bundled_views.iter().enumerate() {
    for i in 0..n_rows {
        let bin = view.get_bin(row).unwrap_or(0);
        histogram[layout.offset + bin as usize] += gradient;
    }
}

// During split finding: iterate bundled histogram, decode bin to feature
for bin in 0..bundled_layout.n_bins {
    let (grad_sum, hess_sum) = histogram[layout.offset + bin];
    let (orig_feat, orig_bin) = decoder.decode_bin(col_idx, bin);
    update_best_split(orig_feat, orig_bin, grad_sum, hess_sum);
}
```

**Pros:**

- O(n_samples × n_bundled_columns) - true bundling performance benefit
- Split finding is just O(total_bins), same as before

**Cons:**

- More complex: histogram layout must accommodate bundled columns
- Split finder needs bundling awareness
- Larger change to the codebase

## Decision

**Start with Option A** for code cleanliness, then optimize with Option B if profiling shows bundled histogram building is the bottleneck.

The current quality issue was due to mixing gradient statistics across features. Option A maintains the fix while cleaning up the architecture.

## Changes

### 1. Move BundleDecoder to bundling.rs

Move from `histograms/ops.rs` to `data/binned/bundling.rs`. It belongs with the other bundling types.

### 2. Replace tuple types with proper structs

```rust
// OLD
bundles: Vec<(u32, Vec<(usize, u32, u32)>)>

// NEW
pub struct SubFeatureInfo {
    pub original_idx: usize,
    pub bin_offset: u32,
    pub n_bins: u32,
}

pub struct BundleInfo {
    pub total_bins: u32,
    pub sub_features: Vec<SubFeatureInfo>,
}

bundles: Vec<BundleInfo>
```

### 3. Create OriginalFeatureAccessor

A struct that provides `OriginalFeatureAccess` for each original feature:

```rust
pub struct OriginalFeatureAccessor<'a> {
    // For each original feature, store the access mechanism
    accesses: Vec<OriginalFeatureAccess<'a>>,
}

impl<'a> OriginalFeatureAccessor<'a> {
    pub fn new(
        bundled_views: &'a [FeatureView<'a>],
        bundle_plan: &BundlePlan,
    ) -> Self { ... }
    
    pub fn get(&self, feature: usize) -> &OriginalFeatureAccess<'a> {
        &self.accesses[feature]
    }
}
```

### 4. Unify histogram building in HistogramBuilder

Remove the separate unbundled functions. HistogramBuilder uses `OriginalFeatureAccessor`:

```rust
impl HistogramBuilder {
    pub fn build_contiguous(
        &self,
        histogram: &mut [HistogramBin],
        ordered_grad_hess: &[GradsTuple],
        start_row: usize,
        feature_accessor: &OriginalFeatureAccessor<'_>,
        feature_metas: &[HistogramLayout],
    ) { ... }
}
```

### 5. Grower no longer needs bundle_decoder dispatch

Grower creates `OriginalFeatureAccessor` once from the bundle plan, passes it to histogram builder. No if/else branching needed.

## Files Changed

- `crates/boosters/src/data/binned/bundling.rs`: Add BundleDecoder, SubFeatureInfo, BundleInfo
- `crates/boosters/src/data/binned/mod.rs`: Export new types
- `crates/boosters/src/training/gbdt/histograms/ops.rs`: Remove BundleDecoder, unbundled functions
- `crates/boosters/src/training/gbdt/grower.rs`: Use OriginalFeatureAccessor

## Future Work

- RFC-0020: Adopt LightGBM-style bundled histogram building for performance
- Profile to determine if Option B is needed

## References

- LightGBM `feature_group.h`: bin_offsets_ for sub-feature ranges
- Original EFB fix: build histograms for original features to maintain quality
