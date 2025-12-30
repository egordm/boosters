# RFC-0019: RawFeaturesView for GBLinear

**Status**: Draft  
**Created**: 2025-12-30  
**Author**: Team  
**Related**: RFC-0018 (Raw Feature Storage)

## Summary

Define `RawFeaturesView` - a minimal trait/type for accessing raw feature values per-feature. This enables GBLinear training and prediction to work with both the legacy `FeaturesView` (wrapping `ArrayView2<f32>`) and `BinnedDataset` (via `raw_feature_slice()`), enabling migration to a single dataset type.

## Motivation

Currently:
- **`FeaturesView`** wraps `ArrayView2<f32>` with feature-major layout
- **`BinnedDataset`** stores raw values per-feature with `raw_feature_slice(idx) -> Option<&[f32]>`

Both provide the same access pattern that GBLinear needs:
```rust
for feat_idx in 0..n_features {
    let feature_values: &[f32] = /* contiguous slice */;
    for (sample_idx, &value) in feature_values.iter().enumerate() {
        // use value
    }
}
```

The goal is to make GBLinear work with `BinnedDataset` directly without:
1. Creating parallel interfaces (`train_binned`, `predict_binned`)
2. Allocating contiguous matrices (`to_raw_feature_matrix()`)
3. Adding new traits that force infrastructure changes

## Design

### Option A: Thin Wrapper Type (Recommended)

Create `RawFeaturesView` as an enum that can wrap either source:

```rust
/// Zero-cost view into raw feature values.
/// 
/// Provides per-feature iteration for GBLinear training and prediction.
/// Works with both legacy ArrayView2 and BinnedDataset.
pub enum RawFeaturesView<'a> {
    /// Wraps an ArrayView2 in feature-major layout [n_features, n_samples]
    Array(ArrayView2<'a, f32>),
    /// Wraps a BinnedDataset, accessing raw_feature_slice() per feature
    Binned(&'a BinnedDataset),
}

impl<'a> RawFeaturesView<'a> {
    #[inline]
    pub fn n_features(&self) -> usize {
        match self {
            Self::Array(arr) => arr.nrows(),
            Self::Binned(ds) => ds.n_features(),
        }
    }
    
    #[inline]
    pub fn n_samples(&self) -> usize {
        match self {
            Self::Array(arr) => arr.ncols(),
            Self::Binned(ds) => ds.n_samples(),
        }
    }
    
    /// Get a contiguous slice of raw values for a feature.
    /// 
    /// Returns `None` for categorical features or sparse storage in BinnedDataset.
    /// For Array variant, always returns Some.
    #[inline]
    pub fn feature(&self, idx: usize) -> Option<&[f32]> {
        match self {
            Self::Array(arr) => {
                let row = arr.row(idx);
                // ArrayView1 is contiguous if the original array was C-order
                row.as_slice()
            }
            Self::Binned(ds) => ds.raw_feature_slice(idx),
        }
    }
    
    /// Iterate over all features that have raw values.
    /// 
    /// Yields (feature_idx, slice) pairs. Skips categorical/sparse features.
    pub fn iter_features(&self) -> impl Iterator<Item = (usize, &[f32])> + '_ {
        (0..self.n_features()).filter_map(move |idx| {
            self.feature(idx).map(|slice| (idx, slice))
        })
    }
}

// Conversions
impl<'a> From<ArrayView2<'a, f32>> for RawFeaturesView<'a> {
    fn from(arr: ArrayView2<'a, f32>) -> Self {
        Self::Array(arr)
    }
}

impl<'a> From<&'a BinnedDataset> for RawFeaturesView<'a> {
    fn from(ds: &'a BinnedDataset) -> Self {
        Self::Binned(ds)
    }
}

impl<'a> From<FeaturesView<'a>> for RawFeaturesView<'a> {
    fn from(fv: FeaturesView<'a>) -> Self {
        Self::Array(fv.view())
    }
}
```

### Why Enum Over Trait

1. **No dynamic dispatch**: Match compiles to branch, inlined in hot loops
2. **No generic proliferation**: Methods take `RawFeaturesView<'_>` not `impl RawFeatures`
3. **Easy migration**: After `types::Dataset` is removed, simplify to just the Binned variant
4. **Explicit**: Callers see exactly what types are supported

### Migration Path

**Phase 1 (Immediate)**:
1. Create `RawFeaturesView` enum
2. Update `LinearModel::predict_into()` to accept `RawFeaturesView` 
3. Update `Updater::update_round()` to accept `RawFeaturesView`
4. Both still work with FeaturesView via `.into()`

**Phase 2 (After Dataset removal)**:
1. Remove `Array` variant from enum
2. Or replace enum with direct `&BinnedDataset` parameter
3. Delete `FeaturesView` entirely

### Code Changes

#### LinearModel Prediction

Before:
```rust
pub fn predict_into(&self, features: FeaturesView<'_>, mut output: ArrayViewMut2<'_, f32>) {
    for feat_idx in 0..n_features {
        let feature_values = features.feature(feat_idx);
        for (sample_idx, &value) in feature_values.iter().enumerate() {
            // ...
        }
    }
}
```

After:
```rust
pub fn predict_into(&self, features: impl Into<RawFeaturesView<'_>>, mut output: ArrayViewMut2<'_, f32>) {
    let features = features.into();
    for (feat_idx, feature_values) in features.iter_features() {
        for (sample_idx, &value) in feature_values.iter().enumerate() {
            // ...
        }
    }
}
```

#### Updater

Before:
```rust
pub fn update_round(
    &self,
    model: &mut LinearModel,
    data: &FeaturesView<'_>,
    // ...
)
```

After:
```rust
pub fn update_round(
    &self,
    model: &mut LinearModel,
    data: impl Into<RawFeaturesView<'_>>,
    // ...
)
```

The inner loop is identical - `feature_values.iter().enumerate()` works on `&[f32]`.

### GBDT Prediction (No Change)

GBDT prediction already uses `SampleBlocks::for_each_with()` which materializes contiguous blocks. This path is unchanged.

### GBLinear Training

The `Updater` methods that currently take `&FeaturesView<'_>` will take `impl Into<RawFeaturesView<'_>>`. The main `GBLinearTrainer::train()` method will:

1. Accept `&BinnedDataset` directly (new signature)
2. Build `RawFeaturesView::Binned(dataset)` internally
3. Pass to Updater

Eventually, when `types::Dataset` is deleted, `train()` only accepts `&BinnedDataset`.

## Handling Categorical/Sparse Features

For features where `raw_feature_slice()` returns `None`:
- **GBLinear**: Skip these features (they can't contribute to linear weights)
- **Validation**: Check at model creation that n_features matches expected

The `iter_features()` method naturally skips these.

## Performance Analysis

### Zero-Allocation
- No matrix copies
- Feature slices are borrowed from existing storage
- Match on enum variant is a single branch

### Cache Efficiency
- Same iteration order as before (feature-major)
- BinnedDataset raw values are contiguous per-feature
- No change to memory access patterns

### Expected Overhead
- One branch per feature access (match on enum)
- Negligible compared to actual computation

## Alternatives Considered

### A. Trait-Based Approach
```rust
trait RawFeatures {
    fn n_features(&self) -> usize;
    fn feature(&self, idx: usize) -> Option<&[f32]>;
}
```
Rejected: Requires generics everywhere, complicates API.

### B. to_raw_feature_matrix() (REJECTED - Already Tried)
Creates O(n × m) allocation. Negates memory benefits of BinnedDataset.

### C. Separate Methods (REJECTED - Already Tried)
`train_binned()`, `predict_binned()` duplicate code and create parallel interfaces.

## Success Criteria

- [ ] GBLinear prediction works with BinnedDataset via RawFeaturesView
- [ ] GBLinear training works with BinnedDataset via RawFeaturesView  
- [ ] No new public methods on LinearModel/GBLinearTrainer (signature changes ok)
- [ ] No allocation in conversion path
- [ ] Benchmark shows <5% overhead vs direct FeaturesView

## Open Questions

1. **Sparse features**: Should we add a `sparse_feature()` method that returns `(indices, values)` for sparse raw storage? Not needed for current GBLinear (skips sparse).

2. **Naming**: `RawFeaturesView` vs `FeatureSlices` vs `RawFeatureAccess`?

3. **Location**: `crate::data::RawFeaturesView` or `crate::data::binned::RawFeaturesView`?

## Implementation Plan

See updated backlog in `docs/backlogs/dataset-consolidation.md` Epic 3.
