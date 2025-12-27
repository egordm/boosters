# RFC-0017: EFB Training Integration

- **Status**: Draft
- **Created**: 2025-12-27
- **Updated**: 2025-12-27
- **Depends on**: RFC-0011 (Feature Bundling)
- **Scope**: Training integration for feature bundling

## Summary

This RFC proposes completing the Exclusive Feature Bundling (EFB) integration to close the 6-10x single-threaded performance gap between boosters and LightGBM on sparse datasets. **The root cause is that bundle plans are computed but never used during histogram building.**

## Motivation

### Problem Statement

Benchmarks show boosters is **6x slower than LightGBM** in single-threaded mode:

| Mode | Boosters | LightGBM | Ratio |
|------|----------|----------|-------|
| Single-threaded | 0.46s | 0.08s | **6.06x slower** |
| Multi-threaded (8 cores) | 0.26s | 0.46s | **1.7x faster** |

Detailed per-tree timing on covertype shows even worse:
- **Boosters**: 2343 ms/tree
- **LightGBM**: 235 ms/tree
- **Ratio**: **10x slower per tree**

### Root Cause: Bundle Plan Computed But Not Used

RFC-0011 implemented bundle plan creation in the dataset builder:

```rust
// builder.rs - This works correctly
let bundle_plan = if config.enable_bundling {
    Some(self.create_bundle_plan(n_rows, config))
} else {
    None
};
BinnedDataset::with_bundle_plan(..., bundle_plan)
```

But the training code **ignores the bundle plan entirely**:

```rust
// grower.rs line 252 - THIS IS THE BUG
let bin_views = dataset.feature_views();  // Returns ALL 54 features!
```

The `feature_views()` method returns views for all original features, not bundled columns. Searching for "bundle" in `crates/boosters/src/training/gbdt/**` returns only **1 match** - a test that passes `None` for the bundle plan.

### Covertype Dataset Analysis

| Metric | Value |
|--------|-------|
| Total features | 54 |
| Sparse features (>90% zeros) | 41 |
| Dense features | 13 |
| LightGBM bundles | ~3 (inferred from "Total Bins 2262") |
| LightGBM effective columns | ~16 |
| Boosters effective columns | **54 (no bundling used)** |
| Feature reduction | **3.4× fewer columns** |

**LightGBM verbose output:**
```
[LightGBM] [Info] Total Bins 2262
```

Without bundling, 54 features × 256 bins = 13,824 bins. LightGBM's 2,262 bins represents an **84% bin reduction**.

### RFC-0011 Claim Was Wrong

The RFC-0011 "Measured Performance" section claimed:
> Training time unchanged — EFB is memory optimization, not compute optimization.

This is **incorrect**. EFB is absolutely a compute optimization:
- Fewer features = fewer histogram passes per node
- Fewer bins = faster split finding
- LightGBM's 10x speed advantage on covertype demonstrates this

## Design

### Overview

Complete the EFB integration by modifying the training code to use bundled columns:

1. **Phase 1**: Create `bundled_feature_views()` that returns bundled column views
2. **Phase 2**: Modify `TreeGrower` to use bundled views
3. **Phase 3**: Decode bundle splits back to original features

### Phase 1: Bundled Feature Views

Add a method to `BinnedDataset` that returns views for bundled columns:

```rust
impl BinnedDataset {
    /// Get feature views for bundled columns (fewer than original features).
    ///
    /// If bundling is active, returns:
    /// - One view per bundle (mutually exclusive sparse features combined)
    /// - One view per standalone dense feature
    ///
    /// If bundling is not active, returns same as `feature_views()`.
    pub fn bundled_feature_views(&self) -> Vec<FeatureView<'_>> {
        match &self.bundle_plan {
            Some(plan) if plan.is_effective() => {
                let mut views = Vec::with_capacity(plan.binned_column_count());
                
                // Add bundle views (bundled columns with offset-encoded bins)
                for (bundle_idx, bundle) in plan.bundles.iter().enumerate() {
                    views.push(self.bundle_view(bundle_idx, bundle));
                }
                
                // Add standalone feature views (unchanged)
                for &feature_idx in &plan.standalone_features {
                    views.push(self.feature_view(feature_idx));
                }
                
                views
            }
            _ => self.feature_views(),
        }
    }
    
    /// Get the effective number of columns for histogram building.
    pub fn n_bundled_columns(&self) -> usize {
        match &self.bundle_plan {
            Some(plan) if plan.is_effective() => plan.binned_column_count(),
            _ => self.n_features(),
        }
    }
}
```

### Phase 2: Bundle View Creation

Create views that return offset-encoded bins for bundled columns:

```rust
impl BinnedDataset {
    /// Create a view for a bundled column.
    ///
    /// The view encodes bins using the offset scheme:
    /// - bin 0 = "all features in bundle are zero"
    /// - bin offset[i] + k = "feature i has bin value k"
    fn bundle_view(&self, bundle_idx: usize, bundle: &FeatureBundle) -> FeatureView<'_> {
        // For now, create a virtual view that computes encoded bins on-demand
        // This can be optimized later with pre-computed bundled storage
        FeatureView::Bundle {
            bundle_idx,
            features: &bundle.members,
            offsets: &bundle.bin_offsets,
            dataset: self,
        }
    }
}

/// Extended FeatureView to support bundled columns.
pub enum FeatureView<'a> {
    // Existing variants...
    U8 { bins: &'a [u8], stride: usize },
    U16 { bins: &'a [u16], stride: usize },
    SparseU8 { row_indices: &'a [u32], bin_values: &'a [u8] },
    SparseU16 { row_indices: &'a [u32], bin_values: &'a [u16] },
    
    // New variant for bundled columns
    Bundle {
        bundle_idx: usize,
        features: &'a [usize],      // Original feature indices in bundle
        offsets: &'a [u32],         // Bin offsets for each feature
        dataset: &'a BinnedDataset, // Reference back to dataset
    },
}

impl<'a> FeatureView<'a> {
    /// Get bin value for a row (works for all variants).
    pub fn get_bin(&self, row: usize) -> Option<u32> {
        match self {
            // Existing implementations...
            
            FeatureView::Bundle { features, offsets, dataset, .. } => {
                // Find which feature in bundle is non-zero (if any)
                for (i, &feature_idx) in features.iter().enumerate() {
                    if let Some(bin) = dataset.get_bin(row, feature_idx) {
                        if bin > 0 {
                            // Found non-zero feature, return offset-encoded bin
                            return Some(offsets[i] + bin);
                        }
                    }
                }
                // All features zero -> bin 0
                Some(0)
            }
        }
    }
}
```

### Phase 3: TreeGrower Integration

Modify `TreeGrower` to use bundled views and decode splits:

```rust
impl TreeGrower {
    pub fn grow(&mut self, dataset: &BinnedDataset, gradients: &Gradients, ...) {
        // Use bundled views instead of original views
        let bin_views = dataset.bundled_feature_views();
        
        // ... rest of tree growing ...
    }
    
    /// Decode a split on a bundled column to original feature + threshold.
    fn decode_split(&self, dataset: &BinnedDataset, split: &SplitInfo) -> DecodedSplit {
        if let Some(plan) = dataset.bundle_plan() {
            if split.feature_idx < plan.bundles.len() {
                // This is a bundle column - decode to original feature
                if let Some((orig_feature, orig_bin)) = 
                    plan.decode_bundle_split(split.feature_idx, split.threshold) 
                {
                    return DecodedSplit {
                        feature: orig_feature,
                        threshold: orig_bin,
                        is_bundled: true,
                    };
                }
            }
        }
        
        // Not bundled or decode failed - use as-is
        DecodedSplit {
            feature: split.feature_idx,
            threshold: split.threshold,
            is_bundled: false,
        }
    }
}
```

### Phase 4: Histogram Layout for Bundles

Update histogram layout to accommodate bundled columns:

```rust
impl TreeGrower {
    fn compute_feature_metas(&self, dataset: &BinnedDataset) -> Vec<HistogramLayout> {
        match dataset.bundle_plan() {
            Some(plan) if plan.is_effective() => {
                let mut metas = Vec::with_capacity(plan.binned_column_count());
                
                // Bundle columns have more bins (sum of constituent features)
                for bundle in &plan.bundles {
                    metas.push(HistogramLayout {
                        n_bins: bundle.total_bins,
                        offset: metas.iter().map(|m| m.n_bins).sum(),
                    });
                }
                
                // Standalone features use their original bin counts
                for &feature_idx in &plan.standalone_features {
                    metas.push(HistogramLayout {
                        n_bins: dataset.n_bins(feature_idx),
                        offset: metas.iter().map(|m| m.n_bins).sum(),
                    });
                }
                
                metas
            }
            _ => {
                // Original behavior for unbundled datasets
                (0..dataset.n_features())
                    .map(|f| HistogramLayout { /* ... */ })
                    .collect()
            }
        }
    }
}
```

## Design Decisions

### DD-1: Virtual vs Pre-computed Bundled Storage

**Context**: Should bundled columns be computed on-demand or pre-stored?

**Options**:
1. **Virtual views**: Compute encoded bins on-demand during histogram building
2. **Pre-computed storage**: Create additional storage for bundled columns

**Decision**: Start with virtual views.

**Rationale**:
- Simpler implementation
- No additional memory overhead
- The encoding is cheap (one non-zero lookup per bundle member)
- Can optimize later if profiling shows it's a bottleneck

### DD-2: Column Ordering

**Context**: How should bundled columns be ordered in the view?

**Options**:
1. **Bundles first, then standalone**: Simple, deterministic
2. **Interleaved by original order**: Preserves some locality
3. **By density**: Dense features first for better cache behavior

**Decision**: Bundles first, then standalone features in original order.

**Rationale**:
- Simple and predictable
- Matches how LightGBM orders features
- Column sampling can sample from the bundled view

### DD-3: Split Decoding Timing

**Context**: When should bundle splits be decoded to original features?

**Options**:
1. **At split time**: Decode immediately when split is found
2. **At tree finalization**: Keep bundle indices during growth, decode at end
3. **At model export**: Keep bundle indices internally, decode only for external format

**Decision**: Decode at split time.

**Rationale**:
- Tree structure remains in terms of original features
- Feature importance tracking works unchanged
- Model serialization doesn't need bundle awareness
- Simpler debugging (can see which original feature was split)

## Expected Impact

### Performance

On covertype (54 features → 16 bundled columns):
- **Histogram building**: ~3.4× faster (54/16 fewer features)
- **Split finding**: ~2-3× faster (fewer histograms to scan)
- **Overall per-tree**: ~3× faster

Combined with existing optimizations, this should reduce the gap from 10× slower to 2-3× slower.

### Accuracy

No impact on model accuracy:
- Bundle encoding is lossless
- Split decoding restores original feature indices
- Predictions are identical to unbundled training

## Testing Plan

1. **Unit tests**: Bundle view creation, split decoding
2. **Integration tests**: Train on covertype, compare predictions with bundling off
3. **Benchmark**: Compare single-threaded performance with LightGBM
4. **Accuracy validation**: Ensure metrics match unbundled baseline

## Implementation Plan

1. [ ] Add `FeatureView::Bundle` variant
2. [ ] Implement `BinnedDataset::bundled_feature_views()`
3. [ ] Update histogram layout computation for bundles
4. [ ] Modify `TreeGrower::grow()` to use bundled views
5. [ ] Implement split decoding in tree construction
6. [ ] Update column sampling to work with bundled columns
7. [ ] Add benchmarks comparing bundled vs unbundled

## Open Questions

1. **Row partitioning**: Does partitioning need bundle awareness?
   - Partitioning uses bin values to route rows
   - Bundle bins encode multiple features - need to verify routing still works

2. **Categorical features**: How do native categoricals interact with bundling?
   - RFC-0012 handles raw categoricals differently
   - May need to exclude categoricals from bundling

3. **Feature importance**: Should we track importance per bundle or per original feature?
   - Decision: Track per original feature (decode splits during tracking)

## References

- RFC-0011: Feature Bundling and Sparse Optimization (bundle plan creation)
- [LightGBM dataset.cpp](https://github.com/microsoft/LightGBM/blob/master/src/io/dataset.cpp) - EFB implementation
- [LightGBM feature_group.hpp](https://github.com/microsoft/LightGBM/blob/master/src/io/feature_group.hpp) - Feature group histogram building

## Changelog

- 2025-12-27: Initial draft identifying bundle plan not being used during training
