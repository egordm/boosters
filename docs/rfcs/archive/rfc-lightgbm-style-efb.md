# RFC: LightGBM-Style Exclusive Feature Bundling

**Status**: Approved (Design Review Complete)  
**Created**: 2025-12-27  
**Author**: Development Team  

## Summary

Refactor the Exclusive Feature Bundling (EFB) implementation to follow LightGBM's elegant approach where bundles are treated as regular columns rather than special-cased feature types. This eliminates decoding during histogram building (hot path) and significantly simplifies the codebase.

## Motivation

### Current State

The current EFB implementation uses a complex approach:

1. **Special `FeatureView` variants**: `BundledU8` and `BundledU16` contain decode information `(bin_offset, hist_offset)` per original feature
2. **Decoding in histogram building**: The histogram kernel must decode each bundled value to determine which original feature's histogram bin to update
3. **Multiple code paths**: Separate `build_bundled_*` and `build_unbundled_*` functions with different signatures
4. **Complex cache**: `BundleHistogramCache` pre-computes decode information for zero-allocation building

### Problems

1. **Performance**: Decoding in the hot path adds overhead
2. **Complexity**: Multiple code paths for bundled vs unbundled features
3. **Scattered histogram writes**: Bundled features may have non-contiguous histogram regions
4. **Maintenance burden**: Special cases throughout the codebase

### Training Time Gap

Current training time: **4.1s**  
Target (competitive with LightGBM): **<1.8s**

## LightGBM's Approach

LightGBM treats bundled features elegantly:

1. **Bundles are columns**: A bundle of N features becomes ONE column with offset-encoded bins
2. **No decoding during histogram building**: The encoded bin directly indexes into a contiguous histogram region
3. **Decoding only at split time**: When a split is found on a bundle column, decode to original feature (cold path)
4. **Uniform treatment**: All columns (bundled or not) use the same histogram building code

### Example

```
Original features: [0, 1, 2, 3, 4, 5, 6, 7]  (8 features, 8 histogram regions)

With bundling:
  Bundle0 = features [1, 4, 6] → 1 column with offset-encoded bins
  Bundle1 = features [2, 7]    → 1 column with offset-encoded bins  
  Standalone = [0, 3, 5]       → 3 columns (unchanged)

Result: 5 columns total, 5 histogram regions
```

### Histogram Layout

```
Without bundling:
  Histogram: [feat0_bins][feat1_bins][feat2_bins]...[feat7_bins]
             scattered, one region per original feature

With LightGBM-style bundling:
  Histogram: [Bundle0_bins][Bundle1_bins][feat0_bins][feat3_bins][feat5_bins]
             contiguous, one region per column
```

## Proposed Design

### Phase 1: Use Existing Infrastructure

The `BinnedDataset` already has the necessary building blocks:

```rust
// Already implemented:
dataset.bundled_feature_views()  // Returns views for bundled columns
dataset.bundled_bin_counts()     // Returns bin counts per bundled column
dataset.decode_bundle_split()    // Decodes bundle split to original feature
dataset.compute_bundled_columns() // Pre-computes encoded bins
```

The training code simply needs to use these instead of the original feature APIs.

### Phase 2: Simplify FeatureView

Remove the bundled variants since bundles are just regular columns:

```rust
// BEFORE (complex)
pub enum FeatureView<'a> {
    U8 { bins: &'a [u8], stride: usize },
    U16 { bins: &'a [u16], stride: usize },
    SparseU8 { row_indices: &'a [u32], bin_values: &'a [u8] },
    SparseU16 { row_indices: &'a [u32], bin_values: &'a [u16] },
    BundledU8 { bins: &'a [u8], stride: usize, decode: &'a [(u32, u32)] },  // REMOVE
    BundledU16 { bins: &'a [u16], stride: usize, decode: &'a [(u32, u32)] }, // REMOVE
}

// AFTER (simple)
pub enum FeatureView<'a> {
    U8 { bins: &'a [u8], stride: usize },
    U16 { bins: &'a [u16], stride: usize },
    SparseU8 { row_indices: &'a [u32], bin_values: &'a [u8] },
    SparseU16 { row_indices: &'a [u32], bin_values: &'a [u16] },
}
```

### Phase 3: Update Histogram Layout

The `HistogramLayout` and `HistogramPool` work per-column:

```rust
// In TreeGrower initialization:
fn create_histogram_layout(dataset: &BinnedDataset) -> Vec<HistogramLayout> {
    // Use bundled column counts, not original feature counts
    let bin_counts = dataset.bundled_bin_counts();
    let mut offset = 0;
    bin_counts.iter().map(|&n_bins| {
        let layout = HistogramLayout { offset, n_bins };
        offset += n_bins;
        layout
    }).collect()
}
```

### Phase 4: Update Split Finding

Split finding iterates over bundled columns:

```rust
struct SplitCandidate {
    column_idx: usize,      // Bundled column index
    bin: u32,               // Bin threshold (may be encoded for bundles)
    gain: f64,
    // ... other fields
}

impl SplitCandidate {
    /// Convert to tree-storable split (original feature index)
    fn to_tree_split(&self, dataset: &BinnedDataset) -> TreeSplit {
        let n_bundles = dataset.bundle_plan().map(|p| p.bundles.len()).unwrap_or(0);
        
        if self.column_idx < n_bundles {
            // Bundle column: decode to original feature
            let (orig_feature, orig_bin) = dataset
                .decode_bundle_split(self.column_idx, self.bin)
                .expect("valid bundle split");
            TreeSplit { feature: orig_feature, threshold: orig_bin, ... }
        } else {
            // Standalone column: map to original feature index
            let standalone_idx = self.column_idx - n_bundles;
            let orig_feature = dataset.bundle_plan()
                .map(|p| p.standalone_features[standalone_idx])
                .unwrap_or(self.column_idx);
            TreeSplit { feature: orig_feature, threshold: self.bin, ... }
        }
    }
}
```

### Phase 5: Update Partitioning

Partitioning uses original feature values, not bundled columns:

```rust
// Tree stores original feature index
fn partition_node(tree: &Tree, split: &TreeSplit, dataset: &BinnedDataset) {
    // Use original feature for partitioning
    let bin = dataset.get_bin(row, split.feature);
    // ... partition based on bin vs threshold
}
```

**Key insight**: Partitioning doesn't need bundles because the tree stores original feature indices.

### Column Index Mapping

We need a clear mapping between bundled columns and original features:

```
Bundled column layout:
  [bundle0, bundle1, ..., bundleN, standalone0, standalone1, ...]
   ↓        ↓                       ↓            ↓
   n_bundles columns                n_standalone columns

Column to original feature mapping:
  - column_idx < n_bundles: decode_bundle_split(column_idx, bin) → (orig_feat, orig_bin)
  - column_idx >= n_bundles: standalone_features[column_idx - n_bundles] → orig_feat
```

### Bundle Bin Encoding Scheme

For a bundle containing features [A, B, C] with bin counts [5, 10, 8]:

```
Bin offsets (computed during bundling):
  Feature A: offset = 1,  bins 0-4 → encoded as 1-5
  Feature B: offset = 6,  bins 0-9 → encoded as 6-15  
  Feature C: offset = 16, bins 0-7 → encoded as 16-23

Special bin 0: "All features at default value"
  - Rows where all features have their default (most frequent) bin
  - Does not represent any actual feature bin
  - decode_bundle_split(bundle_idx, 0) returns None

Total bundle bins: 1 + 5 + 10 + 8 = 24
  (bin 0 + feature A bins + feature B bins + feature C bins)

Encoding example:
  Row has feature A at bin 3 → encoded bin = 1 + 3 = 4
  Row has feature B at bin 7 → encoded bin = 6 + 7 = 13
  Row has feature C at bin 0 → encoded bin = 16 + 0 = 16
  Row has all defaults        → encoded bin = 0

Decoding example:
  Encoded bin 4  → feature A, bin 3  (4 >= 1 and 4 < 6)
  Encoded bin 13 → feature B, bin 7  (13 >= 6 and 13 < 16)
  Encoded bin 16 → feature C, bin 0  (16 >= 16 and 16 < 24)
  Encoded bin 0  → None (all defaults)
```

## API Changes

### BinnedDataset (existing, no changes needed)

```rust
impl BinnedDataset {
    // Already available:
    fn bundled_feature_views(&self) -> Vec<FeatureView<'_>>;
    fn bundled_bin_counts(&self) -> Vec<u32>;
    fn n_bundled_columns(&self) -> usize;
    fn decode_bundle_split(&self, bundle_idx: usize, encoded_bin: u32) -> Option<(usize, u32)>;
    
    // May need to add:
    fn bundled_column_to_original(&self, column_idx: usize) -> ColumnMapping;
}

enum ColumnMapping {
    Bundle { bundle_idx: usize },
    Standalone { original_feature: usize },
}
```

### TreeGrower Changes

```rust
impl TreeGrower {
    // Change from n_features to n_bundled_columns
    fn new(dataset: &BinnedDataset, ...) -> Self {
        let n_columns = dataset.n_bundled_columns();
        let bin_counts = dataset.bundled_bin_counts();
        let feature_metas = create_histogram_layout(&bin_counts);
        // ...
    }
    
    // Change feature views
    fn build_histogram(&self, ...) {
        let views = self.dataset.bundled_feature_views();
        // ... build using views (same code path for all columns)
    }
}
```

### HistogramBuilder Changes

Remove bundled-specific code:

```rust
impl HistogramBuilder {
    // REMOVE: build_bundled_contiguous, build_bundled_gathered
    // REMOVE: build_contiguous_with_cache, build_gathered_with_cache
    
    // KEEP: build_contiguous, build_gathered (work for all column types)
}
```

## Code Removal

The following can be removed:

1. **storage.rs**: `BundledU8`, `BundledU16` variants from `FeatureView`
2. **bundling.rs**: `BundleHistogramCache` and related types
3. **ops.rs**: All bundled-specific kernel functions
4. **grower.rs**: `bundle_histogram_cache` field and related logic

## Migration Path

1. **Phase 1**: Update grower to use `bundled_feature_views()` and `bundled_bin_counts()`
2. **Phase 2**: Update split finding to decode bundle splits
3. **Phase 3**: Remove `BundledU8`/`BundledU16` from `FeatureView`
4. **Phase 4**: Remove `BundleHistogramCache` and related code
5. **Phase 5**: Clean up unused bundle-specific kernel functions

## Testing Strategy

### Unit Tests

1. Verify `bundled_feature_views()` returns correct views
2. Verify `bundled_bin_counts()` matches expected layout
3. Verify `decode_bundle_split()` correctly decodes all cases
4. Verify histogram building produces identical results with/without bundling

### Edge Case Tests

1. **All-default bundle**: Bundle where all rows have encoded bin = 0
2. **Multi-activation bundle**: Different rows activate different sub-features
3. **Sparse standalone**: Standalone feature with sparse storage
4. **Mixed bit-width**: Mix of u8 and u16 columns in same training
5. **Single-feature bundle**: Bundle with one feature (should behave like standalone)
6. **Maximum bins**: Bundle with features summing to > 255 bins (requires u16)

### Bit-Exact Validation

Create test that:
1. Builds histograms with bundling enabled
2. Builds histograms with bundling disabled (original features)
3. Verifies histogram sums are bit-exact after accounting for layout differences

### Integration Tests

1. Train model with bundling enabled, verify accuracy matches non-bundled
2. Verify splits are stored with original feature indices
3. Verify prediction works correctly on new data
4. Verify feature importance attributes correctly to original features

### Regression Tests

1. Train on known dataset with fixed seed and hyperparameters
2. Compare tree structure against golden file
3. Ensure test covers bundled and non-bundled paths

### Performance Tests

1. Benchmark histogram building time on:
   - Airlines dataset (large, many sparse features)
   - Higgs dataset (medium, dense features)
   - Synthetic sparse dataset (high sparsity, many bundles)
2. Benchmark full training time
3. Compare with LightGBM on same datasets

## Expected Benefits

| Metric | Before | After |
|--------|--------|-------|
| Code paths for histogram building | 4+ | 1 |
| Decode operations in hot path | O(n_rows × n_bundles) | 0 |
| `FeatureView` variants | 6 | 4 |
| Special cache structures | 1 (BundleHistogramCache) | 0 |
| Lines of code | ~500 extra | removed |

## Risks and Mitigations

### Risk 1: Breaking existing functionality

**Mitigation**: Comprehensive test suite, phase-by-phase migration with validation at each step.

### Risk 2: Edge cases in bundle decoding

**Mitigation**: The existing `decode_bundle_split()` already handles edge cases. Add more test coverage. Ensure out-of-range encoded bins return `None` (no panic).

### Risk 3: Performance regression in non-bundled case

**Mitigation**: Non-bundled case uses `feature_views()` which falls back to original behavior.

### Risk 4: Reduced parallelism with fewer columns

**Consideration**: With bundling, we have fewer columns to parallelize over. For example, 100 sparse features → 10 bundles reduces parallel work items. Monitor in benchmarks; if problematic, consider sub-column parallelization.

### Risk 5: Bin 0 semantics in split finding

**Issue**: Bin 0 in a bundle means "all features at default". Rows with bin 0 contribute to bin 0's histogram, but a split at bin 0 doesn't map to any meaningful original feature split.

**Mitigation**: The split finder should skip bin 0 for bundle columns, or `decode_bundle_split(bundle_idx, 0)` should return `None`. Verify current behavior and add test.

## Design Clarifications (Round 1)

### Categorical Features and Bundling

Categorical features are NOT bundled. The bundling algorithm only considers sparse numerical features. Categorical features remain as standalone columns. This is already enforced in the current implementation.

### Feature Importance

When computing feature importance, we must attribute histogram contributions back to original features. The approach:

1. During split finding, we find the best split on a bundled column
2. When recording importance, we decode to the original feature and attribute the gain there
3. The `decode_bundle_split()` function provides this mapping

### Prediction Path

Prediction uses `get_bin(row, original_feature)` which is unchanged. Trees store original feature indices, so prediction works without any bundling awareness.

### Memory Layout of Bundled Columns

Currently `Vec<Vec<u16>>` which has pointer indirection. For better cache efficiency, consider a future optimization:

```rust
struct BundledColumns {
    flat_data: Vec<u16>,      // All bundle data contiguous
    column_offsets: Vec<usize>, // Start of each bundle
    ...
}
```

This is a potential follow-up optimization, not blocking for initial implementation.

### Standalone Feature Storage

Standalone features retain their original storage layout (may have stride > 1 for row-major groups). The `bundled_feature_views()` function returns the original `feature_view()` for standalone features, preserving their access pattern.

### Backward Compatibility

- Trees store original feature indices → prediction works unchanged
- No persistent state for `BundleHistogramCache` → no migration needed
- Public API: `BundleHistogramCache` is internal, not exposed publicly

## Open Questions

1. Should `HistogramLayout` track whether a column is a bundle or standalone?
   - **Proposal**: No, treat all columns uniformly during histogram building

2. How to handle the "0 bin" case in bundles (all features at default)?
   - **Current behavior**: Encoded as 0, maps to first feature's default bin
   - **Proposal**: Keep current behavior, it's correct for histogram building

3. Should we cache the column-to-original mapping?
   - **Proposal**: Yes, create `BundledColumnMapping` struct computed once during grower initialization

## Implementation Checklist

### Phase 1: Preparation (Non-breaking)

- [ ] Verify `bundled_feature_views()` returns correct views
- [ ] Verify `bundled_bin_counts()` returns correct counts  
- [ ] Verify `decode_bundle_split()` handles all edge cases
- [ ] Run existing test suite to establish baseline

### Phase 2: Update Histogram Building

- [ ] Cache `bundled_feature_views()` in `TreeGrower` (avoid re-allocation)
- [ ] Update `TreeGrower` to use `bundled_bin_counts()` for `HistogramLayout`
- [ ] Update histogram building to iterate over bundled columns
- [ ] Add column-to-original feature mapping in grower

### Phase 3: Update Split Finding

- [ ] Update split finder to iterate over `n_bundled_columns`
- [ ] Decode bundle splits to original feature before storing in tree
- [ ] Verify tree dumps show original feature indices
- [ ] Add test: bundled vs non-bundled produces equivalent trees

### Phase 4: Code Cleanup

- [ ] Remove `BundledU8`/`BundledU16` from `FeatureView` in storage.rs
- [ ] Remove `BundleHistogramCache` from bundling.rs
- [ ] Remove `bundle_histogram_cache` from grower.rs
- [ ] Remove unused bundled kernel functions from ops.rs
- [ ] Run clippy, verify no dead code warnings

### Phase 5: Validation

- [ ] Run full test suite
- [ ] Benchmark histogram building time
- [ ] Benchmark full training time
- [ ] Compare accuracy with LightGBM on test datasets
