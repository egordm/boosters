# Backlog: BinnedDataset Redesign with Raw Feature Storage

**RFC**: [RFC-0018](../rfcs/0018-raw-feature-storage.md)  
**Created**: 2025-12-28  
**Revised**: 2025-12-29  
**Status**: Ready for Implementation  
**Refinement Rounds**: 4 of 4 (Complete)

## Overview

Comprehensive redesign of `BinnedDataset` storage to store raw feature values alongside bins, enabling architectural cleanup and improved API.

### Approach: Full Isolation + Parallel Implementation

**Key insight**: Rather than incrementally modifying existing code, we:

1. **Physically move** ALL existing binned/dataset/bundling code into a `deprecated/` folder
2. **Fix imports** so existing code continues to work during migration
3. **Build new implementation from scratch** in the main modules, following RFC-0018 exactly
4. **Switch over** once new implementation is complete and tested
5. **Delete deprecated folder** after migration is verified

This gives us a clean slate without mixing old and new patterns. The old code remains working during the transition.

### Primary Goals

1. **Architectural cleanup**: Remove dead code (`GroupLayout`, strided access, `BinType`) - ~530 lines
2. **Unified API**: Single `BinnedDataset` type with `from_array()` and `RowBlocks`
3. **Raw value storage**: Enable linear trees and gblinear to access original feature values

### Quality Gates

- No regression in linear trees mlogloss (baseline: ~0.38 on covertype)
- No regression >5% in histogram building performance
- Memory overhead <50% of binned size
- ~130 net lines removed

---

## Epic 0: Deprecation and Isolation

*Physically move ALL code that will be replaced into a `deprecated/` folder.*

### Story 0.1: Baseline Benchmarks (Speed + Quality)

**Status**: COMPLETE  
**Estimate**: 30 min  
**Priority**: BLOCKING

**Description**: Run full benchmark suite to capture current performance and quality baselines.

**Baseline Results** (captured 2025-12-28):

| Dataset | Model | Library | Metric | Value |
|---------|-------|---------|--------|-------|
| covertype | gbdt | boosters | mlogloss | 0.3807±0.0061 |
| covertype | linear_trees | boosters | mlogloss | 0.3772±0.0060 |
| covertype | linear_trees | lightgbm | mlogloss | 0.4053±0.0091 |
| california | linear_trees | boosters | rmse | 0.4726±0.0074 |
| synthetic_reg_medium | linear_trees | boosters | rmse | 18.7762±0.9837 |

**Definition of Done**:
- ✅ Full benchmark run completed
- ✅ Results documented

---

### Story 0.2: Move All Deprecated Code to Deprecated Folder

**Status**: COMPLETE  
**Estimate**: 1.5 hours  
**Priority**: BLOCKING

**Description**: Move the entire `data/binned/` directory AND `data/dataset.rs` AND `data/column.rs` AND `data/accessor.rs` AND `data/schema.rs` AND `data/views.rs` to `data/deprecated/`. The `deprecated` module itself is marked with `#![deprecated]` at the module level.

**Files to Move**:
```
data/binned/*                 → data/deprecated/binned/
data/dataset.rs               → data/deprecated/dataset.rs
data/column.rs                → data/deprecated/column.rs
```

**Tasks**:
1. Create `data/deprecated/` directory
2. Create `data/deprecated/mod.rs` with `#![deprecated(note = "Use new binned implementation")]` 
3. Move entire `data/binned/` to `data/deprecated/binned/`
4. Move `data/dataset.rs` to `data/deprecated/dataset.rs`
5. Move `data/column.rs` to `data/deprecated/column.rs`
6. Create new empty `data/binned/` directory
7. Create stub `data/binned/mod.rs` that re-exports from deprecated for now

**Definition of Done**:
- `data/deprecated/` contains all old binned/dataset/column code
- `deprecated` module has `#![deprecated]` attribute at module level
- `data/binned/mod.rs` re-exports from deprecated
- All existing code compiles and tests pass
- No behavior change

---

### Story 0.3: Fix All Imports After Move

**Status**: COMPLETE  
**Estimate**: 1 hour  
**Priority**: HIGH

**Description**: After moving files, fixed all imports throughout the codebase to use the re-exports.

**Changes Made**:
- Added `#![allow(deprecated)]` to lib.rs and quality_benchmark.rs during migration
- Added `#![allow(clippy::all)]`, `#![allow(dead_code)]` to deprecated module
- Fixed internal imports in deprecated files to use `super::`
- Fixed `add_binned` test calls to use 4-argument signature
- Re-exports all deprecated types at original paths

---

### Story 0.4: Verify Clean Separation

**Status**: COMPLETE  
**Estimate**: 15 min

**Description**: Verified the codebase is in a clean state with deprecated code isolated.

**Results**:
- ✅ Full test suite passes: `cargo test --package boosters`
- ✅ Clippy passes: `cargo clippy --package boosters -- -D warnings`
- ✅ All old binned/dataset/column/accessor/schema/views code in `data/deprecated/`
- ✅ `data/binned/` is now empty (except placeholder mod.rs)
- ✅ Committed: "refactor(data): move deprecated code to data/deprecated module [RFC-0018/0.2-0.4]"

---

### Story 0.5: Stakeholder Feedback Check (Epic 0)

**Status**: COMPLETE  
**Estimate**: 15 min

**Description**: Review stakeholder feedback file before proceeding to implementation of new storage types.

**Results**:
- ✅ Reviewed `workdir/tmp/stakeholder_feedback.md`
- ✅ No new feedback relevant to deprecation approach
- ✅ All previous feedback items marked as addressed

---

## Epic 1: New Storage Types

*Create new typed storage hierarchy from scratch, following RFC-0018 exactly.*

### Story 1.1: Create BinData Enum

**Status**: COMPLETE  
**Estimate**: 30 min

**Description**: Create the new `BinData` enum in `data/binned/bin_data.rs`.

**Location**: `data/binned/bin_data.rs` (NEW FILE)

**Implementation** (from RFC):

```rust
/// Bin data container. The variant encodes the bin width.
pub enum BinData {
    U8(Box<[u8]>),
    U16(Box<[u16]>),
}

impl BinData {
    pub fn is_u8(&self) -> bool;
    pub fn is_u16(&self) -> bool;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn get(&self, idx: usize) -> Option<u32>;
    pub fn get_unchecked(&self, idx: usize) -> u32;
    pub fn size_bytes(&self) -> usize;
    pub fn max_bins(&self) -> u32;
    pub fn needs_u16(n_bins: u32) -> bool;
    pub fn as_u8(&self) -> Option<&[u8]>;
    pub fn as_u16(&self) -> Option<&[u16]>;
}
```

**Testing Requirements**:

- Unit tests for all methods
- Edge cases: empty data, single element, max u8 bins (255), u16 threshold
- Size calculation tests

**Definition of Done**:

- `BinData` enum created with all methods
- Unit tests for all methods (including edge cases)
- Compiles without using any deprecated code

---

### Story 1.2: Create NumericStorage

**Status**: COMPLETE  
**Estimate**: 45 min

**Description**: Create dense numeric storage with bins + raw values.

**Location**: `data/binned/storage.rs` (NEW FILE)

**Implementation** (from RFC):
```rust
/// Dense numeric storage: [n_features × n_samples], column-major.
/// Raw values store actual f32 values including NaN for missing.
pub struct NumericStorage {
    bins: BinData,
    raw_values: Box<[f32]>,
}

impl NumericStorage {
    pub fn new(bins: BinData, raw_values: Box<[f32]>) -> Self;
    
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> u32;
    
    #[inline]
    pub fn raw(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> f32;
    
    #[inline]
    pub fn raw_slice(&self, feature_in_group: usize, n_samples: usize) -> &[f32];
    
    pub fn n_features(&self, n_samples: usize) -> usize;
    pub fn size_bytes(&self) -> usize;
}
```

**Definition of Done**:
- `NumericStorage` struct implemented
- All access methods working
- Unit tests for bin/raw access
- No use of deprecated code

---

### Story 1.3: Create CategoricalStorage

**Status**: COMPLETE  
**Estimate**: 30 min

**Description**: Create dense categorical storage (bins only, no raw values).

**Location**: `data/binned/storage.rs`

**Implementation** (from RFC):
```rust
/// Dense categorical storage: [n_features × n_samples], column-major.
/// No raw values - bin = category ID (lossless).
pub struct CategoricalStorage {
    bins: BinData,
}

impl CategoricalStorage {
    pub fn new(bins: BinData) -> Self;
    
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> u32;
    
    pub fn n_features(&self, n_samples: usize) -> usize;
    pub fn size_bytes(&self) -> usize;
}
```

**Definition of Done**:
- `CategoricalStorage` struct implemented
- Unit tests
- No raw value methods (categorical is lossless)

---

### Story 1.4: Create Sparse Storage Types

**Status**: COMPLETE  
**Estimate**: 1 hour

**Description**: Create CSC-like sparse storage for numeric and categorical features.

**Location**: `data/binned/storage.rs`

**Implementation**:
```rust
/// Sparse numeric storage: CSC-like, single feature.
pub struct SparseNumericStorage {
    sample_indices: Box<[u32]>,
    bins: BinData,
    raw_values: Box<[f32]>,
    n_samples: usize,
}

/// Sparse categorical storage: CSC-like, single feature.
pub struct SparseCategoricalStorage {
    sample_indices: Box<[u32]>,
    bins: BinData,
    n_samples: usize,
}
```

**Definition of Done**:
- Both sparse storage types implemented
- Binary search for random access
- Iteration methods for sequential access
- Unit tests

---

### Story 1.5: Create FeatureStorage Enum

**Status**: COMPLETE  
**Estimate**: 30 min

**Description**: Create unified enum wrapping all storage types.

**Location**: `data/binned/storage.rs`

**Implementation**:
```rust
/// Feature storage with bins and optional raw values.
pub enum FeatureStorage {
    Numeric(NumericStorage),
    Categorical(CategoricalStorage),
    SparseNumeric(SparseNumericStorage),
    SparseCategorical(SparseCategoricalStorage),
    // Bundle variant added in Story 1.6
}

impl FeatureStorage {
    pub fn has_raw_values(&self) -> bool;
    pub fn is_categorical(&self) -> bool;
    pub fn is_sparse(&self) -> bool;
    pub fn size_bytes(&self) -> usize;
}
```

**Definition of Done**:
- `FeatureStorage` enum with all variants
- Helper methods implemented
- Unit tests

---

### Story 1.6: Create BundleStorage (Optional/Deferred)

**Status**: Deferred  
**Estimate**: 1.5 hours

**Description**: Create EFB bundle storage type.

**Note**: This can be deferred until after core functionality works. EFB bundling is an optimization.

---

## Epic 2: New FeatureGroup

*Create new FeatureGroup that uses FeatureStorage.*

### Story 2.1: Create New FeatureGroup

**Status**: COMPLETE  
**Estimate**: 1.5 hours

**Description**: Create new `FeatureGroup` struct using `FeatureStorage`.

**Location**: `data/binned/group.rs` (NEW FILE)

**Implementation** (from RFC):
```rust
pub struct FeatureGroup {
    /// Global feature indices in this group.
    feature_indices: Box<[u32]>,
    /// Number of samples.
    n_samples: usize,
    /// Storage (bins + optional raw values).
    storage: FeatureStorage,
    /// Per-feature bin counts.
    bin_counts: Box<[u32]>,
    /// Cumulative bin offsets within group histogram.
    bin_offsets: Box<[u32]>,
}

impl FeatureGroup {
    pub fn new(...) -> Self;
    pub fn n_features(&self) -> usize;
    pub fn n_samples(&self) -> usize;
    pub fn has_raw_values(&self) -> bool;
    pub fn is_categorical(&self) -> bool;
    pub fn bin(&self, sample: usize, feature_in_group: usize) -> u32;
    pub fn raw(&self, sample: usize, feature_in_group: usize) -> Option<f32>;
    pub fn raw_slice(&self, feature_in_group: usize) -> Option<&[f32]>;
    pub fn feature_view(&self, feature_in_group: usize) -> FeatureView;
}
```

**Definition of Done**:
- New `FeatureGroup` struct implemented
- All access methods working
- Unit tests
- No use of deprecated code

---

### Story 2.2: Create New FeatureView

**Status**: COMPLETE  
**Estimate**: 30 min

**Description**: Create simplified `FeatureView` with no stride.

**Location**: `data/binned/view.rs` (NEW FILE - separate from group.rs for clarity)

**Implementation** (from RFC):

```rust
/// Zero-cost view into feature bins. No stride.
pub enum FeatureView<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    SparseU8 { sample_indices: &'a [u32], bin_values: &'a [u8] },
    SparseU16 { sample_indices: &'a [u32], bin_values: &'a [u16] },
}
```

**Definition of Done**:

- `FeatureView` enum with 4 variants (no strided variants!)
- No stride field
- Unit tests

---

### Story 2.3: Property-Based Tests for Storage Types

**Status**: COMPLETE  
**Estimate**: 1 hour

**Description**: Add property-based tests (proptest) for core storage types.

**Location**: Tests in `data/binned/storage.rs` or separate test module

**Properties to test**:

1. NumericStorage: bins.len() == raw_values.len() == n_features * n_samples
2. NumericStorage: raw_slice returns correct slice bounds
3. CategoricalStorage: bin values within valid range
4. Round-trip: construct storage, access all elements, verify match input
5. Edge cases: empty storage, single sample, single feature

**Definition of Done**:

- Property-based tests written using proptest
- All storage types covered
- Tests pass

---

## Epic 3: New BinMapper

*Copy and adapt existing BinMapper to new location.*

### Story 3.1: Copy and Adapt BinMapper

**Status**: COMPLETE  
**Estimate**: 1 hour

**Description**: Copy `BinMapper` from deprecated to new `data/binned/bin_mapper.rs` and adapt as needed.

**Location**: `data/binned/bin_mapper.rs` (NEW FILE)

**Rationale**: The existing BinMapper is fairly standalone and clean. Rather than evaluate whether to reuse, we simply copy it to the new location and adapt as needed. This maintains the "fresh start" principle.

**Tasks**:

1. Copy `deprecated/binned/bin_mapper.rs` to `data/binned/bin_mapper.rs`
2. Remove any dependencies on deprecated types
3. Ensure it works with new `BinData` enum
4. Update tests

---

## Epic 4: New Dataset Builder

*Create new DatasetBuilder that produces new storage types.*

### Story 4.1: Create FeatureAnalysis

**Status**: COMPLETE  
**Estimate**: 1 hour

**Description**: Create feature analysis to detect numeric vs categorical, sparsity, bin width.

**Location**: `data/binned/feature_analysis.rs` (NEW FILE)

**Key Functions**:
```rust
pub struct FeatureAnalysis {
    pub is_numeric: bool,
    pub is_sparse: bool,
    pub needs_u16: bool,
    pub n_unique: usize,
    pub n_zeros: usize,
    // etc.
}

pub fn analyze_feature(values: &[f32], config: &BinningConfig) -> FeatureAnalysis;
```

---

### Story 4.2: Create GroupingStrategy

**Status**: COMPLETE  
**Estimate**: 1 hour

**Description**: Create strategy for grouping features homogeneously.

**Location**: `data/binned/feature_analysis.rs` (added to existing)

**Key Logic**:
- ✅ Partition features by type (numeric/categorical) and bin width (U8/U16)
- ✅ Create homogeneous groups
- ✅ Handle sparse features (each gets own group)
- ✅ Handle trivial features (collected separately)

**Implementation**:
- `GroupType` enum: NumericDense, CategoricalDense, SparseNumeric, SparseCategorical, Bundle
- `GroupSpec` struct with feature_indices, group_type, needs_u16
- `GroupingResult` combining groups with trivial_features
- `compute_groups()` function implementing RFC grouping rules
- 7 comprehensive tests for grouping logic

---

### Story 4.3: Create New BinnedDatasetBuilder

**Status**: COMPLETE  
**Estimate**: 5 hours

**Description**: Create the main builder that produces `BinnedDataset`.

**Location**: `data/binned/builder.rs`

**Implementation**:
- `DatasetBuilder` with batch API (`from_array`, `from_array_with_metadata`) and single-feature API (`add_numeric`, `add_categorical`)
- `PendingFeature` struct for single-feature API
- `BuiltGroups` intermediate result with groups, bin_mappers, analyses, labels, weights
- `DatasetError` enum for error handling
- Helper functions: `bin_numeric()`, `bin_categorical()`, `create_bin_mappers()`
- Storage builders: `build_numeric_dense()`, `build_categorical_dense()`, `build_sparse_numeric()`, `build_sparse_categorical()`
- 10 comprehensive tests covering all functionality

**Key Methods**:
```rust
impl DatasetBuilder {
    pub fn new() -> Self;
    pub fn from_array(data: ArrayView2<f32>, config: &BinningConfig) -> Result<Self, DatasetError>;
    pub fn from_array_with_metadata(...) -> Result<Self, DatasetError>;
    pub fn add_numeric(...) -> Self;
    pub fn add_categorical(...) -> Self;
    pub fn set_labels(...) -> Self;
    pub fn build_groups(self) -> Result<BuiltGroups, DatasetError>;
}
```

**Critical**: This builder:
1. ✅ Analyzes all features
2. ✅ Creates homogeneous groups
3. ✅ Stores raw values for numeric features
4. ✅ Produces new `FeatureStorage` types

---

## Epic 5: New BinnedDataset

*Create new BinnedDataset with full API.*

### Story 5.1: Create New BinnedDataset Struct

**Status**: COMPLETE  
**Estimate**: 2 hours

**Description**: Create the main `BinnedDataset` struct and supporting types.

**Location**: `data/binned/dataset.rs` (NEW FILE)

**Implementation**:
- `FeatureLocation` enum: Direct, Bundled, Skipped
- `BinnedFeatureInfo` with name, bin_mapper, location
- `BinnedDataset` with from_built_groups() constructor
- Basic accessors: n_samples, n_features, n_groups, total_bins
- Labels/weights support
- global_bin_offsets for histogram indexing
- has_raw_values() for linear tree support
- 5 unit tests

**Definition of Done**:

- ✅ `BinnedFeatureInfo`, `FeatureLocation`, and `BinnedDataset` structs created
- ✅ Basic constructor from BuiltGroups
- ✅ Unit tests

---

### Story 5.2: Implement Basic Access Methods

**Status**: COMPLETE  
**Estimate**: 1.5 hours

**Description**: Implement bin/raw access methods.

**Implementation**:
- `bin(sample, feature) -> u32`: Access bin value via FeatureLocation
- `raw_value(sample, feature) -> Option<f32>`: Access raw value (None for categorical)
- `raw_feature_slice(feature) -> Option<&[f32]>`: Contiguous raw access
- Handles Direct, Bundled (panics - not yet implemented), and Skipped (panics) locations
- 4 new unit tests

**Methods**:
```rust
impl BinnedDataset {
    pub fn bin(&self, sample: usize, feature: usize) -> u32;
    pub fn raw_value(&self, sample: usize, feature: usize) -> Option<f32>;
    pub fn raw_feature_slice(&self, feature: usize) -> Option<&[f32]>;
    pub fn has_raw_values(&self) -> bool;
    pub fn n_samples(&self) -> usize;
    pub fn n_features(&self) -> usize;
}
```

---

### Story 5.3: Implement Feature Views for Histogram Building

**Status**: Not Started  
**Estimate**: 1.5 hours

**Description**: Implement `feature_views()` for histogram building (hot path).

**Methods**:
```rust
impl BinnedDataset {
    /// Get feature views for histogram building.
    pub fn feature_views(&self) -> Vec<FeatureView<'_>>;
    
    /// Get view for a single original feature.
    pub fn original_feature_view(&self, feature: usize) -> FeatureView<'_>;
}
```

---

### Story 5.4: Implement Raw Matrix Access (for gblinear)

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Implement bulk raw access methods.

**Methods**:

```rust
impl BinnedDataset {
    /// Returns a view into raw numeric values as a matrix.
    /// Returns None if not all features have raw values.
    pub fn raw_numeric_view(&self) -> Option<ArrayView2<'_, f32>>;
    
    /// Iterate over features that have raw values.
    pub fn raw_feature_iter(&self) -> impl Iterator<Item = (usize, &[f32])>;
}
```

**Note**: Use `ArrayView2` instead of `CowArray` - gblinear doesn't need to own the data.

---

### Story 5.5: Implement RowBlocks for Prediction

**Status**: Not Started  
**Estimate**: 2 hours

**Description**: Create buffered row block iterator for efficient prediction.

**Location**: `data/binned/row_blocks.rs` (NEW FILE)

**Design**:

- Block size: configurable, default 64 samples
- Returns rows as ndarray views
- For numeric features: return raw value
- For categorical features: cast bin index to f32
- Iteration pattern: row-major within each block

**Definition of Done**:

- RowBlocks struct implemented
- Iteration tests (block boundaries, last partial block)
- Edge cases: empty dataset, single sample, single feature

---

### Story 5.6: Review/Demo: New Dataset API

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Demo the new dataset API to stakeholders before integration.

**Tasks**:

1. Prepare demonstration materials showing:
   - New `BinnedDataset` API with `from_array()` and raw access
   - `FeatureView` with exactly 4 variants (compare to old 6+ variants)
   - Memory usage comparison (new vs old)
   - Code examples showing usage
2. Present to stakeholders
3. Document in `workdir/tmp/development_review_<timestamp>.md`

**Definition of Done**:

- Demo completed
- Stakeholder feedback captured
- Documentation written

---

## Epic 6: Integration

*Connect new implementation to training code.*

### Story 6.1: Update Histogram Building to Use New FeatureView

**Status**: Not Started  
**Estimate**: 2 hours

**Description**: Update histogram kernels to work with new `FeatureView` (no stride).

**Key Files**:
- `training/gbdt/histogram/*.rs`

**Changes**:
- Remove strided match arms
- Use new 4-variant `FeatureView`

---

### Story 6.2: Update Training to Use New BinnedDataset

**Status**: Not Started  
**Estimate**: 2 hours

**Description**: Update GBDT training to use new `BinnedDataset`.

**Key Files**:
- `training/gbdt/trainer.rs`
- `training/gbdt/grower.rs`

---

### Story 6.3: Update Linear Trees to Use Raw Values

**Status**: Not Started  
**Estimate**: 1.5 hours

**Description**: Update linear tree implementation to use raw values instead of bin midpoints.

**Location**: `training/gbdt/sample.rs`

**Key Changes**:

1. Update `BinnedSample::feature()` to call `BinnedDataset::raw_value(sample, feature)` instead of computing from bin midpoint
2. Remove `bin_to_midpoint()` calls where raw values are available
3. Handle case where raw value is `None` (categorical) - use bin as f32

**Definition of Done**:

- `BinnedSample::feature()` returns actual raw values
- Linear trees use raw feature values
- Tests pass

---

### Story 6.4: Update gblinear to Use New Dataset

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Update gblinear to use `raw_feature_iter()` for feature access.

**Note**: Use `ArrayView2` instead of `CowArray` for the matrix view - gblinear doesn't need to own the data.

---

## Epic 7: Switchover

*Switch from deprecated to new implementation.*

### Story 7.1: Update Re-exports to Use New Implementation

**Status**: Not Started  
**Estimate**: 45 min

**Description**: Change `data/binned/mod.rs` and `data/mod.rs` to export new types instead of deprecated.

**Before**:

```rust
pub use deprecated::binned::*;
```

**After**:

```rust
// New implementation exports
pub use self::bin_data::BinData;
pub use self::storage::{
    NumericStorage, CategoricalStorage, 
    SparseNumericStorage, SparseCategoricalStorage, 
    FeatureStorage
};
pub use self::group::FeatureGroup;
pub use self::view::FeatureView;
pub use self::bin_mapper::{BinMapper, BinningConfig, BinningStrategy};
pub use self::builder::BinnedDatasetBuilder;
pub use self::dataset::BinnedDataset;
pub use self::row_blocks::RowBlocks;
```

**Bundle Handling**: If deprecated code has bundled features enabled, the new code does not support bundles yet. Add a runtime check in the builder:

```rust
if config.enable_bundling {
    // For now, bundles are not supported in new implementation
    // Fall back to deprecated or return error
    return Err(Error::BundlingNotYetSupported);
}
```

**Definition of Done**:

- New exports in place
- Bundle handling addressed (error or fallback)
- All tests pass

---

### Story 7.2: Run Full Test Suite

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Verify everything works with new implementation.

---

### Story 7.3: Run Benchmarks and Validate Quality

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Run full benchmark suite and compare to baselines.

**Quality Gates**:
- covertype linear_trees mlogloss ≤ baseline (0.3772)
- No regression >5% in histogram building
- All tests pass

---

### Story 7.4: Review/Demo: Full Integration

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Demo the full integration with benchmarks to stakeholders.

**Tasks**:

1. Prepare demonstration materials showing:
   - Performance benchmarks (before/after)
   - Quality metrics (linear trees mlogloss comparison)
   - Code cleanup metrics (lines removed, methods deleted)
   - Interface simplification (old vs new FeatureView variants)
2. Present to stakeholders
3. Document in `workdir/tmp/development_review_<timestamp>.md`

**Definition of Done**:

- Demo completed with performance tables
- Stakeholder feedback captured
- Documentation written

---

## Epic 8: Cleanup

*Delete deprecated code.*

### Story 8.1: Delete Deprecated Folder

**Status**: Not Started  
**Estimate**: 15 min

**Description**: Delete `data/deprecated/` folder entirely.

**Prerequisite**: All tests pass with new implementation.

---

### Story 8.2: Final Code Review

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Review codebase for any remaining references to old patterns.

---

### Story 8.3: Documentation Update

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Update documentation to reflect new API.

---

### Story 8.4: Retrospective

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Conduct retrospective after full implementation is complete.

**Tasks**:

1. Each team member reflects on the implementation process
2. Discuss what went well / not well
3. Identify process improvements
4. Capture action items as new backlog stories where appropriate
5. Document in `workdir/tmp/retrospective.md`

**Definition of Done**:

- Retrospective completed
- Documentation written to `workdir/tmp/retrospective.md`
- Action items captured

---

## Summary

| Epic | Stories | Description |
| ---- | ------- | ----------- |
| 0. Deprecation | 5 | Move all old code to deprecated folder + stakeholder check |
| 1. Storage Types | 6 | Create new BinData, NumericStorage, etc. |
| 2. FeatureGroup | 3 | Create FeatureGroup, FeatureView + property tests |
| 3. BinMapper | 1 | Copy and adapt BinMapper |
| 4. Builder | 3 | Create new BinnedDatasetBuilder |
| 5. Dataset | 6 | Create new BinnedDataset with full API + demo |
| 6. Integration | 4 | Connect to training code |
| 7. Switchover | 4 | Switch from deprecated to new + demo |
| 8. Cleanup | 4 | Delete deprecated code + retrospective |
| **Total** | **36** | |

### Critical Path

```text
Epic 0 (Deprecation/Isolation)
    ↓
Epic 1 (Storage Types) + Epic 3 (BinMapper)
    ↓
Epic 2 (FeatureGroup)
    ↓
Epic 4 (Builder)
    ↓
Epic 5 (Dataset)
    ↓
Epic 6 (Integration)
    ↓
Epic 7 (Switchover)
    ↓
Epic 8 (Cleanup)
```

### Key Principles

1. **No modification of old code** - it's moved to deprecated folder
2. **New code follows RFC exactly** - no shortcuts or deviations
3. **Old code remains working** - via re-exports during transition
4. **Clean separation** - easy to reason about what's old vs new
5. **Delete when done** - deprecated folder is temporary

### Time Estimates

| Epic | Estimate |
| ---- | -------- |
| 0. Deprecation | ~3 hours |
| 1. Storage Types | ~4 hours |
| 2. FeatureGroup | ~3 hours |
| 3. BinMapper | ~1 hour |
| 4. Builder | ~7 hours |
| 5. Dataset | ~8 hours |
| 6. Integration | ~6 hours |
| 7. Switchover | ~3 hours |
| 8. Cleanup | ~2.5 hours |
| **Total** | **~37.5 hours** (~5 days focused work) |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
| ---- | ---------- | ------ | ---------- |
| Builder complexity (Story 4.3) | Medium | High | Build incrementally, test each storage type independently |
| Histogram integration regression (Story 6.1) | Low | High | Keep deprecated code working; frequent benchmarks |
| Linear tree accuracy regression (Story 6.3) | Low | High | Verify exact mlogloss values match baseline |
| Memory overhead exceeds 50% | Low | Medium | Track memory usage throughout; optimize if needed |

### Key Dates

- **Created**: 2025-12-28
- **Full Rewrite**: 2025-12-29
- **Refinement Complete**: 2025-12-29
- **Ready for Implementation**: Yes
