# Backlog: BinnedDataset Redesign with Raw Feature Storage

**RFC**: [RFC-0018](../rfcs/0018-raw-feature-storage.md)  
**Created**: 2025-12-28  
**Status**: Ready for Implementation  
**Refinement Rounds**: 6

## Overview

Comprehensive redesign of `BinnedDataset` storage to store raw feature values alongside bins, enabling architectural cleanup and improved API.

### Primary Goals

1. **Architectural cleanup**: Remove dead code (`GroupLayout`, strided access, `BinType`) - ~530 lines
2. **Unified API**: Single `BinnedDataset` type with `from_array()` and `RowBlocks`
3. **Raw value storage**: Enable linear trees and gblinear to access original feature values

### Quality Gates

- No regression in linear trees mlogloss (baseline: ~0.38 on covertype)
- No regression >5% in histogram building performance
- Memory overhead <50% of binned size
- ~130 net lines removed

### Key Discovery During Refinement

Results.md shows linear_trees on covertype already achieves 0.3754 mlogloss (better than LightGBM's 0.4053). This contradicts RFC's claim of ~0.80. Story 0.1 will investigate this discrepancy. The backlog remains valuable for cleanup regardless.

---

## Epic 0: Prework and Baselines ✅ COMPLETED

*Establish baselines and verify assumptions before implementation.*

### Story 0.1: Baseline Linear Trees Performance and Verify Issue ✅

**Status**: Completed  
**Completed**: 2025-01-27

**Findings**:
- Linear trees on covertype: boosters=0.7611 vs lightgbm=0.4064 (87% worse)
- Root cause confirmed: `BinnedSample::feature()` uses `bin_to_midpoint()` at dataset.rs:888-892
- Documented in `tmp/ndarray-migration-baselines.md`

---

### Story 0.2: Verify Strided Paths Are Dead Code ✅

**Status**: Completed  
**Completed**: 2025-01-27

**Findings**:
- RowMajor was only used in tests (lines 969, 997, 1008 in dataset.rs)
- Production code always used ColumnMajor
- All strided code paths removed in Epic 1

---

### Story 0.3: Verify ndarray CowArray Availability ✅

**Status**: Completed  
**Completed**: 2025-01-27

**Findings**:
- ndarray version: 0.16.1
- CowArray: Available and suitable
- Import: `use ndarray::CowArray;`

---

## Epic 1: Storage Types Foundation ✅ COMPLETED

*Create the new typed storage hierarchy that encodes numeric vs categorical semantics.*

**Milestone**: After this epic, new storage types exist but aren't used yet.

**Completed**: 2025-12-28  
**Commit**: 6acd69f  
**LOC Added**: ~1458 lines (including tests)  
**Tests**: 39 new tests for all storage types

### Story 1.1: Create BinData Enum and Module Structure ✅

**Status**: Completed  
**Estimate**: ~80 LOC

**Description**: Create storage module and `BinData` enum replacing `BinType`.

**Tasks**:

- Create `data/binned/storage/` module structure
- Create `BinData::U8(Box<[u8]>)` and `BinData::U16(Box<[u16]>)` variants
- Add `is_u8()`, `is_u16()`, `len()` helper methods
- Add `get(idx) -> u32` accessor

**Definition of Done**:

- `BinData` enum implemented with all accessors
- Unit tests for construction and access
- Module structure: `storage/mod.rs`, `storage/bin_data.rs`

**Public API Added**: `BinData`, `BinData::U8`, `BinData::U16`

**Testing**:

- Unit tests for U8/U16 construction
- Property test: get(i) returns correct value for all indices

---

### Story 1.2: Create NumericStorage and CategoricalStorage ✅

**Status**: Completed  
**Estimate**: ~120 LOC  
**Depends on**: 1.1

**Description**: Create dense storage types for numeric and categorical features.

**Tasks**:

- Create `NumericStorage` with `bins: BinData` and `raw_values: Box<[f32]>`
- Create `CategoricalStorage` with `bins: BinData` only
- Implement `bin()`, `raw()`, `raw_slice()` accessors for NumericStorage
- Implement `bin()` accessor for CategoricalStorage

**Definition of Done**:

- Both storage types implemented
- Column-major layout (feature × samples)
- Unit tests for access patterns

**Public API Added**: `NumericStorage`, `CategoricalStorage`

**Testing**:

- Unit tests for bin/raw access at various indices
- Verify column-major indexing: `idx = feature_in_group * n_samples + sample`

---

### Story 1.3: Create Sparse Storage Types ✅

**Status**: Completed  
**Estimate**: ~100 LOC  
**Depends on**: 1.1

**Description**: Create CSC-like sparse storage for numeric and categorical features.

**Tasks**:

- Create `SparseNumericStorage` with sample_indices, bins, raw_values
- Create `SparseCategoricalStorage` with sample_indices, bins
- Implement binary search `get()` accessor
- Document sparse semantics (zeros are meaningful, not missing)

**Definition of Done**:

- Both sparse types implemented
- O(log nnz) random access via binary search
- Unit tests for sparse access

**Public API Added**: `SparseNumericStorage`, `SparseCategoricalStorage`

**Testing**:

- Unit tests for samples in/not in indices
- Verify implicit (0, 0.0) for missing samples

---

### Story 1.4: Create BundleStorage ✅

**Status**: Completed  
**Estimate**: ~80 LOC  
**Depends on**: 1.1

**Description**: Consolidate EFB bundle handling into dedicated storage type.

**Tasks**:

- Create `BundleStorage` with encoded_bins, feature_indices, bin_offsets, etc.
- Implement `decode()` method (encoded_bin → original feature + bin)
- Migrate logic from `bundling.rs`

**Definition of Done**:

- `BundleStorage` implemented with decode
- Lossless encoding verified
- Integration with existing EFB logic

**Public API Added**: `BundleStorage`

**Testing**:

- Unit tests for encode/decode roundtrip
- Property test: decode(encode(f, b)) == (f, b)

---

### Story 1.5: Create FeatureStorage Enum ✅

**Status**: Completed  
**Estimate**: ~50 LOC  
**Depends on**: 1.2, 1.3, 1.4

**Description**: Create unified enum wrapping all storage types.

**Tasks**:

- Create `FeatureStorage` with Numeric, Categorical, SparseNumeric, SparseCategorical, Bundle variants
- Add `has_raw_values()` helper
- Add `is_categorical()` helper

**Definition of Done**:

- `FeatureStorage` enum complete
- All storage types wrapped
- Pattern matching examples documented

**Public API Added**: `FeatureStorage`, `FeatureStorageType`

**Testing**:

- Unit tests for helper methods
- Verify exhaustive matching

---

### Story 1.6: Stakeholder Feedback Check - Epic 1 ✅

**Status**: Completed

**Description**: Review stakeholder feedback after storage types are complete.

**Tasks**:

- Check `workdir/tmp/stakeholder_feedback.md` for relevant input
- Discuss feedback with team
- Create follow-up stories if needed

**Definition of Done**:

- Feedback reviewed and documented
- Any new stories added to backlog

---

## Epic 2: FeatureGroup Migration

*Update `FeatureGroup` to use new storage types and remove dead code.*

**Milestone**: After this epic, FeatureGroup uses new storage; old API still works.

### Story 2.1: Update FeatureGroup to Use FeatureStorage

**Status**: Not Started  
**Estimate**: ~100 LOC  
**Depends on**: Epic 1 complete

**Description**: Replace current FeatureGroup internals with FeatureStorage.

**Tasks**:

- Replace bins/data fields with `storage: FeatureStorage`
- Remove `layout` field (always column-major)
- Update all FeatureGroup methods
- Migrate from `BinStorage` to `FeatureStorage`

**Definition of Done**:

- FeatureGroup uses FeatureStorage internally
- All existing tests pass
- No layout field

**Public API Changed**: `FeatureGroup` internals (internal)

**Testing**:

- Existing FeatureGroup tests pass
- New tests for storage access

---

### Story 2.2: Simplify FeatureView

**Status**: Not Started  
**Estimate**: ~60 LOC  
**Depends on**: 2.1

**Description**: Remove stride from FeatureView (always 1 now).

**Tasks**:

- Update `FeatureView` enum to remove stride field
- Rename `row_indices` → `sample_indices`
- Update all histogram kernels (lines 738-745 in dataset.rs)

**Definition of Done**:

- FeatureView has 4 variants (U8, U16, SparseU8, SparseU16)
- No stride field
- All histogram tests pass

**Public API Changed**: `FeatureView` (stride field removed)

**Testing**:

- Histogram building benchmarks show no regression
- All training tests pass

---

### Story 2.3: Add Runtime Assertions for Homogeneous Groups

**Status**: Not Started  
**Estimate**: ~30 LOC  
**Depends on**: 2.1

**Description**: Ensure groups are homogeneous (all numeric OR all categorical).

**Tasks**:

- Add debug assertions in group construction
- Add `has_raw_values()` and `is_categorical()` to FeatureGroup
- Document invariant

**Definition of Done**:

- Assertions catch mixed groups in debug builds
- Helper methods implemented

**Public API Added**: `FeatureGroup::has_raw_values()`, `FeatureGroup::is_categorical()`

**Testing**:

- Test that mixed groups panic in debug
- Test helper methods

---

### Story 2.4: Review/Demo - Storage and FeatureGroup

**Status**: Not Started

**Description**: Demo storage types and FeatureGroup migration.

**Tasks**:

- Prepare demo showing new storage hierarchy
- Show benchmark results (no regression)
- Show lines of code removed
- Document in `workdir/tmp/development_review_*.md`

**Definition of Done**:

- Demo completed
- Review documented
- No performance regression confirmed

---

## Epic 3: Dataset Builder Unification

*Create unified `DatasetBuilder` with batch and single-feature APIs.*

### Story 3.1a: Feature Analysis for Batch Ingestion

**Status**: Not Started  
**Estimate**: ~100 LOC  
**Depends on**: Epic 1 complete

**Description**: Implement feature analysis for auto-detection of types and bin widths.

**Tasks**:

- Create `FeatureAnalysis` struct with detected type, bin count, sparsity
- Implement `analyze_feature(column: ArrayView1<f32>, config: &BinningConfig) -> FeatureAnalysis`
- Auto-detect: numeric vs categorical (based on cardinality)
- Auto-detect: required bins (U8 vs U16)
- Auto-detect: sparsity

**Definition of Done**:

- Feature analysis function implemented
- Returns correct type, bin width, sparsity for test cases

**Public API Added**: `FeatureAnalysis` (internal)

**Testing**:

- Test numeric detection (continuous values)
- Test categorical detection (low cardinality integers)
- Test bin width detection
- Test sparsity detection

---

### Story 3.1b: Homogeneous Group Building Strategy

**Status**: Not Started  
**Estimate**: ~100 LOC  
**Depends on**: 3.1a

**Description**: Implement grouping strategy that partitions features by type and bin width.

**Tasks**:

- Implement `GroupingStrategy` that assigns features to groups
- Partition: numeric_u8, numeric_u16, categorical_u8, categorical_u16, sparse
- Create `GroupSpec` for each group with feature indices

**Definition of Done**:

- Grouping strategy implemented
- Features correctly partitioned by type/width

**Public API Added**: `GroupingStrategy` (internal)

**Testing**:

- Test with mixed feature types
- Test with mixed bin widths
- Verify homogeneous groups

---

### Story 3.1c: from_array Builder Integration

**Status**: Not Started  
**Estimate**: ~100 LOC  
**Depends on**: 3.1a, 3.1b

**Description**: Wire feature analysis and grouping into DatasetBuilder.

**Tasks**:

- Implement `DatasetBuilder::from_array(data: ArrayView2<f32>, config: &BinningConfig)`
- Use feature analysis to detect types
- Use grouping strategy to build homogeneous groups
- Store raw values for numeric features

**Definition of Done**:

- `from_array()` implemented and working
- Raw values stored
- Homogeneous groups created

**Public API Added**: `DatasetBuilder::from_array()`

**Testing**:

- End-to-end test: matrix → builder → dataset
- Verify raw values accessible
- Verify grouping correct

---

### Story 3.2: Create FeatureMetadata Support

**Status**: Not Started  
**Estimate**: ~80 LOC  
**Depends on**: 3.1c

**Description**: Implement `from_array_with_metadata()` for user-specified metadata.

**Tasks**:

- Create `FeatureMetadata` struct with names, categorical_features, max_bins (HashMap)
- Implement builder pattern methods: `names()`, `categorical()`, `max_bins_for()`
- Implement `from_array_with_metadata()`

**Definition of Done**:

- `FeatureMetadata` implemented
- User can specify names, categorical columns, per-feature max_bins
- Metadata overrides auto-detection

**Public API Added**: `FeatureMetadata`, `DatasetBuilder::from_array_with_metadata()`

**Testing**:

- Test with partial metadata
- Test max_bins overrides
- Test categorical feature specification

---

### Story 3.3: Update Single-Feature API

**Status**: Not Started  
**Estimate**: ~60 LOC  
**Depends on**: 3.1c

**Description**: Update `add_feature()`, `add_numeric()`, `add_categorical()` methods.

**Tasks**:

- Update methods to store raw values
- Defer grouping to `build()`
- Implement homogeneous group building at build time

**Definition of Done**:

- Single-feature API stores raw values
- Grouping happens at build()
- Existing tests pass

**Public API Changed**: `add_feature()`, `add_numeric()`, `add_categorical()` now store raw values

**Testing**:

- Test mixed feature types via single-feature API
- Verify grouping at build time

---

### Story 3.4: Stakeholder Feedback Check - Epic 3

**Status**: Not Started

**Description**: Review stakeholder feedback after builder unification.

**Tasks**:

- Check `workdir/tmp/stakeholder_feedback.md`
- Incorporate feedback
- Create follow-up stories if needed

**Definition of Done**:

- Feedback reviewed
- Any new stories added

---

## Epic 4: Dataset API and Access Methods

*Implement access methods on BinnedDataset for raw values and introspection.*

**Milestone**: After this epic, full new API is available.

### Story 4.1: Implement FeatureLocation and Introspection

**Status**: Not Started  
**Estimate**: ~80 LOC  
**Depends on**: Epic 2

**Description**: Add feature location tracking and introspection APIs.

**Tasks**:

- Create `FeatureLocation` enum (Direct, Bundled, Skipped)
- Add `feature_locations` to BinnedDataset
- Implement `feature_storage_type()`, `bundle_features()`

**Definition of Done**:

- Feature location tracking complete
- Introspection APIs working

**Public API Added**: `FeatureLocation`, `BinnedDataset::feature_storage_type()`, `BinnedDataset::bundle_features()`

**Testing**:

- Test all FeatureLocation variants
- Test introspection on mixed datasets

---

### Story 4.2: Implement Raw Value Access Methods

**Status**: Not Started  
**Estimate**: ~100 LOC  
**Depends on**: Epic 3 (builder stores raw values)

**Description**: Add methods for accessing raw feature values.

**Tasks**:

- Implement `raw_value(sample, feature) -> Option<f32>`
- Implement `raw_feature_slice(feature) -> Option<&[f32]>`
- Implement `raw_feature_iter()` for zero-allocation iteration
- Implement `raw_numeric_matrix() -> Option<CowArray<f32, Ix2>>`

**Definition of Done**:

- All raw access methods implemented
- Returns None for categorical/bundled features
- Uses ndarray types

**Public API Added**: `raw_value()`, `raw_feature_slice()`, `raw_feature_iter()`, `raw_numeric_matrix()`

**Testing**:

- Test raw access for numeric features
- Test None for categorical
- Test CowArray allocation behavior

---

### Story 4.3: Implement RowBlocks for Prediction

**Status**: Not Started  
**Estimate**: ~120 LOC  
**Depends on**: 4.2

**Description**: Create buffered row block iterator for efficient prediction.

**Tasks**:

- Create `RowBlocks` struct in dataset module
- Implement `for_each()`, `for_each_with(Parallelism)`, `flat_map_with()`
- Implement `block()`, `block_indices()` for external iteration
- Use ndarray `ArrayView2` for blocks

**Definition of Done**:

- RowBlocks implemented with parallelism support
- Uses existing `Parallelism` enum
- Returns ndarray views

**Public API Added**: `RowBlocks`, `BinnedDataset::row_blocks()`

**Testing**:

- Test sequential iteration
- Test parallel iteration
- Benchmark: ~2x speedup vs column access

---

### Story 4.4: Migrate Prediction Code to RowBlocks

**Status**: Not Started  
**Estimate**: ~-50 LOC (net removal)  
**Depends on**: 4.3

**Description**: Update prediction code to use RowBlocks instead of custom transpose.

**Tasks**:

- Identify prediction code with transpose logic
- Replace with `dataset.row_blocks()`
- Remove duplicate transpose implementations

**Definition of Done**:

- Prediction uses RowBlocks
- Duplicate code removed
- No regression in prediction speed

**Public API Removed**: Custom prediction transpose code (internal)

**Testing**:

- End-to-end prediction tests
- Benchmark prediction performance

---

### Story 4.5: Add describe() and Debugging APIs

**Status**: Not Started  
**Estimate**: ~60 LOC

**Description**: Add debugging and summary methods.

**Tasks**:

- Implement `describe() -> DatasetSummary`
- Include: n_samples, n_features, n_numeric, n_categorical, n_bundled, memory
- Add `raw_storage_bytes()`

**Definition of Done**:

- `describe()` returns useful summary
- Memory accounting accurate

**Public API Added**: `BinnedDataset::describe()`, `DatasetSummary`, `BinnedDataset::raw_storage_bytes()`

**Testing**:

- Test summary on various datasets
- Verify memory calculations

---

### Story 4.6: Review/Demo - Dataset API

**Status**: Not Started

**Description**: Demo raw access APIs and RowBlocks.

**Tasks**:

- Demo raw value access for linear trees
- Demo RowBlocks for prediction
- Show memory usage stats
- Document in `workdir/tmp/development_review_*.md`

**Definition of Done**:

- Demo completed
- Review documented

---

## Epic 5: Serialization

*Update serialization to handle raw values.*

### Story 5.1: Update Serialization Format

**Status**: Not Started  
**Estimate**: ~100 LOC

**Description**: Include raw values in serialization by default.

**Tasks**:

- Generate test file with OLD format BEFORE changes (for Story 5.3)
- Update save format to include raw values
- Add version field for compatibility
- Implement `save()` with raw values

**Definition of Done**:

- Serialization includes raw values
- Version field present
- Old format test file committed

**Testing**:

- Roundtrip test: save → load → compare
- Test with various feature types

---

### Story 5.2: Implement save_without_raw

**Status**: Not Started  
**Estimate**: ~50 LOC  
**Depends on**: 5.1

**Description**: Add option to save without raw values for smaller files.

**Tasks**:

- Implement `save_without_raw()`
- Document that linear trees won't work after reload

**Definition of Done**:

- `save_without_raw()` implemented
- File size measurably smaller

**Public API Added**: `BinnedDataset::save_without_raw()`

**Testing**:

- Compare file sizes
- Verify `has_raw_values() = false` after reload

---

### Story 5.3: Backward Compatibility

**Status**: Not Started  
**Estimate**: ~50 LOC  
**Depends on**: 5.1

**Description**: Handle loading old format files.

**Tasks**:

- Use test file generated in Story 5.1
- Detect old format via version field
- Load with `has_raw_values() = false`
- Document migration path

**Definition of Done**:

- Old files load successfully
- Clear error/warning for linear trees

**Testing**:

- Test loading old format test file
- Verify graceful degradation

---

## Epic 6: Cleanup and Removal

*Remove deprecated code and consolidate.*

### Story 6.1: Remove GroupLayout Enum

**Status**: Not Started  
**Estimate**: ~-50 LOC

**Description**: Delete unused GroupLayout enum and related code.

**Tasks**:

- Remove `GroupLayout` enum from `storage.rs`
- Remove layout field from FeatureGroup
- Update all references
- Update/remove `layout_benchmark.rs` example

**Definition of Done**:

- GroupLayout deleted
- No compilation errors
- ~50 lines removed

**Public API Removed**: `GroupLayout`

**Testing**:

- All tests pass
- Grep confirms no remaining references

---

### Story 6.2: Remove BinType Enum

**Status**: Not Started  
**Estimate**: ~-20 LOC  
**Depends on**: Epic 1 (BinData exists)

**Description**: Delete redundant BinType enum (replaced by BinData).

**Tasks**:

- Remove `BinType` enum
- Update any remaining references to use `BinData::is_u8()`

**Definition of Done**:

- BinType deleted
- ~20 lines removed

**Public API Removed**: `BinType`

**Testing**:

- All tests pass

---

### Story 6.3: Remove Strided Histogram Kernels

**Status**: Not Started  
**Estimate**: ~-60 LOC  
**Depends on**: 2.2 (FeatureView simplified)

**Description**: Delete dead strided access code in histogram building.

**Tasks**:

- Remove strided variants from histogram kernels
- Specifically: lines 738-745 in dataset.rs and related code
- Simplify match arms

**Definition of Done**:

- Strided code deleted
- ~60 lines removed
- 4 match arms instead of 6

**Testing**:

- Histogram tests pass
- Benchmark shows no regression

---

### Story 6.4: Consolidate Bundle Code

**Status**: Not Started  
**Estimate**: ~-150 LOC  
**Depends on**: 1.4 (BundleStorage exists)

**Description**: Remove `BundledColumns`, `BundlePlan` (now in BundleStorage).

**Tasks**:

- Remove `BundledColumns` struct
- Remove `bundle_plan`, `bundled_columns` fields
- Update all callers to use BundleStorage

**Definition of Done**:

- Old bundle structs deleted
- ~150 lines removed

**Public API Removed**: `BundledColumns`, `BundlePlan` (if public)

**Testing**:

- EFB tests pass
- Bundling benchmarks stable

---

### Story 6.5: Deprecate Dataset Type

**Status**: Not Started  
**Estimate**: ~20 LOC + docs

**Description**: Mark old Dataset type as deprecated.

**Tasks**:

- Add `#[deprecated]` attribute with migration message
- Add migration guide in `docs/migration/dataset-to-binneddataset.md`
- Update examples to use BinnedDataset

**Definition of Done**:

- Dataset deprecated with clear message
- Migration guide written
- Examples updated

**Public API Deprecated**: `Dataset`

**Testing**:

- Deprecation warning shown when Dataset used

---

### Story 6.6: Cleanup Verification

**Status**: Not Started

**Description**: Verify all deprecated symbols are removed.

**Tasks**:

- Grep for: `GroupLayout`, `BinType`, `BundledColumns`, `BundlePlan`
- Verify no remaining usages except deprecation notices
- Update `mod.rs` exports

**Definition of Done**:

- All removed symbols confirmed gone
- Public API exports updated

---

### Story 6.7: Stakeholder Feedback Check - Cleanup

**Status**: Not Started

**Description**: Final feedback check before validation.

**Tasks**:

- Check `workdir/tmp/stakeholder_feedback.md`
- Ensure nothing missed
- Create follow-up stories if needed

**Definition of Done**:

- All feedback addressed or captured

---

## Epic 7: Validation and Quality Gates

*Verify the changes meet quality targets.*

### Story 7.1: Linear Trees Quality Validation

**Status**: Not Started

**Description**: Verify linear trees quality is same or better.

**Tasks**:

- Run linear trees on covertype
- Compare with baseline from Story 0.1
- Target: same or better mlogloss (no regression)

**Definition of Done**:

- Covertype mlogloss same or better than baseline
- Results documented

**Testing**:

- End-to-end covertype benchmark
- Comparison table in review

---

### Story 7.2: Performance Benchmarks

**Status**: Not Started

**Description**: Ensure no performance regression.

**Tasks**:

- Run `bench_histogram_building`
- Run `bench_training_covertype`
- Measure memory overhead (raw values vs baseline)

**Definition of Done**:

- No regression >5% in histogram building
- Memory overhead <50% of binned size
- Results documented

**Testing**:

- Criterion benchmarks
- Memory profiling

---

### Story 7.3: Integration Tests

**Status**: Not Started

**Description**: End-to-end tests for all features.

**Specific Test Scenarios**:

1. Mixed feature types (numeric + categorical)
2. All categorical (no raw values)
3. All bundled (EFB)
4. Empty dataset (0 samples, 0 features)
5. Single sample dataset
6. Serialization roundtrip with raw values
7. Serialization roundtrip without raw values
8. RowBlocks prediction

**Definition of Done**:

- All 8 scenarios pass

**Testing**:

- Integration test suite

---

### Story 7.4: Property-Based Tests

**Status**: Not Started

**Description**: Add property-based tests for invariants.

**Properties to Test**:

1. Roundtrip: build → serialize → deserialize → same data
2. Invariant: numeric groups have `raw_values.len() == n_features * n_samples`
3. Invariant: categorical groups return None for raw access
4. Invariant: `feature_views().len()` equals training feature count

**Definition of Done**:

- Property tests pass with proptest
- At least 1000 iterations per property

**Testing**:

- proptest integration

---

### Story 7.5: Final Review/Demo

**Status**: Not Started

**Description**: Final demo of completed work.

**Tasks**:

- Demo linear trees with raw values
- Show code reduction stats (target: ~130 net lines removed)
- Present benchmark results
- Document in `workdir/tmp/development_review_*.md`

**Definition of Done**:

- Review completed
- All quality gates passed

---

### Story 7.6: Retrospective

**Status**: Not Started

**Description**: Team retrospective on implementation.

**Tasks**:

- Reflect on what went well/not well
- Identify improvements
- Document in `workdir/tmp/retrospective.md`
- Create follow-up stories for action items

**Definition of Done**:

- Retrospective completed
- Action items captured as stories (if any)

---

## Summary

| Epic | Stories | Estimated LOC | Description |
|------|---------|---------------|-------------|
| 0. Prework | 3 | 0 | Baselines and verification |
| 1. Storage Types | 6 | +430 | BinData, NumericStorage, etc. |
| 2. FeatureGroup Migration | 4 | +90 | Use new storage, simplify FeatureView |
| 3. Builder Unification | 5 | +440 | from_array, FeatureMetadata |
| 4. Dataset API | 6 | +310 | Raw access, RowBlocks |
| 5. Serialization | 3 | +200 | Raw values in format |
| 6. Cleanup | 7 | -530 | Remove dead code |
| 7. Validation | 6 | +50 | Tests and quality gates |
| **Total** | **40** | **~-130 net** | |

### Dependency Graph

```text
Epic 0 (Prework)
    ↓
Epic 1 (Storage Types)
    ↓
Epic 2 (FeatureGroup) ←──────────────┐
    ↓                                 │
Epic 3 (Builder) ─────────────────────┤
    ↓                                 │
Epic 4 (API) ←────────────────────────┘
    ↓
Epic 5 (Serialization)
    ↓
Epic 6 (Cleanup) [can partially overlap with Epics 2-5]
    ↓
Epic 7 (Validation)
```

### Key Dates

- **Created**: 2025-12-28
- **Refinement Complete**: 2025-12-28 (6 rounds)
- **Ready for Implementation**: Yes
