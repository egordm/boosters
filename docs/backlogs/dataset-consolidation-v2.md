# Backlog: Dataset Type Consolidation

**RFC**: [RFC-0019](../rfcs/rfc-0019-raw-features-view.md), extends RFC-0018  
**Created**: 2025-12-30  
**Status**: In Progress  

## Overview

Consolidate to a single `Dataset` type (currently `BinnedDataset`, renamed after migration).

**Current State**:
- `types::Dataset` - Legacy raw feature container (deprecated)
- `BinnedDataset` - Unified type with binned + raw features

**Goal**: Delete `types::Dataset`, rename `BinnedDataset` → `Dataset`.

## Completed Work

### ✅ Epic 0: Baselines (Complete)

Baselines captured in `docs/benchmarks/dataset-consolidation-baseline.md`:
- BinnedDataset creation: ~16x overhead (binning cost) - acceptable since one-time
- Memory overhead: +23% to +47% - acceptable
- Decision: Proceed with consolidation

### ✅ Epic 2: GBDT Prediction (Complete)

GBDT prediction works with BinnedDataset via `SampleBlocks::for_each_with()`:
- 10.6% overhead measured - within 10% threshold
- No new traits or interfaces added
- Tests pass

---

## Active Work

## Epic 3: GBLinear with RawFeaturesView

*Enable GBLinear to work with BinnedDataset via RawFeaturesView abstraction.*

**Approach**: Create `RawFeaturesView` enum that wraps either `ArrayView2<f32>` or `&BinnedDataset`. GBLinear methods accept `impl Into<RawFeaturesView>`, allowing both paths during migration.

See [RFC-0019](../rfcs/rfc-0019-raw-features-view.md) for design details.

### Story 3.1: Create RawFeaturesView Type

**Status**: Not Started  
**Estimate**: 1.5 hours

**Location**: `crates/boosters/src/data/raw_features_view.rs`

**Implementation**:
```rust
pub enum RawFeaturesView<'a> {
    Array(ArrayView2<'a, f32>),
    Binned(&'a BinnedDataset),
}

impl RawFeaturesView<'_> {
    pub fn n_features(&self) -> usize;
    pub fn n_samples(&self) -> usize;
    pub fn feature(&self, idx: usize) -> Option<&[f32]>;
    pub fn iter_features(&self) -> impl Iterator<Item = (usize, &[f32])>;
}

// From impls for ArrayView2, FeaturesView, &BinnedDataset
```

**Definition of Done**:
- Type compiles and has unit tests
- From impls work for all sources
- `iter_features()` skips None features

---

### Story 3.2: Update LinearModel Prediction

**Status**: Not Started  
**Estimate**: 1 hour

**Change**: `LinearModel::predict_into()` accepts `impl Into<RawFeaturesView<'_>>`

**Before**:
```rust
pub fn predict_into(&self, features: FeaturesView<'_>, output: ArrayViewMut2<'_, f32>)
```

**After**:
```rust
pub fn predict_into(&self, features: impl Into<RawFeaturesView<'_>>, output: ArrayViewMut2<'_, f32>)
```

**Definition of Done**:
- Existing callers using FeaturesView still work (via From impl)
- New callers can pass &BinnedDataset directly
- Tests pass
- No allocation in hot path

---

### Story 3.3: Update Updater for Training

**Status**: Not Started  
**Estimate**: 1.5 hours

**Change**: `Updater::update_round()` and `apply_weight_deltas_to_predictions()` accept RawFeaturesView.

**Methods to update**:
- `update_round(model, data, buffer, selector, output)` - `data` param
- `apply_weight_deltas_to_predictions(data, deltas, output, predictions)` - `data` param
- `compute_weight_update(model, data, buffer, feature, output, config)` - internal

**Definition of Done**:
- Updater works with both FeaturesView and BinnedDataset
- Training tests pass
- No duplicate code

---

### Story 3.4: Update GBLinearTrainer

**Status**: Not Started  
**Estimate**: 1 hour

**Change**: `GBLinearTrainer::train()` internally converts to RawFeaturesView.

Eventually (after Epic 5), signature changes to accept `&BinnedDataset` directly.
For now, keep `&Dataset` signature but internally use RawFeaturesView.

**Definition of Done**:
- Train still works with types::Dataset
- Tests pass
- Ready for signature change in Epic 5

---

### Story 3.5: Update GBLinearModel Prediction

**Status**: Not Started  
**Estimate**: 1 hour

**Change**: High-level `GBLinearModel::predict()` works with BinnedDataset.

- `predict_raw(dataset)` - accepts types::Dataset (deprecated)
- Add RawFeaturesView path internally

After Epic 5, signature changes to `predict(&BinnedDataset)`.

**Definition of Done**:
- Prediction works with both Dataset types
- Tests pass

---

### Story 3.6: Benchmark GBLinear Overhead

**Status**: Not Started  
**Estimate**: 30 min

**Measurements**:
| Operation | FeaturesView | RawFeaturesView(Binned) | Overhead |
|-----------|--------------|-------------------------|----------|
| Training  | baseline     | measure                 | <5%?     |
| Prediction| baseline     | measure                 | <5%?     |

**Definition of Done**:
- Benchmark numbers captured
- Overhead within acceptable threshold

---

## Epic 4: Python Bindings

*Migrate Python bindings to use BinnedDataset.*

### Story 4.1: Create PyDataset Wrapper

**Status**: Not Started  
**Estimate**: 2 hours

**Change**: `PyDataset` internally wraps `BinnedDataset` instead of `types::Dataset`.

**Implementation**:
- Constructor: Use `DatasetBuilder::from_array()` with binning config
- Keep same Python API
- Use feature-major storage

**Definition of Done**:
- PyDataset compiles with BinnedDataset
- Basic unit tests pass

---

### Story 4.2: Update PyGBDTModel

**Status**: Not Started  
**Estimate**: 1.5 hours

**Changes**:
- `fit()`: Pass BinnedDataset to trainer (already does this)
- `predict()`: Use SampleBlocks path

**Definition of Done**:
- All GBDT Python tests pass
- No deprecated API usage

---

### Story 4.3: Update PyGBLinearModel

**Status**: Not Started  
**Estimate**: 1.5 hours

**Changes**:
- `fit()`: Eventually pass BinnedDataset to trainer
- `predict()`: Use RawFeaturesView path

**Definition of Done**:
- All GBLinear Python tests pass

---

### Story 4.4: Python Integration Tests

**Status**: Not Started  
**Estimate**: 1 hour

Run full Python test suite, fix any regressions.

---

## Epic 5: Cleanup

*Delete deprecated code and rename types.*

### Story 5.1: Delete types::Dataset

**Status**: Not Started  
**Estimate**: 2 hours

**Actions**:
1. Delete `crates/boosters/src/data/types/` module
2. Update all imports
3. Fix compile errors

**Prerequisite**: All usages migrated in Epics 3-4.

**Definition of Done**:
- `types::Dataset` deleted
- Code compiles
- Tests pass

---

### Story 5.2: Rename BinnedDataset to Dataset

**Status**: Not Started  
**Estimate**: 1 hour

**Actions**:
1. Rename `BinnedDataset` → `Dataset`
2. Update all imports and docs
3. Keep `BinnedDataset` as deprecated type alias for one release

**Definition of Done**:
- Primary type is `Dataset`
- Old name works as alias

---

### Story 5.3: Simplify RawFeaturesView

**Status**: Not Started  
**Estimate**: 30 min

**Change**: After `types::Dataset` is gone, consider:
- Remove `Array` variant from enum
- Or replace with direct `&Dataset` parameter

**Definition of Done**:
- Cleaner API with single path

---

### Story 5.4: Remove Dead Code

**Status**: Not Started  
**Estimate**: 1 hour

**Actions**:
- Delete `deprecated/` folder if exists
- Remove `#![allow(dead_code)]` from binned module
- Delete any unused helper methods

**Definition of Done**:
- No dead code
- No allow(dead_code) attributes

---

### Story 5.5: Final Validation

**Status**: Not Started  
**Estimate**: 1 hour

**Checklist**:
- [ ] All tests pass
- [ ] Python tests pass
- [ ] Benchmarks show acceptable performance
- [ ] No deprecated warnings in library code
- [ ] Documentation updated

---

## Epic 6: Review and Retrospective

### Story 6.1: Final Review

**Status**: Not Started  
**Estimate**: 30 min

Document what was delivered:
- Lines of code removed
- Performance impact
- API changes

---

### Story 6.2: Retrospective

**Status**: Not Started  
**Estimate**: 30 min

Write to `tmp/retrospective.md`:
- What went well
- What didn't go well
- Lessons learned

---

## Summary

| Epic | Stories | Status |
|------|---------|--------|
| 0: Baselines | 5 | ✅ Complete |
| 2: GBDT Prediction | 5 | ✅ Complete |
| 3: GBLinear | 6 | Not Started |
| 4: Python | 4 | Not Started |
| 5: Cleanup | 5 | Not Started |
| 6: Review | 2 | Not Started |
| **Total** | **27** | |

**Estimated remaining**: ~16 hours
