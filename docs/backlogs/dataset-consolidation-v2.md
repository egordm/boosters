# Backlog: Dataset Type Consolidation

**RFC**: [RFC-0019](../rfcs/rfc-0019-feature-value-iterator.md)  
**Created**: 2025-12-30  
**Status**: In Progress  

## Overview

Consolidate to a single `Dataset` type (currently `BinnedDataset`, renamed after migration).

**Current State**:
- `types::Dataset` - Legacy raw feature container (deprecated)
- `BinnedDataset` - Unified type with binned + raw features

**Goal**: Delete `types::Dataset`, rename `BinnedDataset` → `Dataset`.

**Key Principle**: No legacy compatibility layers. Delete old code, update callers, fix compile errors. Library may not compile during migration—that's fine.

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

## Epic 3: GBLinear with FeatureValueIter

*Enable GBLinear to work with BinnedDataset via zero-cost feature value iteration.*

**Approach**: Add `FeatureValueIter` enum to `BinnedDataset` that yields `(sample_idx, f32)` pairs. This handles:
- Dense features: direct slice iteration (zero-cost)
- Bundled features (EFB): extracts values from bundle encoding
- Sparse features: yields only non-zero samples

**No legacy compatibility**. We update GBLinear to take `&BinnedDataset` directly.

See [RFC-0019](../rfcs/rfc-0019-feature-value-iterator.md) for design details.

### Story 3.1: Add FeatureValueIter to BinnedDataset

**Status**: Not Started  
**Estimate**: 2 hours

**Implementation**:

```rust
/// Iterator yielding (sample_idx, raw_value) for a feature.
pub enum FeatureValueIter<'a> {
    Dense(std::iter::Enumerate<std::slice::Iter<'a, f32>>),
    Bundled(BundledFeatureIter<'a>),
    Sparse(SparseFeatureIter<'a>),
}

impl BinnedDataset {
    /// Iterate over raw values for a feature.
    pub fn feature_values(&self, feature: usize) -> FeatureValueIter<'_>;
    
    /// Iterate over all numeric features.
    pub fn iter_feature_values(&self) -> impl Iterator<Item = (usize, FeatureValueIter<'_>)>;
}
```

**Location**: `crates/boosters/src/data/binned/feature_iter.rs`

**Definition of Done**:
- FeatureValueIter compiles and implements Iterator
- Dense path works and has unit tests
- Bundled path works (or panics with TODO if EFB extraction not ready)
- Sparse path works (or skips sparse features for now)

---

### Story 3.2: Update LinearModel Prediction

**Status**: Not Started  
**Estimate**: 1.5 hours

**Change**: `LinearModel::predict_into()` takes `&BinnedDataset` directly.

**Before**:
```rust
pub fn predict_into(&self, features: FeaturesView<'_>, output: ArrayViewMut2<'_, f32>) {
    for feat_idx in 0..n_features {
        let feature_values = features.feature(feat_idx);
        for (sample_idx, &value) in feature_values.iter().enumerate() {
            // ...
        }
    }
}
```

**After**:
```rust
pub fn predict_into(&self, dataset: &BinnedDataset, output: ArrayViewMut2<'_, f32>) {
    for feat_idx in 0..n_features {
        for (sample_idx, value) in dataset.feature_values(feat_idx) {
            // ...
        }
    }
}
```

**Definition of Done**:
- Signature changed
- All callers updated (will not compile until fixed)
- Tests pass after all callers fixed

---

### Story 3.3: Update Updater for Training

**Status**: Not Started  
**Estimate**: 2 hours

**Change**: `Updater::update_round()` and internal functions take `&BinnedDataset`.

**Methods to update**:
- `update_round(model, data: &BinnedDataset, buffer, selector, output)`
- `apply_weight_deltas_to_predictions(data: &BinnedDataset, deltas, output, predictions)`
- `compute_weight_update(model, data: &BinnedDataset, buffer, feature, output, config)`

**Definition of Done**:
- All signatures changed
- Inner loop uses `data.feature_values(feature)`
- Tests pass

---

### Story 3.4: Update GBLinearTrainer

**Status**: Not Started  
**Estimate**: 1.5 hours

**Change**: `GBLinearTrainer::train()` takes `&BinnedDataset`.

**Before**:
```rust
pub fn train(&self, dataset: &Dataset, ...) -> GBLinearModel
```

**After**:
```rust
pub fn train(&self, dataset: &BinnedDataset, ...) -> GBLinearModel
```

**Definition of Done**:
- Signature changed
- Internally uses feature_values() iteration
- Tests pass

---

### Story 3.5: Update GBLinearModel High-Level API

**Status**: Not Started  
**Estimate**: 1 hour

**Changes**:
- `predict()` takes `&BinnedDataset`
- `predict_raw()` - delete (was for old Dataset)

**Definition of Done**:
- Clean API with single path
- Tests pass

---

### Story 3.6: Benchmark GBLinear Overhead

**Status**: Not Started  
**Estimate**: 30 min

**Measurements**:
| Operation | Old (FeaturesView) | New (FeatureValueIter) | Overhead |
|-----------|-------------------|------------------------|----------|
| Training  | baseline          | measure                | <5%?     |
| Prediction| baseline          | measure                | <5%?     |

**Definition of Done**:
- Benchmark numbers captured
- Overhead within acceptable threshold (<5%)

---

## Epic 4: Python Bindings

*Migrate Python bindings to use BinnedDataset.*

### Story 4.1: Update PyDataset to BinnedDataset

**Status**: Not Started  
**Estimate**: 2 hours

**Change**: `PyDataset` internally wraps `BinnedDataset`.

**Implementation**:
- Constructor: Use `DatasetBuilder::from_array()` with binning config
- Keep same Python API (transparent to users)
- Update all internal references

**Definition of Done**:
- PyDataset uses BinnedDataset
- Basic unit tests pass

---

### Story 4.2: Update PyGBDTModel

**Status**: Not Started  
**Estimate**: 1 hour

Already mostly works since GBDT uses SampleBlocks.

**Changes**:
- Verify fit() works with BinnedDataset
- Verify predict() works with BinnedDataset

**Definition of Done**:
- All GBDT Python tests pass

---

### Story 4.3: Update PyGBLinearModel

**Status**: Not Started  
**Estimate**: 1.5 hours

**Changes**:
- `fit()`: Pass BinnedDataset to trainer
- `predict()`: Pass BinnedDataset to model

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
1. Delete `crates/boosters/src/data/types/` module (or relevant parts)
2. Delete `FeaturesView` struct
3. Update all imports
4. Fix any remaining compile errors

**Prerequisite**: All usages migrated in Epics 3-4.

**Definition of Done**:
- Old `Dataset` deleted
- `FeaturesView` deleted
- Code compiles
- Tests pass

---

### Story 5.2: Rename BinnedDataset to Dataset

**Status**: Not Started  
**Estimate**: 1 hour

**Actions**:
1. Rename `BinnedDataset` → `Dataset`
2. Update all imports and docs
3. Consider type alias for transition if needed

**Definition of Done**:
- Primary type is `Dataset`
- All references updated

---

### Story 5.3: Remove Dead Code

**Status**: Not Started  
**Estimate**: 1 hour

**Actions**:
- Delete `deprecated/` folder if exists
- Remove `#![allow(dead_code)]` from binned module
- Delete any unused helper methods
- Delete old `raw_feature_slice()`, `raw_feature_iter()` if superseded

**Definition of Done**:
- No dead code
- No allow(dead_code) attributes

---

### Story 5.4: Final Validation

**Status**: Not Started  
**Estimate**: 1 hour

**Checklist**:
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean
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
- Types deleted
- Performance impact
- API simplification

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
| 5: Cleanup | 4 | Not Started |
| 6: Review | 2 | Not Started |
| **Total** | **26** | |

**Estimated remaining**: ~15 hours
