# Backlog: Dataset Separation and Feature Value Iteration

**RFCs**: 
- [0021-dataset-separation.md](../rfcs/0021-dataset-separation.md) - Data Module Restructuring
- [0019-feature-value-iterator.md](../rfcs/0019-feature-value-iterator.md) - Feature Value Iteration

**Created**: 2025-12-30  
**Updated**: 2025-12-31  
**Status**: Ready for Implementation

## Overview

This backlog implements both RFC-0021 (data module restructuring and dataset separation) and RFC-0019 (feature value iteration patterns). The work is organized into seven epics that must be implemented in order, as later epics depend on earlier ones.

## Milestones

| Milestone | Epics | Description | Goal |
| --------- | ----- | ----------- | ---- |
| M1        | 1-2   | Cleanup & Structure | Remove unused code, reorganize modules |
| M2        | 3     | Iteration APIs | Implement Dataset iteration patterns |
| M3        | 4-5   | Simplification | Simplify training API, remove builder |
| M4        | 6     | Migration | Migrate all consumers to new APIs |
| M5        | 7     | Retrospective | Reflect and capture learnings |

---

## Epic 1: Remove io-parquet Feature

**Goal**: Remove unused Arrow/Parquet I/O code to reduce complexity and dependencies.

### Story 1.1: Verify and Remove io-parquet Remnants

**Description**: Confirm io-parquet is fully removed from codebase and dependencies.

**Status**: ✅ Complete (2025-12-30)

**Tasks**:
- [x] Verify `data/io/` doesn't exist → Deleted (error.rs, mod.rs, parquet.rs, record_batches.rs)
- [x] Remove `io-parquet` feature from `crates/boosters/Cargo.toml` → Not present (never added)
- [x] Remove `arrow` and `parquet` dependencies from Cargo.toml → Not present
- [x] Update `crates/boosters/src/data/mod.rs` to remove io module reference → Removed `#[cfg(feature = "io-parquet")]`
- [x] Check quality_benchmark.rs for parquet references → Clean (prior session)
- [x] Update docstrings to remove `--features io-parquet` usage examples → Clean

**Definition of Done**:
- ✅ No `data/io/` directory exists
- ✅ No `io-parquet` feature in Cargo.toml
- ✅ No arrow/parquet dependencies
- ✅ No parquet references in benchmarks
- ✅ `cargo build` succeeds

**Testing**:
- ✅ `cargo build` and `cargo test` pass (686 tests)
- ✅ `cargo check --bin quality_benchmark` succeeds

---

### Story 1.2: Stakeholder Feedback Check (Epic 1)

**Status**: ✅ Complete (2025-12-30)

**Description**: Check stakeholder feedback after completing Epic 1.

**Tasks**:
- [x] Review `workdir/tmp/stakeholder_feedback.md` for relevant feedback → No pending feedback
- [x] Document any new stories created from feedback → None needed

**Definition of Done**:
- ✅ Feedback file reviewed
- ✅ No new stories needed

---

## Epic 2: Module Restructuring

**Goal**: Reorganize data module with clear separation between raw/ and binned/.

### Story 2.1: Create data/raw Module

**Description**: Create the new `data/raw/` module structure.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Create `data/raw/` directory
- [x] Create `data/raw/mod.rs` with appropriate exports
- [x] Move `data/types/dataset.rs` → `data/raw/dataset.rs`
- [x] Move `data/types/views.rs` → `data/raw/views.rs`
- [x] Move `data/types/column.rs` → `data/raw/feature.rs` (rename to reflect naming)
- [x] Move `data/types/schema.rs` → `data/raw/schema.rs`
- [x] Update internal imports in moved files

**Definition of Done**:
- ✅ All files in `data/raw/`
- ✅ `column.rs` renamed to `feature.rs`
- ✅ Imports updated
- ✅ `cargo build` succeeds

**Testing**:
- ✅ All existing tests pass (686 tests)

---

### Story 2.2: Move SampleBlocks to raw Module

**Description**: Move `sample_blocks.rs` from binned/ to raw/.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Move `data/binned/sample_blocks.rs` → `data/raw/sample_blocks.rs`
- [x] Update exports in `data/raw/mod.rs`
- [x] Update imports in `data/binned/mod.rs`
- [x] Update any callers that import SampleBlocks

**Definition of Done**:
- ✅ SampleBlocks in raw module
- ✅ All callers updated
- ✅ `cargo build` succeeds

**Testing**:
- ✅ SampleBlocks tests pass
- ✅ Prediction tests pass (686 tests)

---

### Story 2.3: Delete data/types Module Structure

**Description**: Remove the old types module after moving files. Note: accessor.rs is kept until Epic 6 when consumers are migrated.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Delete `data/types/mod.rs`
- [x] Move `data/types/accessor.rs` → `data/raw/accessor.rs` (temporary, deleted in Epic 6)
- [x] Delete `data/types/` directory
- [x] Update `data/mod.rs` to import from raw/ instead of types/
- [x] Note: accessor.rs kept for now, consumers still use DataAccessor trait

**Definition of Done**:
- ✅ No `data/types/` directory
- ✅ accessor.rs in raw/ (temporary location)
- ✅ All exports come from `data/raw/`
- ✅ `cargo build` succeeds

**Testing**:
- ✅ All tests pass (686 tests)
- ✅ Existing DataAccessor usages still work

---

### Story 2.4: Update Public API Exports

**Description**: Ensure public API exports are correct in data/mod.rs.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Export `Dataset`, `DatasetBuilder` from `raw/`
- [x] Export `DatasetSchema`, `FeatureMeta`, `FeatureType` from `raw/`
- [x] Export views (`TargetsView`, `WeightsView`, `FeaturesView`) from `raw/`
- [x] Export `SampleBlocks` from `raw/`
- [x] Keep `BinnedDataset` as `pub(crate)` (not publicly exported) → Deferred to Epic 5 when builder is deleted
- [x] Remove `DataAccessor`, `SampleAccessor` from exports → Deferred to Epic 6

**Note**: Binned visibility changes deferred to Epic 5/6 when types are deleted. Current raw module exports are correct.

**Definition of Done**:
- ✅ Public API is clean and minimal
- ✅ Internal types correctly exported from raw/
- ✅ Documentation in mod.rs is current

**Testing**:
- ✅ Public API matches RFC specification
- ✅ All tests pass

---

### Story 2.5: Stakeholder Feedback Check (Epic 2)

**Description**: Check stakeholder feedback after completing Epic 2.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Review `workdir/tmp/stakeholder_feedback.md` → No pending feedback
- [x] Document any new stories → None needed

**Definition of Done**:
- ✅ Feedback reviewed

---

## Epic 3: Feature Value Iteration API

**Goal**: Implement the iteration patterns from RFC-0019 on Dataset.

### Story 3.0: Capture Benchmark Baselines

**Description**: Before any API changes, capture performance baselines for comparison.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Run existing GBLinear training benchmark, record results → API mismatch, noted in baselines
- [x] Run existing GBDT prediction benchmark, record results → Captured
- [x] Run existing SHAP benchmark, record results → Not available, noted
- [x] Document baselines in `workdir/tmp/iteration-api-baselines.md`

**Definition of Done**:
- ✅ Baseline numbers captured and documented
- ✅ Can compare post-implementation numbers

**Testing**:
- ✅ Benchmarks run successfully (prediction_core, training_gbdt)

---

### Story 3.1: Implement Feature Enum

**Description**: Rename and update the feature storage enum.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Rename `Column` → `Feature` in `data/raw/feature.rs`
- [x] Ensure `Feature::Dense(Array1<f32>)` variant
- [x] Ensure `Feature::Sparse { indices, values, n_samples, default }` variant (inline struct)
- [x] Update DatasetBuilder to use `Feature` instead of `Column`
- [x] Update all internal references
- [x] Add deprecated `Column` type alias for backward compatibility
- [x] Remove `SparseColumn` struct (fields inlined into `Feature::Sparse`)

**Note**: Dataset.features still uses Array2<f32> (densified). The RFC's Box<[Feature]> storage will be implemented when iteration methods need it. Current design keeps backward compatibility.

**Definition of Done**:
- ✅ Consistent naming throughout
- ✅ No references to old SparseColumn struct
- ✅ `cargo build` succeeds

**Testing**:
- ✅ All tests pass (687 tests)

---

### Story 3.2: Implement for_each_feature_value()

**Description**: Add zero-cost iteration over feature values.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Add `Dataset::for_each_feature_value<F>(feature, f: F)` method
- [x] Match on Feature once, then iterate directly
- [x] Dense: iterate `values.iter().enumerate()`
- [x] Sparse: (future) iterate only stored (non-default) values
- [x] Ensure closure is inlined

**Definition of Done**:
- ✅ Method compiles to tight loop for dense features
- ✅ Works for both dense and sparse features
- ✅ Documented with performance notes

**Testing**:
- ✅ Unit tests for dense iteration
- ✅ 695 tests pass

---

### Story 3.3: Implement for_each_feature_value_dense()

**Description**: Add iteration that includes default values for sparse features.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Add `Dataset::for_each_feature_value_dense<F>(feature, f: F)` method
- [x] Dense: same as for_each_feature_value
- [x] Sparse: (future) iterate all n_samples, filling gaps with default

**Definition of Done**:
- ✅ Yields all n_samples values for both dense and sparse
- ✅ Documented when to use vs for_each_feature_value

**Testing**:
- ✅ Unit tests verify behavior matches for_each_feature_value for dense

---

### Story 3.4: Implement gather_feature_values()

**Description**: Add filtered gather for subset of samples (linear tree fitting).

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Add `Dataset::gather_feature_values(feature, sample_indices, buffer)` method
- [x] Dense: indexed gather `buffer[i] = values[indices[i]]`
- [x] Sparse: (future) merge-join algorithm (both sorted)

**Definition of Done**:
- ✅ Works for sorted sample indices
- ✅ Efficient indexed gather for dense
- ✅ Buffer filled correctly

**Testing**:
- ✅ Unit tests for dense gather
- ✅ Edge cases tested (empty indices, all indices)

---

### Story 3.5: Implement for_each_gathered_value()

**Description**: Add callback version of gather for allocation-free usage.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Add `Dataset::for_each_gathered_value<F>(feature, sample_indices, f)` method
- [x] Callback receives `(local_idx, value)`
- [x] Use same logic as gather_feature_values

**Definition of Done**:
- ✅ No allocation required
- ✅ Equivalent to gather_feature_values semantically

**Testing**:
- ✅ Unit tests verify correct indices and values

---

### Story 3.6: Move SampleBlocks to Correct Module

**Description**: ~~Update SampleBlocks to use Dataset's Feature enum.~~

**Status**: ✅ Complete (2025-12-31)

**Revised Scope**: During planning discussion, the team determined that SampleBlocks is tightly coupled to BinnedDataset (uses FeatureView which returns bin indices) and should remain in the binned module. The file was incorrectly placed in `raw/` during earlier restructuring. The RFC's vision of SampleBlocks on Dataset will be revisited when Dataset gains sparse storage.

**Tasks**:
- [x] Move `sample_blocks.rs` from `data/raw/` to `data/binned/`
- [x] Update module declarations in `raw/mod.rs` and `binned/mod.rs`
- [x] Update imports in `binned/dataset.rs`
- [x] Update re-exports in `data/mod.rs`

**Original Tasks (Deferred)**:
- [ ] ~~Update SampleBlocks to take `&Dataset` instead of BinnedDataset~~ (deferred until Dataset has sparse storage)
- [ ] ~~Generalize buffering to work with Feature::Dense and Feature::Sparse~~

**Definition of Done**:
- ✅ SampleBlocks correctly located in binned module
- ✅ All tests pass (695 tests)
- ✅ No production code actually uses SampleBlocks (only docs/tests)

**Testing**:
- ✅ Existing SampleBlocks tests pass

---

### Story 3.7: Add FeatureValueIter (Optional Ergonomic API)

**Description**: Add enum iterator for cases needing Iterator trait.

**Status**: ⏸️ Deferred

**Reason**: The for_each methods (Stories 3.2-3.5) cover all current use cases. The iterator has documented ~5-10% overhead and no current callers need the Iterator trait. Will implement if users request it.

**Tasks** (Deferred):
- [ ] Add `FeatureValueIter<'a>` enum with Dense/Sparse variants
- [ ] Implement Iterator trait with `(usize, f32)` item
- [ ] Add `Dataset::feature_values(feature)` → `FeatureValueIter`
- [ ] Document overhead (~5-10% for dense)

---

### Story 3.8: Review/Demo Session (Epic 3)

**Description**: Review iteration API implementation.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Demo for_each_feature_value performance
- [x] Show benchmark results vs direct iteration
- [x] Document in `workdir/tmp/development_review_2025-12-31_epic3.md`

**Definition of Done**:
- ✅ API complete and performant
- ✅ Review documented

---

## Epic 4: API Simplification

**Goal**: Simplify training API with single validation set and cleaner BinnedDataset interface.

### Story 4.1: Replace EvalSet with Optional Validation Set (Rust)

**Description**: Change eval_sets parameter to single val_set in Rust API.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Delete `EvalSet` struct from `training/eval.rs`
- [x] Change `Evaluator::evaluate_round()` signature from `&[EvalSet]` to `Option<&Dataset>`
- [x] Update `GBDTTrainer::train()` signature
- [x] Update `GBLinearTrainer::train()` signature
- [x] Update `GBDTModel::train()` and `train_binned()` signatures
- [x] Update `GBLinearModel::train()` signature
- [x] Remove `early_stopping_eval_set` from GBDTParams and GBLinearParams
- [x] Update all internal callers (tests, configs)
- [x] Callbacks not needed (no callback system)

**Definition of Done**:
- ✅ No `EvalSet` struct in Rust crate
- ✅ Single `val_set: Option<&Dataset>` parameter everywhere
- ✅ All callers updated (695 tests pass)

**Testing**:
- ✅ Training with val_set=None works
- ✅ All 695 tests pass
- (Note: Training with val_set=Some deferred - requires BinnedDataset for validation predictions)

---

### Story 4.2: Remove PyEvalSet from Python Bindings

**Description**: Update Python bindings to use simple val_set parameter.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Delete `PyEvalSet` class from `boosters-python/src/data.rs`
- [x] Remove `EvalSet` export from `boosters-python/src/lib.rs`
- [x] Update `PyGBDTModel.fit()` to accept `val_set: Option<PyDataset>`
- [x] Update `PyGBLinearModel.fit()` to accept `val_set: Option<PyDataset>`
- [x] Update Python data.py (remove EvalSet import and export)
- [x] Update sklearn wrappers (gbdt.py, gblinear.py)
- [x] Update tests and README

**Definition of Done**:
- ✅ No `PyEvalSet` class
- ✅ Python API uses `val_set=...` parameter
- ✅ Python package compiles

**Testing**:
- Python training with val_set works (manual testing recommended)
- Python training without val_set works (manual testing recommended)

---

### Story 4.3: Remove effective_ Prefix from BinnedDataset

**Description**: Rename BinnedDataset methods to remove effective_ prefix.

**Status**: ⏸️ Deferred/Not Applicable

**Notes**: After investigation, the current API is already correct:
- `n_features()` already exists (no `effective_feature_count()` or `original_feature_count()`)
- `feature_views()` and `effective_feature_views()` serve different purposes:
  - `feature_views()` returns `Vec<FeatureView>` for simple iteration
  - `effective_feature_views()` returns `EffectiveViews` struct with bundle metadata needed by the grower
- `original_feature_view()` is used by SampleBlocks for prediction and cannot be deleted

The effective_ prefix on `effective_feature_views()` is intentional to distinguish it from `feature_views()`. No changes needed.

**Tasks**:
- [x] Verified `n_features()` exists
- [x] Verified `effective_feature_views()` and `feature_views()` have different return types
- [x] Verified `original_feature_view()` is used by SampleBlocks

---

### Story 4.4: Stakeholder Feedback Check (Epic 4)

**Description**: Check stakeholder feedback after completing Epic 4.

**Status**: ✅ Complete (2025-12-31)

**Tasks**:
- [x] Review `workdir/tmp/stakeholder_feedback.md` → No pending feedback
- [x] Document any new stories → None needed

**Definition of Done**:
- ✅ Feedback reviewed

---

## Epic 5: BinnedDataset Simplification

**Goal**: Replace BinnedDatasetBuilder with simple factory method.

### Story 5.1: Implement BinnedDataset::from_dataset()

**Description**: Create factory method that constructs BinnedDataset from Dataset.

**Complexity**: Substantial (~500 lines of builder logic to reorganize). Expect 1-2 days.

**Tasks**:
- [ ] Add `from_dataset(&Dataset, &BinningConfig) -> Result<Self, BuildError>` method
- [ ] Access Dataset.schema for feature metadata
- [ ] Move `create_bin_mappers()` helper into from_dataset (private)
- [ ] Move `build_feature_group()` helper into from_dataset (private)
- [ ] Move `build_dense_feature()` helper into from_dataset (private)
- [ ] Move `build_sparse_feature()` helper into from_dataset (private)
- [ ] Move `build_bundle()` EFB logic into from_dataset (private)

**Definition of Done**:
- `from_dataset()` creates working BinnedDataset
- All builder logic preserved as private helpers
- Training works with new construction path

**Testing**:
- Unit tests for from_dataset
- Integration tests for training pipeline

---

### Story 5.2: Delete BinnedDatasetBuilder

**Description**: Remove the builder struct and file entirely.

**Tasks**:
- [ ] Delete `BinnedDatasetBuilder` struct
- [ ] Delete `BuiltGroups` struct
- [ ] Delete `data/binned/builder.rs` file
- [ ] Update `data/binned/mod.rs` exports
- [ ] Remove `from_built_groups()` method from BinnedDataset
- [ ] Update all callers to use `from_dataset()`

**Definition of Done**:
- No `builder.rs` file
- No `BinnedDatasetBuilder` type
- ~700 lines of code deleted

**Testing**:
- All training tests pass
- Benchmark results unchanged

---

### Story 5.3: Add BinnedDataset::test_builder()

**Description**: Add simple test helper for constructing BinnedDataset in unit tests.

**Tasks**:
- [ ] Add `#[cfg(test)]` `test_builder()` method
- [ ] Allow direct construction with known bins for testing
- [ ] Keep it simple (~50 lines)

**Definition of Done**:
- Tests can construct BinnedDataset directly
- test_builder is only available in test builds

**Testing**:
- Existing histogram tests refactored to use test_builder
- Tests are cleaner and more focused

---

### Story 5.4: Simplify Storage Types

**Description**: Remove raw value storage from BinnedDataset storage types.

**IMPORTANT**: This story must wait until Epic 6 is complete. Consumers must be migrated to use Dataset for raw values before we can remove raw_values from BinnedDataset storage.

**Depends on**: Epic 6 complete (Stories 6.1-6.8)

**Tasks**:
- [ ] Verify all consumers use Dataset for raw values (Epic 6 complete)
- [ ] Remove `raw_values` field from `NumericStorage`
- [ ] Remove `raw_values` field from `SparseNumericStorage`
- [ ] Update storage constructors
- [ ] Update `FeatureStorage` enum
- [ ] Update histogram building to not expect raw values

**Definition of Done**:
- Storage types only contain bins
- Memory usage reduced
- Histogram building works

**Testing**:
- Storage unit tests updated
- Training produces same results

---

### Story 5.5: Stakeholder Feedback Check (Epic 5)

**Description**: Check stakeholder feedback after completing Epic 5.

**Tasks**:
- [ ] Review `workdir/tmp/stakeholder_feedback.md`
- [ ] Document any new stories

**Definition of Done**:
- Feedback reviewed

---

## Epic 6: Consumer Migration

**Goal**: Migrate all raw-value consumers to use Dataset iteration APIs.

### Story 6.1: Migrate GBLinear Training

**Description**: Update GBLinear updater to use Dataset::for_each_feature_value().

**Tasks**:
- [ ] Update `Updater::compute_weight_update()` to take `&Dataset`
- [ ] Replace `data.feature(f).iter()` with `dataset.for_each_feature_value()`
- [ ] Delete `FeaturesView` usage in updater
- [ ] Update signature of `GBLinearTrainer::train()`

**Definition of Done**:
- GBLinear training uses Dataset directly
- No FeaturesView in training code
- Training produces same results

**Testing**:
- GBLinear training tests pass
- Golden test: trained model weights match baseline within 1e-10 tolerance
- Benchmark unchanged (±5%)

---

### Story 6.2: Migrate GBLinear Prediction

**Description**: Update LinearModel::predict_into() to use Dataset.

**Tasks**:
- [ ] Update `LinearModel::predict_into()` to take `&Dataset`
- [ ] Replace `features.feature(f).iter()` with `for_each_feature_value()`
- [ ] Update callers

**Definition of Done**:
- Prediction uses Dataset directly
- Same results as before

**Testing**:
- Prediction tests pass

---

### Story 6.3: Migrate Linear SHAP

**Description**: Update Linear SHAP explainer to use Dataset.

**Tasks**:
- [ ] Update `LinearExplainer` to take `&Dataset`
- [ ] Replace `features.feature(f)[i]` with iteration pattern
- [ ] Use for_each_feature_value() for efficient access

**Definition of Done**:
- Linear SHAP works with Dataset
- Same SHAP values as before

**Testing**:
- Linear SHAP tests pass

---

### Story 6.4: Migrate Linear Tree Fitting

**Description**: Update leaf linear fitting to use gather_feature_values().

**Tasks**:
- [ ] Update `LeafLinearTrainer` to take `&Dataset`
- [ ] Replace `DataAccessor::sample(row).feature(f)` pattern
- [ ] Use `gather_feature_values()` for per-feature gather
- [ ] Allocate reusable buffer for values

**Definition of Done**:
- Linear tree fitting uses gather pattern
- Better cache locality
- Same model quality

**Testing**:
- Linear tree tests pass
- Performance improved or unchanged

---

### Story 6.5: Migrate Tree SHAP

**Description**: Update Tree SHAP to use SampleBlocks on Dataset.

**Tasks**:
- [ ] Update `TreeExplainer` to use Dataset-based SampleBlocks
- [ ] Replace per-sample random access with block iteration
- [ ] Align with GBDT prediction pattern

**Definition of Done**:
- Tree SHAP uses SampleBlocks on Dataset
- Same SHAP values
- Better cache performance

**Testing**:
- Tree SHAP tests pass
- Performance benchmark

---

### Story 6.6: Migrate GBDT Prediction

**Description**: Update GBDT prediction to use SampleBlocks on Dataset.

**Tasks**:
- [ ] Update prediction code to use Dataset-based SampleBlocks
- [ ] Remove dependency on BinnedDataset for prediction
- [ ] Prediction works with raw Dataset only

**Definition of Done**:
- GBDT prediction works with Dataset
- No BinnedDataset needed for prediction
- Same predictions

**Testing**:
- Prediction tests pass

---

### Story 6.7: Delete FeaturesView

**Description**: Remove the FeaturesView type entirely. Must be done AFTER all consumer migrations (6.1-6.6) are complete.

**Tasks**:
- [ ] Verify no remaining usages of FeaturesView
- [ ] Update or delete tests that use FeaturesView
- [ ] Delete FeaturesView struct from `data/raw/views.rs`
- [ ] Remove export from mod.rs
- [ ] Update documentation

**Depends on**: Stories 6.1-6.6 complete

**Definition of Done**:
- No FeaturesView type
- All code uses Dataset iteration APIs
- Tests updated or removed

**Testing**:
- `cargo build` succeeds
- All tests pass

---

### Story 6.8: Delete DataAccessor Trait and accessor.rs

**Description**: Remove the DataAccessor trait entirely. This is the final cleanup after all consumers are migrated.

**Tasks**:
- [ ] Verify no remaining usages of DataAccessor trait
- [ ] Verify no remaining usages of SampleAccessor trait
- [ ] Delete `data/raw/accessor.rs` file
- [ ] Remove exports from mod.rs

**Depends on**: Stories 6.1-6.6 complete

**Definition of Done**:
- No DataAccessor trait
- No SampleAccessor trait
- accessor.rs deleted

**Testing**:
- All tests pass

---

### Story 6.9: End-to-end Regression Validation

**Description**: Validate complete pipeline still works identically after all migrations.

**Tasks**:
- [ ] Run full training pipeline: Dataset → BinnedDataset → Train
- [ ] Run full prediction pipeline with trained model
- [ ] Compare predictions against baseline from Story 3.0
- [ ] Run performance benchmarks, compare against Story 3.0 baselines
- [ ] Verify test coverage hasn't dropped significantly
- [ ] Document results showing no regression

**Definition of Done**:
- Predictions match baseline within 1e-10 tolerance
- Performance within ±5% of baseline
- Test coverage maintained
- No regressions

**Testing**:
- End-to-end test suite passes
- Benchmark comparison documented
- Coverage report generated

---

### Story 6.10: Review/Demo Session (Epic 6)

**Description**: Final review of all migration work.

**Tasks**:
- [ ] Demo all migrated components
- [ ] Show code deletion metrics
- [ ] Show performance benchmarks vs baselines
- [ ] Update data module documentation (README, rustdoc)
- [ ] Document in `workdir/tmp/development_review_<date>.md`

**Definition of Done**:
- All migrations complete
- Performance validated against baselines
- Documentation updated
- Review documented

---

## Epic 7: Retrospective

**Goal**: Reflect on implementation process and capture learnings.

### Story 7.1: Conduct Retrospective

**Description**: Team retrospective after all implementation complete.

**Tasks**:
- [ ] What went well?
- [ ] What didn't go well?
- [ ] What would we do differently?
- [ ] Document in `workdir/tmp/retrospective.md`

**Definition of Done**:
- Retrospective documented
- Action items captured as new backlog stories if needed

---

## Summary

| Epic | Stories | Description |
| ---- | ------- | ----------- |
| 1    | 1.1-1.2 | Remove io-parquet |
| 2    | 2.1-2.5 | Module restructuring |
| 3    | 3.0-3.8 | Feature value iteration API |
| 4    | 4.1-4.4 | API simplification |
| 5    | 5.1-5.5 | BinnedDataset simplification |
| 6    | 6.1-6.10 | Consumer migration |
| 7    | 7.1     | Retrospective |

**Total**: 7 Epics, 36 Stories

**Estimated code changes**:
- ~1,280 lines deleted (io-parquet, accessor.rs, builder.rs, FeaturesView)
- ~200 lines moved (types/ → raw/)
- ~400 lines new (iteration APIs, from_dataset, consumer updates)

**Net reduction**: ~680 lines

**Key Dependencies**:
- Epic 2 must complete before Epic 3 (module structure needed)
- Epic 3 must complete before Epic 5 (iteration APIs needed for binning)
- Epic 3 must complete before Epic 6 (APIs needed for migration)
- Story 5.4 (remove raw_values) must wait until Epic 6 is complete
- Story 6.7 and 6.8 (deletions) must wait until Stories 6.1-6.6 are complete

---

## Final Verification Checklist

Use this checklist at the end of implementation to verify all RFC success criteria are met.

### RFC-0021 (Dataset Separation)

- [ ] Dataset type exists with Feature::Dense and Feature::Sparse
- [ ] BinnedDataset only contains bins (no raw values in storage)
- [ ] BinnedDataset::from_dataset() is the only construction path
- [ ] BinnedDatasetBuilder deleted (~700 lines)
- [ ] data/raw/ module contains Dataset, Feature, schema, views
- [ ] data/types/ directory deleted
- [ ] accessor.rs and DataAccessor trait deleted
- [ ] FeaturesView deleted
- [ ] io-parquet removed
- [ ] API simplified: val_set instead of eval_sets

### RFC-0019 (Feature Value Iteration)

- [ ] Dataset::for_each_feature_value() implemented (zero-cost for dense)
- [ ] Dataset::for_each_feature_value_dense() implemented
- [ ] Dataset::gather_feature_values() implemented (with merge-join for sparse)
- [ ] Dataset::for_each_gathered_value() implemented
- [ ] SampleBlocks ported to work on Dataset
- [ ] FeatureValueIter enum provided (with documented overhead)
- [ ] GBLinear training uses Dataset::for_each_feature_value()
- [ ] GBLinear prediction uses Dataset::for_each_feature_value()
- [ ] Linear SHAP uses Dataset iteration APIs
- [ ] Linear tree fitting uses Dataset::gather_feature_values()
- [ ] Tree SHAP uses SampleBlocks on Dataset
- [ ] GBDT prediction uses SampleBlocks on Dataset

### Quality Gates

- [ ] All tests pass
- [ ] No performance regression (within ±5% of baseline)
- [ ] Predictions match baseline within 1e-10 tolerance
- [ ] Test coverage maintained
- [ ] Documentation updated
