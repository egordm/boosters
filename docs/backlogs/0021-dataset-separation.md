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

**Tasks**:
- [ ] Export `Dataset`, `DatasetBuilder` from `raw/`
- [ ] Export `DatasetSchema`, `FeatureMeta`, `FeatureType` from `raw/`
- [ ] Export views (`TargetsView`, `WeightsView`, `FeaturesView`) from `raw/`
- [ ] Export `SampleBlocks` from `raw/`
- [ ] Keep `BinnedDataset` as `pub(crate)` (not publicly exported)
- [ ] Remove `DataAccessor`, `SampleAccessor` from exports

**Definition of Done**:
- Public API is clean and minimal
- Internal types are not exported
- Documentation updated

**Testing**:
- Public API matches RFC specification
- External crates can import expected types

---

### Story 2.5: Stakeholder Feedback Check (Epic 2)

**Description**: Check stakeholder feedback after completing Epic 2.

**Tasks**:
- [ ] Review `workdir/tmp/stakeholder_feedback.md`
- [ ] Document any new stories

**Definition of Done**:
- Feedback reviewed

---

## Epic 3: Feature Value Iteration API

**Goal**: Implement the iteration patterns from RFC-0019 on Dataset.

### Story 3.0: Capture Benchmark Baselines

**Description**: Before any API changes, capture performance baselines for comparison.

**Tasks**:
- [ ] Run existing GBLinear training benchmark, record results
- [ ] Run existing GBDT prediction benchmark, record results
- [ ] Run existing SHAP benchmark, record results
- [ ] Document baselines in `workdir/tmp/iteration-api-baselines.md`

**Definition of Done**:
- Baseline numbers captured and documented
- Can compare post-implementation numbers

**Testing**:
- Benchmarks run successfully

---

### Story 3.1: Implement Feature Enum

**Description**: Rename and update the feature storage enum.

**Tasks**:
- [ ] Rename `FeatureColumn` → `Feature` in `data/raw/feature.rs`
- [ ] Ensure `Feature::Dense(Box<[f32]>)` variant
- [ ] Ensure `Feature::Sparse { indices, values, default }` variant
- [ ] Update Dataset struct: `columns` → `features: Box<[Feature]>`
- [ ] Update all internal references

**Definition of Done**:
- Consistent naming throughout
- No references to old names
- `cargo build` succeeds

**Testing**:
- Unit tests pass

---

### Story 3.2: Implement for_each_feature_value()

**Description**: Add zero-cost iteration over feature values.

**Tasks**:
- [ ] Add `Dataset::for_each_feature_value<F>(feature, f: F)` method
- [ ] Match on Feature once, then iterate directly
- [ ] Dense: iterate `values.iter().enumerate()`
- [ ] Sparse: iterate only stored (non-default) values
- [ ] Ensure closure is inlined

**Definition of Done**:
- Method compiles to tight loop for dense features
- Works for both dense and sparse features
- Documented with performance notes

**Testing**:
- Unit tests for dense and sparse iteration
- Benchmark shows 0% overhead vs direct slice iteration

---

### Story 3.3: Implement for_each_feature_value_dense()

**Description**: Add iteration that includes default values for sparse features.

**Tasks**:
- [ ] Add `Dataset::for_each_feature_value_dense<F>(feature, f: F)` method
- [ ] Dense: same as for_each_feature_value
- [ ] Sparse: iterate all n_samples, filling gaps with default

**Definition of Done**:
- Yields all n_samples values for both dense and sparse
- Documented when to use vs for_each_feature_value

**Testing**:
- Unit tests verify all samples yielded
- Sparse defaults are correct

---

### Story 3.4: Implement gather_feature_values()

**Description**: Add filtered gather for subset of samples (linear tree fitting).

**Tasks**:
- [ ] Add `Dataset::gather_feature_values(feature, sample_indices, buffer)` method
- [ ] Dense: indexed gather `buffer[i] = values[indices[i]]`
- [ ] Sparse: merge-join algorithm (both sorted)
- [ ] Add helper `gather_sparse_values()` for sparse logic

**Definition of Done**:
- Works for sorted sample indices
- Efficient merge-join for sparse
- Buffer filled correctly

**Testing**:
- Unit tests for dense and sparse gather
- Edge cases tested:
  - Empty sample_indices
  - Single element sample_indices
  - sample_indices covering all samples
  - Sparse feature with no stored values (all defaults)
  - Sparse feature where sample_indices don't overlap with stored indices

---

### Story 3.5: Implement for_each_gathered_value()

**Description**: Add callback version of gather for allocation-free usage.

**Tasks**:
- [ ] Add `Dataset::for_each_gathered_value<F>(feature, sample_indices, f)` method
- [ ] Callback receives `(local_idx, value)`
- [ ] Use same logic as gather_feature_values

**Definition of Done**:
- No allocation required
- Equivalent to gather_feature_values semantically

**Testing**:
- Unit tests verify correct indices and values

---

### Story 3.6: Port SampleBlocks to Work on Dataset

**Description**: Update SampleBlocks to use Dataset's Feature enum.

**Note**: Current SampleBlocks is tightly coupled to BinnedDataset's storage. This port requires generalizing the block buffering to work with Dataset's Feature enum (dense slices and sparse storage) rather than binned indices.

**Tasks**:
- [ ] Update SampleBlocks to take `&Dataset` instead of BinnedDataset
- [ ] Generalize buffering to work with Feature::Dense and Feature::Sparse
- [ ] Access features via `dataset.features[feature]`
- [ ] Handle both Dense (slice access) and Sparse (lookup) features
- [ ] Keep block buffering pattern for cache efficiency

**Definition of Done**:
- SampleBlocks works with Dataset
- Efficient batched row-major access
- Cache-friendly

**Testing**:
- Existing SampleBlocks tests pass
- Works with both dense and sparse features

---

### Story 3.7: Add FeatureValueIter (Optional Ergonomic API)

**Description**: Add enum iterator for cases needing Iterator trait.

**Tasks**:
- [ ] Add `FeatureValueIter<'a>` enum with Dense/Sparse variants
- [ ] Implement Iterator trait with `(usize, f32)` item
- [ ] Add `Dataset::feature_values(feature)` → `FeatureValueIter`
- [ ] Document overhead (~5-10% for dense)

**Definition of Done**:
- Iterator works correctly
- Documented as having overhead vs for_each

**Testing**:
- Unit tests for iteration
- Benchmark shows overhead is acceptable

---

### Story 3.8: Review/Demo Session (Epic 3)

**Description**: Review iteration API implementation.

**Tasks**:
- [ ] Demo for_each_feature_value performance
- [ ] Show benchmark results vs direct iteration
- [ ] Document in `workdir/tmp/development_review_<date>.md`

**Definition of Done**:
- API complete and performant
- Review documented

---

## Epic 4: API Simplification

**Goal**: Simplify training API with single validation set and cleaner BinnedDataset interface.

### Story 4.1: Replace EvalSet with Optional Validation Set (Rust)

**Description**: Change eval_sets parameter to single val_set in Rust API.

**Tasks**:
- [ ] Delete `EvalSet` struct from `training/eval.rs`
- [ ] Change `Evaluator::evaluate_round()` signature from `&[EvalSet]` to `Option<&Dataset>`
- [ ] Update `GBDTTrainer::train()` signature
- [ ] Update all internal callers
- [ ] Update callback signatures if needed

**Definition of Done**:
- No `EvalSet` struct
- Single `val_set: Option<&Dataset>` parameter
- All callers updated

**Testing**:
- Training with val_set=Some works
- Training with val_set=None works
- Early stopping works with validation set

---

### Story 4.2: Remove PyEvalSet from Python Bindings

**Description**: Update Python bindings to use simple val_set parameter.

**Tasks**:
- [ ] Delete `PyEvalSet` class from `boosters-python/src/data.rs`
- [ ] Remove `EvalSet` export from `boosters-python/src/lib.rs`
- [ ] Update `PyGBDTModel.fit()` to accept `val_set: Option<PyDataset>`
- [ ] Update Python type stubs if they exist
- [ ] Update Python examples and documentation

**Definition of Done**:
- No `PyEvalSet` class
- Python API uses `val_set=...` parameter
- Python tests pass

**Testing**:
- Python training with val_set works
- Python training without val_set works

---

### Story 4.3: Remove effective_ Prefix from BinnedDataset

**Description**: Rename BinnedDataset methods to remove effective_ prefix.

**Tasks**:
- [ ] Rename `effective_feature_views()` → `feature_views()`
- [ ] Rename `effective_feature_count()` → `n_features()`
- [ ] Delete `original_feature_view()` method
- [ ] Delete `original_feature_count()` method
- [ ] Update all callers in histogram building, training

**Definition of Done**:
- No `effective_` prefix on any methods
- No `original_` methods
- All callers updated

**Testing**:
- Histogram building works
- Training produces same results

---

### Story 4.4: Stakeholder Feedback Check (Epic 4)

**Description**: Check stakeholder feedback after completing Epic 4.

**Tasks**:
- [ ] Review `workdir/tmp/stakeholder_feedback.md`
- [ ] Document any new stories

**Definition of Done**:
- Feedback reviewed

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
