# Backlog 13: API Cleanup Round 2

**Status**: In Progress  
**Created**: 2024-12-24  

## Overview

Further API cleanup based on stakeholder feedback. Focus on consistency, removing
redundant methods, and using standardized view types throughout.

---

## Epic 1: Objectives and Metrics Use Views

**SKIPPED** - ArrayView1 for per-output targets works fine, no change needed.

---

## Epic 2: BinnedDatasetBuilder Redesign ✅

### Story 2.1: Redesign BinnedDatasetBuilder ✅
- [x] Make `add_binned` `pub(crate)` (internal use only)
- [x] Builder takes built BinningConfig, no additional config needed
- [x] `new(config: BinningConfig)` - creates builder with config
- [x] `add_dataset(&Dataset, parallelism)` - bins all features using config
- [x] `add_features(FeaturesView, parallelism)` - bins features
- [x] `build()` - finalize
- [x] Remove from_dataset - user calls `new(config).add_dataset(ds, par).build()`
- [x] Updated all tests and examples to use new API

---

## Epic 3: GBDTModel Training Cleanup ✅

### Story 3.1: Train takes EvalSets directly ✅
- [x] Changed `train(&Dataset, &[&Dataset], ...)` to `train(&Dataset, &[EvalSet<'_>], ...)`
- [x] User passes `&[]` for no eval data
- [x] User creates EvalSets themselves with `EvalSet::new(name, &dataset)`

### Story 3.2: Don't Clone Eval Data ✅
- [x] EvalSet contains reference to dataset
- [x] No intermediate allocations

---

## Epic 4: High-Level Model Prediction ✅

### Story 4.1: GBDTModel predict takes Dataset ✅
- [x] Changed `predict(FeaturesView)` to `predict(&Dataset)`
- [x] Targets are just ignored
- [x] Same for predict_raw
- [x] predict internally calls predict_raw + transform (no code duplication)

---

## Epic 5: LinearModel Prediction Cleanup ✅

### Story 5.1: Unify FeaturesView Types ✅
- [x] Use single FeaturesView from `dataset::views`
- [x] Make schema optional (`Option<&DatasetSchema>`)
- [x] Add `from_array(data)` for schema-less views
- [x] Add `from_slice(data, n_samples, n_features)` for convenient construction
- [x] Remove/deprecate `data::ndarray::FeaturesView`

### Story 5.2: Consolidate to 3 Methods ✅
- [x] Keep: `predict_row_into(row: &[f32], output: &mut [f32])`
- [x] Keep: `predict_into(features: FeaturesView, output: ArrayViewMut2)`
- [x] Keep: `predict(features: FeaturesView) -> Array2`
- [x] Removed: base_score parameter (baked into model bias during XGBoost load)

### Story 5.3: GBLinearModel takes Dataset ✅
- [x] `predict(dataset: Dataset)` - returns transformed predictions
- [x] `predict_raw(dataset: Dataset)` - returns raw margins
- [x] Internally uses `dataset.features()` to get FeaturesView
- [x] Updated all tests to use new API

---

## Epic 6: GBDT Predictor Cleanup

### Story 6.1: Consolidate to 3 Methods
- [ ] `predict_row_into(row: &[f32], output: &mut [f32])`
- [ ] `predict_into(features: FeaturesView, parallelism, output: ArrayViewMut2)`
- [ ] `predict(features: FeaturesView, parallelism) -> Array2`
- [ ] Remove: `predict_from_feature_major_into` - merge into predict_into
- [ ] Clean implementation like current predict_into (zip, no block allocations)

---

## Epic 7: Import Cleanup

### Story 7.1: Remove Fully Qualified Names
- [ ] Replace `crate::dataset::Dataset` with `Dataset` and proper import
- [ ] Audit all files for fully qualified paths

---

## Quality Gate

- [x] All lib tests pass (556/556)
- [x] All integration tests pass (34/34)
- [x] All examples compile
- [ ] No new clippy warnings
- [ ] All doc tests pass

