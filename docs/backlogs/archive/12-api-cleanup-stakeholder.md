# Backlog 12: API Cleanup from Stakeholder Feedback

**Status**: Complete  
**Created**: 2024-12-24  
**Completed**: 2024-12-24  
**RFCs**: RFC-0019, RFC-0020

## Overview

This backlog addresses stakeholder feedback on API cleanliness and reduces interface bloat.
The focus is on simplifying constructors, removing convenience methods that add confusion,
and consolidating duplicate code.

---

## Epic 1: Builder and Binning Cleanup

### Story 1.1: Remove Type Aliases and Fix Imports
- [x] Remove `UnifiedDataset` type alias from builder.rs
- [x] Move function-local imports to top of files (builder.rs, predictor.rs)
- [x] Use short `Parallelism` instead of `crate::utils::Parallelism` with proper import

### Story 1.2: Consolidate BinnedDatasetBuilder Methods
- [x] Remove `from_matrix` - keep only `new()` with explicit parameters
- [x] Simplify `from_dataset` to `from_dataset(dataset, config, parallelism)`
- [x] Consolidate `bin_feature` and `bin_numeric_feature_from_slice` into `bin_numeric()`
- [x] Rename `bin_categorical_feature` to `bin_categorical()`

### Story 1.3: Add BinningConfig to GBDTConfig
- [x] Add `binning: BinningConfig` field to `GBDTConfig`
- [x] Remove hardcoded `max_bins=256` in `GBDTModel::train`
- [x] Update training to use config.binning

---

## Epic 2: Dataset Constructor Cleanup

### Story 2.1: Simplify Dataset Constructors
- [x] Modify `new()` to accept optional targets and optional weights
- [x] Remove `from_features()` - now covered by `new(features, None, None)`
- [x] Remove `from_column_major()` and `from_column_major_features()`
- [x] Make `new()` expect **feature-major** input `[n_features, n_samples]` by default

### Story 2.2: Update All Call Sites
- [x] Update tests to use new constructor
- [x] Update examples to use new constructor
- [x] Update benchmarks to use new constructor

---

## Epic 3: Prediction API Cleanup

### Story 3.1: Remove Redundant Predict Methods on Models
- [x] Remove `predict_array()` from GBDTModel - predict() takes FeaturesView only
- [x] Remove `predict_raw_array()` from GBDTModel
- [x] Remove `predict_array()` from GBLinearModel
- [x] Remove `predict_raw_array()` from GBLinearModel
- [x] Make `predict(FeaturesView)` the API, not `predict(&Dataset)`

### Story 3.2: Rename as_array() to view()
- [x] Rename `FeaturesView::as_array()` to `view()`
- [x] Rename `TargetsView::as_array()` to `view()`
- [x] Rename `SamplesView::as_array()` to `view()`
- [x] Follow ndarray naming convention

### Story 3.3: Update All Prediction Call Sites
- [x] Update tests
- [x] Update examples  
- [x] Update benchmarks

---

## Epic 4: Predictor Code Cleanup

### Story 4.1: Fix predict_from_feature_major_into
- [x] Simplify to use `maybe_par_bridge_for_each()` instead of `is_parallel()` checks
- [x] Remove ugly unsafe pointer arithmetic and intermediate allocations
- [x] Write directly to output buffer matching the sequential pattern

### Story 4.2: Move Imports to Top of File
- [x] Imports organized at top of predictor.rs
- [x] Imports organized at top of builder.rs

---

## Quality Gate

- [x] All tests pass
- [x] No new clippy warnings
- [x] All examples compile and run
- [x] Documentation updated

---

## Notes

**Breaking Changes**: This is API cleanup. Per CONTRIBUTING.md, breaking changes are
allowed since the library has no external users yet.

**Key Principles from Feedback**:
1. Fewer methods, explicit parameters over multiple convenience constructors
2. No type aliases unless absolutely necessary
3. Imports at top of file, avoid fully qualified names
4. Immediate cleanup - don't defer
5. Follow existing patterns (e.g., how predict_into handles parallel writes)
