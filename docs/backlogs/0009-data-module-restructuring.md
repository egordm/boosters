# Backlog: Data Module Restructuring

**Status**: Superseded by `docs/backlogs/0021-dataset-separation.md`  
**Created**: 2025-01-25  
**Epic**: Data Module Cleanup and Simplification

## Overview

This backlog captured early work to restructure and simplify the data module based on RFC-0021 design decisions.

**Update (2025-12-31)**: This work was fully implemented and tracked under `docs/backlogs/0021-dataset-separation.md`. This document is kept for historical context, but should not be used for current status tracking.

## Epic 1: Remove Unused I/O Infrastructure

### Story 1.1: Remove io-parquet Feature and Module
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

Remove the unused io-parquet feature and all associated code:

- [ ] Delete `/crates/boosters/src/data/io/` directory (error.rs, mod.rs, parquet.rs, record_batches.rs)
- [ ] Remove `io-parquet = ["dep:parquet", "dep:arrow"]` from Cargo.toml
- [ ] Remove `#[cfg(feature = "io-parquet")] pub mod io;` from data/mod.rs
- [ ] Remove any re-exports related to io module

**Definition of Done**:
- io module completely removed
- `cargo check` passes without io-parquet feature
- No orphaned imports or dead code

---

### Story 1.2: Remove Parquet Support from Quality Benchmark
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

Remove all parquet/loaded dataset support from quality_benchmark.rs:

- [ ] Remove `#[cfg(feature = "io-parquet")]` imports and use statements
- [ ] Remove `DataSource::Parquet` variant
- [ ] Remove `real_world_configs()` function (both implementations)
- [ ] Remove parquet file path constants (CALIFORNIA_HOUSING_PATH, ADULT_PATH, COVERTYPE_PATH)
- [ ] Remove `--mode real` option (only synthetic mode remains)
- [ ] Update documentation comments to reflect synthetic-only support
- [ ] Remove io-parquet from example cargo commands in docstrings

**Definition of Done**:
- quality_benchmark.rs compiles without io-parquet feature
- Only synthetic datasets supported
- Documentation updated

---

## Epic 2: Data Module Structure Cleanup

### Story 2.1: Restructure Data Module Layout
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

Reorganize the data module for clear separation:

Current structure:
```
data/
  binned/
    sample_blocks.rs  <- should be in raw
    ...
  types/              <- buried too deep
    dataset.rs
    views.rs
    ...
  io/                 <- to be removed
  mod.rs
```

Target structure:
```
data/
  raw/                <- renamed from types/
    mod.rs
    dataset.rs
    views.rs
    schema.rs
    accessor.rs
    column.rs
    sample_blocks.rs  <- moved from binned/
  binned/
    mod.rs
    dataset.rs
    builder.rs
    ...
  mod.rs              <- re-exports from raw/ and binned/
```

Tasks:
- [ ] Rename `types/` to `raw/`
- [ ] Move `sample_blocks.rs` from `binned/` to `raw/`
- [ ] Update all internal imports
- [ ] Update data/mod.rs re-exports
- [ ] Update all external consumers

**Definition of Done**:
- Clear separation between raw and binned modules
- sample_blocks in raw module (since it's used for prediction on raw data)
- All tests pass
- No broken imports

---

## Epic 3: Simplify Evaluation API

### Story 3.1: Replace EvalSets with Single Validation Set
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

Simplify from `eval_sets: &[EvalSet<'_>]` to `val_set: Option<&Dataset>`:

Files affected:
- `crates/boosters/src/training/eval.rs` - EvalSet struct, Evaluator
- `crates/boosters/src/model/gbdt/model.rs` - GBDTModel::train()
- `crates/boosters/src/model/gblinear/model.rs` - GBLinearModel::train()
- `packages/boosters-python/src/model/gbdt.rs` - Python bindings
- `packages/boosters-python/src/model/gblinear.rs` - Python bindings
- `packages/boosters-python/src/data.rs` - PyEvalSet removal

Tasks:
- [ ] Change `eval_sets: &[EvalSet<'_>]` to `val_set: Option<&Dataset>` in core model APIs
- [ ] Update Evaluator to work with single validation set
- [ ] Remove EvalSet struct (or make it internal to Evaluator)
- [ ] Update Python bindings: `eval_set` becomes `valid: Option<PyDataset>`
- [ ] Remove PyEvalSet class from Python bindings
- [ ] Update all call sites
- [ ] Update training loop early stopping logic

**Definition of Done**:
- Single validation set API throughout
- EvalSet struct removed from public API
- Python bindings simplified
- All tests pass

---

## Epic 4: Simplify BinnedDataset

### Story 4.1: Remove Original/Effective Feature Distinction
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

Remove the original_feature vs effective_feature distinction:

Current API:
```rust
// Current - confusing
dataset.effective_feature_views()  // returns EffectiveViews
dataset.original_feature_view(idx) // returns single view
dataset.effective_n_features()
```

Target API:
```rust
// Simplified - just "features"
dataset.feature_views()  // returns Vec<BinnedFeatureView>
dataset.feature_view(idx) // returns single view
dataset.n_features()
```

Tasks:
- [ ] Rename `effective_feature_views()` → `feature_views()`
- [ ] Rename `EffectiveViews` → `FeatureViews` or inline
- [ ] Remove `original_feature_view()` method
- [ ] Remove `effective_n_features()` - just use `n_features()`
- [ ] Remove all `effective_to_original()` mappings if no longer needed
- [ ] Update all consumers (grower, histogram builder, etc.)

**Definition of Done**:
- No `effective_` or `original_` prefixes in public API
- Clean feature access interface
- All tests pass

---

### Story 4.2: Simplify BinnedDataset Builder
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

Since BinnedDataset is internal and always created from Dataset, simplify the builder:

Current complexity:
- Multiple construction paths (from_array, single features, from_array_with_metadata)
- Targets/weights handling in builder
- BuiltGroups intermediate representation

Target simplicity:
```rust
// Primary API - always from Dataset
impl BinnedDataset {
    pub fn from_dataset(dataset: &Dataset, config: &BinningConfig) -> Result<Self>;
}
```

Tasks:
- [ ] Create `BinnedDataset::from_dataset()` as primary constructor
- [ ] Move binning config from builder to method parameter
- [ ] Remove `set_targets()`, `set_weights()` from builder (get from Dataset)
- [ ] Simplify or remove DatasetBuilder (or make it internal)
- [ ] Keep internal builder for test utilities if needed
- [ ] Update all construction sites

**Definition of Done**:
- Single clear construction path: `BinnedDataset::from_dataset()`
- No targets/weights on BinnedDataset (reference back to Dataset)
- Simpler internal structure

---

### Story 4.3: Fix labels vs targets Inconsistency
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

BinnedDataset uses `labels` while Dataset uses `targets`. Standardize on `targets`:

Tasks:
- [ ] Rename `has_labels()` → access through Dataset or remove
- [ ] Rename `labels()` → access through Dataset or remove  
- [ ] Update tests that use `set_labels()`
- [ ] Consider if BinnedDataset even needs targets (maybe reference Dataset?)

**Definition of Done**:
- Consistent naming throughout
- Clear ownership of targets

---

## Epic 5: Review and Cleanup

### Story 5.1: Final Review and Documentation
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

- [ ] Update RFC-0021 with final implementation details
- [ ] Verify all public APIs are documented
- [ ] Run full test suite
- [ ] Run benchmarks to verify no performance regression

---

## Meta-Tasks

### Stakeholder Feedback Check
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

- [ ] Review `workdir/tmp/stakeholder_feedback.md` for relevant input
- [ ] Incorporate feedback into implementation decisions

### Review/Demo Session
**Status**: Superseded (implemented in `docs/backlogs/0021-dataset-separation.md`)

After Epic 4 completion:
- [ ] Demonstrate simplified API
- [ ] Show code reduction metrics
- [ ] Document in `workdir/tmp/development_review_<timestamp>.md`

---

## Notes

- Keep TargetsView and WeightsView as they are useful abstractions
- SampleBlocks is used for prediction batching, belongs with raw dataset
- BinnedDataset builder tests can use internal test utilities
