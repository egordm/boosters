# Backlog: Dataset Format and Data Access Layer

**RFCs**: RFC-0019 (Dataset Format), RFC-0020 (Data Access Layer)
**Created**: 2025-12-24
**Status**: Approved

## Overview

This backlog implements the unified Dataset type and internal data access layer as specified in RFC-0019 and RFC-0020. The work focuses on:

1. Creating the new Dataset type with feature-major storage
2. Implementing view types (FeaturesView, TargetsView) in same module
3. Integrating with existing algorithms (GBDT, GBLinear)
4. Deleting old code immediately after each migration (no lingering code)

**Module Location**: New code goes in `src/dataset/` as a new top-level module. This keeps new and old code separate during migration. Old `src/data/dataset.rs` deleted after migrations complete.

**Python Bindings**: Out of scope for this backlog - covered in future Python API backlog.

## Milestone: Core API Complete

After Epic 1 + Epic 2 complete: New Dataset and views are usable, old code still works.

## Milestone: Full Migration Complete

After Epic 3 complete: All algorithms use new Dataset, old code deleted, benchmarks verified.

---

## Epic 1: Dataset Core Implementation

Implement the new Dataset type as specified in RFC-0019.

### Story 1.1: Core Types and Storage

Implement Dataset, Column, SparseColumn, DatasetSchema, FeatureMeta, FeatureType.

**Tasks:**

- [ ] 1.1.1: Create `src/dataset/` module structure (mod.rs, dataset.rs, schema.rs, column.rs, error.rs, views.rs)
- [ ] 1.1.2: Implement `FeatureType` enum (Numeric, Categorical)
- [ ] 1.1.3: Implement `FeatureMeta` and `DatasetSchema` types
- [ ] 1.1.4: Implement `Column` enum (Dense, Sparse) and `SparseColumn` struct
  - Note: Sparse support is type definition only; full sparse training is out of scope
- [ ] 1.1.5: Implement `Dataset` struct with feature-major storage `[n_features, n_samples]`
- [ ] 1.1.6: Implement `DatasetError` with thiserror, integrate with existing error types
- [ ] 1.1.7: Unit tests for all types
- [ ] 1.1.8: Add `pub use` exports to `src/lib.rs` (Dataset, DatasetBuilder, Column, SparseColumn, DatasetSchema, FeatureMeta, FeatureType, FeaturesView, TargetsView, DatasetError)

**Definition of Done:**

- All types compile and have rustdoc
- Unit tests cover construction, accessors, error cases
- Types are Send + Sync
- Types exported from crate root

### Story 1.2: Dataset Construction API

Implement Dataset::new() and DatasetBuilder for flexible construction.

**Tasks:**

- [ ] 1.2.1: Implement `Dataset::new(features, targets)` with transpose from `[n_samples, n_features]`
- [ ] 1.2.2: Implement `DatasetBuilder` with fluent API
- [ ] 1.2.3: Implement `add_feature()`, `add_categorical()`, `add_sparse()` builder methods
- [ ] 1.2.4: Implement validation (shape consistency, sparse indices sorted, no duplicates)
- [ ] 1.2.5: Unit tests for validation cases:
  - Empty dataset (0 samples) → OK
  - Mismatched row counts → DatasetError
  - Duplicate sparse indices → DatasetError
  - Unsorted sparse indices → DatasetError
  - n_categories = 0 for categorical → DatasetError
- [ ] 1.2.6: Integration tests for construction paths

**Definition of Done:**

- Can construct Dataset from dense matrix
- Can construct Dataset via builder with mixed column types
- Validation errors have clear messages
- Tests cover all DatasetError variants

### Story 1.3: Dataset View Access

Implement features_view(), targets_view(), weights() accessors.

**Tasks:**

- [ ] 1.3.1: Implement `Dataset::features_view() -> FeaturesView`
- [ ] 1.3.2: Implement `Dataset::targets_view() -> Option<TargetsView>`
- [ ] 1.3.3: Implement `Dataset::weights() -> Option<ArrayView1>`
- [ ] 1.3.4: Implement `Dataset::n_samples()`, `Dataset::n_features()`
- [ ] 1.3.5: Implement `Dataset::has_categorical()` for GBLinear validation

**Definition of Done:**
- Views return correct shapes
- n_samples/n_features consistent with storage
- has_categorical scans schema correctly

## Epic 2: View Types Implementation

Implement FeaturesView and TargetsView as specified in RFC-0020.

### Story 2.1: FeaturesView Implementation

Implement FeaturesView with feature-major semantics.

**Tasks:**

- [ ] 2.1.1: Implement `FeaturesView` struct in `src/dataset/views.rs`, wrapping `ArrayView2<f32>` and schema
- [ ] 2.1.2: Implement `n_samples()`, `n_features()` with correct dimension mapping
- [ ] 2.1.3: Implement `get(sample, feature)` with `[feature, sample]` indexing
- [ ] 2.1.4: Implement `feature(idx) -> ArrayView1` (contiguous)
- [ ] 2.1.5: Implement `sample(idx) -> ArrayView1` (strided, with doc warning)
- [ ] 2.1.6: Implement `feature_type(idx) -> FeatureType`
- [ ] 2.1.7: Implement `as_array() -> ArrayView2`
- [ ] 2.1.8: Unit tests verifying semantics match storage

**Definition of Done:**

- feature() returns contiguous slice (verify with as_slice())
- sample() is strided (document in rustdoc)
- Indexing is (sample, feature) conceptually but [feature, sample] in storage

### Story 2.2: TargetsView Implementation

Implement TargetsView for target access.

**Tasks:**

- [ ] 2.2.1: Implement `TargetsView` struct in `src/dataset/views.rs`
- [ ] 2.2.2: Implement `n_samples()`, `n_outputs()`
- [ ] 2.2.3: Implement `as_single_output()` with panic on n_outputs != 1
- [ ] 2.2.4: Implement `as_array() -> ArrayView2`
- [ ] 2.2.5: Unit tests

**Definition of Done:**

- Single-output and multi-output both work
- as_single_output panics clearly on multi-output

---

## Epic 3: Algorithm Integration and Cleanup

Integrate new Dataset with GBDT and GBLinear. Delete old code immediately after each migration.

### Story 3.1: Baseline Performance Capture

Capture baseline performance before any changes.

**Tasks:**

- [ ] 3.1.1: Run existing prediction benchmarks, record baseline numbers
- [ ] 3.1.2: Run existing training benchmarks, record baseline numbers
- [ ] 3.1.3: Document baselines in benchmark report

**Definition of Done:**

- Baseline numbers documented with commit hash
- Can compare after migration

### Story 3.2: GBDT Training Integration

Update GBDT training to use new Dataset.

**Tasks:**

- [ ] 3.2.1: Update `GBDTModel::train()` signature to accept `&Dataset`
- [ ] 3.2.2: Implement `Dataset::to_binned()` or use existing `BinnedDatasetBuilder`
- [ ] 3.2.3: Update categorical handling in binning (float → i32 cast)
- [ ] 3.2.4: Update tests to use new Dataset construction
- [ ] 3.2.5: Verify existing integration tests pass (`tests/inference_xgboost.rs`, `tests/training_*.rs`)

**Definition of Done:**

- GBDT training works with new Dataset
- Existing integration tests pass with same reference values
- No regression in training quality

### Story 3.3: GBDT Prediction with Block Buffering

Implement predictor-side block buffering as specified in RFC-0020.

**Tasks:**

- [ ] 3.3.1: Audit existing prediction code in `src/trees/predict.rs` to understand current structure
- [ ] 3.3.2: Implement `predict_into()` with parallel block iteration
- [ ] 3.3.3: Implement thread-local transpose buffer via `thread_local!`
  - Define `const PREDICT_BLOCK_SIZE: usize = 256;`
  - Buffer = 256 × n_features × 4 bytes (100KB for 100 features, fits L2 cache)
  - Use `RefCell<Vec<f32>>`, resize if n_features changes between calls
- [ ] 3.3.4: Implement `predict_block_into()` receiving buffer as parameter
- [ ] 3.3.5: Implement `transpose_block()` helper: `[n_features, block_samples]` → `[block_samples, n_features]`
- [ ] 3.3.6: Benchmark: compare against baseline from Story 3.1, verify ≤10% overhead (target 5%)
- [ ] 3.3.7: Update existing predict methods to use new infrastructure

**Definition of Done:**

- Prediction works with feature-major Dataset
- Thread-local buffers work correctly under parallel load
- Unit test verifies buffer resizes correctly when n_features changes
- Benchmark shows ≤10% overhead vs baseline captured in 3.1

### Story 3.4: GBLinear Training Integration

Update GBLinear training to use new Dataset.

**Tasks:**

- [ ] 3.4.1: Update `GBLinearModel::train()` to accept `&Dataset`
- [ ] 3.4.2: Add categorical feature validation (error if present)
- [ ] 3.4.3: Use `Dataset::features_view()` for coordinate descent
- [ ] 3.4.4: Update tests

**Definition of Done:**

- GBLinear training works with new Dataset
- Clear error on categorical features
- Existing tests pass

### Story 3.5: GBLinear Prediction

Ensure efficient prediction with per-feature iteration.

**Tasks:**

- [ ] 3.5.1: Audit existing prediction code for GBLinear
- [ ] 3.5.2: Ensure prediction uses per-feature iteration with column-major output
- [ ] 3.5.3: Benchmark: expect ~21% overhead vs row-major baseline (based on prior benchmark)

**Definition of Done:**

- GBLinear prediction works with feature-major Dataset
- Benchmark confirms overhead in expected range (~20-25%)

### Story 3.6: Delete Old Dataset and Cleanup

Remove old Dataset implementation completely. This happens immediately after Stories 3.2-3.5.

**Tasks:**

- [ ] 3.6.1: Identify all files using old types:
  - Old `Dataset` from `src/data/dataset.rs`
  - Old `FeaturesView`/`SamplesView` from `src/data/ndarray.rs`
- [ ] 3.6.2: Verify all usages migrated:
  - `grep -r "use.*data::Dataset"` - should find nothing
  - `grep -r "use super::\*"` in test modules - check for hidden imports
- [ ] 3.6.3: Delete `src/data/dataset.rs`
- [ ] 3.6.4: Update `src/data/mod.rs` exports
- [ ] 3.6.5: Run `cargo clippy --all-features --all-targets` - no dead code warnings
- [ ] 3.6.6: Run full test suite with `cargo test --all-features`
- [ ] 3.6.7: Document what remains in `src/data/` and why (binning, histograms, etc.)

**Definition of Done:**

- `src/data/dataset.rs` deleted
- No compile errors, no clippy warnings about dead code
- All tests pass

### Story 3.7: Cleanup Prediction API

Consolidate prediction functions to clean high/medium level separation.

**Tasks:**

- [ ] 3.7.1: Audit existing predict methods on `GBDTModel`, `GBLinearModel`, and predictors
- [ ] 3.7.2: High-level API (model level): `predict(&Dataset) -> Array2<f32>` only
  - Takes Dataset, returns owned output
  - Internally uses fastest path (column-major with buffering)
- [ ] 3.7.3: Medium-level API (predictor level): `predict_into`, `predict_row_into` as public
  - Allocation-free hot paths
  - Used by high-level API internally
- [ ] 3.7.4: Remove other predict variants (no longer needed with column-major decision)
- [ ] 3.7.5: Update GBLinear to match pattern: high-level `predict(&Dataset)`, medium-level `predict_into`

**Definition of Done:**

- High-level: `Model::predict(&Dataset)` is the only public prediction entry point
- Medium-level: `predict_into`, `predict_row_into` are public for advanced use
- Other variants removed (dead code cleaned)
- Both GBDT and GBLinear follow same pattern

### Story 3.8: Cleanup Examples and Benchmarks

Update all examples and benchmarks to use new Dataset.

**Tasks:**

- [ ] 3.8.1: Update examples in `crates/boosters/examples/`
- [ ] 3.8.2: Update benchmarks that construct datasets
- [ ] 3.8.3: Verify examples run correctly

**Definition of Done:**

- All examples compile and run
- All benchmarks work with new Dataset

---

## Epic 4: Stakeholder Feedback and Review

### Story 4.1: Stakeholder Feedback Check

Review stakeholder feedback after core implementation.

**Tasks:**

- [ ] 4.1.1: After Stories 1.1-1.3 complete, review `tmp/stakeholder_feedback.md`
- [ ] 4.1.2: Address any new feedback items
- [ ] 4.1.3: Update backlog if scope changes needed

**Definition of Done:**

- All feedback items reviewed
- New items either addressed or added to backlog

### Story 4.2: Implementation Review

Conduct review after algorithm integration.

**Tasks:**

- [ ] 4.2.1: After Epic 3 complete, prepare demo of:
  - New Dataset construction API
  - GBDT train/predict with new Dataset
  - GBLinear train/predict with new Dataset
  - Benchmark results (block buffering overhead, GBLinear overhead)
- [ ] 4.2.2: Document review in `tmp/development_review_<timestamp>.md`
- [ ] 4.2.3: Address any issues found

**Definition of Done:**

- Review completed and documented
- No blocking issues remain

### Story 4.3: Retrospective

Conduct retrospective after full implementation.

**Tasks:**

- [ ] 4.3.1: After Story 3.8 complete, run retrospective
- [ ] 4.3.2: Document in `tmp/retrospective.md`
- [ ] 4.3.3: Create backlog items for top improvement(s)

**Definition of Done:**

- Retrospective documented
- At least one improvement added to future backlog if warranted

---

## Quality Gate

Before closing this backlog, verify:

- [ ] All tests pass: `cargo test --all-features`
- [ ] No clippy warnings: `cargo clippy --all-features --all-targets`
- [ ] GBDT prediction overhead ≤10% vs baseline
- [ ] GBLinear prediction overhead ~20-25% vs row-major baseline
- [ ] `src/data/dataset.rs` deleted
- [ ] Old view types from `src/data/ndarray.rs` removed
- [ ] All new types exported from crate root

---

## Implementation Order

```text
Epic 1 (Dataset Core)
    ↓
Epic 2 (View Types) ← can overlap with late Epic 1
    ↓
Story 3.1 (Baseline Capture)
    ↓
Story 3.2 (GBDT Training) → Story 3.3 (GBDT Prediction)
    ↓
Story 3.4 (GBLinear Training) → Story 3.5 (GBLinear Prediction)
    ↓
Story 3.6 (Delete Old Code) → Story 3.7 (Cleanup Predict API) → Story 3.8 (Update Examples)
    ↓
Epic 4 (Reviews) ← interspersed throughout
```

## Risk Register

| Risk | Mitigation |
| ---- | ---------- |
| Breaking existing tests | Migrate incrementally, run tests often |
| Performance regression | Baseline capture (3.1), benchmark early (3.3.6, 3.5.3) |
| Incomplete cleanup | Explicit delete tasks, clippy for dead code |
| View API confusion | Clear rustdoc, semantic naming (feature/sample not row/col) |

## Notes

- We prioritize cleanup - old code deleted same sprint as migration
- Keep tasks granular enough to be checkpoints but not bureaucratic
- Run `cargo test` frequently during migration
- Benchmark block buffering and GBLinear prediction overhead
- Python bindings are out of scope - see future backlog
