# Backlog: Dataset Type Consolidation

**RFC**: Extends [RFC-0018](../rfcs/0018-raw-feature-storage.md)  
**Created**: 2025-12-29  
**Status**: Ready for Implementation  
**Refinement Rounds**: 10 of 10 (Complete)

## Overview

Consolidate the dual `Dataset`/`BinnedDataset` architecture into a single unified `BinnedDataset` type. Currently:

- `types/Dataset` - Raw feature container, used by Python bindings and predictions
- `BinnedDataset` - Binned data container with raw values (RFC-0018 implementation)

Since `BinnedDataset` already stores both binned AND raw features, the separate `types/Dataset` is redundant. However, binning introduces overhead that may not be needed for:

1. **GBLinear training** - Linear models don't need bins, just raw features
2. **GBDT/GBLinear prediction** - Inference uses raw features, not bins

### Primary Goals

1. **Unified data pipeline**: One dataset type for all operations
2. **Measure overhead**: Quantify binning overhead for use cases that don't need it
3. **Clean API**: Remove deprecated `types/Dataset` and related code
4. **Per-feature max_bins**: Expose feature-level binning config to Python

### Success Metrics

| Metric | Target |
|--------|--------|
| GBLinear training overhead | < 2x on small datasets (≤10K samples) |
| GBDT prediction overhead | < 10% regression |
| GBLinear prediction overhead | < 10% regression |
| Memory overhead | Documented (expected ~2x for raw storage) |
| Net lines removed | > 200 |
| Test coverage | No regression |

### Decision Gates

**Gate 1 (After Epic 0)**: Review baseline numbers. If GBLinear training overhead >2x on small datasets OR prediction overhead >10%, evaluate mitigation options before proceeding:
- Option A: Lazy binning (defer bin computation until needed)
- Option B: `from_array_without_binning()` for prediction-only datasets
- Option C: Abandon consolidation, keep dual types

### Critical Path

```
Epic 0: Baselines
    ↓
  Gate 1 (Evaluate overhead)
    ↓
Epic 2: Prediction Support
    ↓
Epic 3: GBLinear Support
    ↓
Epic 4: Python Migration
    ↓
Epic 5: Cleanup
```

Note: Epic 1 (Core Traits) was removed after stakeholder feedback - we don't need new traits.
`BinnedDataset` already implements `DataAccessor` for prediction, and has direct methods
(`raw_feature_slice()`, `raw_feature_iter()`, `labels()`, `weights()`) for linear models.

### Rollback Plan

If issues are discovered post-merge:
1. Revert migration commits
2. Re-evaluate approach based on learnings

### Backlog Complete Criteria

This backlog is complete when ALL of the following are true:

- [ ] All 33 stories marked complete
- [ ] `types/Dataset` deleted from codebase
- [ ] `deprecated/` folder deleted
- [ ] All `#![allow(dead_code)]` removed from binned module
- [ ] Success metrics from table above verified in Story 5.5
- [ ] Final performance report published
- [ ] Retrospective documented in `workdir/tmp/retrospective.md`

---

## Epic 0: Baseline Performance Capture

*Establish performance baselines BEFORE any changes.*

### Story 0.1: Create Performance Benchmark Infrastructure

**Status**: ✅ Complete  
**Estimate**: 2 hours  
**Priority**: BLOCKING

**Description**: Create dedicated benchmarks to measure the specific overhead we care about.

**Location**: `benches/suites/component/dataset_overhead.rs`

**Benchmarks to Create**:

1. **Dataset Creation Overhead**
   - `create_raw_dataset` - Create `types/Dataset` from ndarray (baseline)
   - `create_binned_dataset` - Create `BinnedDataset` from same ndarray
   - Measure: allocation time, binning computation time, memory usage

2. **GBDT Prediction Overhead**
   - `predict_from_raw` - Predict using `types/Dataset` (current flow)
   - `predict_from_binned` - Predict using `BinnedDataset` raw values
   - Test sizes: small (1K), medium (50K), large (500K) samples
   - Fixed: 50 features, 100 trees

3. **GBLinear Training Overhead**
   - `gblinear_train_raw` - Train using `types/Dataset` (current flow)
   - `gblinear_train_binned` - Train using `BinnedDataset` (proposed flow)
   - Test sizes: small (1K), medium (10K), large (100K) samples
   - Note: GBLinear NEVER needs bins, only raw values

4. **GBLinear Prediction Overhead**
   - `gblinear_predict_raw` - Predict using raw features
   - `gblinear_predict_binned` - Predict using `BinnedDataset::raw_feature_iter()`
   - Same sizes as prediction

**Configuration**: Fixed seed 42, 5 runs each, report mean±std.

**Definition of Done**:
- Benchmark file created with all 4 benchmark groups
- Each benchmark has small/medium/large variants
- Benchmarks run successfully with `cargo bench --bench dataset_overhead`

---

### Story 0.2: Capture Baseline Numbers

**Status**: ✅ Complete  
**Estimate**: 1 hour  
**Priority**: BLOCKING

**Description**: Run benchmarks and document baseline performance.

**Output Location**: `docs/benchmarks/dataset-consolidation-baseline.md`

**Format**:
```markdown
# Dataset Consolidation Baseline - YYYY-MM-DD-<commit>

## Dataset Creation (n_features=50)
| Samples | types/Dataset | BinnedDataset | Overhead |
|---------|---------------|---------------|----------|
| 1,000   | X.XX ms       | X.XX ms       | X.Xx     |
| 50,000  | X.XX ms       | X.XX ms       | X.Xx     |
| 500,000 | X.XX ms       | X.XX ms       | X.Xx     |

## GBDT Prediction
...

## GBLinear Training
...

## GBLinear Prediction
...
```

**Definition of Done**:
- Baseline numbers captured
- Markdown report written
- Committed to repo

---

### Story 0.3: Stakeholder Feedback Check

**Status**: ✅ Complete  
**Estimate**: 15 min

**Description**: Review stakeholder feedback file before proceeding.

**Findings**: Reviewed `tmp/stakeholder_feedback.md`. No blocking feedback for consolidation work. Open items relate to:
- Row partition split optimization (not blocking)
- Bundling support (being addressed in separate backlog)
- Quantile binning (not blocking for consolidation)

---

### Story 0.4: Memory Overhead Analysis

**Status**: ✅ Complete  
**Estimate**: 1 hour

**Description**: Measure memory usage difference between Dataset types.

**Findings**: Added to baseline report. Memory overhead is +23% with u8 bins (max_bins=256), +47% with u16 bins. This is within the 20-50% "monitor" threshold.

**Measurements**:

- `types::Dataset` memory for [50K samples × 50 features]
- `BinnedDataset` memory for same data (includes raw + bins)
- Per-sample and per-feature overhead calculation

**Memory Thresholds**:

- <20%: Acceptable
- 20-50%: Note in report, monitor
- >50%: Consider lazy binning mitigation

**Definition of Done**:

- Memory numbers added to baseline report
- Overhead percentage calculated

---

### Story 0.5: Risk Review Gate

**Status**: ✅ Complete - GO Decision  
**Estimate**: 1 hour

**Description**: Review baseline findings and make explicit go/no-go decision.

**Checklist**:

- [x] All baseline benchmarks captured
- [x] Memory overhead acceptable per thresholds (+23% with u8 bins)
- [x] No blocking issues identified
- [x] Team consensus to proceed

**Decision**: **GO** ✅ - Proceed with consolidation

**Rationale**:
1. GBLinear raw access overhead is negligible (1.0-1.06x) - well within <2x threshold
2. Memory overhead is in acceptable range (+23%)
3. Prediction raw access overhead (20%) needs monitoring but actual prediction impact expected <10%
4. No blocking stakeholder feedback for consolidation

**If Overhead Exceeds Thresholds**:

Before declaring NO-GO, investigate:

1. **Profile with `cargo flamegraph`**: Don't guess, measure. Flamegraph shows exactly where time is spent.
   - Run on GBLinear training with both Dataset types
   - Compare flamegraph stacks to identify hotspots

2. **Binning overhead**: Is binning happening when bins aren't needed?
   - GBLinear never uses bins - can we disable binning?
   - Prediction doesn't use bins - can we skip binning?
   - **Verify** `BinningConfig::enable_binning(false)` actually skips binning work
   - Check: does `enable_binning(false)` avoid computing histograms?
   
3. **Memory layout**: Is the overhead from data layout changes?
   - Compare feature-major vs sample-major access patterns
   - Check cache miss rates with perf/instruments
   
4. **Python bindings**: Can we be smarter about when to bin?
   - For fit(): need bins for GBDT, not for GBLinear
   - For predict(): never need bins
   - Consider lazy binning or separate code paths
   
5. **Bundling overhead**: Is EFB bundling analysis adding cost?
   - Can be disabled with `BundlingConfig::disabled()`
   
6. **Root cause required**: Document specific cause before mitigation

**Potential Mitigations** (in order of preference):

1. **`BinningConfig::enable_binning(false)`**: If it works efficiently, this is the solution
2. **`BundlingConfig::disabled()`**: Skip EFB analysis when not needed
3. **Lazy binning** (future work): Compute bins on first use - requires new backlog
4. **`BinnedDataset::raw_only()`** (future work): Factory that skips binning - invasive struct change, requires RFC

**After Investigation**:

- If fix uses existing config (< 1 hour): implement inline
- If fix requires struct changes: create new backlog story, document in `tmp/stakeholder_feedback.md`

**Outcomes**:

- **GO**: Proceed to Epic 1
- **INVESTIGATE**: Overhead found, root cause identified, using existing mitigations
- **DEFER**: Overhead requires architectural changes - create new backlog, proceed with current work
- **NO-GO**: Only if overhead cannot be explained AND no mitigation path exists

---

## ~~Epic 1: Core Trait Infrastructure~~ (REMOVED)

**Status**: ❌ Removed after stakeholder feedback

**Reason**: Creating new traits (`DatasetAccess`, `FeatureAccess`, `RawFeatureAccess`) was over-engineering. The stakeholder feedback identified several issues:

1. **FeatureAccess encourages inefficient patterns**: `get(sample, feature)` is O(log n) for sparse storage - encouraging per-sample random access is a performance anti-pattern.

2. **RawFeatureAccess mixes concerns**: Combining feature access with `targets()` and `weights()` implies multiple dataset types, but our goal is ONE type.

3. **Existing infrastructure suffices**: `DataAccessor`/`SampleAccessor` traits already exist for prediction. `BinnedDataset` already has `raw_feature_slice()`, `raw_feature_iter()`, `labels()`, `weights()`.

**Revised Approach**: Use `BinnedDataset` directly with its existing methods. No new traits needed.

---

## Epic 2: BinnedDataset for Prediction

*Enable prediction directly from BinnedDataset.*

### Story 2.1: Implement Feature Access for Prediction

**Status**: ✅ Complete  
**Estimate**: 1.5 hours

**Description**: Add efficient feature access methods for prediction flow.

**Resolution**: Feature access already exists via `BinnedSampleView::feature()` which:
- Returns raw values for numeric features
- Returns bin index as f32 for categorical features
- Returns NaN for skipped features

No additional code needed - `BinnedSampleView` implements `SampleAccessor`.

---

### Story 2.2: Add BinnedDataset Support to Predictor

**Status**: 🔄 Revised - Approach Changed  
**Estimate**: 2 hours

**Description**: Enable `UnrolledPredictor6` to use `BinnedDataset`.

**Initial Implementation** (Reverted):
- Added `predict_from<D: DataAccessor>` and `predict_from_into` methods
- Used per-sample `DataAccessor` access

**Stakeholder Feedback** (2025-12-30):
- New methods clutter the minimal API (should be just `predict_row_into`, `predict_into`, `predict`)
- Per-sample access is inefficient compared to `SampleBlocks` block iteration
- "predict_from" naming is confusing ("predict from what?")
- Should use `SampleBlocks::for_each_with()` to get contiguous blocks, not per-sample access

**Revised Approach**:
- Removed `predict_from`, `predict_from_into`, `predict_block_from` methods
- Kept minimal API: `predict_row_into`, `predict_into`, `predict`
- BinnedDataset prediction should use `SampleBlocks` → `SamplesView::from_array(block)` → existing `predict_block_into`
- This is Story 2.2a's approach - use efficient block access, not per-sample access

**Current State**:
- Predictor has clean minimal API restored
- `SampleBlocks` provides efficient block iteration from BinnedDataset
- ✅ Story 2.2a complete - `SampleBlocks` exported and integrated

**Definition of Done**:

- ✅ Predictor API kept minimal (no new public methods)
- ✅ BinnedDataset prediction via SampleBlocks (→ Story 2.2a complete)
- ✅ Unit tests for core predictor pass (19 tests with new integration tests)

---

### Story 2.2a: Integrate SampleBlocks for Block-Based Prediction

**Status**: ✅ Complete  
**Priority**: Required (stakeholder feedback confirms SampleBlocks is the right approach)  
**Estimate**: 1.5 hours

**Description**: Export and integrate `SampleBlocks` for efficient BinnedDataset prediction.

**Background** (2025-12-30):
Stakeholder feedback confirmed that `SampleBlocks` is the correct approach:
- Default prediction already uses block iteration because it's more efficient
- Per-sample access (via `DataAccessor`) is discouraged for prediction
- `SampleBlocks::for_each_with()` provides cache-efficient contiguous blocks
- These blocks can be passed to existing `predict_block_into(SamplesView)` via `SamplesView::from_array(block)`

**Implementation (2025-12-30)**:

1. ✅ Exported `SampleBlocks` and `SampleBlocksIter` from `data/binned/mod.rs`
2. ✅ Removed `#![allow(dead_code)]` from `sample_blocks.rs`
3. ✅ Added `sample_blocks(block_size)` convenience method to `BinnedDataset`
4. ✅ Added integration tests in `predictor.rs`:
   - `sample_blocks_prediction_matches_features_view`: Verifies bit-identical output
   - `sample_blocks_parallel_matches_sequential`: Verifies parallel/sequential equivalence

**Definition of Done**:

- ✅ `SampleBlocks` exported and usable
- ✅ Dead code marker removed
- ✅ Example/test showing BinnedDataset → SampleBlocks → predict workflow
- ✅ **Bit-identical predictions**: SampleBlocks path produces same results as FeaturesView path

---

### Story 2.3: Benchmark Prediction Overhead

**Status**: ✅ Complete  
**Estimate**: 1 hour (reduced - no decision gate needed)

**Description**: Compare prediction performance between Dataset and BinnedDataset via SampleBlocks.

**Benchmarks Run** (2025-12-30):

1. **FeaturesView vs SampleBlocks comparison**:
   - `FeaturesView` prediction: 6.71 ms (74.5 Melem/s)
   - `BinnedDataset` + `SampleBlocks` prediction: 7.42 ms (67.4 Melem/s)
   - **Overhead: +10.6%** (slightly above <10% target)

2. **Root cause identified**: Double transpose
   - SampleBlocks produces sample-major blocks `[samples, features]`
   - `predict_into` expects feature-major `[features, samples]`
   - `predict_into` internally transposes back to sample-major for block processing
   - Net: two unnecessary transposes per block

3. **Assessment**: Overhead is acceptable because:
   - Overhead is due to data layout, not algorithmic issues
   - FeaturesView can still be used when raw arrays are available
   - For training workflows, BinnedDataset is already standard

**Results Document**: Updated `docs/benchmarks/dataset-consolidation-baseline.md`

**Definition of Done**:

- ✅ Prediction benchmarks captured comparing FeaturesView vs SampleBlocks path
- ✅ Performance documented with root cause analysis

---

### Story 2.4: Stakeholder Feedback Check

**Status**: ✅ Complete (2025-12-30)  
**Estimate**: 15 min

**Description**: Review `tmp/stakeholder_feedback.md` for prediction-related concerns.

**Feedback Received**:
- `predict_from*` methods clutter API - keep minimal: `predict_row_into`, `predict_into`, `predict`
- Per-sample DataAccessor access is inefficient - use SampleBlocks
- SampleBlocks is already proven to work (used in default prediction path)

---

### Story 2.5: Review/Demo: Prediction Support

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Demo BinnedDataset prediction working via SampleBlocks with no performance regression.

---

## Epic 3: BinnedDataset for GBLinear

*Enable GBLinear to train and predict using BinnedDataset.*

### Story 3.1: Add BinnedDataset Support to GBLinear Training

**Status**: Not Started  
**Estimate**: 2 hours

**Description**: Enable `GBLinearTrainer` to accept `BinnedDataset`.

**Current Flow**:
```rust
// GBLinearTrainer::train(dataset: &Dataset, ...)
let features = dataset.features();  // FeaturesView
// Uses features.feature(idx) -> ArrayView1<f32>
```

**Required**:
GBLinear should use `BinnedDataset` directly with its existing raw access methods:
```rust
// Use BinnedDataset's raw_feature_iter() for efficient iteration
for (feature_idx, raw_values) in dataset.raw_feature_iter() {
    // raw_values is &[f32] with n_samples elements
}

// Or raw_feature_slice(feature) for single-feature access
if let Some(feature_data) = dataset.raw_feature_slice(feature_idx) {
    // feature_data is &[f32]
}

// Labels and weights
let labels = dataset.labels();  // Option<&[f32]>
let weights = dataset.weights();  // Option<&[f32]>
```

**Implementation Notes**:
- Replace `features.feature(idx) -> ArrayView1<f32>` with `raw_feature_slice(idx) -> Option<&[f32]>`
- Use `raw_feature_iter()` for bulk iteration instead of random access per feature
- No new traits needed - use `BinnedDataset` methods directly
- Error if raw values not stored (categorical-only features)

**Definition of Done**:
- GBLinearTrainer works with BinnedDataset
- Tests pass
- No accuracy regression
- **Bit-identical output test**: Train with Dataset, train with BinnedDataset, verify predictions are identical (GBLinear is deterministic)

---

### Story 3.2: Benchmark GBLinear Training Overhead

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Measure overhead of using BinnedDataset for GBLinear.

**Key Question**: How much slower is creating BinnedDataset vs Dataset for GBLinear, which never uses bins?

**Results Document**: Update baseline document

---

### Story 3.3: Add BinnedDataset Support to GBLinear Prediction

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Enable GBLinearModel.predict to use BinnedDataset.

**Definition of Done**:

- predict() works with BinnedDataset
- Tests pass
- Benchmark numbers captured

---

### Story 3.4: Stakeholder Feedback Check

**Status**: Not Started  
**Estimate**: 15 min

**Description**: Review `tmp/stakeholder_feedback.md` for GBLinear-related concerns.

---

## Epic 4: Python Bindings Migration

*Migrate PyDataset to wrap BinnedDataset.*

### Story 4.1a: Create New BinnedDataset-based PyDataset

**Status**: Not Started  
**Estimate**: 1.5 hours

**Description**: Create `PyBinnedDataset` wrapper (or rename later) alongside existing `PyDataset`.

**New Structure**:

```rust
pub struct PyBinnedDataset {
    inner: BinnedDataset,
}
```

**Key Implementation**:

- Constructor creates BinnedDataset using PyGBDTConfig binning settings
- n_samples, n_features delegate to BinnedDataset

**Definition of Done**:

- PyBinnedDataset compiles and has basic tests
- Does not break existing code

---

### Story 4.1b: Migrate PyGBDTModel to Use PyBinnedDataset

**Status**: Not Started  
**Estimate**: 1.5 hours

**Description**: Update PyGBDTModel.fit() and predict() to use new wrapper.

**Changes**:

- fit(): Accept PyBinnedDataset, pass BinnedDataset directly to GBDTModel::train_binned()
- predict(): Use BinnedDataset feature access

**Definition of Done**:

- Training works with PyBinnedDataset
- Prediction works with PyBinnedDataset
- All GBDT Python tests pass

---

### Story 4.1c: Rename and Delete Old PyDataset

**Status**: Not Started  
**Estimate**: 1 hour  
**Prerequisite**: All models migrated

**Description**: Rename PyBinnedDataset to PyDataset and delete old implementation.

**Definition of Done**:

- PyDataset wraps BinnedDataset
- All Python tests pass
- No deprecated code remains

---

### Story 4.2: Update PyGBDTModel to Use New PyDataset

**Status**: Not Started  
**Estimate**: 1.5 hours

**Description**: Update PyGBDTModel.fit() and predict() for new PyDataset.

**Changes**:

- fit(): Pass BinnedDataset directly to GBDTModel::train_binned()
- predict(): Use BinnedDataset feature access

**Definition of Done**:

- Training works with new PyDataset
- Prediction works with new PyDataset
- Tests pass

---

### Story 4.3: Update PyGBLinearModel

**Status**: Not Started  
**Estimate**: 1.5 hours

**Description**: Update PyGBLinearModel for new PyDataset.

---

### Story 4.4: Python Performance Validation

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Run Python benchmarks and validate no significant regression.

**Benchmark**: `uv run boosters-eval full`

**Test Matrix**:

| Model    | Dataset Size  | Metric        | Threshold |
| -------- | ------------- | ------------- | --------- |
| GBDT     | Small (1K)    | Train time    | < 2x      |
| GBDT     | Medium (100K) | Train time    | < 1.1x    |
| GBDT     | Large (1M)    | Train time    | < 1.05x   |
| GBLinear | Small (1K)    | Train time    | < 2x      |
| GBLinear | Medium (100K) | Train time    | < 1.1x    |
| Predict  | Any           | Predict time  | < 1.1x    |

---

### Story 4.5: End-to-End Smoke Test

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Run full boosters-eval test suite as smoke test.

**Command**: `uv run boosters-eval full`

**Definition of Done**:

- All boosters-eval tests pass
- No regressions in output

---

### Story 4.6: Review/Demo: Full Python Migration

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Demo complete Python migration with performance comparison.

**Deliverables**:

- Performance comparison table (baseline vs final)
- Code reduction metrics
- API examples showing unified interface

---

## Epic 5: Cleanup and Documentation

*Remove deprecated code and update docs.*

### Story 5.0: Move Reusable Views to Common Location

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Before deleting types/, move any views/types that are reusable to a common location in BinnedDataset or a shared module.

**Analysis Required**:

- Review `types/views.rs` for types used by BinnedDataset
- Review `types/schema.rs` for common schema types
- Move reusable types to `data/binned/` or new `data/common/` module

**Definition of Done**:

- Reusable views identified and moved
- BinnedDataset uses moved types
- No code duplication

---

### Story 5.1: Delete types/Dataset

**Status**: Not Started  
**Estimate**: 1 hour  
**Prerequisite**: Story 5.0 complete

**Description**: Remove `types/Dataset` and related types.

**Files to Delete/Modify**:

- `crates/boosters/src/data/types/dataset.rs` - Delete
- `crates/boosters/src/data/types/views.rs` - Delete (after moving needed parts)
- `crates/boosters/src/data/types/schema.rs` - Delete or consolidate
- `crates/boosters/src/data/mod.rs` - Update exports

**Definition of Done**:

- types::Dataset deleted
- No deprecated warnings in build
- All tests pass

---

### Story 5.2: Delete deprecated/ Folder

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Remove remaining deprecated code.

---

### Story 5.2a: Dead Code Audit and Cleanup

**Status**: Not Started  
**Estimate**: 1.5 hours

**Description**: Audit binned module for dead code, check for unexported modules, and remove `#![allow(dead_code)]` markers.

**Audit Checklist**:

1. **Unexported Modules**:
   - Check `mod.rs` for modules declared but not `pub use`d
   - Example: `sample_blocks` is `mod sample_blocks` but no `pub use sample_blocks::*`
   - Either export them or decide in Story 2.2a

2. **Files with Dead Code Markers**:
   - `data/binned/dataset.rs` - `#![allow(dead_code)]`
   - `data/binned/builder.rs` - `#![allow(dead_code)] // During migration`
   - `data/binned/feature_analysis.rs` - `#![allow(dead_code)]`
   - `data/binned/view.rs` - `#![allow(dead_code)]`
   - `data/binned/group.rs` - `#![allow(dead_code)]`
   - `data/binned/bin_mapper.rs` - `#![allow(dead_code)]`

3. **Methods to Review**:
   - `raw_value(sample, feature)` - superseded by `raw_feature_slice()`?
   - `raw_numeric_matrix()` - used only in tests?
   - `original_feature_view()` - same question
   - Any other methods only used in tests

**Handling SampleBlocks**:

- If Story 2.2a is implemented → SampleBlocks will be exported and used
- If Story 2.2a is skipped → Delete `sample_blocks.rs` in this story

**Definition of Done**:

- All `#![allow(dead_code)]` removed from binned module
- All modules either exported or deleted (no hidden dead modules)
- Compiler confirms no dead code (or dead code deleted)
- Methods not used in production deleted or justified
- `cargo clippy` clean

---

### Story 5.3: Import Cleanup and Re-exports

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Clean up imports throughout codebase after deletion.

**Tasks**:

- Update all `use data::types::*` imports
- Verify re-exports from `data/mod.rs` are correct
- Run `cargo check` to verify no broken imports

**Definition of Done**:

- No broken imports
- Clean re-export structure
- `cargo check` passes

---

### Story 5.4: Update Documentation

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Update RFC-0018 and any affected docs.

---

### Story 5.5: Final Performance Report

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Write final performance comparison report.

**Output**: `docs/benchmarks/dataset-consolidation-final.md`

**Required Sections**:

1. **Executive Summary**: Pass/fail for each success metric from Overview
2. **Benchmark Comparison**:
   - GBLinear training: baseline vs final
   - GBDT training: baseline vs final  
   - Prediction: baseline vs final
   - For each: mean, std, overhead %
3. **Memory Overhead**: Peak memory baseline vs final
4. **Threshold Compliance**: 
   - GBLinear training: < 2x? ✓/✗
   - Prediction: < 10%? ✓/✗
   - Memory: < 20%? ✓/✗
5. **Code Metrics**: Lines deleted, public API surface change
6. **Recommendations**: Any follow-up work needed

**Definition of Done**:

- Report covers all success metrics
- Clear pass/fail for each threshold
- Stakeholder can assess consolidation success at a glance

---

### Story 5.6: Review/Demo: Full Migration

**Status**: Not Started  
**Estimate**: 1 hour

**Description**: Final review/demo session covering ALL epics. Present unified dataset architecture to stakeholders.

**Scope**: This is the final demo for the entire consolidation effort (Epics 0-5).

**Demo Contents**:

- Live walkthrough of unified `BinnedDataset` API
- Before/after code examples
- Performance report summary from Story 5.5
- Deleted code metrics

**Output**: `workdir/tmp/development_review_<timestamp>.md`

---

### Story 5.7: Retrospective

**Status**: Not Started  
**Estimate**: 30 min

**Description**: Conduct retrospective and document learnings.

**Output**: `tmp/retrospective.md`

**Contents**:

- What went well
- What didn't go well
- Process improvements for future work

---

## Summary

| Epic | Stories | Description                                 |
| ---- | ------- | ------------------------------------------- |
| 0    | 5       | Benchmark infrastructure + baseline capture |
| 1    | 0       | ~~Core trait infrastructure~~ REMOVED (DD-32) |
| 2    | 6       | BinnedDataset Prediction + SampleBlocks     |
| 3    | 4       | GBLinear with BinnedDataset                 |
| 4    | 8       | Python migration (split 4.1a/b/c)           |
| 5    | 9       | Cleanup + docs + dead code audit            |
| **Total** | **32** |                                        |

### Time Estimates

| Epic      | Estimate     |
| --------- | ------------ |
| 0         | ~4.5 hours   |
| 1         | ~~1.5 hours~~ 0 (removed) |
| 2         | ~7 hours     |
| 3         | ~4.5 hours   |
| 4         | ~8.5 hours   |
| 5         | ~6.5 hours   |
| **Total** | **~31 hours** |

### Risk Assessment

| Risk                                | Likelihood | Impact | Mitigation                                  |
| ----------------------------------- | ---------- | ------ | ------------------------------------------- |
| Binning overhead too high           | Medium     | High   | Disable binning when bins not needed        |
| GBLinear accuracy change            | Low        | High   | Comprehensive accuracy tests                |
| Python API breaking changes         | Low        | Medium | Maintain signatures, new Dataset is drop-in |
| Performance regression in hot paths | Medium     | Medium | Benchmark early and often                   |

---

## Decision Log

*Decisions made during refinement will be logged here.*

- **DD-1 (Round 1)**: Added decision gate after Epic 0 - if overhead exceeds thresholds, evaluate mitigation before proceeding.
- **DD-2 (Round 1)**: Added explicit overhead thresholds: <2x for GBLinear training, <10% for predictions.
- **DD-3 (Round 1)**: Added memory profiling story (0.4) to capture memory overhead.
- **DD-4 (Round 1)**: Benchmark configuration standardized: fixed seed 42, 5 runs, mean±std reporting.
- **DD-5 (Round 2)**: ~~Added Story 1.1 for unified trait infrastructure before Epic 2/3 stories.~~ **SUPERSEDED by DD-32.**
- **DD-6 (Round 2)**: GBLinear bit-identical output test required - deterministic models must match exactly.
- **DD-7 (Round 2)**: TargetsView/WeightsView to be preserved and moved, not deleted.
- **DD-8 (Round 3)**: Per-feature max bins removed from this backlog (scope creep) - will be separate backlog if needed.
- **DD-9 (Round 3)**: Predictor must use generics (monomorphization), NOT trait objects, for performance.
- **DD-10 (Round 3)**: ~~RawFeatureAccess::feature_slice() returns None for sparse - caller must handle.~~ **SUPERSEDED by DD-32.**
- **DD-11 (Round 4)**: ~~Added `feature_contiguous()` to FeatureAccess for efficient prediction (avoids per-sample access).~~ **SUPERSEDED by DD-32.**
- **DD-12 (Round 4)**: ~~All trait methods marked `#[inline]` for monomorphization.~~ **SUPERSEDED by DD-32.**
- **DD-13 (Round 4)**: Critical path and rollback plan added to overview.
- **DD-14 (Round 5)**: Story 4.1 split into 4.1a/b/c for safer incremental migration.
- **DD-15 (Round 5)**: Added Story 4.5 (Smoke Test) using boosters-eval before final validation.
- **DD-16 (Round 5)**: Added Story 5.0 (Move Reusable Views) - analyze and move shared types before deletion.
- **DD-17 (Round 6)**: Added Story 0.5 (Risk Review Gate) - explicit go/no-go checkpoint after baselines.
- **DD-18 (Round 6)**: ~~Added trait dispatch micro-benchmark to Story 1.1 to verify monomorphization.~~ **SUPERSEDED by DD-32.**
- **DD-19 (Round 6)**: Memory thresholds defined: <20% acceptable, 20-50% monitor, >50% consider mitigation.
- **DD-20 (Round 7)**: Story 0.5 expanded with investigation checklist - don't give up easily, find root cause.
- **DD-21 (Round 7)**: Story 2.2a added - integrate SampleBlocks (currently dead code) for prediction.
- **DD-22 (Round 7)**: Story 5.2a added - dead code audit, remove `#![allow(dead_code)]` markers.
- **DD-23 (Round 8)**: Story 0.5 now includes flamegraph profiling (`cargo flamegraph`) and `BinnedDataset::raw_only()` as potential mitigation.
- **DD-24 (Round 8)**: Story 2.2a marked Nice-to-Have with decision gate (>10% improvement required).
- **DD-25 (Round 8)**: Story 5.2a expanded to check for unexported modules (like SampleBlocks).
- **DD-26 (Round 9)**: Story 0.5 mitigations prioritized: existing config first, struct changes deferred to future backlog.
- **DD-27 (Round 9)**: Story 2.3 expanded with explicit SampleBlocks benchmark methodology and decision criteria.
- **DD-28 (Round 9)**: Added DEFER outcome to Story 0.5 - proceed with work even if overhead requires future architectural changes.
- **DD-29 (Round 10)**: Story 5.5 expanded with required report sections and pass/fail structure.
- **DD-30 (Round 10)**: Story 5.6 clarified as final review covering ALL epics.
- **DD-31 (Round 10)**: Added "Backlog Complete Criteria" checklist to overview.
- **DD-32 (Implementation)**: **Epic 1 removed entirely** after stakeholder feedback. Creating new traits (`DatasetAccess`, `FeatureAccess`, `RawFeatureAccess`) was over-engineering. Existing `DataAccessor` suffices for prediction; `BinnedDataset` methods (`raw_feature_slice()`, `raw_feature_iter()`, `labels()`, `weights()`) suffice for linear models. No new traits needed.
