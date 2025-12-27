# Backlog: LightGBM-Style EFB Implementation

**RFC**: [docs/rfcs/rfc-lightgbm-style-efb.md](../rfcs/rfc-lightgbm-style-efb.md)  
**Created**: 2025-12-27  
**Status**: Active

## Overview

Implement LightGBM-style Exclusive Feature Bundling where bundles are treated as regular columns, eliminating decoding during histogram building.

---

## Epic 1: Preparation and Verification

### Story 1.1: Verify Bundled API Correctness ✅ COMPLETE
**Description**: Verify existing bundled APIs work correctly before making changes.

**Tasks**:
- [x] Verify `bundled_feature_views()` returns correct views
- [x] Verify `bundled_bin_counts()` returns correct counts
- [x] Verify `decode_bundle_split()` handles edge cases (bin 0 → None)
- [x] Run existing test suite

**Definition of Done**:
- All existing tests pass
- `test_bundle_plan_decode_bundle_split` verifies edge cases

**Notes**: Verified during design review. Existing implementation is correct.

---

## Epic 2: Update Histogram Building

### Story 2.1: Cache Bundled Views in TreeGrower
**Description**: Store bundled feature views and layout in grower to avoid re-allocation per histogram build.

**Tasks**:
- [ ] Add `bundled_views: Vec<FeatureView<'static>>` field to TreeGrower (or store dataset reference)
- [ ] Add `n_bundled_columns: usize` field
- [ ] Compute and cache during `TreeGrower::new()`

**Definition of Done**:
- Views are computed once, not per-histogram-build
- No new allocations in hot path

### Story 2.2: Update Histogram Layout to Use Bundled Columns
**Description**: Change `HistogramLayout` to be based on bundled columns, not original features.

**Tasks**:
- [ ] In `TreeGrower::new()`, use `dataset.bundled_bin_counts()` for layout
- [ ] Update `n_features` references to `n_bundled_columns` where appropriate
- [ ] Verify histogram pool has correct total bins

**Definition of Done**:
- Histogram size = sum of bundled column bins
- Layout has one entry per bundled column

### Story 2.3: Update Build Methods to Use Bundled Views
**Description**: Change histogram building to iterate over bundled column views.

**Tasks**:
- [ ] Update `build_contiguous()` to use bundled views
- [ ] Update `build_gathered()` to use bundled views
- [ ] Ensure column index matches layout index

**Definition of Done**:
- Histogram building uses bundled views
- All existing histogram tests pass (with updated expectations)

### Story 2.4: Stakeholder Feedback Check
**Description**: Review stakeholder feedback for Epic 2.

**Tasks**:
- [ ] Check `tmp/stakeholder_feedback.md` for relevant feedback
- [ ] Incorporate feedback or defer to new stories

---

## Epic 3: Update Split Finding

### Story 3.1: Add Column-to-Feature Mapping
**Description**: Add helper to map bundled column index to original feature.

**Tasks**:
- [ ] Add method `bundled_column_to_split(column_idx, bin)` that returns `(orig_feature, orig_bin)`
- [ ] Handle bundle columns via `decode_bundle_split()`
- [ ] Handle standalone columns via direct lookup

**Definition of Done**:
- All column types correctly mapped to original features
- Unit tests cover bundle and standalone cases

### Story 3.2: Update Split Finder to Use Bundled Columns
**Description**: Change split finding to iterate over bundled columns.

**Tasks**:
- [ ] Update loop to iterate `0..n_bundled_columns`
- [ ] Use bundled layout for histogram indexing
- [ ] Decode split before storing in tree

**Definition of Done**:
- Splits found on bundled columns are correctly decoded
- Tree stores original feature indices

### Story 3.3: Verify Tree Output
**Description**: Verify trained trees have correct original feature indices.

**Tasks**:
- [ ] Add test that trains with bundling, inspects tree splits
- [ ] Verify feature indices are original (not bundled column indices)

**Definition of Done**:
- Tree dump shows original feature names/indices
- Prediction works correctly

### Story 3.4: Review and Demo - Epic 2 & 3
**Description**: Review completed work and demonstrate value.

**Tasks**:
- [ ] Prepare demonstration: before/after histogram building
- [ ] Show metrics: code paths reduced, performance improvement
- [ ] Document in `tmp/development_review_<timestamp>.md`

---

## Epic 4: Code Cleanup

### Story 4.1: Remove BundledU8/BundledU16 from FeatureView
**Description**: Remove the now-unused bundled variants from FeatureView enum.

**Tasks**:
- [ ] Remove `BundledU8` and `BundledU16` variants from storage.rs
- [ ] Remove associated helper methods (`is_bundled()`, decode-related)
- [ ] Update any match arms that handled these variants

**Definition of Done**:
- `FeatureView` has only 4 variants (U8, U16, SparseU8, SparseU16)
- No compile errors

### Story 4.2: Remove BundleHistogramCache
**Description**: Remove the now-unused bundle histogram cache.

**Tasks**:
- [ ] Remove `BundleHistogramCache` struct from bundling.rs
- [ ] Remove related `BundleDecoder` types if unused
- [ ] Remove `bundle_histogram_cache` field from TreeGrower
- [ ] Remove cache creation code from grower initialization

**Definition of Done**:
- No references to `BundleHistogramCache` remain
- Code compiles without dead code warnings

### Story 4.3: Remove Unused Kernel Functions
**Description**: Remove bundled-specific histogram kernel functions.

**Tasks**:
- [ ] Remove `build_bundled_*` functions from ops.rs
- [ ] Remove `build_*_with_cache` functions from ops.rs
- [ ] Remove any bundled-specific dispatch code

**Definition of Done**:
- ops.rs contains only unified kernel functions
- No dead code warnings

### Story 4.4: Final Cleanup and Clippy
**Description**: Run final code quality checks.

**Tasks**:
- [ ] Run `cargo clippy` and fix warnings
- [ ] Run `cargo test` to verify all tests pass
- [ ] Verify no unused imports or dead code

**Definition of Done**:
- Zero clippy warnings related to bundling
- All tests pass

---

## Epic 5: Validation and Benchmarking

### Story 5.1: Add Bundled Histogram Tests
**Description**: Add specific tests for bundled histogram building.

**Tasks**:
- [ ] Test: bundled vs non-bundled histogram sums are equivalent
- [ ] Test: split on bundle column decodes correctly
- [ ] Test: end-to-end training with bundling produces valid model

**Definition of Done**:
- New tests in appropriate test module
- Tests cover edge cases (all-default rows, single-feature bundle)

### Story 5.2: Benchmark Performance
**Description**: Measure performance improvement from this change.

**Tasks**:
- [ ] Benchmark histogram building on Airlines dataset
- [ ] Benchmark full training time
- [ ] Compare with baseline (before changes)
- [ ] Compare with LightGBM

**Definition of Done**:
- Performance metrics documented
- Training time improved (target: <1.8s competitive with LightGBM)

### Story 5.3: Retrospective
**Description**: Conduct retrospective on implementation.

**Tasks**:
- [ ] Document what went well
- [ ] Document what could be improved
- [ ] Capture action items as future backlog stories
- [ ] Write to `tmp/retrospective.md`

---

## Dependencies

- Story 2.2 depends on 2.1
- Story 2.3 depends on 2.2
- Story 3.2 depends on 3.1
- Story 3.3 depends on 3.2
- Epic 4 depends on Epic 3 completion
- Epic 5 depends on Epic 4 completion
