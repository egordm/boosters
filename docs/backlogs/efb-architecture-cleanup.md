# Backlog: EFB Architecture Cleanup

**RFC**: RFC-0019-efb-architecture-cleanup.md

## Epic: EFB Code Quality Improvements

Clean up the EFB (Exclusive Feature Bundling) architecture to improve code organization, reduce duplication, and prepare for future performance optimizations.

---

## Story 1: Create Proper Types for Bundle Metadata

**Status**: ✅ Complete

**Description**:
Replace the complex tuple type `bundles: Vec<(u32, Vec<(usize, u32, u32)>)>` with properly named structs that clearly express the data structure.

**Changes**:

- Added `SubFeatureInfo` struct to `bundling.rs`
- Added `n_bins: Vec<u32>` field to `FeatureBundle`
- Added `sub_features()` and `sub_feature()` methods to `FeatureBundle`
- Exported `SubFeatureInfo` from `data/binned/mod.rs`

**Definition of Done**:

- [x] New structs added to `bundling.rs`
- [x] Types exported from `data/binned/mod.rs`
- [x] Documentation with examples
- [x] All 33 bundling tests pass

---

## Story 2: Move BundleDecoder to Bundling Module

**Status**: ✅ Complete

**Description**:
Move `BundleDecoder` from `histograms/ops.rs` to `data/binned/bundling.rs` where it belongs with other bundling types.

**Changes**:

- Moved `BundleDecoder` struct to `bundling.rs`
- `BundleDecoder` now stores `Vec<FeatureBundle>` directly instead of tuple types
- Simplified `from_plan()` - no longer needs `n_bins_fn` closure
- Added `n_bundles()`, `n_standalone()`, `n_columns()` methods
- Updated imports throughout codebase

**Definition of Done**:

- [x] `BundleDecoder` moved to `bundling.rs`
- [x] Uses `FeatureBundle` directly
- [x] All tests pass
- [x] Re-exported from `histograms/mod.rs` for convenience

---

## Story 3: Create Unified Histogram Building Interface

**Status**: ✅ Complete

**Description**:
Instead of separate accessor abstraction, added unified methods to `HistogramBuilder` that handle bundling internally.

**Changes**:

- Added `build_contiguous_with_decoder()` method to `HistogramBuilder`
- Added `build_gathered_with_decoder()` method to `HistogramBuilder`
- Grower now uses single code path with `decoder.as_ref()`
- Removed duplicate branching logic from grower

**Definition of Done**:

- [x] Unified methods on `HistogramBuilder`
- [x] Grower uses single code path
- [x] All tests pass

---

## Story 4: Unify Histogram Building Functions

**Status**: ✅ Complete

**Description**:
Remove the separate `build_unbundled_contiguous` and `build_unbundled_gathered` functions from public exports.

**Changes**:

- Made `build_unbundled_contiguous` private (removed `pub`)
- Made `build_unbundled_gathered` private (removed `pub`)
- Removed from `histograms/mod.rs` re-exports
- Removed imports from `grower.rs`

**Definition of Done**:

- [x] Unbundled functions are private (not exported)
- [x] Grower uses `HistogramBuilder` unified methods
- [x] All tests pass

---

## Story 5: Cleanup and Review

**Status**: ✅ Complete

**Description**:
Final cleanup, ensure all unused code is deleted, run benchmarks to verify no regression.

**Changes**:

- Verified no unused imports
- All public exports are intentional
- Ran quality benchmarks - results match expectations
- 637 tests pass

**Definition of Done**:

- [x] No unused imports or dead code (clippy clean)
- [x] All 637 tests pass
- [x] Quality benchmarks verified
- [x] Code review complete

---

## Summary of Changes

### Files Modified

1. **`crates/boosters/src/data/binned/bundling.rs`**:
   - Added `SubFeatureInfo` struct
   - Added `n_bins: Vec<u32>` to `FeatureBundle`
   - Added `sub_features()` and `sub_feature()` methods
   - Added `BundleDecoder` struct (moved from ops.rs)

2. **`crates/boosters/src/data/binned/mod.rs`**:
   - Exported `SubFeatureInfo` and `BundleDecoder`

3. **`crates/boosters/src/training/gbdt/histograms/ops.rs`**:
   - Removed old `BundleDecoder` (96 lines deleted)
   - Added `build_contiguous_with_decoder()` and `build_gathered_with_decoder()`
   - Made `build_unbundled_*` functions private

4. **`crates/boosters/src/training/gbdt/histograms/mod.rs`**:
   - Removed `build_unbundled_*` from exports
   - Re-export `BundleDecoder` from data module

5. **`crates/boosters/src/training/gbdt/grower.rs`**:
   - Removed duplicate if/else branching (~30 lines)
   - Uses unified `build_*_with_decoder()` methods
   - Simplified `BundleDecoder::from_plan()` call (no closure needed)

### Metrics

- **Lines removed**: ~130 lines of code
- **Tests passing**: 637 (all)
- **Clippy warnings**: 0 new (1 pre-existing)

---

## Future Work (Not in Scope)

- **RFC-0020**: LightGBM-style bundled histogram building
  - Build histograms on bundled columns directly
  - Decode during split finding using bin offsets
  - Would provide O(n_samples × n_bundled_columns) performance
