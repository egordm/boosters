# EFB (Exclusive Feature Bundling) Quality Fix

## Status: Blocked

## Problem Summary

The current EFB implementation degrades model quality significantly:
- With EFB enabled: log_loss ~0.56, accuracy ~76%
- With EFB disabled: log_loss ~0.38, accuracy ~84%
- LightGBM reference: log_loss ~0.42, accuracy ~83%

## Root Cause Analysis

The current implementation has a fundamental architectural flaw:

1. **Current approach**: We build a **single histogram** for the entire bundled column (all sub-features combined) and find splits across the combined bins.

2. **Problem**: A split at encoded bin X (e.g., bin 5) mixes gradient statistics from different original features:
   - Left child: bins 0-5 may include all-defaults + feature_A + part of feature_B
   - Right child: bins 6+ may include rest of feature_B + feature_C
   
3. **Inference issue**: When we decode a bundled split to (feature, bin) for inference, we only apply the split to that single feature. But rows where *other* features were active in the bundle are not correctly handled.

## How LightGBM Handles This

LightGBM's approach (conceptually):
1. Features are grouped for **storage efficiency** (one column per group)
2. But histograms are still built **per-sub-feature** (not per-group)
3. Each sub-feature is evaluated independently for splits
4. Split finding uses `min_bin` and `max_bin` to isolate sub-feature ranges

Key code patterns from LightGBM:
- `Split(uint32_t min_bin, uint32_t max_bin, ...)` - partition checks if bin is in range
- Per-feature histogram arrays indexed by feature, not by group
- Feature groups used for memory layout, not histogram computation

## Proposed Fix

### Option A: Per-Sub-Feature Histogram Building (Recommended)
1. Keep bundled column storage for memory efficiency
2. During histogram building, iterate over bundled column but:
   - Decode each row's encoded bin to (sub_feature, original_bin)
   - Accumulate into the correct sub-feature's histogram slice
3. Split finding remains unchanged (per-feature)
4. Partition can use bundled column with bin-range checking
5. Inference uses original features (no decode needed)

**Pros**: Clean separation, correct semantics
**Cons**: Histogram building is more complex, some speedup lost

### Option B: Split on Bundled Column, Infer on Bundled Column
1. Keep current bundled histogram approach
2. Store splits as (bundled_col, encoded_bin) instead of decoding
3. During inference, compute bundled columns for test data
4. Apply splits using encoded bins

**Pros**: Fastest training and inference
**Cons**: Requires bundled column computation at inference time, complex

### Option C: Hybrid Approach
1. Use bundled columns for histogram building speedup
2. But find splits by iterating over sub-feature ranges within the histogram
3. Store splits with original features
4. Partition uses bin-range checking

**Pros**: Balanced complexity and correctness
**Cons**: Split finding needs sub-feature awareness

## Implementation Steps (Option A)

1. [ ] Modify histogram builder to accept bundled columns but output per-sub-feature histograms
2. [ ] Update histogram layout to account for sub-features within bundles
3. [ ] Ensure split finding iterates over original features, not bundled columns
4. [ ] Keep partition using bundled columns with bin-range checking for speed
5. [ ] Remove decode logic from `apply_split_to_builder` (splits already on original features)
6. [ ] Update tests to verify quality matches non-bundled training

## Definition of Done

- [ ] EFB-enabled training achieves same accuracy as EFB-disabled (within 0.5%)
- [ ] Training time improvement of at least 1.5x on sparse datasets
- [ ] All existing tests pass
- [ ] New test: verify bundled vs non-bundled training produces equivalent models

## References

- LightGBM source: `include/LightGBM/feature_group.h`, `src/io/dense_bin.hpp`
- LightGBM paper: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NeurIPS 2017)
- Current implementation: `crates/boosters/src/data/binned/bundling.rs`
