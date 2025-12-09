# Epic 5: GBTree Training Phase 2

**Status**: Complete  
**Priority**: High  
**Depends on**: Epic 4 (GBTree Phase 1 — Complete)

## Overview

Phase 2 extended the GBTree training implementation with advanced features:

- ✅ Categorical feature handling (LightGBM-style gradient-sorted splits)
- ✅ Sampling strategies (GOSS, row subsampling, column subsampling)
- ✅ Multi-output support (one-tree-per-output strategy)
- ⏸️ Monotonic and interaction constraints (delayed for proper implementation)
- ❌ Exclusive feature bundling (deferred — low priority for dense data)
- ❌ Multi-output trees with vector leaves (removed — see RFC-0024)

---

## Testing Philosophy

Testing was a core part of Phase 2 development. Each story included:

1. **Unit Tests**: Test individual functions and components in isolation
2. **Integration Tests**: Test feature end-to-end within training pipeline
3. **Validation Tests**: Compare output against XGBoost/LightGBM baselines
4. **Performance Tests**: Measure speed/memory against expectations
5. **Qualitative Tests**: Verify trained models are accurate and sensible

---

## RFCs

| RFC | Title | Status |
|-----|-------|--------|
| RFC-0016 | Categorical Feature Training | Implemented |
| RFC-0017 | Sampling Strategies | Implemented |
| RFC-0024 | Unified Multi-Output Training | Accepted (one-tree-per-output only) |
| RFC-0019 | Exclusive Feature Bundling | Deferred |
| RFC-0023 | Training Constraints | Delayed |

---

## Story 1: Categorical Feature Training ✅

**Goal**: Train trees that can split on categorical features

**RFCs**: RFC-0016

### Tasks

- [x] 1.1: Add `CategoricalInfo` to track which features are categorical
- [x] 1.2: Implement gradient summation by category in histogram builder
- [x] 1.3: Implement gradient-sorted partition finding (O(k log k)) in `find_best_categorical_split`
- [x] 1.4: Generate bitset for categories going left (`categories_left`)
- [x] 1.5: Integrate with `SplitInfo.is_categorical` and `categories_to_bitset()`
- [x] 1.6: Tests with synthetic categorical data (`test_categorical_split_basic`)
- [x] 1.7: Partition tests (`test_apply_split_categorical`)

### Implementation Notes

- Located in `src/training/gbtree/split.rs` and `src/training/gbtree/quantize.rs`
- Uses LightGBM-style gradient-sorted approach for optimal categorical splits
- Bitset encoding compatible with inference code

---

## Story 2: GOSS Sampling ✅

**Goal**: Gradient-based One-Side Sampling for faster training

**RFCs**: RFC-0017

### Tasks

- [x] 2.1: Add `GossParams` with `top_rate` and `other_rate` fields
- [x] 2.2: Implement gradient magnitude computation in `GossSampler`
- [x] 2.3: Implement top-gradient selection (keep `top_rate` fraction)
- [x] 2.4: Implement random sampling of remaining (`other_rate` fraction)
- [x] 2.5: Apply weight amplification to small gradients
- [x] 2.6: Create `GossSample` with indices and weights
- [x] 2.7: Integrate with trainer loop via `RowSamplingStrategy::Goss`
- [x] 2.8: Tests for GOSS correctness (`test_goss_*` in sampling.rs)

### Implementation Notes

- Located in `src/training/gbtree/sampling.rs`
- `GossSampler::sample()` returns indices + weight multipliers
- Weight amplification: small gradients get weight = `(1 - top_rate) / other_rate`

---

## Story 3: Row/Column Subsampling ✅

**Goal**: Bootstrap and feature subsampling for regularization

**RFCs**: RFC-0017

### Tasks

- [x] 3.1: Add `subsample` parameter to `TreeParams`
- [x] 3.2: Add `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` to params
- [x] 3.3: Implement `RowSampler` for row sampling per tree
- [x] 3.4: Implement `FeatureSampler` for hierarchical column sampling
- [x] 3.5: Ensure sampling is reproducible with seed-based RNG
- [x] 3.6: Tests: `test_train_with_subsample`, `test_subsample_reproducibility`

### Implementation Notes

- `RowSampler` in `src/training/gbtree/sampling.rs`
- `FeatureSampler` supports XGBoost-style hierarchical: bytree → bylevel → bynode
- Seeds derived from `trainer_seed + round + class_idx`

---

## Story 4: Multi-output Support (One Output Per Tree) ✅

**Goal**: Train separate trees per output (multiclass via softmax)

**RFCs**: RFC-0024

### Tasks

- [x] 4.1: Implement `train_multiclass()` method in trainer
- [x] 4.2: Create single-output gradient views per class
- [x] 4.3: Train K trees per round (one per class)
- [x] 4.4: Combine predictions via `SoAForest` with tree groups
- [x] 4.5: Tests with multiclass classification
- [x] 4.6: `freeze_multiclass_forest()` for inference format

### Implementation Notes

- Located in `src/training/gbtree/trainer.rs::train_multiclass()`
- Uses `GradientBuffer` with `n_outputs = num_classes`
- Each class has its own `RowPartitioner` and tree collection
- Forest stores trees in groups: group 0 = class 0 trees, etc.

---

## Story 5: Multi-output Support (Vector Leaves) — REMOVED

**Goal**: ~~Trees with vector-valued leaves~~

**Status**: Removed per RFC-0024

### Decision Rationale (RFC-0024, DD-1)

After research into XGBoost/LightGBM internals:
- Neither XGBoost nor LightGBM actually trains multi-output trees with vector leaves
- XGBoost's `multi_output_tree` parameter only affects JSON serialization, not training
- One-tree-per-output is the universal approach for gradient boosting
- Vector leaves are only useful for **inference** of pre-trained models

All K-output histogram/split infrastructure was removed to simplify codebase:
- `KVec<f32; 4>` (SmallVec) removed from `SplitInfo` and `BuildingNode`
- `smallvec` dependency removed
- Histogram simplified to scalar `f32` totals

See `docs/design/research/xgboost-gbtree/training/multi_output.md` for full analysis.

---

## Story 6: Exclusive Feature Bundling — DEFERRED

**Goal**: ~~Bundle mutually exclusive sparse features~~

**Status**: Deferred to future epic

### Rationale

- Primary use case is high-dimensional sparse data (CTR, one-hot encoded)
- Current focus is on dense numerical/categorical data
- Can be added later if needed for specific use cases

---

## Story 7: Monotonic Constraints ⏸️

**Status**: Delayed — implementation removed for clean API, to be reimplemented properly in future epic.

**Goal**: Enforce monotonic relationships

**RFCs**: RFC-0023

### Monotonic Tasks (Delayed)

- [ ] 7.1: Add `monotone_constraints` to `TreeParams`
- [ ] 7.2: Implement `MonotonicBounds` for bound tracking
- [ ] 7.3: Implement `MonotonicChecker` for split validation
- [ ] 7.4: Implement bounds propagation in `BuildingNode.compute_child_bounds()`
- [ ] 7.5: Implement leaf value clamping via `clamp_leaf_weight()`
- [ ] 7.6: Tests: `test_train_with_monotonic_constraints`, `test_monotonic_constraint_enforced_in_predictions`
- [ ] 7.7: Tests verify predictions are actually monotonic

---

## Story 8: Interaction Constraints ⏸️

**Status**: Delayed — implementation removed for clean API, to be reimplemented properly in future epic.

**Goal**: Limit feature interactions

**RFCs**: RFC-0023

### Interaction Tasks (Delayed)

- [ ] 8.1: Add `interaction_constraints` to `TreeParams`
- [ ] 8.2: Implement `InteractionConstraints` structure
- [ ] 8.3: Track allowed features per node via `allowed_features` in `BuildingNode`
- [ ] 8.4: Filter candidate features in split finder
- [ ] 8.5: Tests verifying interaction limits enforced
- [ ] 8.6: Integration with `TreeGrower`

---

## Dependencies

```text
Story 1 (Categorical) ──────────────────────── ✅ ─┐
Story 2 (GOSS) ─────────────────────────────── ✅ ─┤
Story 3 (Row/Col Sampling) ─────────────────── ✅ ─┤
                                                   ├──► Phase 2 Complete
Story 4 (Multi-output Per Tree) ────────────── ✅ ─┤
Story 5 (Multi-output Tree) ────────────────── ❌ ─┤ (removed)
Story 6 (Feature Bundling) ─────────────────── ⏸️ ─┤ (deferred)
Story 7 (Monotonic) ────────────────────────── ⏸️ ─┤ (delayed)
Story 8 (Interaction) ──────────────────────── ⏸️ ─┘ (delayed)
```

## Summary

| Story | Status | Notes |
|-------|--------|-------|
| 1. Categorical | ✅ Complete | LightGBM-style gradient-sorted |
| 2. GOSS | ✅ Complete | With weight amplification |
| 3. Row/Col Sampling | ✅ Complete | Hierarchical XGBoost-style |
| 4. Multi-output Per Tree | ✅ Complete | `train_multiclass()` |
| 5. Multi-output Tree | ❌ Removed | See RFC-0024 |
| 6. Feature Bundling | ⏸️ Deferred | Low priority |
| 7. Monotonic | ⏸️ Delayed | To be reimplemented properly |
| 8. Interaction | ⏸️ Delayed | To be reimplemented properly |
