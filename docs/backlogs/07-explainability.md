# Explainability Backlog

**Source**: [RFC-0020: Explainability](../rfcs/0020-explainability.md)  
**Created**: 2025-12-19  
**Status**: ✅ COMPLETE - 2025-12-20

---

## Overview

This backlog implements feature importance and SHAP value computation for model explainability.

**Dependencies**: [Epic 2: Model API](06-model-api-and-python.md) (for model types) ✅  
**Enables**: Python explainability APIs, SHAP library integration ✅

**Estimated Total Effort**: ~6-8 hours (implementation) + ~2 hours (reference data generation)

**Testing Setup Note**: SHAP tests require reference values. Generate using `shap` Python library on equivalent XGBoost models. See [Reference Values](#reference-values) section.

---

## Epic 4: Explainability

**Goal**: Provide feature importance and SHAP values for all model types.

**Why**: Users need to understand model predictions for debugging, feature selection, and regulatory compliance.

---

### Story 4.0: Extend Tree with Optional Statistics

**RFC Section**: RFC-0020 "Note on Node Statistics"  
**Effort**: S (< 30min)

**Description**: Add optional `gains` and `covers` fields to `Tree` struct. These are needed for feature importance and SHAP computation.

**Note**: This story must be completed before Epic 1 (Storage) serializes these fields. Coordinate with storage work.

**Where Stats Come From**:  
During `Trainer.train()`, the trainer populates `TreeStats` for each tree as a byproduct of the split-finding process (gain, cover, sample counts are computed anyway). These stats flow: Trainer → Forest → ForestPayload → .bstr file.

**If training is skipped** (loading pre-trained model): Stats come from the serialized file if present, or are marked as unavailable (returning `None`).

**Tasks**:

- [ ] 4.0.1 Add `gains: Option<Box<[f32]>>` to `Tree<L>` struct
- [ ] 4.0.2 Add `covers: Option<Box<[f32]>>` to `Tree<L>` struct
- [ ] 4.0.3 Add accessor methods: `gain(node_idx) -> Option<f32>`, `cover(node_idx) -> Option<f32>`
- [ ] 4.0.4 Update existing `Tree` constructors to default these to `None`
- [ ] 4.0.5 Verify existing tests still pass (fields are optional)

**Definition of Done**:

- `cargo test` passes
- Tree struct has optional stats fields
- No behavior change for existing code

**Testing Criteria**:

- Existing tests pass unchanged
- New accessor methods work correctly with `None` values
- Accessor returns `None` when stats not populated

---

### Story 4.1: Implement Full Feature Importance

**RFC Section**: RFC-0020 "Feature Importance"  
**Effort**: M (1-2h)

**Description**: Extend the basic split-count importance (from Epic 2) to support gain-based and cover-based importance types.

**Prerequisite**: Basic `feature_importance()` returning split counts exists in `GBDTModel` (Epic 2, Story 2.3).

**Tasks**:

- [ ] 4.1.1 Create `src/explainability/mod.rs` module
- [ ] 4.1.2 Implement `ImportanceType` enum (Split, Gain, AverageGain, Cover, AverageCover)
- [ ] 4.1.3 Implement `FeatureImportance` struct:
  - values: Vec<f64>
  - importance_type: ImportanceType
  - feature_names: Option<Vec<String>>
- [ ] 4.1.4 Implement `FeatureImportance` methods:
  - `get(idx)`, `get_by_name(name)`
  - `iter()`, `sorted()`, `top_k(k)`
  - `to_map() -> HashMap`
  - `normalized()` (sums to 1.0)
- [ ] 4.1.5 Extend `Forest::feature_importance(imp_type)` to support Gain/Cover:
  - For Split: use existing implementation
  - For Gain/Cover: sum values (requires gains/covers on Tree)
  - Return `ExplainError::MissingNodeStats` if gain/cover needed but not present
- [ ] 4.1.6 Implement `LinearModel::feature_importance()`:
  - Split = non-zero weight count
  - Gain = abs(weight) optionally * std(feature)
- [ ] 4.1.7 Update `GBDTModel::feature_importance()` to accept `ImportanceType` parameter

**Definition of Done**:

- Feature importance computable for both model types
- All ImportanceType variants work (with stats available)
- Clear error when stats are missing

**Testing Criteria**:

- Split importance works without node stats
- Gain/Cover importance fails gracefully without stats
- top_k returns correct ordering
- normalized() sums to 1.0

---

### Story 4.2: Populate Tree Statistics During Training

**RFC Section**: RFC-0020 "Training Integration"  
**Effort**: M (1-2h)

**Description**: Capture gain and cover values during tree building for explainability.

**Tasks**:

- [ ] 4.2.1 Modify `TreeBuilder::apply_split()` to accept gain and cover
- [ ] 4.2.2 Store gains and covers in growing tree structure
- [ ] 4.2.3 Transfer gains/covers to final `Tree` struct after build
- [ ] 4.2.4 Update `GBDTTrainer` to pass gain/cover to TreeBuilder
- [ ] 4.2.5 Update serialization to include gains/covers

**Definition of Done**:

- Freshly trained models have gains/covers populated
- Gain-based importance works on trained models

**Testing Criteria**:

- Train model → feature_importance(Gain) works
- Train model → save → load → feature_importance(Gain) still works
- Gains are non-negative, covers are positive

---

### Story 4.3: Implement PathState for TreeSHAP

**RFC Section**: RFC-0020 "TreeSHAP Algorithm"  
**Effort**: M (1h)

**Description**: Implement the path tracking structure used by TreeSHAP algorithm.

**Tasks**:

- [ ] 4.3.1 Create `src/explainability/shap/mod.rs` and `path.rs`
- [ ] 4.3.2 Implement `PathState` struct:
  - features: Vec<i32>
  - zero_fractions: Vec<f64>
  - one_fractions: Vec<f64>
  - weights: Vec<f64>
- [ ] 4.3.3 Implement `PathState::new(n_features)`
- [ ] 4.3.4 Implement `PathState::extend(feature, zero_fraction, one_fraction)`:
  - Update weights using SHAP path algorithm
- [ ] 4.3.5 Implement `PathState::unwind()` for backtracking
- [ ] 4.3.6 Implement `PathState::unwound_sum(target_idx)` for contribution computation

**Definition of Done**:

- PathState correctly tracks path weights
- extend/unwind are inverse operations

**Testing Criteria**:

- Simple path: extend(f1) → unwind() leaves empty state
- Weight computation matches expected values for known toy paths
- Multiple extend/unwind cycles work correctly

---

### Story 4.4: Implement ShapValues Container

**RFC Section**: RFC-0020 "SHAP Values"  
**Effort**: S (30min)

**Description**: Container for SHAP values with proper indexing and verification.

**Tasks**:

- [ ] 4.4.1 Create `src/explainability/shap/values.rs`
- [ ] 4.4.2 Implement `ShapValues` struct:
  - values: Vec<f64>
  - n_samples, n_features, n_outputs
- [ ] 4.4.3 Implement indexing: `get(sample, feature, output)`
- [ ] 4.4.4 Implement `base_value(sample, output)`
- [ ] 4.4.5 Implement `sample(idx)` to get slice for one sample
- [ ] 4.4.6 Implement `verify(predictions, tolerance)` for consistency check
- [ ] 4.4.7 Implement `to_3d_array()` for Python conversion

**Definition of Done**:

- ShapValues stores and retrieves values correctly
- verify() checks sum property

**Testing Criteria**:

- Indexing works: get(0, 0, 0) returns first value
- base_value returns correct element
- verify() returns true for correct values, false for incorrect

---

### Story 4.5: Implement TreeExplainer Core

**RFC Section**: RFC-0020 "TreeSHAP Algorithm"  
**Effort**: L (2-3h)

**Description**: Core TreeSHAP implementation for single trees and forests.

**Tasks**:

- [ ] 4.5.1 Create `src/explainability/shap/tree_explainer.rs`
- [ ] 4.5.2 Implement `TreeExplainer` struct holding forest reference
- [ ] 4.5.3 Implement `shap_values(data) -> ShapValues`:
  - Allocate output array
  - Process each sample
  - Add base values
- [ ] 4.5.4 Implement `shap_for_sample()`:
  - Initialize PathState
  - Recurse through each tree
  - Accumulate contributions
- [ ] 4.5.5 Implement `tree_shap_recursive()`:
  - Handle leaf nodes (compute contributions)
  - Handle internal nodes (recurse into hot/cold paths)
  - Track fractions based on node covers
- [ ] 4.5.6 Handle missing values using default_left
- [ ] 4.5.7 Return `ExplainError::MissingNodeStats` if covers unavailable

**Definition of Done**:

- SHAP values computed for forest
- sum(shap) + base = prediction property holds

**Testing Criteria**:

- Single tree: SHAP sums to prediction
- Multi-tree forest: SHAP sums to prediction
- All samples satisfy verification within tolerance (1e-6)
- Missing covers produces clear error

---

### Story 4.6: Implement LinearExplainer

**RFC Section**: RFC-0020 "Linear SHAP"  
**Effort**: M (1h)

**Description**: SHAP values for linear models (closed-form solution).

**Tasks**:

- [ ] 4.6.1 Create `src/explainability/shap/linear_explainer.rs`
- [ ] 4.6.2 Implement `LinearExplainer` with model and feature_means
- [ ] 4.6.3 Implement `shap_values(data)`:
  - For each sample, feature: contribution = weight * (x - mean)
  - Base value = sum(weight * mean) + bias
- [ ] 4.6.4 Return `ExplainError::MissingFeatureStats` if means not available

**Definition of Done**:

- Linear SHAP produces correct values
- Closed-form computation is fast

**Testing Criteria**:

- Linear SHAP matches manual calculation: w * (x - mean)
- sum(shap) + base = prediction
- Works for multi-output linear models

---

### Story 4.7: Integrate with Model API

**Effort**: S (30min)

**Description**: Add explainability methods to high-level model types.

**Tasks**:

- [ ] 4.7.1 Add `feature_importance(type)` to `GBDTModel`
- [ ] 4.7.2 Add `shap_values(data)` to `GBDTModel`
- [ ] 4.7.3 Add `feature_importance()` to `GBLinearModel`
- [ ] 4.7.4 Add `shap_values(data)` to `GBLinearModel`
- [ ] 4.7.5 Store `FeatureStats` (means) during training for linear SHAP

**Definition of Done**:

- Explainability accessible via model objects
- Feature stats captured during training

**Testing Criteria**:

- `model.feature_importance(Gain)` works for GBDT
- `model.shap_values(X)` works for both model types

---

### Story 4.8: Add Python Explainability Bindings

**Effort**: M (1-2h)

**Description**: Expose explainability to Python.

**Tasks**:

- [ ] 4.8.1 Add `feature_importance(importance_type="gain")` to PyGBDTBooster:
  - Return dict mapping feature name → importance
- [ ] 4.8.2 Add `shap_values(data)` to PyGBDTBooster:
  - Return NumPy array of shape (n_samples, n_features)
- [ ] 4.8.3 Add `expected_value` property for base value
- [ ] 4.8.4 Add same methods to PyGBLinearBooster
- [ ] 4.8.5 Handle errors gracefully (MissingNodeStats → clear Python error)

**Definition of Done**:

- Python users can compute importance and SHAP
- Output format compatible with `shap` library

**Testing Criteria**:

- `model.feature_importance()` returns dict
- `model.shap_values(X)` returns correct shape array
- Works with shap.summary_plot (manual verification)

---

## Summary

### Story Order

| Order | Story | Effort | Blocked By |
| ----- | ----- | ------ | ---------- |
| 1 | 4.0 Extend Tree with Stats | S | None |
| 2 | 4.1 Feature Importance | M | 4.0 |
| 3 | 4.2 Populate Stats During Training | M | 4.0 |
| 4 | 4.3 PathState | M | None |
| 5 | 4.4 ShapValues Container | S | None |
| 6 | 4.5 TreeExplainer | L | 4.2, 4.3, 4.4 |
| 7 | 4.6 LinearExplainer | M | 4.4 |
| 8 | 4.7 Model Integration | S | 4.5, 4.6, Epic 2 |
| 9 | 4.8 Python Bindings | M | 4.7, Epic 3 |

### Coordination Notes

**With Epic 1 (Storage)**: Story 4.0 adds optional fields to Tree struct. Epic 1 can serialize these fields (if present). Complete Story 4.0 before or alongside Epic 1, Story 1.2 (Payloads).

**Parallel Work**: Stories 4.3, 4.4 can proceed in parallel with 4.0, 4.1, 4.2 since they have no dependencies.

### Deferred to Post-MVP

| Item | Reason |
| ---- | ------ |
| SHAP interaction values | O(M²) complexity, advanced use case |
| GPU TreeSHAP | Optimization |
| Approximate SHAP | Complexity |
| Permutation importance | Different algorithm |
| Interventional SHAP | Higher computational cost |

### Verification Checklist

After all stories complete:

- [ ] `cargo test` — all tests pass
- [ ] Feature importance works for GBDT (all types)
- [ ] Feature importance works for GBLinear
- [ ] SHAP verification passes: sum + base = prediction
- [ ] Python bindings work correctly
- [ ] Works with shap library plotting (manual verification)
- [ ] Error messages are clear for missing stats

**Epic 4 Complete When**:
- [ ] `model.feature_importance(method="gain")` returns importance scores
- [ ] `model.shap_values(X)` returns per-sample, per-feature SHAP values
- [ ] SHAP values sum to `(prediction - expected_value)` for all test samples
- [ ] `TreeExplainer` predictions match `model.predict()` exactly
- [ ] Python bindings expose all explainability methods
- [ ] SHAP verification tests pass with reference data

### Test File Locations

| Test Type | Location |
|-----------|----------|
| PathState unit tests | Inline in `src/explainability/shap/path.rs` |
| ShapValues unit tests | Inline in `src/explainability/shap/values.rs` |
| TreeExplainer tests | `tests/explainability/tree_shap.rs` |
| LinearExplainer tests | `tests/explainability/linear_shap.rs` |
| Feature importance tests | `tests/explainability/importance.rs` |
| SHAP verification (sum property) | `tests/explainability/shap_verification.rs` |
| Python explainability tests | `boosters-python/tests/test_explainability.py` |

### Reference Values

For SHAP testing, generate reference values using the `shap` library on equivalent XGBoost models. Store in `tests/test-cases/shap/` as JSON files.

---

**Previous Epic**: [Model API and Python](06-model-api-and-python.md)

---

**Document Status**: Ready for Implementation  
**Reviewed By**: PO, Architect, Senior Engineer, QA Engineer
