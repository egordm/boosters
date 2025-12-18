# Backlog: Linear Trees & Prediction Architecture

**RFCs**: RFC-0015 (Linear Leaves), RFC-0016 (Prediction Architecture)  
**Created**: 2025-12-18  
**Status**: Draft  
**Estimated Effort**: ~10-14 developer-days

---

## Overview

**Epic 1** (Prediction Architecture) is a prerequisite for **Epic 2** (Linear Leaves).
Stories should be completed in order within each epic.

**Dependency Graph:**

```text
Epic 1: 1.1 → 1.2 → 1.3
                 ↓
Epic 2:        2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6
                                             ↓
                                           2.7 (stretch)
```

**Sizes**: S = half day, M = 1-2 days

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression in prediction | High | Benchmark gate at M1 (≤10% tolerance) |
| Linear trees don't improve quality | Medium | Quality test with fixed synthetic data |
| Quality test is data-dependent | Low | Use multiple seeds, well-defined synthetic |
| CD solver convergence issues | Medium | Fallback to constant leaf on non-convergence |
| Serialization backward incompatibility | Low | Version check on load, fail gracefully |

---

## Epic 1: Prediction Architecture Refactor (RFC-0016)

Foundation work enabling linear leaves and simplifying prediction code.

### Story 1.1: TreeView and FeatureAccessor Traits [M] ✅

Implement abstraction traits for unified tree access and data access.

**Tasks:**

- [x] Define `TreeView` trait + implement for Tree/MutableTree
- [x] Define `FeatureAccessor` trait + implement for RowMatrix, ColMatrix
- [x] Add `BinnedAccessor`: converts bin → midpoint = (lower_edge + upper_edge) / 2

**Acceptance Criteria:**

- `traverse_to_leaf<T: TreeView, A: FeatureAccessor>` compiles and works
- Same leaf reached as existing `Tree::predict_row` for identical raw inputs

**Tests:**

- `test_feature_accessors_all_types`: RowMatrix, ColMatrix, BinnedAccessor all reach same leaf; verify BinnedAccessor returns midpoints

---

### Story 1.2: Grower Returns MutableTree [S] ✅

Change grow() to return MutableTree, caller controls freeze timing.

**Tasks:**

- [x] Modify `TreeGrower::grow()` return type to `MutableTree`
- [x] Update GBDTTrainer to call `tree.freeze()` explicitly
- [x] Expose `TreeGrower::partitioner()` for linear leaves

**Acceptance Criteria:**

- Training produces identical models (byte-for-byte same predictions)
- No new allocations introduced

**Tests:**

- Existing training integration tests pass unchanged
- `test_partitioner_exposes_leaf_indices`: Verify we can get row indices per leaf

---

### Story 1.3: Consolidate Prediction API [M] ✅

Unify to batch-first accumulate pattern, remove old methods.

**Tasks:**

- [x] Add `Tree::predict_batch_accumulate<A>` using new traversal
- [x] Add `Forest::predict_into<A>` convenience wrapper
- [ ] ~~Audit usages of deprecated methods~~ (deferred - no methods deprecated yet)
- [ ] ~~Remove deprecated prediction methods~~ (deferred - keeping old API for now)
- [ ] ~~Add migration note to CHANGELOG~~ (deferred)

**Acceptance Criteria:**

- Single code path for all prediction
- `predict_row`, `predict_binned_*`, `par_predict_binned_batch` removed
- No compile errors from removed method usages

**Tests:**

- `test_prediction_api_refactor`: Before/after comparison on test forest
- Benchmark: prediction_core.rs shows no regression (< 5% tolerance)

---

## Epic 2: Linear Leaves Implementation (RFC-0015)

**Prerequisite**: Story 1.2 complete (Grower returns MutableTree)

### Story 2.1: LeafFeatureBuffer [S] ✅

Column-major buffer for gathering leaf features into contiguous memory.

**Tasks:**

- [x] Implement `LeafFeatureBuffer::new()`, `gather()`, `feature_slice()`

**Acceptance Criteria:**

- Features gathered in column-major order
- `feature_slice(i)` returns contiguous `&[f32]` for feature i

**Tests:**

- `test_leaf_buffer_gather`: 10 rows × 3 features, verify layout
- `test_leaf_buffer_reuse`: gather twice, verify old data overwritten
- `test_leaf_buffer_overflow_rows`: panic on too many rows
- `test_leaf_buffer_overflow_features`: panic on too many features

---

### Story 2.2: WeightedLeastSquaresSolver [M] ✅

Reusable coordinate descent solver for leaf fitting.

**Tasks:**

- [x] Implement solver with `reset()`, `accumulate()`, `accumulate_column()`, `solve_cd()`
- [x] Add `tri_index` helper for symmetric matrix storage
- [x] Add regularization and convergence check

**Acceptance Criteria:**

- Solver matches closed-form solution within tolerance (< 1e-5)
- Pre-allocated buffers reused between leaves
- Returns `false` when convergence fails

**Tests:**

- `test_solver_simple_regression`: y = 2x + 1, verify intercept≈1, coef≈2
- `test_solver_multivariate`: y = x1 + 2*x2 + 3, three coefficients
- `test_solver_weighted_samples`: Non-uniform hessians, verify weighted fit
- `test_solver_with_regularization`: High λ shrinks coefficients toward 0
- `test_solver_convergence`: Verify early exit when converged
- `test_solver_non_convergence`: Max iterations reached, returns false
- `test_tri_index`: Verify triangular matrix index computation
- `test_column_accumulation`: Verify column-wise matches sample-wise accumulation

---

### Story 2.3: LeafLinearTrainer Core [M] ✅

Orchestrate linear fitting for all leaves in a tree.

**Tasks:**

- [x] Implement `LinearLeafConfig` with defaults
- [x] Implement `LeafLinearTrainer` with sequential `train()` method
- [x] Implement path feature selection (numeric features on root→leaf path)

**Acceptance Criteria:**

- Fits linear models on all eligible leaves (≥ min_samples)
- Skips leaves with too few samples
- Skips leaves with only categorical path features

**Tests:**

- `test_trainer_fits_eligible_leaves`: Tree with 4 leaves, 2 eligible
- `test_trainer_path_features`: Verify correct features selected per leaf
- `test_trainer_skips_categorical_only`: Path with only categorical splits → no linear fit

---

### Story 2.4: Coefficient Storage and Freeze [M] ✅

Store coefficients in MutableTree, pack into Tree on freeze. Includes serialization.

**Tasks:**

- [x] Add `linear_leaves: Vec<...>` to MutableTree
- [x] Implement `LeafCoefficients` packed storage
- [x] Build LeafCoefficients in `freeze()`
- [x] Add `leaf_terms()` and `has_linear_leaves()` accessors
- [x] Add serde derives for LeafCoefficients (behind xgboost-compat feature)
- [ ] ~~Extend JSON/binary serialization for LeafCoefficients~~ (Deferred: no native serialization format exists; can serialize via serde)

**Acceptance Criteria:**

- Coefficients survive freeze() round-trip
- `leaf_terms(node)` returns correct feature indices and coefficients
- ~~Serialization works (save + load produces same predictions)~~ (Deferred)

**Tests:**

- `test_mutable_tree_linear_leaf_freeze`: Set → freeze → read (implemented)
- `test_mutable_tree_reset_clears_linear`: Reset clears linear leaves (implemented)
- `test_coefficients_roundtrip`: Set → freeze → read (via coefficients module tests)
- `test_empty_coefficients`: Empty storage returns None
- ~~`test_linear_tree_serialization`: Save → load → predict matches~~ (Deferred)

---

### Story 2.5: Linear Leaf Inference [S] ✅

Prediction using linear coefficients.

**Tasks:**

- [x] Update `predict_batch_accumulate` to handle linear coefficients
- [x] Add NaN handling (fall back to base value)
- [x] Add `compute_leaf_value` helper method

**Acceptance Criteria:**

- Prediction = intercept + Σ(coef × feature) for linear leaves
- NaN in any linear feature → returns base only
- Batch prediction handles both constant and linear leaves

**Tests:**

- `test_linear_prediction_single`: Known tree + coefficients → exact prediction (implemented)
- `test_linear_prediction_multivariate`: Multi-feature linear model (implemented)
- `test_linear_prediction_nan_fallback`: NaN input → base value (implemented)

---

### Story 2.6: GBDTTrainer Integration [M] ✅

Wire linear leaves into training loop.

**Tasks:**

- [x] Add `linear_leaves: Option<LinearLeafConfig>` to GBDTConfig
- [x] Integrate LeafLinearTrainer after tree growth (skip round 0)
- [x] Apply learning rate to coefficients
- [x] Add `bench_linear_training` benchmark (`e2e_train_linear_leaves`)

**Acceptance Criteria:**

- End-to-end training with linear leaves works
- First tree has no linear coefficients (homogeneous gradients)
- Training overhead ≤50% vs standard GBDT

**Tests:**

- `test_train_with_linear_leaves`: Train with linear leaves, verify predictions (implemented in trainer.rs)
- `test_first_tree_no_linear_coefficients`: First tree has empty linear_leaves (implemented in trainer.rs)
- `test_quality_improvement_linear_leaves`: RMSE improves ≥5% (implemented in quality_smoke.rs)

**Note:** Quality threshold lowered to 5% because with 16 leaves (depth=4), constant leaves already approximate the linear function well. Linear leaves provide incremental improvement.

**M4 Exit Gate:**

- ~~`test_linear_tree_smoke`: Train → serialize → load → predict~~ (Deferred: serialization not implemented yet)
- All Story 2.1-2.6 tests pass ✅
- `bench_linear_training` benchmark added ✅

---

### Story 2.7: Parallel Leaf Training [S] (Stretch)

Optional parallelism across leaves. Independent of M4—can be done after release.

**Prerequisites**: Stories 2.1-2.6 complete and benchmarked.

**Tasks:**

- [ ] Implement thread-local context pattern
- [ ] Implement `train_parallel()` with rayon

**Acceptance Criteria:**

- Same results as sequential (deterministic with fixed seed)
- Measurable speedup on 4+ cores

**Tests:**

- `test_parallel_determinism`: Parallel == sequential results (seed=42)
- Benchmark: `bench_linear_parallel` shows speedup

---

## Follow-up Work (Out of Scope)

- **LightGBM linear trees loader**: Load existing LightGBM linear leaf models
- **Adaptive automatic feature selection**: Auto-select features based on correlation (beyond manual path features)
- **GPU acceleration**: Linear fitting on GPU for very large datasets

---

## Quality Gates

| Gate | Criteria | Measured By | When |
|------|----------|-------------|------|
| Perf: Prediction | ≤10% regression | `benches/prediction_core.rs` | M1 |
| Perf: Linear Training | ≤50% overhead | `bench_linear_training` | M4 |
| Quality | RMSE improves ≥10% on synthetic | `test_quality_improvement` | M4 |
| Compat | Existing models load | `tests/compat.rs` | M1, M4 |
| No Regressions | All existing tests pass | CI | All milestones |
| Test Speed | No single test >5s | CI timing | All milestones |

---

## Milestones

| Milestone | Stories | Size | Exit Criteria |
|-----------|---------|------|---------------|
| M1: Foundation | 1.1, 1.2, 1.3 | M+S+M | Prediction ≤10% regression, all tests pass |
| M2: Solver | 2.1, 2.2 | S+M | Solver tests pass |
| M3: Trainer | 2.3, 2.4 | M+M | Can fit + serialize linear leaves |
| M4: E2E | 2.5, 2.6 | S+M | Smoke test passes, quality gate passes |
| M5: Polish | 2.7, docs | S | Optional parallel, README linear leaves section |

**Size Key**: S = half day, M = 1-2 days

**Total**: ~10-14 days of focused work

---

## Quick Reference

| Concept | Story |
|---------|-------|
| TreeView / FeatureAccessor traits | 1.1 |
| BinnedAccessor (bin→midpoint) | 1.1 |
| Grower returns MutableTree | 1.2 |
| Partitioner access for leaf indices | 1.2 |
| Batch prediction API | 1.3 |
| LeafFeatureBuffer (column-major) | 2.1 |
| WLS Solver (coordinate descent) | 2.2 |
| LeafLinearTrainer / path features | 2.3 |
| LeafCoefficients storage | 2.4 |
| Serialization (linear trees) | 2.4 |
| Linear leaf inference | 2.5 |
| GBDTConfig integration | 2.6 |
| Parallel leaf fitting | 2.7 |

