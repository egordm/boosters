# Backlog: Linear Trees Comparison (LightGBM)

**RFCs**: RFC-0015 (Linear Leaves)  
**Created**: 2025-12-18  
**Status**: Complete (Stories 1-4), Story 5 is optional follow-up  
**Depends on**: Linear Trees Implementation (Epic 2 from linear-trees-prediction-refactor.md)

---

## Overview

This epic covers comparing booste-rs linear trees against LightGBM's implementation
for both quality and performance. The goal is to verify we achieve parity with
LightGBM and document the results.

**Dependency Graph:**

```text
Story 1 (LightGBM Loader) ──┬──→ Story 3 (Performance Benchmarks)
                            │                    ↓
Story 2 (Quality Benchmark) ┴──→ Story 4 (Documentation)
```

**Known Limitation**: The lightgbm3 Rust crate crashes (SIGSEGV) when training
with `linear_tree=True`. Python LightGBM works fine. Workarounds:

- Use Python LightGBM to train models, load via our LightGBM loader
- Skip LightGBM training comparison for linear trees
- Try alternative crates (lgbm, lightgbm-rust)

---

## Story 1: LightGBM Linear Tree Loader ✅

**Status**: Complete

Load LightGBM linear tree models and verify 1:1 prediction matching.

**Tasks:**

- [x] Add linear tree fields to `LgbTree` struct (`leaf_const`, `num_features_per_leaf`, `leaf_features`, `leaf_coeff`)
- [x] Implement grouped array parsing (`parse_grouped_int_array`, `parse_grouped_double_array`)
- [x] Update `convert_tree()` to populate `LeafCoefficients`
- [x] Fix `Forest::predict_row()` to handle linear leaves (compute linear prediction)
- [x] Generate test case with Python LightGBM linear tree model
- [x] Verify predictions match LightGBM within tolerance

**Acceptance Criteria:**

- Load LightGBM models trained with `linear_tree=True`
- Predictions match LightGBM output exactly (< 1e-3 tolerance)

**Tests:**

- `convert_linear_tree`: Parser correctly extracts linear tree fields
- `predict_linear_tree`: 20-row prediction matches LightGBM output

---

## Story 2: Quality Benchmark Integration ✅

**Status**: Complete

Add linear trees to quality_benchmark.rs to compare booste-rs quality metrics.

**Tasks:**

- [x] Add `linear_leaves: bool` to `BenchmarkConfig`
- [x] Add linear tree configs: `regression_linear_small`, `regression_linear_medium`
- [x] Add real-world linear config: `california_housing_linear`
- [x] Update `train_boosters()` to use `LinearLeafConfig` when enabled
- [x] Add `linear_tree=true` to LightGBM params (but skip due to crash)
- [x] Update config table to show "Linear" column
- [x] Skip LightGBM and XGBoost for linear tree configs

**Acceptance Criteria:**

- Quality benchmark runs with linear tree configurations
- booste-rs linear trees produce competitive RMSE on synthetic linear data
- Report clearly indicates which libraries support linear trees

**Tests:**

- Run `quality_benchmark` with synthetic + linear configs
- Verify output markdown is correctly formatted

**Notes:**

- LightGBM training comparison skipped due to lightgbm3 crate crash
- XGBoost doesn't support linear trees, skipped by design

---

## Story 3: Performance Benchmarks ✅

**Status**: Complete

Create training and inference benchmarks comparing booste-rs vs LightGBM.

**Tasks:**

- [x] Create inference benchmark: load LightGBM linear tree model, predict
- [x] Create training benchmark: compare training time with linear trees enabled
- [x] Measure overhead of linear tree fitting vs standard GBDT
- [x] Document benchmark methodology and results

**Results:**

| Metric | booste-rs | LightGBM | Status |
|--------|-----------|----------|--------|
| Training overhead | +10.4% | +11.9% | ✅ On par |
| Prediction overhead | +5.4x | +1.75x | ⚠️ Needs optimization |
| Prediction throughput | 324 Kelem/s | 254 Kelem/s | ✅ 1.3x faster |

**Benchmark Report**: [2025-12-18-6958442-linear-trees-performance.md](../benchmarks/2025-12-18-6958442-linear-trees-performance.md)

**Follow-up**: Story 5 created for prediction optimization.

---

## Story 4: Documentation Updates ✅

**Status**: Complete

Review and update RFC-0015 and research docs, create comprehensive benchmark report.

**Tasks:**

- [x] Review RFC-0015 for accuracy against implementation
- [x] Update research/gbdt/training/linear-trees.md with implementation insights
- [x] Create comprehensive benchmark report in docs/benchmarks/
- [x] Document LightGBM comparison results (quality + performance)
- [x] Add any discovered design decisions to RFC changelog

**Completed:**

- RFC-0015 status updated to "Implemented", changelog added
- Research doc updated with performance findings and LightGBM loader details
- Benchmark report: [2025-12-18-6958442-linear-trees-performance.md](../benchmarks/2025-12-18-6958442-linear-trees-performance.md)

---

## Story 5: Linear Tree Prediction Optimization 🆕

**Status**: Not Started  
**Priority**: High  
**Depends on**: Story 3 (benchmark baseline established)

Optimize linear tree prediction to reduce overhead from 5.4x to <2x.

**Problem Statement:**

Current implementation falls back to per-row traversal for linear trees,
losing the benefits of block-optimized traversal. LightGBM achieves 1.75x
overhead while booste-rs has 5.4x.

**Tasks:**

- [ ] Profile current prediction path to identify bottlenecks
- [ ] Implement block-level leaf index collection
- [ ] Batch linear coefficient computation per leaf
- [ ] Vectorize `intercept + Σ(coef × feature)` using SIMD or unrolled loops
- [ ] Benchmark and compare with baseline

**Acceptance Criteria:**

- Prediction overhead reduced to <2x (vs standard trees)
- No regression in prediction accuracy
- Maintains current training quality

**Design Notes:**

```text
Current:
  for row in block:
    leaf_idx = traverse(row)
    value = compute_linear(leaf_idx, row)  // per-row linear math

Optimized:
  leaf_indices = traverse_block(block)  // batch traversal
  for unique_leaf in leaf_indices:
    batch_compute_linear(leaf, rows_with_leaf)  // vectorized
```

---

## Follow-up Stories (If Needed)

### Story 5: Try Alternative LightGBM Crates (Stretch)

If lgbm crate works with linear trees, migrate benchmarks to use it.

**Tasks:**

- [ ] Test lgbm crate with `linear_tree=True`
- [ ] Update Cargo.toml with lgbm dependency
- [ ] Refactor benchmark code to use new crate
- [ ] Re-run benchmarks with actual LightGBM comparison

### Story 6: Quality Improvements (If Needed)

If booste-rs linear trees are significantly worse than LightGBM:

**Tasks:**

- [ ] Investigate coefficient fitting differences
- [ ] Compare solver approaches (CD vs direct solve)
- [ ] Tune regularization parameters
- [ ] Re-benchmark after fixes

---

## Quality Gates

| Gate | Criteria | When |
|------|----------|------|
| Loader Accuracy | Predictions match LightGBM < 1e-3 | Story 1 |
| Quality Parity | RMSE within 10% of LightGBM | Story 2 |
| Benchmark Coverage | All configs run successfully | Story 3 |
| Documentation | RFC + research docs updated | Story 4 |

---

## Notes

### lightgbm3 Crate Crash

The lightgbm3 Rust crate v1.0.8 crashes with SIGSEGV when training with
`linear_tree=True`. This was verified with a minimal reproduction example.
Python LightGBM 4.x works correctly with the same parameters.

**Workarounds implemented:**

1. LightGBM linear tree **loading** works (text format parser)
2. Training comparison uses booste-rs only for linear tree configs
3. Quality benchmark skips LightGBM for linear tree configs

**Future options:**

- Try lgbm crate (0.0.6, updated 4 months ago)
- File issue with lightgbm3 upstream
- Use Python subprocess for training benchmarks
