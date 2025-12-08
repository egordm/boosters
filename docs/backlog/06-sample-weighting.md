# Epic 6: Sample Weighting

**Status**: In Progress  
**Priority**: Medium  
**Depends on**: Epic 3 (GBTree Training Phase 1)  
**RFC**: [RFC-0026](../design/rfcs/0026-sample-weighting.md)

## Overview

Add per-instance sample weights to training, enabling class imbalance handling,
importance sampling, survey data analysis, and curriculum learning. This is a
standard feature in both XGBoost and LightGBM.

**Key insight**: Weights are applied at gradient computation time (multiply grad/hess
by weight), so histogram building, split finding, and tree growing remain unchanged.

---

## Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| **Correctness** | Weighted gradients match formula | Exact |
| **Compatibility** | Predictions match XGBoost with same weights | Correlation > 0.99 |
| **Performance** | No regression for unweighted training | < 1% overhead |
| **Quality** | Weighted model improves imbalanced data | Measurable improvement |

---

## Testing Strategy Overview

### Test Categories

| Category | Purpose | When to Run | Failure Action |
|----------|---------|-------------|----------------|
| **Unit tests** | Verify weighted gradient computation | Every commit | Fix immediately |
| **Integration tests** | Compare weighted training vs XGBoost | PR merge | Fix before merge |
| **Quality tests** | Model accuracy on imbalanced datasets | Story completion | Investigate |
| **Performance tests** | Ensure no overhead for unweighted | Story completion | Optimize if >2% regression |

### Reference Data

Integration tests use pre-computed outputs from XGBoost:

- Generate weighted training baselines with `tools/data_generation/generate_weighted_training.py`
- Store in `tests/test-cases/xgboost/gbtree/training/weighted/`
- Include: predictions, base scores, final metrics

---

## Story 1: Loss Trait Extension ✅

**Goal**: Add weights parameter to `Loss` trait and implement for all loss functions.

**Status**: Complete

### Tasks

- [x] 1.1: Add `weights: Option<&[f32]>` parameter to `Loss::compute_gradients()`
- [x] 1.2: Update `SquaredLoss` to multiply grad/hess by weight
- [x] 1.3: Update `LogisticLoss` to multiply grad/hess by weight
- [x] 1.4: Update `SoftmaxLoss` to multiply grad/hess by weight
- [x] 1.5: Update `QuantileLoss`, `PseudoHuberLoss`, `HingeLoss`
- [x] 1.6: Ensure unweighted path (`None`) has same behavior as before

### Unit Tests

- [x] All weighted gradient tests passing (55 loss tests total)

---

## Story 2: Trainer API Integration ✅

**Goal**: Add weights parameter to trainer and validate at API boundary.

**Status**: Complete

### Tasks

- [x] 2.1: Add `weights: Option<&[f32]>` to `GBTreeTrainer::train()`
- [x] 2.2: Validate `weights.len() == labels.len()` when provided
- [x] 2.3: Add `TrainError::WeightLengthMismatch` error variant
- [x] 2.4: Pass weights through to `loss.compute_gradients()` in training loop
- [x] 2.5: Update `train_quantized()` and `train_internal()` to accept weights
- [x] 2.6: Updated all callers (benches, examples, tests) to pass `None`

### Unit Tests

- [x] `test_train_with_weights` - Training with weights works
- [x] `test_train_weights_length_mismatch` - Error for mismatched lengths
- [x] All trainer tests passing (9 tests)

---

## Story 3: Weighted Base Score ✅

**Goal**: Use weighted labels for base score initialization.

**Status**: Complete (RFC-0024 already implemented with weights)

### Tasks

- [x] 3.1: Verified `SquaredLoss::init_base_score` uses weighted mean
- [x] 3.2: Verified `LogisticLoss::init_base_score` uses weighted log-odds
- [x] 3.3: Verified `SoftmaxLoss::init_base_score` uses weighted class frequencies
- [x] 3.4: Added unit tests for weighted base score

### Unit Tests

- [x] `squared_loss_weighted_base_score` - Weighted mean matches expected
- [x] `logistic_loss_weighted_base_score` - Weighted log-odds correct
- [x] `softmax_loss_weighted_base_score` - Weighted class proportions correct
- [x] `hinge_loss_weighted_base_score` - Weighted margin initial score

---

## Story 4: Weighted Evaluation Metrics ✅

**Goal**: Metrics support optional weights for proper evaluation.

**Status**: Complete

### Tasks

- [x] 4.1: Simplify `Metric` trait to single `evaluate(preds, labels, weights: Option, n_outputs)` method
- [x] 4.2: Remove `SimpleMetric` trait (unnecessary abstraction)
- [x] 4.3: Implement weighted `evaluate` for `Rmse`
- [x] 4.4: Implement weighted `evaluate` for `Mae`
- [x] 4.5: Implement weighted `evaluate` for `LogLoss`
- [x] 4.6: Implement weighted `evaluate` for `Accuracy`
- [x] 4.7: Implement weighted `evaluate` for `Auc` (O(n log n) sorting algorithm)
- [x] 4.8: Implement weighted `evaluate` for `Mape`
- [x] 4.9: Implement weighted `evaluate` for `MulticlassAccuracy`
- [x] 4.10: Implement weighted `evaluate` for `MulticlassLogLoss`
- [x] 4.11: Implement weighted `evaluate` for `QuantileMetric`
- [x] 4.12: Update `EvalMetric` enum to pass weights to implementations
- [x] 4.13: Add `weights` field to `EvalSet` with `with_weights()` constructor
- [x] 4.14: Add `weights` field to `QuantizedEvalSet` with `with_weights()` constructor
- [x] 4.15: Update `EarlyStopping::should_stop()` to accept `weights` and `n_outputs`
- [x] 4.16: Split `metric.rs` into `metric/mod.rs`, `metric/regression.rs`, `metric/classification.rs`

### Unit Tests

- [x] `weighted_rmse_uniform_weights_equals_unweighted`
- [x] `weighted_rmse_emphasizes_high_weight_samples`
- [x] `weighted_mae_emphasizes_high_weight_samples`
- [x] `weighted_logloss_emphasizes_high_weight_samples`
- [x] `weighted_accuracy_emphasizes_correct_samples`
- [x] `weighted_auc_emphasizes_pairs`
- [x] `weighted_mape_emphasizes_high_weight_samples`
- [x] `weighted_multiclass_accuracy`
- [x] `weighted_mlogloss`
- [x] `weighted_quantile_loss`
- [x] `auc_matches_naive_implementation` (verifies O(n log n) algorithm)
- [x] All 40 metric tests passing

---

## Story 5: XGBoost Compatibility Tests

**Goal**: Verify weighted training matches XGBoost predictions.

### Tasks

- [ ] 5.1: Create `generate_weighted_training.py` test data generator
- [ ] 5.2: Generate weighted regression test case
- [ ] 5.3: Generate weighted binary classification test case
- [ ] 5.4: Generate weighted multiclass test case
- [ ] 5.5: Generate class-imbalance test case (high weights on minority)
- [ ] 5.6: Store baselines in `tests/test-cases/xgboost/gbtree/training/weighted/`
- [ ] 5.7: Create `tests/training_weighted.rs` integration tests

### Test Cases

| Name | Type | Rows | Weight Distribution |
|------|------|------|---------------------|
| `weighted_regression` | Regression | 1000 | Random [0.5, 2.0] |
| `weighted_binary` | Binary | 1000 | Random [0.5, 2.0] |
| `weighted_multiclass` | 3-class | 1200 | Random [0.5, 2.0] |
| `class_imbalance` | Binary | 1000 | Minority 10x weight |
| `zero_weights` | Regression | 1000 | 10% zeros |

### Integration Tests

- [ ] 5.I1: Weighted regression predictions correlate > 0.99 with XGBoost
- [ ] 5.I2: Weighted binary predictions correlate > 0.99 with XGBoost
- [ ] 5.I3: Weighted multiclass predictions correlate > 0.99 with XGBoost
- [ ] 5.I4: Class imbalance weights improve minority class recall

---

## Story 6: Quality Validation

**Goal**: Verify weighted training improves model quality on appropriate tasks.

### Quality Tests

| Test | Dataset | Weights | Expected Outcome |
|------|---------|---------|------------------|
| 6.Q1 | Imbalanced binary (10:1 ratio) | 10x on minority | Higher minority recall than unweighted |
| 6.Q2 | Imbalanced binary (10:1 ratio) | None | Baseline minority recall |
| 6.Q3 | Survey data simulation | Population weights | Better population-level estimates |
| 6.Q4 | Importance sampling | Higher on recent samples | Better recent performance |

### Metrics to Track

- **Imbalanced classification**: Recall on minority class, F1 score, balanced accuracy
- **Overall**: Compare weighted RMSE/AUC vs unweighted

### Tasks

- [ ] 6.1: Create imbalanced classification test dataset (10:1 class ratio)
- [ ] 6.2: Train unweighted model, measure minority recall
- [ ] 6.3: Train with minority weighted 10x, measure minority recall
- [ ] 6.4: Document improvement in quality test results

---

## Story 7: Documentation & Examples

**Goal**: Document weighted training API and provide examples.

### Tasks

- [ ] 7.1: Add rustdoc for `weights` parameter in trainer
- [ ] 7.2: Add rustdoc for `weights` parameter in loss functions
- [ ] 7.3: Add rustdoc for weighted metrics
- [ ] 7.4: Update `examples/train_classification.rs` with weighted example
- [ ] 7.5: Create `examples/train_imbalanced.rs` showing class weights
- [ ] 7.6: Document weight normalization behavior (none, match XGBoost)

---

## Dependencies

```
Story 1 (Loss Trait Extension)
    ↓
Story 2 (Trainer API) ──────────────────────────┐
    ↓                                           │
Story 3 (Weighted Base Score)                   │
    ↓                                           │
Story 4 (Weighted Metrics)                      │
    ↓                                           ├──► Epic Complete
Story 5 (XGBoost Compatibility) ◄───────────────┤
    ↓                                           │
Story 6 (Quality Validation)                    │
    ↓                                           │
Story 7 (Documentation) ◄───────────────────────┘
```

---

## Definition of Done

- [ ] All unit tests passing
- [ ] All integration tests passing (correlation > 0.99 with XGBoost)
- [ ] Quality tests show expected improvements for imbalanced data
- [ ] No performance regression for unweighted training (< 1%)
- [ ] Rustdoc complete for public APIs
- [ ] Example code working
- [ ] No compiler warnings
- [ ] `cargo clippy` clean

---

## Open Questions (from RFC-0026)

1. **Weighted quantile computation for bin cuts?**
   - Currently bin cuts use uniform quantiles
   - Weighted quantiles would better represent weighted distribution
   - Deferred to future work (minor impact on accuracy)

---

## Future Work

- [ ] Weighted quantile computation for bin cuts
- [ ] `scale_pos_weight` parameter (class imbalance shorthand)
- [ ] Per-class weights for multiclass (separate from instance weights)
- [ ] Dynamic weight updates (for boosting variants like AdaBoost)

---

## References

- [RFC-0026: Sample Weighting](../design/rfcs/0026-sample-weighting.md)
- [XGBoost weight parameter](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.DMatrix)
- [LightGBM weight parameter](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Dataset.html)
