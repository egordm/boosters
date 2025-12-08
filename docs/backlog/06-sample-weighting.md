# Epic 6: Sample Weighting

**Status**: Not Started  
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

## Story 1: Loss Trait Extension

**Goal**: Add weights parameter to `Loss` trait and implement for all loss functions.

### Tasks

- [ ] 1.1: Add `weights: Option<&[f32]>` parameter to `Loss::compute_gradients()`
- [ ] 1.2: Update `SquaredLoss` to multiply grad/hess by weight
- [ ] 1.3: Update `LogisticLoss` to multiply grad/hess by weight
- [ ] 1.4: Update `SoftmaxLoss` to multiply grad/hess by weight
- [ ] 1.5: Update `PoissonLoss`, `TweedieRegressorLoss` (if present)
- [ ] 1.6: Ensure unweighted path (`None`) has same behavior as before

### Unit Tests

- [ ] 1.T1: `SquaredLoss` unweighted matches current behavior exactly
- [ ] 1.T2: `SquaredLoss` with uniform weights (all 1.0) matches unweighted
- [ ] 1.T3: `SquaredLoss` with weights: `grad[i] = w[i] * (pred - label)`
- [ ] 1.T4: `SquaredLoss` with weights: `hess[i] = w[i] * 1.0`
- [ ] 1.T5: `LogisticLoss` unweighted matches current behavior
- [ ] 1.T6: `LogisticLoss` with uniform weights matches unweighted
- [ ] 1.T7: `LogisticLoss` weighted: `grad[i] = w[i] * (sigmoid(pred) - label)`
- [ ] 1.T8: `LogisticLoss` weighted: `hess[i] = w[i] * p * (1 - p)`
- [ ] 1.T9: `SoftmaxLoss` unweighted matches current behavior
- [ ] 1.T10: `SoftmaxLoss` with uniform weights matches unweighted
- [ ] 1.T11: Zero weight produces zero gradient and hessian
- [ ] 1.T12: Negative weight produces negative gradient (with warning)

### Performance Tests

| Test | Setup | Before | After | Acceptable |
|------|-------|--------|-------|------------|
| 1.P1 | `compute_gradients` 1M samples, no weights | Baseline | ≤ Baseline | < 1% slower |
| 1.P2 | `compute_gradients` 1M samples, with weights | N/A | ≤ 1.2x Baseline | Expected overhead |

---

## Story 2: Trainer API Integration

**Goal**: Add weights parameter to trainer and validate at API boundary.

### Tasks

- [ ] 2.1: Add `weights: Option<&[f32]>` to `GBTreeTrainer::train()`
- [ ] 2.2: Validate `weights.len() == labels.len()` when provided
- [ ] 2.3: Add `TrainError::WeightLengthMismatch` error variant
- [ ] 2.4: Pass weights through to `loss.compute_gradients()` in training loop
- [ ] 2.5: Update `train_multiclass()` to pass weights for each class
- [ ] 2.6: Warn if any weight is negative (log warning, don't fail)

### Unit Tests

- [ ] 2.T1: Training with `weights: None` produces same model as before
- [ ] 2.T2: `WeightLengthMismatch` error when `weights.len() != labels.len()`
- [ ] 2.T3: Weights correctly passed to loss in each round
- [ ] 2.T4: Multiclass training uses same weights for all classes

### Integration Tests

- [ ] 2.I1: Train regression with uniform weights, predictions match unweighted
- [ ] 2.I2: Train classification with uniform weights, predictions match unweighted
- [ ] 2.I3: Error message is clear for weight length mismatch

---

## Story 3: Weighted Base Score

**Goal**: Use weighted labels for base score initialization.

**Note**: RFC-0024 already implemented `init_base_score` with weights support.
This story verifies and tests it.

### Tasks

- [ ] 3.1: Verify `SquaredLoss::init_base_score` uses weighted mean
- [ ] 3.2: Verify `LogisticLoss::init_base_score` uses weighted log-odds
- [ ] 3.3: Verify `SoftmaxLoss::init_base_score` uses weighted class frequencies
- [ ] 3.4: Add tests if missing

### Unit Tests

- [ ] 3.T1: `SquaredLoss` weighted base score = weighted mean of labels
- [ ] 3.T2: `LogisticLoss` weighted base score = log(weighted_pos / weighted_neg)
- [ ] 3.T3: `SoftmaxLoss` weighted base scores reflect weighted class proportions
- [ ] 3.T4: Uniform weights produce same base score as unweighted

### Integration Tests

- [ ] 3.I1: Weighted base score matches XGBoost's `base_score` for same weights

---

## Story 4: Weighted Evaluation Metrics

**Goal**: Metrics support optional weights for proper evaluation.

### Tasks

- [ ] 4.1: Add `weights: Option<&[f32]>` to `rmse()` function
- [ ] 4.2: Add `weights: Option<&[f32]>` to `mae()` function
- [ ] 4.3: Add `weights: Option<&[f32]>` to `logloss()` function
- [ ] 4.4: Add `weights: Option<&[f32]>` to `auc()` function (if applicable)
- [ ] 4.5: Add `weights: Option<&[f32]>` to accuracy/multiclass metrics
- [ ] 4.6: Use weighted metrics in training callbacks/early stopping

### Unit Tests

- [ ] 4.T1: `rmse()` with no weights matches current behavior
- [ ] 4.T2: `rmse()` with uniform weights matches unweighted
- [ ] 4.T3: `rmse()` weighted formula: `sqrt(sum(w * (p-l)²) / sum(w))`
- [ ] 4.T4: `logloss()` with no weights matches current behavior
- [ ] 4.T5: `logloss()` with uniform weights matches unweighted
- [ ] 4.T6: `logloss()` weighted formula correct
- [ ] 4.T7: Zero-weight samples excluded from metric computation

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
