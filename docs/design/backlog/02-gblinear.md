# Epic: GBLinear Support

**Status**: 🔄 Active (feature parity work)
**Priority**: High  
**RFCs**: [0008](../rfcs/0008-gblinear-inference.md), [0009](../rfcs/0009-gblinear-training.md)

Add support for XGBoost's linear booster — both inference and training.
Validates training infrastructure before GBTree training.

---

## Completed Stories

### Story 1: GBLinear Inference ✓

- [x] 1.1 `LinearModel` struct with `Box<[f32]>` weight storage
- [x] 1.2 Weight indexing: `weights[feature * num_groups + group]`
- [x] 1.3 `predict_row()` and `predict_batch()` methods
- [x] 1.4 `par_predict_batch()` with Rayon
- [x] 1.5 `Booster::Linear` variant in model enum
- [x] 1.6 XGBoost JSON parser for `gblinear` section
- [x] 1.7 Integration tests vs Python XGBoost

### Story 2: Training Infrastructure ✓

- [x] 2.1 `GradientPair` struct (grad, hess)
- [x] 2.2 `Loss` trait — compute gradients from predictions + labels
- [x] 2.3 Common losses: squared error, logistic, softmax
- [x] 2.4 `Metric` trait — evaluate model quality
- [x] 2.5 Common metrics: RMSE, MAE, logloss, AUC, accuracy
- [x] 2.6 `EarlyStopping` callback
- [x] 2.7 `TrainingLogger` with verbosity levels

### Story 3: GBLinear Training ✓

- [x] 3.1 `CSCMatrix` — column-sparse format for efficient column access
- [x] 3.2 `CSCMatrix::from_dense()` and column iteration
- [x] 3.3 Coordinate descent update with elastic net regularization
- [x] 3.4 Parallel updater — all features with stale gradients (default)
- [x] 3.5 Sequential updater — features in order with stale gradients
- [x] 3.6 `CyclicSelector` and `ShuffleSelector` for feature order
- [x] 3.7 `LinearTrainer` high-level API
- [x] 3.8 Integration tests comparing to XGBoost

### Story 4: Matrix Layout Refactor ✓

- [x] 4.1 Add `Layout` trait with `RowMajor` and `ColMajor` implementations
- [x] 4.2 Refactor `DenseMatrix` to `DenseMatrix<T, L: Layout = RowMajor>`
- [x] 4.3-4.8 Full layout support with conversions and iterators

### Story 5: Training Validation ✓

- [x] 5.1-5.5 Full validation vs XGBoost (weight correlation > 0.9, good test RMSE)

### Story 6: Benchmarks & Optimization ✓

- [x] 6.1-6.6 Performance validated and documented

---

## Active Stories (Feature Parity)

### Story 7: Fix Multiclass Training 🔴 HIGH

**Goal**: Multiclass classification currently broken — all groups get identical gradients.

**Problem**: In `LinearTrainer::compute_gradients()`, we use the same gradient
for all output groups instead of per-class softmax gradients.

- [ ] 7.1 Update `compute_gradients` to handle multiclass properly
- [ ] 7.2 Use `SoftmaxLoss::compute_multiclass_gradient()` for each sample
- [ ] 7.3 Store gradients per (sample, class) pair
- [ ] 7.4 Update each group's weights with group-specific gradients
- [ ] 7.5 Enable `train_multiclass_classification` test
- [ ] 7.6 Validate vs XGBoost multiclass

---

### Story 8: Quantile Regression 🟡 MEDIUM

**Goal**: Add quantile loss for uncertainty quantification.

Pinball loss: `L = α(y-ŷ)⁺ + (1-α)(ŷ-y)⁺`

- [ ] 8.1 Implement `QuantileLoss` with configurable α
- [ ] 8.2 Gradient: `grad = (1-α) if pred >= label else -α`
- [ ] 8.3 Add Python test case generation for quantile regression
- [ ] 8.4 Integration test vs XGBoost `reg:quantileerror`
- [ ] 8.5 Document multi-quantile training (use num_groups = num_quantiles)

---

### Story 9: Additional Loss Functions 🟢 LOW

**Goal**: Add commonly used loss functions for feature parity.

- [ ] 9.1 `HuberLoss` — robust regression (grad clipped for large residuals)
- [ ] 9.2 `HingeLoss` — SVM-style binary classification
- [ ] 9.3 `PseudoHuberLoss` — smooth approximation of Huber
- [ ] 9.4 Integration tests for each

---

### Story 10: Additional Feature Selectors 🟢 LOW

**Goal**: XGBoost-compatible feature selection strategies.

- [ ] 10.1 `GreedySelector` — select feature with largest gradient magnitude
- [ ] 10.2 `ThriftySelector` — approximate greedy (sort by magnitude, iterate)
- [ ] 10.3 `RandomSelector` — with replacement
- [ ] 10.4 Benchmark feature selector impact

---

## Feature Parity Checklist

### Loss Functions

| Objective | XGBoost | booste-rs | Story |
|-----------|---------|-----------|-------|
| `reg:squarederror` | ✅ | ✅ | Done |
| `reg:quantileerror` | ✅ | ❌ | 8 |
| `reg:pseudohubererror` | ✅ | ❌ | 9 |
| `binary:logistic` | ✅ | ✅ | Done |
| `binary:hinge` | ✅ | ❌ | 9 |
| `multi:softmax` | ✅ | ⚠️ Broken | 7 |

### Feature Selectors

| Selector | XGBoost | booste-rs | Story |
|----------|---------|-----------|-------|
| Cyclic | ✅ | ✅ | Done |
| Shuffle | ✅ | ✅ | Done |
| Greedy | ✅ | ❌ | 10 |
| Thrifty | ✅ | ❌ | 10 |
| Random | ✅ | ❌ | 10 |

---

## Success Criteria

1. ✅ Load XGBoost GBLinear JSON models and predict correctly
2. ✅ Train models matching Python XGBoost quality (metrics within 5%)
3. ✅ Training performance equal to or faster than XGBoost
4. ✅ Training infrastructure (losses, metrics, callbacks) is reusable
5. ✅ Early stopping and logging work correctly
6. ✅ Trained model predictions correlate highly with XGBoost predictions
7. ⬜ Multiclass classification works correctly
8. ⬜ Quantile regression supported
