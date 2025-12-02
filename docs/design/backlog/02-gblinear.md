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

### Story 7: Fix Multiclass Training ✓

- [x] 7.1 Add `MulticlassLoss` trait for proper per-class gradient computation
- [x] 7.2 Implement `MulticlassLoss` for `SoftmaxLoss`
- [x] 7.3 Add `gradient_stride` to updater for strided gradient indexing
- [x] 7.4 Add `train_multiclass()` method using `MulticlassLoss`
- [x] 7.5 Store K gradients per sample (sample × class layout)
- [x] 7.6 Enable `train_multiclass_classification` integration test

### Story 8: Quantile Regression ✓

- [x] 8.1 Implement `QuantileLoss` with configurable α (0-1)
- [x] 8.2 Gradient: `(1-α)` if pred >= label else `-α`; hessian = 1
- [x] 8.3 Generate XGBoost quantile test data (α=0.1, 0.5, 0.9)
- [x] 8.4 Integration tests validating quantile behavior
- [x] 8.5 Test that different quantiles produce different predictions

---

## Active Stories (Feature Parity)

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

### Story 11: Multi-Quantile Regression ✅ COMPLETE

**Goal**: Train multiple quantiles simultaneously (like XGBoost's `quantile_alpha` array).

- [x] 11.1 `QuantileLoss::multi(&[f32])` for multiple quantiles (e.g., `[0.1, 0.5, 0.9]`)
- [x] 11.2 Use `num_groups = num_quantiles` to leverage existing multi-output infra
- [x] 11.3 Per-quantile gradient computation (each output gets its own α)
- [x] 11.4 Generate XGBoost multi-quantile test data
- [x] 11.5 Integration tests — 3 quantiles in one model vs 3 separate models

**Results**:
- Multi-quantile model produces high correlation with XGBoost (0.94-0.99)
- Quantiles are correctly ordered (q0.1 < q0.5 < q0.9) for 100% of samples
- Multi-quantile model is identical to training 3 separate models (correlation = 1.0)
- This is expected: quantile gradients are independent (unlike softmax)

**Note**: XGBoost supports this via `reg:quantileerror` with `quantile_alpha=[...]`.
Our multi-output training infrastructure already handles `num_groups > 1`, so this
was a natural extension of Story 8.

**API**:

- `QuantileLoss::new(0.5)` — single-quantile (implements both `Loss` and `MulticlassLoss`)
- `QuantileLoss::multi(&[0.1, 0.5, 0.9])` — multi-quantile (same type, uses `MulticlassLoss`)

---

### Story 12: Gradient Batch Optimization ✅ COMPLETE (No Change Needed)

**Goal**: Vectorized gradient computation for SIMD potential.

- [x] 12.1 ~~Add `gradient_batch` method~~ — Not needed, see results
- [x] 12.2 ~~Implement batch gradient~~ — Default impl already optimal
- [x] 12.3 ~~Implement for `MulticlassLoss`~~ — Already implemented
- [x] 12.4 Benchmark single vs batch gradient computation
- [ ] 12.5 Explore SIMD intrinsics for exp/log (future, softmax-specific)
- [x] 12.6 Document performance findings in `docs/benchmarks/`

**Results** (see [2025-11-29-gradient-batch.md](../benchmarks/2025-11-29-gradient-batch.md)):

| Loss | Best Method | Throughput |
|------|-------------|------------|
| SquaredLoss | gradient_buffer (default) | 5.6 Gelem/s |
| LogisticLoss | per_sample ≈ gradient_buffer | 422 Melem/s |
| QuantileLoss | per_sample | 1.0 Gelem/s |
| Softmax | All ~equal | 203 Melem/s |

**Key Finding**: LLVM auto-vectorization makes explicit batch methods unnecessary.
The current `Loss::compute_gradient` + default `gradient_buffer` is already optimal.

**Hypothesis Rejected**: Explicit batch/vectorized methods are NOT faster.
The compiler already does this well with `#[inline]` functions.

---

### Story 13: SoA Gradient Storage ✅ COMPLETE

**Goal**: Replace `Vec<GradientPair>` with Structure-of-Arrays layout.

**Hypothesis**: Separate `grads: Vec<f32>` and `hess: Vec<f32>` arrays will be:

1. **Faster** — better cache utilization, auto-vectorization friendly
2. **Cleaner code** — no `gradient_stride` hacks for multiclass/multi-quantile
3. **More ergonomic** — natural `[n_samples, n_outputs]` shape for multi-output

**Tasks**:

- [x] 13.1 Create `GradientBuffer` struct with `grads: Vec<f32>`, `hess: Vec<f32>`
- [x] 13.2 Add shape info: `n_samples`, `n_outputs` (1 for regression, K for multiclass)
- [x] 13.3 Indexing: `grads[sample * n_outputs + output]` (row-major per sample)
- [x] 13.4 Refactor `Loss::compute_gradient` to write to slices
- [x] 13.5 Refactor `MulticlassLoss` to use same buffer (no separate trait needed?)
- [x] 13.6 Update `LinearUpdater` to use SoA gradients
- [x] 13.7 Remove `gradient_stride` parameter — shape is in buffer

**Benchmarks** (before/after):

- [x] 13.8 Gradient computation throughput (single-output regression)
- [x] 13.9 Gradient computation throughput (multiclass K=10)
- [x] 13.10 Full training loop (regression, binary, multiclass)
- [x] 13.11 Document findings in `docs/benchmarks/gradient-soa.md`

**Results**:

- ✅ Performance equal (AoS and SoA identical: ~26.5 Melem/s single-output, ~4.4 Melem/s 5-class)
- ✅ Code complexity reduced (no stride parameters, unified multi-output handling)
- ✅ Cleaner API for multiclass and multi-quantile

**Findings**: SoA provides **code quality benefits** (cleaner API, no stride hacks) without
performance penalty. Performance is memory-bound on coordinate descent regardless of layout.
See `docs/benchmarks/2025-11-29-gradient-soa.md` for detailed analysis.

---

## Feature Parity Checklist

### Loss Functions

| Objective | XGBoost | booste-rs | Story |
|-----------|---------|-----------|-------|
| `reg:squarederror` | ✅ | ✅ | Done |
| `reg:quantileerror` | ✅ | ✅ | Done (8) |
| `reg:pseudohubererror` | ✅ | ❌ | 9 |
| `binary:logistic` | ✅ | ✅ | Done |
| `binary:hinge` | ✅ | ❌ | 9 |
| `multi:softmax` | ✅ | ✅ | Done (7) |

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
7. ✅ Multiclass classification works correctly
8. ✅ Quantile regression supported
