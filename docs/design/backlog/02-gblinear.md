# Epic: GBLinear Support

**Status**: 🔄 Active  
**Priority**: High  
**RFCs**: [0008](../rfcs/0008-gblinear-inference.md), [0009](../rfcs/0009-gblinear-training.md)

Add support for XGBoost's linear booster — both inference and training.
Validates training infrastructure before GBTree training.

---

## Story 1: GBLinear Inference ✓

**Goal**: Load and predict with XGBoost GBLinear models.

- [x] 1.1 `LinearModel` struct with `Box<[f32]>` weight storage
- [x] 1.2 Weight indexing: `weights[feature * num_groups + group]`
- [x] 1.3 `predict_row()` and `predict_batch()` methods
- [x] 1.4 `par_predict_batch()` with Rayon
- [x] 1.5 `Booster::Linear` variant in model enum
- [x] 1.6 XGBoost JSON parser for `gblinear` section
- [x] 1.7 Integration tests vs Python XGBoost

**Refs**: [RFC-0008](../rfcs/0008-gblinear-inference.md)

---

## Story 2: Training Infrastructure ✓

**Goal**: Core training types reusable for GBLinear and GBTree.

- [x] 2.1 `GradientPair` struct (grad, hess)
- [x] 2.2 `Loss` trait — compute gradients from predictions + labels
- [x] 2.3 Common losses: squared error, logistic, softmax
- [x] 2.4 `Metric` trait — evaluate model quality
- [x] 2.5 Common metrics: RMSE, MAE, logloss, AUC, accuracy
- [x] 2.6 `EarlyStopping` callback
- [x] 2.7 `TrainingLogger` with verbosity levels

---

## Story 3: GBLinear Training ✓

**Goal**: Train linear models via coordinate descent.

- [x] 3.1 `CSCMatrix` — column-sparse format for efficient column access
- [x] 3.2 `CSCMatrix::from_dense()` and column iteration
- [x] 3.3 Coordinate descent update with elastic net regularization
- [x] 3.4 `ShotgunUpdater` — parallel feature updates (default)
- [x] 3.5 `CoordinateUpdater` — sequential feature updates
- [x] 3.6 `CyclicSelector` and `ShuffleSelector` for feature order
- [x] 3.7 `LinearTrainer` high-level API
- [x] 3.8 Integration tests comparing to XGBoost

**Refs**: [RFC-0009](../rfcs/0009-gblinear-training.md)

---

## Story 4: Matrix Layout Refactor ✓

**Goal**: Support both row-major and column-major dense matrices via zero-cost abstraction.

**Motivation**: Current `DenseMatrix` is row-major, which is optimal for tree
prediction (iterate rows, access features). But coordinate descent iterates
over features (columns), so column-major storage would give better cache
locality for training.

**Design Decision**: Use generic `DenseMatrix<T, L: Layout>` with `RowMajor`/`ColMajor`
type parameters. This is zero-cost (monomorphized) and allows writing code generic
over layout. Keep CSC/CSR as separate types since they have fundamentally different
storage structures, not just different indexing.

**RFC**: [0010-matrix-layouts.md](../rfcs/0010-matrix-layouts.md) ✓ Accepted

Tasks:

- [x] 4.1 Add `Layout` trait with `RowMajor` and `ColMajor` implementations
- [x] 4.2 Refactor `DenseMatrix` to `DenseMatrix<T, L: Layout = RowMajor>`
- [x] 4.3 Update `DataMatrix` impl to use `L::index()`
- [x] 4.4 Add `to_layout<L2>()` conversion method
- [x] 4.5 Add layout-specific slice methods (`row_slice`, `col_slice`, `rows_slice`)
- [x] 4.6 Add strided iterators for non-contiguous dimension
- [x] 4.7 Type aliases: `RowMatrix`, `ColMatrix` for convenience
- [x] 4.8 Verify existing tests still pass (backward compatibility)
- [ ] 4.9 Add benchmarks comparing layouts for training

---

## Story 5: Training Validation

**Goal**: Verify trained models match XGBoost quality.

- [ ] 5.1 Generate reference training data with Python XGBoost
- [ ] 5.2 Compare final metrics (RMSE, logloss) within tolerance
- [ ] 5.3 Compare predictions on held-out test set
- [ ] 5.4 Verify weight correlation (Pearson r > 0.95)
- [ ] 5.5 Test convergence on regression, binary, multiclass tasks

**Validation approach**: Since exact weight matching is unlikely due to
randomness and floating-point differences, we validate:

1. Final metrics are within 5% of XGBoost
2. Test predictions have < 1e-3 RMSE vs XGBoost predictions
3. Weight vectors are highly correlated (not necessarily identical)

---

## Story 6: Benchmarks

**Goal**: Validate performance.

- [ ] 6.1 Inference benchmark vs Python XGBoost
- [ ] 6.2 Training benchmark vs Python XGBoost
- [ ] 6.3 Compare CSC vs ColMajor vs RowMajor for training
- [ ] 6.4 Document results and recommend defaults

---

## Success Criteria

1. Load XGBoost GBLinear JSON models and predict correctly
2. Train models matching Python XGBoost quality (metrics within 5%)
3. Training performance equal to or faster than XGBoost
4. Training infrastructure (losses, metrics, callbacks) is reusable
5. Early stopping and logging work correctly
6. Trained model predictions correlate highly with XGBoost predictions
