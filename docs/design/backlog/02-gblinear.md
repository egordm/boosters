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

## Story 3: GBLinear Training

**Goal**: Train linear models via coordinate descent.

- [ ] 3.1 `CSCMatrix` — column-sparse format for efficient column access
- [ ] 3.2 `CSCMatrix::from_dense()` and column iteration
- [ ] 3.3 Coordinate descent update with elastic net regularization
- [ ] 3.4 `ShotgunUpdater` — parallel feature updates (default)
- [ ] 3.5 `CoordinateUpdater` — sequential feature updates
- [ ] 3.6 `CyclicSelector` and `ShuffleSelector` for feature order
- [ ] 3.7 `LinearTrainer` high-level API
- [ ] 3.8 Early stopping + logging integration

**Refs**: [RFC-0009](../rfcs/0009-gblinear-training.md)

---

## Story 4: Training Validation

**Goal**: Verify trained models match XGBoost quality.

- [ ] 4.1 Generate reference training data with Python XGBoost
- [ ] 4.2 Compare final metrics (RMSE, logloss) within tolerance
- [ ] 4.3 Compare predictions on held-out test set
- [ ] 4.4 Verify weight correlation (Pearson r > 0.95)
- [ ] 4.5 Test convergence on regression, binary, multiclass tasks

**Validation approach**: Since exact weight matching is unlikely due to
randomness and floating-point differences, we validate:

1. Final metrics are within 5% of XGBoost
2. Test predictions have < 1e-3 RMSE vs XGBoost predictions
3. Weight vectors are highly correlated (not necessarily identical)

---

## Story 5: Benchmarks

**Goal**: Validate performance.

- [ ] 5.1 Inference benchmark vs Python XGBoost
- [ ] 5.2 Training benchmark vs Python XGBoost
- [ ] 5.3 Compare CSC vs dense column access for training
- [ ] 5.4 Document results and recommend defaults
- [ ] 4.3 Document results

---

## Success Criteria

1. Load XGBoost GBLinear JSON models and predict correctly
2. Train models matching Python XGBoost quality (metrics within 5%)
3. Training infrastructure (losses, metrics, callbacks) is reusable
4. Early stopping and logging work correctly
5. Trained model predictions correlate highly with XGBoost predictions
