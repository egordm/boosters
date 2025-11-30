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

## Story 2: Training Infrastructure

**Goal**: Core training types reusable for GBLinear and GBTree.

- [ ] 2.1 `GradientPair` struct (grad, hess)
- [ ] 2.2 `Objective` trait — compute gradients from predictions + labels
- [ ] 2.3 Common objectives: squared error, logistic, softmax
- [ ] 2.4 `Metric` trait — evaluate model quality
- [ ] 2.5 Common metrics: RMSE, MAE, logloss, AUC, accuracy
- [ ] 2.6 `EarlyStopping` callback
- [ ] 2.7 `TrainingLogger` with verbosity levels

---

## Story 3: GBLinear Training

**Goal**: Train linear models via coordinate descent.

- [ ] 3.1 `CSCMatrix` — column-sparse format
- [ ] 3.2 `CSCMatrix::from_dense()` conversion
- [ ] 3.3 Coordinate descent update with elastic net
- [ ] 3.4 `ShotgunUpdater` — parallel (default)
- [ ] 3.5 `CoordinateUpdater` — sequential
- [ ] 3.6 `CyclicSelector` and `ShuffleSelector`
- [ ] 3.7 `LinearTrainer` high-level API
- [ ] 3.8 Early stopping + logging integration
- [ ] 3.9 Integration tests vs Python XGBoost

**Refs**: [RFC-0009](../rfcs/0009-gblinear-training.md)

---

## Story 4: Benchmarks

**Goal**: Validate performance.

- [ ] 4.1 Inference benchmark vs Python XGBoost
- [ ] 4.2 Training benchmark vs Python XGBoost
- [ ] 4.3 Document results

---

## Success Criteria

1. Load XGBoost GBLinear JSON models and predict correctly
2. Train models matching Python XGBoost quality
3. Training infrastructure (objectives, metrics, callbacks) is reusable
4. Early stopping and logging work correctly
