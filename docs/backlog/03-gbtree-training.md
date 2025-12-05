# Epic: GBTree Training (Phase 1)

**Status**: � In Progress  
**Priority**: High  
**RFCs**: [0011](../design/rfcs/0011-quantization-binning.md), [0012](../design/rfcs/0012-histogram-building.md), [0013](../design/rfcs/0013-split-finding.md), [0014](../design/rfcs/0014-row-partitioning.md), [0015](../design/rfcs/0015-tree-growing.md)

Implement histogram-based gradient boosting tree training matching XGBoost/LightGBM accuracy and performance.

---

## Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| **Accuracy (Depth-wise)** | Model predictions correlate with XGBoost | Correlation > 0.99 |
| **Accuracy (Leaf-wise)** | Model predictions correlate with LightGBM | Correlation > 0.99 |
| **Accuracy** | Test set metrics (RMSE, AUC) | Within 2% of reference |
| **Performance (Depth-wise)** | Training time vs XGBoost | ≤ XGBoost (ideally faster) |
| **Performance (Leaf-wise)** | Training time vs LightGBM | ≤ LightGBM (ideally faster) |
| **Correctness** | All unit tests pass | 100% |
| **Integration** | Can load trained models for inference | Works with existing inference |

**Performance Goals**: If we achieve better performance than XGBoost/LightGBM, document
the reasons (e.g., better memory layout, more aggressive SIMD, etc.) in benchmarks.

---

## Testing Strategy Overview

### Test Categories

| Category | Purpose | When to Run | Failure Action |
|----------|---------|-------------|----------------|
| **Unit tests** | Verify algorithm correctness | Every commit | Fix immediately |
| **Integration tests** | End-to-end with XGBoost comparison | PR merge | Fix before merge |
| **Quality tests** | Model accuracy on real datasets | Story completion | Investigate root cause |
| **Performance tests** | Training speed benchmarks | Story completion | Document, investigate if >20% regression |

### Reference Data

Integration tests use pre-computed outputs from XGBoost and LightGBM:

- **Depth-wise tests**: Compare against XGBoost (`tests/test-cases/xgboost/`)
- **Leaf-wise tests**: Compare against LightGBM (`tests/test-cases/lightgbm/`)

Generated using Python scripts in `tools/data_generation/`:

- `generate_gbtree_training_data.py` - XGBoost training test cases
- `generate_lightgbm_training_data.py` - LightGBM training test cases
- Expected outputs: predictions, feature importances, tree structures

### Quality Expectations

**Depth-wise (vs XGBoost)**:

| Dataset | Metric | XGBoost Baseline | Acceptable Range |
|---------|--------|------------------|------------------|
| California Housing | RMSE | ~0.46 | 0.44 - 0.48 |
| Breast Cancer | AUC | ~0.99 | 0.98 - 1.00 |
| Iris (multiclass) | Accuracy | ~0.97 | 0.95 - 1.00 |
| Synthetic regression | Correlation | 1.00 | > 0.99 |

**Leaf-wise (vs LightGBM)**:

| Dataset | Metric | LightGBM Baseline | Acceptable Range |
|---------|--------|-------------------|------------------|
| California Housing | RMSE | ~0.45 | 0.43 - 0.47 |
| Breast Cancer | AUC | ~0.99 | 0.98 - 1.00 |
| Iris (multiclass) | Accuracy | ~0.97 | 0.95 - 1.00 |
| Synthetic regression | Correlation | 1.00 | > 0.99 |

### Performance Baselines

Measured on Apple M1 Pro, 10 cores, 16GB RAM.

**Depth-wise (vs XGBoost)**:

| Dataset | Rows | Features | Trees | XGBoost Time | Target |
|---------|------|----------|-------|--------------|--------|
| 10K regression | 10,000 | 20 | 100 | ~0.5s | ≤ 0.5s |
| 100K regression | 100,000 | 50 | 100 | ~3s | ≤ 3s |
| 1M regression | 1,000,000 | 100 | 100 | ~30s | ≤ 30s |

**Leaf-wise (vs LightGBM)**:

| Dataset | Rows | Features | Leaves | LightGBM Time | Target |
|---------|------|----------|--------|---------------|--------|
| 10K regression | 10,000 | 20 | 31 | ~0.3s | ≤ 0.3s |
| 100K regression | 100,000 | 50 | 31 | ~2s | ≤ 2s |
| 1M regression | 1,000,000 | 100 | 31 | ~20s | ≤ 20s |

**Note**: Initial implementation may be slower. Track progress across stories.
If we beat these baselines, document the optimization that enabled it.

---

## Story 1: Quantization & Binning ✅

**Goal**: Implement RFC-0011 — discretize continuous features into bins.

**RFC**: [0011-quantization-binning.md](../design/rfcs/0011-quantization-binning.md)

### Tasks

- [x] 1.1 Implement `BinCuts` struct with CSR-style storage
- [x] 1.2 Implement `BinCuts::feature_cuts()` and `num_bins()`
- [x] 1.3 Implement `BinCuts::bin_value()` with binary search
- [x] 1.4 Implement `QuantizedMatrix<B: BinIndex>` with column-major storage
- [x] 1.5 Implement `QuantizedMatrix::get()` and `feature_column()`
- [x] 1.6 Implement `BinIndex` trait for u8, u16, u32
- [x] 1.7 Implement `ExactQuantileCuts` — sort-based cut finding
- [x] 1.8 Implement `Quantizer` — transform raw features to quantized
- [x] 1.9 Handle missing values (NaN → bin 0)
- [x] 1.10 Parallel quantization per feature (Rayon)

### Unit Tests

- [x] 1.T1 `BinCuts` correctly maps values to bins (edge cases: exact boundaries, below min, above max)
- [x] 1.T2 `QuantizedMatrix` column-major layout is correct
- [x] 1.T3 Missing values mapped to bin 0
- [x] 1.T4 Quantile cuts produce expected bin boundaries
- [x] 1.T5 Different `BinIndex` types work (u8, u16)

### Integration Tests

- [ ] 1.I1 Quantize XGBoost training data, verify bin distribution similar to XGBoost's internal bins

### Quality Tests

- [ ] 1.Q1 Quantization with 256 bins preserves feature information (correlation with original > 0.99)

### Performance Tests

| Test | Dataset | Expected | Notes |
|------|---------|----------|-------|
| 1.P1 Quantize 100K×100 | Dense matrix | < 1s | Measures cut finding + transform |
| 1.P2 Quantize 1M×100 | Dense matrix | < 10s | Should scale linearly |

---

## Story 2: Histogram Building ✅

**Goal**: Implement RFC-0012 — aggregate gradients into per-bin histograms.

**RFC**: [0012-histogram-building.md](../design/rfcs/0012-histogram-building.md)

### Tasks

- [x] 2.1 Implement `FeatureHistogram` with SoA layout (sum_grad, sum_hess, count)
- [x] 2.2 Implement `FeatureHistogram::add()`, `reset()`, `bin_stats()`
- [x] 2.3 Implement `FeatureHistogram::subtract_from()` for histogram subtraction
- [x] 2.4 Implement `NodeHistogram` — collection of feature histograms
- [x] 2.5 Implement `NodeHistogram::new(cuts)` with proper bin counts
- [x] 2.6 Implement `HistogramBuilder::build()` — core accumulation kernel
- [x] 2.7 Implement `HistogramBuilder::build_parallel()` — per-feature parallelism
- [ ] 2.8 Implement `HistogramSubtractor::compute_sibling()` *(deferred to Story 10)*
- [ ] 2.9 Implement `HistogramSubtractor::select_build_child()` *(deferred to Story 10)*

### Unit Tests

- [x] 2.T1 `FeatureHistogram::add()` correctly accumulates grad/hess/count
- [x] 2.T2 `subtract_from()` produces correct sibling histogram
- [x] 2.T3 Histogram totals match sum of bins
- [x] 2.T4 `build()` produces same result as manual accumulation
- [x] 2.T5 `build_parallel()` matches sequential `build()`

### Integration Tests

- [ ] 2.I1 Build histogram from quantized XGBoost data, verify bin sums match XGBoost's internal histograms (if extractable)

### Quality Tests

N/A (histogram building is deterministic)

### Performance Tests

| Test | Setup | Expected | Notes |
|------|-------|----------|-------|
| 2.P1 Build histogram 10K rows × 100 features | 256 bins | < 10ms | Single-threaded |
| 2.P2 Build histogram 100K rows × 100 features | 256 bins | < 100ms | Single-threaded |
| 2.P3 Parallel histogram 100K × 100 | 8 threads | < 20ms | ~5x speedup expected |
| 2.P4 Histogram subtraction 100 features × 256 bins | - | < 0.1ms | Pure memory bandwidth |

---

## Story 3: Split Finding ✅

**Goal**: Implement RFC-0013 — find best splits from histograms.

**RFC**: [0013-split-finding.md](../design/rfcs/0013-split-finding.md)

### Tasks

- [x] 3.1 Implement `GainParams` with L1/L2 regularization
- [x] 3.2 Implement `leaf_objective()` and `leaf_weight()` functions
- [x] 3.3 Implement `split_gain()` — XGBoost-compatible gain formula
- [x] 3.4 Implement `SplitInfo` struct with all split metadata
- [x] 3.5 Implement `SplitFinder` trait
- [x] 3.6 Implement `GreedySplitFinder::find_best_split()`
- [x] 3.7 Implement `find_best_split_for_feature()` — scan bins for best split
- [x] 3.8 Implement `choose_default_direction()` — learn missing value direction
- [x] 3.9 Implement `find_best_split_parallel()` — parallel over features

### Unit Tests

- [x] 3.T1 `split_gain()` matches hand-computed examples
- [x] 3.T2 `leaf_weight()` with L2 regularization correct
- [x] 3.T3 `leaf_weight()` with L1 regularization (soft thresholding) correct
- [x] 3.T4 `min_child_weight` constraint filters invalid splits
- [x] 3.T5 Best split selected correctly from multiple features
- [x] 3.T6 Default direction chosen based on gain comparison
- [x] 3.T7 Parallel split finding matches sequential

### Integration Tests

- [ ] 3.I1 **Critical**: Split gain matches XGBoost's gain for same histogram
  - Generate XGBoost model, extract split gains from JSON
  - Build same histogram, verify our gain matches
- [ ] 3.I2 Best split matches XGBoost's choice (same feature, threshold)

### Quality Tests

N/A (split finding is deterministic given histogram)

### Performance Tests

| Test | Setup | Expected | Notes |
|------|-------|----------|-------|
| 3.P1 Find split 100 features × 256 bins | Pre-built histogram | < 1ms | Single-threaded |
| 3.P2 Parallel split 1000 features × 256 bins | 8 threads | < 5ms | Should scale |

**Gain Formula Verification**

The gain formula is critical. Verify against XGBoost exactly:

```
XGBoost: gain = 0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ
```

Our implementation must match this exactly (within floating-point tolerance).

---

## Story 4: Row Partitioning ✅

**Goal**: Implement RFC-0014 — track and update row-to-node assignments.

**RFC**: [0014-row-partitioning.md](../design/rfcs/0014-row-partitioning.md)

### Tasks

- [x] 4.1 Implement `RowPartitioner` with position list storage
- [x] 4.2 Implement `RowPartitioner::new()` — all rows in root
- [x] 4.3 Implement `node_rows()` — get row slice for a node
- [x] 4.4 Implement `node_size()` — count rows in node
- [x] 4.5 Implement `apply_split()` — partition rows into children
- [x] 4.6 Implement `partition_numerical()` — Dutch flag partition
- [x] 4.7 Implement `partition_categorical()` — bitset lookup
- [x] 4.8 Implement `find_threshold_bin()` — map threshold to bin

### Unit Tests

- [x] 4.T1 Initial partitioner has all rows in root
- [x] 4.T2 `apply_split()` correctly partitions rows
- [x] 4.T3 Missing values go to default direction
- [x] 4.T4 Categorical partition uses bitset correctly
- [x] 4.T5 Multiple splits maintain correct row assignments
- [x] 4.T6 No rows lost or duplicated after splits

### Integration Tests

- [ ] 4.I1 After building same tree as XGBoost, row assignments to leaves match

### Quality Tests

N/A (partitioning is deterministic)

### Performance Tests

| Test | Setup | Expected | Notes |
|------|-------|----------|-------|
| 4.P1 Partition 100K rows | Single split | < 1ms | Memory bandwidth limited |
| 4.P2 Build full tree partitions | 100 splits on 100K rows | < 50ms | Cumulative |

---

## Story 5: Tree Builder ✅

**Goal**: Implement RFC-0015 — coordinate tree building with growth policies.

**RFC**: [0015-tree-growing.md](../design/rfcs/0015-tree-growing.md)

### Tasks

- [x] 5.1 Implement `BuildingNode` and `BuildingTree` structures
- [x] 5.2 Implement `BuildingTree::new()` — create root node
- [x] 5.3 Implement `BuildingTree::expand()` — split a leaf
- [x] 5.4 Implement `BuildingTree::leaves()` — iterate current leaves
- [x] 5.5 Implement `NodeCandidate` for tracking expansion candidates
- [x] 5.6 Implement `GrowthPolicy` trait
- [x] 5.7 Implement `DepthWisePolicy` — level-by-level growth
- [x] 5.8 Implement `DepthWiseState` — track current level
- [x] 5.9 Implement `TreeBuilder<G: GrowthPolicy>` — main builder struct
- [x] 5.10 Implement `TreeBuilder::build_tree()` — core training loop

### Unit Tests

- [x] 5.T1 `BuildingTree` starts with single root leaf
- [x] 5.T2 `expand()` creates correct child structure
- [x] 5.T3 `DepthWisePolicy` expands all nodes at level before going deeper
- [x] 5.T4 `max_depth` constraint respected
- [x] 5.T5 `min_samples_split` filters small nodes
- [x] 5.T6 Learning rate applied to leaf weights

### Integration Tests

- [ ] 5.I1 **Critical**: Single tree matches XGBoost structure
  - Train single tree with same params
  - Compare tree depth, number of nodes, split features
- [ ] 5.I2 Tree predictions match XGBoost (correlation > 0.99)

### Quality Tests

See Story 8 (Full Training Loop)

### Performance Tests

See Story 8

---

## Story 6: Leaf-wise Growth ✅

**Goal**: Implement leaf-wise (best-first) tree growth as alternative to depth-wise.

**RFC**: [0015-tree-growing.md](../design/rfcs/0015-tree-growing.md)

### Tasks

- [x] 6.1 Implement `LeafWisePolicy` — expand best gain leaf
- [x] 6.2 Implement `LeafWiseState` — priority queue of candidates
- [x] 6.3 Implement `LeafCandidate` with `Ord` by gain
- [x] 6.4 Implement `max_leaves` constraint
- [x] 6.5 Update `TreeGrower` to work with `LeafWisePolicy`
- [x] 6.6 Implement `GrowthStrategy` enum for runtime selection

### Unit Tests

- [x] 6.T1 `LeafWisePolicy` always expands highest gain leaf
- [x] 6.T2 `max_leaves` constraint respected
- [x] 6.T3 Priority queue correctly updated after expansion
- [x] 6.T4 `GrowthStrategy` enum creates correct builder

### Integration Tests

- [ ] 6.I1 **Critical**: Leaf-wise predictions match LightGBM (correlation > 0.99)
  - Train with same `max_leaves`, `learning_rate`, regularization
  - Compare predictions on test set
- [ ] 6.I2 Leaf-wise with `max_leaves=31` produces similar tree structure to LightGBM
- [ ] 6.I3 Compare leaf-wise vs depth-wise on same dataset (document tradeoffs)

### Quality Tests

| Test | Expected |
|------|----------|
| 6.Q1 Leaf-wise RMSE on California Housing | Within 2% of LightGBM |
| 6.Q2 Leaf-wise vs depth-wise accuracy | Document which is better per dataset |
| 6.Q3 Leaf-wise trains faster for same accuracy | Document findings |

### Performance Tests

| Test | Setup | Expected | Notes |
|------|-------|----------|-------|
| 6.P1 Leaf-wise 100 trees on 100K rows | max_leaves=31 | ≤ LightGBM | Compare against LightGBM baseline |
| 6.P2 Leaf-wise vs depth-wise same dataset | 100K rows | Document | Which is faster for same quality? |

---

## Story 7: Gradient Boosting Trainer ✅

**Goal**: Implement full boosting loop with multiple trees.

**RFC**: [0015-tree-growing.md](../design/rfcs/0015-tree-growing.md)

### Tasks

- [x] 7.1 Implement `TreeParams` — training hyperparameters
- [x] 7.2 Implement `GBTreeTrainer<G: GrowthPolicy>` struct
- [x] 7.3 Implement `GBTreeTrainer::train()` — main training loop
- [x] 7.4 Implement gradient computation using existing `Loss` traits
- [x] 7.5 Implement `update_predictions()` — add tree predictions
- [x] 7.6 Implement `predict_row()` — traverse building tree
- [x] 7.7 Integrate `EarlyStopping` callback
- [x] 7.8 Integrate `TrainingLogger` for progress output
- [x] 7.9 Implement `freeze_forest()` — convert to inference format

### Unit Tests

- [x] 7.T1 Base score initialized correctly from labels
- [x] 7.T2 Gradients computed using `Loss` trait
- [x] 7.T3 Predictions updated after each tree
- [x] 7.T4 Early stopping triggers after patience exceeded
- [x] 7.T5 Logger called with correct metrics (verified via verbosity)

### Integration Tests

- [ ] 7.I1 **Critical**: Train regression, predictions match XGBoost
  - Train 100 trees, same hyperparameters
  - Compare predictions on test set (correlation > 0.99)
- [ ] 7.I2 Train binary classification, AUC matches XGBoost
- [ ] 7.I3 Early stopping stops at similar round as XGBoost

### Quality Tests

See Story 8

### Performance Tests

See Story 8

---

## Story 8: Full Training Validation

**Goal**: Comprehensive validation of training quality and performance.

### Quality Tests

#### Regression

| Test | Dataset | Trees | Expected RMSE | XGBoost RMSE |
|------|---------|-------|---------------|--------------|
| 8.Q1 | California Housing (20K rows) | 100 | 0.44 - 0.48 | ~0.46 |
| 8.Q2 | Synthetic linear (10K rows) | 50 | < 0.1 | < 0.1 |
| 8.Q3 | Synthetic nonlinear (10K rows) | 100 | < 0.5 | TBD |

**Failure criteria**: RMSE > 10% worse than XGBoost → investigate before proceeding.

#### Binary Classification

| Test | Dataset | Trees | Expected AUC | XGBoost AUC |
|------|---------|-------|--------------|-------------|
| 8.Q4 | Breast Cancer (569 rows) | 50 | 0.98 - 1.00 | ~0.99 |
| 8.Q5 | Synthetic binary (10K rows) | 100 | > 0.95 | TBD |

**Failure criteria**: AUC > 2% worse than XGBoost → investigate.

#### Multiclass Classification

| Test | Dataset | Trees | Expected Accuracy | XGBoost Accuracy |
|------|---------|-------|-------------------|------------------|
| 8.Q6 | Iris (150 rows) | 50 | 0.95 - 1.00 | ~0.97 |
| 8.Q7 | Synthetic multiclass (10K rows) | 100 | > 0.90 | TBD |

### Performance Tests

All times measured on Apple M1 Pro (10 cores, 16GB RAM).

| Test | Dataset | Trees | XGBoost Time | Target | Acceptable |
|------|---------|-------|--------------|--------|------------|
| 8.P1 | 10K × 20 features | 100 | ~0.5s | ≤ 0.5s | ≤ 1.0s |
| 8.P2 | 100K × 50 features | 100 | ~3s | ≤ 3s | ≤ 5s |
| 8.P3 | 100K × 50, 8 threads | 100 | ~1s | ≤ 1s | ≤ 2s |
| 8.P4 | 1M × 100 features | 100 | ~30s | ≤ 30s | ≤ 60s |

**Failure criteria**: >2x slower than XGBoost → investigate and optimize.

### Integration Tests

- [ ] 8.I1 Trained model saves to JSON (XGBoost format)
- [ ] 8.I2 Saved model loads correctly with existing inference code
- [ ] 8.I3 Inference on trained model matches training predictions
- [ ] 8.I4 Round-trip: Train → Save → Load → Predict matches

---

## Story 9: Test Data Generation

**Goal**: Create reference datasets and XGBoost baselines for all tests.

### Tasks

- [ ] 9.1 Create `tools/data_generation/generate_gbtree_training_data.py` (XGBoost)
- [ ] 9.2 Create `tools/data_generation/generate_lightgbm_training_data.py` (LightGBM)
- [ ] 9.3 Generate synthetic regression datasets (linear, nonlinear)
- [ ] 9.4 Generate synthetic classification datasets (binary, multiclass)
- [ ] 9.5 Generate XGBoost trained models (depth-wise) for each dataset
- [ ] 9.6 Generate LightGBM trained models (leaf-wise) for each dataset
- [ ] 9.7 Export expected predictions, metrics, tree structures
- [ ] 9.8 Store XGBoost baselines in `tests/test-cases/xgboost/gbtree/`
- [ ] 9.9 Store LightGBM baselines in `tests/test-cases/lightgbm/gbtree/`
- [ ] 9.10 Create Rust test case loader in `tests/test_data.rs`
- [ ] 9.11 Document dataset generation process in `tests/test-cases/README.md`

### Test Case Structure

```text
tests/test-cases/
├── xgboost/gbtree/          # Depth-wise baselines
│   ├── regression/
│   │   ├── california_housing/
│   │   │   ├── train_data.csv
│   │   │   ├── test_data.csv
│   │   │   ├── model.json
│   │   │   ├── expected_predictions.csv
│   │   │   └── expected_metrics.json
│   │   └── synthetic_linear/
│   ├── binary_classification/
│   └── multiclass/
│
└── lightgbm/gbtree/         # Leaf-wise baselines
    ├── regression/
    │   ├── california_housing/
    │   │   ├── train_data.csv
    │   │   ├── test_data.csv
    │   │   ├── model.txt
    │   │   ├── expected_predictions.csv
    │   │   └── expected_metrics.json
    │   └── synthetic_linear/
    ├── binary_classification/
    └── multiclass/
```

---

## Story 10: Histogram Subtraction Optimization

**Goal**: Implement histogram subtraction for ~2x training speedup.

**RFC**: [0012-histogram-building.md](../design/rfcs/0012-histogram-building.md)

### Tasks

- [ ] 10.1 Track parent histograms during tree building
- [ ] 10.2 Select smaller child for direct histogram build
- [ ] 10.3 Derive larger child via `parent - smaller = larger`
- [ ] 10.4 Validate subtraction produces identical histograms
- [ ] 10.5 Benchmark improvement

### Unit Tests

- [ ] 10.T1 Subtraction produces identical histogram to direct build
- [ ] 10.T2 `select_build_child()` correctly chooses smaller child

### Performance Tests

| Test | Setup | Before | After | Expected Speedup |
|------|-------|--------|-------|------------------|
| 10.P1 | 100K rows, 100 trees | TBD | TBD | ~1.5-2x |

---

## Story 11: Performance Optimization

**Goal**: Profile and optimize training performance to match XGBoost.

### Tasks

- [ ] 11.1 Profile full training with `perf` or `Instruments`
- [ ] 11.2 Identify hotspots (expect: histogram building, split finding)
- [ ] 11.3 Optimize histogram accumulation (SIMD if beneficial)
- [ ] 11.4 Optimize memory allocation (histogram pooling)
- [ ] 11.5 Optimize parallel scaling (tune Rayon chunk sizes)
- [ ] 11.6 Document optimizations in `docs/benchmarks/`

### Performance Targets

After optimization, must meet Story 8 performance targets.

---

## Story 12: Documentation & Examples

**Goal**: User-facing documentation and examples.

### Tasks

- [ ] 12.1 Add rustdoc for all public training APIs
- [ ] 12.2 Create `examples/train_regression.rs`
- [ ] 12.3 Create `examples/train_classification.rs`
- [ ] 12.4 Update README with training quickstart
- [ ] 12.5 Document hyperparameters in API docs
- [ ] 12.6 Add benchmark results to `docs/benchmarks/`

---

## Dependencies

```
Story 1 (Quantization)
    ↓
Story 2 (Histograms) ←─────────────────┐
    ↓                                   │
Story 3 (Split Finding)                 │
    ↓                                   │
Story 4 (Row Partitioning)              │
    ↓                                   │
Story 5 (Tree Builder)                  │
    ↓                                   │
Story 6 (Leaf-wise)                     │
    ↓                                   │
Story 7 (Full Trainer) ─────────────────┤
    ↓                                   │
Story 8 (Validation)                    │
                                        │
Story 9 (Test Data) ────────────────────┘
Story 10 (Histogram Subtraction) ───────┘
Story 11 (Optimization) ← After Story 8
Story 12 (Documentation) ← After Story 8
```

---

## Hyperparameter Reference

Default hyperparameters to match XGBoost:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_depth` | 6 | Maximum tree depth |
| `max_leaves` | 0 (unlimited) | Max leaves for leaf-wise |
| `learning_rate` | 0.3 | Shrinkage factor |
| `lambda` | 1.0 | L2 regularization |
| `alpha` | 0.0 | L1 regularization |
| `gamma` | 0.0 | Minimum split gain |
| `min_child_weight` | 1.0 | Minimum hessian sum per leaf |
| `subsample` | 1.0 | Row subsampling ratio |
| `colsample_bytree` | 1.0 | Column subsampling ratio |
| `max_bin` | 256 | Maximum histogram bins |

---

## Definition of Done

- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] Quality tests within acceptable range
- [ ] Performance tests meeting targets
- [ ] Rustdoc complete for public APIs
- [ ] Examples working
- [ ] No compiler warnings
- [ ] `cargo clippy` clean

---

## References

- [RFC-0011: Quantization & Binning](../design/rfcs/0011-quantization-binning.md)
- [RFC-0012: Histogram Building](../design/rfcs/0012-histogram-building.md)
- [RFC-0013: Split Finding](../design/rfcs/0013-split-finding.md)
- [RFC-0014: Row Partitioning](../design/rfcs/0014-row-partitioning.md)
- [RFC-0015: Tree Growing](../design/rfcs/0015-tree-growing.md)
- [Feature Overview](../design/FEATURE_OVERVIEW.md)
- [Phase 2 Research Notes](../design/research/phase2-notes.md)
