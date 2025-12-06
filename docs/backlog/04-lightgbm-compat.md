# Epic 04: LightGBM Compatibility

**Status**: ✅ Inference Complete (Stories 1-5)  
**Priority**: P1 — Complete before Phase 2 optimizations  
**Estimate**: 1-2 weeks

---

## Overview

Add LightGBM model loading and inference support to booste-rs. This enables:
- Loading pre-trained LightGBM models
- Validating leaf-wise training against LightGBM baselines
- Generating training test cases for comparison

**Why before Phase 2?** Our leaf-wise growth (Story 6) implements LightGBM's approach. 
Proper validation requires loading LightGBM models and comparing predictions. This also
unblocks generating LightGBM training baselines for Story 9.

---

## Goals

1. **Inference parity**: ✅ Load LightGBM models and predict with identical results
2. **Training validation**: Generate LightGBM baselines for leaf-wise comparison
3. **Performance**: Match or exceed LightGBM C++ inference speed

---

## Non-Goals (Phase 2+)

- Linear trees (leaf contains linear model) — separate RFC
- Categorical feature training — separate RFC (RFC-0016)
- GOSS sampling — Phase 2 optimization

---

## Research Summary

LightGBM model format options:
- **Text format** (`.txt`): Human-readable, easier to parse ✅ Implemented
- **Model string**: Same as text, but in memory ✅ Implemented
- **Binary format**: Faster but undocumented (deferred)

**Key implementation insight**: LightGBM's `leaf_value` has shrinkage (learning rate) 
already applied during training via `Tree::Shrinkage()`. The `shrinkage` field in the 
model file is purely informational — do NOT multiply leaf values by it again!

### LightGBM Text Format Structure

```
tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=9
...

Tree=0
num_leaves=31
num_cat=0
split_feature=...
split_gain=...
threshold=...
decision_type=...
left_child=...
right_child=...
leaf_value=...
shrinkage=1.0
...
```

Key differences from XGBoost:
- Uses `num_leaves` not `max_depth` as primary constraint
- `decision_type` encodes split type (numerical, categorical, default direction)
- Leaf indices are negative (left_child < 0 means leaf index = ~left_child)
- Shrinkage baked into leaf values (not applied at inference)

---

## Stories

### Story 1: LightGBM Model Format Research ✅ COMPLETE

**Goal**: Document LightGBM text format thoroughly.

#### Tasks

- [x] 1.1 Study LightGBM source code for text format parsing
- [x] 1.2 Document tree structure encoding (nodes, leaves, splits)
- [x] 1.3 Document categorical feature handling
- [x] 1.4 Document multi-class model structure
- [x] 1.5 Create research notes in `docs/design/research/lightgbm/`

#### Deliverables

- ✅ `docs/design/research/lightgbm/model-format.md` (448 lines)
- ✅ Example models for regression, binary, multiclass, missing values

---

### Story 2: LightGBM Model Parser ✅ COMPLETE

**Goal**: Parse LightGBM text format into internal representation.

#### Tasks

- [x] 2.1 Create `src/compat/lightgbm/mod.rs` module structure
- [x] 2.2 Implement `LgbModel` parser for text format
- [x] 2.3 Parse header section (num_class, num_trees, feature info)
- [x] 2.4 Parse tree section (splits, thresholds, leaf values)
- [x] 2.5 Handle decision_type bitfield (default direction, categorical)
- [x] 2.6 Handle multi-class models (num_tree_per_iteration)
- [x] 2.7 Comprehensive error handling with context

#### Unit Tests

- [x] 2.T1 Parse single regression tree
- [x] 2.T2 Parse binary classification model
- [x] 2.T3 Parse multiclass model (3+ classes)
- [x] 2.T4 Parse model with missing value handling
- [x] 2.T5 Error on malformed input

---

### Story 3: LightGBM to Internal Conversion ✅ COMPLETE

**Goal**: Convert parsed LightGBM model to booste-rs forest format.

#### Tasks

- [x] 3.1 Map LightGBM tree structure to `SoATreeStorage`
- [x] 3.2 Convert leaf indices (negative → positive using bitwise NOT)
- [x] 3.3 Convert split conditions to `SplitCondition`
- [x] 3.4 Handle default direction from decision_type
- [x] 3.5 Handle shrinkage correctly (already baked in!)
- [x] 3.6 Map objective to appropriate forest type

#### Unit Tests

- [x] 3.T1 Converted tree has correct structure
- [x] 3.T2 Leaf values match original (no double shrinkage!)
- [x] 3.T3 Split thresholds match original
- [x] 3.T4 Default direction preserved

---

### Story 4: LightGBM Inference Validation ✅ COMPLETE

**Goal**: Validate predictions match LightGBM exactly.

#### Tasks

- [x] 4.1 Create test case generation script (`tools/data_generation/generate_lightgbm.py`)
- [x] 4.2 Generate regression test cases
- [x] 4.3 Generate binary classification test cases
- [x] 4.4 Generate multiclass test cases
- [x] 4.5 Generate test cases with missing values
- [x] 4.6 Store in `tests/test-cases/lightgbm/inference/`

#### Integration Tests

- [x] 4.I1 Regression predictions match (tolerance 1e-2)
- [x] 4.I2 Binary logistic predictions match
- [x] 4.I3 Multiclass softmax predictions match
- [x] 4.I4 Missing value handling matches

#### Test Case Structure

```
tests/test-cases/lightgbm/inference/
├── small_tree/           # 3 trees, simple model
├── regression/           # Full regression model
├── binary_classification/ # Binary classifier
├── multiclass/           # 3-class classifier
└── regression_missing/   # Missing value handling
```

---

### Story 5: LightGBM Feature Flag & API ✅ COMPLETE

**Goal**: Expose LightGBM loading behind feature flag.

#### Tasks

- [x] 5.1 Add `lightgbm-compat` feature flag to Cargo.toml
- [x] 5.2 Create `LgbModel::from_file()` API
- [x] 5.3 Create `LgbModel::from_string()` API
- [x] 5.4 Implement `LgbModel::to_forest()` conversion
- [x] 5.5 Add integration tests (`tests/inference_lightgbm.rs`)

#### Final API

```rust
#[cfg(feature = "lightgbm-compat")]
use booste_rs::compat::lightgbm::LgbModel;

let model = LgbModel::from_file("model.txt")?;
let forest = model.to_forest()?;
let predictions = forest.predict_row(&features);
```

---

### Story 6: LightGBM Training Baselines 📋 PLANNED

**Goal**: Generate LightGBM training baselines for leaf-wise validation.

#### Tasks

- [ ] 6.1 Add LightGBM generation to `generate_lightgbm.py`
- [ ] 6.2 Generate leaf-wise regression baseline
- [ ] 6.3 Generate leaf-wise classification baseline
- [ ] 6.4 Store train data, config, predictions
- [ ] 6.5 Add integration tests comparing our training to LightGBM
- [ ] 6.6 Update Story 9 in GBTree training backlog

---

### Story 7: Performance Benchmarks 📋 PLANNED

**Goal**: Benchmark inference performance against LightGBM C++.

#### Tasks

- [ ] 7.1 Create benchmark suite for LightGBM models
- [ ] 7.2 Benchmark single-row latency
- [ ] 7.3 Benchmark batch prediction throughput
- [ ] 7.4 Compare against LightGBM C++ inference
- [ ] 7.5 Document results in `docs/benchmarks/`

---

## Test Summary

| Test Suite | Tests | Status |
|------------|-------|--------|
| Parser unit tests | 4 | ✅ Pass |
| Conversion unit tests | 4 | ✅ Pass |
| Prediction unit tests | 5 | ✅ Pass |
| Integration tests | 8 | ✅ Pass |
| **Total** | **21** | ✅ All Pass |

---

## Dependencies

- **Requires**: None (builds on existing inference infrastructure)
- **Blocks**: 
  - Story 9 LightGBM baselines (GBTree training)
  - Phase 2 leaf-wise optimizations

---

## References

- [LightGBM Source: tree.h](https://github.com/microsoft/LightGBM/blob/master/include/LightGBM/tree.h)
- [LightGBM Source: tree.cpp](https://github.com/microsoft/LightGBM/blob/master/src/io/tree.cpp)
- [LightGBM Source: gbdt_model_text.cpp](https://github.com/microsoft/LightGBM/blob/master/src/boosting/gbdt_model_text.cpp)
- [Research notes: model-format.md](../design/research/lightgbm/model-format.md)

---

## Changelog

- 2024-11-30: Initial epic created, moved from future backlog
- 2024-11-30: Stories 1-5 completed (inference support)
  - Fixed shrinkage bug (leaf values already have shrinkage applied)
  - Added comprehensive test suite (21 tests)
  - Integration tests in `tests/inference_lightgbm.rs`
