# Epic 04: LightGBM Compatibility

**Status**: � Active  
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

1. **Inference parity**: Load LightGBM models and predict with identical results
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
- **Text format** (`.txt`): Human-readable, easier to parse
- **Model string**: Same as text, but in memory
- **Binary format**: Faster but undocumented

Recommended approach: Start with text format (well-documented), add binary later if needed.

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
...
```

Key differences from XGBoost:
- Uses `num_leaves` not `max_depth` as primary constraint
- `decision_type` encodes split type (numerical, categorical, default direction)
- Leaf indices are negative (left_child < 0 means leaf)
- Different base score handling

---

## Stories

### Story 1: LightGBM Model Format Research

**Goal**: Document LightGBM text format thoroughly.

#### Tasks

- [ ] 1.1 Study LightGBM source code for text format parsing
- [ ] 1.2 Document tree structure encoding (nodes, leaves, splits)
- [ ] 1.3 Document categorical feature handling
- [ ] 1.4 Document multi-class model structure
- [ ] 1.5 Create research notes in `docs/design/research/lightgbm/`

#### Deliverables

- `docs/design/research/lightgbm/model-format.md`
- Example models for each objective type

---

### Story 2: LightGBM Model Parser

**Goal**: Parse LightGBM text format into internal representation.

#### Tasks

- [ ] 2.1 Create `src/compat/lightgbm/mod.rs` module structure
- [ ] 2.2 Implement `LightGBMModelParser` for text format
- [ ] 2.3 Parse header section (num_class, num_trees, feature info)
- [ ] 2.4 Parse tree section (splits, thresholds, leaf values)
- [ ] 2.5 Handle decision_type bitfield (default direction, categorical)
- [ ] 2.6 Handle multi-class models (num_tree_per_iteration)
- [ ] 2.7 Comprehensive error handling with context

#### Unit Tests

- [ ] 2.T1 Parse single regression tree
- [ ] 2.T2 Parse binary classification model
- [ ] 2.T3 Parse multiclass model (3+ classes)
- [ ] 2.T4 Parse model with categorical features
- [ ] 2.T5 Parse model with missing value handling
- [ ] 2.T6 Error on malformed input

---

### Story 3: LightGBM to Internal Conversion

**Goal**: Convert parsed LightGBM model to booste-rs forest format.

#### Tasks

- [ ] 3.1 Map LightGBM tree structure to `SoATreeStorage`
- [ ] 3.2 Convert leaf indices (negative → positive)
- [ ] 3.3 Convert split conditions to `SplitCondition`
- [ ] 3.4 Handle default direction from decision_type
- [ ] 3.5 Convert categorical splits to bitset format
- [ ] 3.6 Map objective to `PredictStrategy`
- [ ] 3.7 Handle base score differences

#### Unit Tests

- [ ] 3.T1 Converted tree has correct structure
- [ ] 3.T2 Leaf values match original
- [ ] 3.T3 Split thresholds match original
- [ ] 3.T4 Default direction preserved
- [ ] 3.T5 Categorical splits converted correctly

---

### Story 4: LightGBM Inference Validation

**Goal**: Validate predictions match LightGBM exactly.

#### Tasks

- [ ] 4.1 Create test case generation script for LightGBM
- [ ] 4.2 Generate regression test cases
- [ ] 4.3 Generate binary classification test cases
- [ ] 4.4 Generate multiclass test cases
- [ ] 4.5 Generate test cases with missing values
- [ ] 4.6 Store in `tests/test-cases/lightgbm/inference/`

#### Integration Tests

- [ ] 4.I1 Regression predictions match (tolerance 1e-6)
- [ ] 4.I2 Binary logistic predictions match
- [ ] 4.I3 Multiclass softmax predictions match
- [ ] 4.I4 Missing value handling matches

#### Test Case Structure

```
tests/test-cases/lightgbm/
├── inference/
│   ├── regression.model.txt
│   ├── regression.input.json
│   ├── regression.expected.json
│   ├── binary_classification.model.txt
│   ├── binary_classification.input.json
│   ├── binary_classification.expected.json
│   └── ...
└── training/  # Added in Story 6
```

---

### Story 5: LightGBM Feature Flag & API

**Goal**: Expose LightGBM loading behind feature flag.

#### Tasks

- [ ] 5.1 Add `lightgbm-compat` feature flag to Cargo.toml
- [ ] 5.2 Create `LightGBMModel::load()` API
- [ ] 5.3 Create `LightGBMModel::load_string()` API
- [ ] 5.4 Implement `From<LightGBMModel> for EnsembleModel`
- [ ] 5.5 Add rustdoc with examples
- [ ] 5.6 Update README with LightGBM support

#### API Design

```rust
#[cfg(feature = "lightgbm-compat")]
pub mod lightgbm {
    pub struct LightGBMModel { /* ... */ }
    
    impl LightGBMModel {
        /// Load from text format file
        pub fn load(path: impl AsRef<Path>) -> Result<Self, LightGBMError>;
        
        /// Load from text format string
        pub fn load_string(content: &str) -> Result<Self, LightGBMError>;
    }
    
    impl From<LightGBMModel> for EnsembleModel {
        fn from(model: LightGBMModel) -> Self { /* ... */ }
    }
}
```

---

### Story 6: LightGBM Training Baselines

**Goal**: Generate LightGBM training baselines for leaf-wise validation.

#### Tasks

- [ ] 6.1 Add LightGBM generation to `generate_lightgbm.py`
- [ ] 6.2 Generate leaf-wise regression baseline
- [ ] 6.3 Generate leaf-wise classification baseline
- [ ] 6.4 Store train data, config, predictions
- [ ] 6.5 Add integration tests comparing our training to LightGBM
- [ ] 6.6 Update Story 9 in GBTree training backlog

#### Test Cases

| Name | Rows | Features | Trees | Leaves | Type |
|------|------|----------|-------|--------|------|
| leaf_wise_regression | 160 | 10 | 30 | 31 | Regression |
| leaf_wise_binary | 160 | 10 | 30 | 31 | Binary |
| leaf_wise_multiclass | 240 | 10 | 30 | 31 | Multiclass |

---

### Story 7: Performance Benchmarks

**Goal**: Benchmark inference performance against LightGBM C++.

#### Tasks

- [ ] 7.1 Create benchmark suite for LightGBM models
- [ ] 7.2 Benchmark single-row latency
- [ ] 7.3 Benchmark batch prediction throughput
- [ ] 7.4 Compare against LightGBM C++ inference
- [ ] 7.5 Document results in `docs/benchmarks/`

#### Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Single-row latency | ≤ LightGBM C++ | Should be competitive |
| Batch throughput | ≥ LightGBM C++ | Vectorized traversal |

---

## Dependencies

- **Requires**: None (builds on existing inference infrastructure)
- **Blocks**: 
  - Story 9 LightGBM baselines (GBTree training)
  - Phase 2 leaf-wise optimizations

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Undocumented format details | Study LightGBM source code, create comprehensive test suite |
| Categorical encoding differences | Research LightGBM categorical handling thoroughly |
| Binary format complexity | Start with text format, defer binary to later |

---

## References

- [LightGBM Model Format](https://lightgbm.readthedocs.io/en/latest/Parameters.html#model-parameters)
- [LightGBM Source: model.cpp](https://github.com/microsoft/LightGBM/blob/master/src/io/model.cpp)
- [LightGBM Source: tree.cpp](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/tree.cpp)

---

## Changelog

- 2024-11-30: Initial epic created, moved from future backlog
