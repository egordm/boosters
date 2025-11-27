# booste-rs Implementation Roadmap

## Philosophy

**Slice-wise implementation**: Build thin vertical slices that work end-to-end, then expand horizontally.

**Guiding principle**: At each milestone, we should be able to load a real XGBoost model and produce correct predictions.

---

## Phase 1: Minimal Viable Prediction

**Goal**: Load a simple XGBoost regression model (JSON) and predict on dense data.

### Milestone 1.1: Core Types (Leaf & Node)

Implement the foundational types from RFC-0002.

- [x] `LeafValue` trait + `ScalarLeaf` implementation
- [x] `Node<L>` enum (Split/Leaf variants)
- [x] `SplitCondition` struct (feature index, threshold, default direction)
- [x] Basic tests for node construction

**Files**: `src/trees/node.rs`, `src/trees/leaf.rs`

### Milestone 1.2: Tree Storage

Implement tree storage from RFC-0002.

- [x] `SoATreeStorage<L>` — flat arrays for nodes
- [x] Node indexing (left/right child access)
- [x] Tree construction from node data (`TreeBuilder`)
- [x] Traversal: `predict_row(&[f32]) -> LeafValue`

**Files**: `src/trees/soa.rs`

### Milestone 1.3: Forest

Implement forest from RFC-0001.

- [x] `SoAForest<L>` — collection of trees with group assignments
- [x] `SoATreeView<'a, L>` — borrowed view into a single tree (from RFC-0002)
- [x] `Forest::predict_row()` — sum leaf values across trees
- [x] Tree iteration via views

**Files**: `src/forest/mod.rs`, `src/forest/soa.rs`

### Milestone 1.4: XGBoost JSON Loader

Refactor existing loader code per RFC-0007.

- [x] Foreign types: `XgbModel`, `XgbTree`, `XgbNode` (serde) — already existed
- [x] Conversion: `XgbModel::to_forest()` → `SoAForest<ScalarLeaf>`
- [x] Basic model metadata extraction (base_score, num_class)
- [x] Feature-gate behind `xgboost-compat`

**Files**: `src/compat/mod.rs`, `src/compat/xgboost/json.rs`, `src/compat/xgboost/convert.rs`

### Milestone 1.5: Simple Prediction API

Minimal prediction without full `Model` wrapper.

- [x] `SoAForest::predict_batch()` — implemented in M1.3
- [x] Integration test: load XGBoost model, predict, compare to Python

**Files**: `tests/xgboost_compat.rs`, `tools/data_generation/scripts/generate_test_cases.py`

### ✅ Phase 1 Complete

**Deliverable**: Can load XGBoost JSON regression model and predict correctly.

**Validation**: Compare predictions to Python XGBoost on test data.

---

## Phase 2: Full Inference Pipeline

**Goal**: Complete inference API with proper abstractions.

### Milestone 2.1: DataMatrix Trait

Implement core data abstraction from RFC-0004.

- [x] `DataMatrix` trait definition
- [x] `DenseMatrix<f32>` implementation
- [x] `RowView` trait and dense row view
- [x] Missing value handling (NaN)

**Files**: `src/data/mod.rs`, `src/data/dense.rs`, `src/data/traits.rs`

### Milestone 2.2: Predictor & Visitor

Implement traversal patterns from RFC-0003.

- [ ] `TreeVisitor` trait
- [ ] `PredictVisitor` — simple row-at-a-time prediction
- [ ] `Predictor` struct wrapping forest + visitor strategy
- [ ] Wire up `DataMatrix` input

**Files**: `src/predict/mod.rs`, `src/predict/visitor.rs`

### Milestone 2.3: Model Wrapper

Complete `Model` type from RFC-0007.

- [ ] `Model` struct (booster, meta, features, objective)
- [ ] `Booster` enum (Tree variant only for now)
- [ ] `ModelMeta`, `FeatureInfo`, `Objective` types
- [ ] `Model::predict()` high-level API

**Files**: `src/model.rs`

### Milestone 2.4: Objective Transforms

Post-prediction transformations.

- [ ] `Objective` enum with common objectives
- [ ] `Objective::transform()` — apply sigmoid, softmax, etc.
- [ ] Wire into `Model::predict()`

**Files**: `src/objective.rs`

### ✅ Phase 2 Complete

**Deliverable**: Clean public API: `Model::load()` + `Model::predict()`.

**Validation**: Binary classification, multiclass models work correctly.

---

## Phase 3: Advanced Features

### Milestone 3.1: DART Support

- [ ] `Booster::Dart` variant with tree weights
- [ ] DART-aware prediction (weighted tree contributions)
- [ ] XGBoost JSON: parse DART models

### Milestone 3.2: Categorical Features

- [ ] Categorical split condition (bitset-based)
- [ ] `FeatureType::Categorical` in metadata
- [ ] XGBoost JSON: parse categorical splits

### Milestone 3.3: Block Traversal

Performance optimization from RFC-0003.

- [ ] `BlockVisitor` — process multiple rows together
- [ ] SIMD-friendly traversal (future)
- [ ] Benchmark vs row-at-a-time

### Milestone 3.4: Sparse Data

- [ ] `SparseMatrix` (CSR format, possibly via `sprs`)
- [ ] `DataMatrix` impl for sparse
- [ ] Sparse-aware traversal

---

## Phase 4: Native Serialization

### Milestone 4.1: Schema Types

- [ ] `ModelSchema`, `TreeSchema`, etc. (stable interchange)
- [ ] `Model` ↔ `ModelSchema` conversion

### Milestone 4.2: Binary Format

- [ ] `bincode` serialization of schema
- [ ] Magic bytes, version header
- [ ] `save()` / `load()` API

### Milestone 4.3: JSON Format

- [ ] `serde_json` serialization of schema
- [ ] Pretty-print for debugging

---

## Phase 5: Ecosystem Integration

### Milestone 5.1: Python Bindings (PyO3)

- [ ] `booste-rs-python` crate
- [ ] `Model` wrapper with `predict()` method
- [ ] NumPy array input support

### Milestone 5.2: Arrow Integration

- [ ] `ArrowMatrix` implementing `DataMatrix`
- [ ] Zero-copy from PyArrow/polars

### Milestone 5.3: CLI Tool

- [ ] `xgbrs` binary for model inspection
- [ ] Convert between formats
- [ ] Benchmark predictions

---

## Current Focus

```text
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Minimal Viable Prediction  ✅ COMPLETE                │
│  ═══════════════════════════════════                            │
│                                                                  │
│  [x] 1.1 Core Types                                             │
│  [x] 1.2 Tree Storage                                           │
│  [x] 1.3 Forest                                                 │
│  [x] 1.4 XGBoost JSON Loader                                    │
│  [x] 1.5 Simple Prediction                                      │
│                                                                  │
│  PHASE 2: Full Inference Pipeline ◄── IN PROGRESS               │
│  ════════════════════════════════                               │
│                                                                  │
│  [x] 2.1 DataMatrix Trait                                       │
│  [ ] 2.2 Predictor & Visitor ◄── NEXT                           │
│  [ ] 2.3 Model Wrapper                                          │
│  [ ] 2.4 Objective Transforms                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Test Data Strategy

For validation, we need reference predictions from Python XGBoost.

### Test Models

1. **Simple regression** — 10 trees, 5 features, no missing
2. **Binary classification** — logistic objective
3. **Multiclass** — 3+ classes, softmax
4. **With missing values** — NaN handling
5. **DART** — dropout trees with weights
6. **Categorical** — native categorical splits

### Generation Script

```bash
# tools/data_generation/
uv run python scripts/generate_test_cases.py
```

Generates per test case: `{name}.model.json`, `{name}.input.json`, `{name}.expected.json`

---

## Dependencies

### Phase 1 (Minimal)

```toml
[dependencies]
thiserror = "1.0"

[dependencies.serde]
version = "1.0"
optional = true

[dependencies.serde_json]
version = "1.0"
optional = true

[features]
default = []
xgboost-compat = ["dep:serde", "dep:serde_json"]
```

### Added in Later Phases

- `bincode` — native binary format (Phase 4)
- `sprs` — sparse matrices (Phase 3)
- `arrow` — Arrow integration (Phase 5)
- `pyo3` — Python bindings (Phase 5)

---

## Notes

- **Don't over-engineer early**: Get something working, then refactor
- **Test against Python**: Every milestone should have a validation test
- **Feature flags**: Keep optional stuff behind features from the start
- **Document as you go**: Update RFCs if implementation diverges
