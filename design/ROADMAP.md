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

- [x] `TreeVisitor` trait
- [x] `ScalarVisitor` — simple row-at-a-time prediction
- [x] `Predictor` struct wrapping forest + visitor
- [x] `PredictionOutput` flat buffer with shape metadata
- [x] Wire up `DataMatrix` input

**Files**: `src/predict/mod.rs`, `src/predict/visitor.rs`, `src/predict/output.rs`

### Milestone 2.3: Model Wrapper

Complete `Model` type from RFC-0007.

- [x] `Model` struct (booster, meta, features, objective)
- [x] `Booster` enum (Tree and Dart variants)
- [x] `ModelMeta`, `FeatureInfo`, `Objective` types
- [x] `Model::predict()` high-level API

**Files**: `src/model.rs`

### Milestone 2.4: Objective Transforms

Post-prediction transformations.

- [x] `Objective` enum with common objectives
- [x] `Objective::transform()` — apply sigmoid, softmax, etc.
- [x] Wire into `Model::predict()`

**Files**: `src/objective.rs`

### Milestone 2.5: Model Integration Tests

Validate full pipeline against Python XGBoost.

- [x] Test `Model::predict()` for regression models
- [x] Test `Model::predict()` for binary classification (sigmoid)
- [x] Test `Model::predict()` for multiclass (softmax)
- [x] Generate test data with Python script

**Files**: `tests/model_integration.rs`, `tools/data_generation/`

### ✅ Phase 2 Complete

**Deliverable**: Clean public API: `Model::load()` + `Model::predict()`.

**Validation**: Binary classification, multiclass models work correctly.

---

## Phase 3: Performance Optimization

**Goal**: Match or beat XGBoost C++ on batch prediction benchmarks.

**Context**: We already win on single-row latency (4.9x faster). XGBoost's batch
advantage comes from ArrayTreeLayout + SIMD. This phase closes that gap.

### ✅ Milestone 3.1: DART Support

- [x] `Booster::Dart` variant with tree weights (implemented in M2.3)
- [x] DART-aware prediction (weighted tree contributions)
- [x] `XgbModel::to_booster()` returns proper `Booster::Dart` with weights
- [x] `XgbModel::is_dart()` helper method
- [x] XGBoost JSON: parse DART models (already working)

**Implementation**: `Predictor::predict_weighted()` applies per-tree weights
during accumulation, matching XGBoost C++ DART inference behavior.

**Files**: `src/predict/visitor.rs`, `src/model.rs`, `src/compat/xgboost/convert.rs`

### ✅ Milestone 3.2: Categorical Features

- [x] `SplitType` enum: `Numeric` / `Categorical`
- [x] `CategoriesStorage` for bitset-based categorical splits (in `src/trees/categories.rs`)
- [x] `categories_to_bitset()` helper for building bitsets from category values
- [x] `float_to_category()` with debug validation for safe f32→u32 conversion
- [x] `TreeBuilder::add_categorical_split()` with bitset
- [x] `SoATreeStorage` / `SoATreeView` categorical accessors
- [x] `ScalarVisitor` handles categorical splits
- [x] `FeatureType::Categorical` and `FeatureType::Quantitative` in metadata
- [x] XGBoost JSON: parse categorical splits (categories → bitset conversion)
- [x] Integration tests with XGBoost categorical models

**Implementation**: Categorical splits store categories as packed u32 bitsets
using bit operations (`>> 5` and `& 31` for efficient indexing). Categories
in the set go RIGHT, others go LEFT (matching XGBoost behavior). XGBoost JSON
stores category integer values which are converted to bitsets during loading.

**Files**: `src/trees/categories.rs`, `src/trees/node.rs`, `src/trees/soa.rs`,
`src/forest/soa.rs`, `src/predict/visitor.rs`, `src/compat/xgboost/json.rs`,
`src/compat/xgboost/convert.rs`

### ✅ Milestone 3.3: Benchmarking Infrastructure

Establish baseline performance measurements before optimization work.

- [x] Add `criterion` dev-dependency for benchmarks
- [x] Add `xgb` (XGBoost Rust bindings) as optional bench dependency
- [x] Create `benches/` directory with benchmark harness
- [x] Implement benchmark scenarios:
  - [x] Small model (10 trees, 5 features) - single row
  - [x] Medium model (100 trees, 50 features) - batch 1K rows
  - [x] Large model (500 trees, 100 features) - batch 10K rows
  - [x] Varying batch sizes (1, 10, 100, 1K, 10K rows)
- [x] Baseline: booste-rs vs `xgb` crate (XGBoost C++ via FFI)
- [x] Document baseline results in `docs/benchmarks/`

**Files**: `benches/prediction.rs`, `Cargo.toml`, `docs/benchmarks/`, `README.md`

**Key findings** (baseline, pre-optimization):

- **Single-row latency**: booste-rs ~860ns vs XGBoost C++ ~4.2µs (4.9x faster!)
- **Batch prediction**: XGBoost C++ significantly faster (SIMD-optimized)
- **Opportunity**: Block traversal (M3.4) should close the batch gap

**Notes**:

- Using `xgb` crate (Rust bindings to XGBoost C++) provides fair comparison
- Both load from same model file, predict on same data
- Avoids Python overhead issues
- `xgb` uses prebuilt XGBoost libs, requires `libclang-dev` on Linux

### ✅ Milestone 3.4: Block Traversal

Performance optimization from RFC-0003.

- [x] `BlockPredictor` — process multiple rows in blocks of 64
- [x] Block-based feature loading for cache locality
- [x] Benchmark vs row-at-a-time predictor

**Implementation**: `BlockPredictor` processes rows in blocks (default 64),
loading features into a buffer and processing all trees per block. This keeps
tree nodes in L1/L2 cache while processing multiple rows.

**Benchmark results** (bench_medium model):

- 100 rows: 30% faster (87.6µs → 61.2µs)
- 1,000 rows: 7% faster (2.18ms → 2.04ms)
- 10,000 rows: 7% faster (22.0ms → 20.5ms)

**Files**: `src/predict/block.rs`, `benches/prediction.rs`

### Milestone 3.5: ArrayTreeLayout

Unroll top tree levels into flat arrays for cache-friendly level-by-level traversal.
This is XGBoost's key batch optimization.

- [ ] `ArrayTreeLayout` struct — flatten top K levels (typically 6-8)
- [ ] Level-by-level traversal instead of pointer-chasing
- [ ] `ArrayTreePredictor` using the new layout
- [ ] Conversion from `SoATreeStorage` to `ArrayTreeLayout`
- [ ] Benchmark: target 2x improvement on 1K+ row batches

**Theory**: A complete binary tree of depth K has `2^K - 1` nodes. Unrolling
the top 6 levels gives 63 nodes in a contiguous array. For each row, we can
traverse these 6 levels with simple index arithmetic (`2*i+1`, `2*i+2`) before
falling back to the regular tree for deeper nodes.

**Files**: `src/trees/array_layout.rs`, `src/predict/array.rs`

### Milestone 3.6: SIMD Traversal

Process multiple rows simultaneously using SIMD instructions.

- [ ] Research: portable SIMD (`std::simd` or `wide` crate)
- [ ] `SimdVisitor` — process 4/8 rows in parallel with AVX2/AVX-512
- [ ] SIMD-friendly comparison operations
- [ ] Fallback for non-SIMD platforms
- [ ] Benchmark: target 2-4x improvement on batch prediction

**Theory**: With ArrayTreeLayout, all rows at the same tree level can be
processed together. SIMD lets us compare 4-8 thresholds simultaneously and
compute the next node index for all rows in one instruction.

**Files**: `src/predict/simd.rs`

### Milestone 3.7: Memory Prefetching

Hint CPU about upcoming memory accesses to reduce cache misses.

- [ ] Research: `core::arch` prefetch intrinsics
- [ ] Prefetch next tree nodes during traversal
- [ ] Prefetch feature data for upcoming rows
- [ ] Benchmark: target 10-30% improvement

**Files**: `src/predict/prefetch.rs` or inline in existing predictors

### Milestone 3.8: Performance Validation

Comprehensive benchmarking to validate optimization gains.

- [ ] Re-run all benchmarks with optimizations enabled
- [ ] Compare against XGBoost C++ across all batch sizes
- [ ] Profile to identify remaining bottlenecks
- [ ] Document final performance characteristics
- [ ] Update README with benchmark results

**Success criteria**: Match or beat XGBoost C++ on batch prediction for
1K-10K row batches while maintaining single-row latency advantage.

---

## Phase 4: Compatibility & Data Formats

### Milestone 4.1: Sparse Data

- [ ] `SparseMatrix` (CSR format, possibly via `sprs`)
- [ ] `DataMatrix` impl for sparse
- [ ] Sparse-aware traversal (skip missing features)
- [ ] Benchmark sparse vs dense for high-sparsity data

**Files**: `src/data/sparse.rs`

### Milestone 4.2: Native Serialization

- [ ] `ModelSchema`, `TreeSchema`, etc. (stable interchange)
- [ ] `Model` ↔ `ModelSchema` conversion
- [ ] `bincode` serialization with magic bytes, version header
- [ ] `save()` / `load()` API

### Milestone 4.3: Additional XGBoost Formats

- [ ] XGBoost binary format (.bin) loader
- [ ] XGBoost UBJSON format loader
- [ ] LightGBM model loader (stretch goal)

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
│  PHASE 2: Full Inference Pipeline  ✅ COMPLETE                   │
│  ════════════════════════════════                               │
│                                                                  │
│  [x] 2.1 DataMatrix Trait                                       │
│  [x] 2.2 Predictor & Visitor                                    │
│  [x] 2.3 Model Wrapper                                          │
│  [x] 2.4 Objective Transforms                                   │
│  [x] 2.5 Model Integration Tests                                │
│                                                                  │
│  PHASE 3: Advanced Features                                      │
│  ═════════════════════════                                      │
│                                                                  │
│  [x] 3.1 DART Support                                           │
│  [x] 3.2 Categorical Features                                   │
│  [x] 3.3 Benchmarking Infrastructure                            │
│  [x] 3.4 Block Traversal                                        │
│  [ ] 3.5 ArrayTreeLayout          ◄── NEXT                      │
│  [ ] 3.6 SIMD Traversal                                         │
│  [ ] 3.7 Memory Prefetching                                     │
│  [ ] 3.8 Performance Validation                                 │
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

- `wide` or `std::simd` — SIMD operations (Phase 3)
- `sprs` — sparse matrices (Phase 4)
- `bincode` — native binary format (Phase 4)
- `arrow` — Arrow integration (Phase 5)
- `pyo3` — Python bindings (Phase 5)

---

## Notes

- **Don't over-engineer early**: Get something working, then refactor
- **Test against Python**: Every milestone should have a validation test
- **Feature flags**: Keep optional stuff behind features from the start
- **Document as you go**: Update RFCs if implementation diverges
