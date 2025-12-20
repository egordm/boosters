# Model API and Python Bindings Backlog

**Source**: [RFC-0019: Python Bindings & Model Abstraction](../rfcs/0019-python-bindings.md)  
**Created**: 2025-12-19  
**Status**: ✅ COMPLETE (Phase 1 MVP) - 2025-12-20

---

## Overview

This backlog implements the high-level model abstraction layer in Rust and Python bindings via PyO3.

**Completed**: Epic 2 (Model API), Epic 3 (Python Bindings Stories 3.1-3.5)  
**Deferred**: Epic 3b (Converters), Story 3.6 (Sklearn Wrappers)

**Dependencies**: [Epic 1: Storage Format](05-storage-format.md) (for save/load) ✅  
**Enables**: [Epic 4: Explainability](07-explainability.md) ✅

**Parallel Work Opportunities**:
- Epic 1 (Storage) + Epic 2.1-2.3 (Core API) can run in parallel
- Epic 2.4 (File I/O) blocks on Epic 1 completion
- Epic 3 (Python) blocks on Epic 2 completion  
- Epic 4 (Explainability) can start after Epic 2.3
- Phase 1b (Converters) blocks on Epic 3 completion

```
                    ┌─── Epic 2.1-2.3 ──┐
    Prerequisites ──┤                    ├──► Epic 2.4 ──► Epic 3 ──► Converters
                    └─── Epic 1 ────────┘         │
                                                  └──► Epic 4
```

**Estimated Total Effort**: ~10-14 hours (implementation) + ~3-4 hours (testing setup)

**CI Requirements**:
- Rust tests: `cargo test` (existing CI)
- Python tests: `pytest` with maturin-built package
- Converter tests: Require `xgboost` and `lightgbm` optional dependencies
- CI matrix should test with/without optional ML libraries

---

## Epic 2: High-Level Model API (Rust)

**Goal**: Create unified `GBDTModel` and `GBLinearModel` types that wrap training, prediction, and serialization.

**Why**: Current API requires users to manually compose trainers, predictors, and serialization. High-level types provide a simple, type-safe interface.

---

### Story 2.1: Implement Parameter Types

**RFC Section**: RFC-0019 "Objectives and Metrics", "Common Training Parameters"  
**Effort**: M (1-2h)

**Description**: Create the hierarchical parameter structs and objective/metric enums.

**Tasks**:

- [ ] 2.1.1 Create `src/training/params.rs` module
- [ ] 2.1.2 Implement `ObjectiveKind` enum with all variants from RFC:
  - SquaredError, AbsoluteError, Huber, Quantile, Tweedie
  - BinaryLogistic, BinaryHinge
  - MultiSoftmax, MultiOneVsAll
- [ ] 2.1.3 Implement `ObjectiveKind::from_str()` for XGBoost-compatible parsing
- [ ] 2.1.4 Implement `ObjectiveKind::is_classification()`, `is_regression()`
- [ ] 2.1.5 Implement `MetricKind` enum (Rmse, Mae, Mape, LogLoss, Auc, Accuracy, Ndcg)
- [ ] 2.1.6 Implement `CommonParams` struct with defaults from RFC
- [ ] 2.1.7 Implement `GBDTParams` struct with common + tree-specific params
- [ ] 2.1.8 Implement `GBLinearParams` struct with common + linear-specific params
- [ ] 2.1.9 Implement `Default` for all param types
- [ ] 2.1.10 Add `#[derive(Clone, Debug, Serialize, Deserialize)]` for persistence

**Definition of Done**:

- All param types compile with Serde derives
- Defaults match RFC-0019 specification
- ObjectiveKind parsing handles XGBoost prefixes

**Testing Criteria**:

- `ObjectiveKind::from_str("reg:squared_error")` returns `SquaredError`
- `ObjectiveKind::from_str("binary:logistic")` returns `BinaryLogistic`
- Params serialize/deserialize round-trip correctly

---

### Story 2.2: Implement ModelMeta

**RFC Section**: RFC-0019 "Model Metadata"  
**Effort**: S (30min)

**Description**: Create shared metadata struct for model introspection.

**Tasks**:

- [ ] 2.2.1 Create `src/repr/model.rs` module
- [ ] 2.2.2 Implement `ModelMeta` struct:
  - feature_names: Option<Vec<String>>
  - feature_types: Option<Vec<FeatureType>>
  - n_features: usize
  - n_groups: usize
  - task: TaskKind
  - best_iteration: Option<usize>
- [ ] 2.2.3 Implement `FeatureType` enum (Numeric, Categorical { n_categories })
- [ ] 2.2.4 Implement `TaskKind` enum (Regression, BinaryClassification, MulticlassClassification, Ranking)
- [ ] 2.2.5 Add Serde derives for serialization

**Definition of Done**:

- ModelMeta captures all fields from RFC
- Serializable for inclusion in saved models

**Testing Criteria**:

- ModelMeta can be constructed with and without optional fields
- Round-trip serialization preserves all data

---

### Story 2.3: Implement GBDTModel

**RFC Section**: RFC-0019 "GBDTModel - Tree Ensemble Model"  
**Effort**: L (3-4h)

**Description**: High-level wrapper around Forest with training, prediction, and serialization.

**Tasks**:

- [ ] 2.3.1 Define `GBDTModel` struct in `src/repr/model.rs`:
  - forest: Forest<ScalarLeaf>
  - meta: ModelMeta
  - params: GBDTParams
  - optimized_layout: UnrolledLayout (always populated)
- [ ] 2.3.2 Implement `GBDTModel::train(data, labels, params)`:
  - Convert params to internal trainer config
  - Call existing GBDTTrainer
  - Populate ModelMeta from dataset (feature names, n_features, task type)
  - Call `UnrolledLayout::from_forest(&forest)` to optimize
- [ ] 2.3.3 Implement `predict(&self, features)` using optimized layout predictor
- [ ] 2.3.4 Implement `predict_margin(&self, features)` for raw output
- [ ] 2.3.5 Implement `n_trees()`, `n_features()` accessors
- [ ] 2.3.6 Implement basic `feature_importance()` returning split counts per feature
- [ ] 2.3.7 Implement `save(path)`, `load(path)` using Epic 1 codec
- [ ] 2.3.8 Implement `to_bytes()`, `from_bytes()` for in-memory serialization
- [ ] 2.3.9 In `load()` and `from_bytes()`: call `UnrolledLayout::from_forest()` after deserializing
- [ ] 2.3.10 Add comprehensive rustdoc with examples

**Note**: Advanced explainability (gain-based importance, SHAP values) is added in [Epic 4: Explainability](07-explainability.md). This story provides only basic split-count importance.

**Definition of Done**:

- Training produces working model
- Predictions match existing trainer output
- Save/load preserves predictions exactly
- Layout is always optimized (no manual call needed)

**Testing Criteria**:

- Train model → GBDTModel predictions match raw GBDTTrainer predictions
- Optimized layout predictions == simple traversal predictions (correctness check)
- Save → load → predict matches original predictions
- `n_trees()` returns correct count
- Feature names are captured if provided in dataset

---

### Story 2.4: Implement GBLinearModel

**RFC Section**: RFC-0019 "GBLinearModel - Linear Booster Model"  
**Effort**: M (1-2h)

**Description**: High-level wrapper for linear booster with same API pattern as GBDTModel.

**Tasks**:

- [ ] 2.4.1 Define `GBLinearModel` struct:
  - weights: Box<[f32]>
  - bias: Box<[f32]>
  - meta: ModelMeta
  - params: GBLinearParams
- [ ] 2.4.2 Implement `train(data, labels, params)` using GBLinearTrainer
- [ ] 2.4.3 Implement `predict(&self, features)`
- [ ] 2.4.4 Implement `weights()`, `bias()`, `n_features()` accessors
- [ ] 2.4.5 Implement `save(path)`, `load(path)`
- [ ] 2.4.6 Implement `to_bytes()`, `from_bytes()`
- [ ] 2.4.7 Add rustdoc with examples

**Definition of Done**:

- Training produces working model
- Predictions match existing linear trainer output
- Save/load works correctly

**Testing Criteria**:

- Train → predict matches raw trainer
- Save → load → predict matches
- Weights and bias are accessible

---

## Epic 3: Python Bindings

**Goal**: Create PyO3 bindings for training and prediction from Python.

**Why**: Python is the primary ML ecosystem. Without Python bindings, adoption is blocked.

---

### Story 3.1: Setup Python Crate

**Effort**: S (30min)

**Description**: Create the boosters-python workspace crate with PyO3.

**Tasks**:

- [ ] 3.1.1 Create `boosters-python/` directory in workspace
- [ ] 3.1.2 Create `boosters-python/Cargo.toml`:
  - name = "boosters-python"
  - crate-type = ["cdylib"]
  - Dependencies: pyo3, numpy, boosters (path)
- [ ] 3.1.3 Add to workspace Cargo.toml members
- [ ] 3.1.4 Create basic `src/lib.rs` with `#[pymodule]`
- [ ] 3.1.5 Export `__version__` string
- [ ] 3.1.6 Create `pyproject.toml` for maturin build
- [ ] 3.1.7 Verify `maturin develop` works and module imports

**Definition of Done**:

- `import boosters` works in Python
- `boosters.__version__` returns correct version

**Testing Criteria**:

- `maturin develop` succeeds
- Python import works
- Version string matches Cargo.toml

---

### Story 3.2: Implement Dataset Wrapper

**RFC Section**: RFC-0019 "Dataset - Zero-Copy Data Wrapper"  
**Effort**: M (1-2h)

**Description**: Python class that wraps NumPy arrays with zero-copy semantics.

**Tasks**:

- [ ] 3.2.1 Create `PyDataset` struct in `boosters-python/src/dataset.rs`
- [ ] 3.2.2 Implement `new(data, label, weight, feature_names)`:
  - Accept `PyReadonlyArray2<f32>` for data
  - Accept `PyReadonlyArray1<f32>` for label/weight
  - Keep references to prevent GC
- [ ] 3.2.3 Handle dtype conversion:
  - f32: zero-copy (ideal)
  - f64, int32, int64: convert to f32 once with warning
  - Other dtypes: raise TypeError
- [ ] 3.2.4 Handle input types:
  - np.ndarray: accept (with dtype handling above)
  - pd.DataFrame: accept, extract .values, infer feature_names from columns
  - Python list: reject with TypeError "Use np.array() instead of Python lists"
  - Masked arrays: reject with "Masked arrays not supported; fill missing values first"
- [ ] 3.2.5 Implement `n_rows`, `n_cols`, `feature_names` properties
- [ ] 3.2.6 Implement `validate_for_training(params)` method
- [ ] 3.2.7 Add to pymodule exports

**Definition of Done**:

- Dataset wraps NumPy arrays
- Properties return correct values
- Lists are rejected with helpful error

**Testing Criteria**:

- Create Dataset from NumPy array: works
- Create Dataset from list: raises TypeError with message
- Zero-copy verification: memory shouldn't double for large arrays (manual check)
- n_rows, n_cols match input array shape

---

### Story 3.3: Implement Parameter Bindings

**Effort**: M (1h)

**Description**: Expose parameter classes to Python.

**Tasks**:

- [ ] 3.3.1 Create `boosters-python/src/params.rs`
- [ ] 3.3.2 Implement `PyCommonParams` with all fields from Rust
- [ ] 3.3.3 Implement `PyGBDTParams` with common + tree params
- [ ] 3.3.4 Implement `PyGBLinearParams` with common + linear params
- [ ] 3.3.5 Implement `quick(**kwargs)` classmethod for flat construction
- [ ] 3.3.6 Implement `for_regression(**kwargs)` and `for_classification(n_classes, **kwargs)` factories
- [ ] 3.3.7 Add to pymodule exports

**Definition of Done**:

- All param classes accessible from Python
- Factory methods create appropriate configs
- Flat kwargs work for convenience

**Testing Criteria**:

- `GBDTParams()` creates default params
- `GBDTParams.quick(learning_rate=0.1, max_depth=5)` works
- `GBDTParams.for_classification(n_classes=3)` sets correct objective

---

### Story 3.4: Implement GBDTBooster

**RFC Section**: RFC-0019 "GBDTBooster - Tree Model"  
**Effort**: L (2-3h)

**Description**: Main training/prediction class for Python.

**Tasks**:

- [ ] 3.4.1 Create `boosters-python/src/gbdt.rs`
- [ ] 3.4.2 Implement `PyGBDTBooster` struct wrapping `GBDTModel`
- [ ] 3.4.3 Implement `train(params, train_data, eval_data=None)` classmethod:
  - Extract data from PyDataset
  - Call Rust GBDTModel::train
  - Return new PyGBDTBooster
- [ ] 3.4.4 Implement `predict(data, output_margin=False)`:
  - Accept Dataset or raw NumPy array
  - Return NumPy array of predictions
- [ ] 3.4.5 Implement `feature_importance()` returning dict (feature name → split count)
- [ ] 3.4.6 Implement `save(path)` and `load(path)`
- [ ] 3.4.7 Implement `to_bytes()` and `from_bytes(data)`
- [ ] 3.4.8 Implement `__getstate__` and `__setstate__` for pickle
- [ ] 3.4.9 Implement properties: `n_trees`, `n_features`, `best_iteration`, `params`
- [ ] 3.4.10 Add to pymodule exports

**Note**: SHAP values and gain-based importance are added in [Epic 4: Explainability](07-explainability.md).

**Definition of Done**:

- Training works from Python
- Predictions return NumPy arrays
- Save/load works
- Pickle works

**Testing Criteria**:

- Train on California Housing dataset → RMSE within 10% of XGBoost default
- Save → load → predict matches original predictions exactly
- `pickle.dumps(model)` → `pickle.loads()` produces identical predictions
- Properties return correct values (`n_trees`, `n_features`, `params`)
- Training with early_stopping sets `best_iteration` property

---

### Story 3.5: Implement GBLinearBooster

**Effort**: M (1-2h)

**Description**: Python binding for linear booster.

**Tasks**:

- [ ] 3.5.1 Create `boosters-python/src/gblinear.rs`
- [ ] 3.5.2 Implement `PyGBLinearBooster` wrapping `GBLinearModel`
- [ ] 3.5.3 Implement `train()`, `predict()`, `save()`, `load()`
- [ ] 3.5.4 Implement `weights`, `bias` properties returning NumPy arrays
- [ ] 3.5.5 Implement pickle support
- [ ] 3.5.6 Add to pymodule exports

**Definition of Done**:

- Training works
- Weights/bias accessible as NumPy arrays
- Save/load and pickle work

**Testing Criteria**:

- Train → predict works
- `model.weights` has correct shape
- Pickle round-trip works

---

### Story 3.6: Implement Sklearn Wrappers (Phase 2)

**RFC Section**: RFC-0019 "Sklearn-Compatible Wrappers"  
**Effort**: M (2h)  
**Status**: Phase 2 (after MVP)

**Description**: Sklearn-compatible estimators for pipeline integration.

**Tasks**:

- [ ] 3.6.1 Create `boosters/sklearn.py` pure Python module
- [ ] 3.6.2 Implement `GBDTRegressor(BaseEstimator, RegressorMixin)`:
  - Flat kwargs matching sklearn conventions
  - `fit(X, y)`, `predict(X)`, `feature_importances_`
- [ ] 3.6.3 Implement `GBDTClassifier(BaseEstimator, ClassifierMixin)`:
  - Auto-detect n_classes from y
  - `predict()` returns class labels, `predict_proba()` returns probabilities
- [ ] 3.6.4 Implement `GBLinearRegressor` and `GBLinearClassifier`
- [ ] 3.6.5 Add sklearn to optional dependencies

**Definition of Done**:

- Works with sklearn cross_val_score
- Works with sklearn Pipeline
- GridSearchCV finds correct parameters

**Testing Criteria**:

- `cross_val_score(GBDTRegressor(), X, y, cv=3)` works
- `Pipeline([('model', GBDTClassifier())]).fit(X, y)` works
- `clone(model)` works correctly

---

## Epic 3b: Python Converters

**Goal**: Provide Python utilities to convert XGBoost/LightGBM models to native format.

**Dependencies**: Story 3.4 (GBDTBooster must exist to save converted models)

**Note**: The converters extract model structure using XGBoost/LightGBM introspection APIs, then construct a `GBDTBooster` internally and save it.

---

### Story 3b.1: Implement xgboost_to_bstr Converter

**RFC Section**: RFC-0021 "Python Conversion Utilities"  
**Effort**: M (1-2h)

**Description**: Convert XGBoost Booster to native .bstr format using introspection APIs.

**Tasks**:

- [ ] 3b.1.1 Create `boosters/convert.py` module
- [ ] 3b.1.2 Implement `xgboost_to_bstr(model, path, include_feature_names=True)`:
  - Use `model.trees_to_dataframe()` for tree structure
  - Use `model.get_config()` for objective, base_score
  - Extract feature names from model
- [ ] 3b.1.3 Handle gbtree, dart, and gblinear boosters
- [ ] 3b.1.4 Document conversion workflow in docstring
- [ ] 3b.1.5 Add xgboost to optional dependencies

**Definition of Done**:

- XGBoost models convert to .bstr
- Loaded .bstr produces identical predictions

**Testing Criteria**:

- Train XGBoost model → convert → load → predict matches XGBoost predictions (rtol=1e-6)
- DART model converts correctly
- GBLinear model converts correctly

---

### Story 3b.2: Implement lightgbm_to_bstr Converter

**Effort**: M (1-2h)

**Description**: Convert LightGBM Booster to native format.

**Tasks**:

- [ ] 3b.2.1 Implement `lightgbm_to_bstr(model, path, include_feature_names=True)`:
  - Use `model.trees_to_dataframe()` for tree structure
  - Use `model.dump_model()` for linear tree coefficients
  - Use `model.params` for configuration
- [ ] 3b.2.2 Handle standard GBDT and linear trees
- [ ] 3b.2.3 Handle categorical features
- [ ] 3b.2.4 Add lightgbm to optional dependencies

**Definition of Done**:

- LightGBM models convert to .bstr
- Linear trees preserve coefficients

**Testing Criteria**:

- Train LightGBM → convert → load → predict matches (rtol=1e-6)
- Linear tree model converts with correct predictions

---

### Story 3b.3: Document Conversion Workflow

**Effort**: S (30min)

**Description**: Add documentation for model conversion.

**Tasks**:

- [ ] 3b.3.1 Add "Converting from XGBoost/LightGBM" section to README
- [ ] 3b.3.2 Add example notebook demonstrating conversion workflow
- [ ] 3b.3.3 Document that converters require respective library installed

**Definition of Done**:

- README has conversion examples
- Workflow is clear for users

---

### Converter Test Corpus

Converter tests require example model files:
- `tests/fixtures/converters/xgb_binary_model.json` - XGBoost JSON format
- `tests/fixtures/converters/xgb_ubj_model.ubj` - XGBoost UBJ format  
- `tests/fixtures/converters/lgb_model.txt` - LightGBM text format

Generate using: `python tools/generate_converter_corpus.py`

---

## Summary

### Story Order (Phases)

**Phase 1: Core (MVP)**

| Order | Story | Effort | Blocked By |
|-------|-------|--------|------------|
| 1 | 2.1 Parameter Types | M | None |
| 2 | 2.2 ModelMeta | S | None |
| 3 | 2.3 GBDTModel | M | 2.1, 2.2, Epic 1 |
| 4 | 2.4 GBLinearModel | M | 2.1, 2.2, Epic 1 |
| 5 | 3.1 Python Crate Setup | S | None |
| 6 | 3.2 Dataset Wrapper | M | 3.1 |
| 7 | 3.3 Parameter Bindings | M | 3.1, 2.1 |
| 8 | 3.4 GBDTBooster | L | 3.2, 3.3, 2.3 |
| 9 | 3.5 GBLinearBooster | M | 3.2, 3.3, 2.4 |

**Phase 1b: Converters (after Phase 1 MVP)**

| Story | Effort | Blocked By |
|-------|--------|------------|
| 3b.1 xgboost_to_bstr | M | 3.4 (GBDTBooster) |
| 3b.2 lightgbm_to_bstr | M | 3.4, 3.5 (need GBLinearBooster for gblinear) |
| 3b.3 Documentation | S | 3b.1, 3b.2 |

**Phase 2: Sklearn Integration**

| Story | Effort | Blocked By |
|-------|--------|------------|
| 3.6 Sklearn Wrappers | M | 3.4, 3.5 |

### Deferred to Post-MVP

| Item | Reason |
|------|--------|
| Callbacks | Not critical for basic usage |
| ObjectiveConfig for parameterized objectives | Can use defaults initially |
| Arrow support | Optimization, not essential |
| CLI for conversion | Python API sufficient |

### Verification Checklist

After all Phase 1 stories complete:

- [ ] `cargo test` in boosters crate — all tests pass
- [ ] `maturin develop` — Python package builds
- [ ] `pytest tests/` — Python tests pass
- [ ] Training from Python produces reasonable predictions
- [ ] Save/load round-trip works
- [ ] Pickle works
- [ ] XGBoost conversion produces matching predictions
- [ ] Documentation builds

**Epic 2 Complete When**:
- [ ] `GBDTModel.predict()` works on California Housing dataset
- [ ] `model.save()` / `Model.load()` roundtrip produces identical predictions
- [ ] `model.feature_importance()` returns split counts
- [ ] All public API is documented

**Epic 3 Complete When**:
- [ ] `pip install boosters` works (from wheel)
- [ ] `import boosters; b = boosters.GBDTBooster()` works
- [ ] Python can train on numpy arrays and pandas DataFrames
- [ ] `pickle.dumps(booster)` / `pickle.loads()` works
- [ ] Example notebook runs end-to-end

**Converters Complete When**:
- [ ] `boosters.from_xgboost(xgb_model)` produces matching predictions
- [ ] `boosters.from_lightgbm(lgb_model)` produces matching predictions
- [ ] Converter test corpus exists and is used in CI

### Test File Locations

| Test Type | Location |
|-----------|----------|
| Rust Model API unit tests | Inline in `src/repr/model.rs` |
| Rust params tests | Inline in `src/training/params.rs` |
| Python Dataset tests | `boosters-python/tests/test_dataset.py` |
| Python training tests | `boosters-python/tests/test_training.py` |
| Python pickle tests | `boosters-python/tests/test_serialization.py` |
| Converter tests | `boosters-python/tests/test_convert.py` |

### Cross-Epic Integration Test

After Epics 1, 2, 3, 4 are all complete, verify this end-to-end workflow:

```python
# Train in Python
model = GBDTBooster.train(params, dataset)

# Save to .bstr format
model.save("model.bstr")

# Load and verify predictions unchanged
loaded = GBDTBooster.load("model.bstr")
assert np.allclose(model.predict(X), loaded.predict(X))

# Compute SHAP values
shap_values = loaded.shap_values(X[:10])
base = loaded.expected_value
preds = loaded.predict(X[:10])
assert np.allclose(shap_values.sum(axis=1) + base, preds, atol=1e-6)

# Verify SHAP compatible with shap library
import shap
shap.summary_plot(shap_values, X[:10])  # Should not crash
```

This test lives in `boosters-python/tests/test_e2e.py`.

---

**Previous Epic**: [Storage Format](05-storage-format.md)  
**Next Epic**: [Explainability](07-explainability.md)

---

**Document Status**: Ready for Implementation  
**Reviewed By**: PO, Architect, Senior Engineer, QA Engineer
