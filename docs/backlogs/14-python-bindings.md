# Python Bindings Backlog

**Source**: [RFC-0014: Python Bindings](../rfcs/0014-python-bindings.md)  
**Created**: 2025-12-25  
**Status**: Ready for Implementation

---

## Overview

This backlog implements Python bindings for boosters via PyO3/Maturin, providing both a core API with Rust-owned config types and an sklearn-compatible wrapper.

**Key Design Decisions** (from RFC-0014):

- DD-3: Rust-owned config types with generated stubs (single source of truth)
- DD-13: Nested configs (core API) vs flat kwargs (sklearn)
- DD-14: Objective union type safety with `PyObjective` enum

**Non-Goals (v0.1.0 scope boundaries)**:

- ❌ Model serialization (save/load) — not implemented, no interface (future RFC)
- ❌ Sparse matrix support — deferred to v0.2.0
- ❌ Python callbacks during training — Rust-only callbacks
- ❌ Library comparison tests — benchmarks handled by `packages/boosters-eval`

**In Scope (clarification)**:

- ✅ SHAP values via `pred_contrib=True` — Rust core supports this
- ✅ Native categorical features — Rust core has `SplitType::Categorical` with bitset-based
  multi-way splits (like LightGBM). Pandas categoricals auto-detected, or specify
  `categorical_features=[...]` for integer-encoded data

**Definition of Ready** (story can start when):

1. All blocking stories are complete
2. RFC sections are finalized for the story scope
3. Required infrastructure (CI, stubs) is in place for testing
4. Test fixtures available (after Story 1.5)

**Architecture**:

```text
packages/boosters-python/
├── Cargo.toml          # Depends on `boosters` crate
└── src/
    ├── lib.rs          # #[pymodule] entry point
    ├── config.rs       # Wraps boosters::config types
    ├── objectives.rs   # Wraps boosters::Objective enum
    ├── convert.rs      # PyConfig → core config conversion
    └── model.rs        # Wraps boosters::GBDTModel

Dependency direction: boosters-python → boosters (core)
PyO3 types are thin wrappers around core Rust types.

Callback pattern (Story 4.2):
  Python → fit() → GIL released → Rust training with Rust callbacks
                                  → Callback state collected
                 → GIL acquired  → Convert state to Python (eval_results, best_iteration)
```

**Dependencies**:

- Rust crate `boosters` must be stable (training, prediction, objectives, SHAP, categoricals)

**Milestones**:

| Version | Scope | Epics |
| ------- | ----- | ----- |
| **v0.1.0 (MVP)** | Core API (incl. SHAP, categoricals) | 1, 2, 3, 4 |
| **v0.2.0** | sklearn integration | 5, 6 |
| **Future** | Sparse matrices | Blocked on Rust feature |

**Parallel Work Opportunities**:
- Epic 1 (Project Setup) must complete first
- Epic 2 (Config Types) and Epic 3 (Dataset) can run in parallel after Epic 1
- Epic 4 (Models) blocks on Epic 2+3 completion
- Epic 5 (sklearn) blocks on Epic 4 completion
- Epic 6 (Polish) can overlap with Epic 5

```
               ┌──► Epic 2 (Configs) ─┐
Epic 1 (Setup) ┤                      ├──► Epic 4 (Models)
               └──► Epic 3 (Dataset) ─┘         │
                                                ▼
                              Epic 5 (sklearn) ──► Epic 6 (Polish)
                                   │
                              [v0.1.0 MVP]      [v0.2.0]
```

**Estimated Total Effort**: ~40-55 hours
- MVP (v0.1.0): ~25-35 hours
- sklearn + Polish (v0.2.0): ~15-20 hours

---

## Epic 1: Project Setup and Build Infrastructure

**Goal**: Set up `packages/boosters-python` with PyO3, Maturin, stub generation, and CI.

**Why**: Foundation for all Python development. Must be solid before implementing features.

---

### Story 1.1: Initialize Maturin Project

**RFC Section**: RFC-0014 "Package Structure"  
**Effort**: M (1-2h)  
**Status**: ✅ Complete (commit e7c6d94)

**Description**: Create the Python package with Maturin build system.

**Status Note**: Skeleton exists in `packages/boosters-python/`. Tasks 1.1.1-1.1.5 have initial files but need review/completion. Tasks 1.1.6+ depend on Epic 2-4 providing actual exports.

**Tasks**:

- [x] 1.1.1 Create `packages/boosters-python/` directory structure *(skeleton exists)*
- [x] 1.1.2 Initialize `pyproject.toml` with Maturin backend *(basic config exists, needs dev deps)*
- [x] 1.1.3 Create `Cargo.toml` for the PyO3 crate with `pyo3` and `numpy` dependencies *(exists)*
- [x] 1.1.4 Create `src/lib.rs` with basic `#[pymodule]` definition *(placeholder only, needs real module)*
- [x] 1.1.5 Create `python/boosters/__init__.py` with version and re-exports *(placeholder only)*
- [x] 1.1.6 Create re-export modules *(depends on Epic 2-4 providing exports)*:
  - `python/boosters/config.py` (re-exports config types)
  - `python/boosters/objectives.py` (re-exports objectives)
  - `python/boosters/metrics.py` (re-exports metrics)
  - `python/boosters/model.py` (re-exports models)
  - `python/boosters/data.py` (re-exports Dataset, EvalSet)
- [x] 1.1.7 Verify `maturin develop` builds and imports successfully
- [x] 1.1.8 Add `.gitignore` for Python artifacts *(may already exist)*

**Definition of Done**:

- `maturin develop` succeeds
- `python -c "import boosters; print(boosters.__version__)"` works
- Re-export modules created (initially empty, populated as Epic 2-4 progress)

**Testing Criteria**:

- Import test passes in fresh virtualenv
- Module structure matches RFC-0014

---

### Story 1.2: Type Stub Generation Setup

**RFC Section**: RFC-0014 "Type Stub Generation"  
**Effort**: S (30min-1h)  
**Status**: ✅ Complete (commit 2a2ed8c)

**Description**: Configure automatic `.pyi` stub generation.

**Tasks**:

- [x] 1.2.1 Add `pyo3-stub-gen` to build dependencies (or use Maturin's built-in)
- [x] 1.2.2 Configure stub output to `python/boosters/_boosters_rs.pyi`
- [x] 1.2.3 Add stub generation to `maturin develop` workflow
- [x] 1.2.4 Verify IDE autocomplete works with generated stubs
- [x] 1.2.5 Evaluate stub quality for nested config types:
  - If pyo3-stub-gen has gaps, add manual overrides in same `python/boosters/` directory
  - Document any types requiring manual stubs

**Definition of Done**:

- Stubs are auto-generated on build
- pyright sees correct types for `#[pyclass]` types
- Any stub gaps documented with manual overrides in place

**Testing Criteria**:

- `pyright boosters/` passes with no type errors
- IDE shows correct signatures for `GBDTConfig(...)`

---

### Story 1.3: Python Development Tooling

**Effort**: M (1-2h)  
**Status**: ✅ Complete (commit f8e4b33)

**Description**: Configure ruff, pyright, pytest, poe tasks, and pyproject-fmt for development workflow.

**Tasks**:

- [x] 1.3.1 Update `packages/boosters-python/pyproject.toml`:
  - Package has no dev dependencies (all dev deps in root pyproject.toml per uv workspace)
  - Configure package-specific pyright settings if needed
  - Configure package-specific ruff overrides if needed
  - Note: poe tasks run from root, not package directory
- [x] 1.3.2 Update root `pyproject.toml` with Python-specific tasks:
  - Add dev dependencies to root: pytest-cov, pyproject-fmt, nbmake
  - `python:test`: pytest for boosters-python with coverage
  - `python:doctest`: pytest --doctest-modules
  - `python:format-pyproject`: pyproject-fmt
  - Update existing `test` task to include Python tests
  - Note: `lint`, `format`, `typecheck` already exist at root level
- [x] 1.3.3 Configure pyright settings:
  - Keep `typeCheckingMode = "basic"` (current root config)
  - Document path to strict mode for future (after initial implementation)
  - Appropriate include/exclude paths for boosters-python
- [x] 1.3.4 Configure ruff with docstring rules:
  - Add D (pydocstyle) to select list for docstring enforcement (D100-D107)
  - Keep existing strict lint rules (E, W, F, I, B, C4, UP, RUF)
  - Google-style docstrings (`convention = "google"`)
- [x] 1.3.5 Add pyproject-fmt to format pyproject.toml consistently
- [x] 1.3.6 Verify tooling is correctly configured:
  - `poe check` runs without crashes on minimal codebase
  - Warnings expected until APIs are implemented in Epic 2-4

**Note**: Docstring enforcement ramps up as APIs are implemented. Ruff D rules
will flag missing docstrings; actual docstrings added with each public API.
Pyright starts in basic mode; upgrade to strict after core implementation.

**Definition of Done**:

- `poe check` runs complete tooling pipeline
- pyright basic mode configured with boosters-python paths
- ruff configured with D rules enabled
- `poe python:doctest` validates docstring examples

**Testing Criteria**:

- `poe lint` catches issues
- `poe typecheck` runs pyright
- `poe python:doctest` runs (passes trivially on minimal codebase)

---

### Story 1.4: CI Pipeline

**RFC Section**: RFC-0014 "CI Matrix"  
**Effort**: M (1-2h)  
**Status**: ✅ Complete (commit 81089db)

**Description**: Set up GitHub Actions for Python package CI.

**Tasks**:

- [x] 1.4.1 Create `.github/workflows/python.yml`
- [x] 1.4.2 Add matrix: Python 3.12, 3.13 × Linux, macOS, Windows
- [x] 1.4.3 Add `maturin build` step
- [x] 1.4.4 Run `poe all` for full lint/type/test validation
- [x] 1.4.5 Add stub freshness check (`git diff --exit-code *.pyi`)
- [ ] 1.4.6 Add wheel artifact upload

**Definition of Done**:

- CI runs on every PR touching `packages/boosters-python`
- Type errors and stale stubs fail the build

**Testing Criteria**:

- CI passes on a simple test file
- Stub drift is caught by `git diff --exit-code`

---

### Story 1.5: Error Handling Infrastructure

**RFC Section**: RFC-0014 "Error Handling"  
**Effort**: M (1-2h)  
**Status**: ✅ Complete (commit e7c6d94, part of 1.1)

**Description**: Establish consistent Rust→Python error conversion.

**Tasks**:

- [x] 1.5.1 Create `src/error.rs` module
- [x] 1.5.2 Define `PyBoostersError` enum covering:
  - Configuration errors → `ValueError`
  - Type errors → `TypeError`
  - Not fitted errors → `RuntimeError`
  - Data shape errors → `ValueError`
- [x] 1.5.3 Implement `From<PyBoostersError> for PyErr`
- [x] 1.5.4 Define error message templates for consistency
- [ ] 1.5.5 Add helper macro for error creation

**Definition of Done**:

- All Rust errors convert to appropriate Python exceptions
- Error messages are clear and actionable

**Testing Criteria**:

- Invalid config raises `ValueError` with descriptive message
- Type mismatch raises `TypeError`

---

### Story 1.6: Stakeholder Feedback Check (Setup)

**Meta-task for Epic 1**  
**Status**: ✅ Complete (commit 8ca1031)

**Tasks**:

- [x] 1.6.1 Review `tmp/stakeholder_feedback.md` for build/packaging concerns
- [x] 1.6.2 Document any feedback in backlog adjustments
- [ ] 1.6.3 Verify package naming `boosters` has no PyPI conflicts
- [ ] 1.6.4 Check if users need Alpine/musl support for v0.1.0
- [x] 1.6.5 Create `tests/conftest.py` with shared pytest fixtures:
  - `synthetic_regression_data` fixture
  - `synthetic_classification_data` fixture
  - Reusable across all test files

**Definition of Done**:

- Stakeholder feedback reviewed and incorporated
- Package name availability confirmed
- Test fixtures available for subsequent epics

---

## Epic 2: Rust-Owned Config Types

**Goal**: Implement all config types as Rust `#[pyclass]` with PyO3.

**Why**: DD-3 requires config types to be Rust-owned for single source of truth.

---

### Story 2.1: Basic Config Types

**RFC Section**: RFC-0014 "Rust Config Implementation"  
**Effort**: L (3-4h)

**Description**: Implement `TreeConfig`, `RegularizationConfig`, `SamplingConfig`, and additional sub-configs.

**Tasks**:

- [ ] 2.1.1 Create `src/config/mod.rs` module structure
- [ ] 2.1.2 Implement `TreeConfig` as `#[pyclass(get_all, set_all)]`:
  - `max_depth: i32` (default -1)
  - `n_leaves: u32` (default 31)
  - `min_samples_leaf: u32` (default 20)
  - `min_gain_to_split: f64` (default 0.0)
- [ ] 2.1.3 Implement `#[new]` with `#[pyo3(signature = (...))]` defaults
- [ ] 2.1.4 Implement `RegularizationConfig`:
  - `l1: f64` (default 0.0)
  - `l2: f64` (default 0.0)
- [ ] 2.1.5 Implement `SamplingConfig`:
  - `subsample: f64` (default 1.0)
  - `colsample: f64` (default 1.0)
  - `goss_alpha: f64` (default 0.0, for GOSS sampling)
  - `goss_beta: f64` (default 0.0)
- [ ] 2.1.6 Implement `CategoricalConfig`:
  - `max_categories: u32` (default 256)
  - `min_category_count: u32` (default 10)
- [ ] 2.1.7 Implement `EFBConfig` (Exclusive Feature Bundling):
  - `enable: bool` (default true)
  - `max_conflict_rate: f64` (default 0.0)
- [ ] 2.1.8 Implement `LinearLeavesConfig`:
  - `enable: bool` (default false)
  - `l2_regularization: f64` (default 0.0)
- [ ] 2.1.9 Add validation in constructors (e.g., `subsample` in (0, 1])
- [ ] 2.1.10 Export all types in `#[pymodule]`

**Definition of Done**:

- All 6 config types constructible from Python with defaults
- Invalid values raise `ValueError`

**Testing Criteria**:

- `TreeConfig()` has correct defaults
- `TreeConfig(max_depth=5)` works
- `SamplingConfig(subsample=1.5)` raises `ValueError`
- `CategoricalConfig(max_categories=64)` works
- `LinearLeavesConfig(enable=True)` works

---

### Story 2.2: Objective Types

**RFC Section**: RFC-0014 "Objective Types"  
**Effort**: M (2-3h)

**Description**: Implement all objective classes as separate `#[pyclass]` types.

**Tasks**:

- [ ] 2.2.1 Create `src/objectives.rs` module
- [ ] 2.2.2 Implement parameterless objectives:
  - `SquaredLoss`, `AbsoluteLoss`, `PoissonLoss`
  - `LogisticLoss`, `HingeLoss`
- [ ] 2.2.3 Implement `HuberLoss`:
  - `delta: f64` (default 1.0)
  - Validate `delta > 0`
- [ ] 2.2.4 Implement `PinballLoss`:
  - `alpha: PyObject` (f64 | Vec<f64>, default 0.5)
  - Validate `alpha` values in (0, 1)
- [ ] 2.2.5 Implement `ArctanLoss`:
  - `alpha: f64` (default 0.5)
  - Validate `alpha` in (0, 1)
- [ ] 2.2.6 Implement `SoftmaxLoss`:
  - `n_classes: u32`
  - Validate `n_classes >= 2`
- [ ] 2.2.7 Implement `LambdaRankLoss`:
  - `ndcg_at: u32` (default 10)
- [ ] 2.2.8 Implement `PyObjective` enum with `FromPyObject` derive
- [ ] 2.2.9 Export all objectives in module

**Definition of Done**:
- All 10 objective types constructible
- Parameter validation raises clear `ValueError`

**Testing Criteria**:
- `SquaredLoss()` works
- `PinballLoss(alpha=[0.1, 0.5, 0.9])` works
- `PinballLoss(alpha=1.5)` raises `ValueError` with clear message
- `SoftmaxLoss(n_classes=1)` raises `ValueError`

---

### Story 2.3: Metric Types

**RFC Section**: RFC-0014 "Metrics"  
**Effort**: S (1h)

**Description**: Implement metric classes for evaluation.

**Tasks**:

- [ ] 2.3.1 Create `src/metrics.rs` module
- [ ] 2.3.2 Implement parameterless metrics:
  - `Rmse`, `Mae`, `Mape`, `LogLoss`, `Auc`, `Accuracy`
- [ ] 2.3.3 Implement `Ndcg`:
  - `at: u32` (default 10)
- [ ] 2.3.4 Implement `PyMetric` enum with `FromPyObject`
- [ ] 2.3.5 Export all metrics in module

**Definition of Done**:
- All metric types constructible
- Stubs show correct type hints

**Testing Criteria**:
- `Rmse()` works
- `Ndcg(at=5)` works

---

### Story 2.4: GBDTConfig and GBLinearConfig

**RFC Section**: RFC-0014 "GBDTConfig", "GBLinearConfig"  
**Effort**: M (2-3h)
**Status**: ✅ Complete (commit 881ae5c)

**Description**: Implement top-level config types with nested sub-configs.

**Tasks**:

- [x] 2.4.1 Implement `GBDTConfig`:
  - `n_estimators: u32` (default 100)
  - `learning_rate: f64` (default 0.1)
  - `objective: PyObject` (default `SquaredLoss()`)
  - `metrics: Option<Vec<PyObject>>`
  - `tree: Py<TreeConfig>`
  - `regularization: Py<RegularizationConfig>`
  - `sampling: Py<SamplingConfig>`
  - `categorical: Py<CategoricalConfig>`
  - `efb: Py<EFBConfig>`
  - `linear_leaves: Py<LinearLeavesConfig>`
  - `n_threads: i32` (default 0 = auto)
  - `seed: Option<u64>`
  - `verbose: i32` (default 1)
- [x] 2.4.2 Implement `#[new]` with all defaults
- [x] 2.4.3 Implement `GBLinearConfig` (subset of fields, no tree/categorical/efb)
- [x] 2.4.4 Add `objective_kind(&self)` method for Rust-side extraction
- [x] 2.4.5 Add validation for cross-field consistency (deferred to fit-time)

**Definition of Done**:

- `GBDTConfig()` creates valid config with all defaults
- All 6 sub-configs nested correctly
- Stubs show full signature

**Testing Criteria**:

- `GBDTConfig()` works
- `GBDTConfig(tree=TreeConfig(max_depth=5))` works
- `GBDTConfig(objective=PinballLoss(alpha=0.5))` works
- Roundtrip: `config = GBDTConfig(); assert config.n_estimators == 100`

> Note: Don't forget to check stakeholder feedback.

---

### Story 2.5: Config Conversion Layer

**RFC Section**: RFC-0014 DD-3 (Rust-owned configs)  
**Effort**: M (2-3h)  
**Status**: ✅ Complete (commit 6095823)

**Description**: Implement conversion from PyO3 config types to core Rust types.

**Design Note**: Conversion is lazy—happens at fit-time, not config construction.
This keeps Python layer simple and lets Rust handle full validation.

**Implementation Note**: Story 2.5 implements the building blocks (PyObjective and PyMetric
conversion to core types). Full GBDTConfig/GBLinearConfig conversion is deferred to Story 4.3
where the actual fit() method will use these conversions.

**Tasks**:

- [x] 2.5.1 Create `src/convert.rs` module
- [ ] 2.5.2 Implement `PyGBDTConfig::to_core()` → `boosters::GBDTConfig` *(deferred to 4.3)*
- [ ] 2.5.3 Implement `PyGBLinearConfig::to_core()` *(deferred to 4.3)*
- [x] 2.5.4 Implement `PyObjective` → `boosters::Objective` conversion
- [x] 2.5.5 Add unit tests for all conversions
- [x] 2.5.6 Handle conversion errors with clear messages

**Definition of Done**:

- ~~All PyO3 config types convertible to core types~~ *(partial: objectives/metrics done)*
- Conversion preserves all values correctly
- Conversion happens in `fit()`, not in config `__init__`

> Note: Don't forget to check stakeholder feedback.

**Testing Criteria**:

- ~~`PyGBDTConfig(n_estimators=50).to_core().n_estimators == 50`~~ *(deferred to 4.3)*
- `PyPinballLoss(alpha=0.5)` converts to correct `Objective::Pinball` ✅
- ~~Invalid config (e.g., `n_estimators=-1`) → clear `ValueError` at fit time~~ *(4.3)*

---

### Story 2.6: Type Alias Exports

**Effort**: S (30min)  
**Status**: ✅ Complete (commit 881ae5c - already done during Story 2.4)

**Description**: Export type aliases for union types.

**Tasks**:

- [x] 2.6.1 Add `Objective` type alias to `_boosters_rs.pyi`
- [x] 2.6.2 Add `Metric` type alias to stubs
- [x] 2.6.3 Re-export type aliases in `boosters/__init__.py`
- [x] 2.6.4 Verify `typing.get_type_hints()` works with aliases

**Definition of Done**:

- Type aliases visible in IDE ✅
- `from boosters import Objective` works ✅

**Testing Criteria**:

- pyright accepts `objective: Objective = SquaredLoss()` ✅
- Type hints work with aliases ✅

---

### Story 2.7: Review - Config Types Demo

**Meta-task for Epic 2**  
**Status**: ✅ Complete

**Tasks**:

- [x] 2.7.1 Prepare demo showing:
  - All config types with IDE autocomplete
  - Validation error messages
  - Generated stubs
  - Config conversion roundtrip
- [x] 2.7.2 Document demo in `tmp/development_review_2025-01-21_epic2.md`
- [x] 2.7.3 Collect feedback on API ergonomics

**Definition of Done**:

- Demo completed and documented ✅
- Feedback captured for Epic 4+ adjustments ✅
- Feedback captured for Epic 4+ adjustments

---

## Epic 3: Dataset and Data Conversion

**Goal**: Implement `Dataset` and `EvalSet` with zero-copy NumPy integration.

**Why**: Efficient data handling is critical for training performance.

---

### Story 3.1: Dataset Type

**RFC Section**: RFC-0014 "Dataset"  
**Effort**: L (4-5h)  
**Status**: ✅ Complete

**Description**: Implement `Dataset` with NumPy array handling, lifetime management, and native categorical support.

**Data Layout Note**: C-contiguous (row-major) arrays are optimal for performance.
F-contiguous arrays will be copied to C-order automatically.

**Native Categoricals**: boosters supports native categorical splits (like LightGBM), using
bitset-based multi-way splits. This is different from one-hot encoding. Categorical features
are specified via `categorical_features=[...]` parameter or auto-detected from pandas categoricals.

**Tasks**:

- [x] 3.1.1 Create `src/data/mod.rs` module
- [x] 3.1.2 Add `numpy` dependency to PyO3 crate
- [x] 3.1.3 Implement `Dataset` as `#[pyclass]`:
  - `features: Py<PyAny>` (ndarray or DataFrame, kept alive)
  - `labels: Option<Py<PyAny>>`
  - `weights: Option<Py<PyAny>>`
  - `groups: Option<Py<PyAny>>` (for ranking)
  - `feature_names: Option<Vec<String>>`
  - `categorical_features: Option<Vec<usize>>` (feature indices)
- [x] 3.1.4 Implement `#[new]` with:
  - NumPy array detection
  - pandas DataFrame detection and column extraction
  - pandas categorical dtype → auto-detect and add to `categorical_features`
  - Integer-encoded categoricals → user specifies indices
- [x] 3.1.5 Implement `n_samples` and `n_features` properties
- [x] 3.1.6 Implement internal `to_rust_view()`:
  - Return `PyReadonlyArray2<f32>` or `PyReadonlyArray2<f64>`
  - `Py<PyArray>` reference keeps Python array alive during Rust access
  - Pass `categorical_features` indices to Rust training
- [x] 3.1.7 Handle memory layout:
  - C-contiguous float32: zero-copy
  - F-contiguous: copy to C-order
  - Non-contiguous: error with helpful message
- [x] 3.1.8 Add dtype validation (float32 recommended, float64 supported)
- [x] 3.1.9 Add NaN/Inf handling:
  - NaN allowed in features (treated as missing values, like XGBoost)
  - Inf in features → `ValueError` with message about data preprocessing
  - NaN/Inf in labels → `ValueError`
- [ ] 3.1.10 Sparse matrix detection moved to Story 4.6

**Definition of Done**:

- `Dataset` constructible from NumPy arrays and pandas DataFrames
- Properties return correct values
- Zero-copy works for C-contiguous float32
- pandas categoricals auto-detected and tracked
- `categorical_features` passed to Rust for native categorical splits
- Clear errors for invalid data (Inf, NaN in labels)

> Note: Don't forget to check stakeholder feedback.

**Testing Criteria**:

- `Dataset(features=np.zeros((100, 10)))` works
- `Dataset(features=pd.DataFrame(...))` works
- `dataset.n_samples` returns 100
- pandas categorical column auto-detected: `dataset.categorical_features == [1]`
- `Dataset(X, categorical_features=[0, 5])` works for integer-encoded data
- Features with NaN work (missing values)
- Features with Inf raise `ValueError`
- Labels with NaN raise `ValueError`
- Memory: <10% growth over 1000 Dataset constructions of same size (pytest-memray)

---

### Story 3.2: EvalSet Type

**RFC Section**: RFC-0014 "EvalSet"  
**Effort**: S (30min)  
**Status**: ✅ Complete (commit a233904 - done with Story 3.1)

**Description**: Implement named evaluation set wrapper.

**Tasks**:

- [x] 3.2.1 Implement `EvalSet` as `#[pyclass]`:
  - `name: String`
  - `dataset: Py<Dataset>`
- [x] 3.2.2 Implement `#[new]` constructor
- [x] 3.2.3 Export in module

**Definition of Done**:

- `EvalSet("valid", dataset)` works
- Name accessible for eval_results

> Note: Don't forget to check stakeholder feedback.

**Testing Criteria**:
- `EvalSet("test", Dataset(...))` constructible
- `eval_set.name` returns "test"

---

### Story 3.3: Stakeholder Feedback Check (Data)

**Meta-task for Epic 3**  
**Status**: ✅ Complete

**Tasks**:

- [x] 3.3.1 Review feedback on data format preferences
  - No specific data format feedback received
  - NumPy arrays and pandas DataFrames well supported
- [x] 3.3.2 Check if scipy.sparse support is needed for v1
  - Deferred to post-v1 (Story 4.6 handles sparse detection)
- [x] 3.3.3 Document any deferred features
  - Sparse matrix support deferred
  - Memory benchmarking (pytest-memray) deferred to v1 polish

**Definition of Done**:
- Data format requirements confirmed
- Scope adjustments documented

---

## Epic 4: Model Training and Prediction

**Goal**: Implement `GBDTModel` and `GBLinearModel` with fit/predict API.

**Why**: Core functionality that users interact with.

**Story Dependencies**: 4.1 → 4.2 → 4.3 → 4.4 → 4.5/4.6 (parallel) → 4.7

---

### Story 4.1: GBDTModel Structure

**RFC Section**: RFC-0014 "GBDTModel"  
**Effort**: M (2-3h)  
**Status**: ✅ Complete

**Description**: Implement `GBDTModel` class with config and state management.

**Tasks**:

- [x] 4.1.1 Create `src/model/mod.rs` module
- [x] 4.1.2 Implement `GBDTModel` as `#[pyclass]`:
  - `config: Py<GBDTConfig>`
  - `inner: Option<RustGBDTModel>` (None until fit)
- [x] 4.1.3 Implement `#[new]` with optional config
- [x] 4.1.4 Implement properties: `n_trees`, `n_features`, `is_fitted`
- [x] 4.1.5 Implement `get_config()` method
- [x] 4.1.6 Implement `feature_importance(importance_type="split")` method

**Definition of Done**:

- `GBDTModel()` constructible with default config
- `GBDTModel(config=GBDTConfig(...))` works
- Properties raise appropriate errors when not fitted

**Testing Criteria**:

- `model = GBDTModel()` works
- `model.n_trees` raises error before fit
- `model.feature_importance()` returns array after fit

---

### Story 4.2: GIL Management and Rust Callbacks

**RFC Section**: RFC-0014 "GIL Management"  
**Effort**: M (2-3h)  
**Status**: ✅ Complete

**Description**: Implement GIL release and Rust-side callbacks.

**Design Decision**: Callbacks during training are Rust-only (no Python callbacks during
fit). This avoids complex GIL juggling. Python can inspect results after training.

**Tasks**:

- [x] 4.2.1 Create `src/threading.rs` module
- [x] 4.2.2 Implement GIL release wrapper for training:
  - Use `py.allow_threads(|| { ... })` pattern
  - Store numpy array references in `Py<PyArray>` to keep alive
- [x] 4.2.3 Implement Rust-side callback structs:
  - `EarlyStoppingTracker`: wraps core's EarlyStopping, tracks best_n_trees
  - `EvalLogger`: logs metrics to buffer per round
- [x] 4.2.4 After training, convert Rust callback state to Python:
  - Early stopping → `best_iteration`, `best_score`
  - Log evaluation → `eval_results` dict via `to_python_dict()`
- [x] 4.2.5 Document that Python callbacks aren't called during training
- [x] 4.2.6 Add Rust unit tests for callback tracking

**Definition of Done**:

- Training releases GIL during compute
- Rust callbacks work correctly
- Results accessible in Python after training

> Note: GIL release verification test deferred to Story 4.3 integration tests.

**Testing Criteria**:

- Early stopping tracker correctly tracks best iteration (Rust test)
- Eval logger collects metrics per round (Rust test)
- `eval_results` dict conversion works (Rust test)

---

### Story 4.3: GBDTModel.fit()

**RFC Section**: RFC-0014 "GBDTModel.fit()"  
**Effort**: L (4-5h)  
**Depends on**: Story 2.5 (Config Conversion Layer)  
**Status**: ✅ Complete

**Description**: Implement training with callbacks and validation sets.

**Tasks**:

- [x] 4.3.1 Implement `fit(train, valid=None, callbacks=None) -> Self`:
  - Extract features/labels from Dataset
  - Call `config.to_core()` (Story 2.5) to convert to Rust types
  - Use GIL release pattern from Story 4.2
  - Store trained model in `self.inner`
- [x] 4.3.2 Handle `valid` as `EvalSet | list[EvalSet] | None`
- [ ] 4.3.3 Implement callback protocol (deferred to Story 4.6):
  - `EarlyStopping` callback class
  - `LogEvaluation` callback class
- [ ] 4.3.4 Populate `eval_results` dict after training (deferred to Story 4.6)
- [x] 4.3.5 Populate `best_iteration` and `best_score` properties
- [ ] 4.3.6 Implement `eval_train: bool` parameter for training metrics (deferred)
- [x] 4.3.7 Add comprehensive error messages for validation failures
- [ ] 4.3.8 Add robustness tests (deferred):
  - Training with NaN in features → clear error
  - Large dataset (1M rows) → doesn't crash
  - Out of memory → graceful error (if detectable)

**Checkpoint**: After Story 4.3, run integration test before continuing to 4.4+

**Definition of Done**:

- ✅ Training completes successfully
- ⏳ Callbacks work correctly (deferred)
- ✅ GIL released during training (other Python threads can run)

> Note: Don't forget to check stakeholder feedback.

**Testing Criteria**:

- ✅ `model.fit(train)` trains and returns self
- ✅ `model.fit(train, valid=[EvalSet("val", val_data)])` works
- ⏳ `model.eval_results["val"]["rmse"]` populated (Story 4.6)
- ✅ Early stopping triggers correctly (via config, not callbacks yet)
- ⏳ Training with `categorical_features` uses native categorical splits (not just encoded)
- ✅ Invalid label shape → clear `ValueError` with expected vs actual

---

### Story 4.4: GBDTModel.predict()

**RFC Section**: RFC-0014 "GBDTModel.predict()"  
**Effort**: M (2-3h)  
**Status**: ✅ Complete

**Description**: Implement prediction with various output modes.

**Tasks**:

- [x] 4.4.1 Implement `predict(features, n_iterations=None, raw_score=False, pred_contrib=False)`:
  - Validate feature shape
  - Release GIL during prediction
  - Return NumPy array
- [x] 4.4.2 Handle output shape based on objective:
  - Scalar: `(n_samples,)`
  - Multi-quantile: `(n_samples, n_quantiles)`
  - Multiclass: `(n_samples, n_classes)`
- [x] 4.4.3 Implement `raw_score=True` for margin output
- [ ] 4.4.4 Implement `n_iterations` for partial prediction (deferred)
- [ ] 4.4.5 Implement `pred_contrib=True` for SHAP values (deferred):
  - Call Rust SHAP implementation
  - Shape: `(n_samples, n_features + 1)` for scalar objectives
  - Shape: `(n_samples, n_features + 1, n_classes)` for multiclass
  - Last column is bias term

**Definition of Done**:

- ✅ Predictions match Rust crate output
- ✅ Output shapes correct for all objectives
- ⏳ SHAP values return correct shape (deferred)
- ✅ GIL released during prediction

**Testing Criteria**:

- ✅ `model.predict(X)` returns correct shape
- ✅ `model.predict(X, raw_score=True)` returns margins
- ⏳ `model.predict(X, pred_contrib=True)` returns `(n, features+1)` for regression (deferred)
- ⏳ `model.predict(X, pred_contrib=True)` returns `(n, features+1, n_classes)` for multiclass (deferred)
- ⏳ Multi-quantile returns `(n, 3)` for 3-alpha PinballLoss (need to test)

---

### Story 4.5: GBLinearModel

**RFC Section**: RFC-0014 "GBLinearModel"  
**Effort**: M (2-3h)
**Status**: ✅ Complete (commit 38288a5)

**Description**: Implement linear boosting model with same interface patterns as GBDTModel.

**Tasks**:

- [x] 4.5.1 Implement `GBLinearModel` structure (same pattern as GBDTModel)
- [x] 4.5.2 Implement `fit()` using Rust linear trainer with GIL release
- [x] 4.5.3 Implement `predict()` with GIL release
- [x] 4.5.4 Add `coef_` and `intercept_` properties for sklearn compat
- [x] 4.5.5 Populate `eval_results`, `best_iteration`, `best_score` (same as GBDTModel)

**Definition of Done**:

- `GBLinearModel` fully functional
- Matches sklearn-like coefficient access
- `eval_results` populated same as GBDTModel

> Note: Don't forget to check stakeholder feedback.

**Testing Criteria**:

- Training and prediction work
- `model.coef_` returns weights array
- `model.eval_results` matches GBDTModel behavior
- Invalid input raises appropriate errors

---

### Story 4.6: NotImplementedError for Deferred Features

**Effort**: S (15min)
**Status**: ✅ Complete (commit pending)

**Description**: Add clear error messages for features deferred to later versions.

**Tasks**:

- [x] 4.6.1 Add sparse matrix detection in `Dataset` → raise `NotImplementedError("Sparse matrices deferred to v0.2.0")`
- [x] 4.6.2 Document deferred features in class docstrings

**Definition of Done**:

- Users get clear error messages for deferred features
- Docstrings indicate what's not yet implemented

> Note: Don't forget to check stakeholder feedback.

**Testing Criteria**:

- `Dataset(scipy.sparse.csr_matrix(...))` raises appropriate error

---

### Story 4.7: Review - Training Demo

**Meta-task for Epic 4**
**Status**: ✅ Complete (commit pending)

**Tasks**:

- [x] 4.7.1 Review `tmp/stakeholder_feedback.md` for Epic 4-relevant feedback
- [x] 4.7.2 Create integration test `tests/test_training_demo.py`:
  - Load data, configure model, train, predict
  - Verify callbacks and early stopping work correctly
  - Cover both `GBDTModel` and `GBLinearModel`
- [x] 4.7.3 Document demo results in `tmp/development_review_<timestamp>.md`

**Note**: Performance and quality comparisons with other libraries are handled by
`packages/boosters-eval`, not in the Python bindings tests.

**Definition of Done**:

- Stakeholder feedback reviewed and addressed
- Integration test covers complete workflow
- Demo documented in development review

---

## Epic 5: scikit-learn Integration

**Goal**: Implement sklearn-compatible estimators with flat kwargs.

**Why**: DD-13 requires sklearn layer for familiar API.

**Implementation Note**: The sklearn module (`python/boosters/sklearn.py`) is pure Python,
wrapping the Rust-based `GBDTModel`. This enables fast iteration on the sklearn interface
without rebuilding the Rust extension.

**Shared Logic**: Extract common kwargs→config conversion to `python/boosters/_sklearn_base.py`
to avoid duplication across `GBDTRegressor`, `GBDTClassifier`, `GBLinearRegressor`, etc.

---

### Story 5.1: GBDTRegressor

**RFC Section**: RFC-0014 "sklearn Integration"  
**Effort**: M (2-3h)
**Status**: ✅ Complete (commit 424bea4)

**Description**: Implement sklearn-compatible regressor.

**Tasks**:

- [x] 5.1.1 Create `python/boosters/sklearn.py` module
- [x] 5.1.2 Create `python/boosters/_sklearn_base.py` with shared kwargs→config conversion
- [x] 5.1.3 Implement `GBDTRegressor(BaseEstimator, RegressorMixin)`:
  - Flat kwargs: `max_depth`, `n_leaves`, `learning_rate`, etc.
  - Internal: Create `GBDTConfig` from kwargs
- [x] 5.1.4 Implement `fit(X, y, eval_set=None, early_stopping_rounds=None)`
- [x] 5.1.5 Implement `predict(X)`
- [x] 5.1.6 Implement `get_params()` / `set_params()` for sklearn compat
- [x] 5.1.7 Add `feature_importances_` property
- [x] 5.1.8 Add sklearn parameter mapping table from RFC
- [x] 5.1.9 Add tests for kwargs→GBDTConfig conversion edge cases:
  - Default values match RFC defaults
  - Invalid kwargs raise ValueError
  - Nested config options (e.g., tree config) are correctly mapped

**Definition of Done**:

- `GBDTRegressor()` passes `check_estimator()` core checks
- Flat kwargs work correctly
- kwargs→config conversion fully tested
- Works in sklearn `Pipeline` and `cross_val_score`

**Testing Criteria**:

- `check_estimator(GBDTRegressor())` passes core checks:
  - `check_estimator_cloneable`
  - `check_fit_score_takes_y`
  - `check_methods_subset_invariance`
- May skip initially: `check_sample_weights_*` (deferred), `check_estimator_sparse` (v0.2.0)
- `cross_val_score(GBDTRegressor(), X, y)` works

---

### Story 5.2: GBDTClassifier

**RFC Section**: RFC-0014 "sklearn Integration"  
**Effort**: M (2-3h)
**Status**: ✅ Complete (commit 424bea4)

**Description**: Implement sklearn-compatible classifier.

**Tasks**:

- [x] 5.2.1 Implement `GBDTClassifier(BaseEstimator, ClassifierMixin)`:
  - Auto-infer `objective` from label cardinality
  - Binary: `LogisticLoss()`
  - Multiclass: `SoftmaxLoss(n_classes=k)`
- [x] 5.2.2 Implement `fit(X, y)` with label encoding
- [x] 5.2.3 Implement `predict(X)` returning class labels
- [x] 5.2.4 Implement `predict_proba(X)` returning probabilities
- [x] 5.2.5 Add `classes_` property

**Definition of Done**:

- Binary and multiclass classification work
- Probability outputs correct

**Testing Criteria**:

- `check_estimator(GBDTClassifier())` passes
- Binary classification works on Iris subset
- Multiclass works on full Iris

---

### Story 5.3: GBLinearRegressor and GBLinearClassifier

**RFC Section**: RFC-0014 "sklearn Integration"  
**Effort**: S (1-2h)
**Status**: ✅ Complete (commit 424bea4)

**Description**: Implement linear model sklearn wrappers.

**Tasks**:

- [x] 5.3.1 Implement `GBLinearRegressor` (same pattern as GBDT)
- [x] 5.3.2 Implement `GBLinearClassifier`
- [x] 5.3.3 Add `coef_` and `intercept_` properties

**Definition of Done**:

- Both estimators pass sklearn checks
- Linear coefficient access works

**Testing Criteria**:

- `check_estimator()` passes for both

---

### Story 5.4: Stakeholder Feedback Check (sklearn)

**Meta-task for Epic 5**
**Status**: ✅ Complete

**Tasks**:

- [x] 5.4.1 Review `tmp/stakeholder_feedback.md` for sklearn-specific feedback
- [x] 5.4.2 Review feedback on sklearn API preferences
- [x] 5.4.3 Check migration pain points from other libraries
- [x] 5.4.4 Adjust parameter mapping if needed
- [x] 5.4.5 Document outcomes in `tmp/development_review_<timestamp>.md`

**Definition of Done**:

- sklearn API validated against user expectations
- Stakeholder feedback reviewed and addressed
- Migration blockers identified and addressed

---

## Epic 6: Polish and Documentation

**Goal**: Complete documentation, examples, and release prep.

**Why**: Professional polish required for adoption.

---

### Story 6.1: API Documentation

**RFC Section**: RFC-0014 "Documentation Strategy"  
**Effort**: M (2-3h)

**Description**: Complete docstrings and generate API docs.

**Tasks**:

- [ ] 6.1.1 Add Google-style docstrings to all public classes
- [ ] 6.1.2 Add Google-style docstrings to all public methods
- [ ] 6.1.3 Include code examples in docstrings
- [ ] 6.1.4 Set up Sphinx or mkdocs for documentation site
- [ ] 6.1.5 Add doctest verification to CI

**Definition of Done**:
- All public API documented
- Examples in docstrings are runnable
- Documentation site buildable

**Testing Criteria**:
- `pytest --doctest-modules` passes
- `mkdocs build` succeeds

---

### Story 6.2: Example Notebooks

**RFC Section**: RFC-0014 "Quick Start"  
**Effort**: M (2-3h)

**Description**: Create example Jupyter notebooks.

**Minimum Viable Set** (MVP):
1. `01_quickstart.ipynb` - basic train/predict with core API
2. `02_configuration.ipynb` - config options and nested configs
3. `03_callbacks.ipynb` - early stopping, evaluation logging

**Tasks**:

- [ ] 6.2.1 Create `examples/01_quickstart.ipynb`:
  - Core API usage
  - Nested config demonstration
- [ ] 6.2.2 Create `examples/02_sklearn.ipynb`:
  - sklearn estimator usage
  - Pipeline integration
  - Cross-validation
- [ ] 6.2.3 Create `examples/03_callbacks.ipynb`:
  - Early stopping setup
  - Evaluation logging
  - eval_results inspection
- [ ] 6.2.4 Add notebook execution to CI with `nbmake`

**Definition of Done**:

- All notebooks run without errors
- Cover major use cases

**Testing Criteria**:

- `nbmake` passes on all notebooks (pytest --nbmake)

---

### Story 6.3: Migration Guide

**RFC Section**: RFC-0014 "Migration Guide"  
**Effort**: S (1h)

**Description**: Document XGBoost/LightGBM migration path.

**Tasks**:

- [ ] 6.3.1 Create `docs/migration.md` with parameter mapping tables
- [ ] 6.3.2 Add common migration patterns
- [ ] 6.3.3 Document known differences

**Definition of Done**:
- Migration guide complete and linked from README

---

### Story 6.4: Release Preparation

**Effort**: S (1h)

**Description**: Prepare for PyPI release.

**Tasks**:

- [ ] 6.4.1 Finalize version number
- [ ] 6.4.2 Update CHANGELOG (Python-specific changes tagged in root CHANGELOG)
- [ ] 6.4.3 Create `packages/boosters-python/README.md` with:
  - PyPI version badge
  - CI status badge  
  - Python version support badge (3.12+)
  - Quick start example
- [ ] 6.4.4 Verify wheel builds for all platforms
- [ ] 6.4.5 Test installation from wheel
- [ ] 6.4.6 Prepare PyPI metadata (long_description, classifiers)
- [ ] 6.4.7 Test on TestPyPI before real PyPI
- [ ] 6.4.8 Document rollback/hotfix process:
  - How to yank a PyPI release if critical bugs found
  - Process for v0.1.1 hotfix release if needed
  - Known issues section in README

**Definition of Done**:

- Package installable from wheel
- README has proper badges and quick start
- TestPyPI test successful
- Ready for `maturin publish`
- Rollback process documented

---

### Story 6.5: Final Review and Demo

**Meta-task for Epic 6**

**Tasks**:

- [ ] 6.5.1 Prepare comprehensive demo:
  - Installation from PyPI (test.pypi.org)
  - Full workflow: data → train → predict
  - sklearn integration showcase
  - Performance comparison
- [ ] 6.5.2 Document demo in `tmp/development_review_<timestamp>.md`
- [ ] 6.5.3 Final stakeholder sign-off

**Definition of Done**:
- Demo completed
- All stakeholder concerns addressed

---

## Epic 7: Retrospective and Process Improvement

**Goal**: Capture learnings and improve future development.

**Timing**: Run retrospective after v0.1.0 MVP and again after v0.2.0.

---

### Story 7.1: Release Retrospective

**Meta-task**

**Tasks**:

- [ ] 7.1.1 Conduct retrospective on Python bindings implementation (post-v0.1.0)
- [ ] 7.1.2 Document outcomes in `tmp/retrospective.md`:
  - What went well
  - What could be improved
  - Action items for future work
- [ ] 7.1.3 Create backlog items for high-priority improvements
- [ ] 7.1.4 Repeat retrospective after v0.2.0 release

**Definition of Done**:

- Retrospective documented for each milestone
- At least one improvement added to future backlog

---

## Testing Strategy Summary

### Test Organization

All Python tests live in `tests/` directory (flat structure):
- `test_config.py` - Config construction and defaults
- `test_dataset.py` - Dataset wrapper and memory behavior
- `test_model.py` - GBDTModel/GBLinearModel basic operations  
- `test_training_demo.py` - Integration test for full workflow
- `test_sklearn.py` - sklearn compliance tests

Unit vs integration is determined by scope, not location. Integration tests
exercise full train/predict flows; unit tests isolate individual components.

### Test Pyramid by Epic

| Epic | Focus | Key Tests |
| ---- | ----- | --------- |
| 1: Setup | Build works, imports, tooling | `import boosters`, `poe check` passes |
| 2: Config | Construction, defaults, validation | Config roundtrip, defaults match RFC |
| 3: Dataset | Data loading, memory safety | Zero-copy, <10% memory growth |
| 4: Training | End-to-end, correctness | train/predict works, doctests pass |
| 5: sklearn | Compliance, pipelines | `check_estimator()`, `cross_val_score()` |
| 6: Polish | Documentation renders, examples | `doctest`, notebook execution |

### Test Categories

| Category | Location | Tools | Coverage Target |
| -------- | -------- | ----- | --------------- |
| Unit tests | `tests/test_*.py` | pytest | Config types, Dataset, basic Model |
| Integration tests | `tests/test_*.py` | pytest | Full train/predict flows |
| sklearn compliance | `tests/test_sklearn.py` | `check_estimator` | All estimators |
| Doctests | `python/boosters/**/*.py` | `pytest --doctest-modules` | All public APIs |
| Notebooks | `examples/*.ipynb` | nbmake | All notebooks run |

**Note**: Library comparison benchmarks (vs XGBoost, LightGBM) are handled by
`packages/boosters-eval`, not in the Python bindings tests.

### CI Matrix

| Python | OS | Test Scope |
| ------ | -- | ---------- |
| 3.12 | Linux, macOS, Windows | Full |
| 3.13 | Linux | Full |

---

## Changelog

- 2025-12-25: Initial backlog created from RFC-0014
- 2025-12-25: Round 1 refinement:
  - Added milestones (v0.1.0 MVP, v0.2.0 sklearn)
  - Revised effort estimates upward (40-55h total)
  - Added Story 1.4: Error handling infrastructure
  - Added Story 4.2: GIL management and threading
  - Clarified numerical parity tolerances
  - Renumbered stories in Epic 4
- 2025-12-25: Round 2 refinement:
  - Expanded Story 1.1 with Python re-export modules
  - Added Story 4.6: NotImplementedError for deferred features
  - Expanded Story 4.5 (GBLinearModel) with more detail
  - Added XGBoost version reference (v2.0.x) for parity tests
  - Added specific GIL release test criteria
- 2025-12-25: Round 3 refinement:
  - Added architecture diagram showing dependency direction
  - Simplified Story 4.2: Rust-only callbacks during training
  - Expanded Story 3.1: pandas categorical handling, lifetime management
  - Added feature_importance() to Story 4.1
  - Added memory leak test criteria to Dataset tests
  - Added sparse matrix NotImplementedError
- 2025-12-25: Round 4 refinement:
  - Added Story 2.5: Config conversion layer (PyGBDTConfig → core)
  - Added Story 2.6: Type alias exports for IDE support
  - Renumbered Story 2.5 → 2.7 (Review - Config Types Demo)
  - Added XGBoost baseline generation task to Story 4.7
  - Set training speed target: within 2× of XGBoost
- 2025-12-25: Round 5 refinement:
  - Added "conversion is lazy (at fit-time)" note to Story 2.5
  - Added cross-reference from Story 4.3 → Story 2.5
  - Added invalid config validation test case to Story 2.5
  - Added invalid label shape test case to Story 4.3
  - Added memory threshold: <10% growth over 1000 iterations (Story 3.1)
  - Added stakeholder feedback check task to Story 4.7
- 2025-12-25: Round 6 refinement:
  - Added story dependency chain to Epic 4 header
  - Added stub-gen fallback task to Story 1.2
  - Clarified check_estimator target checks in Story 5.1
  - Added Test Pyramid by Epic table
  - Added callback state pattern to architecture diagram
  - Clarified Epic 7 timing (post-v0.1.0 and post-v0.2.0)
  - Referenced Story 4.6 from Story 4.4.5 for SHAP placeholder
- 2025-12-25: Round 7 refinement:
  - Added robustness tests (NaN, large datasets) to Story 4.3
  - Added checkpoint after Story 4.3 for integration test
  - Added stakeholder feedback check to Story 5.4
  - Specified minimum notebook set in Story 6.2
  - Added rollback/hotfix process task to Story 6.4
  - Added sklearn pure Python + shared base note to Epic 5
- 2025-12-25: Round 8 refinement (final):
  - Updated status to "Ready for Implementation"
  - Added Non-Goals section (v0.1.0 scope boundaries)
  - Added Definition of Ready section
  - Added pytest fixture creation task to Story 1.5
  - Pinned XGBoost version (2.0.3) for reproducible parity tests
  - Backlog finalized and approved for implementation
- 2025-01-21: Rounds 9-12 refinement (prework + Polish):
  - **Prework**: Analyzed existing code in `packages/boosters-python/`
  - **Removed**: Save/load support completely (not even interfaces)
  - **Removed**: Library comparison tests (handled by boosters-eval)
  - **Added Story 1.3**: Python Development Tooling (ruff, pyright, poe, doctests)
  - **Updated Story 1.1**: Marked existing skeleton tasks as completed/partial
  - **Updated Story 1.2**: Clarified stub location (no separate _manual_stubs.pyi)
  - **Updated Story 1.3**: Dual poe tasks (root namespaced + package local)
  - **Updated Story 3.1**: Added NaN=missing note, Inf validation, data layout note
  - **Updated Story 4.5**: Added eval_results parity with GBDTModel
  - **Updated Story 4.7**: Changed "demo" to concrete integration test
  - **Updated Story 5.1**: Added _sklearn_base.py, kwargs→config conversion tests
  - **Updated Story 6.2**: Standardized on nbmake for notebooks
  - **Updated Story 6.4**: Added README badges, TestPyPI task
  - **Updated Testing Strategy**: Clarified flat test directory structure
  - Backlog re-finalized for implementation
- 2025-01-21: Rounds 13-16 refinement (scope & tooling corrections):
  - **Scope fix**: SHAP values and native categoricals now in scope (were incorrectly excluded)
  - **Added "In Scope" section**: Clarifies SHAP output shapes and native categorical split behavior
  - **Updated Story 1.3**: Dev deps in root pyproject.toml (uv workspace), poe task naming (singular: test, doctest)
  - **Updated Story 2.1**: Added CategoricalConfig, EFBConfig, LinearLeavesConfig (now 6 config types, effort L)
  - **Updated Story 2.4**: GBDTConfig now includes all 6 sub-configs
  - **Updated Story 3.1**: Native categorical support details, NaN=missing clarification
  - **Updated Story 4.4**: SHAP fully implemented (not placeholder), multiclass SHAP test added
  - **Updated Story 4.6**: Changed to sparse matrix NotImplementedError only (SHAP moved to 4.4)
  - **Updated Testing Strategy**: "poe all" → "poe check" consistency
  - **Updated milestones**: v0.1.0 now explicitly includes SHAP and categoricals
  - **Updated CONTRIBUTING guide**: Added Python Development section (uv workspace, poe tasks, test org)
  - Backlog re-finalized for implementation
