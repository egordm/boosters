# Python Bindings Polish Backlog

**Source**: Stakeholder feedback from `tmp/stakeholder_feedback.md` 
**Created**: 2025-12-25 
**Status**: ✅ COMPLETE

---

## Overview

Post-MVP polish work for the Python bindings based on stakeholder feedback and quality improvements.

**Scope**:

1. Fix Python type errors (pyright) to ensure CI passes ✅
2. Improve stub generation workflow ✅ (documented hybrid approach)
3. Add executable example scripts ✅
4. Update poe tasks with check/fix pattern ✅
5. Remove frivolous tests (simplify test suite) ✅
6. Consider refactoring Dataset conversion to Python (deferred)

---

## Epic 8: Python CI and Typing Fixes ✅

**Goal**: Ensure all Python tooling passes (pyright, ruff, pytest) before any commit.

---

### Story 8.1: Fix Type Stubs ✅

**Effort**: M (1-2h)

**Description**: Complete the stub file with missing methods and fix type errors.

**Tasks**:

- [x] 8.1.1 Add `GBLinearModel` class to stubs
- [x] 8.1.2 Add `fit()` method to `GBDTModel` stubs
- [x] 8.1.3 Add `predict()` method to `GBDTModel` stubs
- [x] 8.1.4 Add `fit()` and `predict()` to `GBLinearModel` stubs
- [x] 8.1.5 Add `coef_`, `intercept_`, `n_features_in_` properties to GBLinearModel

**Definition of Done**:

- ✅ All model classes have complete type stubs
- ✅ `from boosters import GBLinearModel` has no type errors

---

### Story 8.2: Fix sklearn Type Errors ✅

**Effort**: M (1-2h)

**Description**: Resolve sklearn-related type errors without requiring sklearn stubs.

**Stakeholder Feedback**: "I see a lot of python type errors. Perhaps it makes sense to
install some type of sklearn stubs so they don't cause unnecessary type errors for us?"

**Tasks**:

- [x] 8.2.1 Fix BaseEstimator/ClassifierMixin/RegressorMixin type assignment errors
- [x] 8.2.2 Fix check_array/check_X_y function signature type errors
- [x] 8.2.3 Fix NDArray[float32] vs float64 return type mismatches
- [x] 8.2.4 Add `# type: ignore` with comments where unavoidable (sklearn interaction)
- [x] 8.2.5 Fix growth_strategy str vs Literal type mismatch in _sklearn_base.py

**Definition of Done**:

- ✅ `pyright packages/boosters-python` reports 0 errors
- ✅ No sklearn stubs required (use type: ignore where needed)

---

### Story 8.3: Fix Test Type Errors ✅

**Effort**: S (30min)

**Description**: Fix type errors in test files that test invalid inputs.

**Tasks**:

- [x] 8.3.1 Add `# type: ignore` to tests that intentionally pass invalid types
- [x] 8.3.2 Document why the ignore is needed (testing error handling)

**Definition of Done**:

- ✅ Test files have no type errors
- ✅ Ignores are documented

---

### Story 8.4: Update Poe Tasks ✅

**Effort**: S (30min)

**Stakeholder Request**: Match poe task pattern with --check flags from other project.

**Tasks**:

- [x] 8.4.1 Update task structure with check/fix pattern
- [x] 8.4.2 Add check task (format-check + lint + typecheck)
- [x] 8.4.3 Add fix task (format + lint-fix)
- [x] 8.4.4 Add python:examples task to run example scripts

**Definition of Done**:

- ✅ `poe check` runs all checks without modification
- ✅ `poe fix` applies auto-fixes
- ✅ `poe python:examples` runs all examples

---

## Epic 9: Examples and Documentation ✅

**Goal**: Add executable example scripts that match README examples.

---

### Story 9.1: Add Example Scripts ✅

**Effort**: M (1-2h)

**Description**: Create executable Python scripts for key examples.

**Tasks**:

- [x] 9.1.1 Create `examples/` directory in boosters-python package
- [x] 9.1.2 Create `examples/01_sklearn_quickstart.py` - sklearn estimators
- [x] 9.1.3 Create `examples/02_core_api.py` - core API with nested configs
- [x] 9.1.4 Create `examples/03_sklearn_integration.py` - Pipeline, GridSearchCV
- [x] 9.1.5 Create `examples/04_linear_models.py` - GBLinear models

**Definition of Done**:

- ✅ All examples run without error
- ✅ Examples are self-contained (no external data needed)
- ✅ Examples match README code snippets

---

### Story 9.2: Example Execution Task ✅

**Effort**: S (15min)

**Description**: Add poe task to verify examples run without error.

**Tasks**:

- [x] 9.2.1 Add python:examples poe task that executes each example script
- [x] 9.2.2 Verified all examples run successfully

**Definition of Done**:

- ✅ Examples can be run via poe task
- ✅ All examples pass

---

## Epic 10: Code Quality Improvements ✅

**Goal**: Address remaining stakeholder feedback on code quality.

---

### Story 10.1: Stub Generation Workflow ✅

**Effort**: S (30min)

**Stakeholder Feedback**: "Would it make sense to add [stubs] to gitignore and add a
script perhaps also to poe to regenerate it?"

**Design Decision**: Stubs are manually augmented with additional type info that
pyo3-stub-gen cannot generate (method signatures, property types). The stubs are kept
in git and documented as a hybrid auto-generated + manually-augmented file.

**Tasks**:

- [x] 10.1.1 Documented hybrid workflow in stub file header
- [x] 10.1.2 Stubs remain in git for IDE support

**Definition of Done**:

- ✅ Stub workflow documented
- ✅ Stubs work correctly with pyright

---

### Story 10.2: Simplify Default Tests ✅

**Effort**: S (30min)

**Stakeholder Feedback**: "Creating tests to verify defaults feels a bit flakey...
We don't care what the actual value is of the defaults while testing bindings."

**Tasks**:

- [x] 10.2.1 Review test_config.py for frivolous default value assertions
- [x] 10.2.2 Replace specific value tests with type and range checks
- [x] 10.2.3 Updated test_gbdt_config.py and test_gblinear_config.py

**Definition of Done**:

- ✅ Tests verify behavior, not specific default values
- ✅ Fewer tests that would break on harmless default changes

---

### Story 10.3: Dataset Conversion Refactor (DEFERRED)

**Effort**: L (3-4h)

**Stakeholder Feedback**: "Would it make sense to implement things like
try_extract_dataframe in python, prepare the array and schema and then pass it to a
more simple method?"

**Decision**: Deferred for future work. Current Rust implementation is working and
tested. The refactor would be a larger undertaking best done as a separate backlog item.

---

## Epic 11: Final CI Validation ✅

**Goal**: Ensure full CI passes before considering complete.

---

### Story 11.1: CI Green Check ✅

**Effort**: S (15min)

**Description**: Run all checks and fix any remaining issues.

**Tasks**:

- [x] 11.1.1 Run `poe check` - 0 errors
- [x] 11.1.2 Run pytest - 238 tests pass
- [x] 11.1.3 Run examples - all 4 examples pass

**Definition of Done**:

- ✅ `poe check` passes
- ✅ All Python tests pass
- ✅ All examples run successfully

---

### Story 11.2: Update Stakeholder Feedback ✅

**Effort**: S (5min)

**Description**: Mark stakeholder feedback items as addressed.

**Tasks**:

- [x] 11.2.1 Update tmp/stakeholder_feedback.md with completion status
- [x] 11.2.2 Note items that were evaluated but not implemented

**Definition of Done**:

- ✅ All feedback items marked as addressed or documented

---

## Changelog

- 2025-12-25: Initial backlog created from stakeholder feedback
- 2025-12-25: Completed all stories - CI passes, examples work, stakeholder feedback addressed
