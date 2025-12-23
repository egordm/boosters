# Backlog 12: ndarray Cleanup and API Consolidation

**RFC**: RFC-0021  
**Priority**: High  
**Status**: In Progress  
**Started From**: Commit `2702fbe` (mid-refactor)

---

## Overview

Complete the ndarray migration by finishing the cleanup started in commit 2702fbe.
The primary goal is a **clean API** - ndarray is just the means to that end.

**Core Principles** (from stakeholder):
- Less layers of functions (at most two)
- Prefer explicit and easy defaults over many convenience functions
- Traversal returns leaf index; aggregation delegated to prediction
- Parallelism enum (Sequential/Parallel) passed to components
- Model-level functions take n_threads and set up rayon thread pool
- Components assume thread pool is already configured

---

## Epic 1: Fix Compilation

Get the codebase compiling from the mid-refactor state.

### Story 1.1: Fix Parallelism Enum Naming

**Goal**: Fix `Parallelism::SEQUENTIAL` → `Parallelism::Sequential` across codebase.

**Tasks**:
- [ ] 1.1.1: Fix in `training/gbdt/histograms/ops.rs` (4 occurrences)
- [ ] 1.1.2: Fix in `training/gbdt/split/find.rs` (1 occurrence)

---

### Story 1.2: Fix Model Prediction API

**Goal**: Update GBDTModel to use new `predict_into` API.

**Tasks**:
- [ ] 1.2.1: Rewrite `predict()` to use `predictor.predict_into()`
- [ ] 1.2.2: Rewrite `predict_raw()` to use `predictor.predict_into()`
- [ ] 1.2.3: Use ndarray for output allocation

---

### Story 1.3: Fix GBLinear Model

**Goal**: Fix closure argument issue in gblinear model.

**Tasks**:
- [ ] 1.3.1: Fix closure in `model/gblinear/model.rs:112`

---

### Story 1.4: Fix Predictor Tests

**Goal**: Update predictor tests to use new API.

**Tasks**:
- [ ] 1.4.1: Update tests using `predict_row()` to use `predict_row_into()`
- [ ] 1.4.2: Update tests using `predict()` to use `predict_into()`
- [ ] 1.4.3: Update tests using `par_predict()` to use `predict_into()` with Parallelism
- [ ] 1.4.4: Update weighted prediction tests

---

### Story 1.5: Verify Compilation

**Goal**: Ensure `cargo build --release` and `cargo test` pass.

**Tasks**:
- [ ] 1.5.1: Run `cargo build --release`
- [ ] 1.5.2: Run `cargo test`
- [ ] 1.5.3: Run `cargo clippy`

---

## Epic 2: Clean Up Training Pipeline

Apply same cleanup principles to training code.

### Story 2.1: Trainer Parallelism API

**Goal**: Update trainer to receive Parallelism enum consistently.

**Tasks**:
- [ ] 2.1.1: Audit trainer API for parallelism handling
- [ ] 2.1.2: Ensure Parallelism flows through to components
- [ ] 2.1.3: Components use helpers for conditional parallelism

---

### Story 2.2: Histogram Building Cleanup

**Goal**: Simplify histogram building parallelism.

**Tasks**:
- [ ] 2.2.1: Use Parallelism helpers consistently
- [ ] 2.2.2: Remove redundant conditionals

---

## Epic 3: Clean Up GBLinear

Apply same cleanup to linear model inference and training.

### Story 3.1: GBLinear Inference

**Goal**: Clean up linear model prediction API.

**Tasks**:
- [ ] 3.1.1: Consolidate prediction methods
- [ ] 3.1.2: Use ndarray for input/output
- [ ] 3.1.3: Apply Parallelism pattern

---

### Story 3.2: GBLinear Training

**Goal**: Clean up linear model training API.

**Tasks**:
- [ ] 3.2.1: Audit parallelism usage
- [ ] 3.2.2: Apply same patterns as GBDT

---

## Epic 4: Final Validation

Ensure everything works correctly.

### Story 4.1: Test Suite

**Goal**: All tests pass.

**Tasks**:
- [ ] 4.1.1: Run full test suite
- [ ] 4.1.2: Fix any remaining issues

---

### Story 4.2: Benchmarks

**Goal**: Verify no performance regression.

**Tasks**:
- [ ] 4.2.1: Run prediction benchmarks
- [ ] 4.2.2: Compare with baseline

---

## Changelog

- 2025-12-22: Created backlog from mid-refactor commit 2702fbe

