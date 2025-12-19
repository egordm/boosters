# Audit Followup Backlog

**Source**: [Code Audit 2025-12-19](../audits/2025-12-19-code-audit.md)  
**Created**: 2025-12-19  
**Status**: In Progress (Epic 1 & 2 Complete)

---

## Overview

This backlog addresses findings from the comprehensive pre-1.0 code audit.
All items should be completed before the 1.0.0 release.

**Estimated Total Effort**: ~6 hours

---

## Epic 1: API Cleanup ✅ COMPLETE

**Goal**: Establish clean, consistent public API with canonical import paths.

**Why**: The audit found confusing re-export chains (A-1.1) and missing convenience methods (U-3.1, U-3.3). Fixing these before 1.0 ensures a stable API surface.

### Story 1.1: Establish Canonical Import Paths ✅

**Findings Addressed**: A-1.1 (Confusing Re-export Chains)  
**Effort**: M (30min - 2h)  
**Status**: ✅ COMPLETE

**Description**: Remove duplicate re-exports so each public type has exactly one import path.

**Canonical Paths**:

| Type | Path | Reason |
|------|------|--------|
| `Forest`, `Tree`, `Node` | `boosters::repr::gbdt::*` | Data structures |
| `Predictor`, `TreeTraversal` | `boosters::inference::gbdt::*` | Prediction logic |
| `LinearModel` | `boosters::repr::gblinear::*` | Data structure |
| `GBDTTrainer`, `Objective` | `boosters::training::*` | Training |

**Tasks**:

- [x] 1.1.1 Remove `repr::gbdt` re-exports from `src/inference/gbdt/mod.rs` (lines 43-48)
- [x] 1.1.2 Remove `Forest`, `Tree`, `Node` etc. from `src/inference/mod.rs` re-exports
- [x] 1.1.3 Update `src/training/gbdt/trainer.rs` to import from `crate::repr::gbdt`
- [x] 1.1.4 Update `src/compat/xgboost/convert.rs` to import from `crate::repr::gbdt`
- [x] 1.1.5 Update `src/inference/gbdt/predictor.rs` internal imports
- [x] 1.1.6 Update `src/inference/gbdt/traversal.rs` internal imports
- [x] 1.1.7 Update `benches/common/models.rs` to use `boosters::repr::gbdt`
- [x] 1.1.8 Update doctests in `src/repr/gbdt/tree.rs` and `forest.rs`
- [x] 1.1.9 Update module docs in `src/inference/mod.rs` (Quick Start example)
- [x] 1.1.10 Verify `cargo doc` shows types at canonical paths only

**Definition of Done**:
- ✅ `cargo test` passes (476 tests)
- ✅ `cargo doc --no-deps` builds without warnings
- ✅ Each public type appears at exactly one path in docs
- ✅ No internal code imports repr types from inference

---

### Story 1.2: Add XgbModel::from_file() ✅

**Findings Addressed**: U-3.3 (XGBoost Missing from_file())  
**Effort**: S (< 30min)  
**Status**: ✅ COMPLETE

**Description**: Add convenience method for loading XGBoost models from file, matching LightGBM's `LgbModel::from_file()` pattern.

**Current usage** (awkward):
```rust
let model: XgbModel = serde_json::from_reader(File::open(path)?)?;
```

**Proposed usage** (ergonomic):
```rust
let model = XgbModel::from_file("model.json")?;
```

**Tasks**:

- [x] 1.2.1 Add `from_file()` method to `XgbModel` in `src/compat/xgboost/json.rs`
- [x] 1.2.2 Return `Result<Self, std::io::Error>` (wrap serde errors as `InvalidData`)
- [x] 1.2.3 Add doctest example
- [x] 1.2.4 Update Quick Start in `src/lib.rs` to use `from_file()`

**Definition of Done**:
- ✅ Method exists with rustdoc
- ✅ Doctest compiles (uses `ignore` due to file I/O)
- ✅ Error handling matches LightGBM pattern
- ✅ Quick Start updated

---

### Story 1.3: Add BinnedDatasetBuilder::from_row_matrix() ✅

**Findings Addressed**: U-3.1 (Training Requires Column-Major Matrix)  
**Effort**: S (< 30min)  
**Status**: ✅ COMPLETE

**Description**: Add convenience method to build binned dataset from row-major data.

**Current usage** (awkward):
```rust
let row_matrix: RowMatrix<f32> = DenseMatrix::from_vec(...);
let col_matrix: ColMatrix<f32> = row_matrix.to_layout();
let dataset = BinnedDatasetBuilder::from_matrix(&col_matrix, 256).build()?;
```

**Proposed usage** (ergonomic):
```rust
let row_matrix: RowMatrix<f32> = DenseMatrix::from_vec(...);
let dataset = BinnedDatasetBuilder::from_row_matrix(&row_matrix, 256).build()?;
```

**Tasks**:

- [x] 1.3.1 Add `from_row_matrix()` to `BinnedDatasetBuilder`
- [x] 1.3.2 Internally convert to column-major using `to_layout()`
- [x] 1.3.3 Add doctest example
- [x] 1.3.4 Test added: `test_from_row_matrix_matches_manual_conversion`

**Definition of Done**:
- ✅ Method exists with rustdoc
- ✅ Doctest compiles
- ✅ Produces identical results to manual conversion (verified by test)

---

## Epic 2: Documentation ✅ COMPLETE

**Goal**: Ensure documentation is accurate and provides working examples.

**Why**: The audit found outdated Quick Start examples and missing README code (U-3.4, U-3.6).

### Story 2.1: Add README Code Examples ✅

**Findings Addressed**: U-3.6 (No Code Examples in README)  
**Effort**: S (< 30min)  
**Status**: ✅ COMPLETE

**Description**: Add working code examples to README for training and loading models.

**Tasks**:

- [x] 2.1.1 Add "Training Example" section with regression example
- [x] 2.1.2 Add "Loading XGBoost Models" section
- [x] 2.1.3 Ensure examples match current API (use canonical imports)
- [x] 2.1.4 Keep examples concise (show essentials, link to docs for details)

**Definition of Done**:
- ✅ README has 2 code examples (Training + XGBoost loading)
- ✅ Examples use canonical import paths
- ✅ Syntax is valid Rust

---

### Story 2.2: Improve train() Documentation ✅

**Findings Addressed**: U-3.2 (Unclear Train Method Parameters)  
**Effort**: S (< 15min)  
**Status**: ✅ COMPLETE

**Description**: Clarify what the empty arrays in `train(&dataset, &labels, &[], &[])` mean.

**Tasks**:

- [x] 2.2.1 Update rustdoc on `GBDTTrainer::train()` to explain parameters
- [x] 2.2.2 Document that `&[]` means "no sample weights" and "no validation set"
- [x] 2.2.3 Add example showing non-empty validation usage

**Definition of Done**:
- ✅ `train()` rustdoc explains all parameters clearly
- ✅ Examples show basic, weighted, and validation usage

---

## Epic 3: GPU Preparation

**Goal**: Document GPU extension points to ensure 1.0 API doesn't block future GPU work.

**Why**: The audit confirmed architecture is GPU-ready (A-1.10, E-4.2). Document this for future contributors.

### Story 3.1: Create GPU Acceleration RFC

**Findings Addressed**: A-1.10 (Architecture Supports GPU), E-4.4 (Histogram Backend Needs Abstraction)  
**Effort**: M (1-2h)

**Description**: Create RFC-0020 documenting the GPU acceleration roadmap and extension points.

**Content Outline**:
1. **Phase 1: GPU Prediction** — Implement `GpuTraversal: TreeTraversal` trait
2. **Phase 2: GPU Histogram Building** — Extract `HistogramBackend` trait
3. **Phase 3: Full GPU Training** — Future scope (major refactor)

**Tasks**:

- [ ] 3.1.1 Create `docs/rfcs/0020-gpu-acceleration.md`
- [ ] 3.1.2 Document Phase 1 approach (trait-based, no core changes)
- [ ] 3.1.3 Document Phase 2 approach (histogram abstraction)
- [ ] 3.1.4 Document current extension points in codebase
- [ ] 3.1.5 Add to RFC README index

**Definition of Done**:
- RFC exists with clear phase definitions
- Links to relevant code locations
- No code changes required (documentation only)

---

## Summary

### Sprint Order

| Order | Story | Effort | Dependency |
|-------|-------|--------|------------|
| 1 | 1.1 Canonical Imports | M | None |
| 2 | 1.2 XgbModel::from_file() | S | None |
| 3 | 1.3 from_row_matrix() | S | None |
| 4 | 2.1 README Examples | S | 1.1 (uses canonical imports) |
| 5 | 2.2 train() Docs | S | None |
| 6 | 3.1 GPU RFC | M | None |

### Deferred Items (Post-1.0)

Items identified in audit but not blocking 1.0:

| Finding | Reason to Defer |
|---------|-----------------|
| Q-2.3 (Ignored doctests) | Nice-to-have, requires external setup |
| Q-2.5 (Coverage gaps) | Covered by integration tests |

### Verification Checklist

After all stories complete:

- [ ] `cargo test` — all tests pass
- [ ] `cargo test --doc` — all doctests pass
- [ ] `cargo doc --no-deps` — no warnings
- [ ] `cargo clippy` — no warnings
- [ ] Manual review of `cargo doc` output for duplicate paths
- [ ] README examples manually verified for syntax

---

**Document Status**: Ready for Implementation  
**Reviewed By**: PO, Architect, Senior Engineer, QA Engineer
