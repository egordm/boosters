# Backlog 10: Stakeholder Feedback Items

**RFC**: Various  
**Priority**: Medium  
**Status**: In Progress

---

## Overview

Follow-up items from stakeholder feedback received during Backlog 09 completion.

---

## Story 10.1: Performance Comparison Benchmarks ✅

Run comparison benchmarks against XGBoost and LightGBM to confirm performance advantage.

**Status**: Completed (informal verification)

**Result**: boosters competitive with XGBoost/LightGBM:
- Training: ~1.6s (XGBoost ~2.1s, LightGBM ~1.55s)
- Prediction: Faster than XGBoost across thread counts

---

## Story 10.2: Serialization Format Removal ✅

**Status**: Completed (commit `a706775`)

Removed native serialization format to enable clean redesign:
- Deleted io/native.rs, io/payload.rs, io/convert.rs
- Removed storage and storage-compression features
- Removed postcard and crc32fast dependencies
- Removed save/load/to_bytes/from_bytes from models
- Deleted native_format.rs tests and test-cases/native/

**Decision**: Clean slate for future serialization RFC.

---

## Story 10.3: Prediction API Alignment with RFC-0020

**Context**: RFC-0020 specifies `predict()` and `predict_raw()` as the only batch prediction methods.
Both methods assume batch prediction on a dataset (not single-row).

Current API has:
- `predict_row()` - Single row (returns Vec<f32>)
- `predict_batch()` - Batch (returns flat Vec<f32>)

RFC-0020 specifies:
- `predict()` - Returns ColMatrix<f32>, applies transformation
- `predict_raw()` - Returns ColMatrix<f32>, no transformation

**Tasks**:

- [ ] 10.3.1: Team discussion on prediction API alignment
- [ ] 10.3.2: Implement `predict()` returning `ColMatrix<f32>`
- [ ] 10.3.3: Implement `predict_raw()` returning `ColMatrix<f32>`
- [ ] 10.3.4: Evaluate whether to keep/deprecate/remove `predict_row()` and `predict_batch()`
- [ ] 10.3.5: Update tests and docs

**Definition of Done**:

- `predict()` and `predict_raw()` implemented per RFC-0020
- Decision made on legacy methods
- Tests passing, clippy clean

---

> Note: Story 10.3 supersedes original Story 10.3 (API naming for serialization).
