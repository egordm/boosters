# Backlog 10: Stakeholder Feedback Items

**RFC**: Various  
**Priority**: Medium  
**Status**: Complete

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

## Story 10.3: Prediction API Alignment with RFC-0020 ✅

**Status**: Completed (commit `9c7cd23`)

**Result**: `predict()` and `predict_raw()` now accept `impl DataMatrix<Element = f32>`:
- Both return `ColMatrix<f32>` (unchanged)
- Works with any matrix layout (RowMatrix, ColMatrix, etc.)
- `predict_row()` and `predict_batch()` marked as legacy in docs
