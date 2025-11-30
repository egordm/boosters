# Backlog: Future Work

**Status**: 📋 Backlog  
**Priority**: Low — defer until core features complete

Items here are parked until there's capacity or user demand.

---

## Sparse Data Support

- `SparseMatrix` (CSR format)
- `DataMatrix` impl for sparse
- Sparse-aware traversal
- Benchmark sparse vs dense

---

## Native Serialization

- `ModelSchema` for stable interchange
- `bincode` serialization
- `save()` / `load()` API

---

## Additional Formats

- XGBoost binary (.bin) loader
- XGBoost UBJSON loader

---

## LightGBM Support

- Research LightGBM model format
- JSON/binary loader
- Linear trees (leaf contains linear model)

**Note**: Research after GBTree training to understand abstraction needs.

---

## Python Bindings

- `booste-rs-python` crate (PyO3)
- NumPy array input

---

## Arrow Integration

- `ArrowMatrix` implementing `DataMatrix`
- Zero-copy from PyArrow/polars

---

## CLI Tool

- Model inspection
- Format conversion
- Benchmark runner

---

## GBTree Training

- Histogram-based split finding
- Tree growing algorithms
- Major undertaking — defer until GBLinear validates infrastructure

---

## Notes

When picking up an item, create a new epic file with detailed stories.
