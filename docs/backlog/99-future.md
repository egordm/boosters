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

## Training Performance Optimizations

**Context**: The core coordinate descent loop is a weighted dot product:

```rust
for (row, value) in column {
    sum_grad += gradients[row].grad() * value;
    sum_hess += gradients[row].hess() * value * value;
}
```

### SIMD / BLAS Acceleration

- Dense columns are contiguous — classic dot product pattern
- Consider `faer`, `ndarray`, or `nalgebra` for optimized BLAS routines
- Alternatively, use `std::simd` (nightly) or `wide` crate for manual SIMD
- Sparse matrices have indirect indexing (harder to vectorize)

### Gradient Storage Layout (SoA vs AoS)

Currently: `Vec<GradientPair>` (Array of Structs, 8 bytes per sample)

Alternative: Separate `Vec<f32>` for gradients and hessians (SoA)

- May enable better SIMD vectorization
- Better cache utilization when only one component needed
- Benchmark to validate benefit

---

## Notes

When picking up an item, create a new epic file with detailed stories.
