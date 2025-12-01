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

## Additional Loss Functions

See [Feature Parity Analysis](../research/gblinear-feature-parity.md) for full comparison.

### High Priority

**Quantile Loss** — Pinball loss for quantile regression

- Single quantile: `L = α(y-ŷ)⁺ + (1-α)(ŷ-y)⁺`
- Multi-quantile: Train as multi-output with different α per group
- Essential for uncertainty quantification and prediction intervals

**Fix Multiclass Training** — Current implementation broken

- All output groups receive identical gradients
- Need per-group softmax gradient computation

### Medium Priority

- Huber Loss (robust regression)
- Hinge Loss (SVM-style binary classification)
- Greedy/Thrifty feature selectors

### Lower Priority

- Gamma/Tweedie deviance (insurance/count data)
- Log error (`reg:squaredlogerror`)
- Raw logit output (`binary:logitraw`)

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
