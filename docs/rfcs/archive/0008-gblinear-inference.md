# RFC-0008: GBLinear Inference

- **Status**: Approved
- **Created**: 2024-11-29
- **Depends on**: RFC-0007 (Serialization)
- **Scope**: Linear booster model loading and prediction

## Summary

Add support for XGBoost's linear booster (GBLinear) inference. Prediction is
simply weighted sums of features — no trees, just dot products.

## Motivation

- **Complete XGBoost compatibility**: Some users train with `booster='gblinear'`
- **Simple implementation**: Minimal code with clear scope
- **Fast baseline**: Linear models useful for comparison

## Design

### Model Structure

```rust
pub struct LinearModel {
    /// Flat weight array: (num_features + 1) × num_groups
    /// Layout: feature-major, group-minor. Last row is bias.
    weights: Box<[f32]>,
    num_features: usize,
    num_groups: usize,
}
```

**Weight indexing**:

```text
weights[feature * num_groups + group]     → coefficient
weights[num_features * num_groups + group] → bias
```

### Prediction

For each output group:

```text
output[g] = base_score + bias[g] + Σ(feature[i] × weight[i, g])
```

Batch prediction parallelizes over rows (independent predictions).

### Integration

```rust
pub enum Model {
    GBTree(SoAForest),
    GBLinear(LinearModel),
}
```

Same objective post-processing (sigmoid, softmax, etc.) applies to both.

## Design Decisions

### DD-1: Flat Weight Storage (`Box<[f32]>`)

Matches XGBoost's layout — direct copy from JSON, good cache locality.
`Box<[f32]>` over `Vec` since size is fixed after loading.

### DD-2: No SIMD Optimization

XGBoost doesn't use SIMD for GBLinear either. Workload is memory-bound,
sparse data is common, and the compiler auto-vectorizes dense loops.

### DD-3: Separate LinearModel Type

Not shoehorned into forest structures. Different structure, different logic,
cleaner API.

## References

- [Research: GBLinear](../research/gblinear/)
- [XGBoost Linear Booster Docs](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-linear-booster-booster-gblinear)
