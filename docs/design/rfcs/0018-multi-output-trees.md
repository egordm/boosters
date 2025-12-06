# RFC-0018: Multi-output Trees

- **Status**: Draft
- **Created**: 2024-12-01
- **Updated**: 2024-12-01
- **Depends on**: RFC-0011 (GBTree Core), RFC-0014 (Split Finding)
- **Scope**: Multi-output/multi-class training strategies

## Summary

This RFC defines multi-output training strategies where a model produces K outputs per sample.
Use cases include multi-class classification (K classes) and multi-target regression (K targets).

**Two strategies** (following XGBoost's `multi_strategy` parameter):

1. **One output per tree** (`one_output_per_tree`): Build K separate tree ensembles, one per output.
   Current behavior for multi-class softmax.

2. **Multi-output trees** (`multi_output_tree`): Single tree with K-dimensional leaf values.
   More efficient when outputs are correlated.

## Motivation

### Efficiency for Correlated Outputs

Multi-target regression often has correlated outputs (e.g., predicting x,y,z coordinates).
Building K separate forests:

- Traverses data K times per boosting round
- May learn redundant splits
- 3-10x slower than necessary

Multi-output trees share the traversal and split structure.

### Library Compatibility

| Library | Multi-class approach | Parameter |
|---------|---------------------|-----------|
| XGBoost | Both strategies | `multi_strategy` |
| LightGBM | Multi-output trees | `num_class` |

## Design

### Strategy Selection

```rust
pub enum MultiStrategy {
    /// Build K trees per iteration (current behavior)
    OneOutputPerTree,
    /// Build single tree with K-dim leaves
    MultiOutputTree,
}
```

User selects strategy based on use case:

- Separate forests: When outputs are independent or interpretability matters
- Multi-output trees: When outputs are correlated or speed matters

### Existing Infrastructure

**`GradientBuffer`** already supports multi-output! (See `src/training/buffer.rs`)

```rust
// Already implemented:
pub struct GradientBuffer {
    grads: Vec<f32>,  // [sample * n_outputs + output]
    hess: Vec<f32>,
    n_samples: usize,
    n_outputs: usize,  // K for multi-class
}

// Already has:
buffer.sample_grads(sample)  // Returns &[f32] of length K
buffer.n_outputs()
```

### Multi-output Histogram

Extend `FeatureHistogram` to store K grad/hess sums per bin:

```rust
// Current (single output):
struct FeatureHistogram {
    grad_sum: Vec<f32>,   // [num_bins]
    hess_sum: Vec<f32>,
}

// Multi-output extension:
struct MultiOutputFeatureHistogram {
    grad_sum: Vec<f32>,   // [num_bins * K] - K values per bin
    hess_sum: Vec<f32>,
    n_outputs: usize,
}
```

### Gain Aggregation

Split gain is **sum of per-output gains**:

```
total_gain = Σ_k gain(left_grad_k, left_hess_k, right_grad_k, right_hess_k, λ) - γ
```

Single γ penalty (not K × γ) — we're making one split decision, not K.

### Leaf Values

Multi-output leaf stores K values:

```rust
// For multi_output_tree strategy
struct VectorLeafValue {
    values: Vec<f32>,  // [K]
}

// Computed from stats:
values[k] = -grad_sum[k] / (hess_sum[k] + λ)
```

### Integration with Existing Code

| Component | Current | Multi-output extension |
|-----------|---------|------------------------|
| `GradientBuffer` | Already supports K outputs | No change needed |
| `FeatureHistogram` | Single output | Add `MultiOutputFeatureHistogram` |
| `split_gain()` | Returns single f32 | Sum across K outputs |
| `BuildingNode.leaf_value` | `f32` | `Vec<f32>` for multi-output trees |
| `SoATreeStorage` | `leaves: Box<[f32]>` | `leaves: Box<[f32]>` with K per leaf |

## Design Decisions

### DD-1: Support Both Strategies

**Decision**: Implement both `one_output_per_tree` and `multi_output_tree`.

**Rationale**: Different use cases benefit from different strategies:

- Independent outputs → separate trees (more flexible, interpretable)
- Correlated outputs → shared trees (faster, better splits)

XGBoost offers both, we should too.

### DD-2: Sum Gains Across Outputs

**Decision**: Split gain = sum of per-output gains.

**Rationale**: Treats all outputs equally. A split that helps all outputs is preferred.
Single γ penalty since we're making one structural decision.

### DD-3: Reuse GradientBuffer

**Decision**: Keep existing `GradientBuffer` — it already handles multi-output.

**Rationale**: Well-tested, correct layout. Adding a separate type would duplicate code.

### DD-4: Separate MultiOutputFeatureHistogram

**Decision**: Create separate type rather than parameterizing `FeatureHistogram`.

**Rationale**: Single-output path should remain fast (no conditional branching).
Multi-output is a distinct code path used only when K > 1.

## Design Decisions (Continued)

### DD-5: No Mixed Strategy

**Decision**: Single strategy per model (no mixing).

**Rationale**: Not worth the complexity. Neither XGBoost nor LightGBM support
mixed strategies. Users can train separate models if they need different
strategies for different outputs.

### DD-6: No Per-output Learning Rates

**Decision**: Single learning rate applied to all outputs.

**Rationale**: Neither XGBoost nor LightGBM support per-output learning rates.
If outputs have different scales, users should normalize targets instead.

### DD-7: Equal Output Weighting

**Decision**: Weight all outputs equally in gain computation.

**Rationale**: Most use cases (multi-class, multi-quantile) treat all outputs
equally. Adding output weights adds complexity without clear benefit. If needed
later, can be added as a separate story.

## Testing Strategy

### Unit Tests

- Multi-output histogram accumulates K values per bin
- Gain aggregation sums correctly across outputs
- Vector leaf computation produces K values
- GradientBuffer sample_grads returns correct slice

### Integration Tests

- Multi-class classification produces valid probabilities (sum to 1)
- Multi-target regression produces reasonable predictions
- Both strategies train successfully on standard datasets

### Validation Tests

- Compare multi-class predictions against XGBoost `multi_strategy` variants
- Tolerance: predictions within 1e-2 for same hyperparameters
- If deviations exceed tolerance, compare source code implementations

### Performance Tests

- Measure training time: one_output_per_tree vs multi_output_tree
- Expected: multi_output_tree ~K× faster for correlated outputs
- Document memory usage differences

### Qualitative Tests

- Train multi-class on MNIST/IRIS, verify reasonable accuracy
- Train multi-target regression on correlated outputs
- Set accuracy expectations before training; investigate if not met

## References

- [XGBoost multi_strategy parameter](https://xgboost.readthedocs.io/en/latest/parameter.html)
- [XGBoost Multiple Outputs](https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html)
- Existing `GradientBuffer` implementation in [buffer.rs](../../src/training/buffer.rs)
