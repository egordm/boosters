# RFC-0026: Sample Weighting

**Status**: Approved  
**Created**: 2024-12-02  
**Updated**: 2024-12-03  
**Depends on**: RFC-0012 (Histogram Building), RFC-0015 (Tree Growing)

## Summary

Add support for per-instance sample weights that affect the loss function and
gradient computation during training. This is a standard feature in XGBoost and
LightGBM, commonly used for class imbalance, importance sampling, and survey data.

## Motivation

Sample weights are essential for many real-world ML applications:

1. **Class imbalance**: Weight minority class samples higher
2. **Importance sampling**: Emphasize recent or high-value samples
3. **Survey data**: Account for sampling design weights
4. **Curriculum learning**: Gradually increase difficulty
5. **Boosting variants**: AdaBoost-style instance reweighting

Without sample weights, users must resort to oversampling (memory inefficient)
or custom loss functions (complex and error-prone).

**XGBoost/LightGBM parity**: Both support `weight` parameter in DMatrix/Dataset.

## Design Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Sample Weight Flow                           │
│                                                                 │
│  train(features, labels, weights)                               │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Gradient Computation (Loss trait)                          ││
│  │                                                             ││
│  │  Without weights:  grad[i] = ∂L/∂pred[i]                    ││
│  │  With weights:     grad[i] = weight[i] * ∂L/∂pred[i]        ││
│  │                    hess[i] = weight[i] * ∂²L/∂pred[i]²      ││
│  └─────────────────────────────────────────────────────────────┘│
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Histogram Building (unchanged)                             ││
│  │                                                             ││
│  │  sum_grad[bin] += grad[row]    // Already weighted          ││
│  │  sum_hess[bin] += hess[row]    // Already weighted          ││
│  │  count[bin] += 1               // Unweighted count          ││
│  └─────────────────────────────────────────────────────────────┘│
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Split Finding & Evaluation (unchanged)                     ││
│  │                                                             ││
│  │  Gain formula uses sum_grad/sum_hess (already weighted)     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: Weights are applied at gradient computation time, so histogram
building, split finding, and tree growing remain unchanged.

## Detailed Design

### Loss Trait Extension

We extend the existing `Loss` trait (in `src/training/loss/mod.rs`) to accept
optional weights. The trait currently has:

```rust
// Current signature
fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer);
```

**New signature**:

```rust
/// Loss function trait for gradient boosting.
/// 
/// Note: This trait was previously named `Loss` in our codebase.
/// We document it as `Loss` here since that's the current name.
pub trait Loss: Send + Sync {
    fn num_outputs(&self) -> usize;
    
    /// Compute gradients and hessians for all samples.
    ///
    /// If `weights` is Some, multiply gradients and hessians by the weight.
    fn compute_gradients(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,  // NEW parameter
        buffer: &mut GradientBuffer,
    );
    
    /// Compute initial base scores, optionally weighted.
    /// 
    /// Already implemented per RFC-0024 (base score init).
    fn init_base_score(&self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32>;
    
    fn name(&self) -> &'static str;
}
```

### Implementation Examples

**SquaredLoss** (currently in `src/training/loss/regression.rs`):

```rust
impl Loss for SquaredLoss {
    fn compute_gradients(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        buffer: &mut GradientBuffer,
    ) {
        let (grads, hess) = buffer.as_mut_slices();
        
        match weights {
            Some(w) => {
                for i in 0..preds.len() {
                    let residual = preds[i] - labels[i];
                    grads[i] = w[i] * residual;
                    hess[i] = w[i];
                }
            }
            None => {
                for i in 0..preds.len() {
                    grads[i] = preds[i] - labels[i];
                    hess[i] = 1.0;
                }
            }
        }
    }
}
```

**LogisticLoss** (currently in `src/training/loss/classification.rs`):

```rust
impl Loss for LogisticLoss {
    fn compute_gradients(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        buffer: &mut GradientBuffer,
    ) {
        let (grads, hess) = buffer.as_mut_slices();
        
        match weights {
            Some(w) => {
                for i in 0..preds.len() {
                    let pred = sigmoid(preds[i]);
                    grads[i] = w[i] * (pred - labels[i]);
                    hess[i] = w[i] * pred * (1.0 - pred).max(1e-6);
                }
            }
            None => {
                for i in 0..preds.len() {
                    let pred = sigmoid(preds[i]);
                    grads[i] = pred - labels[i];
                    hess[i] = pred * (1.0 - pred).max(1e-6);
                }
            }
        }
    }
}
```

### Trainer API

The trainer accepts weights as an optional parameter:

```rust
impl<L: Loss> GBTreeTrainer<L> {
    /// Train with optional sample weights.
    pub fn train<M: DataMatrix>(
        &self,
        features: &M,
        labels: &[f32],
        weights: Option<&[f32]>,
        params: &TrainParams,
    ) -> Result<Forest, TrainError> {
        // Validate weights
        if let Some(w) = weights {
            if w.len() != labels.len() {
                return Err(TrainError::WeightLengthMismatch {
                    expected: labels.len(),
                    got: w.len(),
                });
            }
        }
        
        // Compute weighted base score (RFC-0024 already supports this)
        let base_scores = self.loss.init_base_score(labels, weights);
        
        // Training loop - pass weights to loss
        for round in 0..params.num_rounds {
            self.loss.compute_gradients(&predictions, labels, weights, &mut buffer);
            // ... rest unchanged
        }
    }
}
```

### Weighted Base Score

RFC-0024 (base score init) is already implemented and includes the `weights`
parameter in `init_base_score`. This RFC uses that functionality:

- For `SquaredLoss`: weighted mean of labels
- For `LogisticLoss`: weighted log-odds of positive rate
- For `SoftmaxLoss`: weighted class frequencies → log probabilities

### Evaluation Metrics

Metrics should also support weighted computation. The existing `Metric` trait
(if any) or metric functions gain an optional weights parameter:

```rust
/// Compute weighted RMSE.
pub fn rmse(predictions: &[f32], labels: &[f32], weights: Option<&[f32]>) -> f32 {
    match weights {
        Some(w) => {
            let (sum_sq, sum_w) = predictions.iter()
                .zip(labels)
                .zip(w)
                .fold((0.0, 0.0), |(ss, sw), ((&p, &l), &w)| {
                    (ss + w * (p - l).powi(2), sw + w)
                });
            (sum_sq / sum_w).sqrt()
        }
        None => {
            let mse: f32 = predictions.iter()
                .zip(labels)
                .map(|(p, l)| (p - l).powi(2))
                .sum::<f32>() / predictions.len() as f32;
            mse.sqrt()
        }
    }
}
```

## Design Decisions

### DD-1: Where to Apply Weights

**Context**: Weights could be applied at different points in the pipeline.

**Options considered**:

1. **At gradient computation** (chosen): Multiply grad/hess by weight
2. **At histogram accumulation**: Add weight to accumulator
3. **At split finding**: Weight the gain formula

**Decision**: Apply at gradient computation.

**Rationale**:

- Simplest implementation — single point of change
- Matches XGBoost/LightGBM behavior
- Histogram building remains unchanged (already sums gradients)
- Split finding formula unchanged (works on sums)
- Mathematically equivalent to other approaches

### DD-2: API Design — TrainingData struct vs Separate Parameters

**Context**: How to pass weights to the trainer?

**Options considered**:

1. **Bundled struct** (`TrainingData { features, labels, weights }`):
   - Pro: Single argument, extensible
   - Con: Requires `dyn DataMatrix` or generics on struct
   - Con: Additional indirection

2. **Separate parameters** (chosen): `train(features, labels, weights, params)`
   - Pro: Zero-cost, no trait objects needed
   - Pro: Generic over `M: DataMatrix` naturally
   - Pro: Matches current API style
   - Con: More parameters

3. **Builder pattern**: `Trainer::new().features(f).labels(l).weights(w).train()`
   - Pro: Very flexible
   - Con: More complex, runtime errors for missing fields

**Decision**: Separate parameters with generics.

**Rationale**:

- Avoids `dyn DataMatrix` overhead (virtual dispatch)
- Consistent with existing trainer API
- Clear and explicit at call site
- Easy to add more optional parameters later (just add more `Option<>`)

### DD-3: Weight Normalization

**Context**: Should weights be normalized to sum to N?

**Options considered**:

1. **No normalization** (chosen): Use weights as-is
2. **Normalize to N**: `w_i' = w_i * N / sum(w)`
3. **Normalize to 1**: `w_i' = w_i / sum(w)`

**Decision**: No normalization (match XGBoost/LightGBM).

**Rationale**:

- Users can normalize if desired
- Regularization strength changes with total weight — intentional
- XGBoost/LightGBM don't normalize

### DD-4: Histogram Count Handling

**Context**: Should histogram `count` be weighted or unweighted?

**Options considered**:

1. **Unweighted count** (chosen): `count[bin] += 1`
2. **Weighted count**: `count[bin] += weight[row]`

**Decision**: Unweighted count.

**Rationale**:

- `min_child_weight` uses sum_hess (already weighted)
- Unweighted count useful for debugging/analysis
- Matches XGBoost behavior

### DD-5: Zero/Negative Weights

**Context**: How to handle zero or negative weights?

**Decision**: Allow but warn for negative.

**Rationale**:

- Zero weights effectively exclude samples (useful)
- Negative weights are unusual but mathematically valid
- Warning helps catch bugs without restricting valid use cases

### DD-6: Separate Code Paths for Weighted/Unweighted

**Context**: Should we have separate code paths for weighted vs unweighted?

**Options considered**:

1. **Single code path with match**: `match weights { Some(w) => ..., None => ... }`
2. **Unified with default weights**: Always iterate, use 1.0 if no weights
3. **Separate functions**: `compute_gradients` and `compute_gradients_weighted`

**Decision**: Single code path with match (option 1).

**Rationale**:

- Gradient computation runs O(rounds × samples), not the hottest path
- Histogram building dominates training time
- Code duplication from option 3 not worth the small speedup
- Match is cleaner than always iterating with default (option 2)
- If profiling shows this matters, can optimize later

## Integration

### Changes Required

| Component | Change | Notes |
|-----------|--------|-------|
| `Loss` trait | Add `weights` parameter to `compute_gradients` | Breaking change |
| `SquaredLoss`, `LogisticLoss`, etc. | Implement weighted gradients | Update all impls |
| `SoftmaxLoss` | Implement weighted gradients | Multi-output case |
| `GBTreeTrainer::train` | Add `weights` parameter | API change |
| `init_base_score` | Already has weights (RFC-0024) | No change |
| `HistogramBuilder` | Unchanged | Gradients already weighted |
| `SplitFinder` | Unchanged | Works on weighted sums |
| `TreeGrower` | Unchanged | No weight awareness needed |
| Metrics (`rmse`, `logloss`, etc.) | Add weights parameter | New functionality |

### XGBoost/LightGBM Compatibility

| Feature | XGBoost | LightGBM | This RFC |
|---------|---------|----------|----------|
| Weight parameter | `DMatrix(weight=...)` | `Dataset(weight=...)` | `train(..., weights)` |
| Where applied | Gradient computation | Gradient computation | Gradient computation |
| Normalization | None | None | None |
| Negative weights | Allowed | Allowed | Allowed (with warning) |
| Weighted metrics | Yes | Yes | Yes |
| Weighted base score | Yes | Yes | Yes (RFC-0024) |

## Testing Strategy

1. **Unit tests**: Weighted gradient computation for each loss function
2. **Property tests**: Uniform weights (all 1.0) produces identical results to no weights
3. **Numerical tests**: Doubling weights doubles gradients
4. **Integration tests**: Train with weights, verify convergence
5. **Compatibility tests**: Compare predictions to XGBoost/LightGBM with same weights

## Open Questions

1. **Weighted quantile computation for bin cuts?**
   - Currently bin cuts use uniform quantiles
   - Weighted quantiles would better represent weighted distribution
   - Defer to future work (minor impact on accuracy)

## Future Work

- [ ] Weighted quantile computation for bin cuts
- [ ] `scale_pos_weight` parameter (class imbalance shorthand)
- [ ] Per-class weights for multiclass
- [ ] Dynamic weight updates (boosting variants)

## References

- [XGBoost weight parameter](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.DMatrix)
- [LightGBM weight parameter](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Dataset.html)
- RFC-0024: Base Score Initialization (implemented, includes weighted base score)

## Changelog

- 2024-12-02: Initial draft
- 2024-12-02: Removed `TrainingData` struct in favor of separate parameters (DD-2)
- 2024-12-02: Updated to use existing `Loss` trait name (not `Objective`)
- 2024-12-02: Added DD-6 on code path separation trade-offs
- 2024-12-02: Clarified weighted base score uses RFC-0024 (already implemented)
- 2024-12-02: Simplified testing section to text descriptions
- 2024-12-03: Status → Approved
