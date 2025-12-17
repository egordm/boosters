# RFC-0014: GBLinear

**Status**: Implemented

## Summary

GBLinear is a linear booster using coordinate descent with elastic net regularization. It stores weights in a flat array with feature-major layout and supports multi-output (multiclass) training.

## Design

### Model Structure

`LinearModel` stores weights in a flat `Box<[f32]>` array with feature-major, group-minor layout:

```text
weights[feature * num_groups + group] → coefficient
weights[num_features * num_groups + group] → bias (last row)
```

Total size: `(num_features + 1) × num_groups`

**Example** (2 features, 2 groups):
```rust
// [feat0_g0, feat0_g1, feat1_g0, feat1_g1, bias_g0, bias_g1]
let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
```

### Training

**Coordinate Descent**: Updates one weight at a time per feature. Two modes:
- `Sequential`: Exact gradients, updates one feature then next
- `Parallel` (shotgun): Computes all deltas in parallel, applies sequentially

**Elastic Net Regularization**:
```text
grad_l2 = Σ(gradient × feature) + λ × w
hess_l2 = Σ(hessian × feature²) + λ
delta = soft_threshold(-grad_l2 / hess_l2, α / hess_l2) × learning_rate
```

Where `soft_threshold(x, t) = sign(x) × max(|x| - t, 0)` for L1 sparsity.

**Feature Selectors** (order of updates):

| Kind | Description |
|------|-------------|
| `Cyclic` | Sequential: 0, 1, 2, ... |
| `Shuffle` | Random permutation per round (default) |
| `Random` | Random with replacement |
| `Greedy` | Largest gradient magnitude first (O(n²)) |
| `Thrifty` | Sort once per round (O(n log n)) |

**Bias Update**: No regularization applied, uses simple gradient step.

**Multi-output**: Each output group updated independently within a round. Gradients stored in SoA layout via `Gradients` struct.

### Inference

Prediction is a dot product:

```text
output[g] = base_score[g] + bias[g] + Σ(feature[i] × weight[i, g])
```

Both single-row and batch prediction supported. `par_predict_batch` uses Rayon for parallel row processing.

## Key Types

```rust
// Model storage (inference/gblinear/model.rs)
pub struct LinearModel {
    weights: Box<[f32]>,      // (num_features + 1) × num_groups
    num_features: usize,
    num_groups: usize,
}

// Trainer (training/gblinear/trainer.rs)
pub struct GBLinearTrainer<O: Objective, M: Metric> {
    objective: O,
    metric: M,
    params: GBLinearParams,
}

pub struct GBLinearParams {
    pub n_rounds: u32,           // Default: 100
    pub learning_rate: f32,      // Default: 0.5
    pub alpha: f32,              // L1, default: 0.0
    pub lambda: f32,             // L2, default: 1.0
    pub parallel: bool,          // Default: true (shotgun)
    pub feature_selector: FeatureSelectorKind,
    pub early_stopping_rounds: u32,
    // ...
}

// Updaters (training/gblinear/updater.rs)
pub enum UpdaterKind {
    Sequential,  // Exact gradients
    Parallel,    // Shotgun (stale gradients, faster)
}

// Feature selectors (training/gblinear/selector.rs)
pub enum FeatureSelectorKind {
    Cyclic,
    Shuffle,           // Default
    Random,
    Greedy { top_k: usize },
    Thrifty { top_k: usize },
}
```

## Notes

- Requires column-major data (`ColMatrix`) for efficient feature iteration
- Gradients computed fresh each round (not truly "stale" like some implementations)
- Parallel mode computes all deltas from same gradient state, applies sequentially
- `top_k` in Greedy/Thrifty limits features updated per round
