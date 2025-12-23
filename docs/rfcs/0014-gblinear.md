# RFC-0014: GBLinear

- **Status**: Implemented
- **Created**: 2025-12-17
- **Updated**: 2025-01-21
- **Scope**: Linear booster with coordinate descent

## Summary

GBLinear is a linear booster using coordinate descent with elastic net regularization. It stores weights in a flat array with feature-major layout and supports multi-output (multiclass) training.

## Design

### Model Structure

`LinearModel` stores weights in an `Array2<f32>` with shape `[n_features + 1, n_groups]`:

```text
weights[[feature, group]] → coefficient
weights[[n_features, group]] → bias (last row)
```

Total size: `(n_features + 1) × n_groups`

**Example** (3 features, 2 groups):

```rust
use ndarray::array;
let weights = array![
    [0.1, 0.2],  // feature 0: group 0, group 1
    [0.3, 0.4],  // feature 1
    [0.5, 0.6],  // feature 2
    [0.0, 0.0],  // bias
];
let model = LinearModel::new(weights);
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

Both single-row and batch prediction supported. `par_predict` uses Rayon for parallel row processing.

## Key Types

### High-Level API (model layer)

For most users, the `GBLinearModel` in `model::gblinear` provides the primary interface:

```rust
// High-level model (model/gblinear/model.rs)
pub struct GBLinearModel {
    linear: LinearModel,      // Weight storage
    meta: ModelMeta,          // n_features, n_groups
    config: GBLinearConfig,   // Objective, defaults
}

impl GBLinearModel {
    pub fn predict(&self, features: ArrayView2<f32>) -> Array2<f32>;
    pub fn predict_raw(&self, features: ArrayView2<f32>) -> Array2<f32>;
    pub fn n_features(&self) -> usize;
    pub fn n_groups(&self) -> usize;
}
```

### Low-Level API (repr layer)

For advanced use (custom training loops, model surgery):

```rust
// Model storage (repr/gblinear/model.rs)
pub struct LinearModel {
    weights: Array2<f32>,  // Shape [n_features + 1, n_groups]
}

impl LinearModel {
    pub fn new(weights: Array2<f32>) -> Self;
    pub fn zeros(n_features: usize, n_groups: usize) -> Self;
    pub fn n_features(&self) -> usize;
    pub fn n_groups(&self) -> usize;
    pub fn weight(&self, feature: usize, group: usize) -> f32;
    pub fn bias(&self, group: usize) -> f32;
    pub fn predict(&self, data: SamplesView<'_>, base_score: &[f32]) -> Array2<f32>;
    pub fn predict_into(
        &self,
        data: SamplesView<'_>,
        base_score: &[f32],
        parallelism: Parallelism,
        output: ArrayViewMut2<'_, f32>,
    );
    pub fn predict_row_into(&self, features: &[f32], base_score: &[f32], output: &mut [f32]);
}
```

### Training Types

```rust
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

- Requires feature-major data (`FeaturesView`) for efficient feature iteration
- Gradients computed fresh each round (not truly "stale" like some implementations)
- Parallel mode computes all deltas from same gradient state, applies sequentially
- `top_k` in Greedy/Thrifty limits features updated per round

## Changelog

- 2025-01-23: Added dual-layer API documentation (model vs repr), added predict_row_into, fixed storage type
- 2025-01-23: Fixed `ColMatrix` → `FeaturesView` to match current implementation
- 2025-01-21: Terminology update — `num_features`→`n_features`, `num_groups`→`n_groups`
