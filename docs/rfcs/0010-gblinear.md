# RFC-0010: GBLinear

**Status**: Implemented  
**Created**: 2025-12-01  
**Updated**: 2026-01-02  
**Scope**: Gradient boosted linear model training and inference

## Summary

GBLinear trains a linear model using gradient boosting with coordinate descent.
It's an alternative to GBDT when interpretability or linear relationships are
desired. Compatible with XGBoost's gblinear booster.

## Why GBLinear?

| Aspect | GBDT | GBLinear |
| ------ | ---- | -------- |
| Model | Trees | Linear weights |
| Complexity | Non-linear | Linear |
| Interpretability | Feature importance | Coefficients |
| Training speed | Slower (histogram building) | Faster (coordinate descent) |
| Accuracy | Usually better | Good for linear data |

GBLinear is useful when:
- Features have roughly linear relationships with target
- Model interpretability is important
- Training speed matters and trees aren't necessary

## Layers

### High Level

Users call `GBLinearModel::train`:

```rust
let model = GBLinearModel::train(&dataset, eval_set, config, seed)?;
let predictions = model.predict(&test_data, n_threads);
```

### Medium Level (Trainer)

```rust
pub struct GBLinearTrainer<O: ObjectiveFn, M: MetricFn> {
    objective: O,
    metric: M,
    params: GBLinearParams,
    updater: Updater,
}

impl<O, M> GBLinearTrainer<O, M> {
    pub fn train(
        &self,
        train: &Dataset,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        val_set: Option<&Dataset>,
    ) -> Option<LinearModel>;
}
```

Returns `None` if dataset contains categorical features (unsupported).

### Medium Level (Model)

```rust
pub struct LinearModel {
    weights: Array2<f32>,  // Shape: [n_features + 1, n_groups]
}

impl LinearModel {
    pub fn weight(&self, feature: usize, group: usize) -> f32;
    pub fn bias(&self, group: usize) -> f32;
    pub fn predict(&self, data: &Dataset, parallelism: Parallelism) -> Array2<f32>;
}
```

Weights layout: rows 0..n_features are coefficients, last row is bias.

### Low Level (Updater)

```rust
pub struct Updater {
    strategy: UpdateStrategy,
    alpha: f32,      // L1 regularization
    lambda: f32,     // L2 regularization
    learning_rate: f32,
    max_delta_step: f32,
}

impl Updater {
    pub fn compute_delta(
        &self,
        grad_sum: f64,
        hess_sum: f64,
    ) -> f32;
}
```

## Coordinate Descent

GBLinear optimizes one weight at a time:

```text
For each round:
    Compute gradients from objective
    For each feature (selected by feature_selector):
        sum_g = Σ grad[i] * x[i, feature]
        sum_h = Σ hess[i] * x[i, feature]²
        delta = compute_delta(sum_g, sum_h)
        weight[feature] += delta
        Update predictions: pred += delta * x[:, feature]
    Update bias similarly
```

### Update Strategies

```rust
pub enum UpdateStrategy {
    /// Full Newton step: Δw = -learning_rate * sum_g / (sum_h + λ)
    Standard,
    /// Shotgun: Same as Standard, updates all features in parallel
    Shotgun,
}
```

Shotgun uses parallel feature updates (approximation). Standard is sequential
but more accurate for correlated features.

**Performance**: Shotgun is ~3-5× faster than Standard on many-feature datasets.
May require more rounds to converge with highly correlated features.

**Convergence**: Training runs for exactly `n_rounds`. No early convergence
detection within coordinate descent—this matches XGBoost behavior.

### Feature Selection

```rust
pub enum FeatureSelectorKind {
    /// Cyclic: iterate 0, 1, ..., n_features-1
    Cyclic,
    /// Shuffle: random permutation each round
    Shuffle { seed: u64 },
    /// Random: sample with replacement
    Random { seed: u64 },
}
```

Shuffle typically converges faster than cyclic by reducing coordinate dependencies.

## Parameters

```rust
pub struct GBLinearParams {
    pub n_rounds: u32,           // Boosting rounds
    pub learning_rate: f32,      // Step size
    pub alpha: f32,              // L1 regularization
    pub lambda: f32,             // L2 regularization
    pub update_strategy: UpdateStrategy,
    pub feature_selector: FeatureSelectorKind,
    pub max_delta_step: f32,     // Clip step size
    pub early_stopping_rounds: u32,
    pub verbosity: Verbosity,
    pub seed: u64,
}
```

### Weight Access

```rust
impl LinearModel {
    // Full weight matrix access
    pub fn as_slice(&self) -> &[f32];        // Flat view [n_features + 1, n_groups]
    pub fn weight_view(&self) -> ArrayView2<f32>;  // [n_features, n_groups] excluding bias

    // Per-group combined view (preferred for training updates)
    pub fn weights_and_bias(&self, group: usize) -> ArrayView1<f32>;     // [n_features + 1]
    pub fn weights_and_bias_mut(&mut self, group: usize) -> ArrayViewMut1<f32>;

    // Individual element access
    pub fn weight(&self, feature: usize, group: usize) -> f32;
    pub fn bias(&self, group: usize) -> f32;
    pub fn biases(&self) -> ArrayView1<f32>;  // [n_groups]
}
```

The `weights_and_bias(group)` view is the primary API for training updates—it
returns a contiguous slice `[w_0, w_1, ..., w_{n-1}, bias]` for a single output group.
Individual `weight()` and `bias()` methods exist for inspection.

## Multi-Output

For K-class classification:

- Weights shape: `[n_features + 1, K]`
- Each output group has its own weight vector
- Coordinate descent updates all K outputs per feature

## Prediction

Linear prediction is matrix multiplication:

```rust
impl LinearModel {
    pub fn predict(&self, data: &Dataset, parallelism: Parallelism) -> Array2<f32> {
        // output = data · weights[:-1, :] + bias
        let mut output = data.features.dot(&self.weight_matrix());
        output += &self.biases();
        output
    }
}
```

Parallelized via ndarray's built-in BLAS bindings or rayon for row blocks.

## Files

| Path | Contents |
| ---- | -------- |
| `repr/gblinear/model.rs` | `LinearModel`, weight storage, prediction |
| `training/gblinear/trainer.rs` | `GBLinearTrainer`, `GBLinearParams` |
| `training/gblinear/updater.rs` | `Updater`, delta computation |
| `training/gblinear/selector.rs` | `FeatureSelectorKind`, feature selection |
| `model/gblinear.rs` | `GBLinearModel` high-level wrapper |

## Design Decisions

**DD-1: No categorical support.** GBLinear requires numeric features for
meaningful linear coefficients. If categoricals are needed, users should
one-hot encode or use GBDT.

**DD-2: XGBoost-compatible layout.** Weight storage matches XGBoost's gblinear
for model interoperability.

**DD-3: Shotgun parallelism.** Shotgun assumes feature updates are independent
(approximately true with regularization). Faster but may need more rounds
for highly correlated features.

**DD-4: L1 via soft thresholding.** L1 regularization uses soft thresholding
formula: `sign(x) * max(|x| - α, 0)` for sparse solutions.

## XGBoost Differences

| Aspect | Boosters | XGBoost |
| ------ | -------- | ------- |
| Feature selection | Cyclic, Shuffle, Random | Same options |
| Update strategy | Standard, Shotgun | coord_descent, shotgun |
| Regularization | L1 + L2 | Same |
| Multi-output | Supported | Supported |

Behavior is designed to match XGBoost for model interoperability.

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Coordinate descent | Single-feature convergence to optimal |
| L1 sparsity | Zero weights with sufficient α |
| XGBoost compat | Load XGBoost gblinear, compare predictions |
| Multi-output | K-class weights correctly shaped |
