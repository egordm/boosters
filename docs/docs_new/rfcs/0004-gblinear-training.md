# RFC-0004: GBLinear Training

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: RFC-0001 (Data Matrix), RFC-0002 (Objectives), RFC-0005 (Linear Model), RFC-0006 (Evaluation)
- **Scope**: Coordinate descent training for linear gradient boosting

## Summary

GBLinear trains a linear model via gradient boosting with coordinate descent. Each boosting round computes gradients, then updates the model weights one feature at a time. This RFC covers the training loop, coordinate descent algorithms (Sequential and Shotgun), and feature selection strategies.

## Overview

### Component Hierarchy

```text
GBLinearTrainer<O: Objective, F: FeatureSelector>
│
├── objective: O                          ← Gradient computation
│
├── updater: CoordDescentUpdater<F>
│   ├── selector: F                       ← Feature iteration order
│   └── params: UpdateParams              ← alpha, lambda, learning_rate
│
├── strategy: UpdateStrategy              ← Sequential or Shotgun
│
└── training_params: TrainingParams       ← n_rounds, seed

Feature Selectors:
├── CyclicSelector         ← 0, 1, 2, ...
├── ShuffleSelector        ← Random permutation
├── RandomSelector         ← Random with replacement
├── GreedySelector         ← Largest gradient first
└── ThriftySelector        ← Approximate greedy
```

### Data Flow

```text
features [n_samples × n_features]  ──┐
labels [n_samples × n_labels]      ──┼─► Training Loop
weights [n_samples × 1] (opt)      ──┘        │
                                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  for round in 0..n_rounds:                                      │
│    objective.compute_gradients(preds, ..., &mut grads, &mut hess)│
│    for output_idx in 0..n_outputs:                              │
│      updater.update(model, data, &grads, &hess, output_idx)     │
│    preds = model.predict_into(features)                         │
│    match evaluator.evaluate(round, ...):                        │
│      NewBest => best_model = model.clone()                      │
│      Stop => break                                              │
└─────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                                         LinearModel (RFC-0005)
```

### Coordinate Descent vs Tree Growing

| Aspect | GBLinear | GBTree (RFC-0007) |
|--------|----------|-------------------|
| Model | Linear weights | Decision trees |
| Update unit | One feature at a time | One tree per round |
| Complexity | O(n_samples × n_features) per round | O(n_samples × n_features × n_bins) |
| Interpretability | High (coefficients) | Medium (feature importance) |
| Non-linearity | None (unless feature engineering) | Captures automatically |

## Components

### GBLinearTrainer

```rust
pub struct GBLinearTrainer<
    O: Objective = ObjectiveFunction,
    F: FeatureSelector = ShuffleSelector,
> {
    pub objective: O,
    pub updater: CoordDescentUpdater<F>,
    pub strategy: UpdateStrategy,
    pub training_params: TrainingParams,
}

pub struct TrainingParams {
    pub n_rounds: usize,
    pub seed: u64,
}
```

### CoordDescentUpdater

Owns the selector and regularization parameters:

```rust
pub struct CoordDescentUpdater<F: FeatureSelector = ShuffleSelector> {
    pub selector: F,
    pub params: UpdateParams,
}

pub struct UpdateParams {
    pub alpha: f32,          // L1 regularization (lasso)
    pub lambda: f32,         // L2 regularization (ridge)
    pub learning_rate: f32,
}

impl<F: FeatureSelector> CoordDescentUpdater<F> {
    /// Update bias and weights for one output.
    pub fn update<S: AsRef<[f32]> + Sync>(
        &mut self,
        model: &mut LinearModel,
        data: &ColMatrix<f32, S>,
        grads: &ColMatrix<f32>,
        hess: &ColMatrix<f32>,
        output_idx: usize,
        strategy: UpdateStrategy,
    );
}
```

### UpdateStrategy

```rust
#[derive(Clone, Copy, Debug)]
pub enum UpdateStrategy {
    /// Update features one at a time, gradients always exact.
    Sequential,
    /// Compute all feature deltas in parallel, then apply.
    /// Slight approximation due to stale gradients, but faster.
    Shotgun,
}
```

## Algorithms

### Training Loop

```text
train(dataset, eval_sets, evaluator) -> LinearModel:
  n_samples = dataset.n_samples()
  n_features = dataset.n_features()
  n_outputs = objective.n_outputs(dataset.n_labels())
  
  // Initialize model with base scores
  model = LinearModel::zeros(n_features, n_outputs)
  base_scores = objective.base_score_vec(dataset.labels, dataset.weights)
  for k in 0..n_outputs:
    model.set_bias(k, base_scores[k])
  
  // Allocate buffers (separate matrices, consistent with Objective trait)
  predictions = ColMatrix::zeros(n_samples, n_outputs)
  grads = ColMatrix::zeros(n_samples, n_outputs)
  hess = ColMatrix::zeros(n_samples, n_outputs)
  best_model: Option<LinearModel> = None
  
  // Initialize predictions from model
  model.predict_into(dataset.features, &mut predictions)
  
  for round in 0..training_params.n_rounds:
    // Compute gradients for all outputs
    objective.compute_gradients(
      &predictions, dataset.labels, dataset.weights,
      &mut grads, &mut hess
    )
    
    // Update each output independently
    for output_idx in 0..n_outputs:
      updater.update(&mut model, dataset.features, &grads, &hess, output_idx, strategy)
    
    // Update predictions for next round
    model.predict_into(dataset.features, &mut predictions)
    
    // Evaluation with action-based flow (RFC-0006)
    match evaluator.evaluate(round, &predictions, eval_sets):
      EvalAction::Skip => {}
      EvalAction::Continue { results } => log_results(results)
      EvalAction::NewBest { results } => {
        log_results(results)
        best_model = Some(model.clone())
      }
      EvalAction::Stop { results } => {
        log_results(results)
        break
      }
  
  // Return best model if early stopping was used
  return best_model.unwrap_or(model)
```

### Sequential Update

Updates bias, then features one at a time. After each update, gradients remain exact.

```text
sequential_update(model, data, grads, hess, output_idx, params):
  // 1. Bias update (no regularization)
  sum_grad = sum(grads.col_slice(output_idx))
  sum_hess = sum(hess.col_slice(output_idx))
  if sum_hess > ε:
    delta_bias = -sum_grad / sum_hess * params.learning_rate
    model.add_bias(output_idx, delta_bias)
  
  // 2. Feature updates
  selector.setup(model, data, grads, hess, output_idx, params)
  for feature in selector:
    delta = compute_weight_delta(model, data, grads, hess, feature, output_idx, params)
    if abs(delta) > ε:
      model.add_weight(feature, output_idx, delta)
```

### Shotgun Update

Computes all feature deltas in parallel, then applies. Faster but uses slightly stale gradients.

```text
shotgun_update(model, data, grads, hess, output_idx, params):
  // 1. Bias update
  sum_grad = sum(grads.col_slice(output_idx))
  sum_hess = sum(hess.col_slice(output_idx))
  if sum_hess > ε:
    delta_bias = -sum_grad / sum_hess * params.learning_rate
    model.add_bias(output_idx, delta_bias)
  
  // 2. Collect features to update
  selector.setup(model, data, grads, hess, output_idx, params)
  features: Vec<u32> = selector.collect()
  
  // 3. Parallel: compute all deltas (using current gradients)
  deltas = features.par_iter().map(|&f|
    compute_weight_delta(model, data, grads, hess, f, output_idx, params)
  ).collect()
  
  // 4. Sequential: apply deltas
  for (feature, delta) in zip(features, deltas):
    if abs(delta) > ε:
      model.add_weight(feature, output_idx, delta)
```

### Weight Delta Computation

Elastic net coordinate descent with soft-thresholding for L1:

```text
compute_weight_delta(model, data, grads, hess, feature, output_idx, params) -> f32:
  w = model.weight(feature, output_idx)
  grad_col = grads.col_slice(output_idx)
  hess_col = hess.col_slice(output_idx)
  feature_col = data.col_slice(feature)  // contiguous access
  
  // Accumulate gradient/hessian contribution
  sum_grad = 0.0
  sum_hess = 0.0
  for i in 0..n_samples:
    x = feature_col[i]
    sum_grad += grad_col[i] * x
    sum_hess += hess_col[i] * x * x
  
  // L2 regularization: add to both gradient and hessian
  grad_l2 = sum_grad + params.lambda * w
  hess_l2 = sum_hess + params.lambda
  
  if hess_l2 < ε:
    return 0.0
  
  // Raw update (Newton step)
  raw_delta = -grad_l2 / hess_l2
  
  // L1 regularization: soft-thresholding
  threshold = params.alpha / hess_l2
  delta = soft_threshold(raw_delta, threshold)
  
  return delta * params.learning_rate

soft_threshold(x, t) -> f32:
  if x > t:  return x - t
  if x < -t: return x + t
  return 0.0
```

## Feature Selectors

### FeatureSelector Trait

```rust
/// Controls feature iteration order for coordinate descent.
pub trait FeatureSelector: Iterator<Item = usize> + Send {
    /// Setup for a round. Called before iteration.
    ///
    /// Simple selectors (Cyclic, Shuffle): reset state.
    /// Gradient-based selectors (Greedy, Thrifty): compute ordering.
    fn setup<S: AsRef<[f32]>>(
        &mut self,
        model: &LinearModel,
        data: &ColMatrix<f32, S>,
        grads: &ColMatrix<f32>,
        hess: &ColMatrix<f32>,
        output_idx: usize,
        params: &UpdateParams,
    );
}
```

### Built-in Selectors

| Selector | Order | Setup Cost | Per-Feature Cost | Notes |
|----------|-------|------------|------------------|-------|
| Cyclic | 0, 1, 2, ... | O(1) | O(1) | Deterministic baseline |
| Shuffle | Random permutation | O(n) | O(1) | Better convergence |
| Random | Random w/ replacement | O(1) | O(1) | Can repeat features |
| Greedy | Largest delta first | O(n × n_samples) | O(1) | Optimal order, expensive |
| Thrifty | Approx gradient order | O(n log n) | O(1) | Sort once per round |

### Greedy/Thrifty Setup

```text
setup_gradient_order(model, data, grads, hess, output_idx, params, top_k):
  magnitudes = []
  for feature in 0..n_features:
    delta = abs(compute_weight_delta(model, data, grads, hess, feature, output_idx, params))
    magnitudes.push((feature, delta))
  
  // Sort descending by magnitude
  magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1))
  
  if let Some(k) = top_k:
    magnitudes.truncate(k)
  
  self.order = magnitudes.into_iter().map(|(f, _)| f).collect()
```

**Greedy** recomputes magnitudes after each feature update (O(n²) total).
**Thrifty** computes once per round (O(n log n)), using potentially stale ordering.

## Design Decisions

### DD-1: Algorithm-Centric Naming (Sequential/Shotgun)

**Context**: What to call the "parallel" update strategy?

**Decision**: "Shotgun" instead of "Parallel".

**Rationale**:

- **Describes algorithm**: "Shotgun coordinate descent" is the academic term
- **Avoids confusion**: "Parallel" suggests implementation detail, not algorithm
- **Matches literature**: [Bradley et al., 2011](https://arxiv.org/abs/1105.5379)
- **Clear semantics**: Sequential = exact gradients, Shotgun = stale gradients

### DD-2: FeatureSelector as setup() + Iterator

**Context**: How should selectors provide feature ordering?

**Decision**: Single trait with `setup()` method and `Iterator` impl.

**Rationale**:

```rust
// Usage in updater:
selector.setup(model, data, grad_hess, output_idx, params);
for feature in &mut selector {
    // ... update feature
}
```

- **Simple selectors**: `setup()` resets internal index
- **Gradient-based**: `setup()` computes ordering
- **Standard iteration**: Uses Rust's `Iterator` trait
- **Shotgun**: `selector.collect::<Vec<_>>()` for parallel access

### DD-3: Bias Initialized from Base Score

**Context**: Should base_score be stored separately or in model bias?

**Decision**: Store in `model.bias`.

**Rationale**:

- **Self-contained model**: Model predicts correctly without external state
- **Simpler prediction**: `output = bias + Σ(w[i] * x[i])`
- **Better starting point**: Training starts from optimal constant prediction
- **Clean serialization**: Single model struct captures everything

### DD-4: Per-Output Training Loop

**Context**: How to handle multi-output models?

**Decision**: Outer loop over rounds, inner loop over outputs.

**Rationale**:

```text
for round in 0..n_rounds:
  compute_gradients(...)  // All outputs at once
  for output in 0..n_outputs:
    updater.update(..., output)  // One output at a time
```

- **Matches GBTree**: Same structure as RFC-0007
- **Shared gradients**: Computed once per round for all outputs
- **Independent weights**: Each output has its own coefficient vector
- **XGBoost compatible**: Same training dynamics

### DD-5: Updater Owns Selector and Params

**Context**: Where should selector and regularization params live?

**Decision**: `CoordDescentUpdater<F>` owns both.

**Rationale**:

- **Encapsulation**: All update-related state in one place
- **Generic selector**: Type parameter allows different selectors
- **Clear ownership**: Selector has mutable state (RNG, position)
- **Reusable**: Same updater across multiple training runs

## Integration

| Component | How GBLinear Uses It |
|-----------|---------------------|
| RFC-0001 (Dataset) | Features as `ColMatrix`, labels for gradients |
| RFC-0002 (Objective) | `compute_gradients()` per round |
| RFC-0005 (LinearModel) | `add_weight()`, `predict_into()` |
| RFC-0006 (Evaluator) | `evaluate()` returns `EvalAction` |

### Linear vs Tree Training Comparison

```text
GBLinear:                           GBTree:
┌────────────────────┐              ┌────────────────────┐
│ compute_gradients  │              │ compute_gradients  │
└────────────────────┘              └────────────────────┘
         │                                   │
         ▼                                   ▼
┌────────────────────┐              ┌────────────────────┐
│ for output:        │              │ for output:        │
│   coord_descent()  │              │   grow_tree()      │
└────────────────────┘              └────────────────────┘
         │                                   │
         ▼                                   ▼
┌────────────────────┐              ┌────────────────────┐
│ predict_into()     │              │ predict_into()     │
└────────────────────┘              └────────────────────┘
```

## Future Work

- [ ] Adaptive learning rate schedules
- [ ] Feature interaction terms (polynomial features)
- [ ] Online/streaming coordinate descent
- [ ] GPU acceleration for Shotgun (parallel delta computation)
- [ ] Proximal operators beyond soft-thresholding

## References

- [XGBoost Linear Booster](https://xgboost.readthedocs.io/en/latest/tutorials/linear.html)
- [Shotgun Coordinate Descent](https://arxiv.org/abs/1105.5379) - Bradley et al., 2011
- [Coordinate Descent Methods](https://arxiv.org/abs/1502.04759) - Wright, 2015
