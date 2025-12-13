# RFC-0002: Objectives (Loss Functions)

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: RFC-0001 (Data Matrix)
- **Scope**: Loss functions and gradient computation for training

## Summary

Objectives define what gradient boosting optimizes. Given predictions and labels, the objective computes gradients and hessians that guide each boosting round. This RFC covers the objective trait, built-in implementations, and the computation patterns that make multi-output training seamless.

## Overview

### Component Hierarchy

```text
Objective (trait)
│
└── ObjectiveFunction (enum)          ← Zero-cost dispatch for built-in objectives
    ├── SquaredError
    ├── AbsoluteError
    ├── PseudoHuber { delta }
    ├── PinballLoss { alphas }        ← Multi-quantile support
    ├── Logistic
    ├── Hinge
    └── Softmax { n_classes }         ← 1 label → K outputs

Trainer<O: Objective = ObjectiveFunction>
└── Uses default type parameter for zero-cost dispatch
```

### Data Flow

```text
predictions [n_samples × n_outputs]  ─┐
labels [n_samples × n_labels]        ─┼─► Objective.compute_gradients()
weights [n_samples × 1] (optional)   ─┘              │
                                                     ▼
                                        gradients [n_samples × n_outputs]
                                        hessians [n_samples × n_outputs]
                                                     │
                                                     ▼
                                             TreeGrower / CoordDescent
```

### Labels vs Outputs

The objective transforms `n_labels` columns into `n_outputs` columns:

| Objective | n_labels | n_outputs | Transformation |
|-----------|----------|-----------|----------------|
| SquaredError | k | k | Identity |
| Logistic | 1 | 1 | Identity |
| Softmax | 1 (class indices) | K (probabilities) | One-hot expansion |
| PinballLoss | 1 or Q | Q | Broadcast or identity |

The trainer calls `objective.n_outputs(n_labels)` to determine buffer sizes.

## Components

### Objective Trait

```rust
/// A loss function for gradient boosting training.
pub trait Objective: Send + Sync {
    /// Number of output columns for given number of label columns.
    fn n_outputs(&self, n_labels: usize) -> usize { n_labels }

    /// Compute gradients and hessians for all samples and outputs.
    ///
    /// # Layout
    /// All matrices are column-major. Each column = one output.
    fn compute_gradients<S: AsRef<[f32]>>(
        &self,
        predictions: &ColMatrix<f32, S>,
        labels: &ColMatrix<f32, S>,
        weights: Option<&ColMatrix<f32, S>>,
        gradients: &mut ColMatrix<f32>,
        hessians: &mut ColMatrix<f32>,
    );

    /// Compute optimal initial predictions (base scores).
    ///
    /// Written to `out` (shape: [1 × n_outputs]).
    fn init_base_score<S: AsRef<[f32]>>(
        &self,
        labels: &ColMatrix<f32, S>,
        weights: Option<&ColMatrix<f32, S>>,
        out: &mut ColMatrix<f32>,
    );

    /// Convenience: returns base scores as Vec.
    fn base_score_vec<S: AsRef<[f32]>>(
        &self,
        labels: &ColMatrix<f32, S>,
        weights: Option<&ColMatrix<f32, S>>,
    ) -> Vec<f32> {
        let n_outputs = self.n_outputs(labels.n_cols());
        let mut out = ColMatrix::zeros(1, n_outputs);
        self.init_base_score(labels, weights, &mut out);
        out.into_vec()
    }

    /// Name of the objective (for logging).
    fn name(&self) -> &'static str;
}
```

### ObjectiveFunction Enum

```rust
/// Built-in objective functions with zero-cost dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveFunction {
    // Regression
    SquaredError,
    AbsoluteError,
    PseudoHuber { delta: f32 },
    
    // Quantile regression
    PinballLoss { alphas: Vec<f32> },
    
    // Binary classification
    Logistic,
    Hinge,
    
    // Multiclass classification
    Softmax { n_classes: usize },
}

impl Default for ObjectiveFunction {
    fn default() -> Self { Self::SquaredError }
}

impl Objective for ObjectiveFunction {
    fn n_outputs(&self, n_labels: usize) -> usize {
        match self {
            Self::Softmax { n_classes } => *n_classes,
            Self::PinballLoss { alphas } => alphas.len(),
            _ => n_labels,
        }
    }
    // ... dispatch to specific implementations
}
```

### Objective Summary Table

| Objective | Loss Formula | Gradient | Hessian | Base Score |
|-----------|-------------|----------|---------|------------|
| SquaredError | $(y - \hat{y})^2$ | $\hat{y} - y$ | $1$ | Weighted mean |
| AbsoluteError | $\|y - \hat{y}\|$ | $\text{sign}(\hat{y} - y)$ | $1$ | Weighted median |
| PseudoHuber | $\delta^2(\sqrt{1 + (e/\delta)^2} - 1)$ | $e / \sqrt{1 + (e/\delta)^2}$ | varies | Weighted mean |
| PinballLoss | $\rho_\alpha(y - \hat{y})$ | $\begin{cases} -\alpha & y > \hat{y} \\ 1-\alpha & y \le \hat{y} \end{cases}$ | $1$ | Weighted α-quantile |
| Logistic | $-y\log(p) - (1-y)\log(1-p)$ | $p - y$ | $p(1-p)$ | Log-odds |
| Softmax | $-\log(p_y)$ | $p_k - \mathbb{1}_{k=y}$ | $p_k(1-p_k)$ | Class log-probs |

## Algorithms

All algorithms operate on **column-major** matrices. The outer loop iterates outputs (columns), inner loop iterates samples (rows within a column).

### Squared Error

```text
SquaredErrorGradients(predictions, labels, weights, gradients, hessians):
  for k in 0..n_outputs:
    pred_col = predictions.col_slice(k)
    label_col = labels.col_slice(k)
    grad_col = gradients.col_slice_mut(k)
    hess_col = hessians.col_slice_mut(k)
    
    for i in 0..n_samples:
      w = weights.map(|w| w[0][i]).unwrap_or(1.0)
      grad_col[i] = w * (pred_col[i] - label_col[i])
      hess_col[i] = w

SquaredErrorBaseScore(labels, weights, out):
  for k in 0..n_outputs:
    out[0][k] = weighted_mean(labels.col_slice(k), weights)
```

### Logistic Loss

```text
LogisticGradients(predictions, labels, weights, gradients, hessians):
  pred_col = predictions.col_slice(0)
  label_col = labels.col_slice(0)
  grad_col = gradients.col_slice_mut(0)
  hess_col = hessians.col_slice_mut(0)
  
  for i in 0..n_samples:
    p = sigmoid(pred_col[i])        // 1 / (1 + exp(-x))
    w = weights.map(|w| w[0][i]).unwrap_or(1.0)
    grad_col[i] = w * (p - label_col[i])
    hess_col[i] = w * p * (1.0 - p)

LogisticBaseScore(labels, weights, out):
  p_pos = weighted_mean(labels.col_slice(0), weights)
  out[0][0] = log(p_pos / (1.0 - p_pos))  // log-odds
```

### Pinball Loss (Multi-Quantile)

Supports Q quantiles. Labels can be `[n_samples × 1]` (broadcast) or `[n_samples × Q]`.

```text
PinballGradients(predictions, labels, weights, gradients, hessians, alphas):
  Q = alphas.len()
  broadcast = labels.n_cols() == 1
  
  for k in 0..Q:
    alpha = alphas[k]
    label_col = if broadcast { labels.col_slice(0) } else { labels.col_slice(k) }
    pred_col = predictions.col_slice(k)
    grad_col = gradients.col_slice_mut(k)
    hess_col = hessians.col_slice_mut(k)
    
    for i in 0..n_samples:
      residual = label_col[i] - pred_col[i]
      w = weights.map(|w| w[0][i]).unwrap_or(1.0)
      grad_col[i] = if residual > 0 { w * -alpha } else { w * (1.0 - alpha) }
      hess_col[i] = w  // constant hessian

PinballBaseScore(labels, weights, out, alphas):
  broadcast = labels.n_cols() == 1
  for k in 0..alphas.len():
    label_col = if broadcast { labels.col_slice(0) } else { labels.col_slice(k) }
    out[0][k] = weighted_quantile(label_col, weights, alphas[k])
```

### Softmax (Multiclass)

Maps 1 label column (class indices) to K output columns. Requires row-wise softmax.

```text
SoftmaxGradients(predictions, labels, weights, gradients, hessians):
  K = n_classes
  
  for i in 0..n_samples:
    // Gather predictions for sample i (strided access)
    probs = [0.0; K]
    max_val = -∞
    for k in 0..K:
      max_val = max(max_val, predictions[k][i])
    
    // Compute softmax
    sum_exp = 0.0
    for k in 0..K:
      probs[k] = exp(predictions[k][i] - max_val)
      sum_exp += probs[k]
    for k in 0..K:
      probs[k] /= sum_exp
    
    // Scatter gradients (strided write)
    y = labels[0][i] as usize  // true class
    w = weights.map(|w| w[0][i]).unwrap_or(1.0)
    for k in 0..K:
      indicator = if k == y { 1.0 } else { 0.0 }
      gradients[k][i] = w * (probs[k] - indicator)
      hessians[k][i] = w * probs[k] * (1.0 - probs[k])
```

**Performance Note**: The per-sample loop has strided column access. For better cache locality, consider block transposition (process BLOCK_SIZE samples at a time, transpose to row-major, compute, transpose back). Profile to determine if this helps for typical K values.

## Design Decisions

### DD-1: Separate Gradient and Hessian Matrices (SoA)

**Context**: Gradients and hessians are always computed together. Should they share a struct or be separate?

**Decision**: Separate `ColMatrix` for each.

**Rationale**:

- **Cache efficiency**: Training often accesses only gradients OR only hessians
- **SIMD friendly**: Contiguous column data enables vectorization
- **Consistent layout**: Both follow col-major convention from RFC-0001
- **Multi-output natural**: Each column = one output

Alternative (AoS: `GradHess { grad: f32, hess: f32 }`): Would interleave data, hurting cache when only one is needed.

### DD-2: Buffer-Writing API

**Context**: Should `compute_gradients` return new allocations or write to buffers?

**Decision**: Write to pre-allocated `&mut ColMatrix<f32>` buffers.

**Rationale**:

- **Zero allocation in hot path**: Gradients computed every boosting round
- **Buffer reuse**: Same buffers across all rounds
- **Consistent with Rust patterns**: Output parameters for performance-critical code
- **Caller controls memory**: Enables arena allocation, GPU memory, etc.

### DD-3: Weights Multiply Both Gradient and Hessian

**Context**: How should sample weights affect gradient computation?

**Decision**: Multiply both: `grad[i] *= w[i]`, `hess[i] *= w[i]`.

**Rationale**:

The leaf weight formula is: $w^* = -\frac{\sum_i g_i}{\sum_i h_i + \lambda}$

With sample weights $w_i$, this becomes: $w^* = -\frac{\sum_i w_i \cdot g_i}{\sum_i w_i \cdot h_i + \lambda}$

Weighting both preserves the correct optimization behavior.

### DD-4: Default Type Parameter Pattern

**Context**: How to support custom objectives ergonomically while optimizing built-in ones?

**Decision**: `Trainer<O: Objective = ObjectiveFunction>`.

**Rationale**:

```rust
// Most users: zero-cost enum dispatch
let trainer = GBTreeTrainer::new(ObjectiveFunction::Logistic);

// Custom objective: just specify the type
let trainer = GBTreeTrainer::<MyCustomObjective>::new(my_obj);
```

- **Default case**: No boxing, enum dispatch is a single `match`
- **Custom case**: Works without modification
- **No trait objects**: Avoids `Box<dyn Objective>` overhead

### DD-5: Constant Hessians Filled Explicitly

**Context**: Some objectives (SquaredError, PinballLoss) have constant hessian = 1.

**Decision**: Fill hessian buffer with the constant, no special-casing.

**Rationale**:

- **Uniform interface**: All objectives behave the same way
- **No branching downstream**: Histogram building doesn't check "is hessian constant"
- **Efficient fill**: `hess_col.fill(1.0)` is very fast
- **Simplicity**: Trait interface stays clean

### DD-6: Base Score Written to Buffer

**Context**: How should `init_base_score` return its result?

**Decision**: Write to `&mut ColMatrix<f32>` (shape `[1 × n_outputs]`).

**Rationale**:

- **Consistent with `compute_gradients`**: Same pattern
- **Multi-output**: Shape matches output count
- **Convenience method**: `base_score_vec()` wraps this for simple use cases

## Integration

| Component | How Objective is Used |
|-----------|----------------------|
| RFC-0001 (Dataset) | Labels and weights as input |
| RFC-0003 (Metrics) | Predictions compared against labels (metrics ≠ objectives) |
| RFC-0004 (GBLinear) | Gradient computation per round |
| RFC-0007 (GBTree) | Gradient computation per round |

### Usage in Training Loop

```text
// Setup
n_outputs = objective.n_outputs(dataset.labels.n_cols())
base_score = objective.base_score_vec(dataset.labels, dataset.weights)
predictions.fill_from_base_score(base_score)
gradients = ColMatrix::zeros(n_samples, n_outputs)
hessians = ColMatrix::zeros(n_samples, n_outputs)

// Per round
objective.compute_gradients(predictions, labels, weights, &mut gradients, &mut hessians)
// ... train tree/linear model using gradients, hessians ...
// ... update predictions ...
```

## Future Work

- [ ] Ranking objectives (LambdaMART, LambdaRank) - require pairwise gradients
- [ ] Survival analysis (Cox, AFT)
- [ ] Focal loss for imbalanced classification
- [ ] Tweedie regression for insurance/count data
- [ ] Custom objective via Python callback

## References

- [XGBoost Objectives](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)
- [LightGBM Objectives](https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective)
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
