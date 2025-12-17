# RFC-0008: Objectives and Losses

**Status**: Implemented

## Summary

Objectives define loss functions that compute gradients and hessians for gradient boosting optimization. The module provides regression, classification, and ranking objectives with multi-output support and sample weighting.

## Design

### Objective Trait

```rust
pub trait Objective: Send + Sync {
    /// Number of outputs per sample (1 for regression, K for multiclass).
    fn n_outputs(&self) -> usize { 1 }

    /// Compute gradients and hessians for given predictions.
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],    // column-major [n_outputs * n_rows]
        targets: &[f32],        // column-major [n_targets * n_rows]
        weights: &[f32],        // sample weights (empty for unweighted)
        grad_hess: &mut [GradsTuple],
    );

    /// Compute initial base score (bias) from targets.
    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],
    );

    /// High-level task kind (Regression, BinaryClassification, etc.).
    fn task_kind(&self) -> TaskKind;

    /// Expected target encoding (Continuous, Binary01, MulticlassIndex, etc.).
    fn target_schema(&self) -> TargetSchema;

    /// Default evaluation metric for this objective.
    fn default_metric(&self) -> MetricKind;

    /// Transform raw margins to semantic predictions (in-place).
    fn transform_prediction_inplace(&self, raw: &mut PredictionOutput) -> PredictionKind;

    fn name(&self) -> &'static str;
}
```

### Implemented Objectives

**Regression:**
| Objective | Loss Formula | Gradient | Hessian | Base Score |
|-----------|--------------|----------|---------|------------|
| `SquaredLoss` | `0.5 * (pred - target)²` | `pred - target` | `1.0` | Weighted mean |
| `AbsoluteLoss` | `\|pred - target\|` | `sign(pred - target)` | `1.0` | Weighted median |
| `PinballLoss` | Quantile loss for α | `α - 1` if under, `α` if over | `1.0` | Weighted quantile |
| `PseudoHuberLoss` | Smooth L1 with delta | `residual / sqrt(factor)` | `1 / factor^1.5` | Weighted median |
| `PoissonLoss` | `exp(pred) - target * pred` | `exp(pred) - target` | `exp(pred)` | `log(mean)` |

**Classification:**
| Objective | Description | Gradient | Hessian | Base Score |
|-----------|-------------|----------|---------|------------|
| `LogisticLoss` | Binary cross-entropy | `σ(pred) - target` | `σ * (1-σ)` | Log-odds of class ratio |
| `HingeLoss` | SVM-style margin loss | `-y` if margin < 1 | `1.0` | Zero |
| `SoftmaxLoss` | Multiclass cross-entropy | `p_c - 1{c=label}` | `p_c * (1 - p_c)` | Log class frequencies |

**Ranking:**
| Objective | Description |
|-----------|-------------|
| `LambdaRankLoss` | LambdaMART for NDCG optimization with pairwise gradients |

### Multi-Output Layout

All data uses **column-major** order: `[output0_row0, output0_row1, ..., output0_rowN, output1_row0, ...]`

```
predictions: [n_outputs * n_rows]
targets:     [n_targets * n_rows]  
grad_hess:   [n_outputs * n_rows]
```

Relationship between outputs and targets:
- `SquaredLoss`: `n_outputs == n_targets` (1:1 mapping)
- `PinballLoss`: Multiple quantiles can share single target column
- `SoftmaxLoss`: `n_outputs = num_classes`, `n_targets = 1` (class indices)

### Base Score Initialization

Each objective computes the optimal constant prediction before trees:

```rust
fn init_base_score(&self, targets: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
    let n_outputs = self.n_outputs();
    let mut base_scores = vec![0.0f32; n_outputs];
    self.compute_base_score(n_rows, n_outputs, targets, weights, &mut base_scores);
    base_scores
}
```

| Objective | Base Score Strategy |
|-----------|---------------------|
| SquaredLoss | Weighted mean of targets |
| AbsoluteLoss, PseudoHuber | Weighted median |
| PinballLoss | Weighted α-quantile |
| LogisticLoss | Log-odds: `ln(p / (1-p))` |
| SoftmaxLoss | Log of class frequencies |
| PoissonLoss | `ln(mean(targets))` |

### Sample Weight Support

All objectives support per-sample weights via the `weights` parameter:
- Empty slice `&[]` = unweighted (all weights = 1.0)
- Non-empty = weights applied to both gradient and hessian

```rust
// Helper used internally
fn weight_iter(weights: &[f32], n_rows: usize) -> impl Iterator<Item = f32> {
    let use_weights = !weights.is_empty();
    (0..n_rows).map(move |i| if use_weights { weights[i] } else { 1.0 })
}
```

### ObjectiveFunction Enum

Convenience enum wrapping all objectives for easy configuration:

```rust
pub enum ObjectiveFunction {
    SquaredError,
    AbsoluteError,
    Logistic,
    Hinge,
    Softmax { num_classes: usize },
    Quantile { alpha: f32 },
    MultiQuantile { alphas: Vec<f32> },
    PseudoHuber { delta: f32 },
    Poisson,
}
```

### Gradient Storage

`GradsTuple` stores interleaved gradient/hessian pairs:

```rust
#[repr(C)]
pub struct GradsTuple {
    pub grad: f32,
    pub hess: f32,
}
```

`Gradients` buffer provides column-major storage optimized for histogram building:
- `output_pairs(k)` returns contiguous slice for output k (zero-copy)
- `sum(output, rows)` computes gradient/hessian sums with f64 accumulation

## Key Types

```rust
// Core trait
pub trait Objective: Send + Sync { ... }

// Extension trait for Gradients integration
pub trait ObjectiveExt: Objective {
    fn compute_gradients_buffer(&self, predictions, targets, weights, buffer: &mut Gradients);
}

// Task/target semantics
pub enum TaskKind { Regression, BinaryClassification, MulticlassClassification, Ranking }
pub enum TargetSchema { Continuous, Binary01, BinarySigned, MulticlassIndex, CountNonNegative }

// Concrete objectives
pub struct SquaredLoss;
pub struct AbsoluteLoss;
pub struct PinballLoss { pub alphas: Vec<f32> }
pub struct PseudoHuberLoss { pub delta: f32 }
pub struct PoissonLoss;
pub struct LogisticLoss;
pub struct HingeLoss;
pub struct SoftmaxLoss { pub num_classes: usize }
pub struct LambdaRankLoss { pub query_groups: Vec<usize>, pub sigma: f32 }

// Gradient storage
pub struct GradsTuple { pub grad: f32, pub hess: f32 }
pub struct Gradients { ... }
pub enum ObjectiveFunction { ... }
```
