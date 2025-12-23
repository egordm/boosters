# RFC-0008: Objectives and Losses

- **Status**: Implemented
- **Created**: 2024-12-15
- **Updated**: 2025-01-21
- **Scope**: Loss functions for gradient computation

## Summary

Objectives define loss functions that compute gradients and hessians for gradient boosting optimization. The module provides regression, classification, and ranking objectives with multi-output support and sample weighting.

## Design

### ObjectiveFn Trait

The core trait uses ndarray views for type-safe multi-dimensional access:

```rust
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2};

pub trait ObjectiveFn: Send + Sync {
    /// Number of outputs per sample (1 for regression, K for multiclass).
    fn n_outputs(&self) -> usize;

    /// Compute gradients and hessians for given predictions.
    ///
    /// # Arguments
    /// - `predictions`: Shape `[n_outputs, n_samples]`, model outputs
    /// - `targets`: Shape `[n_samples]`, ground truth (single target column)
    /// - `weights`: Optional per-sample weights
    /// - `grad_hess`: Shape `[n_outputs, n_samples]`, output buffer for (grad, hess) pairs
    fn compute_gradients(
        &self,
        predictions: ArrayView2<f32>,
        targets: ArrayView1<f32>,
        weights: Option<ArrayView1<f32>>,
        grad_hess: ArrayViewMut2<GradsTuple>,
    );

    /// Compute initial base score (bias) from targets.
    fn compute_base_score(
        &self,
        targets: ArrayView1<f32>,
        weights: Option<ArrayView1<f32>>,
        outputs: &mut [f32],
    );

    /// High-level task kind (Regression, BinaryClassification, etc.).
    fn task_kind(&self) -> TaskKind;

    /// Expected target encoding (Continuous, Binary01, MulticlassIndex, etc.).
    fn target_schema(&self) -> TargetSchema;

    /// Transform raw margins to semantic predictions (in-place).
    fn transform_predictions(&self, predictions: ArrayViewMut2<f32>) -> PredictionKind;

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

All data uses **group-major (row-major by group)** order via ndarray:

```
predictions: ArrayView2<f32>        // Shape [n_outputs, n_samples]
targets:     ArrayView1<f32>        // Shape [n_samples]
grad_hess:   ArrayViewMut2<GradsTuple>  // Shape [n_outputs, n_samples]
```

Each output's values are contiguous in memory (group-major layout).

Relationship between outputs and targets:
- `SquaredLoss`: `n_outputs == n_targets` (1:1 mapping)
- `PinballLoss`: Multiple quantiles can share single target column
- `SoftmaxLoss`: `n_outputs = num_classes`, `n_targets = 1` (class indices)

### Base Score Initialization

Each objective computes the optimal constant prediction before trees. Two convenience methods are provided:

```rust
/// Compute base scores and return as Vec.
fn base_scores(&self, n_samples: usize, targets: &[f32], weights: &[f32]) -> Vec<f32>;

/// Fill column-major prediction buffer with computed base scores.
fn fill_base_scores(&self, predictions: &mut [f32], n_samples: usize, targets: &[f32], weights: &[f32]);
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

- `None` = unweighted (all weights = 1.0)
- `Some(weights)` = weights applied to both gradient and hessian

```rust
// Helper used internally
fn weight_iter(weights: &[f32], n_samples: usize) -> impl Iterator<Item = f32> {
    let use_weights = !weights.is_empty();
    (0..n_samples).map(move |i| if use_weights { weights[i] } else { 1.0 })
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
    Softmax { n_classes: usize },
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

**Precision strategy**: Gradients are stored as f32 for memory efficiency (2× savings vs f64). However, histogram accumulation uses f64 internally to prevent numerical drift when summing many small values. This provides a good balance between memory usage and numerical stability.

`Gradients` buffer provides group-major storage optimized for histogram building:

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
pub struct SoftmaxLoss { pub n_classes: usize }
pub struct LambdaRankLoss { pub query_groups: Vec<usize>, pub sigma: f32 }

// Gradient storage
pub struct GradsTuple { pub grad: f32, pub hess: f32 }
pub struct Gradients { ... }
pub enum ObjectiveFunction { ... }
```

## Changelog

- 2025-01-23: Updated trait to use ndarray views (`ArrayView2`, `ArrayView1`, `ArrayViewMut2`) instead of slices. Renamed trait to `ObjectiveFn`. Clarified group-major layout.
- 2025-01-21: Updated terminology (n_samples, n_classes) to match codebase conventions
