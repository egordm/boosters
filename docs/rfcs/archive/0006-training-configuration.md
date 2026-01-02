# RFC-0006: Training Configuration

- **Status**: Implemented
- **Created**: 2024-12-15
- **Updated**: 2025-01-24
- **Depends on**: RFC-0004 (Binning), RFC-0005 (Tree Growing)
- **Scope**: Objectives, metrics, sampling, and multi-output training

## Summary

This RFC consolidates training-related configuration: loss functions for gradient computation (objectives), quality metrics for monitoring, sampling strategies for regularization, and multi-output training strategy.

---

## 1. Objectives and Losses

### ObjectiveFn Trait

The core trait uses ndarray views for type-safe multi-dimensional access:

```rust
use ndarray::{ArrayView2, ArrayViewMut2};
use crate::training::TargetsView, WeightsView;

pub trait ObjectiveFn: Send + Sync {
    fn n_outputs(&self) -> usize;

    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        grad_hess: ArrayViewMut2<GradsTuple>,
    );

    fn compute_base_score(
        &self,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> Vec<f32>;

    fn task_kind(&self) -> TaskKind;
    fn target_schema(&self) -> TargetSchema;
    fn transform_predictions_inplace(&self, predictions: ArrayViewMut2<f32>) -> PredictionKind;
    fn name(&self) -> &'static str;
}
```

### Implemented Objectives

**Regression:**
| Objective | Gradient | Hessian | Base Score |
|-----------|----------|---------|------------|
| `SquaredLoss` | `pred - target` | `1.0` | Weighted mean |
| `AbsoluteLoss` | `sign(pred - target)` | `1.0` | Weighted median |
| `PinballLoss` | `α - 1` if under, `α` if over | `1.0` | Weighted quantile |
| `PseudoHuberLoss` | `residual / sqrt(factor)` | `1 / factor^1.5` | Weighted median |
| `PoissonLoss` | `exp(pred) - target` | `exp(pred)` | `log(mean)` |

**Classification:**
| Objective | Gradient | Hessian | Base Score |
|-----------|----------|---------|------------|
| `LogisticLoss` | `σ(pred) - target` | `σ * (1-σ)` | Log-odds |
| `HingeLoss` | `-y` if margin < 1 | `1.0` | Zero |
| `SoftmaxLoss` | `p_c - 1{c=label}` | `p_c * (1 - p_c)` | Log class frequencies |

**Ranking:**
| Objective | Description |
|-----------|-------------|
| `LambdaRankLoss` | LambdaMART for NDCG optimization |

### Gradient Storage

`GradsTuple` stores interleaved gradient/hessian pairs:

```rust
#[repr(C)]
pub struct GradsTuple { pub grad: f32, pub hess: f32 }
```

**Precision strategy**: Gradients stored as f32 for memory efficiency. Histogram accumulation uses f64 internally to prevent numerical drift.

---

## 2. Evaluation Metrics

### MetricFn Trait

```rust
pub trait MetricFn: Send + Sync {
    fn compute(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> f64;

    fn expected_prediction_kind(&self) -> PredictionKind;
    fn higher_is_better(&self) -> bool;
    fn name(&self) -> &'static str;
    fn is_enabled(&self) -> bool { true }
}
```

### Implemented Metrics

**Regression:** RMSE, MAE, MAPE, Huber, PoissonDeviance, Quantile (all lower-is-better)

**Classification:** LogLoss, Accuracy, MarginAccuracy, AUC, MulticlassLogLoss, MulticlassAccuracy

### Early Stopping

`EarlyStopping` monitors validation metrics and stops training when no improvement occurs:

```rust
pub struct EarlyStopping {
    patience: usize,
    higher_is_better: bool,
    best_value: Option<f64>,
    best_round: usize,
}
```

Stops when `current_round - best_round > patience`.

---

## 3. Sampling Strategies

### Row Sampling

Row sampling modifies gradients in-place by zeroing out unsampled rows.

**Uniform Sampling**: Standard bagging with reservoir sampling.

**GOSS Sampling** (Gradient-based One-Side Sampling):
1. Keep top `top_rate` fraction (high gradient magnitude)
2. Randomly sample `other_rate` fraction from remaining
3. Apply amplification factor to sampled small gradients

**Warmup**: GOSS skips sampling for the first `⌊1/learning_rate⌋` iterations.

### Column Sampling

Three-level cascading feature sampling:

| Level | When Applied |
|-------|--------------|
| `colsample_bytree` | Once per tree |
| `colsample_bylevel` | When depth changes |
| `colsample_bynode` | Every split finding |

**Effective rate**: `bytree × bylevel × bynode`

### Key Types

```rust
pub enum RowSamplingParams {
    None,
    Uniform { subsample: f32 },
    Goss { top_rate: f32, other_rate: f32 },
}

pub enum ColSamplingParams {
    None,
    Sample {
        colsample_bytree: f32,
        colsample_bylevel: f32,
        colsample_bynode: f32,
    },
}
```

---

## 4. Multi-Output Training

### One-Tree-Per-Output Strategy

For K outputs, each boosting round trains K separate trees with scalar leaves:

```rust
for round in 0..n_trees {
    objective.compute_gradients(...);
    
    for output in 0..n_outputs {
        let grad_hess = gradients.output_pairs_mut(output);
        let tree = grower.grow(dataset, &gradients, output, ...);
        forest.push_tree(tree, output as u32);
    }
}
```

**Benefits:**
- Simpler training: reuses scalar-leaf grower unchanged
- Same quality: empirically equivalent to vector-leaf approach
- Memory efficient: no K-dimensional leaf allocations
- Inference flexibility: trees can be evaluated independently

### Gradient Buffer Layout

Column-major layout for per-output training:

```rust
pub struct Gradients {
    data: Vec<GradsTuple>,  // [n_samples * n_outputs]
    n_samples: usize,
    n_outputs: usize,
}
// output_pairs(k) returns contiguous slice for output k
```

### Forest Group Assignment

Trees are grouped by output via `tree_groups`:

```rust
pub struct Forest<L: LeafValue> {
    trees: Vec<Tree<L>>,
    tree_groups: Vec<u32>,  // tree_groups[i] = output for tree i
    n_groups: u32,
    base_score: Vec<f32>,
}
```

---

## Key Types Summary

| Type | Purpose |
|------|---------|
| `ObjectiveFn` | Loss function trait |
| `GradsTuple` | Interleaved grad/hess pair |
| `Gradients` | Per-output gradient buffer |
| `MetricFn` | Evaluation metric trait |
| `EarlyStopping` | Patience-based stopping |
| `RowSamplingParams` | Uniform/GOSS config |
| `ColSamplingParams` | Three-level column sampling |

## Integration

| Component | Integration Point |
| --------- | ----------------- |
| RFC-0005 (Growing) | Grower uses gradients and sampling |
| RFC-0002 (Trees) | Forest stores tree groups |
| RFC-0003 (Inference) | Metrics evaluate predictions |

## Changelog

- 2025-01-24: Merged RFC-0008 (Objectives), RFC-0009 (Metrics), RFC-0010 (Sampling), RFC-0011 (Multi-output) into unified training RFC.
- 2025-01-21: Updated terminology to match codebase conventions.
