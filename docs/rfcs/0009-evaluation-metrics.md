# RFC-0009: Evaluation Metrics

- **Status**: Implemented
- **Created**: 2024-12-15
- **Updated**: 2025-01-21
- **Scope**: Metrics for model quality monitoring

## Summary

Evaluation metrics provide scalar measures of model quality during training. Unlike loss functions (used for gradient computation), metrics monitor validation performance, enable early stopping, and support optional sample weighting.

## Design

### MetricFn Trait

The core `MetricFn` trait defines how quality is measured:

```rust
use ndarray::ArrayView2;
use crate::training::{TargetsView, WeightsView};

pub trait MetricFn: Send + Sync {
    /// Compute metric value.
    /// Predictions: column-major [n_outputs, n_samples].
    fn compute(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> f64;

    /// Expected prediction format (Value, Probability, or Margin).
    fn expected_prediction_kind(&self) -> PredictionKind;

    /// True for accuracy/AUC; false for RMSE/LogLoss.
    fn higher_is_better(&self) -> bool;

    /// Display name (e.g., "rmse", "auc").
    fn name(&self) -> &'static str;

    /// Whether this metric is enabled (default: true).
    /// `NoMetric` returns false to skip computation entirely.
    fn is_enabled(&self) -> bool { true }
}
```

### Implemented Metrics

#### Regression Metrics

| Metric | Name | Formula | Direction |
|--------|------|---------|-----------|
| `Rmse` | `rmse` | RMSE = √(Σ(p-y)²/n) | Lower ↓ |
| `Mae` | `mae` | MAE = Σ\|p-y\|/n | Lower ↓ |
| `Mape` | `mape` | MAPE = 100×Σ(\|p-y\|/\|y\|)/n | Lower ↓ |
| `HuberMetric` | `huber` | Quadratic for small residuals, linear for large (configurable δ) | Lower ↓ |
| `PoissonDeviance` | `poisson` | 2×Σ[y×ln(y/μ) - (y-μ)]/n | Lower ↓ |
| `QuantileMetric` | `quantile` | Pinball loss: τ×max(y-q,0) + (1-τ)×max(q-y,0) | Lower ↓ |

#### Classification Metrics

| Metric | Name | Formula | Direction |
|--------|------|---------|-----------|
| `LogLoss` | `logloss` | -Σ[y×ln(p) + (1-y)×ln(1-p)]/n | Lower ↓ |
| `Accuracy` | `accuracy` | Σ1[ŷ=y]/n (threshold=0.5) | Higher ↑ |
| `MarginAccuracy` | `margin_accuracy` | Accuracy on raw margin scores (threshold=0.0) | Higher ↑ |
| `Auc` | `auc` | Area under ROC curve (O(n log n) algorithm) | Higher ↑ |
| `MulticlassLogLoss` | `mlogloss` | -Σln(p_y)/n | Lower ↓ |
| `MulticlassAccuracy` | `multiclass_accuracy` | Argmax prediction vs true class | Higher ↑ |

### Weighted Computation

All metrics support optional sample weights. When weights are provided:

```text
Weighted RMSE: sqrt(sum(w * (p - y)²) / sum(w))
Weighted Accuracy: sum(w * correct) / sum(w)
```

Pass empty slice `&[]` for unweighted computation.

### Multi-Output Support

For multiclass or multi-quantile models, predictions have shape `(n_samples, n_outputs)` in row-major order. The metric computes across all outputs appropriately (e.g., `MulticlassLogLoss` extracts probability of true class).

### Early Stopping

`EarlyStopping` monitors validation metrics and stops training when no improvement occurs:

```rust
pub struct EarlyStopping {
    patience: usize,        // Rounds without improvement before stopping
    higher_is_better: bool, // Metric direction
    best_value: Option<f64>,
    best_round: usize,
    current_round: usize,
}

impl EarlyStopping {
    pub fn new(patience: usize, higher_is_better: bool) -> Self;
    
    /// Returns true when training should stop
    pub fn should_stop(&mut self, value: f64) -> bool;
    
    pub fn best_value(&self) -> Option<f64>;
    pub fn best_round(&self) -> usize;
    pub fn reset(&mut self);
}
```

**Behavior**: Stops when `current_round - best_round > patience`. Counter resets on any improvement.

### EvalSet

Named evaluation datasets for monitoring during training:

```rust
pub struct EvalSet<'a> {
    pub name: &'a str,
    pub dataset: &'a Dataset,
}

impl<'a> EvalSet<'a> {
    pub fn new(name: &'a str, dataset: &'a Dataset) -> Self;
}
```

**Usage**:
```rust
let eval_sets = vec![
    EvalSet::new("train", &train_data),
    EvalSet::new("valid", &valid_data),
];
```

### Evaluator Component

The `Evaluator` encapsulates evaluation logic during training:

```rust
pub struct Evaluator<'a, O: Objective, M: Metric> {
    objective: &'a O,
    metric: &'a M,
    n_outputs: usize,
    transform_buffer: Vec<f32>,
}

impl<'a, O: Objective, M: Metric> Evaluator<'a, O, M> {
    pub fn new(objective: &'a O, metric: &'a M, n_outputs: usize) -> Self;
    
    /// Compute metric on single dataset
    pub fn compute(&mut self, predictions: &[f32], targets: &[f32], 
                   weights: &[f32], n_samples: usize) -> f64;
    
    /// Compute and wrap as MetricValue
    pub fn compute_metric(&mut self, name: impl Into<String>, predictions: &[f32],
                          targets: &[f32], weights: &[f32], n_samples: usize) -> MetricValue;
    
    /// Evaluate on training + eval sets for one round
    pub fn evaluate_round(&mut self, train_predictions: &[f32], train_targets: &[f32],
                          train_weights: &[f32], train_n_samples: usize,
                          eval_sets: &[EvalSet<'_>], eval_predictions: &[Vec<f32>]) 
                          -> Vec<MetricValue>;
    
    /// Get early stopping value from metrics
    pub fn early_stop_value(metrics: &[MetricValue], eval_set_idx: usize) -> f64;
}
```

The Evaluator handles prediction transforms (sigmoid/softmax) when required by the metric, manages its own transform buffer, and produces `MetricValue` wrappers with metadata.

### MetricValue

A computed metric value with metadata:

```rust
pub struct MetricValue {
    pub name: String,           // e.g., "train-rmse", "valid-logloss"
    pub value: f64,
    pub higher_is_better: bool,
}

impl MetricValue {
    pub fn new(name: impl Into<String>, value: f64, higher_is_better: bool) -> Self;
    pub fn is_better_than(&self, other: &Self) -> bool;
    pub fn is_better_than_value(&self, other_value: f64) -> bool;
}
```

## Key Types

| Type | Purpose |
|------|---------|
| `MetricFn` | Trait for computing evaluation scores |
| `Metric` | Enum for configuration/defaults (Rmse, Mae, LogLoss, etc.) |
| `MetricValue` | Computed metric with name, value, and direction metadata |
| `EvalSet` | Named dataset reference for evaluation |
| `Evaluator` | Component managing evaluation logic and transform buffers |
| `EarlyStopping` | Callback that monitors metric and signals when to stop |
| `PredictionKind` | Expected prediction format (Value, Probability, Margin) |

## Implementation Notes

- AUC uses O(n log n) Wilcoxon-Mann-Whitney algorithm with tie handling
- Epsilon clipping (1e-15) prevents log(0) in LogLoss/MulticlassLogLoss  
- MAPE uses epsilon denominator to handle zero labels
- All metrics implement `Send + Sync` for parallel evaluation

## Changelog

- 2025-01-23: Renamed `Metric` trait to `MetricFn`. Updated signatures to use `ArrayView2`, `TargetsView`, `WeightsView`. Added `is_enabled()` method.
- 2025-01-21: Updated terminology (n_samples) to match codebase conventions
