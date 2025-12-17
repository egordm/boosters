# RFC-0009: Evaluation Metrics

**Status**: Implemented

## Summary

Evaluation metrics provide scalar measures of model quality during training. Unlike loss functions (used for gradient computation), metrics monitor validation performance, enable early stopping, and support optional sample weighting.

## Design

### Metric Trait

The core `Metric` trait defines how quality is measured:

```rust
pub trait Metric: Send + Sync {
    /// Compute metric value.
    /// Predictions: row-major (n_rows, n_outputs). Empty weights = unweighted.
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64;

    /// Expected prediction format (Value, Probability, or Margin).
    fn expected_prediction_kind(&self) -> PredictionKind;

    /// True for accuracy/AUC; false for RMSE/LogLoss.
    fn higher_is_better(&self) -> bool;

    /// Display name (e.g., "rmse", "auc").
    fn name(&self) -> &'static str;
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

For multiclass or multi-quantile models, predictions have shape `(n_rows, n_outputs)` in row-major order. The metric computes across all outputs appropriately (e.g., `MulticlassLogLoss` extracts probability of true class).

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

## Key Types

| Type | Purpose |
|------|---------|
| `Metric` | Trait for computing evaluation scores |
| `MetricKind` | Enum for configuration/defaults (Rmse, Mae, LogLoss, etc.) |
| `EvalSet` | Named dataset reference for evaluation |
| `EarlyStopping` | Callback that monitors metric and signals when to stop |
| `PredictionKind` | Expected prediction format (Value, Probability, Margin) |

## Implementation Notes

- AUC uses O(n log n) Wilcoxon-Mann-Whitney algorithm with tie handling
- Epsilon clipping (1e-15) prevents log(0) in LogLoss/MulticlassLogLoss  
- MAPE uses epsilon denominator to handle zero labels
- All metrics implement `Send + Sync` for parallel evaluation
