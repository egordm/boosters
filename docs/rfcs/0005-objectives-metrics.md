# RFC-0005: Objectives and Metrics

**Status**: Implemented  
**Created**: 2025-12-15  
**Updated**: 2026-01-02  
**Scope**: Loss functions, evaluation metrics, early stopping

## Summary

Objectives compute gradients for training; metrics evaluate model quality.
Separation allows training with one loss and evaluating with another.

## Why Separate Objectives and Metrics?

| Concern | Objective | Metric |
| ------- | --------- | ------ |
| When used | During training (each tree) | During evaluation (each round) |
| Output | Gradients and Hessians | Single score |
| Requirements | Must be differentiable | Any computable |
| Example | LogLoss (differentiable) | AUC (non-differentiable) |

Train with `LogisticLoss`, evaluate with `AUC`.

## ObjectiveFn Trait

```rust
pub trait ObjectiveFn: Send + Sync {
    /// Number of outputs per sample (1 for regression, K for K-class)
    fn n_outputs(&self) -> usize;
    
    /// Compute gradients and hessians into output buffer
    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        grad_hess: ArrayViewMut2<GradsTuple>,
    );
    
    /// Initial prediction before any trees
    fn compute_base_score(
        &self,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> Vec<f32>;
    
    /// Task type for output interpretation
    fn task_kind(&self) -> TaskKind;
    
    /// Expected target encoding
    fn target_schema(&self) -> TargetSchema;
    
    /// Transform raw predictions (e.g., sigmoid, softmax)
    fn transform_predictions_inplace(&self, predictions: ArrayViewMut2<f32>) -> PredictionKind;
    
    fn name(&self) -> &'static str;
}
```

## Implemented Objectives

### Regression

| Objective | Gradient | Hessian | Base Score |
| --------- | -------- | ------- | ---------- |
| `SquaredLoss` | `pred - target` | `1.0` | Weighted mean |
| `AbsoluteLoss` | `sign(pred - target)` | `1.0` | Weighted median |
| `PinballLoss` | `α - 1` or `α` | `1.0` | Weighted quantile |
| `PseudoHuberLoss` | Huber gradient | Huber hessian | Weighted median |
| `PoissonLoss` | `exp(pred) - target` | `exp(pred)` | `log(mean)` |

### Classification

| Objective | Gradient | Hessian | Base Score |
| --------- | -------- | ------- | ---------- |
| `LogisticLoss` | `σ(pred) - target` | `σ(1-σ)` | Log-odds |
| `HingeLoss` | `-y` if margin < 1 | `1.0` | Zero |
| `SoftmaxLoss` | `p_c - I{c=label}` | `p_c(1-p_c)` | Log class freqs |

### Ranking

| Objective | Description |
| --------- | ----------- |
| `LambdaRankLoss` | LambdaMART for NDCG optimization |

## MetricFn Trait

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

## Implemented Metrics

### Regression

- `Rmse`: Root Mean Squared Error (lower is better)
- `Mae`: Mean Absolute Error
- `Mape`: Mean Absolute Percentage Error
- `QuantileMetric`: Pinball loss for quantile regression
- `PoissonDeviance`: For count data

### Classification

- `LogLoss`: Binary cross-entropy
- `Accuracy`: Classification accuracy (threshold-based)
- `Auc`: Area Under ROC Curve (higher is better)
- `MulticlassLogLoss`: Softmax cross-entropy
- `MulticlassAccuracy`: Multiclass accuracy

## Early Stopping

```rust
pub struct EarlyStopping {
    patience: usize,
    higher_is_better: bool,
    best_value: Option<f64>,
    best_round: usize,
}

impl EarlyStopping {
    pub fn should_stop(&mut self, round: usize, value: f64) -> bool;
    pub fn best_round(&self) -> usize;
}
```

Stops training when validation metric doesn't improve for `patience` rounds.

## Target Schema

```rust
pub enum TargetSchema {
    Continuous,        // Real values (regression)
    Binary01,          // {0, 1} (logistic)
    BinarySigned,      // {-1, +1} (hinge)
    MulticlassIndex,   // [0, K) class indices
    CountNonNegative,  // Non-negative counts (Poisson)
}
```

Validates targets match objective expectations.

## Prediction Kinds

```rust
pub enum PredictionKind {
    Raw,          // Untransformed predictions
    Probability,  // Sigmoid-transformed (binary)
    Probabilities,// Softmax-transformed (multiclass)
    Quantile,     // Quantile estimates
}
```

Metrics specify expected prediction kind; trainer applies transformation.

## Multi-Output Layout

Data stored in **output-major** order:

```text
predictions: [output0_sample0, output0_sample1, ..., output1_sample0, ...]
gradients:   [same layout]
```

For 3 samples, 2 outputs:
- `predictions[0..3]` = output 0 for all samples
- `predictions[3..6]` = output 1 for all samples

## Files

| Path | Contents |
| ---- | -------- |
| `training/objectives/mod.rs` | `ObjectiveFn` trait, re-exports |
| `training/objectives/regression.rs` | Squared, Absolute, Pinball, Poisson |
| `training/objectives/classification.rs` | Logistic, Hinge, Softmax |
| `training/metrics/mod.rs` | `MetricFn` trait, `Metric` enum |
| `training/metrics/regression.rs` | RMSE, MAE, MAPE |
| `training/metrics/classification.rs` | LogLoss, Accuracy, AUC |
| `training/callback.rs` | `EarlyStopping` |

## Design Decisions

**DD-1: Trait-based objectives.** Enables custom objectives without modifying
library. Generic over objective type eliminates virtual dispatch.

**DD-2: Metric enum for runtime selection.** Common case is selecting metric
by name (from config). Enum provides dynamic dispatch without boxing.

**DD-3: Separate base score.** Initial prediction computed once, not per-tree.
Objective-specific: mean for squared loss, log-odds for logistic.

**DD-4: f64 for metric results.** Metrics may accumulate over many samples.
Higher precision for final score, even if predictions are f32.

**DD-5: WeightsView for optional weights.** Empty slice signals unweighted.
Avoids Option overhead in hot paths.

## Custom Objectives

Users can implement `ObjectiveFn` for custom losses:

```rust
struct MyObjective;

impl ObjectiveFn for MyObjective {
    fn n_outputs(&self) -> usize { 1 }
    fn compute_gradients_into(&self, preds, targets, weights, grad_hess) {
        // Custom gradient/hessian computation
    }
    // ... other required methods
}

let config = GBDTConfig::builder()
    .objective(MyObjective)
    .build()?;
```

Python: Custom objectives can be passed as callable (slower due to FFI overhead).

## Objective Selection

From config string (for Python/CLI):

```python
config = bst.GBDTConfig(
    objective="reg:squarederror",  # or "binary:logistic", "multi:softmax"
    metric="rmse",                  # or "auc", "logloss"
)
```

Objective names match XGBoost for compatibility.

## Prediction Transformation Flow

```
Raw predictions → Metric.expected_prediction_kind?
                      ↓
               Objective.transform_predictions_inplace()
                      ↓
               Transformed predictions → Metric.compute()
```

Metrics declare expected kind (Raw, Probability, etc.); trainer applies
appropriate transformation before evaluation.

## Performance

Gradient computation is vectorized where possible:

- LLVM auto-vectorizes simple element-wise operations
- LogLoss/Softmax use exp() which vectorizes on x86 (vexpf)
- No explicit SIMD intrinsics—compiler does well

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Gradient correctness | Compare to numerical gradient (finite diff) |
| Hessian correctness | Compare to numerical Hessian |
| Base score | Matches expected value for known distributions |
| Metric computation | Compare to sklearn metrics |
| Edge cases | All zeros, all ones, extreme values |
