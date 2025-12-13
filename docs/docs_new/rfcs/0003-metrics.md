# RFC-0003: Metrics

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: RFC-0001 (Data Matrix), RFC-0002 (Objectives)
- **Scope**: Evaluation metrics for model performance tracking

## Summary

Metrics measure model quality for monitoring, early stopping, and model selection. Unlike objectives (which provide optimization gradients), metrics report what we actually care about. This RFC defines the metric trait, built-in implementations, and their integration with the evaluation workflow.

## Overview

### Metric vs Objective

| Aspect | Objective (RFC-0002) | Metric (this RFC) |
|--------|---------------------|-------------------|
| Purpose | Optimization target | Quality measurement |
| Output | Gradients, Hessians | Single scalar value |
| Examples | LogLoss → gradients | AUC, Accuracy |
| When used | Every boosting round | Periodic evaluation |
| Multi-output | Per-output gradients | Aggregated scalar |

A model optimized with `Logistic` objective might be evaluated with `AUC` or `Accuracy` metrics.

### Component Hierarchy

```text
Metric (trait)
│
└── EvalMetric (enum)               ← Zero-cost dispatch for built-in metrics
    ├── Rmse
    ├── Mae
    ├── Mape
    ├── RSquared
    ├── PinballLoss { alphas }
    ├── LogLoss
    ├── Accuracy { threshold }
    ├── Auc
    ├── F1 { threshold }
    ├── MultiLogLoss
    └── MultiAccuracy

Evaluator (RFC-0006)
└── Uses Metric for periodic evaluation
```

### Data Flow

```text
predictions [n_samples × n_outputs]  ─┐
labels [n_samples × n_labels]        ─┼─► Metric.compute() ──► f64 (scalar)
weights [n_samples × 1] (optional)   ─┘

                    │
                    ▼
           Early stopping decision
           Logging / Callbacks
           Model selection
```

## Components

### Metric Trait

```rust
/// A metric for evaluating model quality.
///
/// Returns a single scalar value. For multi-output models,
/// the metric typically averages across outputs.
pub trait Metric: Send + Sync {
    /// Compute the metric value.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions: [n_samples × n_outputs]
    /// * `labels` - Ground truth: [n_samples × n_labels]
    /// * `weights` - Optional sample weights: [n_samples × 1]
    ///
    /// # Returns
    /// Scalar metric value (f64 for precision).
    fn compute<S: AsRef<[f32]>>(
        &self,
        predictions: &ColMatrix<f32, S>,
        labels: &ColMatrix<f32, S>,
        weights: Option<&ColMatrix<f32, S>>,
    ) -> f64;

    /// Whether higher values indicate better performance.
    fn higher_is_better(&self) -> bool;

    /// Name of the metric (for logging).
    fn name(&self) -> &'static str;
}
```

### EvalMetric Enum

```rust
/// Built-in evaluation metrics with zero-cost dispatch.
#[derive(Debug, Clone, PartialEq)]
pub enum EvalMetric {
    // Regression
    Rmse,
    Mae,
    Mape,
    RSquared,
    
    // Quantile regression
    PinballLoss { alphas: Vec<f32> },
    
    // Binary classification
    LogLoss,
    Accuracy { threshold: f32 },
    Auc,
    F1 { threshold: f32 },
    
    // Multiclass
    MultiLogLoss,
    MultiAccuracy,
}

impl Default for EvalMetric {
    fn default() -> Self { Self::Rmse }
}

impl Metric for EvalMetric {
    // ... dispatch to specific implementations
}
```

### Metric Summary Table

| Metric | Formula | Higher is Better | Multi-Output Handling |
|--------|---------|------------------|----------------------|
| Rmse | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | No | Average per-output RMSE |
| Mae | $\frac{1}{n}\sum\|y - \hat{y}\|$ | No | Average per-output MAE |
| Mape | $\frac{100}{n}\sum\|\frac{y - \hat{y}}{y}\|$ | No | Average per-output MAPE |
| RSquared | $1 - \frac{SS_{res}}{SS_{tot}}$ | Yes | Average per-output R² |
| PinballLoss | $\frac{1}{n}\sum\rho_\alpha(y - \hat{y})$ | No | Average across quantiles |
| LogLoss | $-\frac{1}{n}\sum[y\log p + (1-y)\log(1-p)]$ | No | Single output |
| Accuracy | Correct / Total | Yes | Single output |
| Auc | Area under ROC | Yes | Single output |
| F1 | $2 \cdot \frac{P \cdot R}{P + R}$ | Yes | Single output |
| MultiLogLoss | $-\frac{1}{n}\sum\log p_{y_i}$ | No | n_outputs = n_classes |
| MultiAccuracy | Correct class / Total | Yes | argmax across classes |

## Algorithms

All algorithms operate on **column-major** matrices.

### Root Mean Squared Error (RMSE)

```text
RMSE(predictions, labels, weights) -> f64:
  total_rmse = 0.0
  
  for k in 0..n_outputs:
    pred_col = predictions.col_slice(k)
    label_col = labels.col_slice(k)
    
    sum_sq_error = 0.0
    sum_weights = 0.0
    for i in 0..n_samples:
      error = pred_col[i] - label_col[i]
      w = weights.map(|w| w[0][i]).unwrap_or(1.0)
      sum_sq_error += w * error * error
      sum_weights += w
    
    total_rmse += sqrt(sum_sq_error / sum_weights)
  
  return total_rmse / n_outputs as f64
```

### Binary Log Loss

```text
LogLoss(predictions, labels, weights) -> f64:
  ε = 1e-15  // numerical stability
  
  pred_col = predictions.col_slice(0)
  label_col = labels.col_slice(0)
  
  sum_loss = 0.0
  sum_weights = 0.0
  for i in 0..n_samples:
    p = clamp(pred_col[i], ε, 1.0 - ε)
    y = label_col[i]
    w = weights.map(|w| w[0][i]).unwrap_or(1.0)
    sum_loss += w * -(y * log(p) + (1.0 - y) * log(1.0 - p))
    sum_weights += w
  
  return sum_loss / sum_weights
```

### Area Under ROC Curve (AUC)

Uses the Mann-Whitney U statistic approach:

```text
AUC(predictions, labels, weights) -> f64:
  // Sort indices by prediction (descending)
  indices = argsort_desc(predictions.col_slice(0))
  
  // Compute weighted class totals
  total_pos = 0.0
  total_neg = 0.0
  for i in 0..n_samples:
    w = weights.map(|w| w[0][i]).unwrap_or(1.0)
    if labels[0][i] == 1.0:
      total_pos += w
    else:
      total_neg += w
  
  if total_pos == 0 or total_neg == 0:
    return 0.5  // undefined, return random baseline
  
  // Compute AUC via ranked pairs
  auc = 0.0
  cum_pos = 0.0
  for idx in indices:
    w = weights.map(|w| w[0][idx]).unwrap_or(1.0)
    if labels[0][idx] == 1.0:
      cum_pos += w
    else:
      auc += w * cum_pos
  
  return auc / (total_pos * total_neg)
```

**Complexity**: O(n log n) for sorting, O(n) for accumulation.

### Multiclass Accuracy

```text
MultiAccuracy(predictions, labels, weights) -> f64:
  K = predictions.n_cols()  // number of classes
  
  correct = 0.0
  total = 0.0
  for i in 0..n_samples:
    // argmax across classes (strided access)
    pred_class = 0
    max_prob = predictions[0][i]
    for k in 1..K:
      if predictions[k][i] > max_prob:
        max_prob = predictions[k][i]
        pred_class = k
    
    true_class = labels[0][i] as usize
    w = weights.map(|w| w[0][i]).unwrap_or(1.0)
    
    if pred_class == true_class:
      correct += w
    total += w
  
  return correct / total
```

## Design Decisions

### DD-1: f64 Return Type

**Context**: Metrics accumulate many small values. What precision is needed?

**Decision**: Return `f64`.

**Rationale**:

- **Accumulation precision**: Summing 1M small values in f32 loses significant digits
- **Scientific convention**: Statistics typically use double precision
- **Negligible cost**: Metrics computed infrequently (not every round)
- **Consistent interface**: All metrics return same type

### DD-2: Single Scalar Return

**Context**: Multi-output models could report per-output metrics.

**Decision**: Return single aggregated scalar.

**Rationale**:

- **Early stopping simplicity**: One value to compare against best
- **Logging clarity**: One number per metric in output
- **Per-output option**: Call metric on single column if needed
- **Consistent with XGBoost/LightGBM**: Same behavior

### DD-3: Separate Metric and Objective Traits

**Context**: Should metrics and objectives share an interface?

**Decision**: Completely separate traits.

**Rationale**:

| Aspect | Objective | Metric |
|--------|-----------|--------|
| Signature | `(preds, labels, ...) → (&mut grads, &mut hess)` | `(preds, labels, ...) → f64` |
| Purpose | Optimization | Evaluation |
| Some have no counterpart | LambdaRank → ??? | AUC → ??? |
| Called | Every round | Periodically |

Merging would force unused methods or awkward defaults.

### DD-4: higher_is_better() Method

**Context**: How does the evaluator know which direction is "better"?

**Decision**: Each metric reports via `higher_is_better() -> bool`.

**Rationale**:

- **Early stopping**: Knows whether to look for increase or decrease
- **Model selection**: Picks model with best (not just different) score
- **Self-documenting**: No external lookup table needed

```rust
match metric.higher_is_better() {
    true => new_value > best_value + min_delta,   // improvement
    false => new_value < best_value - min_delta,  // improvement
}
```

### DD-5: Matrix Types for Input

**Context**: Use `&[f32]` slices or typed matrices?

**Decision**: `&ColMatrix<f32, S>` for all inputs.

**Rationale**:

- **Consistent with RFC-0001/0002**: Same types throughout
- **Dimensions explicit**: n_samples, n_outputs always known
- **Generic storage**: Owned or borrowed views work
- **Multi-output natural**: Each column = one output

## Integration

| Component | How Metrics are Used |
|-----------|---------------------|
| RFC-0006 (Evaluator) | Computes metrics on eval sets |
| Early Stopping | Tracks best metric value |
| Callbacks | Receives metric results for logging |
| Model Selection | Picks best iteration based on metric |

### Usage in Evaluation

```text
// In Evaluator.maybe_evaluate():
for eval_set in eval_sets:
    value = metric.compute(
        eval_set.predictions,
        eval_set.dataset.labels,
        eval_set.dataset.weights
    )
    results.push(EvalResult {
        name: eval_set.name,
        metric: metric.name(),
        value,
        higher_is_better: metric.higher_is_better(),
    })
```

## Metric Selection Guidelines

| Task | Recommended Metrics | Notes |
|------|--------------------| ------|
| Regression | RMSE, MAE, R² | RMSE penalizes large errors more |
| Binary Classification | AUC, LogLoss, F1 | AUC threshold-independent |
| Multiclass | MultiLogLoss, MultiAccuracy | LogLoss for probabilities |
| Quantile Regression | PinballLoss | Must match training alphas |
| Imbalanced Data | AUC, F1 | Accuracy misleading |

## Future Work

- [ ] Ranking metrics (NDCG, MAP, MRR)
- [ ] Calibration metrics (Brier score, reliability diagrams)
- [ ] Fairness metrics (demographic parity, equalized odds)
- [ ] Custom metric via Python callback
- [ ] Per-output metric reporting option

## References

- [scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [XGBoost Evaluation Metrics](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)
- [LightGBM Metrics](https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters)
