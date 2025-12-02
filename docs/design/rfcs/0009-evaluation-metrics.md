# RFC-0009: Evaluation Metrics

- **Status**: Draft
- **Created**: 2024-11-30
- **Updated**: 2024-11-30
- **Depends on**: RFC-0008 (GBLinear Training)
- **Scope**: Training evaluation metrics and multiple evaluation sets

## Summary

Add a metrics module for evaluating model performance during training. Support configurable metrics for logging and early stopping, multiple named evaluation sets, and batch computation for multi-output models.

## Motivation

Currently, training uses hardcoded metrics (accuracy for classification, RMSE for regression). Users need:

1. **Configurable metrics** — Choose RMSE vs MAE vs custom metrics
2. **Early stopping** — Stop training when a metric stops improving
3. **Multiple evaluation sets** — Track train + validation + test performance
4. **Named datasets** — Identify metrics in logs (e.g., `train-rmse`, `val-rmse`)
5. **Multi-output support** — Metrics for multiclass and multi-quantile models

XGBoost supports these via `eval_metric` parameter and `evals` list. We should provide similar flexibility.

## Design

### Overview

Metrics are stateless functions that compute a scalar from predictions and labels. They support:
- Single-output (regression, binary classification)
- Multi-output (multiclass, multi-quantile)
- Batch computation over all samples
- Optional sample weights

### Core Trait

```rust
/// Evaluation metric for model performance.
pub trait Metric: Send + Sync {
    /// Metric name for logging (e.g., "rmse", "mae", "accuracy").
    fn name(&self) -> &str;
    
    /// Evaluate metric on predictions vs labels.
    ///
    /// - `predictions`: Flat buffer, shape (n_samples, n_outputs)
    /// - `labels`: Ground truth, shape (n_samples,) or (n_samples, n_outputs)
    /// - `n_outputs`: Number of output columns (1 for regression/binary)
    fn evaluate(&self, predictions: &[f32], labels: &[f32], n_outputs: usize) -> f32;
}
```

### Standard Metrics

| Metric | Name | Use Case |
|--------|------|----------|
| `Rmse` | `"rmse"` | Regression |
| `Mae` | `"mae"` | Regression |
| `Mape` | `"mape"` | Regression (percentage error) |
| `LogLoss` | `"logloss"` | Binary classification |
| `Accuracy` | `"accuracy"` | Classification |
| `MulticlassLogLoss` | `"mlogloss"` | Multiclass |
| `QuantileLoss` | `"quantile"` | Quantile regression |

### Evaluation Set

```rust
/// Named dataset for evaluation during training.
pub struct EvalSet<'a, D> {
    /// Dataset name (appears in logs as prefix)
    pub name: &'a str,
    /// Feature matrix
    pub data: &'a D,
    /// Labels
    pub labels: &'a [f32],
}
```

### Training Integration

```rust
pub struct LinearTrainerConfig {
    // ... existing fields ...
    
    /// Metrics to evaluate each round.
    pub eval_metrics: Vec<Box<dyn Metric>>,
    
    /// Which eval set index to use for early stopping (default: last).
    pub early_stopping_eval_set: usize,
    
    /// Which metric index to use for early stopping (default: first).
    pub early_stopping_metric: usize,
}

impl LinearTrainer {
    /// Train with multiple evaluation sets.
    pub fn train_with_evals<D, L>(
        &self,
        train_data: &D,
        train_labels: &[f32],
        loss: &L,
        eval_sets: &[EvalSet<D>],
    ) -> LinearModel;
}
```

### Logging Format

```
[0]  train-rmse:15.2341  val-rmse:16.1234  test-rmse:16.5678
[1]  train-rmse:12.4521  val-rmse:13.2345  test-rmse:13.8901
...
```

## Design Decisions

### DD-1: Stateless Metrics vs Cached State

**Context**: Some metrics could cache intermediate results.

**Decision**: Keep metrics stateless. Cache computation is minimal for element-wise metrics, and stateless design is simpler and thread-safe.

### DD-2: Multi-Output Handling

**Context**: Multiclass and multi-quantile produce multiple predictions per sample.

**Decision**: Pass `n_outputs` parameter. Metrics interpret the flat buffer according to their semantics:
- Classification metrics use argmax across outputs
- Quantile metrics average pinball loss across quantiles
- Regression metrics (on multi-output) average across outputs

### DD-3: Default Early Stopping Metric

**Context**: Which metric/eval_set should control early stopping?

**Decision**: Default to last eval set and first metric. This matches XGBoost behavior where validation set is typically passed last.

## Integration

| Component | Integration Point | Notes |
|-----------|------------------|-------|
| RFC-0008 | `LinearTrainer` | Add `eval_metrics` config, `train_with_evals` method |
| Logging | `TrainingLogger` | Format metric values with dataset prefixes |
| Early Stopping | `EarlyStopping` callback | Use configurable metric for patience check |

## Future Work

- [ ] Weighted metrics (sample weights)
- [ ] Ranking metrics (NDCG, MAP)
- [ ] AUC and ROC metrics
- [ ] Custom metric functions (closures)
- [ ] Per-output metric breakdown in logs

## References

- [XGBoost Metric Interface](https://github.com/dmlc/xgboost/blob/master/include/xgboost/metric.h)
- [XGBoost Elementwise Metrics](https://github.com/dmlc/xgboost/blob/master/src/metric/elementwise_metric.cu)
- [XGBoost EvalOneIter](https://github.com/dmlc/xgboost/blob/master/src/learner.cc#L1176)

## Changelog

- 2024-11-30: Initial draft
