# RFC-0024: Base Score Initialization

| Status | Implemented |
|--------|-------------|
| Created | 2025-12-02 |
| Updated | 2025-12-02 |

## Summary

Define how base scores (initial predictions) are computed for different loss functions,
ensuring proper initialization for both single-output and multi-output models.

## Motivation

Currently, booste-rs computes base scores inconsistently:

```rust
let base_scores: Vec<f32> = if num_outputs == 1 {
    vec![self.compute_base_score(labels)]  // Mean/fixed/zero
} else {
    vec![0.0; num_outputs]  // Always zero for multi-output!
};
```

This causes:

1. **Slow convergence for multiclass**: Starting from 0 means uniform class probabilities
2. **Inconsistent behavior**: Single-output uses mean, multi-output uses zero
3. **Different from XGBoost/LightGBM**: Both compute objective-specific initial scores

## Research: XGBoost Approach

XGBoost delegates base score computation to the objective function via `InitEstimation`:

### Default Behavior (`objective.cc`)
```cpp
void ObjFunction::InitEstimation(MetaInfo const& info, linalg::Vector<float>* base_score) const {
  auto n_targets = this->Targets(info);
  *base_score = linalg::Constant(this->ctx_, DefaultBaseScore(), n_targets);
}
// DefaultBaseScore() = 0.5 for most objectives
```

### FitIntercept (Regression-like objectives)
Fits a "stump" by computing optimal leaf weight from initial gradients:
```cpp
void FitIntercept::InitEstimation(...) {
  // Compute gradients at predictions=0
  new_obj->GetGradient(dummy_predt, info, 0, &gpair);
  // Fit optimal weight: -sum(grad) / sum(hess)
  tree::FitStump(this->ctx_, info, gpair, n_targets, base_score);
  // Transform back (e.g., sigmoid for logistic)
  this->PredTransform(base_score->Data());
}
```

### FitInterceptGlmLike (Simple mean)
For objectives where mean is appropriate:
```cpp
void FitInterceptGlmLike::InitEstimation(...) {
  common::SampleMean(this->ctx_, info.IsColumnSplit(), info.labels, base_score);
}
```

### Key Insight: Link Functions
XGBoost applies `ProbToMargin` to convert between probability space and margin space:
- **Regression L2**: `margin = base_score` (identity)
- **Logistic**: `margin = log(p / (1-p))` (logit)
- **Poisson**: `margin = log(base_score)`

## Research: LightGBM Approach

LightGBM uses `BoostFromScore(class_id)` on the objective function:

### Regression L2
```cpp
double BoostFromScore(int) const override {
  // Weighted mean of labels
  return suml / sumw;
}
```

### Binary Logistic
```cpp
double BoostFromScore(int) const override {
  // Compute weighted mean of positive labels
  double pavg = suml / sumw;
  pavg = std::clamp(pavg, kEpsilon, 1.0 - kEpsilon);
  // Return log-odds (logit)
  return std::log(pavg / (1.0 - pavg)) / sigmoid_;
}
```

### Multiclass Softmax
```cpp
double BoostFromScore(int class_id) const override {
  // Return log of class probability (prior)
  return std::log(std::max(kEpsilon, class_init_probs_[class_id]));
}
```
Where `class_init_probs_[k]` = count of class k / total samples.

### Poisson
```cpp
double BoostFromScore(int) const override {
  // Mean in log-space
  return Common::SafeLog(RegressionL2loss::BoostFromScore(0));
}
```

## Design Decision

**Chosen approach**: Add `init_base_score()` method to `Loss` trait. The trainer
always uses this method - no user override needed.

Rejected alternatives:

- **BaseScore enum** (Mean/Fixed/Zero/Auto): Over-engineered. Users never need to
  override objective-specific initialization.
- **boost_from_average boolean**: Even a simple toggle is unnecessary. If someone
  wants zeros, they can implement a custom loss.

## Implementation

### Loss trait method

```rust
pub trait Loss {
    /// Compute optimal initial scores for this objective.
    fn init_base_score(&self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32>;
}
```

### Trainer usage

```rust
// Always uses loss-specific initialization
let base_scores = self.loss.init_base_score(labels, None);
```

### Base Score Formulas

| Objective | Formula | Space |
|-----------|---------|-------|
| SquaredError | `mean(labels)` | Raw |
| PseudoHuber | `mean(labels)` | Raw |
| Quantile | `mean(labels)` | Raw |
| Logistic | `log(p / (1-p))` | Margin (pre-sigmoid) |
| Softmax | `[log(p_k)]` per class | Margin (pre-softmax) |
| Hinge | `0` | Raw |

## References

- XGBoost: `src/objective/init_estimation.cc`, `src/tree/fit_stump.h`
- LightGBM: `src/objective/*_objective.hpp` - `BoostFromScore()`

## Changelog

- 2025-12-02: Initial draft
- 2025-12-02: Implemented - always use `loss.init_base_score()`, no user override
