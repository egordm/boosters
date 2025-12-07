# RFC-0024: Base Score Initialization

| Status | Draft |
|--------|-------|
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
    vec![0.0; num_outputs]  // Always zero for multi-output
};
```

This is problematic because:

1. **Slow convergence for multiclass**: Starting from 0 for softmax means the model
   starts with uniform class probabilities, even when classes are imbalanced.

2. **Inconsistent behavior**: Single-output regression uses mean of labels,
   but multi-output regression (e.g., multi-quantile) uses zero.

3. **Different from XGBoost/LightGBM**: Both libraries compute objective-specific
   initial scores.

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

## Design Options

### Option A: Objective Method (Recommended)

Add `init_base_score()` method to `Loss` trait:

```rust
pub trait Loss: Send + Sync {
    fn gradient(&self, preds: &[f32], labels: &[f32], out: &mut [(f32, f32)]);
    
    /// Compute initial base scores for this objective.
    /// 
    /// Returns one score per output. For multi-output models (softmax, multi-quantile),
    /// returns `num_outputs` values.
    fn init_base_score(&self, labels: &[f32], num_outputs: usize) -> Vec<f32> {
        // Default: zeros (let BaseScore config override)
        vec![0.0; num_outputs]
    }
}
```

Implementations:

```rust
impl Loss for SquaredErrorLoss {
    fn init_base_score(&self, labels: &[f32], _num_outputs: usize) -> Vec<f32> {
        let mean = labels.iter().sum::<f32>() / labels.len() as f32;
        vec![mean]
    }
}

impl Loss for LogisticLoss {
    fn init_base_score(&self, labels: &[f32], _num_outputs: usize) -> Vec<f32> {
        let pos_count = labels.iter().filter(|&&l| l > 0.5).count();
        let p = (pos_count as f32 / labels.len() as f32).clamp(1e-7, 1.0 - 1e-7);
        // Return log-odds (in margin space)
        vec![(p / (1.0 - p)).ln()]
    }
}

impl Loss for SoftmaxLoss {
    fn init_base_score(&self, labels: &[f32], num_outputs: usize) -> Vec<f32> {
        // Count class frequencies
        let mut counts = vec![0usize; num_outputs];
        for &label in labels {
            let class_idx = label.round() as usize;
            if class_idx < num_outputs {
                counts[class_idx] += 1;
            }
        }
        
        let total = labels.len() as f32;
        // Return log-probabilities (in margin space, before softmax)
        counts.iter()
            .map(|&c| ((c as f32 / total).max(1e-7)).ln())
            .collect()
    }
}
```

### Option B: BaseScore Enum Extension

Extend `BaseScore` enum to support auto-detection:

```rust
pub enum BaseScore {
    /// Automatically compute from objective and labels.
    Auto,
    /// Use mean of labels (regression).
    Mean,
    /// Use a fixed value per output.
    Fixed(Vec<f32>),
    /// Use zero.
    Zero,
}
```

With `Auto`, the trainer would call `loss.init_base_score()`.

### Option C: Separate Initializer Trait

Create a separate trait for initialization:

```rust
pub trait BaseScoreInit {
    fn compute_base_scores(&self, labels: &[f32], num_outputs: usize) -> Vec<f32>;
}

// Implement for LossFunction enum
impl BaseScoreInit for LossFunction {
    fn compute_base_scores(&self, labels: &[f32], num_outputs: usize) -> Vec<f32> {
        match self {
            LossFunction::SquaredError => { /* mean */ }
            LossFunction::Logistic => { /* log-odds */ }
            LossFunction::Softmax { .. } => { /* log-priors */ }
            // ...
        }
    }
}
```

## Recommendation

**Option A (Objective Method)** is recommended because:

1. **Follows XGBoost/LightGBM pattern**: Both delegate to the objective
2. **Type-safe**: Each loss knows its own initialization
3. **Extensible**: Custom losses can provide custom initialization
4. **Default behavior**: Objectives can return zeros if no special init needed

## Implementation Plan

### Phase 1: Add trait method
1. Add `init_base_score()` with default implementation to `Loss` trait
2. Implement for `SquaredErrorLoss` (mean)
3. Implement for `LogisticLoss` (log-odds)

### Phase 2: Multi-output support
4. Implement for `SoftmaxLoss` (log-priors per class)
5. Implement for `QuantileLoss` (median of labels for all outputs)

### Phase 3: Trainer integration
6. Update `GBTreeTrainer` to use `loss.init_base_score()` when `BaseScore::Auto`
7. Make `BaseScore::Auto` the default

### Phase 4: Documentation
8. Document base score semantics for each objective
9. Add tests comparing convergence with/without proper initialization

## Base Score Formulas by Objective

| Objective | Base Score Formula | Space |
|-----------|-------------------|-------|
| SquaredError | `mean(labels)` | Raw |
| AbsoluteError | `median(labels)` | Raw |
| Logistic | `log(p / (1-p))` where `p = mean(labels > 0.5)` | Margin (pre-sigmoid) |
| Softmax | `[log(p_0), ..., log(p_K)]` where `p_k = count_k / n` | Margin (pre-softmax) |
| Poisson | `log(mean(labels))` | Log space |
| Quantile | `quantile(labels, alpha)` | Raw |
| Huber | `mean(labels)` | Raw |

## Migration

The change is backward compatible if we:
1. Keep `BaseScore::Mean` and `BaseScore::Fixed` working as before
2. Add `BaseScore::Auto` as new default
3. Existing code with explicit `BaseScore::Mean` continues to work

## Open Questions

1. **Weight support**: Should `init_base_score` accept optional sample weights?
   - XGBoost: Yes, uses weighted mean
   - LightGBM: Yes, uses weighted statistics
   - Recommendation: Add weights parameter for completeness

2. **Validation set initialization**: Should we use train labels or combined train+val?
   - XGBoost: Train only
   - LightGBM: Train only
   - Recommendation: Train only (validation is for evaluation)

3. **User override**: If user specifies `BaseScore::Fixed(0.5)`, should we still
   allow auto-detection?
   - Recommendation: No, user-specified values take precedence

## References

- XGBoost `src/objective/init_estimation.cc`
- XGBoost `src/tree/fit_stump.h`
- LightGBM `src/objective/regression_objective.hpp` - `BoostFromScore()`
- LightGBM `src/objective/binary_objective.hpp` - `BoostFromScore()`
- LightGBM `src/objective/multiclass_objective.hpp` - `BoostFromScore()`

## Changelog

- 2025-12-02: Initial draft
