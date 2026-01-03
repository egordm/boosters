# RFC-0005: Objectives and Metrics

**Status**: Implemented (OutputTransform pending for schema v3)  
**Created**: 2025-12-15  
**Updated**: 2026-01-03  
**Scope**: Loss functions, evaluation metrics, output transformations, early stopping

## Summary

Objectives compute gradients for training; metrics evaluate model quality.
Separation allows training with one loss and evaluating with another.

Output transformations convert raw predictions to interpretable values
(probabilities, counts). They are decoupled from objectives for clean persistence.

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
    
    /// Task type for output interpretation (Regression, Classification, Ranking)
    fn task_kind(&self) -> TaskKind;
    
    /// Expected target encoding
    fn target_schema(&self) -> TargetSchema;
    
    /// Transform raw predictions (e.g., sigmoid, softmax)
    fn transform_predictions_inplace(&self, predictions: ArrayViewMut2<f32>) -> PredictionKind;
    
    /// Return the output transformation for inference. [PROPOSED for v3]
    /// This decouples persistence from training-specific parameters.
    fn output_transform(&self) -> OutputTransform;
    
    fn name(&self) -> &'static str;
}
```

### TaskKind vs OutputTransform

These serve different purposes:

| Concept | TaskKind | OutputTransform |
| ------- | -------- | --------------- |
| Describes | ML problem type | Mathematical transform |
| Values | Regression, Binary, Multiclass, Ranking | Identity, Sigmoid, Softmax |
| Used for | Model metadata, validation | Inference-time prediction |
| Overlap | Different tasks can share transforms | (see above) |

`TaskKind` is semantic (what problem are we solving?). `OutputTransform` is mechanical
(what math do we apply?). Both regression and ranking use Identity transform but have
different TaskKinds.

## Implemented Objectives

### Regression

| Objective | Gradient | Hessian | Base Score | Transform |
| --------- | -------- | ------- | ---------- | --------- |
| `SquaredLoss` | `pred - target` | `1.0` | Weighted mean | Identity |
| `AbsoluteLoss` | `sign(pred - target)` | `1.0` | Weighted median | Identity |
| `PinballLoss` | `α - 1` or `α` | `1.0` | Weighted quantile | Identity |
| `PseudoHuberLoss` | Huber gradient | Huber hessian | Weighted median | Identity |
| `PoissonLoss` | `exp(pred) - target` | `exp(pred)` | `log(mean)` | Identity |

### Classification

| Objective | Gradient | Hessian | Base Score | Transform |
| --------- | -------- | ------- | ---------- | --------- |
| `LogisticLoss` | `σ(pred) - target` | `σ(1-σ)` | Log-odds | Sigmoid |
| `HingeLoss` | `-y` if margin < 1 | `1.0` | Zero | Identity |
| `SoftmaxLoss` | `p_c - I{c=label}` | `p_c(1-p_c)` | Log class freqs | Softmax |

### Ranking

| Objective | Description | Transform |
| --------- | ----------- | --------- |
| `LambdaRankLoss` | LambdaMART for NDCG optimization | Identity |

## Output Transformations

### Design Rationale

Previously, the full `Objective` enum was persisted in model files to support
prediction-time transformations. This caused issues:

1. **Training-only parameters persisted** — `PinballLoss { alphas }` and
   `PseudoHuberLoss { delta }` stored hyperparameters never used at inference.

2. **Custom objectives couldn't load** — Models trained with custom objectives
   serialized as `Custom { name }` but couldn't deserialize back.

3. **Schema bloat** — The schema mirrored all training objectives when only
   3 transformation behaviors exist.

### OutputTransform Enum

Only **3 transformation types** exist:

```rust
/// Minimal information needed for inference-time output transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTransform {
    /// No transformation (identity function).
    /// Used by: SquaredLoss, AbsoluteLoss, PinballLoss, PseudoHuberLoss,
    /// PoissonLoss, HingeLoss, LambdaRankLoss.
    Identity,
    
    /// Sigmoid: σ(x) = 1 / (1 + exp(-x))
    /// For binary classification (LogisticLoss).
    Sigmoid,
    
    /// Softmax: exp(x_i) / Σexp(x_j) across classes.
    /// For multiclass classification (SoftmaxLoss).
    /// n_classes derived from prediction array shape, not persisted.
    Softmax,
}

impl OutputTransform {
    /// Apply transformation in-place.
    ///
    /// Layout: predictions[output, sample] (output-major, shape [n_outputs, n_samples]).
    /// Softmax normalizes across outputs (axis 0) for each sample.
    ///
    /// # Edge Cases
    /// - Sigmoid clamps extreme values to avoid overflow
    /// - Softmax uses max-subtraction for numerical stability
    /// - NaN inputs produce NaN outputs (garbage-in, garbage-out)
    pub fn transform_inplace(&self, predictions: ArrayViewMut2<f32>) {
        match self {
            Self::Identity => { /* no-op */ }
            Self::Sigmoid => {
                predictions.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
            }
            Self::Softmax => {
                // Softmax across outputs (axis 0) for each sample (axis 1)
                // predictions shape: [n_classes, n_samples]
                for mut col in predictions.axis_iter_mut(Axis(1)) {
                    let max = col.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_sum: f32 = col.iter().map(|&x| (x - max).exp()).sum();
                    col.mapv_inplace(|x| (x - max).exp() / exp_sum);
                }
            }
        }
    }
}
```

### Objective → Transform Mapping

| Objective | OutputTransform |
| --------- | --------------- |
| SquaredLoss | `Identity` |
| AbsoluteLoss | `Identity` |
| PinballLoss | `Identity` |
| PseudoHuberLoss | `Identity` |
| PoissonLoss | `Identity` |
| LogisticLoss | `Sigmoid` |
| HingeLoss | `Identity` |
| SoftmaxLoss | `Softmax` |
| LambdaRankLoss | `Identity` |
| Custom | Via `ObjectiveFn::output_transform()` |

**Note on Poisson**: Uses identity transform (raw log-lambda values).
This differs from XGBoost which applies `exp()` for expected counts.
Users requiring counts should apply `exp()` themselves.

### Model Storage

```rust
pub struct GBDTModel {
    meta: ModelMeta,           // Includes objective_name for debugging
    forest: Forest<ScalarLeaf>,
    output_transform: OutputTransform,
}

impl GBDTModel {
    pub fn predict(&self, data: &Dataset, n_threads: usize) -> Array2<f32> {
        let mut output = self.predict_raw(data, n_threads);
        self.output_transform.transform_inplace(output.view_mut());
        output
    }
}
```

### Schema Representation

Schema v3 replaces `objective: ObjectiveSchema` with `output_transform: OutputTransformSchema`.
See RFC-0016 for schema versioning strategy.

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputTransformSchema {
    Identity,
    Sigmoid,
    Softmax,
}

pub struct GBDTModelSchema {
    pub meta: ModelMetaSchema,
    pub forest: ForestSchema,
    pub output_transform: OutputTransformSchema,
}
```

The objective name is stored in `ModelMeta.objective_name: Option<String>` for
debugging and reproducibility without affecting the transformation logic.

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

`PredictionKind` is **semantic**—it describes what the predictions *mean*.
`OutputTransform` is **mechanical**—it describes what math to apply.

They are related but not 1:1:

| OutputTransform | PredictionKind | Notes |
| --------------- | -------------- | ----- |
| Identity | Raw | Regression, ranking |
| Identity | Quantile | PinballLoss (semantic differs) |
| Sigmoid | Probability | Binary classification |
| Softmax | Probabilities | Multiclass classification |

The `OutputTransform` enum provides a helper:

```rust
impl OutputTransform {
    /// Returns the semantic prediction kind after this transform.
    pub fn prediction_kind(&self) -> PredictionKind {
        match self {
            Self::Identity => PredictionKind::Raw,
            Self::Sigmoid => PredictionKind::Probability,
            Self::Softmax => PredictionKind::Probabilities,
        }
    }
}
```

Note: PinballLoss returns `PredictionKind::Quantile` from `transform_predictions_inplace()`
despite using Identity transform—the semantic meaning comes from the objective, not the math.
This is why `PredictionKind` exists separately from `OutputTransform`: semantic meaning can
differ even when the mathematical transformation is identical.

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

| Path | Contents | Status |
| ---- | -------- | ------ |
| `training/objectives/mod.rs` | `ObjectiveFn` trait, re-exports | Existing |
| `training/objectives/regression.rs` | Squared, Absolute, Pinball, Poisson | Existing |
| `training/objectives/classification.rs` | Logistic, Hinge, Softmax | Existing |
| `training/metrics/mod.rs` | `MetricFn` trait, `Metric` enum | Existing |
| `training/metrics/regression.rs` | RMSE, MAE, MAPE | Existing |
| `training/metrics/classification.rs` | LogLoss, Accuracy, AUC | Existing |
| `training/callback.rs` | `EarlyStopping` | Existing |
| `model/transform.rs` | `OutputTransform` enum | **New (v3)** |
| `persist/schema.rs` | `OutputTransformSchema` | **Update (v3)** |

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

### Schema v3 Changes (DD-6 through DD-8)

**DD-6: Decoupled OutputTransform.** Objectives specify their output transform
via trait method, but only the transform is persisted—not training parameters.
This enables custom objectives and smaller model files.

**DD-7: Objective name in ModelMeta.** Store `objective_name: Option<String>`
for debugging/reproducibility without coupling persistence to objective types.

**DD-8: Derive n_classes from array shape.** Softmax transform derives class
count from prediction array dimensions rather than persisting it. Reduces
schema complexity and potential for inconsistency.

## Custom Objectives

Users can implement `ObjectiveFn` for custom losses:

```rust
struct MyObjective;

impl ObjectiveFn for MyObjective {
    fn n_outputs(&self) -> usize { 1 }
    
    fn output_transform(&self) -> OutputTransform {
        OutputTransform::Identity  // Specify transformation for persistence
    }
    
    fn transform_predictions_inplace(&self, predictions: ArrayViewMut2<f32>) -> PredictionKind {
        // No transformation needed, return semantic meaning
        PredictionKind::Raw
    }
    
    fn compute_gradients_into(&self, preds, targets, weights, grad_hess) {
        // Custom gradient/hessian computation
    }
    // ... other required methods (task_kind, target_schema, etc.)
}

let config = GBDTConfig::builder()
    .objective(MyObjective)
    .build()?;
```

Custom objectives now serialize correctly because only `OutputTransform` (not
the full objective) is persisted. The objective name is stored in metadata.

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

There are two related but distinct transform points:

1. **Training-time**: `ObjectiveFn::transform_predictions_inplace()` transforms
   predictions before metric evaluation. Returns `PredictionKind` so metrics
   know what they're receiving.

2. **Inference-time**: `OutputTransform::transform_inplace()` applies the same
   mathematical transformation during `model.predict()`. The OutputTransform
   is persisted with the model.

Both perform the same math (sigmoid for binary, softmax for multiclass, identity
for regression), but the training version is coupled to the objective while
the inference version is standalone.

```text
Training:  raw preds → ObjectiveFn.transform_predictions_inplace() → MetricFn.compute()
Inference: raw preds → OutputTransform.transform_inplace() → user
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
| Transform correctness | Compare to numpy sigmoid/softmax |
| Transform edge cases | NaN/Inf inputs, extreme values (±100) |
| Transform properties | sigmoid ∈ (0,1), softmax sums to 1.0 |
| Schema round-trip | Each objective type persists/loads correctly |
| Predict round-trip | train → save → load → predict gives same results |
| Schema migration | v2 models fail with clear error (clean break) |

## Changelog

- 2026-01-03: Added OutputTransform section (DD-6, DD-7, DD-8); decoupled
  transformations from objectives for cleaner persistence; 4 rounds of team review
- 2026-01-02: Updated ObjectiveFn trait with `output_transform()` method
- 2025-12-15: Initial RFC
