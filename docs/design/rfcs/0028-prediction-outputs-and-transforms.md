# RFC 0028: Prediction Outputs & Objective Transforms

- Status: Draft
- Author: (TBD)
- Last updated: 2025-12-13

## Summary

We should make **prediction semantics** (raw margin vs probability vs class index, etc.) a first-class part of the library design.

This RFC proposes:

1. A single **objective-owned prediction transform** function (XGBoost-style `PredTransform`, but expressed in Rust) that converts **raw model outputs** into a **default, user-facing prediction representation**.
2. A revised prediction output type that carries explicit **output kind** (margin, probability, class index, value, rank score, …) and supports **shape-changing transforms** (e.g., multiclass logits `K` → class index `1`).
3. A clear separation between:
   - training objectives (gradients/hessians in margin space)
   - prediction transforms (objective → user-facing output)
   - evaluation metrics (computed by metric implementations, not the objective)

This RFC also proposes reorganizing objective code into **task-specific modules** (LightGBM-style): binary, multiclass, ranking, regression, quantile.

## Motivation

### Current pain points

- `PredictionOutput` is a pure matrix container (row-major) with no semantic information about what values represent (margin/logit vs probability). See `PredictionOutput` in `src/inference/common/output.rs`.
- Many places (tests, examples, potential user code) must manually apply `sigmoid` / `softmax` transforms.
  - This is verbose.
  - It is easy to apply the wrong transform, or apply it with the wrong shape assumptions.
  - It is easy to silently compute incorrect metrics (e.g., feeding margins into logloss).

### Why the objective should define the transform

XGBoost explicitly models this separation:

- raw predictions are margins
- objectives define an activation/inverse-link via `PredTransform`
- metrics expect predictions in the appropriate space

Reference (XGBoost):

- `PredTransform`, `EvalTransform`, `ProbToMargin` are part of `ObjFunction` in `xgboost/include/xgboost/objective.h`.

Even if we don’t mirror XGBoost’s full interface, a **single** objective-owned transform gives us:

- consistent inference semantics
- consistent evaluation semantics (via metrics, not objective)
- fewer bugs in tests and downstream usage

## Goals

- **Make prediction semantics explicit**: raw margin/logit vs probability vs class index vs value.
- Support **shape-changing transforms** where needed (e.g., `K` → `1`).
- Keep the API ergonomic:
  - allow calling `predict_raw()` to get margins
  - allow calling `objective.transform_prediction(raw)` to get default predictions
  - later allow `model.predict(&objective, …)` convenience
- Keep metrics separate: evaluation is performed by `Metric` implementations.
- Make multi-output / multi-target behavior explicit enough to avoid “it works by convention” traps.

## Non-goals

- This RFC does not propose changing the training math (gradients/hessians) or the internal margin-space representation used by trainers.
- This RFC does not introduce a full “Learner” abstraction like XGBoost (though it suggests an optional wrapper for ergonomics).
- This RFC does not attempt to cover SHAP/leaf-index contributions (but the type design should not block it).

## Background: Current Design

### Training

- `Objective` defines:
  - `n_outputs()`
  - `compute_gradients()`
  - `compute_base_score()`

The trait lives in `src/training/objectives/mod.rs` and uses **column-major layout** for predictions/targets during training.

### Inference

- Both GBDT and GBLinear inference return `PredictionOutput` (row-major) without semantic labeling.
- Output transforms exist as free functions (`sigmoid_inplace`, `softmax_rows`) in `src/inference/common/mod.rs`.

### Metrics

- Metrics often assume a specific space:
  - `LogLoss` expects probabilities
  - `Accuracy` expects probabilities for thresholding
  - multiclass accuracy expects class indices (currently)

## Proposed Design

### 1) Add a single transform function to the objective

Add one method to the objective interface.

Important style constraint: objectives should **explicitly** declare their metadata (task kind, schemas). Avoid trait defaults unless they are true convenience methods layered on top of required methods.

```rust
use crate::inference::common::PredictionOutput;

pub trait Objective {
    // (existing methods)

    /// Transform raw model output (margins/logits) into the objective’s default
    /// user-facing prediction representation.
    ///
    /// - Input: row-major raw predictions with shape (n_rows, n_outputs)
    /// - Output: predictions with explicit semantics and potentially different shape
  fn transform_prediction(&self, raw: PredictionOutput) -> Predictions;

  /// Task kind (drives defaults, validation, and output semantics).
  fn task_kind(&self) -> TaskKind;

  /// How targets are laid out / interpreted for this objective.
  fn target_schema(&self) -> TargetSchema;

  /// Recommended default metric (trainer may use it if user doesn't specify).
  fn default_metric(&self) -> MetricKind;
}
```

Key points:

- **Single transform function**: avoids multiple vague transform APIs.
- The transform consumes `PredictionOutput`, allowing in-place transforms without extra allocation when shapes don’t change.
- The output is a new type (`Predictions`) that carries semantic information.

Note: objectives that are identity transforms (e.g., squared loss) still implement `transform_prediction` explicitly, returning `PredictionKind::Value`.

### 2) Introduce a semantic prediction type

Keep `PredictionOutput` as a raw matrix container, but introduce a wrapper that carries semantic meaning.

```rust
/// What do these values represent?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionKind {
    /// Raw margins / logits / scores produced directly by the model.
    Margin,

    /// Regression-style value (identity transform) or mean parameter (e.g., exp for Poisson).
    Value,

    /// Probabilities in [0, 1] (binary) or rows that sum to 1 (multiclass).
    Probability,

    /// Predicted class index (0..K-1), stored as f32 for compatibility with existing metric APIs.
    ClassIndex,

    /// Ranking score (typically margin-like; objective decides).
    RankScore,
}

/// A semantic prediction object.
#[derive(Debug, Clone)]
pub struct Predictions {
    pub kind: PredictionKind,
    pub output: PredictionOutput,
}

impl Predictions {
    pub fn raw_margin(output: PredictionOutput) -> Self {
        Self { kind: PredictionKind::Margin, output }
    }
}
```

This enables:

- metrics and user code to assert/check expected spaces
- objective transforms to change both semantics and shape

### 3) Explicitly model shape changes

Some objectives will transform **(n_rows, n_outputs)** into:

- same shape, different space
  - logistic: `(n_rows, 1)` margin → `(n_rows, 1)` probability
  - softmax: `(n_rows, K)` logits → `(n_rows, K)` probability
- different shape
  - softmax default output might be `(n_rows, 1)` class index (optional design choice)

This RFC recommends:

- Keep `PredictionOutput` as a general matrix.
- Allow `Predictions` to carry any shape.
- Provide conventions:
  - `PredictionKind::Probability` expects either `num_groups == 1` or row-wise sum ≈ 1
  - `PredictionKind::ClassIndex` uses `num_groups == 1`

### 4) Don’t embed evaluation into the objective

Evaluation should be done by metrics.

However, objectives should recommend a default metric (similar to XGBoost’s `DefaultEvalMetric`) to reduce configuration burden:

```rust
pub enum MetricKind {
    Rmse,
    Mae,
    LogLoss,
    MLogLoss,
    Accuracy,
    MulticlassAccuracy,
    // ...
}
```

Trainer API can remain:

- user explicitly supplies metric, OR
- trainer uses objective’s default metric if none is specified

#### Metric space expectations

Metrics operate on slices today (for speed and alignment with the objective interface). We can still make the expected space explicit by adding an *introspection-only* method:

```rust
pub trait Metric {
  fn compute(&self, n_rows: usize, n_outputs: usize, predictions: &[f32], targets: &[f32], weights: &[f32]) -> f64;

  /// What space does `predictions` need to be in?
  fn expected_prediction_kind(&self) -> PredictionKind;
}
```

This enables:

- trainers / wrappers to validate (`debug_assert`) that transformed predictions match the metric's expectation
- tests to avoid manual transforms by calling `objective.transform_prediction` and then using the metric

### 5) Make target schema explicit (multi-output / multi-target)

The library already has implicit conventions:

- SoftmaxLoss:
  - predictions: `(n_rows, K)`
  - targets: `(n_rows, 1)` class indices
- LogisticLoss multi-label:
  - predictions: `(n_rows, D)`
  - targets: `(n_rows, D)`
- Quantile:
  - predictions: `(n_rows, Q)`
  - targets: either `(n_rows, 1)` shared label or `(n_rows, Q)` per-quantile labels

To make this robust, introduce a target schema declaration:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSchema {
    /// One target value per row (e.g., class index, regression target).
    PerRow,
    /// One target value per (row, output) pair (e.g., multi-label).
    PerOutputPerRow,
    /// Either PerRow or PerOutputPerRow is acceptable.
    PerRowOrPerOutputPerRow,
}

pub trait Objective {
  fn target_schema(&self) -> TargetSchema;
}
```

This does not change training immediately, but:

- documents expectations
- enables consistent validation
- helps downstream tooling and future APIs

### 6) Model output “task kind” explicitly

Objectives correspond to tasks:

- regression
- binary classification
- multiclass classification
- ranking
- potentially others

Expose that as a property:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Regression,
    BinaryClassification,
    MulticlassClassification,
    Ranking,
    // MultiLabelClassification could be separate if needed.
}

pub trait Objective {
  fn task_kind(&self) -> TaskKind;
}
```

## Base score (intercept) semantics

We should optimize for two things:

1. Performance: trainers and predictors should continue to operate in **margin space** internally.
2. User friendliness: users should be able to reason about base score without knowing implementation details.

### Proposed rule

- Internally, model biases/base-scores are always stored and applied in **margin space**.
- `Objective::compute_base_score` continues to produce margin-space base scores (this is already true for logistic in `LogisticLoss`, which returns log-odds).

### Optional (future) ergonomic link for user-provided base_score

Some objectives have a natural inverse-link (e.g., probability ↔ logit). If we later add a user-facing API where base_score is specified in the *transformed* space (e.g., probability 0.5), we can support it without complicating the core objective trait by adding an **optional extension trait**:

```rust
pub trait ObjectiveLink {
  /// Convert a user-facing base score into margin space.
  fn base_score_to_margin(&self, base_score: f32) -> f32;
}
```

This keeps the core proposal aligned with “single transform function on Objective”, while still leaving room for user-friendly base_score configuration later.

This is primarily for:

- validation
- output selection defaults
- UI/metadata

## Objective-specific transform behavior

### Logistic (binary)

- Task: BinaryClassification
- Raw: margin/logit `(n_rows, 1)`
- Default transformed output: Probability `(n_rows, 1)` via sigmoid

### Hinge

- Task: BinaryClassification
- Raw: margin `(n_rows, 1)`
- Default transformed output: Margin `(n_rows, 1)` (identity)
  - Rationale: hinge is naturally margin-based; users can threshold at 0.

### Softmax (multiclass)

Default = Probability `(n_rows, K)` via softmax.

Rationale: probability output is strictly richer; class index can always be derived.

### Squared / Absolute / PseudoHuber

- Task: Regression
- Raw: Value `(n_rows, 1)`
- Default transformed: Value `(n_rows, 1)` (identity)

### Poisson / Gamma-like

- Task: Regression (count/positive)
- Raw: log-mean `(n_rows, 1)`
- Default transformed: Value `(n_rows, 1)` via exp

### Quantile (pinball)

- Task: Regression
- Raw: Value `(n_rows, Q)` (quantiles)
- Default transformed: Value `(n_rows, Q)` (identity)

### Ranking (LambdaRank)

- Task: Ranking
- Raw: RankScore `(n_rows, 1)` (usually margin-like)
- Default transformed: RankScore `(n_rows, 1)` (identity)

## API Usage Sketch

### Manual predict → transform (what you asked for)

```rust
let raw: PredictionOutput = model.predict_batch(&data, &base_scores);
let preds: Predictions = objective.transform_prediction(raw);

match preds.kind {
    PredictionKind::Probability => {
        // metric can consume these
    }
    PredictionKind::Margin => {
        // margin-based metrics / thresholding at 0
    }
    _ => { /* ... */ }
}
```

### Optional ergonomic wrapper (future)

```rust
pub struct ModelWithObjective<M, O> {
    pub model: M,
    pub objective: O,
}

impl<M, O> ModelWithObjective<M, O>
where
    M: PredictRaw,
    O: Objective,
{
    pub fn predict(&self, data: &ColMatrix<f32>, base: &[f32]) -> Predictions {
        let raw = self.model.predict_raw(data, base);
        self.objective.transform_prediction(raw)
    }
}
```

This keeps the transform definition in the objective, but allows the model to apply it conveniently.

## Compatibility & Migration

### Step 1 (minimal)

- Add `Predictions` + `PredictionKind`.
- Add `Objective::transform_prediction` as a required method.
- Implement `transform_prediction` for existing objectives:
  - LogisticLoss → sigmoid
  - SoftmaxLoss → softmax
  - PoissonLoss → exp
  - others → identity

### Step 2 (reduce duplication)

- Update tests and examples to call `objective.transform_prediction(raw)` instead of manually calling `sigmoid_inplace`/`softmax_rows`.

### Step 3 (optional ergonomics)

- Provide `predict_raw` and `predict` convenience APIs via wrappers or model methods.

## Open Questions

1. Should we introduce an optional `ObjectiveLink` extension trait for base_score conversion (future ergonomics), or keep base_score margin-only for v1?
2. Should we add trainer-level debug validation that `objective.transform_prediction(...).kind` matches `metric.expected_prediction_kind()`?

## Why this matches the feedback

- Single transform function: `transform_prediction`.
- Prediction → transform can be manual now; model/wrapper can do it later.
- Objective does not do evaluation; it only recommends a metric.
- Objectives explicitly declare `target_schema` and `task_kind` (no trait defaults for core behavior).
- Explicitly accounts for objectives with different target schemas and possible output shape changes.
- Introduces explicit output kinds (classification/regression/ranking + raw vs transformed) so the API can evolve without ambiguous conventions.

## Objective module organization (LightGBM-style)

LightGBM organizes objectives by task (binary, multiclass, rank, regression, …). We should mirror that for clarity and extensibility.

### Proposed module layout

Change `src/training/objectives/` from:

- `classification.rs`
- `regression.rs`

to:

- `binary.rs` (logistic, hinge)
- `multiclass.rs` (softmax)
- `ranking.rs` (lambdarank)
- `regression.rs` (squared, absolute, huber, poisson, etc.)
- `quantile.rs` (pinball, multi-quantile)
- `mod.rs` (trait definitions + re-exports)

### Compatibility strategy

- Keep public re-exports unchanged (`pub use ...`) so downstream imports continue to work.
- Move only definitions, not public names.
- Adjust internal unit tests to new module paths.

This is a mechanical refactor but pays off by keeping each objective focused, matching user expectations and upstream libraries.
