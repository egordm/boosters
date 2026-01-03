
# RFC-0005: Objectives, Metrics, and Output Transforms (vNext)

**Status**: Proposed (vNext)  
**Created**: 2025-12-15  
**Updated**: 2026-01-03  
**Scope**: Objectives (loss), metrics, output transforms, early stopping, persistence schema v3

This RFC is intentionally written as the target design “to be”.
Appendix A summarizes what changed from the previous design and can be deleted later.

## Summary

- Training uses an `Objective` enum to compute gradients and a base score.
- Evaluation uses a `Metric` enum to compute a scalar score.
- Inference-time interpretation uses a persisted `OutputTransform` (Identity/Sigmoid/Softmax).
- Default metric selection is centralized (`Objective` → `Metric`) when the user doesn’t specify one.

The API is enum-only by default: consumers almost never touch per-objective structs.

## Goals

- Prefer an enum-only architecture for objectives and metrics.
- Keep persistence minimal: do not persist full objectives; persist only `OutputTransform` (+ objective name for debugging).
- Avoid task-kind branching spread across trainers/models; derive behavior from `Objective`/`Metric` directly.
- Keep the training hot path fast (match dispatch, no allocation).

## Non-Goals

- Full compatibility with schema v2 model files (schema v3 is a clean break; v2 files are rejected with a clear error).
- Advanced metric bundles / multiple metrics per training run (can be added later).
- A comprehensive target-validation subsystem (can be added later if needed).

## Core Types (vNext)

### Objective (enum-only)

All objective behavior is expressed as methods on a single enum. Variants carry configuration
as struct-like fields when needed.

```rust
#[derive(Clone, Debug)]
pub enum Objective {
    // Regression
    SquaredLoss,
    AbsoluteLoss,
    PseudoHuberLoss { delta: f32 },
    PoissonLoss,
    PinballLoss { alphas: Vec<f32> },

    // Classification
    LogisticLoss,
    HingeLoss,
    SoftmaxLoss { n_classes: usize },

    // Optional extension point; still enum-shaped at the API boundary.
    Custom(CustomObjective),
}

impl Objective {
    pub fn compute_gradients_into(&self, /* ... */) { /* match self { ... } */ }
    pub fn compute_base_score(&self, /* ... */) -> Vec<f32> { /* match self { ... } */ }
    pub fn output_transform(&self) -> OutputTransform { /* simple match */ }
    pub fn name(&self) -> &str { /* match; builtins return literals; custom returns stored name */ }
    pub fn n_outputs(&self) -> usize { /* match */ }
}
```

We intentionally do not keep a separate `TaskKind` type in vNext.

### Metric (enum-only)

Similarly, metrics are a single enum with configuration where needed.

```rust
#[derive(Clone, Debug)]
pub enum Metric {
    None,

    // Regression
    Rmse,
    Mae,
    Mape,
    Quantile { alphas: Vec<f32> },
    PoissonDeviance,

    // Classification
    LogLoss,
    Accuracy { threshold: f32 },
    MarginAccuracy,
    Auc,
    MulticlassLogLoss,
    MulticlassAccuracy,

    Custom(CustomMetric),
}

impl Metric {
    pub fn compute(&self, /* ... */) -> f64 { /* match */ }
    pub fn expected_prediction_kind(&self) -> PredictionKind { /* match */ }
    pub fn higher_is_better(&self) -> bool { /* match */ }
    pub fn name(&self) -> &str { /* match; builtins return literals; custom returns stored name */ }
}
```

### OutputTransform (persisted)

Only three inference-time transforms exist, so the persisted form should be a tiny enum.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTransform {
    Identity,
    Sigmoid,
    Softmax,
}
```

Implementation note: `Sigmoid` and `Softmax` should be numerically stable (clamping / max-subtraction).

**Objective → OutputTransform mapping** (`Objective::output_transform()`):

- Regression + hinge + pinball + poisson → `Identity`
- Logistic → `Sigmoid`
- Softmax → `Softmax`

Poisson note: we keep `Identity` (raw log-lambda). If a user wants expected counts, apply `exp()` in user code.

## Removed Concepts

### TaskKind

We do not keep `TaskKind` in vNext.

Rationale:

- It duplicates information already implied by `Objective`.
- It tempts “task-kind switching” in random parts of the codebase.
- The useful bits for plumbing are already captured by `Objective::n_outputs()` and `Metric::expected_prediction_kind()`.

If we ever need user-facing introspection, it should be derived on-demand from the objective/metric
instead of being a separate first-class type.

### TargetSchema

We do not keep a separate `TargetSchema` in vNext.

Rationale:

- In the current codebase it is metadata-only (not actually enforced).
- It duplicates information already implied by the objective choice.

If we later want strict target validation, it should be a single centralized validation step
driven by `Objective` (not by extra types attached to objective implementations).

## Default Metric Selection (centralized)

When the user does not specify a metric, pick a default via a single mapping function.
No task-kind branching spread across multiple call sites.

```rust
pub fn default_metric_for_objective(objective: &Objective) -> Metric {
    match objective {
        // Regression
        Objective::SquaredLoss
        | Objective::AbsoluteLoss
        | Objective::PseudoHuberLoss { .. }
        | Objective::PoissonLoss => Metric::Rmse,
        Objective::PinballLoss { alphas } => Metric::Quantile { alphas: alphas.clone() },

        // Classification
        Objective::LogisticLoss => Metric::LogLoss,
        Objective::HingeLoss => Metric::MarginAccuracy,
        Objective::SoftmaxLoss { .. } => Metric::MulticlassLogLoss,

        // Custom
        Objective::Custom(_) => Metric::None,
    }
}
```

## Evaluation / Transform Flow

Metrics declare what they expect via `Metric::expected_prediction_kind()`.
Evaluation applies *only* the mechanical transform needed for that expectation.

Rule of thumb:

- If a metric expects `Margin` (raw), do not transform.
- Otherwise apply `objective.output_transform()` to a copy buffer and evaluate.

This keeps evaluation logic independent of any “task kind” concept.

## Custom objective / metric (boxed closures)

To keep the API enum-only while still allowing user-provided behavior, the `Custom` variants can hold
boxed callables plus a name.

Example shape:

```rust
pub struct CustomObjective {
    pub name: String,
    pub compute_gradients_into: Box<dyn Fn(/* ... */) + Send + Sync>,
    pub compute_base_score: Box<dyn Fn(/* ... */) -> Vec<f32> + Send + Sync>,
    pub output_transform: OutputTransform,
    pub n_outputs: usize,
}

pub struct CustomMetric {
    pub name: String,
    pub compute: Box<dyn Fn(/* ... */) -> f64 + Send + Sync>,
    pub expected_prediction_kind: PredictionKind,
    pub higher_is_better: bool,
}
```

This stays compatible with the “few functions” principle, but avoids requiring users to define a trait impl.
**Edge case behavior**: NaN/Inf inputs to `compute_gradients_into` result in NaN gradients (garbage-in, garbage-out).
No special handling is performed.

**Python custom objectives**: Python users pass callables to the bindings, which wrap them in `CustomObjective`.
This incurs FFI overhead per batch but keeps the Rust core clean.

## Model Storage (schema v3)

Models persist:

- `output_transform: OutputTransformSchema`
- `meta.objective_name: Option<String>` (debugging/reproducibility only)

Models do not persist:

- the full objective (and therefore do not persist training-only hyperparameters)

```rust
pub struct GBDTModelSchema {
    pub meta: ModelMetaSchema,  // includes objective_name, n_features, etc.
    pub forest: ForestSchema,
    pub output_transform: OutputTransformSchema,
}
```

`ModelMetaSchema` is defined in the persistence layer (see RFC-0016 for schema versioning strategy).
It contains metadata like `objective_name`, `n_features`, `feature_names`, etc.—but not training hyperparameters.

## Early Stopping

Early stopping continues to depend only on metric direction (`higher_is_better`).

## Testing Strategy (vNext)

| Category | What to test |
| -------- | ------------ |
| Gradient correctness | Compare to numerical gradient (finite differences) for each objective. |
| Hessian correctness | Compare to numerical Hessian where applicable. |
| Base score | Verify expected value for known target distributions (e.g., mean for squared loss, log-odds for logistic). |
| Transform properties | Sigmoid ∈ (0,1), softmax sums to 1.0, edge cases (±100, NaN). |
| Default metric exhaustiveness | Every `Objective` variant has a defined default (or explicitly `None`). |
| Custom objective validation | Mismatched `n_outputs` should error early during training setup. |
| Persistence round-trip | train → save → load → predict produces identical results. |
| Schema v2 rejection | Loading a v2 model returns a clear "unsupported schema version" error. |
| Metric::None behavior | Training with `Metric::None` skips evaluation and early stopping. |

## Appendix A: What Changed From The Previous Design

This section is a short “release notes” style summary and can be removed later.

1. **Enum-only architecture**: objectives and metrics are primarily used via enums, so vNext makes that the core design.
2. **No full objective persistence**: schema v3 persists only `OutputTransform` (and objective name in metadata).
3. **No TargetSchema**: removed as redundant metadata.
4. **Central default metrics**: one `Objective → Metric` mapping function.
5. **Transforms are decoupled**: evaluation/inference uses `OutputTransform`, not objective-specific transform methods.
