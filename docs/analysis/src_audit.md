# src/ and tests/ critical audit

Generated: 2025-12-13 (updated post-RFC-0028)

This is a **critical audit** of the crate layout and the test suite: what's valuable, what's superficial, what's missing, and where the current test strategy is robust vs fragile.

Scope:

- This audit is for the `booste-rs` crate.
- It covers both `src/` (unit tests and module boundaries) and `tests/` (integration tests).

## Executive summary

What's strong:

- The split between *representation* (`src/repr`) and *execution* (`src/inference`) is the right architectural move.
- **RFC 0028 implemented**: Objectives now own prediction transforms and declare semantic metadata (`TaskKind`, `TargetSchema`, `default_metric`). This eliminates ad-hoc `sigmoid_inplace` calls in tests and downstream code.
- Compat integration tests with fixture models (`tests/test-cases/*`) provide high-value regression protection against semantic drift (especially around missing values and categorical handling).
- Metrics now declare `expected_prediction_kind()`, enabling trainers to validate metric/transform compatibility.
- GBLinear evaluation applies objective transforms automatically based on metric expectations.

What's weak / risky:

- Some unit tests are essentially "does Default stay the same" or "n_trees equals X" and don't validate semantics.
- Some training tests check only "reasonable-ish" outcomes (thresholds like RMSE < 1.0, accuracy > 0.4). These can become flaky across changes in optimization, SIMD, or platform math.
- Training-set metric reporting was disabled in GBLinear evaluation (training `ColMatrix` is not passed to `evaluate_round`). Consider restoring if needed.

What's missing (most important gaps):

- Negative tests for malformed models / malformed test-case data (ensure errors are returned, not panics).
- Full GBDT trainer integration with objective transforms (GBDT trainer currently doesn't call `transform_prediction_inplace` during eval; GBLinear does).
- Multi-output metric validation: verify metrics handle multiclass row-major properly.

## Status update (as of 2025-12-13)

This audit was originally written as "gaps to address". Since then, several recommendations have been implemented.

Applied:

- **RFC 0028 objective-owned transforms**: Objectives now implement `transform_prediction_inplace` and `transform_prediction`. Classification objectives apply sigmoid/softmax; Poisson applies exp; regression/hinge/ranking stay identity/margin.
- **Objective metadata**: `TaskKind`, `TargetSchema`, and `default_metric()` are declared on every objective. `ObjectiveFunction` enum delegates to the underlying implementations.
- **Metric introspection**: Metrics implement `expected_prediction_kind()` so trainers/tests can verify prediction-space alignment.
- **GBLinear evaluation transform plumbing**: `GBLinearTrainer::evaluate_round` converts column-major predictions to row-major `PredictionOutput`, applies `objective.transform_prediction_inplace` when the metric expects non-Margin predictions, then computes the metric.
- **MarginAccuracy** is now public and used for margin-space binary classification (HingeLoss).
- Traversal semantics spec coverage was strengthened (threshold equality, missing/default direction, categorical membership/unknown category, categorical below-unroll regression).
- SIMD parity is enforced when `feature = simd` is enabled.
- Conversion invariants are now checked via `Tree::validate()` / `Forest::validate()`.
- Integration tests were regrouped to reduce `tests/` root clutter.
- Tests no longer manually call `sigmoid_inplace`; they use objective transforms or margin metrics.

Still open / partially addressed:

- Negative tests for malformed full models are still light.
- GBDT trainer doesn't yet wire objective transforms into evaluation (it uses objectives only for gradient computation).
- Training-set metric logging in GBLinear was removed because training data isn't passed to `evaluate_round`. Could be restored by passing the train ColMatrix.

## Current layout (high-level)

```text
src/
├── compat/         # Loading/parsing models (LightGBM/XGBoost), converting into repr
├── data/           # Matrices, datasets, binning
├── inference/      # Prediction APIs + traversal implementations
│   └── common/     # PredictionOutput, PredictionKind, Predictions, sigmoid/softmax
├── repr/           # Canonical internal representations (GBDT, GBLinear)
├── testing/        # Test utilities (macros, assert helpers, fixture helpers)
├── training/       # Training implementations (GBDT + GBLinear)
│   ├── metrics/    # Metric trait + implementations (Rmse, LogLoss, MarginAccuracy, etc.)
│   └── objectives/ # Objective trait + implementations (RFC 0028 metadata + transforms)
├── lib.rs
├── main.rs
└── utils.rs
```

Notable changes vs older audits:

- **`src/inference/common/predictions.rs`**: Adds `PredictionKind` and `Predictions` wrapper (semantic output type).
- **Objective trait extensions**: `task_kind()`, `target_schema()`, `default_metric()`, `transform_prediction_inplace()`, `transform_prediction()`.
- `src/training/metrics/mod.rs`: Exports `MetricKind` and `MarginAccuracy`.

### Detailed file tree (trimmed to the meaningful seams)

```text
src/
├── compat/
│   ├── lightgbm/ (convert.rs, text.rs, mod.rs)
│   ├── xgboost/  (convert.rs, json.rs, mod.rs)
│   └── mod.rs
├── data/
│   ├── binned/ (bin_mapper.rs, builder.rs, dataset.rs, group.rs, storage.rs, mod.rs)
│   ├── dataset.rs
│   ├── matrix.rs
│   ├── traits.rs
│   └── mod.rs
├── inference/
│   ├── common/ (mod.rs, output.rs, predictions.rs)  # <-- RFC 0028 additions
│   ├── gbdt/   (predictor.rs, traversal.rs, unrolled.rs, simd.rs, mod.rs)
│   ├── gblinear/ (model.rs, mod.rs)
│   └── mod.rs
├── repr/
│   ├── gbdt/ (categories.rs, forest.rs, leaf.rs, node.rs, tree.rs, mod.rs)
│   ├── gblinear/
│   └── mod.rs
├── testing/ (mod.rs, slices.rs, stats.rs, tree.rs, cases.rs)
└── training/
    ├── gbdt/ (categorical.rs, expansion.rs, grower.rs, optimization.rs, partition.rs, trainer.rs, mod.rs)
    ├── gblinear/ (selector.rs, trainer.rs, updater.rs, mod.rs)
    ├── metrics/ (classification.rs, regression.rs, mod.rs)  # <-- MetricKind, MarginAccuracy
    ├── objectives/ (classification.rs, regression.rs, mod.rs)  # <-- TaskKind, TargetSchema, transforms
    ├── sampling/ (row.rs, column.rs, mod.rs)
    ├── callback.rs
    ├── eval.rs
    ├── gradients.rs
    ├── logger.rs
    └── mod.rs
```

## RFC 0028 implementation details

### Objective metadata

Every objective now declares:

| Method | Purpose |
|--------|---------|
| `task_kind()` | Returns `TaskKind::{Regression, BinaryClassification, MulticlassClassification, Ranking}` |
| `target_schema()` | Returns `TargetSchema::{Continuous, Binary01, BinarySigned, MulticlassIndex, CountNonNegative}` |
| `default_metric()` | Returns `MetricKind` (e.g., `Rmse`, `LogLoss`, `MulticlassLogLoss`) |

### Objective transforms

| Method | Behavior |
|--------|----------|
| `transform_prediction_inplace(&self, raw: &mut PredictionOutput) -> PredictionKind` | Applies in-place transform (sigmoid, softmax, exp, or identity) and returns the resulting semantic kind. |
| `transform_prediction(&self, raw: PredictionOutput) -> Predictions` | Consumes the output, calls `transform_prediction_inplace`, and wraps in a semantic `Predictions` struct. |

Objective-specific transforms:

| Objective | Transform | Output `PredictionKind` |
|-----------|-----------|-------------------------|
| `SquaredLoss`, `AbsoluteLoss`, `PinballLoss`, `PseudoHuberLoss` | identity | `Value` |
| `LogisticLoss` | sigmoid | `Probability` |
| `SoftmaxLoss` | softmax (per-row) | `Probability` |
| `HingeLoss` | identity | `Margin` |
| `PoissonLoss` | exp | `Value` |
| `LambdaRankLoss` | identity | `RankScore` |

### Metric introspection

Metrics declare:

```rust
fn expected_prediction_kind(&self) -> PredictionKind;
```

Examples:

| Metric | `expected_prediction_kind()` |
|--------|------------------------------|
| `Rmse`, `Mae`, `Mape`, `QuantileMetric` | `Value` |
| `LogLoss`, `Accuracy` | `Probability` |
| `MarginAccuracy` | `Margin` |
| `MulticlassLogLoss`, `MulticlassAccuracy` | `Probability` |
| `Auc` | `Probability` |

### Trainer integration (GBLinear)

`GBLinearTrainer::evaluate_round`:

1. Computes raw predictions in column-major buffer.
2. Converts column-major → row-major `PredictionOutput`.
3. If `metric.expected_prediction_kind() != Margin`, calls `objective.transform_prediction_inplace`.
4. Computes metric on the (possibly transformed) row-major slice.

This ensures metrics always see the expected prediction space without manual intervention.

## Test suite inventory

### Integration tests (`tests/`)

```text
tests/
├── compat.rs                      # Feature-gated compat suite entrypoint
├── compat/
│   ├── lightgbm.rs
│   ├── xgboost.rs
│   └── test_data.rs
├── training.rs                    # Training suite entrypoint
├── training/
│   ├── gbdt.rs
│   └── gblinear/
│       ├── classification.rs
│       ├── loss_functions.rs      # Uses MarginAccuracy and objective transforms
│       ├── quantile.rs
│       ├── regression.rs
│       ├── selectors.rs
│       └── mod.rs
└── test-cases/                    # Golden fixtures from Python (LightGBM/XGBoost)
```

### Unit tests (`src/`)

Extensive `#[cfg(test)]` unit tests throughout `src/`, especially in:

- `src/inference/gbdt/unrolled.rs` (layout/traversal behavior)
- `src/compat/lightgbm/text.rs` and `src/compat/xgboost/json.rs` (parser behavior)
- `src/data/binned/*` (builder/storage invariants)
- `src/training/*` (gain, split finding, histogram ops, etc.)
- `src/training/objectives/*.rs` (gradient/base-score unit tests)

## Are we doing the right tests?

### Meaningful tests (high value)

- **Golden compat tests**: Compare predictions against Python-generated expected outputs. Catch semantic regressions.
- **Algorithmic unit tests**: Validate behavior (exit routing, missing-value handling, block processing).
- **Training behavior tests with external references**: GBLinear tests comparing weights/predictions against XGBoost are valuable compatibility anchors.
- **RFC 0028 transform tests**: Integration tests now use objective transforms or margin metrics, validating the new API.

### Trivial / low-value tests (still sometimes OK)

- "Default equals X" tests: Lock down public API defaults but don't validate correctness.
- "Shape-only" tests: Catch almost no real regressions beyond catastrophic failures.

## Missing critical tests (most impactful additions)

### 1) GBDT trainer transform integration

GBDT trainer should also apply objective transforms during evaluation. Currently it only uses objectives for gradients.

### 2) Negative tests (errors, not panics)

Add malformed fixture tests that verify structured errors for:

- missing required JSON fields
- invalid decision type codes
- inconsistent array lengths
- invalid feature indices

### 3) Multi-output metric tests

Verify metrics correctly handle multiclass predictions in row-major layout (especially after softmax transform).

### 4) Training-set metric restoration

Consider restoring training-set metric logging in GBLinear by passing the train `ColMatrix` to `evaluate_round`.

## Robustness vs fragility

### Robust patterns

- Fixture-driven golden tests are robust against refactors.
- Deterministic unit tests that build tiny trees and assert exact leaf routing.
- RFC 0028 transform tests that use objective APIs instead of manual sigmoid calls.

### Fragile patterns

- Tests that assert loose training metric thresholds can become flaky.
- Tests comparing floats without a defined tolerance strategy can break on platform differences.

Mitigations:

- Prefer comparing against a reference implementation over "RMSE < X".
- For training, prefer monotonic invariants (loss decreases) over absolute thresholds.

## Summary of RFC 0028 changes

| Component | Before | After |
|-----------|--------|-------|
| Objectives | Training-only (gradients, base score) | Full metadata + prediction transforms |
| Metrics | Implicit prediction-space assumptions | Explicit `expected_prediction_kind()` |
| GBLinear eval | Raw column-major predictions passed to metrics | Row-major + automatic transform based on metric expectations |
| Tests | Manual `sigmoid_inplace` calls | Use `objective.transform_prediction_inplace` or `MarginAccuracy` |
| Public API | `Accuracy` only | `Accuracy` + `MarginAccuracy` + `MetricKind` + `TaskKind` + `TargetSchema` |

## Machine-readable inventory

See `docs/analysis/src_items.json` for the list of Rust source files under `src/`.
