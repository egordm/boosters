# src/ and tests/ critical audit

Generated: 2025-12-13

This is a **critical audit** of the crate layout and the test suite: what’s valuable, what’s superficial, what’s missing, and where the current test strategy is robust vs fragile.

Scope:

- This audit is for the `booste-rs` crate.
- It covers both `src/` (unit tests and module boundaries) and `tests/` (integration tests).

## Executive summary

What’s strong:

- The split between *representation* (`src/repr`) and *execution* (`src/inference`) is the right architectural move.
- Compat integration tests with fixture models (`tests/test-cases/*`) provide high-value regression protection against semantic drift (especially around missing values and categorical handling).
- Several algorithmic “core” units (e.g. unrolled traversal layout) have direct behavioral tests, not just constructor checks.

What’s weak / risky:

- Some unit tests are essentially “does Default stay the same” or “n_trees equals X” and don’t validate semantics.
- Some training tests check only “reasonable-ish” outcomes (thresholds like RMSE < 1.0, accuracy > 0.4). These can become flaky across changes in optimization, SIMD, or platform math.

What’s missing (most important gaps):

- Tight, explicit spec tests for traversal semantics: NaNs, threshold equality, default direction, categorical split membership, and out-of-range category/bin behavior.
- Conversion invariants (LightGBM/XGBoost → canonical repr): domain consistency (bin/category index semantics), group/tree indexing invariants, and “shape” validation.
- Negative tests for malformed models / malformed test-case data (ensure errors are returned, not panics).

## Status update (as of 2025-12-13)

This audit was originally written as “gaps to address”. Since then, several recommendations have been implemented.

Applied:

- Traversal semantics spec coverage was strengthened (threshold equality, missing/default direction, categorical membership/unknown category, categorical below-unroll regression).
- SIMD parity is enforced when `feature = simd` is enabled (SIMD traversal output matches standard/unrolled for representative cases).
- Conversion invariants are now checked via `Tree::validate()` / `Forest::validate()` and are called from compat conversion/integration tests.
- Integration tests were regrouped to reduce `tests/` root clutter and avoid duplicated compilation.

Still open / partially addressed:

- Negative tests for malformed full models are still light (there are some negative parser tests, but the “bad model returns structured error, never panics” surface can be expanded).
- Some GBLinear training integration tests still assert “reasonable-ish” thresholds and print debug output; they’re useful anchors but can be somewhat noisy/flaky.

## Current layout (high-level)

```text
src/
├── compat/         # Loading/parsing models (LightGBM/XGBoost), converting into repr
├── data/           # Matrices, datasets, binning
├── inference/      # Prediction APIs + traversal implementations
├── repr/           # Canonical internal representations (GBDT, GBLinear)
├── testing/        # Test utilities (macros, assert helpers, fixture helpers)
├── training/       # Training implementations (GBDT + GBLinear)
├── lib.rs
├── main.rs
└── utils.rs
```

Notable changes vs older audits:

- Canonical GBDT representation is now under `src/repr/gbdt/*`.
- `src/testing.rs` is gone; test utilities live under `src/testing/`.
- `src/inference/gbdt/` now focuses on traversal/predictor concerns (repr lives in `src/repr/gbdt`).

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
│   ├── common/ (mod.rs, output.rs)
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
    ├── metrics/ (classification.rs, regression.rs, mod.rs)
    ├── objectives/ (classification.rs, regression.rs, mod.rs)
    ├── sampling/ (row.rs, column.rs, mod.rs)
    ├── callback.rs
    ├── eval.rs
    ├── gradients.rs
    ├── logger.rs
    └── mod.rs
```

## Canonical representation vs execution

This separation is the most important architectural boundary:

- `repr::*` should define *what the model is* (trees/forests, nodes, categorical storage, leaf values).
- `inference::*` should define *how the model is executed* (predictors, traversal strategies, SIMD/unrolled kernels).

Audit guidance: keep traversal-specific data layouts in `inference` (e.g. unrolled layouts), but keep “semantic model truth” in `repr`.

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
│       ├── loss_functions.rs
│       ├── quantile.rs
│       ├── regression.rs
│       ├── selectors.rs
│       └── mod.rs
└── test-cases/                    # Golden fixtures from Python (LightGBM/XGBoost)
```

### Unit tests (`src/`)

There are extensive `#[cfg(test)]` unit tests scattered throughout `src/`, especially in:

- `src/inference/gbdt/unrolled.rs` (layout/traversal behavior)
- `src/compat/lightgbm/text.rs` and `src/compat/xgboost/json.rs` (parser behavior)
- `src/data/binned/*` (builder/storage invariants)
- `src/training/*` (gain, split finding, histogram ops, etc.)

## Are we doing the right tests?

### Meaningful tests (high value)

- **Golden compat tests**:
  - `tests/inference_lightgbm.rs` and `tests/inference_xgboost.rs` compare predictions against Python-generated expected outputs.
  - These are “semantic regression” tests: if you break missing-value semantics, default-direction routing, or categorical handling, they should catch it.
- **Algorithmic unit tests**:
  - Example: unrolled traversal layout tests validate behavior (exit routing, missing-value behavior, block processing).
  - These are good because they test *outcomes*, not just sizes.
- **Training behavior tests with external references**:
  - GBLinear tests that compare weights/predictions against XGBoost are valuable “compatibility anchors”.

### Trivial / low-value tests (still sometimes OK, but watch the ratio)

- “Default equals X” tests (e.g., `ScalarLeaf::default()` is zero, basic `Default` params checks).
  - These primarily lock down public API defaults. That’s acceptable, but they do not validate correctness.
- “Shape-only” tests (e.g., `forest.n_trees() == N` after training) without checking any semantic metric.
  - These catch almost no real regressions besides catastrophic failures.

Recommendation: keep a few “defaults are stable” checks, but prioritize semantic tests around traversal + conversion.

## Missing critical tests (most impactful additions)

### 1) Traversal semantics spec tests

Add tests that explicitly define the semantics of:

- Numeric splits at threshold equality (`x == threshold`): go left or right?
- `NaN` routing: what does `default_left` mean in numeric and categorical contexts?
- Categorical membership: which side is “match”, and what happens for unknown category ids?

These tests should be small, deterministic, and ideally share a single “reference traversal” helper.

### 2) SIMD parity tests

If `feature = simd` is enabled, add tests that run the *same model + inputs* through:

- scalar traversal
- SIMD traversal
- unrolled traversal (if used)

and assert identical (or tightly bounded) outputs.

### 3) Conversion invariants tests

Compat conversion (`compat::{lightgbm,xgboost}::* -> repr::*`) needs invariant checks beyond “it parses”:

- node counts match declared sizes
- child indices are in-bounds
- all leaves reachable
- group count vs tree-per-iteration consistent (multiclass)
- categorical domain is canonicalized (bin/category index semantics match internal expectations)

### 4) Negative tests (errors, not panics)

Add malformed fixture tests that verify the library returns structured errors:

- missing required JSON fields
- invalid decision type codes
- inconsistent array lengths
- invalid feature indices

Goal: avoid panics for user-supplied models.

## Robustness vs fragility

### Robust patterns

- Fixture-driven golden tests are robust against refactors, and focus on public behavior.
- Deterministic unit tests that build tiny trees and assert exact leaf routing are robust.

### Fragile patterns

- Tests that assert a loose training metric threshold (e.g., “accuracy > 0.4”) can become flaky as training hyperparameters or numeric details change.
- Tests that compare floats without clearly defined tolerance strategy (absolute + relative) can break on platform differences.

Mitigations:

- Prefer comparing *against a reference implementation* (e.g., scalar vs SIMD) over “RMSE < X”.
- For training, prefer monotonic/qualitative invariants (loss decreases, predictions improve over base score) over absolute thresholds.
- Keep printing in tests minimal (CI logs stay readable).

## Test structure and grouping

Current grouping is good:

- Integration tests are segregated under `tests/` and feature-gated for compat.
- Larger “domains” are grouped (GBLinear training tests are in a folder module).

Potential improvements:

- Expand malformed-model negative tests (conversion should return structured errors for invalid JSON/text, never panic).
- Consider gradually reducing “hard thresholds” in training integration tests by preferring parity checks (e.g. compare against a stable reference) where possible.

## Machine-readable inventory

See `docs/analysis/src_items.json` for the list of Rust source files under `src/`.
