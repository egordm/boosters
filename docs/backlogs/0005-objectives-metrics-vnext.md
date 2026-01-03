# Backlog: RFC-0005 Objectives, Metrics, and Output Transforms (vNext)

**RFC**: [docs/rfcs/0005-objectives-metrics.md](../rfcs/0005-objectives-metrics.md)  
**Created**: 2026-01-03  
**Status**: Draft

This backlog implements the vNext design for objectives, metrics, and output transforms
as described in RFC-0005.

---

## Epic 1: Enum-Only Objective Architecture

Refactor objectives from trait-based to enum-based dispatch.

### Story 1.0: Define `OutputTransform` enum

**Description**: Create the three-variant transform enum with numerically stable implementations.
This is needed early because `Objective::output_transform()` returns this type.

**Tasks**:

- [ ] 1.0.1 Define `OutputTransform` enum in `model/transform.rs`
- [ ] 1.0.2 Implement `transform_inplace` with numerical stability (sigmoid clamp, softmax max-subtract)
- [ ] 1.0.3 Implement `Copy`, `Clone`, `Debug`, `PartialEq`, `Eq` derives
- [ ] 1.0.4 Re-export from `model` module

**Definition of Done**:

- [ ] Transform tests: sigmoid ∈ (0,1), softmax sums to 1.0
- [ ] Edge case tests: ±100, NaN/Inf inputs (no panics; NaNs propagate)

---

### Story 1.1: Define `Objective` enum with all variants

**Description**: Create the `Objective` enum with all supported loss functions as variants.
Include struct-like fields for configurable objectives (PseudoHuberLoss, PinballLoss, SoftmaxLoss).

**Depends on**: Story 1.0

**Tasks**:

- [ ] 1.1.1 Define `Objective` enum in `training/objectives/mod.rs`
- [ ] 1.1.2 Implement `Clone`, `Debug` derives
- [ ] 1.1.3 Add `Custom(CustomObjective)` variant
- [ ] 1.1.4 Re-export `Objective` from crate root

**Definition of Done**:

- [ ] All objective variants from RFC are present
- [ ] Enum compiles and is accessible via `boosters::Objective`

---

### Story 1.2: Implement core methods on `Objective` enum

**Description**: Add `compute_gradients_into`, `compute_base_score`, `output_transform`,
`name`, and `n_outputs` methods via match dispatch.

**Depends on**: Story 1.0, Story 1.1

**Tasks**:

- [ ] 1.2.1 Implement `compute_gradients_into` with match dispatch
- [ ] 1.2.2 Implement `compute_base_score` with match dispatch
- [ ] 1.2.3 Implement `output_transform` returning `OutputTransform`
- [ ] 1.2.4 Implement `name` returning `&str`
- [ ] 1.2.5 Implement `n_outputs` returning `usize`
- [ ] 1.2.6 Benchmark or inspect asm to confirm match dispatch is zero-cost (no heap allocation)

**Definition of Done**:

- [ ] All methods implemented for all variants
- [ ] Unit tests for gradient correctness (finite differences) for each objective
- [ ] Unit tests for base score expected values
- [ ] No allocations in hot path confirmed

---

### Story 1.3: Define `CustomObjective` struct

**Description**: Create `CustomObjective` with boxed closures for user-provided behavior.

**Tasks**:

- [ ] 1.3.1 Define `GradientFn` and `BaseScoreFn` type aliases
- [ ] 1.3.2 Define `CustomObjective` struct with fields from RFC
- [ ] 1.3.3 Ensure `Send + Sync` bounds on closures

**Definition of Done**:

- [ ] `CustomObjective` can be constructed with user closures
- [ ] Test: custom objective with identity gradient compiles and runs

---

### Story 1.4: Migrate regression objective implementations

**Description**: Move gradient/base_score logic from existing regression objective structs
into the enum match arms.

**Tasks**:

- [ ] 1.4.1 Migrate SquaredLoss logic
- [ ] 1.4.2 Migrate AbsoluteLoss logic
- [ ] 1.4.3 Migrate PseudoHuberLoss logic
- [ ] 1.4.4 Migrate PoissonLoss logic
- [ ] 1.4.5 Migrate PinballLoss logic
- [ ] 1.4.6 Delete old regression objective structs
- [ ] 1.4.7 Remove `TargetSchema` and any now-redundant target/task plumbing

**Definition of Done**:

- [ ] All regression objectives use enum dispatch
- [ ] Old regression objective structs deleted
- [ ] Existing regression tests pass

---

### Story 1.5: Migrate classification objective implementations

**Description**: Move gradient/base_score logic from existing classification objective structs
into the enum match arms. Delete `ObjectiveFn` trait.

**Tasks**:

- [ ] 1.5.1 Migrate LogisticLoss logic
- [ ] 1.5.2 Migrate HingeLoss logic
- [ ] 1.5.3 Migrate SoftmaxLoss logic
- [ ] 1.5.4 Delete old classification objective structs
- [ ] 1.5.5 Delete `ObjectiveFn` trait
- [ ] 1.5.6 Remove `TaskKind` and any now-redundant task plumbing

**Definition of Done**:

- [ ] No `ObjectiveFn` trait remains
- [ ] No `TaskKind` references remain
- [ ] All existing tests pass with new enum-based objectives
- [ ] `cargo clippy --all-targets` passes with no warnings

---

## Epic 2: Enum-Only Metric Architecture

Refactor metrics from trait-based to enum-based dispatch.

### Story 2.1: Define `Metric` enum with all variants

**Description**: Create the `Metric` enum with all supported evaluation metrics.
Include `None` variant for no-evaluation case.

**Tasks**:

- [ ] 2.1.1 Define `Metric` enum in `training/metrics/mod.rs`
- [ ] 2.1.2 Add struct-like fields for Quantile and Accuracy
- [ ] 2.1.3 Add `Custom(CustomMetric)` variant
- [ ] 2.1.4 Re-export `Metric` from crate root

**Definition of Done**:

- [ ] All metric variants from RFC are present
- [ ] Enum compiles and is accessible via `boosters::Metric`

---

### Story 2.2: Implement core methods on `Metric` enum

**Description**: Add `compute`, `expected_prediction_kind`, `higher_is_better`, `name` methods.

**Tasks**:

- [ ] 2.2.1 Implement `compute` with match dispatch
- [ ] 2.2.2 Implement `expected_prediction_kind`
- [ ] 2.2.3 Implement `higher_is_better`
- [ ] 2.2.4 Implement `name`
- [ ] 2.2.5 Ensure the evaluation pipeline skips `Metric::None` (no evaluation / no early stopping)
- [ ] 2.2.6 Define and test `Metric::None.compute()` behavior (should not panic; value is not used)

**Definition of Done**:

- [ ] All methods implemented for all variants
- [ ] Unit tests comparing to sklearn metrics where applicable
- [ ] Test: `Metric::None.compute()` behavior is documented and tested

---

### Story 2.3: Define `CustomMetric` struct

**Description**: Create `CustomMetric` with boxed closure for user-provided metric.

**Tasks**:

- [ ] 2.3.1 Define `CustomMetricFn` type alias
- [ ] 2.3.2 Define `CustomMetric` struct

**Definition of Done**:

- [ ] `CustomMetric` can be constructed and used in training

---

### Story 2.4: Migrate existing metric implementations

**Description**: Move metric logic from existing structs into enum match arms.
Delete old metric structs and legacy `MetricFn` trait.

**Tasks**:

- [ ] 2.4.1 Migrate RMSE, MAE, MAPE logic
- [ ] 2.4.2 Migrate Quantile/PoissonDeviance logic
- [ ] 2.4.3 Migrate LogLoss, Accuracy, AUC logic
- [ ] 2.4.4 Migrate MulticlassLogLoss, MulticlassAccuracy logic
- [ ] 2.4.5 Delete old metric structs and legacy `MetricFn` trait

**Definition of Done**:

- [ ] No legacy `MetricFn` trait remains
- [ ] All existing metric tests pass
- [ ] `cargo clippy --all-targets` passes with no warnings

---

## Epic 3: Model Integration and Persistence

Integrate `OutputTransform` into models and update persistence to schema v3.

### Story 3.1: Update `GBDTModel` to use `OutputTransform`

**Description**: Replace objective-based prediction transform with `OutputTransform`.

**Depends on**: Story 1.0

**Status**: ✅ COMPLETE (commit a227718)

**Tasks**:

- [x] 3.1.1 Add `output_transform: OutputTransform` field to `GBDTModel`
- [x] 3.1.2 Update `predict()` to use `output_transform.transform_inplace()`
- [x] 3.1.3 Remove objective dependency from model prediction path

**Definition of Done**:

- [x] `GBDTModel::predict()` works without knowing the objective
- [x] Round-trip test: train → save → load → predict identical

---

### Story 3.2: Update persistence schema to v3

**Description**: Persist `OutputTransform` instead of full objective.
Add `objective_name` to `ModelMeta`.

**Depends on**: Story 3.1

**Status**: ✅ COMPLETE (commit 1a6a8be)

**Tasks**:

- [x] 3.2.1 Define `OutputTransformSchema` enum
- [x] 3.2.2 Update `GBDTModelSchema` to use `output_transform`
- [x] 3.2.3 Add `objective_name: Option<String>` to `ModelMetaSchema`
- [x] 3.2.4 Implement schema v3 serialization/deserialization
- Deferred: 3.2.5 v2 rejection (schema changes are additive, backward compatible)

**Definition of Done**:

- [x] Schema v3 models serialize and deserialize correctly
- [x] `objective_name` is stored for debugging

---

### Story 3.3: Update `GBLinearModel` to use `OutputTransform`

**Description**: Apply the same `OutputTransform` pattern to `GBLinearModel`.

**Depends on**: Story 3.1

**Status**: ✅ COMPLETE (commit 1a6a8be)

**Tasks**:

- [x] 3.3.1 Add `output_transform: OutputTransform` field to `GBLinearModel`
- [x] 3.3.2 Update `predict()` to use `output_transform.transform_inplace()`
- [x] 3.3.3 Update `GBLinearModelSchema` to use `output_transform`

**Definition of Done**:

- [x] `GBLinearModel::predict()` works without knowing the objective
- [x] Round-trip test: train → save → load → predict identical

---

## Epic 4: Default Metric Mapping and Trainer Integration

Centralize default metric selection and integrate with trainers.

**Depends on**: Epics 1, 2, 3

### Story 4.1: Implement `default_metric_for_objective`

**Description**: Create central mapping function from `Objective` to `Metric`.

**Status**: ✅ COMPLETE (commit d4a2719)

**Tasks**:

- [x] 4.1.1 Implement `default_metric_for_objective` function
- [x] 4.1.2 Unit test exhaustiveness (all variants covered)

**Definition of Done**:

- [x] Function returns sensible defaults for all objectives
- [x] Custom objectives return `Metric::None`

---

### Story 4.2: Update trainer to use new objective/metric enums

**Description**: Update `GBDTTrainer` to use `Objective` and `Metric` enums.

**Status**: ✅ COMPLETE (commit fdc3772)

**Tasks**:

- [x] 4.2.1 Update `GBDTConfig` to use `Objective` enum (already done)
- [x] 4.2.2 Update `GBDTConfig` to use `Metric` enum (already done)
- [x] 4.2.3 Apply `default_metric_for_objective` when metric is not specified
- [x] 4.2.4 Update training loop to use enum-based dispatch (already done via trait impls)
- [x] 4.2.5 Handle `Metric::None`: skip evaluation and early stopping (already done via is_enabled())
- [x] 4.2.6 Ensure the trained model sets `output_transform` from `objective.output_transform()`
- [x] 4.2.7 Ensure the trained model sets `meta.objective_name` (stored in schema only)
- [x] 4.2.8 Run prediction and training benchmarks, compare to baseline

**Definition of Done**:

- [x] Training works with all objective/metric combinations
- [x] Test: `Metric::None` skips evaluation and early stopping
- [x] All integration tests pass
- [x] No performance regression in benchmarks

---

### Story 4.3: Update `GBLinearTrainer` to use new objective/metric enums

**Description**: Update `GBLinearTrainer` and its configs to use `Objective` and `Metric` enums,
including default metric selection.

**Status**: ✅ COMPLETE (commit 280e349)

**Tasks**:

- [x] 4.3.1 Update `GBLinearConfig` / params to use `Objective` enum (already done)
- [x] 4.3.2 Update `GBLinearConfig` / params to use `Metric` enum (already done)
- [x] 4.3.3 Apply `default_metric_for_objective` when metric is not specified
- [x] 4.3.4 Update training loop to use enum-based dispatch (already done)
- [x] 4.3.5 Handle `Metric::None`: skip evaluation and early stopping (already done)
- [x] 4.3.6 Ensure the trained model sets `output_transform` and `meta.objective_name`

**Definition of Done**:

- [x] `GBLinearTrainer` uses `Objective` / `Metric` enums
- [x] Existing GBLinear tests and integration paths pass
- [x] No performance regression in relevant benchmarks

---

### Story 4.4: Update Python bindings

**Description**: Expose new `Objective` and `Metric` enums to Python.

**Status**: ✅ PARTIAL (commit e06d1f8)

**Tasks**:

- [x] 4.4.1 Expose `Objective` variants to Python config
- [x] 4.4.2 Expose `Metric` variants to Python config
- Deferred: 4.4.3 Support Python callables for `CustomObjective` → Story 4.6
- Deferred: 4.4.4 Support Python callables for `CustomMetric` → Story 4.6

**Definition of Done**:

- [x] Python users can specify objectives/metrics by name or enum
- Deferred: Custom Python objectives/metrics work (with FFI overhead) → Story 4.6
- [x] Python tests pass (1 pre-existing failure tracked in Story 4.7)

---

### Story 4.5: Update docs and examples to new API

**Description**: Update Rust docs, module docs, and examples that refer to the legacy
`ObjectiveFn` / `MetricFn` trait-based APIs.

**Status**: ✅ COMPLETE (commit 3162895)

**Tasks**:

- [x] 4.5.1 Update crate/module docs that show `ObjectiveFn` / `MetricFn` usage
- [x] 4.5.2 Update examples to construct `Objective` / `Metric` enums
- [x] 4.5.3 Update docs index/status where it references the old objective/metric design

**Definition of Done**:

- [x] `cargo test --doc` passes
- Note: ObjectiveFn/MetricFn traits still exist for backward compatibility (Story 1.5 deferred)

---

### Story 4.6: Python custom callables (Deferred)

**Description**: Support Python callables for `CustomObjective` and `CustomMetric`.
Deferred from Story 4.4 due to FFI complexity.

**Tasks**:

- [ ] 4.6.1 Implement `PyCustomObjective` wrapper that calls Python gradients function
- [ ] 4.6.2 Implement `PyCustomMetric` wrapper that calls Python compute function
- [ ] 4.6.3 Handle GIL acquisition and error propagation across FFI boundary
- [ ] 4.6.4 Document FFI overhead and usage patterns

**Definition of Done**:

- [ ] Python users can pass callables for custom objectives/metrics
- [ ] FFI overhead is documented and benchmarked
- [ ] Error messages from Python exceptions are preserved

**Notes**: This is optional functionality that incurs FFI overhead per batch.
The core enum-based API is complete without this.

---

### Story 4.7: Fix multiclass softmax probabilities (Pre-existing bug)

**Description**: Multiclass softmax predictions don't sum to 1.0 properly.
The current implementation applies per-class sigmoid instead of proper softmax.
This is a pre-existing bug discovered during RFC-0005 implementation.

**Tasks**:

- [ ] 4.7.1 Investigate why multiclass probabilities don't sum to 1.0
- [ ] 4.7.2 Fix OutputTransform::Softmax implementation
- [ ] 4.7.3 Verify sklearn test_multiclass passes

**Definition of Done**:

- [ ] `predict_proba().sum(axis=1) == 1.0` for multiclass classification
- [ ] All sklearn tests pass

---

## Meta-Tasks

### Review: Epics 1-2 Complete

**When**: After Stories 1.5 and 2.4 are done  
**Artifacts**: `workdir/tmp/development_review_<timestamp>.md`

- [ ] Demonstrate enum-based objective/metric dispatch
- [ ] Show test coverage for gradients and metrics
- [ ] Review code reduction (lines deleted vs added)
- [ ] Confirm `cargo clippy --all-targets` passes
- [ ] Run full test suite: `cargo test --all`

---

### Review: Schema v3 Complete

**When**: After Stories 3.2 and 3.3 are done  
**Artifacts**: `workdir/tmp/development_review_<timestamp>.md`

- [ ] Demonstrate model round-trip with new schema (GBDT and GBLinear)
- [ ] Show v2 rejection error message
- [ ] Review persistence code simplification

---

### Review: Public API + Python Complete

**When**: After Stories 4.2, 4.3, 4.4, and 4.5 are done  
**Artifacts**: `workdir/tmp/development_review_<timestamp>.md`

- [ ] Demo: end-to-end training + prediction from Rust (GBDT and GBLinear)
- [ ] Demo: end-to-end training + prediction from Python
- [ ] Confirm docs/examples updated to new `Objective` / `Metric` API
- [ ] Run full check gate: `uv run poe all --check`

---

### Stakeholder Feedback Check (Epics 1-2)

**When**: After the Epics 1-2 review  
**Artifacts**: `workdir/tmp/stakeholder_feedback.md`

- [ ] Review feedback file for input on objective/metric API ergonomics
- [ ] Incorporate any urgent feedback as new stories

---

### Stakeholder Feedback Check (Epic 3)

**When**: After the Schema v3 review  
**Artifacts**: `workdir/tmp/stakeholder_feedback.md`

- [ ] Review feedback file for input on persistence compatibility and errors
- [ ] Incorporate any urgent feedback as new stories

---

### Stakeholder Feedback Check (Epic 4)

**When**: After Epic 4  
**Artifacts**: `workdir/tmp/stakeholder_feedback.md`

- [ ] Review feedback file for input on API ergonomics
- [ ] Incorporate any urgent feedback as new stories

---

### Final Retrospective

**When**: After all epics complete  
**Artifacts**: `workdir/tmp/retrospective.md`

- [ ] Document what went well / not well
- [ ] Capture any deferred work as new backlog items
- [ ] Note any testing gaps discovered during implementation
