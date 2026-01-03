# Backlog: OutputTransform Decoupling (RFC-0005 v3)

**Epic**: Decouple output transformations from objectives for schema v3  
**RFC**: RFC-0005 (Objectives and Metrics)  
**Created**: 2026-01-03  
**Refined**: 2026-01-03 (4 rounds)  
**Status**: Ready

## Prerequisites

- RFC-0005: Objectives and Metrics (OutputTransform section)
- RFC-0016: Model Serialization (schema versioning)

## Overview

Replace `objective: ObjectiveSchema` with `output_transform: OutputTransformSchema` in
persisted models. This enables custom objectives to serialize correctly and removes
training-only parameters from model files.

---

## Story 1: Add OutputTransform Runtime Type

Add the `OutputTransform` enum to the model module for inference-time transformations.

### Tasks (Story 1)

- [ ] 1.1: Create `model/transform.rs` with `OutputTransform` enum (Identity, Sigmoid, Softmax)
- [ ] 1.2: Implement `transform_inplace(&self, predictions: ArrayViewMut2<f32>)`
  - Sigmoid: clamp input to [-88, 88] to avoid exp overflow
  - Softmax: use max-subtraction for numerical stability
- [ ] 1.3: Implement `prediction_kind(&self) -> PredictionKind` helper
- [ ] 1.4: Add unit tests for sigmoid (compare to numpy, verify output ∈ (0,1))
- [ ] 1.5: Add unit tests for softmax (compare to numpy, verify sum=1)
- [ ] 1.6: Add edge case tests (extreme values ±100, single class, NaN input)
- [ ] 1.7: Add benchmark comparing to current `ObjectiveFn::transform_predictions_inplace`
- [ ] 1.8: Verify numerical equivalence with existing objective transforms
- [ ] 1.9: Export from `model/mod.rs`

### Definition of Done (Story 1)

- [ ] `OutputTransform` enum with 3 variants
- [ ] `transform_inplace` applies correct math for each variant
- [ ] `prediction_kind` returns appropriate `PredictionKind`
- [ ] Tests pass for normal and edge cases
- [ ] Exported from crate root

---

## Story 2: Add output_transform() to ObjectiveFn Trait

Extend the `ObjectiveFn` trait with a method to return the output transform.

### Tasks (Story 2)

- [ ] 2.1: Add `fn output_transform(&self) -> OutputTransform` to `ObjectiveFn` trait
- [ ] 2.2: Implement for `SquaredLoss` → `Identity`
- [ ] 2.3: Implement for `AbsoluteLoss` → `Identity`
- [ ] 2.4: Implement for `PinballLoss` → `Identity`
- [ ] 2.5: Implement for `PseudoHuberLoss` → `Identity`
- [ ] 2.6: Implement for `PoissonLoss` → `Identity`
- [ ] 2.7: Implement for `LogisticLoss` → `Sigmoid`
- [ ] 2.8: Implement for `HingeLoss` → `Identity`
- [ ] 2.9: Implement for `SoftmaxLoss` → `Softmax`
- [ ] 2.10: Implement for `LambdaRankLoss` → `Identity`
- [ ] 2.11: Implement for `Objective` enum (delegate to inner)
- [ ] 2.12: Add tests verifying each objective returns correct transform

### Definition of Done (Story 2)

- [ ] All built-in objectives implement `output_transform()`
- [ ] Custom objectives can override with their transform
- [ ] Tests verify correct mapping for each objective

---

## Story 3: Add objective_name to ModelMeta

Store the objective name in model metadata for debugging/reproducibility.

### Tasks (Story 3)

- [ ] 3.1: Add `objective_name: Option<String>` field to `ModelMeta`
- [ ] 3.2: Update `ModelMeta` constructors to accept objective name
- [ ] 3.3: Add `objective_name: Option<String>` to `ModelMetaSchema`
- [ ] 3.4: Update schema conversion (From impls)
- [ ] 3.5: Update Python schema mirror (`ModelMetaSchema` in pydantic)
- [ ] 3.6: Add test for round-trip with objective name

### Definition of Done (Story 3)

- [ ] `ModelMeta.objective_name` stores objective name
- [ ] Schema persists objective name
- [ ] Python schema includes objective_name field
- [ ] Round-trip test passes

---

## Story 4: Update Models to Use OutputTransform

Replace `objective` field with `output_transform` in both `GBDTModel` and `GBLinearModel`.

### Tasks (Story 4)

#### Preparation

- [ ] 4.1: Audit codebase for `model.objective()` usages (`grep -r "\.objective()"`)
- [ ] 4.2: Plan migration for each usage site

#### GBDTModel

- [ ] 4.3: Replace `objective: Objective` with `output_transform: OutputTransform` in `GBDTModel`
- [ ] 4.4: Update `GBDTModel::train()` to extract transform from objective
- [ ] 4.5: Update `GBDTModel::predict()` to use `output_transform.transform_inplace()`
- [ ] 4.6: Add `output_transform(&self) -> OutputTransform` getter
- [ ] 4.7: Remove `objective(&self)` getter

#### GBLinearModel

- [ ] 4.8: Replace `objective: Objective` with `output_transform: OutputTransform` in `GBLinearModel`
- [ ] 4.9: Update `GBLinearModel::train()` to extract transform from objective
- [ ] 4.10: Update `GBLinearModel::predict()` to use `output_transform.transform_inplace()`
- [ ] 4.11: Add `output_transform(&self) -> OutputTransform` getter
- [ ] 4.12: Remove `objective(&self)` getter

#### Testing

- [ ] 4.13: Update all tests using `model.objective()`
- [ ] 4.14: Add integration test: train → predict gives correct transformed output
- [ ] 4.15: Add custom objective round-trip test:
  - Create custom objective with Sigmoid transform
  - Train model
  - Save to disk
  - Load from disk (new model instance)
  - Predict and verify output matches expected sigmoid-transformed values
- [ ] 4.16: Verify no additional allocations in predict path (transform reuses buffer)

### Definition of Done (Story 4)

- [ ] Both models store `OutputTransform` instead of `Objective`
- [ ] `predict()` applies correct transformation for all objective types
- [ ] No compilation errors from removed `objective()` method
- [ ] Custom objective serialization works correctly (train → save → load → predict)
- [ ] All tests pass

---

## Story 5: Update Schema to v3 with OutputTransformSchema

Replace `ObjectiveSchema` with `OutputTransformSchema` in persistence layer.

### Tasks (Story 5)

- [ ] 5.1: Grep for `ObjectiveSchema` usages to ensure safe deletion
- [ ] 5.2: Add `OutputTransformSchema` enum to `persist/schema.rs`
- [ ] 5.3: Replace `objective: ObjectiveSchema` with `output_transform: OutputTransformSchema` in model schemas
- [ ] 5.4: Bump `SCHEMA_VERSION` to 3
- [ ] 5.5: Update `From<OutputTransform>` for schema conversion
- [ ] 5.6: Update `From<OutputTransformSchema>` for loading
- [ ] 5.7: Delete `ObjectiveSchema` enum (after confirming no references)
- [ ] 5.8: Add schema round-trip tests for each transform type:
  - Identity: save → load → predict gives raw values
  - Sigmoid: save → load → predict gives probabilities
  - Softmax: save → load → predict gives probability distribution
- [ ] 5.9: Add test that v2 models fail with error: "Schema version 2 is not supported. Please re-export your model with the current library version."

### Definition of Done (Story 5)

- [ ] Schema v3 uses `OutputTransformSchema`
- [ ] `ObjectiveSchema` removed
- [ ] Round-trip tests pass for each transform type
- [ ] v2 model loading fails with clear error message (task 5.9)

---

## Story 6: Update Python Bindings

Update PyO3 bindings for the new model structure.

### Tasks (Story 6)

- [ ] 6.1: Add `OutputTransform` to Python bindings (as enum with values: Identity, Sigmoid, Softmax)
- [ ] 6.2: Update `GBDTModel.output_transform` property (returns enum)
- [ ] 6.3: Update `GBLinearModel.output_transform` property (returns enum)
- [ ] 6.4: Remove `model.objective` property
- [ ] 6.5: Update Python schema (`OutputTransformSchema` in pydantic)
- [ ] 6.6: Update schema conversion in `convert.py`
- [ ] 6.7: Update Python tests
- [ ] 6.8: Update sklearn wrappers if affected

### Definition of Done (Story 6)

- [ ] Python models expose `output_transform` property
- [ ] Python schema includes `OutputTransformSchema`
- [ ] All Python tests pass

---

## Story 7: Documentation and Examples

Update documentation and examples for the new API.

### Tasks (Story 7)

- [ ] 7.1: Update rustdoc on `OutputTransform` with usage examples
- [ ] 7.2: Update `GBDTModel` rustdoc to mention `output_transform`
- [ ] 7.3: Update Python docstrings
- [ ] 7.4: Update example notebooks if any use `model.objective()`
- [ ] 7.5: Update RFC-0016 (Model Serialization) to document v3 schema format
- [ ] 7.6: Regenerate Python stubs

### Definition of Done (Story 7)

- [ ] Rustdoc complete for new types
- [ ] Python docstrings updated
- [ ] Examples work with new API

---

## Story 8: Stakeholder Feedback and Review

Conduct review and gather feedback after implementation.

### Tasks (Story 8)

- [ ] 8.1: Check `workdir/tmp/stakeholder_feedback.md` for related feedback
- [ ] 8.2: Prepare demo showing:
  - Smaller model files (compare v2 vs v3 file sizes)
  - Custom objective serialization works end-to-end
  - No performance regression in predict path
- [ ] 8.3: Document review in chat and in `workdir/tmp/development_review_<timestamp>.md`
- [ ] 8.4: Create follow-up stories for any deferred work

### Definition of Done (Story 8)

- [ ] Stakeholder feedback reviewed
- [ ] Demo conducted with acceptance criteria met
- [ ] Review documented

---

## Story 9: Retrospective

Conduct retrospective after all stories complete.

### Tasks (Story 9)

- [ ] 9.1: Gather team reflections
- [ ] 9.2: Document in `workdir/tmp/retrospective.md`
- [ ] 9.3: Create backlog items for process improvements

### Definition of Done (Story 9)

- [ ] Retrospective documented
- [ ] Action items captured
- [ ] At least one high-priority improvement turned into backlog work (if warranted)

---

## Dependencies

```text
Story 1 (OutputTransform type) ─┐
Story 2 (ObjectiveFn trait) ────┼──→ Story 4 (Both Models)
Story 3 (ModelMeta) ────────────┘           ↓
                                     Story 5 (Schema v3)
                                            ↓
                                     Story 6 (Python bindings)
                                            ↓
                                     Story 7 (Documentation)
                                            ↓
                                     Story 8 (Review) → Story 9 (Retro)
```

Stories 1, 2, 3 can be done in parallel. Story 4 requires all three to complete.

---

## Risks and Mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Breaking change for users | Library has no external users yet (per CONTRIBUTING.md) |
| v2 models become unloadable | Document in release notes; provide conversion script if needed |
| Custom objectives in Python | Ensure `output_transform()` is exposed via trait object |
| Sigmoid overflow | Clamp input values to [-88, 88] before exp() |

---

## Estimation

| Story | Size | Notes |
| ----- | ---- | ----- |
| 1 | Small | New type, straightforward |
| 2 | Small | Trait method + 10 impls |
| 3 | Small | Field addition |
| 4 | Medium | Both models + test updates |
| 5 | Medium | Schema changes + deletion |
| 6 | Medium | Python binding updates |
| 7 | Small | Documentation |
| 8-9 | Small | Process tasks |

**Total**: ~3-4 days of focused work
