# Backlog 09: API Polish and Cleanup

**RFC**: None (follow-up work from Backlog 08)  
**Priority**: Medium  
**Status**: Ready for Implementation

---

## Overview

Follow-up refinements from stakeholder feedback during API Layering Redesign (Backlog 08).
These are design improvements and cleanup items that don't block the core refactor.

**Scope**: 9 stories, ~2 weeks estimated effort.

---

## Story 9.1: Performance Investigation

Investigate potential performance regression reported before the API refactor.

**Reference**: See `docs/benchmarks/2025-12-16-67d2d89-gblinear-optimization.md`

**Tasks**:

- [ ] 9.1.1: Run benchmark suite on current commit
- [ ] 9.1.2: Compare with baseline from Backlog 08, Story 0.1
- [ ] 9.1.3: Identify any regressions (>5% slowdown)
- [ ] 9.1.4: If regressions found, profile and fix

**Definition of Done**:

- Performance is within 5% of baseline or documented reason for regression

---

## Story 9.2: Convenience Method Cleanup

Review and potentially remove `compute_gradients_buffer` if single-use.

**Tasks**:

- [ ] 9.2.1: Audit usages of `compute_gradients_buffer`
- [ ] 9.2.2: If single use, inline it
- [ ] 9.2.3: If multiple uses, keep but document

**Definition of Done**:

- Convenience methods justified or removed

---

## Story 9.3: MetricKind Revision

Review whether `MetricKind` adds value or hides nuance.

**Tasks**:

- [ ] 9.3.1: Audit where MetricKind is used
- [ ] 9.3.2: Determine if it can be removed
- [ ] 9.3.3: Replace with explicit Metric construction if beneficial

**Definition of Done**:

- MetricKind removed or documented rationale for keeping

---

## Story 9.4: TreeParams Builder Pattern

Add bon builder to TreeParams for consistency.

**Tasks**:

- [ ] 9.4.1: Add `#[bon]` to TreeParams
- [ ] 9.4.2: Keep convenience constructors as shortcuts
- [ ] 9.4.3: Document both patterns in rustdoc

**Definition of Done**:

- `TreeParams::builder().max_depth(8).build()` works
- Convenience constructors still work

---

## Story 9.5: Error Type Structuring

Review error types and consider newtype wrappers.

**Tasks**:

- [ ] 9.5.1: Audit ConfigError variants
- [ ] 9.5.2: Determine if nesting improves usability
- [ ] 9.5.3: Implement if beneficial (e.g., `ConfigError::Regularization(RegularizationError)`)

**Definition of Done**:

- Error types documented and justified

---

## Story 9.6: Config Serialization

Include GBDTConfig in model serialization.

**Tasks**:

- [ ] 9.6.1: Add config field to serialized model format
- [ ] 9.6.2: Implement serialization for GBDTConfig
- [ ] 9.6.3: Update load to restore config
- [ ] 9.6.4: No backwards compatibility needed (no users)

**Definition of Done**:

- Models saved with config
- Loaded models have `config()` returning `Some`

---

## Story 9.7: Params/Config Unification

Reduce duplication between GBDTParams and GBDTConfig.

**Tasks**:

- [ ] 9.7.1: Audit overlap between GBDTParams and GBDTConfig
- [ ] 9.7.2: Consider making GBDTConfig produce GBDTParams directly
- [ ] 9.7.3: Expose all power-user params

**Definition of Done**:

- Single source of truth for parameters
- All params accessible to power users

---

## Story 9.8: TaskKind Inference from Objective

Fix TaskKind to be inferred from objective, not n_outputs.

**Context**: Multi-output regression (e.g., PinballLoss with multiple quantiles) 
is incorrectly classified as MulticlassClassification.

**Tasks**:

- [ ] 9.8.1: Add `task_kind()` method to ObjectiveFn (already exists?)
- [ ] 9.8.2: Use objective's task_kind in model creation
- [ ] 9.8.3: Remove n_outputs-based inference

**Definition of Done**:

- PinballLoss with 3 quantiles → TaskKind::Regression (not Multiclass)

---

## Story 9.9: TaskKind Design Discussion

Evaluate whether TaskKind is needed or can be simplified.

**Tasks**:

- [ ] 9.9.1: Document current TaskKind usage
- [ ] 9.9.2: Identify assumptions that cause issues
- [ ] 9.9.3: Propose simplification or removal

**Definition of Done**:

- Design decision documented in RFC or research doc
