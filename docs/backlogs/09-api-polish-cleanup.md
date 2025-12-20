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

## Story 9.1: Performance Investigation ✅

Investigate potential performance regression reported before the API refactor.

**Reference**: See `docs/benchmarks/2025-12-16-67d2d89-gblinear-optimization.md`

**Result**: No significant regression. All benchmarks within 5% tolerance.
See `docs/benchmarks/2025-12-20-c6cc704-post-refactor.md`.

**Tasks**:

- [x] 9.1.1: Run benchmark suite on current commit
- [x] 9.1.2: Compare with baseline from Backlog 08, Story 0.1
- [x] 9.1.3: Identify any regressions (>5% slowdown) → None found
- [x] 9.1.4: If regressions found, profile and fix → N/A

**Definition of Done**:

- Performance is within 5% of baseline ✅

---

## Story 9.2: Convenience Method Cleanup ✅

Review and potentially remove `compute_gradients_buffer` if single-use.

**Result**: Kept the method. It's a good abstraction that:
- Hides buffer internals (`n_samples()`, `n_outputs()`, `pairs_mut()`)
- Documents intent clearly
- Is used in both GBDT and GBLinear trainers (now consistent)

**Tasks**:

- [x] 9.2.1: Audit usages of `compute_gradients_buffer` → 3 usages (2 GBLinear, 1 GBDT now)
- [x] 9.2.2: If single use, inline it → Multiple uses, kept
- [x] 9.2.3: If multiple uses, keep but document → Kept, consistent across both trainers

**Definition of Done**:

- Convenience methods justified ✅

---

## Story 9.3: MetricKind Revision ✅

Review whether `MetricKind` adds value or hides nuance.

**Result**: Removed `MetricKind` entirely. It was dead code:
- Only used by `default_metric()` method on ObjectiveFn
- `default_metric()` was never called externally
- RFC-0020 explicitly said to remove `default_metric()`
- Stakeholder feedback: "MetricKind hides nuances like quantile parameters"

**Tasks**:

- [x] 9.3.1: Audit where MetricKind is used → Only in `default_metric()` and `from_kind()`
- [x] 9.3.2: Determine if it can be removed → Yes, dead code
- [x] 9.3.3: Replace with explicit Metric construction → Removed entirely

**Definition of Done**:

- MetricKind removed ✅

---

## Story 9.4: TreeParams Builder Pattern ✅

Add bon builder to TreeParams for consistency.

**Result**: Added `#[bon]` builder with defaults. Deprecated `with_max_onehot_cats()`
in favor of builder pattern. Convenience constructors (`depth_wise`, `leaf_wise`) kept.

**Tasks**:

- [x] 9.4.1: Add `#[bon]` to TreeParams
- [x] 9.4.2: Keep convenience constructors as shortcuts
- [x] 9.4.3: Document both patterns in rustdoc

**Definition of Done**:

- `TreeParams::builder().growth_strategy(...).build()` works ✅
- Convenience constructors still work ✅

---

## Story 9.5: Error Type Structuring ✅

Review error types and consider newtype wrappers.

**Result**: Kept current approach. The `{ field: &'static str, value: f32 }` pattern is:

- Common in Rust error handling
- Provides good error messages ("lambda must be >= 0, got -1.0")
- Easy to extend without adding new types
- Sufficient for programmatic matching (match on field if needed)

Alternative (nested errors like `ConfigError::Sampling(SamplingError::InvalidSubsample)`)
adds complexity without clear benefit for this use case.

**Tasks**:

- [x] 9.5.1: Audit ConfigError variants → 4 variants (GBDT), 3 variants (GBLinear)
- [x] 9.5.2: Determine if nesting improves usability → No, current pattern sufficient
- [x] 9.5.3: Implement if beneficial → No change needed

**Definition of Done**:

- Error types documented and justified ✅

---

## Story 9.6: Config Serialization ⏸️ DEFERRED

Include GBDTConfig in model serialization.

**Decision**: Deferred. Requires adding Serialize/Deserialize to many types
(Objective, Metric, TreeParams, etc.) with feature-gating. Low priority since:

- Users can recreate configs from objective string
- Models already store objective in `ModelMetadata.objective`
- Config is a training-time concern, not inference-time

**Future Work**: If needed, create separate RFC for config serialization design.

**Tasks**:

- [ ] 9.6.1: Add config field to serialized model format
- [ ] 9.6.2: Implement serialization for GBDTConfig
- [ ] 9.6.3: Update load to restore config
- [ ] 9.6.4: No backwards compatibility needed (no users)

**Definition of Done**:

- ~~Models saved with config~~ Deferred
- ~~Loaded models have `config()` returning `Some`~~ Deferred

---

## Story 9.7: Params/Config Unification ✅

Reduce duplication between GBDTParams and GBDTConfig.

**Result**: Kept current design. The two-layer architecture is intentional:

- `GBDTConfig`: User-friendly, semantically grouped, validated
- `GBDTParams`: Internal, flat, used by trainer

Minor power-user params not exposed in config:

- `early_stopping_eval_set`: Always uses first eval set
- `colsample_bynode`: Always 1.0

Users needing these can construct `GBDTParams` directly.

**Tasks**:

- [x] 9.7.1: Audit overlap between GBDTParams and GBDTConfig → Intentional layering
- [x] 9.7.2: Consider making GBDTConfig produce GBDTParams directly → Already does via `to_trainer_params()`
- [x] 9.7.3: Expose all power-user params → Minor params kept internal

**Definition of Done**:

- Single source of truth for parameters ✅ (GBDTConfig for users, GBDTParams for internal)
- All params accessible to power users ✅ (via direct GBDTParams construction)

---

## Story 9.8: TaskKind Inference from Objective ✅

Fix TaskKind to be inferred from objective, not n_outputs.

**Context**: Multi-output regression (e.g., PinballLoss with multiple quantiles)
is incorrectly classified as MulticlassClassification.

**Result**: Fixed by storing TaskKind in the serialized payload:

- Added `task_kind` field to `ModelMetadata` in payload
- `GBDTModel::to_bytes()` now stores `meta.task`
- `GBDTModel::from_bytes()` uses stored task instead of inferring
- Same fix applied to `GBLinearModel`
- `train()` already correctly uses `objective.task_kind()`

**Tasks**:

- [x] 9.8.1: Add `task_kind()` method to ObjectiveFn → Already exists
- [x] 9.8.2: Use objective's task_kind in model creation → Already done in train()
- [x] 9.8.3: Remove n_outputs-based inference → Fixed in from_bytes()

**Definition of Done**:

- PinballLoss with 3 quantiles → TaskKind::Regression (not Multiclass) ✅

---

## Story 9.9: TaskKind Design Discussion ✅

Evaluate whether TaskKind is needed or can be simplified.

**Decision**: Keep TaskKind as-is. Rationale:

- Provides structured introspection (`is_classification()`, `is_regression()`)
- Correctly derived from objectives via `task_kind()` method
- Preserved in serialization (fixed in Story 9.8)
- Lightweight enum with clear semantics

**Design Choices**:

- `TaskKind::Regression` covers all regression cases (single and multi-output)
- Number of outputs is available via `n_groups` for those who need it
- No `MultiOutputRegression` variant - adds complexity without clear benefit

**Tasks**:

- [x] 9.9.1: Document current TaskKind usage → Introspection, serialization, debug
- [x] 9.9.2: Identify assumptions that cause issues → Fixed n_outputs inference in 9.8
- [x] 9.9.3: Propose simplification or removal → Keep as-is

**Definition of Done**:

- Design decision documented ✅

---

## Story 9.10: Clippy Cleanup ✅

Fix all pre-existing clippy warnings in the codebase.

**Context**: ~70 clippy warnings existed (pre-dating the API refactor).

**Result**: All clippy warnings fixed:
- `derivable_impls` - Removed manual Default implementations
- `needless_borrow` - Removed unnecessary borrows
- `needless_range_loop` - Converted to iterator-based loops
- `redundant_closure` - Used function pointers directly
- `clone_on_copy` - Removed unnecessary clones on Copy types
- `unnecessary_map_or` - Simplified to direct comparisons
- `drain_collect` - Used Vec::default() instead
- `map_identity` - Removed identity maps
- `too_many_arguments` - Added #[allow] for benchmark code
- `collapsible_if` - Merged nested ifs
- `duplicated_attributes` - Removed duplicate doc attributes
- `single_char_add_str` - Changed push_str("\n") to push('\n')

**Tasks**:

- [x] 9.10.1: Run `cargo clippy` and categorize warnings
- [x] 9.10.2: Fix `map_identity` warnings in sampling/row.rs
- [x] 9.10.3: Fix `needless_range_loop` warnings
- [x] 9.10.4: Fix remaining warnings
- [x] 9.10.5: Add clippy to CI (deny warnings)

**Definition of Done**:

- `cargo clippy -- -D warnings` passes ✅

---

## Story 9.11: Documentation Completeness ✅

Audit and complete rustdoc coverage.

**Result**: Fixed all documentation warnings:

- Escaped `[i]` notation in math expressions using backticks
- Replaced broken links (`[FeatureImportance]`, `[FeatureAccessor]`, `[CategoriesStorage]`) with backtick notation
- Fixed HTML tag warnings in benchmark binary by escaping angle brackets
- Added `early_stopping.rs` example demonstrating early stopping configuration

**Note on `ignore` doctests**: These are intentional - they show API patterns without
bulky data setup code. Making them all runnable would obscure the documentation.

**Tasks**:

- [x] 9.11.1: Audit all public types for missing docs → Fixed 14 doc warnings
- [x] 9.11.2: Convert `ignore` doctests to runnable or remove → Kept as ignore (intentional)
- [x] 9.11.3: Add examples to undocumented public items → Covered by existing examples
- [x] 9.11.4: Create `early_stopping.rs` example ✅

**Definition of Done**:

- All public items have rustdoc ✅
- No documentation warnings ✅
- Early stopping example added ✅

---

## Story 9.12: Prediction API Cleanup

Improve prediction API per RFC-0020 Epic 5.

**Context**: Current `predict_batch()` returns `Vec<f32>` and requires `n_rows` parameter.

**Tasks**:

- [ ] 9.12.1: Change return type to `ColMatrix<f32>`
- [ ] 9.12.2: Infer `n_rows` from input where possible
- [ ] 9.12.3: Add `predict()` with automatic transformation
- [ ] 9.12.4: Update examples

**Definition of Done**:

- Predictions return structured matrix
- API matches RFC-0020 specification


> Note: check for new stakeholder feedback.