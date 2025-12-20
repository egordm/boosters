# Backlog 08: API Layering and Usability Redesign

**RFC**: [0020-api-layering-redesign.md](../rfcs/0020-api-layering-redesign.md)  
**Priority**: High (Critical Refactor)  
**Status**: Accepted - Ready for Implementation

---

> Note: don't forget to address stakeholder feedback in tmp/stakeholder_feedback.md.

## Overview

Comprehensive API redesign establishing clear abstraction layers, consistent naming, and improved usability. This refactor is foundational for all future work including Python bindings.

**Scope**: 8 epics, 24 stories, ~3 weeks estimated effort.

**Guiding Principles**:

- Create module skeleton first, then populate
- Minimize file moves by working in target locations from the start
- `Model` trait omitted for now; models work standalone without shared interface

---

## Breaking Changes

This refactor introduces breaking changes (acceptable for 0.x library with no external users):

1. **Serialization format**: Models saved with old API cannot be loaded with new API
2. **Public API surface**: Type names, module paths, and function signatures change
3. **Configuration structure**: Flat params → nested `GBDTConfig`

No migration path provided. Users must retrain models after upgrade.

> Mini stakeholder feedback: that's okay. No need to add backwards comparibility.There are no users yet.

---

## Epic 0: Preparation

Capture baselines before any code changes.

**Effort**: 2-4 hours

### Story 0.1: Capture Baselines

Record performance and output baselines for regression testing.

**Tasks**:

- [x] 0.1.1: Run benchmark suite and save results:
  - `cargo bench --bench prediction_core`
  - `cargo bench --bench training_gbdt`
  - Save to `docs/benchmarks/<date>-pre-refactor-baseline.md` (use actual date)
- [x] 0.1.2: Select 3 representative XGBoost compat test cases
- [x] 0.1.3: Run selected tests, capture exact prediction outputs to test file
- [x] 0.1.4: Commit baseline file (can be in `tests/` or `docs/`)

**Definition of Done**:

- Benchmark results saved with commit hash
- Prediction outputs for 3 test cases captured

**Testing Criteria**:

- N/A (preparation only)

---

## Epic 1: Foundation - Module Skeleton + Naming

Create target module structure and align naming conventions. These are low-risk changes that set up the foundation.

**Effort**: Small (1-2 days)

### Story 1.1: Create Module Skeleton

Create empty module structure per RFC layout (files can be empty or have placeholder exports).

**Tasks**:

- [x] 1.1.1: Create all new module directories and files:
  - `model/mod.rs`, `model/gbdt.rs`, `model/gblinear.rs`
  - `training/mod.rs`, `training/objectives/mod.rs`, `training/metrics/mod.rs`
  - `training/gbdt/mod.rs`, `training/gblinear/mod.rs`
  - `inference/mod.rs`, `inference/gbdt/mod.rs`, `inference/gblinear/mod.rs`
  - `repr/mod.rs`
- [x] 1.1.2: Update `lib.rs` to declare new modules (can be empty initially)

**Note**: `inference/` module may remain empty initially; created for future compatibility.

**Definition of Done**:

- All module directories/files exist
- `cargo check` passes (modules may be empty)

**Testing Criteria**:

- N/A (structural change only)

---

### Story 1.2: Objective Naming Alignment

Rename `Objective` trait → `ObjectiveFn`, `ObjectiveFunction` enum → `Objective`. Atomic rename - all changes in one commit.

**Tasks**:

- [x] 1.2.1: Rename trait and enum, update all implementations and usages
- [x] 1.2.2: Add convenience constructors to `Objective` enum

**Definition of Done**:

- `ObjectiveFn` trait exists, `Objective` is the selection enum
- All existing objectives implement `ObjectiveFn`
- `cargo test` passes

**Testing Criteria**:

- All existing tests pass with NO test logic changes (only import updates)
- Functionality identical before/after

---

### Story 1.3: Metric Naming Alignment

Apply same pattern: `Metric` trait → `MetricFn`, `MetricFunction` enum → `Metric`.

**Tasks**:

- [x] 1.3.1: Rename trait and enum, update all implementations and usages
- [x] 1.3.2: Add convenience constructors to `Metric` enum

**Definition of Done**:

- `MetricFn` trait exists, `Metric` is the selection enum
- Tests pass

**Testing Criteria**:

- All existing tests pass with NO test logic changes

---

### Story 1.4: Loss-Based Naming

Rename task-based names to loss-based names where applicable.

**Tasks**:

- [x] 1.4.1: Audit existing loss/objective names in `training/objectives/` - identify any needing rename (e.g., `Quantile` → `PinballLoss`)
- [x] 1.4.2: Rename identified types to `FooLoss` pattern
- [x] 1.4.3: Update enum variants and usages

**Definition of Done**:

- All loss types follow `FooLoss` naming convention
- Tests pass

**Testing Criteria**:

- Renamed types produce identical outputs

---

## Epic 2: Objective and Metric Streamlining

Simplify traits and ensure clean delegation patterns.

**Effort**: Small-Medium (2-3 days)

### Story 2.1: Streamline ObjectiveFn Trait

Define minimal required methods with sensible optional defaults.

**Tasks**:

- [x] 2.1.1: Audit existing objective trait in `training/objectives/` - identify current method set
- [x] 2.1.2: Define required methods: `compute_gradients()`, `compute_base_score()`
- [x] 2.1.3: Add optional methods with defaults: `n_outputs()`, `transform_predictions()`, `task_kind()`, `name()`
- [x] 2.1.4: Update all implementations to match new signature

**Definition of Done**:

- `ObjectiveFn` has 2 required + 4 optional methods
- All implementations updated
- Tests pass

**Testing Criteria**:

- Each objective computes correct gradients/hessians (existing tests)
- Default methods return sensible values
- Multi-output objectives (`SoftmaxLoss`, `PinballLoss`) return correct `n_outputs()`

---

### Story 2.2: Objective Enum Delegation

Ensure enum holds pre-constructed structs with trivial delegation.

**Tasks**:

- [x] 2.2.1: Verify enum variants hold actual structs (not raw data)
- [x] 2.2.2: Implement `ObjectiveFn` for `Objective` with simple match-delegation
- [x] 2.2.3: Add `Custom(Arc<dyn ObjectiveFn>)` variant for runtime polymorphism
- [x] 2.2.4: Implement `Default` (returns `SquaredLoss`)

**Definition of Done**:

- `Objective` implements `ObjectiveFn` via delegation
- `Custom` variant exists
- Tests pass

**Testing Criteria**:

- All enum variants delegate correctly (spot-check gradient computation)
- **Custom objective integration test**:
  - Implement `AbsoluteErrorCustom` as separate struct implementing `ObjectiveFn`
  - Train 5-tree model with custom objective
  - Verify predictions match built-in `AbsoluteLoss` within tolerance (`< 1e-5`)

---

### Story 2.3: MetricFn Trait and Metric Enum

Apply same patterns to metrics.

**Tasks**:

- [x] 2.3.1: Audit existing metric trait
- [x] 2.3.2: Define `MetricFn` with `compute()`, `name()`, `higher_is_better()`
- [x] 2.3.3: Implement `Metric` enum with delegation
- [x] 2.3.4: Add `Custom(Arc<dyn MetricFn>)` variant
- [x] 2.3.5: Implement `Default` (returns `Rmse`)

**Definition of Done**:

- `MetricFn` trait and `Metric` enum follow same pattern as objectives
- Tests pass

**Testing Criteria**:

- All metrics compute correctly (existing tests)
- `higher_is_better()` returns correct value:
  - `Metric::rmse().higher_is_better() == false`
  - `Metric::mae().higher_is_better() == false`
  - `Metric::auc().higher_is_better() == true`
  - `Metric::accuracy().higher_is_better() == true`

---

### Story 2.4: TaskKind Consolidation

Ensure single source of truth for TaskKind.

**Tasks**:

- [x] 2.4.1: Audit codebase for TaskKind definitions/duplicates
- [x] 2.4.2: Consolidate to single definition in `model/meta.rs`
- [x] 2.4.3: Update all usages to import from single location

**Definition of Done**:

- One `TaskKind` enum exists
- All code uses the same definition

**Testing Criteria**:

- Compilation succeeds
- No duplicate type errors

---

## Epic 3: Configuration Layer

Establish nested parameter groups and builder pattern with `bon`.

> Note: don't forget to address stakeholder feedback in tmp/stakeholder_feedback.md.

**Effort**: Medium (3-4 days)

### Story 3.1: Add bon Dependency ✅

Add `bon` crate for builder pattern generation.

**Tasks**:

- [x] 3.1.1: Add `bon = "3.8"` to `Cargo.toml`
- [x] 3.1.2: Verify compilation with `cargo check`

**Definition of Done**:

- `bon` is available as dependency
- Crate compiles

**Testing Criteria**:

- `cargo build` succeeds

---

### Story 3.2: Audit Existing Parameters ✅

Before creating new param structs, understand current state. **Time-box: 4 hours max.**

**Tasks**:

- [x] 3.2.1: Inventory all existing parameter/config structs
- [x] 3.2.2: Document current field names and default values
- [x] 3.2.3: Identify what can be grouped vs what stays top-level

**Definition of Done**:

- Clear mapping from old → new param organization
- Document in PR description or temporary doc

**Testing Criteria**:

- N/A (research task)

---

### Story 3.3: Implement Nested Parameter Groups ✅

Create `TreeParams`, `RegularizationParams`, `SamplingParams` structs.

**Tasks**:

- [x] 3.3.1: Create `TreeParams` with `max_depth`, `max_leaves`, `growth_strategy`
- [x] 3.3.2: Create `RegularizationParams` with `lambda`, `alpha`, `min_child_weight`, `min_gain`
- [x] 3.3.3: Create `SamplingParams` with `subsample`, `colsample_bytree`, `colsample_bylevel`
- [x] 3.3.4: Implement `Default` for each with documented values
- [x] 3.3.5: Add validation methods (e.g., `SamplingParams::validate()`)
- [x] 3.3.6: Place in `model::gbdt` module

**Definition of Done**:

- All param structs exist with defaults and validation
- Located in `model::gbdt` module

**Testing Criteria**:

- Default values match RFC documentation
- Validation rejects invalid values:
  - `subsample = 0.0` → error
  - `subsample = 1.0` → valid
  - `subsample = 1.5` → error
  - `colsample_bytree ∈ (0, 1]` enforced
  - `colsample_bylevel ∈ (0, 1]` enforced

---

### Story 3.4: Implement GBDTConfig with Builder ✅

Create high-level config using bon builder pattern.

> Note: don't forget to address stakeholder feedback in tmp/stakeholder_feedback.md.

**Tasks**:

- [x] 3.4.1: Create `GBDTConfig` struct composing param groups
- [x] 3.4.2: Add `#[bon] impl GBDTConfig` with `#[builder(finish_fn = build)]`
- [x] 3.4.3: Add all fields per RFC
- [x] 3.4.4: Implement validation in `build()` returning `Result<Self, ConfigError>`
- [x] 3.4.5: Define `ConfigError` enum with variants:
  - `InvalidLearningRate(f32)` - learning_rate <= 0
  - `InvalidNTrees` - n_trees == 0
  - `InvalidSamplingRatio { field: &'static str, value: f32 }` - sampling ratios out of range
  - Other validation errors as needed

**Definition of Done**:

- `GBDTConfig::builder()...build()` pattern works
- Invalid configs fail at build time with clear errors
- Tests pass

**Testing Criteria**:

- Builder with all defaults produces valid config
- `learning_rate = 0.0` → `Err(ConfigError::InvalidLearningRate)`
- `learning_rate = 1.0` → `Ok` (boundary valid)
- `learning_rate > 1.0` → `Ok` but emit warning (allowed but unusual, matches XGBoost)
- `n_trees = 0` → `Err(ConfigError::InvalidNTrees)`
- Nested param validation cascades correctly
- **Integration test**: Train model with `max_depth = 10`, `subsample = 0.8` on synthetic dataset (1000 samples, 10 features) - verify successful training

---

### Story 3.5: Implement GBLinearConfig ✅

Create parallel config for linear models.

**Tasks**:

- [x] 3.5.1: Create `GBLinearConfig` struct
- [x] 3.5.2: Create `model::gblinear::RegularizationParams` (lambda, alpha only - no tree params)
- [x] 3.5.3: Add bon builder with validation
- [x] 3.5.4: Define GBLinear-specific `ConfigError` if needed

**Definition of Done**:

- `GBLinearConfig` exists with builder ✓
- Tests pass (15 tests) ✓

**Testing Criteria**:

- Builder works with defaults ✓
- Validation catches invalid params ✓

---

## Epic 4: Model Layer Refactoring

Update model types to store config, remove forwarding, and update train signatures.

> Note: don't forget to address stakeholder feedback in tmp/stakeholder_feedback.md.

**Effort**: Medium-Large (4-5 days)

### Story 4.1: Refactor GBDTModel Structure ✅

Update to store config and remove forwarding methods.

**Tasks**:

- [x] 4.1.1: Add `config: GBDTConfig` field to `GBDTModel`
- [x] 4.1.2: Audit and remove forwarding methods (e.g., `n_trees()`, `n_groups()`)
- [x] 4.1.3: Add accessors: `fn forest(&self) -> &Forest`, `fn meta(&self) -> &ModelMeta`, `fn config(&self) -> &GBDTConfig`
- [x] 4.1.4: Update `from_parts()` to accept config
- [x] 4.1.5: Update serialization to include config

**Note**: Serialization serializes metadata but not config yet (can be done later if needed). Config is `Option<GBDTConfig>` for backwards compatibility.

**Definition of Done**:

- `GBDTModel` stores config ✓
- Accessors replace forwarding methods ✓
- Serialization preserves metadata ✓

**Testing Criteria**:

- `model.forest().n_trees()` works (accessor pattern) ✓
- `model.config().map(|c| c.learning_rate)` accessible ✓
- Serialization round-trips: save model, load ✓
- **Note**: Old serialized models will NOT load (breaking change, documented above)

---

### Story 4.2: Update GBDTModel::train()

New signature accepting Dataset, GBDTConfig, and eval_sets.

**Tasks**:

- [ ] 4.2.1: Update signature to `train(dataset: &Dataset, config: GBDTConfig, eval_sets: &[(&str, &Dataset)]) -> Result<Self, TrainError>`
- [ ] 4.2.2: Convert Dataset to BinnedDataset internally (binning config derived automatically from data statistics)
- [ ] 4.2.3: Create trainer using config's nested params
- [ ] 4.2.4: Define `EvalSet` struct in `training/mod.rs` (e.g., `pub(crate) struct EvalSet<'a> { name: &'a str, dataset: &'a Dataset }`), convert eval_sets
- [ ] 4.2.5: Implement early stopping logic (may leverage existing code if present; if new implementation required, may need additional time)
- [ ] 4.2.6: Store config in resulting model

**Definition of Done**:

- New train signature works
- Eval sets supported for monitoring and early stopping

**Testing Criteria**:

- Training with default config produces valid model
- Training with custom objective (logistic) works
- **Early stopping test**:
  - Configure `early_stopping_rounds = 5`, `n_trees = 100`
  - Provide eval_set that will plateau early
  - Verify training stops before 100 trees
  - Verify `model.forest().n_trees() < 100`
- Model stores same config used for training

---

### Story 4.3: Refactor GBLinearModel

Apply same patterns as GBDTModel.

**Tasks**:

- [ ] 4.3.1: Add `config: GBLinearConfig` field
- [ ] 4.3.2: Add accessors: `linear()`, `meta()`, `config()`
- [ ] 4.3.3: Update `train()` signature with eval_sets
- [ ] 4.3.4: Implement early stopping for linear models
- [ ] 4.3.5: Update serialization

**Definition of Done**:

- `GBLinearModel` follows same patterns as `GBDTModel`

**Testing Criteria**:

- Training with default config produces valid model
- Training with custom objective works
- **Early stopping test** (same pattern as Story 4.2):
  - Configure `early_stopping_rounds = 5`
  - Verify training stops when eval metric plateaus
- Config stored and accessible after training

---

### Story 4.4: Trainer Layer Updates

Ensure trainers work with new config types.

**Tasks**:

- [ ] 4.4.1: Audit existing trainer param types (identify any intermediate types to remove)
- [ ] 4.4.2: Update `GBDTTrainer` to accept/use `GBDTConfig` directly
- [ ] 4.4.3: Update `GBLinearTrainer` similarly
- [ ] 4.4.4: Ensure generics work with `ObjectiveFn` trait bound
- [ ] 4.4.5: Remove any intermediate "TrainerParams" types that duplicate config
- [ ] 4.4.6: Verify trainer compiles with both concrete types and `Objective` enum

**Definition of Done**:

- Trainers use config directly, no conversion types
- Generic objective/metric work correctly

**Testing Criteria**:

- Trainer compiles with `GBDTTrainer<O: ObjectiveFn, M: MetricFn>`
- Trainer works with `Objective` enum (not just concrete types)
- Custom objectives work via trainer

---

## Epic 5: Prediction Simplification

Remove wrapper types, return ColMatrix directly.

**Effort**: Small (1-2 days)

> Note: don't forget to address stakeholder feedback in tmp/stakeholder_feedback.md.

### Story 5.1: Remove PredictionOutput

Replace any prediction wrapper with direct ColMatrix returns.

**Tasks**:

- [ ] 5.1.1: Audit existing prediction return types (if already ColMatrix, skip to 5.1.6)
- [ ] 5.1.2: Update `GBDTModel::predict()` to return `ColMatrix<f32>`
- [ ] 5.1.3: Update `GBDTModel::predict_raw()` to return `ColMatrix<f32>`
- [ ] 5.1.4: Update `GBLinearModel` predictions similarly
- [ ] 5.1.5: Remove `PredictionOutput` type if it exists
- [ ] 5.1.6: Update all call sites

**Definition of Done**:

- Predictions return `ColMatrix<f32>` directly
- No wrapper types exist

**Testing Criteria**:

- Predictions have correct shape (n_rows × n_outputs)
- **Regression test**: Compare outputs to baseline captured in Story 0.1
- XGBoost compatibility tests pass with unchanged expected values

---

## Epic 6: Type Relocation and Re-exports

Move types to their final locations and establish clean public API.

**Effort**: Medium (2-3 days)

**Note**: Module skeleton already created in Epic 1. This epic populates the modules.

> Note: don't forget to address stakeholder feedback in tmp/stakeholder_feedback.md.

### Story 6.1: Relocate All Types

Move all types to their target modules in one coordinated effort.

**Approach**: Atomic move - all changes in one commit to avoid broken intermediate states.

**Note**: `compat/` module (XGBoost/LightGBM loading) stays in place - not part of this reorganization.

**Note**: `inference/` module created in skeleton but may remain empty; `Predictor<T>` can be moved there in future iteration.

**Tasks**:

- [ ] 6.1.1: Move model types: `GBDTModel`, `GBDTConfig`, param groups → `model::gbdt`
- [ ] 6.1.2: Move model types: `GBLinearModel`, `GBLinearConfig` → `model::gblinear`
- [ ] 6.1.3: Move objectives: `ObjectiveFn`, `Objective`, loss structs → `training::objectives`
- [ ] 6.1.4: Move metrics: `MetricFn`, `Metric`, metric structs → `training::metrics`
- [ ] 6.1.5: Move `TaskKind` → `training::objectives`
- [ ] 6.1.6: Move trainers → `training::gbdt`, `training::gblinear`
- [ ] 6.1.7: Move repr types: `Forest`, `Tree`, `LinearModel` → `repr`
- [ ] 6.1.8: Update all internal imports throughout crate

**Definition of Done**:

- All types in correct modules per RFC
- All internal references updated

**Testing Criteria**:

- `cargo test` passes with NO changes to test logic (only imports)
- Run `cargo test` before and after, verify same test count
- `cargo doc` builds

---

### Story 6.2: Establish Public API Re-exports

Define clean public API surface in `lib.rs`.

**Tasks**:

- [ ] 6.2.1: Re-export from `lib.rs`:
  - **High-level**: `GBDTModel`, `GBLinearModel`
  - **Config**: `GBDTConfig`, `GBLinearConfig`, `TreeParams`, `RegularizationParams`, `SamplingParams`
  - **Training**: `ObjectiveFn`, `Objective`, `MetricFn`, `Metric`, `TaskKind`
  - **Data**: `Dataset`, `ColMatrix`, `RowMatrix`, `DenseMatrix`
- [ ] 6.2.2: Ensure internal types are NOT re-exported and have `pub(crate)` visibility:
  - `GrowerParams`, `HistogramParams` - internal
  - `Histogram`, `Split`, `SplitInfo` - internal
  - `BinnedDataset` - internal
- [ ] 6.2.3: Add module-level rustdoc to each public module

**Definition of Done**:

- Users can `use boosters::{GBDTModel, GBDTConfig, Objective, Metric, Dataset, ColMatrix}`
- Internal types require full path or are private

**Testing Criteria**:

- Example code from RFC compiles with minimal imports
- `cargo doc` shows clean public API
- Internal types not visible in docs unless explicitly enabled

---

## Epic 7: Documentation Update

Update all documentation for new API.

**Effort**: Small-Medium (2-3 days)

### Story 7.1: Update Rustdoc

Document all public types with examples.

**Tasks**:

- [ ] 7.1.1: Document `GBDTModel` with usage examples (train, predict)
- [ ] 7.1.2: Document `GBDTConfig` with builder examples
- [ ] 7.1.3: Document `ObjectiveFn` trait with custom implementation example
- [ ] 7.1.4: Document `Objective` enum with all variants and convenience constructors
- [ ] 7.1.5: Document `MetricFn` and `Metric` similarly
- [ ] 7.1.6: Document param groups (`TreeParams`, `RegularizationParams`, etc.)
- [ ] 7.1.7: Add module-level docs for `model`, `training`, `repr`

**Definition of Done**:

- All public items have rustdoc
- Examples compile (doctests pass)
- `cargo doc` builds without warnings
- Each public type has at least one doctest example

**Testing Criteria**:

- `cargo test --doc` passes

---

### Story 7.2: Update README and Examples

Update top-level documentation.

**Tasks**:

- [ ] 7.2.1: Update README with new API usage patterns
- [ ] 7.2.2: Create/update examples in `examples/` directory:
  - `basic_training.rs` - default config, train, predict
  - `custom_objective.rs` - implement and use custom loss
  - `early_stopping.rs` - train with eval_set and early stopping
- [ ] 7.2.3: Ensure all examples compile and run

**Definition of Done**:

- README reflects new API
- Three examples exist and work

**Testing Criteria**:

- `cargo run --example basic_training` succeeds
- `cargo run --example custom_objective` succeeds
- `cargo run --example early_stopping` succeeds

---

### Story 7.3: Final Validation

Run final benchmarks and regression tests against baseline from Story 0.1.

**Tasks**:

- [ ] 7.3.1: Run same benchmark suite as Story 0.1
- [ ] 7.3.2: Compare results to baseline - verify no significant regression (< 5% slower)
- [ ] 7.3.3: Run regression tests against captured prediction outputs (tolerance: `< 1e-5` for f32)
- [ ] 7.3.4: Document results in `docs/benchmarks/<date>-post-refactor.md`
- [ ] 7.3.5: If regression detected, investigate and fix before merge

**Definition of Done**:

- Benchmarks show no significant regression
- Prediction outputs match baseline (tolerance: `< 1e-5`)

**Testing Criteria**:

- Benchmark comparison documented
- Regression test passes

---

## Definition of Done (Epic Level)

Each epic is complete when:

1. All stories within the epic are complete
2. All tests pass (`cargo test`)
3. Documentation builds (`cargo doc`)
4. No clippy warnings (`cargo clippy`)
5. Code compiles on stable Rust

---

## Quality Gates

- **Unit Tests**: All existing tests pass with identical behavior
- **Integration Tests**: XGBoost/LightGBM compat tests pass unchanged
- **Regression Tests**: Capture prediction outputs before refactor, verify identical after
- **Benchmarks**: Run before/after comparison - no significant regression
- **Doc Tests**: All rustdoc examples compile and run

---

## Dependencies

| Story | Depends On |
| ----- | ---------- |
| 1.1-1.4 | 0.1 (baselines captured) |
| 2.1 | 1.2 (trait renamed first) |
| 2.2-2.4 | 2.1 |
| 3.3-3.5 | 3.1, 3.2, 3.3 |
| 4.1 | 3.4 (config exists) |
| 4.2 | 4.1 |
| 4.3 | 3.5 |
| 4.4 | 4.1, 4.2, 4.3 |
| 5.1 | 4.1, 4.3 |
| 6.1 | All of Epics 1-5 |
| 6.2 | 6.1 |
| 7.1-7.3 | 6.2 |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Large refactor scope | Breaking changes accumulate | Complete epics incrementally, test after each story |
| Breaking existing tests | False confidence | Run full test suite after every story, capture regression outputs |
| Performance regression | Slower training/inference | Run benchmarks before Epic 1 and after Epic 6, compare |
| Serialization incompatibility | Old models won't load | Document as breaking change (done above) |

**Rollback Strategy**: Create git branch per epic. If epic fails mid-way, revert to branch point. Merge to main only when epic complete and tested.

---

## Estimated Effort

| Epic | Effort | Notes |
| ---- | ------ | ----- |
| Epic 0 | 2-4 hours | Baseline capture |
| Epic 1 | 1-2 days | Module skeleton + naming renames |
| Epic 2 | 2-3 days | Trait streamlining |
| Epic 3 | 3-4 days | Config layer, most complex |
| Epic 4 | 4-5 days | Model refactor, train updates |
| Epic 5 | 1-2 days | Prediction simplification |
| Epic 6 | 1-2 days | Type relocation (consolidated) |
| Epic 7 | 2-3 days | Documentation + final validation |
| **Total** | **14-21 days** | ~3 weeks |

---

## Final Review Checklist

Before marking refactor complete:

- [ ] All stories complete and checked off
- [ ] `cargo test` passes (all tests)
- [ ] `cargo test --doc` passes (all doctests)
- [ ] `cargo clippy` passes (no warnings, no new warnings introduced)
- [ ] `cargo doc` builds without warnings
- [ ] Benchmarks show no significant regression vs baseline
- [ ] Prediction outputs match baseline within tolerance
- [ ] README updated with new API
- [ ] All three examples work
- [ ] Breaking changes documented
- [ ] Git history includes meaningful commits (one per story or epic preferred)

