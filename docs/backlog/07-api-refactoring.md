# Epic 7: API Refactoring

**Status**: Complete ✅  
**Priority**: Medium  
**Depends on**: Epic 5 (Phase 2 Complete)

## Overview

Refactor the training API to be more user-friendly and reduce code duplication.

## Completed Work (2024-12-02)

### API Renames

| Old Name | New Name | Rationale |
|----------|----------|-----------|
| `train` (quantized data) | `train_quantized` | Advanced API for pre-quantized data |
| `train_with_data` (raw data) | `train` | Simple API that handles quantization |
| `train_multiclass` | `train_multioutput` | More general (covers multi-quantile, multi-output regression) |

### GOSS Multi-Output Support

Added `sample_multioutput` method to `GossSampler`:
- Computes L2 norm of gradient vector for row importance
- Enables GOSS sampling for multiclass and multi-quantile training
- Location: `src/training/gbtree/sampling.rs`

## Completed Work (2024-12-03)

### Builder Pattern API (Story 3 - COMPLETE ✅)

Implemented `GBTreeTrainerBuilder` as the primary API for configuring and training models.

**New API:**
```rust
let mut trainer = GBTreeTrainer::builder()
    .loss(SquaredLoss)           // Required
    .num_rounds(100)
    .max_depth(6)                // Depth-wise growth
    // OR: .max_leaves(31)       // Leaf-wise growth
    .learning_rate(0.1)
    .min_child_weight(1.0)
    .reg_lambda(1.0)
    .reg_alpha(0.0)
    .subsample(0.8)
    .colsample_bytree(1.0)
    .max_bins(256)
    .verbosity(Verbosity::Info)
    .seed(42)
    .build();

let forest = trainer.train(&data, &labels);
```

**Key changes:**
- `GBTreeTrainerBuilder` provides fluent configuration
- Policy (depth-wise vs leaf-wise) is now internal, selected via `max_depth` vs `max_leaves`
- Cut finder and max_bins are now part of trainer config, not train() args
- Growth strategy stored in `TrainerParams.growth_strategy`
- `train(&data, &labels)` - simplified API (handles quantization internally)
- `train_with_eval_sets(&data, &labels, &eval_sets)` - with evaluation sets
- `train_quantized()` - advanced API for pre-quantized data

### Files Updated (Full List)

- `src/training/gbtree/trainer.rs` - Method renames, module docs, builder pattern
- `src/training/gbtree/sampling.rs` - Added `sample_multioutput`
- `src/training/gbtree/mod.rs` - Export builder
- `src/training/mod.rs` - Re-export builder
- `src/training/linear/trainer.rs` - Renamed `train_multiclass` → `train_multioutput`
- `tests/training_gbtree_tests.rs` - Updated method calls
- `tests/training_gblinear/*.rs` - Updated method calls
- `examples/train_regression.rs` - Updated to use builder pattern
- `examples/train_classification.rs` - Updated to use builder pattern

## Analysis (2024-12-02)

### File Size Review

| File | Lines | Status |
|------|-------|--------|
| `trainer.rs` (gbtree) | ~2200 | ~1180 lines are tests - acceptable |
| `quantize.rs` | 1328 | Could split, but cohesive |
| `split.rs` | 1261 | Cohesive - keep |
| `grower.rs` | 1076 | Already in module |
| `dense.rs` | 1067 | Contains iterators - could split |
| `metric.rs` | 958 | Could split by metric type |

**Decision**: Files are large but cohesive. Splitting would add complexity without significant benefit. Keep as-is.

### Terminology

- ✅ Renamed `train_multiclass` → `train_multioutput` (more general)
- Keep `n_outputs` internally (already done)
- `multioutput` covers: multiclass, multi-quantile, multi-output regression

### Code Duplication in Trainers

`train` vs `train_multioutput` share:
- Base score computation
- Gradient computation flow
- Tree building loop
- Early stopping logic (missing in multioutput!)
- Logging

Differences:
- num_outputs: 1 vs K
- Trees per round: 1 vs K
- GOSS sampling: supported vs not (gap!)
- Eval metrics: supported vs ignored (gap!)

**Recommendation**: Don't unify now. Instead, add missing features to `train_multioutput`:
1. ✅ Add GOSS/sampling support (infrastructure ready)
2. Add eval set metrics
3. Add early stopping

### API Simplification Options

**Option A: Static helper (simple)**
```rust
// One-liner training
let forest = booste_rs::train_regression(&data, &labels);
let forest = booste_rs::train_multiclass(&data, &labels, num_classes);
```

**Option B: Builder pattern (flexible)**
```rust
let forest = GBTreeTrainer::builder()
    .loss(SquaredLoss)
    .max_depth(6)
    .num_rounds(100)
    .build()
    .train(&data, &labels);
```

**Option C: Config struct (current approach, improved)**
```rust
// Current API is fine, just needs cleaner defaults
let config = TrainingConfig::regression();
let config = TrainingConfig::classification(num_classes);
```

**Recommendation**: ~~Keep current API, add convenience constructors to TrainerParams.~~ → Implemented builder pattern instead.

---

## Stories

### Story 1: API Rename (COMPLETE ✅)

- [x] 1.1: Rename `train` → `train_quantized` (advanced API)
- [x] 1.2: Rename `train_with_data` → `train` (simple API)  
- [x] 1.3: Rename `train_multiclass` → `train_multioutput`
- [x] 1.4: Add `sample_multioutput` to GossSampler
- [x] 1.5: Update all tests and examples

### Story 2: Add Missing Features to Multioutput Training

- [ ] 2.1: Integrate GOSS sampling into `train_multioutput`
- [ ] 2.2: Add row subsampling support  
- [ ] 2.3: Add eval set metrics logging
- [ ] 2.4: Add early stopping support
- [ ] 2.5: Update tests

### Story 3: Builder Pattern API (COMPLETE ✅)

- [x] 3.1: Add `GBTreeTrainerBuilder` struct with fluent API
- [x] 3.2: Move `growth_strategy` and `max_bins` into `TrainerParams`
- [x] 3.3: Simplify `train()` to take only `(data, labels)`
- [x] 3.4: Add `train_with_eval_sets()` for evaluation during training
- [x] 3.5: Update all examples to use builder pattern

### Story 4: Linear Trainer Parity

- [ ] 4.1: Ensure same features available as GBTree
- [ ] 4.2: Unify config naming conventions

---

## Deferred

### File Splitting

Files are large but cohesive. Splitting would:

- Add indirection
- Break IDE navigation
- Increase cognitive load navigating modules

Only split when natural boundaries emerge.

### Deep Unification

Unifying `train` and `train_multiclass` into one generic function would:


Keep separate until a clear abstraction emerges.
