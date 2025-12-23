# Backlog: ndarray Cleanup and API Polish

Epic for completing the ndarray migration, removing legacy patterns, and
polishing the API for consistency.

---

## Story 1: Transpose Helper and Array Test Patterns ✅

Rename `to_c_order` to a transpose-specific helper and clean up test patterns.

### Tasks

- [x] 1.1 Rename `to_c_order` to `transpose_to_c_order` (clarifies purpose)
- [x] 1.2 Update all call sites to use `transpose_to_c_order`
- [x] 1.3 Clean up test assertions to use `array!` macro and `assert_eq!` on arrays
- [x] 1.4 Use approx integration (`assert_abs_diff_eq!`) for float comparisons where needed

### Definition of Done

- `transpose_to_c_order` is the only transpose helper ✅
- Tests use idiomatic ndarray patterns: `array!`, direct comparisons ✅
- No element-by-element assertions in tests ✅

---

## Story 2: Remove PredictionOutput ✅

Replace `PredictionOutput` with `Array2<f32>` throughout.

### Tasks

- [x] 2.1 Audit all uses of `PredictionOutput`
- [x] 2.2 Replace `PredictionOutput` with `Array2<f32>` in return types
- [x] 2.3 Update callers to use ndarray methods
- [x] 2.4 Delete `PredictionOutput` struct and module

### Definition of Done

- No `PredictionOutput` references in codebase ✅
- All prediction methods return `Array2<f32>` ✅

---

## Story 3: Simplify LinearModel with ndarray ✅

Make `LinearModel` a wrapper around `Array2<f32>` for cleaner inference.

### Tasks

- [x] 3.1 Replace `Box<[f32]>` with `Array2<f32>` (shape: `[n_features+1, n_groups]`)
- [x] 3.2 Use ndarray dot product for predictions
- [x] 3.3 Update prediction to write directly to column-major output (no conversion)
- [x] 3.4 Simplify weight/bias access using ndarray slicing

### Definition of Done

- `LinearModel` uses `Array2<f32>` internally ✅
- Prediction uses `dot()` operation ✅
- No post-prediction layout conversion ✅

---

## Story 4: Unify Parallelism Pattern ✅

Remove `predict`/`par_predict` duality, use `Parallelism` enum consistently.

### Tasks

- [x] 4.1 Add `parallelism: Parallelism` parameter to predict methods
- [x] 4.2 Remove separate `par_predict` methods
- [x] 4.3 Update callers to pass `Parallelism`
- [x] 4.4 Audit other places with parallel/sequential duality

### Definition of Done

- Single `predict(parallelism)` method pattern ✅
- `Parallelism` enum used consistently across codebase ✅

---

## Story 5: GBLinearModel Prediction Cleanup ✅

Use predict.rs module and fix constructor patterns.

### Tasks

- [x] 5.1 Remove `compute_predictions_raw` loop, use `LinearModelPredict` trait with dot product
- [x] 5.2 Formulate predict = predict_raw + transform explicitly (already done)
- [x] 5.3 Keep `from_linear_model` for test convenience (well-documented default config)
- [x] 5.4 Keep `from_parts` as main constructor for production

### Definition of Done

- `GBLinearModel` uses `LinearModelPredict` trait ✅
- Clear predict_raw → transform → predict pipeline ✅
- Constructors are well-documented ✅

---

## Story 6: Gradients Struct Evaluation ✅

Evaluate replacing `Gradients` with `Array2<GradsTuple>` or split arrays.

### Decision: Keep Gradients

The `Gradients` struct is well-designed and should be kept because:

1. **Interleaved layout**: Stores `(grad, hess)` pairs together for cache efficiency
2. **Column-major optimization**: Output-major layout optimizes histogram building hot path
3. **ndarray integration**: Already provides `pairs_array()` returning `ArrayView2<GradsTuple>`
4. **Domain methods**: `sum()`, `bias_update()` belong with the data structure

### Tasks

- [x] 6.1 Analyze current `Gradients` usage patterns
- [x] 6.2 Prototype `ArrayView2<GradsTuple>` approach (already exists via pairs_array)
- [x] 6.3 Domain-specific methods are appropriate where they are
- [x] 6.4 Decision: keep Gradients

### Definition of Done

- Clear decision documented ✅
- Gradients struct retained with ndarray integration ✅

---

## Story 7: Eval Sets API Fix ✅

Fix empty eval_sets pattern in high-level train methods.

### Tasks

- [x] 7.1 Add `eval_sets` parameter to `GBDTModel::train`
- [x] 7.2 Add `eval_sets` parameter to `GBLinearModel::train`
- [x] 7.3 Update examples to include eval_sets parameter
- [x] 7.4 Update docstrings

### Definition of Done

- Users can pass eval_sets to high-level train methods ✅
- Examples demonstrate proper usage ✅

---

## Story 8: Base Score Investigation ✅

Investigate LogisticLoss base_score only writing to outputs[0].

### Finding: No Bug

All objectives handle `compute_base_score` correctly:

1. **Single-output** (`LogisticLoss`, `SquaredLoss`): Write to `outputs[0]` or `fill()` - correct
2. **Multi-output** (`SoftmaxLoss`, `PinballLoss`): Iterate over all outputs - correct

`LogisticLoss` writing only to `outputs[0]` is correct because it's a single-output binary classification objective.

### Tasks

- [x] 8.1 Verify LogisticLoss base_score behavior is correct - ✅ Correct (single-output)
- [x] 8.2 Check other objectives for consistency - ✅ All consistent
- [x] 8.3 Add test coverage for multi-output base_scores - ✅ Already covered by quantile/softmax tests
- [x] 8.4 No bugs found

### Definition of Done

- Base score behavior documented and correct ✅
- Test coverage for edge cases ✅

---

## Story 9: Import Organization ✅

Move all imports to top of files/modules.

### Finding: Already Organized

Audited the codebase and found:

1. **Module-level imports**: Already at top of files
2. **Test imports**: Properly in `#[cfg(test)] mod tests {}` blocks
3. **Function-local imports**: Appropriately scoped (e.g., serde deserialize helpers)

The codebase follows Rust idiomatic import patterns. No changes needed.

### Tasks

- [x] 9.1 Audit codebase for local imports
- [x] 9.2 Imports already at module/file top
- [x] 9.3 Import organization follows Rust conventions

### Definition of Done

- All imports at top of files ✅ (or appropriately scoped)
- Consistent import organization ✅

---

## Story 10: predict_raw + transform Pattern ✅

Formalize predict = predict_raw + transform pattern.

### Finding: Already Formalized

Both `GBDTModel` and `GBLinearModel` already implement this pattern:

1. `predict_raw()` - Returns raw margin scores (public, documented)
2. `predict()` - Returns `predict_raw()` + `objective.transform_predictions()`

The pattern is documented in the module docstring at `src/model/mod.rs`.

### Tasks

- [x] 10.1 Rename internal prediction methods for clarity (`compute_predictions_raw` is internal)
- [x] 10.2 Ensure `predict_raw` is public and documented
- [x] 10.3 Document transformation pipeline (in module docs)
- [x] 10.4 Pattern is clear from method names and docs

### Definition of Done

- Clear predict_raw → transform pipeline ✅
- Both raw and transformed prediction accessible ✅
