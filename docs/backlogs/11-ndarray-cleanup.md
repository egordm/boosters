# Backlog: ndarray Cleanup and API Polish

Epic for completing the ndarray migration, removing legacy patterns, and
polishing the API for consistency.

---

## Story 1: Transpose Helper and Array Test Patterns [IN PROGRESS]

Rename `to_c_order` to a transpose-specific helper and clean up test patterns.

### Tasks

- [x] 1.1 Rename `to_c_order` to `transpose_to_c_order` (clarifies purpose)
- [x] 1.2 Update all call sites to use `transpose_to_c_order`
- [x] 1.3 Clean up test assertions to use `array!` macro and `assert_eq!` on arrays
- [ ] 1.4 Use approx integration (`assert_abs_diff_eq!`) for float comparisons where needed

### Definition of Done

- `transpose_to_c_order` is the only transpose helper
- Tests use idiomatic ndarray patterns: `array!`, direct comparisons
- No element-by-element assertions in tests

---

## Story 2: Remove PredictionOutput

Replace `PredictionOutput` with `Array2<f32>` throughout.

### Tasks

- [ ] 2.1 Audit all uses of `PredictionOutput`
- [ ] 2.2 Replace `PredictionOutput` with `Array2<f32>` in return types
- [ ] 2.3 Update callers to use ndarray methods
- [ ] 2.4 Delete `PredictionOutput` struct and module

### Definition of Done

- No `PredictionOutput` references in codebase
- All prediction methods return `Array2<f32>`

---

## Story 3: Simplify LinearModel with ndarray

Make `LinearModel` a wrapper around `Array2<f32>` for cleaner inference.

### Tasks

- [ ] 3.1 Replace `Box<[f32]>` with `Array2<f32>` (shape: `[n_features+1, n_groups]`)
- [ ] 3.2 Use ndarray dot product for predictions
- [ ] 3.3 Update prediction to write directly to column-major output (no conversion)
- [ ] 3.4 Simplify weight/bias access using ndarray slicing

### Definition of Done

- `LinearModel` uses `Array2<f32>` internally
- Prediction uses `dot()` operation
- No post-prediction layout conversion

---

## Story 4: Unify Parallelism Pattern

Remove `predict`/`par_predict` duality, use `Parallelism` enum consistently.

### Tasks

- [ ] 4.1 Add `parallelism: Parallelism` parameter to predict methods
- [ ] 4.2 Remove separate `par_predict` methods
- [ ] 4.3 Update callers to pass `Parallelism`
- [ ] 4.4 Audit other places with parallel/sequential duality

### Definition of Done

- Single `predict(parallelism)` method pattern
- `Parallelism` enum used consistently across codebase

---

## Story 5: GBLinearModel Prediction Cleanup

Use predict.rs module and fix constructor patterns.

### Tasks

- [ ] 5.1 Remove `compute_predictions_raw`, use `LinearModelPredict` trait
- [ ] 5.2 Formulate predict = predict_raw + transform explicitly
- [ ] 5.3 Move `from_linear_model` responsibility to caller (require config)
- [ ] 5.4 Keep `from_parts` as main constructor

### Definition of Done

- `GBLinearModel` uses `LinearModelPredict` trait
- Clear predict_raw → transform → predict pipeline
- No convenience constructors that hide config

---

## Story 6: Gradients Struct Evaluation

Evaluate replacing `Gradients` with `Array2<GradsTuple>` or split arrays.

### Tasks

- [ ] 6.1 Analyze current `Gradients` usage patterns
- [ ] 6.2 Prototype `ArrayView2<GradsTuple>` approach
- [ ] 6.3 Move domain-specific methods to algorithm modules
- [ ] 6.4 Decide: keep Gradients vs. use raw arrays

### Definition of Done

- Clear decision documented
- If migrating: Gradients replaced with ndarray types
- Domain methods moved near their algorithms

---

## Story 7: Eval Sets API Fix

Fix empty eval_sets pattern in high-level train methods.

### Tasks

- [ ] 7.1 Add `eval_sets` parameter to `GBDTModel::train`
- [ ] 7.2 Add `eval_sets` parameter to `GBLinearModel::train`
- [ ] 7.3 Update examples to show eval_sets usage
- [ ] 7.4 Update docstrings

### Definition of Done

- Users can pass eval_sets to high-level train methods
- Examples demonstrate proper usage

---

## Story 8: Base Score Investigation

Investigate LogisticLoss base_score only writing to outputs[0].

### Tasks

- [ ] 8.1 Verify LogisticLoss base_score behavior is correct
- [ ] 8.2 Check other objectives for consistency
- [ ] 8.3 Add test coverage for multi-output base_scores
- [ ] 8.4 Fix any bugs found

### Definition of Done

- Base score behavior documented and correct
- Test coverage for edge cases

---

## Story 9: Import Organization

Move all imports to top of files/modules.

### Tasks

- [ ] 9.1 Audit codebase for local imports
- [ ] 9.2 Move imports to module/file top
- [ ] 9.3 Organize imports (std, external, internal)

### Definition of Done

- All imports at top of files
- Consistent import organization

---

## Story 10: predict_raw + transform Pattern

Formalize predict = predict_raw + transform pattern.

### Tasks

- [ ] 10.1 Rename internal prediction methods for clarity
- [ ] 10.2 Ensure `predict_raw` is public and documented
- [ ] 10.3 Document transformation pipeline
- [ ] 10.4 Update examples showing raw vs transformed

### Definition of Done

- Clear predict_raw → transform pipeline
- Both raw and transformed prediction accessible
