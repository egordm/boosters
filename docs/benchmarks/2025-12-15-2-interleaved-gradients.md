# Interleaved Gradients (AoS) – Benchmark + Quality Report (2025-12-15)

## Summary

This report evaluates the “interleaved gradients” refactor (single canonical `(grad,hess)` AoS layout across training) for both performance and basic quality/accuracy checks.

- Host: Apple M1 Pro, 10 cores
- Toolchain: rustc 1.91.1, cargo 1.91.1
- Git: `2f73f20` (clean)

## What Changed

- `Gradients` now stores interleaved `GradHessF32 { grad: f32, hess: f32 }` in output-major order (per-output slice is contiguous).
- Objectives write directly into `&mut [GradHessF32]`.
- Row sampling (Uniform/GOSS) operates on `&mut [GradHessF32]`.
- TreeGrower gathers `ordered_grad_hess: Vec<GradHessF32>` and builds histograms using interleaved ordered builders.

## Quality / Accuracy Checks

Command:

- `BOOSTERS_RUN_QUALITY=1 cargo test -q --test quality_smoke`

Result:

- Passed (4 tests) in ~22s.

Notes:

- This suite covers regression (RMSE/MAE), binary (LogLoss/Accuracy), and multiclass (MLLogLoss/Accuracy) on synthetic targets.
- The optional `BOOSTERS_RUN_QUALITY=1` test is intended to align with benchmark reporting and provides tighter thresholds than the always-on smoke checks.

## Benchmarks

### Intra-crate (boosters) training benchmarks

Command:

- `cargo bench -q --bench training_gbdt`

Outcome (high level):

- Criterion reported large improvements vs prior local baseline across quantization and end-to-end training workloads (often ~35–45% faster for training scenarios).

### Cross-library: LightGBM training comparison

Command:

- `cargo bench -q --features bench-lightgbm --bench training_lightgbm`

Observed results (regression):

- `small`
  - boosters: ~323 ms (≈ 1.55 Melem/s)
  - lightgbm: ~239 ms (≈ 2.09 Melem/s)
  - boosters is ~1.35× slower (~34% slower)

- `medium`
  - boosters: ~1.82 s (≈ 2.74 Melem/s)
  - lightgbm: ~1.49 s (≈ 3.36 Melem/s)
  - boosters is ~1.22× slower (~22% slower)

### Cross-library: XGBoost training comparison

Command:

- `cargo bench -q --features bench-xgboost --bench training_xgboost`

Observed results (regression):

- `small`
  - boosters: ~299 ms (≈ 1.67 Melem/s)
  - xgboost: ~547 ms (≈ 0.915 Melem/s)
  - boosters is ~1.8× faster

- `medium`
  - boosters: ~1.72 s (≈ 2.90 Melem/s)
  - xgboost: ~2.11 s (≈ 2.37 Melem/s)
  - boosters is ~1.23× faster

## Has the Gap to LightGBM Narrowed?

- The interleaved refactor produced a large speedup inside boosters itself (per `training_gbdt` deltas and also the “boosters” side of the LightGBM comparison benches).
- Despite that, on these regression training benchmarks we are still behind LightGBM (roughly 22–34% slower in the measured cases above).

## Histogram Kernels: Are They All Needed?

`src/training/gbdt/histograms/ops.rs` contains multiple kernels because histogram building must handle combinations of:

- Bin storage types: `u8` and `u16`
- Layouts: dense (stride=1), strided (row-major grouped bins), sparse
- Access patterns:
  - unordered (direct `grad[row]` / `hess[row]`)
  - ordered (pre-gathered gradients, but bin lookup via `indices`)
  - sequential-ordered fast path (contiguous row range, avoids streaming `indices`)
- Gradient layout:
  - SoA (two `&[f32]` slices)
  - AoS interleaved (`&[GradHessF32]`)

**What production uses today**

- Tree training uses the interleaved ordered entry points:
  - `build_histograms_ordered_interleaved`
  - `build_histograms_ordered_sequential_interleaved`

**What’s likely removable / gateable**

- The SoA ordered/unordered entry points were unused by training after the refactor and have now been removed from the public surface area.
- Unit tests in `ops.rs` were updated to validate the interleaved builders against a naive reference implementation instead of AoS-vs-SoA equivalence.

## Possible Lossless Next Optimizations (if we want to close LightGBM gap)

These are candidates that should be lossless (no algorithm/accuracy changes), but require separate benchmarking:

- Reduce overhead in histogram loops (e.g., minimize repeated loads/casts and ensure tight loops stay vectorizable on Apple Silicon).
- Revisit the ordered/sequential strategy thresholds specifically for M1 (the current `ParallelStrategy::auto_select` constants were tuned earlier and may not be optimal).
- Reduce per-node work in TreeGrower outside histogram kernels (e.g., partition bookkeeping / split bookkeeping), since LightGBM’s implementation is extremely optimized around the full training pipeline.

## Addendum: Histogram Strategy Tuning (2025-12-15)

To make strategy tuning more scientific, ephemeral microbenchmarks were created during profiling.
The bench-helper entrypoints (`build_histograms_ordered_*_with_strategy`) are exposed under the
`bench-training` feature flag for future tuning exercises.

Key observation on M1 Pro: for *small leaves* the best choice depends heavily on both feature count and available threads.
For example, `512x10` can prefer sequential (parallel overhead dominates), while `512x100` benefits from feature-parallel even though row count is below the old `MIN_ROWS_PARALLEL` threshold.

As a lossless improvement, `ParallelStrategy::auto_select` was refined to use a simple work-per-thread heuristic (`rows × features` per thread) rather than a hard minimum-row cutoff.

## Repro Notes

- LightGBM and XGBoost comparison benches require features:
  - `--features bench-lightgbm`
  - `--features bench-xgboost`
- This report was originally collected while iterating on the refactor; the final code is now committed as `2f73f20`. If you want, we can re-run the benches again post-commit to regenerate the numbers with a clean provenance.
