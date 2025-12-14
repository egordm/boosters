# Benchmarking Plan (Performance + Quality)

This document describes how we should restructure `criterion` benchmarks and add a **model quality** evaluation suite.

The goal is to make it easy to answer two different questions:

1. **Internal performance engineering**: “Is change X faster, and why?” (e.g. SIMD vs scalar, layout choices, traversal strategy, parallel scaling).
1. **External positioning**: “How do we compare to XGBoost / LightGBM on the same workloads?”

A third, separate axis is **quality**:

1. **Model quality**: “Given comparable settings, are we as accurate as XGBoost / LightGBM?”

---

## 1) Benchmark taxonomy

We should organize benchmarks into three layers, each with a different purpose.

### A) Microbenchmarks (inner loops)

**Purpose**: isolate a single variable and measure it precisely.

Examples:

- Tree traversal kernel variants (scalar vs unrolled vs SIMD)
- Row-major ↔ column-major conversion cost
- Threshold compare / branching patterns
- Histogram / binning kernels (if applicable)
- Gradient / hessian buffer layouts (AoS vs SoA)

Rules:

- No allocations in the timed region.
- Inputs pre-generated; results black-boxed.
- Always record throughput (e.g. rows/sec, nodes/sec) in addition to time.

### B) Component benchmarks (library elements)

**Purpose**: measure meaningful library sub-systems while still being attributable.

Examples (maps to current bench files):

- Prediction: core traversal, different prediction strategies, parallel scaling
- Training: core training loop, access patterns, matrix layout choice

Rules:

- Pre-build models and datasets outside the timed region.
- Measure realistic shapes (narrow/wide/tall) from the benchmark matrix.

### C) End-to-end benchmarks (user-facing)

**Purpose**: measure what users experience.

Examples:

- Training time for a fixed number of rounds/trees (no early stopping)
- Prediction throughput for a fixed trained model
- Train + predict combined

Rules:

- Fixed iteration count and fixed seed.
- Identical evaluation policy across libraries (either disabled, or identical and excluded from timed region).

---

## 2) Shared infrastructure (tests + benches)

### Why share utilities?

Synthetic dataset generation and small “model factory” helpers are useful in **both** unit tests and benchmarks:

- tests need small deterministic matrices and labels
- benches need larger deterministic matrices and consistent splits

### Proposed split

We should have:

1. **Thin Criterion entrypoints** under `benches/` (minimal glue)
1. **Reusable testing utilities inside the library** (shared by unit tests, integration tests, and benchmarks)

#### Library-side utilities (use the existing `testing` module)

We already have `booste_rs::testing` (see `src/testing/`). Instead of adding a second module (e.g. `testkit`), expand the existing module to include **synthetic data generation** and small **fixture helpers**.

Important nuance: Criterion benches compile the library as a normal dependency, so `cfg(test)` is **not** enabled there. Therefore, anything benchmarks need must be available via one of these patterns:

1. **Always-on** under `booste_rs::testing` (acceptable if small and dependency-light)
1. **Feature-gated** under `booste_rs::testing` (recommended for heavier helpers)

Suggested feature naming:

- `testing-utils`: enables synthetic generators + dataset loaders used by benches/tools
- keep purely assertion-style helpers always-on

This keeps the “one testing utilities module” story clean, while still making benchmarks reuse the same code paths as tests.

#### Bench-only utilities

Keep `benches/common/` for Criterion-specific glue:

- Criterion configuration presets
- consistent benchmark naming helpers
- per-benchmark parameter matrices
- rayon threadpool setup helpers

### Principle: benchmarks should be mostly wiring

Benchmarks should ideally call:

- public or semi-public library APIs
- `booste_rs::testing` helpers for dataset generation / loading

It’s OK to have some custom glue (e.g. adapters to XGBoost/LightGBM APIs), but we should avoid re-implementing algorithms in benchmark code.

---

## 3) Standardizing configuration

### Criterion configuration

All benchmarks should use a consistent baseline config (override only when necessary):

- warmup time
- measurement time
- sample size
- noise threshold

We should also standardize:

- benchmark IDs (include rows/cols/trees/depth/threads)
- throughput (set `Throughput::Elements(rows)` or similar)

### Threading control

For internal and cross-library comparisons we must explicitly control:

- number of Rayon threads
- whether XGBoost/LightGBM run single-threaded or multi-threaded

Policy:

- default to single-threaded comparisons for fairness + determinism
- add explicit scaling benches as a separate suite

### Single-thread and multi-thread comparisons vs other libraries

We should explicitly support both:

- **1 thread**: baseline, most fair across implementations
- **N threads**: show real-world throughput scaling

For cross-library comparisons, benchmark IDs should include thread count, e.g. `threads=1` and `threads=<num_cores>`.

When multi-threading, ensure all libraries are actually using the requested thread count (Rayon threadpool, XGBoost `nthread`, LightGBM `num_threads`).

---

## 4) Cache fairness (XGBoost / LightGBM)

External libraries sometimes cache internal representations (e.g. dataset bins, prediction caches). This can make benchmarks accidentally compare “cold” for one library vs “warm” for another.

We should make cache behavior explicit and report it in results.

### Recommended policy

For cross-library benchmarks, provide two modes:

1. **Cold**: clear or recreate cached structures between iterations
1. **Warm/steady-state**: allow caches and measure throughput after warmup

Rules:

- Never do file IO in the timed region.
- If a library has an explicit cache-reset API, use it.
- Otherwise, recreate the relevant handle (e.g. DMatrix/Dataset) outside timing and benchmark only predict, or recreate per-iteration if you truly want “cold”.

Notes specific to this repo:

- The XGBoost dependency comment mentions a fork with a `reset()` method for accurate benchmarking (cache clearing). Prefer that for “cold” mode.
- For LightGBM, be cautious about prediction caching; prefer constructing predictors/datasets in a controlled way and document what is reused.

---

## 5) Dataset matrices

### Performance matrix (synthetic)

Define a small matrix of synthetic datasets that we always run:

- **Small**: 10k rows × 50 cols
- **Narrow**: 100k rows × 20 cols
- **Wide**: 50k rows × 2000 cols
- **Tall**: 1M rows × 50 cols (throughput-focused)

Options:

- dense float32 only (baseline)
- optional “sparse-ish” generator (many zeros + missing) if representative

Rules:

- fixed RNG seeds
- deterministic label generation (regression / binary / multiclass)
- fixed train/valid split

### Quality matrix (real datasets)

Training speed is hard to compare fairly, but **model quality** can be compared more robustly by using standard datasets and the same iteration count.

We should maintain a small suite of standard datasets across domains:

Binary classification:

- Adult income (tabular)
- Higgs (large tabular)

Multiclass classification:

- Iris (tiny sanity)
- a medium tabular multiclass (e.g. Covertype subset)

Regression:

- California housing
- a “wide-ish” regression dataset (optional)

For each dataset define:

- preprocessing rules (missing handling, categorical encoding policy)
- split policy (seeded train/valid/test)
- evaluation metrics (one or two)

Recommended metrics:

- regression: RMSE + MAE (or “RMAE” if that’s the chosen internal name)
- binary: logloss + accuracy/AUC
- multiclass: mlogloss + accuracy

### Storage format for real datasets (compact + reproducible)

We want a format that is:

- compact (doesn’t bloat the repo)
- simple to load from Rust
- easy to generate from Python (scikit-learn / scipy)

Recommended options (in order):

1. **Arrow IPC / Parquet** (preferred)

- store `X` and `y` together in one file as Arrow IPC (`.arrow`/`.feather`) or Parquet (`.parquet`)
- benchmarks/tests load via `booste_rs::data::io::{arrow, parquet}` and then build whatever internal structures they need
- keep benchmark-specific metadata (task type, split seed, params) in a **separate testcase file** (e.g. JSON) owned by the harness

Example layout:

- `data/quality/<dataset>/data.arrow` (or `data.parquet`)
- `data/quality/<dataset>/case.json` (split seed, objective, metrics, iterations, etc)

1. **Zstd-compressed raw arrays + JSON** (fallback)

- store `X` row-major as `f32` (little-endian) and compress with zstd
- store `y` as `f32` (regression) or `u32`/`f32` (classification labels)
- store *shape only* (`rows`, `cols`, dtype/layout) if needed; task/params should still live in testcase config

### Repo bloat policy

- Keep only **small** datasets (or small subsets) in-repo (e.g. < 5–20 MB compressed total).
- For larger datasets (e.g. Higgs), provide a script to download and cache locally (excluded from git).

This keeps “quality comparisons” realistic without turning the repo into a dataset mirror.

---

## 6) Cross-library comparisons (performance)

### Adapter approach

Use optional features already in `Cargo.toml`:

- `bench-xgboost`
- `bench-lightgbm`

Implement adapters that can:

- train for N rounds on the same in-memory data
- predict on the same in-memory data

### Fairness rules

For every comparison, document:

- objective
- iteration count
- tree complexity controls (depth/leaves)
- learning rate
- sampling params (ideally disable sampling: subsample=1, feature_fraction=1)
- threading

Call out mismatches explicitly (e.g. exact split finding algorithm differs).

---

## 7) Cross-library comparisons (quality)

### Why quality belongs here

If we only benchmark runtime, we can accidentally “win” by being less accurate. A lightweight quality suite keeps us honest.

### Proposed workflow

- Train each library on the same dataset + split
- Use a fixed iteration count (no early stopping)
- Compute metrics on the same validation set
- Store results as a small JSON/CSV artifact

This can live as:

- an integration test-like harness under `tools/` (recommended), or
- a `cargo run --example quality_eval` style executable

It should not run in CI by default.

---

## 8) File layout proposal (cleaner)

The current `benches/` folder will look much cleaner if we take advantage of Cargo’s ability to set a custom `path` per benchmark target.

### Bench entrypoints

Keep entrypoints grouped by suite under `benches/suites/`, and declare them in `Cargo.toml` with `path = ...`.

Suggested structure:

- `benches/`
  - `common/` (Criterion-only glue)
    - `criterion_config.rs`
    - `ids.rs` (standard benchmark IDs)
    - `matrix.rs` (dataset matrix enumerations)
    - `threading.rs` (Rayon + external thread config)
    - `cache_policy.rs` (cold vs warm helpers)
  - `suites/`
    - `micro/`
      - `traversal.rs`
      - `layout.rs`
    - `component/`
      - `predict.rs`
      - `train.rs`
    - `e2e/`
      - `predict.rs`
      - `train.rs`
    - `compare/`
      - `predict_xgboost.rs` (feature-gated)
      - `train_xgboost.rs` (feature-gated)
      - `predict_lightgbm.rs` (feature-gated)
      - `train_lightgbm.rs` (feature-gated)

This keeps “what we benchmark” obvious from the file tree.

### Library-side testing utilities

Extend the existing module:

- `src/testing/`
  - `data/` (synthetic generators + splits)
  - `datasets/` (real dataset loaders; likely feature-gated)
  - keep existing assertion helpers and fixtures as-is

Benchmarks should primarily call into `booste_rs::testing` for data generation/loading, and into the normal training/inference APIs for the actual work.

---

## 9) Reporting

We should distinguish:

- **Results**: versioned benchmark writeups in `docs/benchmarks/` (existing process)
- **Methodology + structure**: this plan + a checklist

Add a short checklist doc that every benchmark run references:

- no IO in timed region
- fixed seeds
- thread settings
- how to run SIMD vs non-SIMD
- cold vs warm cache policy

---

## 10) Next steps

1. Introduce `src/testkit` + feature gating (tests always, benches via feature)
1. Move synthetic data generation out of `benches/bench_utils.rs` into the testkit
1. Add `benches/common` with a single Criterion config + naming conventions
1. Refactor existing benches to use the common modules
1. Add end-to-end suites (train N rounds, predict throughput)
1. Add a separate “quality harness” under `tools/` to compare metrics across libs

Note: dataset loading can now standardize on Arrow IPC / Parquet via `booste_rs::data::io`, and benchmark/test harnesses should keep their own testcase metadata separate from the dataset file.
