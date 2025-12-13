# Strict audit (2025-12-13)

This report is a follow-up to the `src/` inventory in `docs/analysis/src_audit.md`.

## What was checked

### 1) Full `src/` tree + symbol inventory

- See: `docs/analysis/src_audit.md` (human-readable)
- See: `docs/analysis/src_items.json` (machine-readable file list)

### 2) “Warnings are errors” health (dead-code / unused items)

The main enforcement mechanism used here is building and testing with warnings denied:

- `RUSTFLAGS='-Dwarnings' cargo check --all-targets`
- `RUSTFLAGS='-Dwarnings' cargo test`

This ensures there are no unused imports/variables, unreachable code, dead private functions, etc. across:

- library
- binary
- tests
- default-compiled benches

### 3) Feature-flag compilation matrix (promised features)

These were validated to compile under `-Dwarnings`:

- `cargo check --no-default-features`
- `cargo check --no-default-features --features xgboost-compat`
- `cargo check --no-default-features --features simd`
- `cargo check --no-default-features --features lightgbm-compat`

Bench feature validation:

- `cargo check --features bench-lightgbm --bench prediction_lightgbm`

## Findings

### Fixed: broken `simd` feature

- Root cause: `#[cfg(feature = "simd")] mod simd;` existed, but the module file was missing.
- Fix: added `src/inference/gbdt/simd.rs` with a correctness-first traversal implementation (keeps the feature functional and public type names stable).

### Fixed: broken `lightgbm-compat` feature

- Root cause: `src/compat/lightgbm/convert.rs` referenced removed legacy path `crate::trees::SoATreeStorage`.
- Fix: updated conversion to return the current `inference::gbdt::TreeStorage`.

### Fixed: broken `bench-lightgbm` benchmark target

- Root cause: `benches/prediction_lightgbm.rs` used removed legacy modules (`model`, `objective`, `predict`).
- Fix: updated the benchmark to load a `Forest` via `compat::LgbModel` and run inference via `inference::{Predictor, ...}`.

### Fixed: stale doc paths

- `src/testing.rs` and `src/inference/gbdt/predictor.rs` contained examples referencing a removed `booste_rs::predict` path.

## Notes on “test-only” code

- The public `testing` module (`src/testing.rs`) is intentionally a test-utility surface and is used by integration tests under `tests/`.
- Within the production modules (`data/`, `training/`, `inference/`, `compat/`, `utils/`), the `-Dwarnings` checks act as the primary guardrail against unused private code.

## Known remaining opt-in items

- The GBLinear training benches are now explicitly gated behind `--features bench-training`.
  They were previously failing due to API drift; gating keeps default strict checks green while preserving the work for later modernization.
