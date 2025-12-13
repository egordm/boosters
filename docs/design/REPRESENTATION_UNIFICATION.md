# Unifying training + inference representations

Status: implemented (2025-12-13)

## Problem (historical)

Historically the repo contained two separate GBDT “model libraries”:

- Inference model representation in `src/inference/gbdt/*` (`TreeStorage`, `Forest`, `Predictor`, traversal strategies).
- Training-side tree/forest representation in `src/training/gbdt/tree/*` (`TreeNode`, `Tree`, `TreeBuilder`, `Forest`) used during training and also for some prediction in binned space.

That training-side representation has since been removed; training now writes directly into the canonical inference representation.

This duplication increases maintenance cost and makes feature parity harder:

- New features tend to get implemented twice (or only in one side).
- Compatibility code can drift ("promised features" compile but are subtly inconsistent).
- It’s easy for evaluation/prediction behavior to differ between training-time and inference-time trees.

## Goal

Have **one canonical GBDT model representation** that:

- is used as the public “trained model” type;
- is used by compatibility loaders;
- is the only representation the inference subsystem optimizes;
- does **not** regress training performance.

## Recommendation (canonical model)

Make `inference::gbdt::{TreeStorage, Forest}` the canonical trained-model representation.

## Current state (2025-12-13)

- Training produces `inference::gbdt::Forest<ScalarLeaf>` and builds `TreeStorage` directly (no post-hoc conversion).
- The training-side tree module (`src/training/gbdt/tree/*`) has been deleted.
- Split semantics are aligned at the numeric boundary and categorical domain:
  - Numeric: thresholds are translated as `next_up(bin_upper_bound)` so training “bin <= threshold_bin goes left” matches inference “value < threshold_value goes left” at exact boundaries.
  - Categorical: the canonical categorical domain is **category indices** (`0..K-1`), i.e. categorical bin indices.

The main invariants are protected by regression tests in `src/training/gbdt/grower.rs`.

Rationale:

- It already encodes the performance intent (SoA layout, traversal abstraction, unrolled layouts).
- It matches the direction of external model imports (XGBoost/LightGBM import -> inference forest).
- It avoids a permanent “training forest” type leaking into the public API.

## Key design decision: keep training fast

Training uses quantized bins heavily (histograms, partitioning). That’s correct.

The proposal is **not** to change training’s dataset/gradient/histogram internals.

Instead:

- Training continues to *compute* splits in bin space (because that is fast and already implemented).
- Training writes the final tree directly into the canonical SoA `TreeStorage`, with a tiny translation step for split thresholds/categories.

That translation happens once per split, not per sample.

## Concrete plan (phased refactor)

### Phase 1 (low risk): unify the public return type

Status: done.

Change `training::GBDTTrainer::train` to return `inference::gbdt::Forest<ScalarLeaf>`.

Implementation detail:

- Keep the existing training tree builder and training forest internally for now.
- Add a conversion step at the end:
  - `training::gbdt::tree::Tree` -> `inference::gbdt::TreeStorage<ScalarLeaf>`
  - `training::gbdt::tree::Forest` -> `inference::gbdt::Forest<ScalarLeaf>`

Why this helps immediately:

- Downstream users see one model type.
- Compatibility loaders and training converge on the same output type.

Performance impact:

- Negligible in practice: conversion is $O(\text{num_nodes})$ per tree, once per tree.

### Phase 2 (medium risk): build canonical trees directly

Status: done.

Remove the conversion by making training build `TreeStorage` directly.

How:

- Extend the canonical builder `inference::gbdt::TreeBuilder` with “mutable building” APIs similar to the training builder:
  - `init_root()` (alloc placeholder node 0)
  - `apply_split(node, ...) -> (left, right)` (allocate children, fill node arrays)
  - `make_leaf(node, value)`
  - `finish() -> TreeStorage`

This allows replacing the former training-side `TreeBuilder` without changing the tree growth logic.

The training-side `TreeNode` / `Tree` / `Forest` types become unnecessary.

### Phase 3 (cleanup): delete or internalize training tree module

Status: done (module deleted).

- Delete `src/training/gbdt/tree/*` or move it under a clearly internal module name (only if something still depends on it). (Done: module deleted.)
- Ensure `training::gbdt` code only exposes training APIs and returns canonical inference forests.

### Phase 4 (semantic alignment): make split semantics explicit and consistent

Status: done.

There is a subtle but important semantic mismatch today:

- Training uses bin splits: “bin <= threshold_bin goes left”.
- Inference numeric traversal uses: “feature_value < threshold_value goes left”.

To avoid any boundary mismatches when exporting training-built models into float-threshold inference trees:

- For numerical splits, compute the threshold as the bin upper bound and adjust with `nextafter(+∞)`.
  - Let `t = mapper.bin_to_value(threshold_bin) as f32`.
  - Store `threshold_value = nextafter(t, +∞)`.

This ensures values exactly equal to the bin boundary still go left.

### Phase 5 (categorical compatibility): decide on one categorical domain

Status: done (Option A).

Inference categorical splits currently assume XGBoost-style encoding:

- categorical feature values are non-negative integers (stored as `f32` but representing `u32` indices)
- bitsets are indexed by those `u32` category indices

Training categorical splits currently operate on **bin indices**.

To unify without slowing inference, pick one of these options:

#### Option A (recommended): canonicalize training categorical mapping to `0..K-1`

- Ensure the categorical bin mapper produces category indices starting at 0.
- Treat “bin index” == “category index”.
- Then training categorical sets can be converted to inference categorical bitsets losslessly.

#### Option B: store per-feature categorical mapping in the trained model

- Put categorical mappers into a small `ModelMeta` stored alongside the forest.
- Add a fast preprocessing step in prediction to map raw category values -> internal index.

Option A tends to be simpler and faster long-term.

## Acceptance criteria

- The only public trained model type for GBDT is `inference::gbdt::Forest<ScalarLeaf>`.
- `RUSTFLAGS='-Dwarnings' cargo test` passes.
- Feature matrix stays green:
  - `--no-default-features`
  - `xgboost-compat`, `lightgbm-compat`, `simd`
- Training benchmarks show no regression attributable to representation changes.

## Suggested invariants to document/test

- Training and inference decisions match for exported models (including missing/default direction).
- Categorical encoding is explicitly defined (and tested) for both imported and trained models.
- Base score semantics are consistent across training outputs and imported models.
