# `repr` refactor plan: canonical representations + naming cleanup

Date: 2025-12-13

## Status: ✅ Complete

All primary goals of this refactor are implemented:

- ✅ Canonical GBDT representation lives in `repr::gbdt` (`Tree`, `Forest`, `MutableTree`, categories, leaves, nodes).
- ✅ `TreeStorage` naming is gone; inference consumes `repr::gbdt::{Tree, Forest}` directly.
- ✅ Sequential builder is removed; construction is via `MutableTree`.
- ✅ `TreeView` wrapper is removed; `Forest` yields `&Tree`.
- ✅ Converters (XGBoost/LightGBM) construct via `MutableTree` (including pre-allocation + explicit child indices).
- ✅ `NodeId` is centralized as `repr::gbdt::NodeId` (type alias).
- ✅ Feature-matrix smoke checks are green (`cargo test`, `--no-default-features`, `--features simd`).
- ✅ Test ergonomics: `scalar_tree!` macro for concise tree construction.

### Deferred items (not blocking, nice-to-have for future work)

These are intentionally not done as part of this refactor. They can be tackled independently if/when valuable:

- `TreeLike` trait: A shared baseline traversal for `Tree` and `MutableTree`. Currently we keep traversal as inherent methods on `Tree`; a trait would only add value if we need polymorphic traversal (e.g., training wants to traverse mutable trees during construction). Low priority.
- Canonical GBLinear in `repr::gblinear`: Linear model weights are currently in `inference::gblinear::LinearModel`. Moving to `repr` is a minor reorganization that doesn't affect functionality. Do it when adding new linear model features.
- Unrolled layout renames: `UnrolledTreeLayout` could become `UnrolledTree` for brevity. Only rename if it reduces confusion at call sites.
- Benchmarks: Should confirm no regressions, but the SoA layout and traversal logic are unchanged, so regressions are unlikely.

---

This doc proposes a focused refactor to (a) introduce a central `repr` module for canonical model representations, and (b) reduce cognitive load by renaming and consolidating the current tree/mutable-tree/optimized-layout types.

Non-goals:

- No public “high-level model enum API” yet (reserved for a future `model` module).
- No performance regressions: SoA traversal and optimized layouts must remain at least as fast and ideally become easier to optimize.

## Motivation

Today, training and inference already share the same canonical representation for GBDT (and partially for linear models), but the types live under `inference::*`.

That creates conceptual friction:

- “Representation” types appear to be inference-specific.
- Builders and converters are scattered and inconsistently named.
- “Optimized layouts” (unrolled/SIMD) are near the representation, but they are really execution strategies.

A `repr` module makes dependencies clearer and sets up the later `model` API layer.

## Guiding principles

- **Canonical representation is neutral**: representation types should not semantically belong to training or inference.
- **Execution stays in inference**: SIMD, unrolled/compiled layouts, predictors, and batch traversal remain in `inference`.
- **Training is allowed to mutate**: training needs an efficient mutable construction/edit phase.
- **Keep SoA**: we keep the current SoA layout and `Box<[T]>` for the frozen form.

## Proposed module layout

### New modules

- `src/repr/mod.rs`
- `src/repr/gbdt/mod.rs`
- `src/repr/gbdt/tree.rs` (canonical tree representation)
- `src/repr/gbdt/forest.rs` (canonical forest representation)
- `src/repr/gbdt/categories.rs` (category bitsets + storage)
- `src/repr/gbdt/leaf.rs` (leaf types/traits)
- `src/repr/gbdt/node.rs` (split enums/types)

- `src/repr/gblinear/mod.rs`
- `src/repr/gblinear/linear_model.rs` (canonical linear model weights/bias)

### Existing modules remain (but depend on `repr`)

- `src/inference/gbdt/*` becomes mostly:
  - predictors/traversal
  - optimized layouts
  - import/compat conversion targets are `repr::gbdt::{Forest, Tree}`

- `src/training/*` uses `repr` types for model output and (optionally) for construction.

### Re-exports (transitional)

During the refactor we can keep churn low by re-exporting:

- `crate::inference::gbdt::{TreeStorage, Forest, ...}` can temporarily re-export from `repr::gbdt`.
- Later we can remove inference-level re-exports if desired.

This repo can break APIs during this redesign, so re-exports are optional; they are just a convenience.

## Naming proposals (Rust-y + consistent)

### Canonical GBDT representation

Current → Proposed

- `TreeStorage<L>` → `Tree<L>`
  - Rationale: it *is* the tree; “Storage” is an implementation detail.

- `Forest<L>` → `Forest<L>`
  - Rationale: already clear.

### Borrowed tree references

Recommendation: do **not** introduce a wrapper type unless it buys something concrete.

- Prefer returning `&Tree<L>` directly from `Forest`.
  - Rationale: `&Tree<L>` is already a “tree reference”, is `Copy`, and already gets all inherent methods via autoderef.

If you still want the *name* `TreeRef` for readability in signatures, use a type alias:

- `type TreeRef<'a, L> = &'a Tree<L>;`

Avoid a newtype wrapper unless we need one of:

- adding methods that are not (or should not be) on `Tree` itself
- implementing foreign traits in a specific way for references
- constraining the API surface (e.g., hiding mutability or hiding some methods)


- `SplitType` stays `SplitType`.
- `CategoriesStorage` stays `CategoriesStorage`.

### Mutable construction

Decision: replace “builder” terminology with a **mutable tree** type.

Current → Proposed

- `MutableTreeBuilder<L>` → `MutableTree<L>`
  - This matches Rust conventions (`String` vs `str`, `Vec<T>` vs `[T]`): one mutable/owned growable structure, one frozen/boxed structure.

Important: we should **remove** the sequential builder.

- The current sequential `TreeBuilder<L>` should be deleted.
- Tests should build trees via `MutableTree`.

Rationale:

- Maintaining two construction APIs is a real long-term burden.
- `MutableTree` can cover both training growth and test ergonomics.
- Conversion `Vec<T> -> Box<[T]>` with `into_boxed_slice()` reuses the allocation (no copy).

### Optimized layouts (inference-only)

Current name: `unrolled` / `UnrolledTreeLayout` etc.

Decision: keep the word **unrolled** (it’s clear and honest).

Rename proposal (optional, minimal churn):

- Keep module `unrolled`.
- Consider renaming `UnrolledTreeLayout` → `UnrolledTree`.
  - Rationale: the optimized thing is effectively “a tree in an unrolled layout”.
  - If we need to distinguish “layout descriptor” from “compiled instance”, we can use:
    - `UnrolledLayout` (descriptor)
    - `UnrolledTree` (the thing used for traversal)

We should avoid renaming just for aesthetics; only rename if it reduces confusion at call sites.

## Where should the tree builder live: `repr` vs `training`?

Recommendation: put the canonical builder in `repr`.

Reasoning:

- The builder constructs the canonical representation and enforces its invariants.
- It is useful for multiple producers:
  - training
  - compatibility loaders/importers
  - tests and benchmarks

But we keep training-only helpers out of `repr`:

- anything that depends on gradients, histograms, datasets, objectives, etc.

### Practical compromise

- `repr::gbdt::MutableTree` = general-purpose construction type (allocate placeholder nodes, apply split, set leaf).
- `training::gbdt` may wrap it with training-specific convenience methods, but does not own the representation.

## Do we need a “tree builder” at all?

Yes, but it should be a “mutable tree”.

### Key distinction: frozen vs mutable

- Frozen `Tree<L>` uses `Box<[T]>` arrays for cache-friendly traversal and a stable memory layout.
- A mutable construction type uses `Vec<T>` for push/resize/edit.

Importantly: converting `Vec<T>` → `Box<[T]>` via `into_boxed_slice()` is *not a copy*; it reuses the allocation.

So we model construction as a real mutable tree type:

- `Tree<L>`: frozen, SoA, boxed slices.
- `MutableTree<L>`: mutable, SoA, vecs.
  - `freeze(self) -> Tree<L>`

This makes the interface feel simpler (“I am editing a tree”) while retaining the performance characteristics in the frozen form.

### Why not just make `Tree` itself use `Vec`?

We can, but it tends to blur invariants:

- inference might accidentally keep trees in a “mutable”/resizable state longer than intended
- hard to enforce “frozen form” invariants (e.g., compact categories storage)

A distinct `TreeMut` / `TreeBuilder` keeps those phases explicit.

## Why do we have two builders today? Should we keep only one?

We should have only one construction API.

Decision:

- Delete the sequential builder.
- Use `MutableTree` everywhere (training, import/conversion, tests).

To keep tests pleasant, `MutableTree` should provide a tiny ergonomic layer:

- `MutableTree::with_capacity(nodes)`
- `MutableTree::init_root()`
- `MutableTree::split_numeric(node, ...) -> (left, right)`
- `MutableTree::split_categorical(node, ...) -> (left, right)`
- `MutableTree::set_leaf(node, value)`
- `MutableTree::freeze() -> Tree<L>`

## Where should “unrolled trees” live: `repr` vs `inference`?

Keep them in `inference`.

Reasoning:

- They are not canonical representation; they are an execution strategy.
- They likely depend on CPU features/SIMD and predictor/traversal logic.
- Keeping them in `repr` would couple representation to a particular runtime optimization pipeline.

However: we can optionally rename *types* to reduce cognitive load (see the `UnrolledTreeLayout` → `UnrolledTree` suggestion above), while keeping “unrolled” terminology.

## Shared traversal / prediction without duplication

Goal: avoid having two divergent “simple traversal” implementations for:

- frozen `Tree`
- `MutableTree`

And enable future work where training can optionally reuse inference optimizations.

### Proposal: `TreeLike` + generic traversal

Introduce a small trait that exposes the SoA accessors required for traversal:

- `trait TreeLike<L: LeafValue>`
  - `fn n_nodes(&self) -> usize`
  - `fn is_leaf(&self, node: u32) -> bool`
  - `fn split_index(&self, node: u32) -> u32`
  - `fn split_type(&self, node: u32) -> SplitType`
  - `fn split_threshold(&self, node: u32) -> f32`
  - `fn left_child(&self, node: u32) -> u32`
  - `fn right_child(&self, node: u32) -> u32`
  - `fn default_left(&self, node: u32) -> bool`
  - `fn leaf_value(&self, node: u32) -> &L`
  - `fn categories(&self) -> &CategoriesStorage`

Implement `TreeLike` for both `Tree` and `MutableTree`.

Then implement one **generic** traversal function (monomorphized, inlinable):

- `fn predict_row<T: TreeLike<L>, L: LeafValue>(tree: &T, features: &[f32]) -> &L`

This gives:

- one correctness path for “simple prediction” used in tests and during training when needed
- no dynamic dispatch (should inline the same as inherent methods)

### Keeping performance + enabling future optimizations

- Inference optimized predictors (SIMD, unrolled layouts) stay in `inference` and do not depend on `TreeLike`.
- If later we want training to reuse those optimizations, we can:
  - add a training-time feature path that predicts on float features via the inference predictor
  - or define a separate “binned feature provider” and compile a traversal specialized for it

Crucially: `TreeLike` is for the correctness-oriented baseline traversal; optimizations remain explicitly opt-in.

## Proposed refactor steps

1. Introduce `repr` module scaffold and move the existing canonical types under it (no behavior changes).
2. Rename `TreeStorage` → `Tree` and update call sites.
3. Move `MutableTreeBuilder` into `repr` and rename it to `MutableTree`.
4. Delete the sequential builder; update tests/importers to use `MutableTree`.
5. Introduce `TreeLike` + one shared baseline traversal for `Tree` + `MutableTree`.
6. Split linear model into `repr::gblinear::LinearModel` (data) and inference prediction code.
7. Keep `unrolled` in inference; optionally rename `UnrolledTreeLayout` → `UnrolledTree` if it improves clarity.
8. Run the feature matrix and benchmarks to confirm no regressions.

## Open questions

- Whether we want the `TreeRef` *alias* (`type TreeRef<'a, L> = &'a Tree<L>;`) or just use `&Tree<L>` everywhere.
- Whether `NodeId` becomes a newtype (e.g., `struct NodeId(u32)`) or stays `u32`.
  - Newtype improves clarity but can add tiny friction; can be phased later.
