# RFC-0016: Prediction Architecture

- **Status**: Draft
- **Created**: 2025-12-17
- **Depends on**: RFC-0002 (Forest and Tree Structures)

## Summary

Unify prediction code by:

1. `TreeGrower::grow()` returns `MutableTree` (caller freezes when ready)
2. `TreeView` trait for shared read-only access to Tree and MutableTree
3. `FeatureAccessor` trait to abstract data access (raw, binned, column-major)
4. Batch-first API: `predict_batch_accumulate<A: FeatureAccessor>`

## Motivation

Prediction methods are scattered across Tree (5 methods), Forest (2), Predictor (2).
This causes:

- Duplicated traversal logic
- Inconsistent APIs (return vs mutate slice)

For linear leaves (RFC-0015), we need to fit coefficients on MutableTree **before**
freezing. Current `grow()` calls `freeze()` internally, preventing this.

## Design

### Grower Returns MutableTree

```rust
impl TreeGrower {
    pub fn grow(&mut self, ...) -> MutableTree<ScalarLeaf> {
        // ... build tree ...
        std::mem::take(&mut self.tree_builder)  // No freeze() here
    }
}
```

Caller controls freeze timing:

```rust
let mut tree = grower.grow(...);
// Optional post-processing here (e.g., linear leaves)
let tree = tree.freeze();
```

**No extra allocations**: `Vec::into_boxed_slice()` is free when capacity equals length.

### TreeView Trait

Read-only interface implemented by both Tree and MutableTree:

```rust
pub trait TreeView {
    type LeafValue: LeafValue;
    
    fn n_nodes(&self) -> usize;
    fn is_leaf(&self, idx: u32) -> bool;
    fn split_index(&self, idx: u32) -> u32;
    fn split_threshold(&self, idx: u32) -> f32;
    fn split_type(&self, idx: u32) -> SplitType;
    fn left_child(&self, idx: u32) -> u32;
    fn right_child(&self, idx: u32) -> u32;
    fn default_left(&self, idx: u32) -> bool;
    fn leaf_value(&self, idx: u32) -> &Self::LeafValue;
    fn category_goes_right(&self, idx: u32, category: u32) -> bool;
    
    fn default_child(&self, idx: u32) -> u32 {
        if self.default_left(idx) { self.left_child(idx) } else { self.right_child(idx) }
    }
}
```

Generic (monomorphized), not trait objects—zero vtable overhead.

### FeatureAccessor Trait

Abstract over data sources—enables single traversal implementation:

```rust
pub trait FeatureAccessor: Sync {
    fn n_rows(&self) -> usize;
    fn get(&self, row: usize, feature: usize) -> f32;
}
```

Implementations:

| Type | Use Case | Notes |
|------|----------|-------|
| `RowMatrix<f32>` | Inference | Row-major raw features |
| `ColMatrix<f32>` | GBLinear, linear leaves | Column-major raw features |
| `BinnedAccessor<'_>` | Training prediction | Converts bin → midpoint of bin range |

**Extensibility**: Third-party code can implement `FeatureAccessor` to wrap
DataFrames (polars, arrow) or custom data structures without conversion.

For column-major data, `get(row, feature)` does indexed access. The trait
abstracts layout—traversal code doesn't care.

**Scope**: FeatureAccessor is for the public prediction API. Internal training
functions (like `compute_weight_update`) work directly on slices—no need to
abstract there.

### Unified Traversal

Single generic function:

```rust
#[inline]
pub fn traverse_to_leaf<T: TreeView, A: FeatureAccessor>(
    tree: &T,
    accessor: &A,
    row: usize,
) -> u32 {
    let mut idx = 0u32;
    while !tree.is_leaf(idx) {
        let fvalue = accessor.get(row, tree.split_index(idx) as usize);
        idx = if fvalue.is_nan() {
            tree.default_child(idx)
        } else {
            match tree.split_type(idx) {
                SplitType::Numeric => {
                    if fvalue < tree.split_threshold(idx) { tree.left_child(idx) }
                    else { tree.right_child(idx) }
                }
                SplitType::Categorical => {
                    if tree.category_goes_right(idx, float_to_category(fvalue)) {
                        tree.right_child(idx)
                    } else { tree.left_child(idx) }
                }
            }
        };
    }
    idx
}
```

### Batch-First API

Primary prediction method:

```rust
impl<L: LeafValue> Tree<L> {
    pub fn predict_batch_accumulate<A: FeatureAccessor>(
        &self,
        accessor: &A,
        predictions: &mut [f32],
    ) {
        for row in 0..accessor.n_rows() {
            let leaf_idx = traverse_to_leaf(self, accessor, row);
            predictions[row] += self.leaf_value(leaf_idx).as_f32();
        }
    }
}
```

**MutableTree**: Gets only `traverse_to_leaf` via TreeView—sufficient for linear
leaf training. Full prediction happens after `freeze()`.

### API After Refactor

| Component | Method |
|-----------|--------|
| Tree | `predict_batch_accumulate<A>` |
| Forest | `predict<A>` (convenience, uses Predictor) |
| Forest | `predict_into<A>` (allocates and returns Vec) |
| Predictor\<T\> | `predict<A>` (optimized with blocking/unrolling) |

**Convenience method**: For simple use cases:

```rust
impl<L: LeafValue> Forest<L> {
    /// Predict and return new Vec (allocates)
    pub fn predict_into<A: FeatureAccessor>(&self, accessor: &A) -> Vec<f32> {
        let mut preds = vec![0.0; accessor.n_rows()];
        self.predict(accessor, &mut preds);
        preds
    }
}
```

Removed: `predict_row`, `predict_binned_*`, `par_predict_binned_batch`.

## Design Decisions

### DD-1: Grower Returns MutableTree

Enables post-processing before freeze. No perf cost (`into_boxed_slice` is free).

### DD-2: Keep Tree and MutableTree Separate

Clear phase separation. `Box<[T]>` signals immutability. TreeView provides shared
behavior without storage duplication.

### DD-3: FeatureAccessor Trait

Zero-cost abstraction. Monomorphized—no trait objects. Works with any data layout.

### DD-4: Batch-First API

Accumulate pattern matches training loop. Caller controls allocation.

## Implementation Plan

1. Add TreeView trait + implementations
2. Add FeatureAccessor trait + implementations
3. Change `grow()` to return MutableTree
4. Update trainer to call `freeze()` explicitly
5. Consolidate prediction methods

## Integration

| RFC | Integration |
|-----|-------------|
| RFC-0015 | Fits coefficients on MutableTree before freeze |
| RFC-0002 | TreeView for Tree/MutableTree |
