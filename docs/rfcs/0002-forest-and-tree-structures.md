# RFC-0002: Forest and Tree Structures

**Status**: Implemented

## Summary

This RFC documents the canonical representation for gradient-boosted decision tree (GBDT) ensembles. The design uses Structure-of-Arrays (SoA) layout for cache-efficient traversal and supports multi-output models via tree groups.

## Design

### Forest

The `Forest<L>` struct manages a collection of trees with multi-output support:

```rust
pub struct Forest<L: LeafValue> {
    trees: Vec<Tree<L>>,
    tree_groups: Vec<u32>,   // Maps each tree to its output group
    n_groups: u32,           // Number of output groups (1 for regression, K for multiclass)
    base_score: Vec<f32>,    // Per-group base scores
}
```

**Multi-output strategy**: Rather than K-dimensional leaves, forests use K separate tree groups where each group contributes to one output dimension. Trees are assigned to groups via `tree_groups[i]`. Predictions accumulate tree outputs per group, starting from `base_score`.

### Tree Storage

Trees use SoA layout for cache-efficient traversal:

```rust
pub struct Tree<L: LeafValue> {
    split_indices: Box<[u32]>,      // Feature index per node
    split_thresholds: Box<[f32]>,   // Threshold per node (numeric splits)
    left_children: Box<[u32]>,      // Left child node ID
    right_children: Box<[u32]>,     // Right child node ID
    default_left: Box<[bool]>,      // Missing value direction
    is_leaf: Box<[bool]>,           // Leaf flag
    leaf_values: Box<[L]>,          // Leaf values (only valid for leaf nodes)
    split_types: Box<[SplitType]>,  // Numeric or Categorical
    categories: CategoriesStorage,   // Packed bitsets for categorical splits
}
```

All arrays are indexed by `NodeId` (u32). Node 0 is always the root. Child indices are local to the tree.

**Split types**:
- `Numeric`: go left if `value < threshold`
- `Categorical`: go left if category NOT in bitset, right if in bitset

### Categorical Split Storage

Categorical splits use packed u32 bitsets:

```rust
pub struct CategoriesStorage {
    categories: Box<[u32]>,         // Flat bitset data (32 categories per word)
    segments: Box<[(u32, u32)]>,    // Per-node (start, size) into categories
}
```

Bit `c` set means category `c` goes RIGHT. This matches XGBoost's partition-based format.

### Building Trees

`MutableTree<L>` provides a builder pattern for tree construction:

1. **Initialize**: `init_root()` or `init_root_with_num_nodes(n)` for pre-allocated trees
2. **Build**: `apply_numeric_split()` / `apply_categorical_split()` allocate children and set split info
3. **Finalize**: `make_leaf()` marks nodes as leaves
4. **Freeze**: `freeze()` converts to immutable `Tree<L>`

For model loaders with known structure, `set_numeric_split()` / `set_categorical_split()` allow explicit child indices.

### Leaf Values

The `LeafValue` trait abstracts over output types:

```rust
pub trait LeafValue: Clone + Default + Send + Sync {
    fn accumulate(&mut self, other: &Self);  // Sum predictions
    fn scale(&mut self, factor: f32);        // Apply learning rate
}
```

**Implementations**:
- `ScalarLeaf(f32)` - single output (regression, binary classification)
- `VectorLeaf { values: Vec<f32> }` - multi-output per leaf (alternative to tree groups)

## Key Types

| Type | Description |
|------|-------------|
| `Forest<L>` | Tree ensemble with group assignments and base scores |
| `Tree<L>` | Immutable SoA tree storage |
| `MutableTree<L>` | Builder for tree construction during training |
| `CategoriesStorage` | Packed bitset storage for categorical splits |
| `ScalarLeaf` | Single f32 leaf value |
| `VectorLeaf` | K-dimensional leaf value |
| `SplitType` | Numeric vs Categorical discriminant |
| `SplitCondition` | Feature index, threshold, default direction |
| `Node<L>` | Enum representation (used by some APIs, not SoA storage) |
| `NodeId` | Node index (u32) |

## Validation

Both `Forest` and `Tree` provide `validate()` methods for structural invariant checks:
- Tree connectivity (no cycles, all nodes reachable)
- Child bounds checking
- Categorical segments alignment
- Tree group assignments in range
- Base score length matches group count
