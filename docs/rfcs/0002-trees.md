# RFC-0002: Tree and Forest Representation

**Status**: Implemented  
**Created**: 2025-12-15  
**Updated**: 2026-01-02  
**Scope**: GBDT model structure

## Summary

Trees use Structure-of-Arrays (SoA) layout for cache-efficient traversal.
`Tree<L>` is immutable for inference; `MutableTree<L>` supports construction.
`Forest<L>` manages collections of trees with multi-output support.

## Why SoA?

Tree traversal accesses node fields in sequence: is_leaf → split_index →
threshold → children. SoA keeps each field contiguous:

| Layout | Cache Behavior |
| ------ | -------------- |
| Array-of-Structs | Load unused fields (waste bandwidth) |
| Struct-of-Arrays | Sequential access to needed arrays |

**Benchmark**: SoA provides ~15% speedup in inference vs AoS layout due to
better cache line utilization (fewer bytes loaded per node access).

## Layers

### High Level

Users interact with `GBDTModel` which wraps `Forest`:

```rust
let model = GBDTModel::train(&dataset, eval_set, config, seed)?;
let preds = model.predict(&test_data, n_threads);
```

### Medium Level (Forest)

```rust
pub struct Forest<L: LeafValue> {
    trees: Vec<Tree<L>>,
    tree_groups: Vec<u32>,  // Maps tree → output group
    n_groups: u32,          // 1 = regression, K = multiclass
    base_scores: Vec<f32>,  // Per-group initial predictions
}
```

**Multi-output**: Each tree contributes to one output group. For K-class,
trees round-robin: tree 0 → group 0, tree 1 → group 1, etc.

### Tree Access

```rust
impl<L: LeafValue> Forest<L> {
    pub fn n_trees(&self) -> usize;
    pub fn tree(&self, idx: usize) -> &Tree<L>;
    pub fn trees(&self) -> impl Iterator<Item = &Tree<L>>;
    pub fn tree_group(&self, idx: usize) -> u32;
}
```

Individual tree access is useful for debugging, explainability, and analysis.

### Medium Level (Tree)

```rust
pub struct Tree<L: LeafValue> {
    // Core SoA arrays (indexed by NodeId)
    split_indices: Box<[u32]>,
    split_thresholds: Box<[f32]>,
    left_children: Box<[u32]>,
    right_children: Box<[u32]>,
    default_left: Box<[bool]>,
    is_leaf: Box<[bool]>,
    leaf_values: Box<[L]>,
    split_types: Box<[SplitType]>,  // Numeric vs Categorical
    
    // Categorical split storage
    categories: CategoriesStorage,
    
    // Optional for explainability
    gains: Option<Box<[f32]>>,
    covers: Option<Box<[f32]>>,
}
```

**Why `Box<[T]>` not `Vec<T>`?** Trees are immutable after construction.
`Box<[T]>` has smaller stack size (no capacity) and signals immutability.

### Medium Level (MutableTree)

```rust
pub struct MutableTree<L: LeafValue> {
    // Same fields as Tree, but Vec for growth
    split_indices: Vec<u32>,
    // ...
}

impl<L: LeafValue> MutableTree<L> {
    pub fn init_root(&mut self) -> NodeId;
    pub fn apply_split(&mut self, node: NodeId, split: &SplitInfo) -> (NodeId, NodeId);
    pub fn make_leaf(&mut self, node: NodeId, value: L);
    pub fn freeze(self) -> Tree<L>;
}
```

Grower produces `MutableTree`, calls `freeze()` to get immutable `Tree`.

### Low Level (TreeView Trait)

```rust
pub trait TreeView {
    type LeafValue: LeafValue;
    
    fn n_nodes(&self) -> usize;
    fn is_leaf(&self, node: NodeId) -> bool;
    fn split_index(&self, node: NodeId) -> u32;
    fn split_threshold(&self, node: NodeId) -> f32;
    // ... traversal primitives
}
```

Both `Tree` and `MutableTree` implement `TreeView`, enabling generic
traversal code.

## LeafValue Trait

```rust
pub trait LeafValue: Clone + Default + Send + Sync {
    fn accumulate(&mut self, other: &Self);
    fn scale(&mut self, factor: f32);
}

pub struct ScalarLeaf(pub f32);  // Standard case
```

Vector leaves exist but aren't used—multi-output uses tree groups instead.

## Categorical Storage

```rust
pub struct CategoriesStorage {
    categories: Box<[u32]>,       // Packed bitsets
    segments: Box<[(u32, u32)]>,  // Per-node (start, len)
}
```

Bit set = category goes RIGHT. Memory-efficient: only allocate for categorical
split nodes.

## Files

| Path | Contents |
| ---- | -------- |
| `repr/gbdt/tree.rs` | `Tree<L>`, `NodeId` |
| `repr/gbdt/mutable.rs` | `MutableTree<L>` |
| `repr/gbdt/forest.rs` | `Forest<L>` |
| `repr/gbdt/view.rs` | `TreeView` trait |
| `repr/gbdt/categories.rs` | `CategoriesStorage` |

## Design Decisions

**DD-1: Separate Tree and MutableTree.** Clear ownership: grower produces
mutable, freezes to immutable. No runtime mutability checks.

**DD-2: Tree groups over vector leaves.** Each tree has scalar output,
contributes to one group. Simpler, matches XGBoost/LightGBM, better parallelism.

**DD-3: Categories go right.** Convention matches XGBoost. Sorted partition
algorithm (high g/h ratio → right) produces right-going sets naturally.

**DD-4: Optional gains/covers.** Only populated when:
1. Training with `store_node_stats: true` in config
2. Loading XGBoost/LightGBM models that include statistics

Not required for inference; needed for explainability (TreeSHAP, feature importance).

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Tree structure | Valid node indices, no orphans, proper leaf marking |
| Traversal | Reaches correct leaf for all samples |
| Freeze correctness | MutableTree → Tree preserves all data |
| Categorical storage | Bitset encode/decode roundtrip |
| Multi-output | Tree groups correctly assigned |
