# RFC-0002: Forest and Tree Structures

- **Status**: Implemented
- **Created**: 2024-11-15
- **Updated**: 2025-01-21
- **Scope**: Tree ensemble representation for GBDT

## Summary

This RFC documents the canonical representation for gradient-boosted decision tree (GBDT) ensembles. The design uses Structure-of-Arrays (SoA) layout for cache-efficient traversal, provides a `TreeView` trait for uniform tree access, and separates mutable construction (`MutableTree`) from immutable inference (`Tree`).

## Design Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Forest<ScalarLeaf>                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ trees: Vec<Tree<L>>                                      ││
│  │ tree_groups: Vec<u32>   // Maps tree → output group      ││
│  │ n_groups: u32           // 1=regression, K=multiclass    ││
│  │ base_score: Vec<f32>    // Per-group base scores         ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
         │
         │ contains
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tree<L: LeafValue>                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ SoA Arrays (all indexed by NodeId):                      ││
│  │   split_indices: Box<[u32]>     // Feature index         ││
│  │   split_thresholds: Box<[f32]>  // Split threshold       ││
│  │   left_children: Box<[u32]>     // Left child NodeId     ││
│  │   right_children: Box<[u32]>    // Right child NodeId    ││
│  │   default_left: Box<[bool]>     // Missing → left?       ││
│  │   is_leaf: Box<[bool]>          // Leaf flag             ││
│  │   leaf_values: Box<[L]>         // Leaf outputs          ││
│  │   split_types: Box<[SplitType]> // Numeric/Categorical   ││
│  │   categories: CategoriesStorage // Categorical bitsets   ││
│  │   leaf_coefficients: LeafCoefficients // Linear leaves   ││
│  │   gains: Option<Box<[f32]>>     // For explainability    ││
│  │   covers: Option<Box<[f32]>>    // For explainability    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
         ▲
         │ implements
         │
┌─────────────────────────────────────────────────────────────┐
│                    trait TreeView                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Minimal read-only interface for tree traversal:          ││
│  │   n_nodes(), is_leaf(), split_index(), split_threshold() ││
│  │   left_child(), right_child(), default_left()            ││
│  │   split_type(), categories(), leaf_value()               ││
│  │   traverse_to_leaf<A: FeatureAccessor>(accessor, row)    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Forest

The `Forest<L>` struct manages a collection of trees with multi-output support:

```rust
pub struct Forest<L: LeafValue> {
    trees: Vec<Tree<L>>,
    tree_groups: Vec<u32>,   // Maps each tree to its output group
    n_groups: u32,           // Number of output groups
    base_score: Vec<f32>,    // Per-group base scores
}

impl<L: LeafValue> Forest<L> {
    pub fn n_trees(&self) -> usize;
    pub fn n_groups(&self) -> u32;
    pub fn trees(&self) -> &[Tree<L>];
    pub fn trees_with_groups(&self) -> impl Iterator<Item = (&Tree<L>, u32)>;
    pub fn base_score(&self) -> &[f32];
    pub fn validate(&self) -> Result<(), ForestValidationError>;
}
```

**Multi-output strategy**: Rather than K-dimensional leaves, forests use K separate tree groups where each group contributes to one output dimension. Trees are assigned to groups via `tree_groups[i]`. Predictions accumulate tree outputs per group, starting from `base_score`.

| `n_groups` | Task |
| ---------- | ---- |
| 1 | Regression or binary classification |
| K (K > 1) | K-class classification (one-vs-all) |

## Tree (Immutable SoA)

`Tree<L>` provides immutable, cache-efficient storage for inference:

```rust
pub struct Tree<L: LeafValue> {
    // Core structure (all arrays indexed by NodeId)
    split_indices: Box<[u32]>,
    split_thresholds: Box<[f32]>,
    left_children: Box<[u32]>,
    right_children: Box<[u32]>,
    default_left: Box<[bool]>,
    is_leaf: Box<[bool]>,
    leaf_values: Box<[L]>,
    split_types: Box<[SplitType]>,
    
    // Categorical support
    categories: CategoriesStorage,
    
    // Linear leaves (RFC-0015)
    leaf_coefficients: LeafCoefficients,
    
    // Explainability (RFC-0022)
    gains: Option<Box<[f32]>>,
    covers: Option<Box<[f32]>>,
}
```

### Node Indexing

- `NodeId` is `u32`, indexing into all arrays
- Node 0 is always the root
- Child indices are local to the tree
- Leaf values are only valid where `is_leaf[node] == true`

### Split Types

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SplitType {
    Numeric,     // go left if value < threshold
    Categorical, // go left if category NOT in bitset
}
```

### Tree Methods

```rust
impl<L: LeafValue> Tree<L> {
    /// Traverse to leaf using any FeatureAccessor.
    fn predict_row(&self, features: &[f32]) -> &L;
    
    /// Batch prediction (adds to predictions buffer).
    fn predict_into<A: FeatureAccessor>(
        &self,
        accessor: &A,
        predictions: &mut [f32],
        parallelism: Parallelism,
    );
    
    /// Linear leaf terms (if present).
    fn leaf_terms(&self, node: NodeId) -> Option<(&[u32], &[f32])>;
    
    /// Structural validation.
    fn validate(&self) -> Result<(), TreeValidationError>;
}
```

## TreeView Trait

The `TreeView` trait provides a read-only interface for tree traversal, implemented by both `Tree<L>` and `MutableTree<L>`:

```rust
pub trait TreeView {
    type LeafValue: LeafValue;
    
    // Node access
    fn n_nodes(&self) -> usize;
    fn is_leaf(&self, node: NodeId) -> bool;
    fn split_index(&self, node: NodeId) -> u32;
    fn split_threshold(&self, node: NodeId) -> f32;
    fn left_child(&self, node: NodeId) -> NodeId;
    fn right_child(&self, node: NodeId) -> NodeId;
    fn default_left(&self, node: NodeId) -> bool;
    fn split_type(&self, node: NodeId) -> SplitType;
    fn categories(&self) -> &CategoriesStorage;
    fn leaf_value(&self, node: NodeId) -> &Self::LeafValue;
    
    // Provided traversal method
    fn traverse_to_leaf<A: FeatureAccessor>(&self, accessor: &A, row: usize) -> NodeId {
        let mut node = 0;
        while !self.is_leaf(node) {
            let feat_idx = self.split_index(node) as usize;
            let fvalue = accessor.get_feature(row, feat_idx);
            
            node = if fvalue.is_nan() {
                if self.default_left(node) { self.left_child(node) }
                else { self.right_child(node) }
            } else {
                match self.split_type(node) {
                    SplitType::Numeric => {
                        if fvalue < self.split_threshold(node) { self.left_child(node) }
                        else { self.right_child(node) }
                    }
                    SplitType::Categorical => {
                        let cat = float_to_category(fvalue);
                        if self.categories().category_goes_right(node, cat) { 
                            self.right_child(node) 
                        } else { 
                            self.left_child(node) 
                        }
                    }
                }
            };
        }
        node
    }
}
```

**Design rationale**: The `TreeView` trait enables generic code that works with both immutable trees (inference) and mutable trees (training-time prediction for gradient updates).

## MutableTree (For Training)

`MutableTree<L>` provides a builder pattern for constructing trees during training:

```rust
pub struct MutableTree<L: LeafValue> {
    // Same arrays as Tree, but Vec for growth
    split_indices: Vec<u32>,
    split_thresholds: Vec<f32>,
    left_children: Vec<u32>,
    right_children: Vec<u32>,
    default_left: Vec<bool>,
    is_leaf: Vec<bool>,
    leaf_values: Vec<L>,
    split_types: Vec<SplitType>,
    categories: CategoriesStorage,
    linear_coefficients: Vec<(NodeId, LinearTerms)>,
}

impl<L: LeafValue> MutableTree<L> {
    /// Create empty tree.
    pub fn new() -> Self;
    
    /// Initialize root node (returns NodeId = 0).
    pub fn init_root(&mut self) -> NodeId;
    
    /// Pre-allocate with known node count.
    pub fn init_root_with_n_nodes(&mut self, n_nodes: usize) -> NodeId;
    
    /// Apply a numeric split, allocating children.
    pub fn apply_numeric_split(
        &mut self, node: NodeId, feature: u32, threshold: f32, default_left: bool
    ) -> (NodeId, NodeId);
    
    /// Apply a categorical split.
    pub fn apply_categorical_split(
        &mut self, node: NodeId, feature: u32, categories_right: &[u32], default_left: bool
    ) -> (NodeId, NodeId);
    
    /// Mark node as leaf with value.
    pub fn make_leaf(&mut self, node: NodeId, value: L);
    
    /// Set linear terms for a leaf (RFC-0015).
    pub fn set_linear_leaf(&mut self, node: NodeId, features: Vec<u32>, 
                           intercept: f32, coefficients: Vec<f32>);
    
    /// Convert to immutable Tree.
    pub fn freeze(self) -> Tree<L>;
}

impl<L: LeafValue> TreeView for MutableTree<L> { ... }
```

**Workflow**:
1. `init_root()` or `init_root_with_n_nodes()` for pre-allocated trees
2. Recursive `apply_*_split()` calls to grow the tree
3. `make_leaf()` to finalize leaf nodes
4. `freeze()` to convert to immutable `Tree<L>`

## Categorical Split Storage

Categorical splits use packed u32 bitsets for memory efficiency:

```rust
pub struct CategoriesStorage {
    categories: Box<[u32]>,        // Flat bitset data (32 categories per word)
    segments: Box<[(u32, u32)]>,   // Per-node (start, size) into categories
}

impl CategoriesStorage {
    /// Check if category goes right (bit is set).
    pub fn category_goes_right(&self, node: NodeId, category: u32) -> bool;
    
    /// Empty storage for numeric-only trees.
    pub fn empty() -> Self;
}
```

**Semantics**: Bit `c` set in the node's bitset means category `c` goes RIGHT. This matches XGBoost's partition-based format.

## Leaf Values

The `LeafValue` trait abstracts over output types:

```rust
pub trait LeafValue: Clone + Default + Send + Sync {
    fn accumulate(&mut self, other: &Self);  // Sum predictions
    fn scale(&mut self, factor: f32);        // Apply learning rate
}
```

**Implementations**:

| Type | Use Case |
| ---- | -------- |
| `ScalarLeaf(f32)` | Regression, binary classification, per-group trees |
| `VectorLeaf { values: Vec<f32> }` | Multi-output per leaf (alternative to tree groups) |

## Linear Leaves (RFC-0015)

Trees can have linear models at leaf nodes for improved accuracy:

```rust
pub struct LeafCoefficients {
    terms: Vec<LinearTerms>,           // Sparse, only for leaves with linear models
    leaf_indices: HashMap<NodeId, usize>,
}

pub struct LinearTerms {
    feature_indices: Vec<u32>,
    intercept: f32,
    coefficients: Vec<f32>,
}
```

Accessed via `Tree::leaf_terms(node)` which returns `Option<(&[u32], &[f32])>`.

## Explainability Support (RFC-0022)

Trees can optionally store per-node statistics for feature importance and SHAP:

```rust
impl<L: LeafValue> Tree<L> {
    pub fn with_stats(self, gains: Vec<f32>, covers: Vec<f32>) -> Self;
    pub fn gains(&self) -> Option<&[f32]>;
    pub fn covers(&self) -> Option<&[f32]>;
}
```

- **gains**: Information gain at each split node (0 for leaves)
- **covers**: Sum of hessians for samples reaching each node

## Key Types Summary

| Type | Description |
| ---- | ----------- |
| `Forest<L>` | Tree ensemble with group assignments and base scores |
| `Tree<L>` | Immutable SoA tree storage for inference |
| `MutableTree<L>` | Mutable tree for construction during training |
| `TreeView` | Read-only trait for uniform tree access |
| `CategoriesStorage` | Packed bitset storage for categorical splits |
| `LeafCoefficients` | Linear model terms at leaf nodes |
| `ScalarLeaf` | Single f32 leaf value |
| `VectorLeaf` | K-dimensional leaf value |
| `SplitType` | Numeric vs Categorical discriminant |
| `NodeId` | Node index (u32) |

## Validation

Both `Forest` and `Tree` provide `validate()` methods:

```rust
pub enum TreeValidationError {
    EmptyTree,
    ChildOutOfBounds { node, side, child, n_nodes },
    SelfLoop { node },
    DuplicateVisit { node },
    CycleDetected { node },
    UnreachableNode { node },
    CategoricalSegmentsLenMismatch { segments_len, n_nodes },
}

pub enum ForestValidationError {
    EmptyForest,
    TreeGroupOutOfBounds { tree_idx, group, n_groups },
    BaseScoreLengthMismatch { expected, got },
    TreeValidationFailed { tree_idx, error },
}
```

## Design Decisions

### DD-1: SoA over AoS

**Context**: Store nodes as array-of-structs or struct-of-arrays?

**Decision**: SoA layout.

**Rationale**:
- Cache-friendly for traversal (accessing `is_leaf`, `split_index`, `split_threshold` in sequence)
- SIMD-friendly for batch operations
- Standard approach in high-performance tree implementations

### DD-2: Separate Tree and MutableTree

**Context**: Single mutable struct or separate types for construction vs inference?

**Decision**: Separate `MutableTree` and `Tree` types.

**Rationale**:
- `Tree` can use `Box<[T]>` (no reallocation overhead, smaller)
- `MutableTree` uses `Vec<T>` for growth
- Clear ownership: grower produces `MutableTree`, freezes to `Tree`
- Training can use `MutableTree` directly via `TreeView`

**Usage guideline**:
- Use `Tree<L>` for inference (immutable, cache-optimal)
- Use `MutableTree<L>` only during tree construction in training
- Call `freeze()` when tree construction is complete

### DD-3: TreeView Trait

**Context**: How to share traversal code between `Tree` and `MutableTree`?

**Decision**: `TreeView` trait with provided `traverse_to_leaf` method.

**Rationale**:
- Generic prediction code works with either type
- Training can predict on in-progress tree without freezing
- Single implementation of traversal logic

### DD-4: Per-Tree Groups vs Vector Leaves

**Context**: Multi-output via K-dimensional leaves or K separate tree groups?

**Decision**: Per-tree groups (each tree assigned to one output).

**Rationale**:
- Matches XGBoost/LightGBM approach
- Simpler tree structure
- Better parallelism (trees can be processed independently)
- `ScalarLeaf` is simpler than `VectorLeaf`

## Integration

| Component | Integration Point |
| --------- | ----------------- |
| RFC-0001 (Data) | `TreeView::traverse_to_leaf<A: FeatureAccessor>` |
| RFC-0003 (Inference) | `Predictor` iterates forest, accumulates per group |
| RFC-0007 (Growing) | `TreeGrower` produces `MutableTree`, freezes to `Tree` |
| RFC-0012 (Compat) | XGBoost/LightGBM loaders build `Tree` from model files |
| RFC-0015 (Linear) | `LeafCoefficients` stored in `Tree` |
| RFC-0022 (Explain) | `gains`/`covers` optional statistics (Draft) |

**Note on explainability fields**: The `gains` and `covers` fields in `Tree` are optional (`Option<Box<[f32]>>`). They are only populated when:
1. Training explicitly computes and stores them
2. Loading from XGBoost/LightGBM models that include gain/cover statistics

These fields are not required for inference. See RFC-0022 for explainability features.

## Changelog

- 2025-01-21: Major rewrite. Removed deprecated `Node`/`SplitCondition` enums. Added `TreeView` trait. Documented `MutableTree`. Added linear leaves and explainability support. Updated terminology (`n_nodes`, `n_groups`).
- 2024-11-15: Initial RFC with SoA tree design
