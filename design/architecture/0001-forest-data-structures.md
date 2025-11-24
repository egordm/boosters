# RFC-0001: Forest Data Structures

- **Status**: Draft
- **Created**: 2024-11-24
- **Depends on**: RFC-0002 (Tree Data Structures)
- **Scope**: Forest-level data structures for training and inference

## Summary

This RFC defines the forest-level data structures that hold collections of trees. We establish two primary representations:

1. **`NodeForest`**: Mutable, AoS-based, node-oriented representation
2. **`SoAForest`**: Immutable, SoA-based, optimized for inference

## Motivation

A gradient boosting model is an ensemble of trees. The forest container must:

- Support efficient tree addition during training
- Enable fast batch prediction during inference
- Handle multi-target/multi-class scenarios with tree groups
- Support conversion between training and inference layouts

XGBoost uses a single `GBTreeModel` for both training and inference, with transient views for optimization. We choose explicit separate types for clarity and to enable more aggressive inference optimizations.

## Design

### Type Hierarchy

```text
                                    ┌─────────────────────────────┐
                                    │     Forest<L: LeafValue>    │
                                    │         (trait)             │
                                    └─────────────┬───────────────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    ▼                             ▼                             ▼
        ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
        │    NodeForest     │         │     SoAForest     │         │   PackedForest    │
        │       <L>         │────────▶│        <L>        │────────▶│       <L>         │
        │   (mutable AoS)   │ freeze  │  (immutable SoA)  │  pack   │   (GPU-ready)     │
        └───────────────────┘         └───────────────────┘         └───────────────────┘
```

### LeafValue Trait

The `LeafValue` associated type enables different leaf representations:

```rust
/// Trait for leaf value types
pub trait LeafValue: Clone + Default + Send + Sync {
    /// Number of output dimensions (1 for scalar, N for vector)
    fn output_dim(&self) -> usize;
    
    /// Accumulate this leaf into an output buffer
    fn accumulate(&self, output: &mut [f32]);
    
    /// Create from a slice (for deserialization)
    fn from_slice(data: &[f32]) -> Self;
    
    /// Write to a slice (for serialization)
    fn to_slice(&self, output: &mut [f32]);
}

/// Scalar leaf value (standard regression/binary classification)
#[derive(Clone, Copy, Default)]
pub struct ScalarLeaf(pub f32);

impl LeafValue for ScalarLeaf {
    fn output_dim(&self) -> usize { 1 }
    
    fn accumulate(&self, output: &mut [f32]) {
        output[0] += self.0;
    }
    
    fn from_slice(data: &[f32]) -> Self {
        ScalarLeaf(data[0])
    }
    
    fn to_slice(&self, output: &mut [f32]) {
        output[0] = self.0;
    }
}

/// Vector leaf value (multi-target regression)
#[derive(Clone)]
pub struct VectorLeaf<const N: usize>(pub [f32; N]);

impl<const N: usize> LeafValue for VectorLeaf<N> {
    fn output_dim(&self) -> usize { N }
    
    fn accumulate(&self, output: &mut [f32]) {
        for (o, &v) in output.iter_mut().zip(&self.0) {
            *o += v;
        }
    }
    // ...
}

/// Dynamic vector leaf (when N not known at compile time)
#[derive(Clone)]
pub struct DynVectorLeaf(pub Vec<f32>);

// Future: LinearLeaf for LightGBM-style linear models in leaves
```

### NodeForest (AoS, Mutable)

```rust
/// Forest with explicit node-based tree representation (AoS layout)
/// 
/// This is a mutable, growable representation suitable for:
/// - Training (tree growing)
/// - Model inspection and modification
/// - Serialization/deserialization
/// 
/// Name rationale: "Node" describes the layout (explicit nodes in AoS),
/// not the use case. This avoids semantic lock-in.
pub struct NodeForest<L: LeafValue = ScalarLeaf> {
    /// Trees in the ensemble
    trees: Vec<NodeTree<L>>,
    
    /// Which output group each tree belongs to (for multi-class)
    /// trees[i] contributes to output group tree_groups[i]
    tree_groups: Vec<u32>,
    
    /// Number of output groups (1 for regression, K for K-class)
    num_groups: u32,
    
    /// Base score added to all predictions
    base_score: Vec<f32>,  // Length = num_groups
    
    /// Feature metadata
    num_features: u32,
    
    /// Categorical feature info (which features are categorical)
    categorical_features: Vec<u32>,
    
    /// Training statistics (optional, for analysis)
    stats: Option<TrainStats>,
}

impl<L: LeafValue> NodeForest<L> {
    /// Create a new empty forest
    pub fn new(num_features: u32, num_groups: u32) -> Self;
    
    /// Add a tree to the forest
    pub fn push_tree(&mut self, tree: NodeTree<L>, group: u32);
    
    /// Get a mutable reference to the last tree (for growing)
    pub fn last_tree_mut(&mut self) -> Option<&mut NodeTree<L>>;
    
    /// Convert to inference-optimized layout
    pub fn freeze(self) -> SoAForest<L>;
    
    /// Number of trees
    pub fn num_trees(&self) -> usize;
    
    /// Predict a single row (for training validation)
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32>;
}
```

### SoAForest (SoA, Immutable)

```rust
/// Forest optimized for inference (Structure-of-Arrays)
pub struct SoAForest<L: LeafValue = ScalarLeaf> {
    /// SoA tree storage (see RFC-0002)
    trees: SoATreeStorage<L>,
    
    /// Tree group assignments
    tree_groups: Box<[u32]>,
    
    /// Number of output groups
    num_groups: u32,
    
    /// Base score per group
    base_score: Box<[f32]>,
    
    /// Feature count
    num_features: u32,
    
    /// Categorical split data (shared across trees)
    categorical_splits: CategoricalSplitStorage,
}

/// Stores all trees in SoA layout
pub struct SoATreeStorage<L: LeafValue> {
    /// Tree boundaries: tree i spans nodes tree_offsets[i]..tree_offsets[i+1]
    tree_offsets: Box<[u32]>,
    
    // ─────────────────────────────────────────────────
    // Node arrays (SoA layout, indexed by global node id)
    // ─────────────────────────────────────────────────
    
    /// Feature index for split (0 for leaves)
    split_index: Box<[u32]>,
    
    /// Split threshold (f32::NAN for leaves)
    split_threshold: Box<[f32]>,
    
    /// Left child offset (relative to tree start), 0 for leaves
    left_child: Box<[u32]>,
    
    /// Right child offset, 0 for leaves
    right_child: Box<[u32]>,
    
    /// Default direction: true = go left when missing
    default_left: BitVec,
    
    /// Is this node a leaf?
    is_leaf: BitVec,
    
    // ─────────────────────────────────────────────────
    // Leaf values (separate storage for cache efficiency)
    // ─────────────────────────────────────────────────
    
    /// Leaf value storage (layout depends on L)
    leaf_values: LeafStorage<L>,
    
    /// Mapping from node index to leaf index (only for leaf nodes)
    /// This indirection allows compact leaf storage
    node_to_leaf: Box<[u32]>,
}
```

### Memory Layout Visualization

```text
NodeForest (AoS)                       SoAForest (SoA)
─────────────────                      ─────────────────

trees: Vec<NodeTree>                   tree_offsets: [0, 7, 15, 22, ...]
  │                                              │
  ├─ Tree 0                            split_index:    [f2, f0, f1, 0, 0, f3, 0, f1, ...]
  │   └─ nodes: Vec<Node>                              ├─tree 0──────────┤├─tree 1──...
  │       ├─ Node { split: f2, ... }   
  │       ├─ Node { split: f0, ... }   split_threshold: [0.5, 0.3, 0.7, NaN, NaN, ...]
  │       └─ ...                       
  │                                    left_child:     [1, 3, 5, 0, 0, 8, 0, ...]
  ├─ Tree 1                            
  │   └─ nodes: Vec<Node>              right_child:    [2, 4, 6, 0, 0, 9, 0, ...]
  │       └─ ...                       
  │                                    is_leaf:        [0, 0, 0, 1, 1, 0, 1, ...]
  └─ ...                               
                                       leaf_values:    [0.1, -0.2, 0.3, ...]  (compact)
                                                        ├─only leaves────────┤
```

### Categorical Split Storage

```rust
/// CSR-like storage for categorical splits across all trees
pub struct CategoricalSplitStorage {
    /// Which nodes have categorical splits
    /// Key: global node index, Value: index into cat_offsets
    categorical_nodes: HashMap<u32, u32>,
    
    /// Bitset data (packed u64s)
    bitset_data: Box<[u64]>,
    
    /// Offsets into bitset_data for each categorical node
    /// Node i's bitset is bitset_data[cat_offsets[i]..cat_offsets[i+1]]
    cat_offsets: Box<[u32]>,
}

impl CategoricalSplitStorage {
    /// Check if category is in the "go left" set for a node
    pub fn contains(&self, node_idx: u32, category: u32) -> Option<bool> {
        let cat_idx = self.categorical_nodes.get(&node_idx)?;
        let start = self.cat_offsets[*cat_idx as usize] as usize;
        let end = self.cat_offsets[*cat_idx as usize + 1] as usize;
        let bitset = &self.bitset_data[start..end];
        
        let word_idx = (category / 64) as usize;
        let bit_idx = category % 64;
        Some(bitset.get(word_idx).map_or(false, |w| (w >> bit_idx) & 1 == 1))
    }
}
```

### Conversion: NodeForest → SoAForest

```rust
impl<L: LeafValue> NodeForest<L> {
    pub fn freeze(self) -> SoAForest<L> {
        // 1. Count total nodes across all trees
        let total_nodes: usize = self.trees.iter()
            .map(|t| t.num_nodes())
            .sum();
        
        // 2. Allocate SoA arrays
        let mut split_index = Vec::with_capacity(total_nodes);
        let mut split_threshold = Vec::with_capacity(total_nodes);
        let mut left_child = Vec::with_capacity(total_nodes);
        let mut right_child = Vec::with_capacity(total_nodes);
        let mut default_left = BitVec::with_capacity(total_nodes);
        let mut is_leaf = BitVec::with_capacity(total_nodes);
        let mut tree_offsets = Vec::with_capacity(self.trees.len() + 1);
        let mut leaf_values = Vec::new();
        let mut node_to_leaf = vec![0u32; total_nodes];
        
        // 3. Flatten trees into SoA
        let mut node_offset = 0u32;
        for tree in &self.trees {
            tree_offsets.push(node_offset);
            for (local_idx, node) in tree.nodes().iter().enumerate() {
                let global_idx = node_offset + local_idx as u32;
                
                split_index.push(node.split_index());
                split_threshold.push(node.split_threshold());
                default_left.push(node.default_left());
                
                if node.is_leaf() {
                    is_leaf.push(true);
                    left_child.push(0);
                    right_child.push(0);
                    node_to_leaf[global_idx as usize] = leaf_values.len() as u32;
                    leaf_values.push(node.leaf_value().clone());
                } else {
                    is_leaf.push(false);
                    // Store relative offsets within tree
                    left_child.push(node.left_child() as u32);
                    right_child.push(node.right_child() as u32);
                }
            }
            node_offset += tree.num_nodes() as u32;
        }
        tree_offsets.push(node_offset);
        
        // 4. Build categorical split storage
        let categorical_splits = self.build_categorical_storage();
        
        SoAForest {
            trees: SoATreeStorage {
                tree_offsets: tree_offsets.into_boxed_slice(),
                split_index: split_index.into_boxed_slice(),
                split_threshold: split_threshold.into_boxed_slice(),
                left_child: left_child.into_boxed_slice(),
                right_child: right_child.into_boxed_slice(),
                default_left,
                is_leaf,
                leaf_values: LeafStorage::from_vec(leaf_values),
                node_to_leaf: node_to_leaf.into_boxed_slice(),
            },
            tree_groups: self.tree_groups.into_boxed_slice(),
            num_groups: self.num_groups,
            base_score: self.base_score.into_boxed_slice(),
            num_features: self.num_features,
            categorical_splits,
        }
    }
}
```

### Forest Trait

```rust
/// Common interface for all forest types
pub trait Forest {
    type Leaf: LeafValue;
    
    /// Number of trees in the forest
    fn num_trees(&self) -> usize;
    
    /// Number of output groups
    fn num_groups(&self) -> u32;
    
    /// Number of features expected
    fn num_features(&self) -> u32;
    
    /// Base score per group
    fn base_score(&self) -> &[f32];
    
    /// Get tree group assignment
    fn tree_group(&self, tree_idx: usize) -> u32;
}

impl<L: LeafValue> Forest for NodeForest<L> {
    type Leaf = L;
    // ... implementations
}

impl<L: LeafValue> Forest for SoAForest<L> {
    type Leaf = L;
    // ... implementations
}
```

## Container Variants

For different ownership needs, we provide type aliases:

```rust
/// Owned, growable (node-based, for training or modification)
pub type NodeForestVec<L> = NodeForest<L>;

/// Owned, fixed-size (inference)
pub type SoAForestOwned<L> = SoAForest<L>;

/// Shared, thread-safe (serving)
pub type SoAForestArc<L> = Arc<SoAForest<L>>;

/// Borrowed view (zero-copy)
pub type SoAForestRef<'a, L> = &'a SoAForest<L>;
```

## GPU Extension Point

```rust
/// GPU-ready forest (future, behind feature flag)
#[cfg(feature = "gpu")]
pub struct PackedGpuForest<L: LeafValue> {
    /// Device-side SoA storage
    device_trees: DeviceBuffer<SoATreeStorage<L>>,
    
    /// Host-side metadata (for orchestration)
    host_meta: ForestMetadata,
}

#[cfg(feature = "gpu")]
impl<L: LeafValue> SoAForest<L> {
    /// Upload to GPU
    pub fn to_device(&self, device: &Device) -> PackedGpuForest<L>;
}
```

## Design Decisions

This section records architectural decisions with rationale. Decisions marked **[DECIDED]** are settled; **[OPEN]** require further analysis.

### DD-1: Container Type for SoA Arrays — `Box<[T]>` vs `Vec<T>` **[DECIDED]**

**Decision**: Use `Box<[T]>` for immutable SoA storage.

**Rationale**:

- `Box<[T]>` signals immutability at the type level (no `push`, `pop`, etc.)
- Same memory layout as `Vec<T>` (pointer + length), no capacity overhead
- Enables `&[T]` slicing without additional indirection
- Conversion: `vec.into_boxed_slice()` is O(1) if capacity == length

**Trade-offs**:

- Cannot grow after creation (by design for inference)
- Slight ergonomic cost vs `Vec` (less familiar to some Rust users)

**Alternatives considered**:

- `Vec<T>`: Keeps capacity field (8 bytes wasted), allows accidental mutation
- `Arc<[T]>`: Use when sharing across threads; adds refcount overhead
- Arena allocation: Defer to future optimization; adds lifetime complexity

### DD-2: Node Indexing — Relative vs Absolute **[DECIDED]**

**Decision**: Use **relative** (tree-local) indices with `u32` type.

**Analysis** (practical limits):

- Max tree depth in practice: ~15-20 (deeper is overfitting)
- Max nodes per tree: 2^20 ≈ 1M nodes (extreme case)
- Max trees in forest: ~10,000 (typical: 100-1000)
- Global index range: up to ~10B nodes theoretically

**Rationale**:

- Relative indices keep values small (fit in u16 for most trees)
- Tree-local indexing enables independent tree processing
- Each `SoATreeView` stores `node_offset`, making absolute calculation O(1)
- XGBoost uses relative indices within trees

**Index Type Considerations**:

```rust
/// Node index within a tree (relative)
pub type NodeIdx = u32;  // u16 would limit trees to 65K nodes

/// Tree index within a forest
pub type TreeIdx = u32;  // u16 would limit to 65K trees

/// Leaf index (for separate leaf storage)
pub type LeafIdx = u32;

/// Feature index
pub type FeatureIdx = u32;  // Could be u16 for most datasets
```

**Future**: Consider a separate RFC for index newtypes if we want stronger typing.

### DD-3: Leaf Storage — Interleaved vs Separate **[DECIDED]**

**Decision**: Use **separate** leaf storage with indirection.

**Rationale**:

- Enables compact leaf arrays (no wasted space for internal nodes)
- Better for SIMD: leaf values are contiguous for vectorized accumulation
- Required for variable-size leaves (vector leaves)
- XGBoost also uses separate leaf storage in the SoA prediction path

**Trade-off**:

- Extra indirection via `node_to_leaf` mapping
- For small trees (< 15 nodes), interleaved might be faster (cache line)

**Mitigation**: For scalar leaves, the mapping lookup is a single array access; the leaf value itself is also a single access. Two sequential accesses to likely-cached memory is acceptable.

### DD-4: Categorical Node Lookup **[DECIDED]**

**Decision**: Use **sorted array with binary search** for default; allow HashMap for high-cardinality cases.

**Analysis**:

- Typical case: 5-20% of nodes have categorical splits
- Category count per split: usually < 100

**Rationale**:

- Sorted array: O(log n) lookup, cache-friendly, no hashing overhead
- For n < 50 nodes with categorical splits, binary search is fast
- HashMap adds per-lookup overhead (hashing, potential cache miss)

**Implementation**:

```rust
pub struct CategoricalSplitStorage {
    /// Sorted by node_idx for binary search
    categorical_nodes: Box<[u32]>,
    // ... bitset storage
}

impl CategoricalSplitStorage {
    pub fn contains(&self, node_idx: u32, category: u32) -> Option<bool> {
        // Binary search for node, then bitset lookup
        let pos = self.categorical_nodes.binary_search(&node_idx).ok()?;
        // ... bitset check
    }
}
```

**Extension point**: Add a `CategoricalLookup` trait if we need HashMap variant later.

### DD-5: Index Type Abstraction **[OPEN]**

**Question**: Should we define newtype wrappers for different index semantics?

```rust
#[repr(transparent)]
pub struct NodeIdx(u32);

#[repr(transparent)]
pub struct FeatureIdx(u32);

#[repr(transparent)]
pub struct BinIdx(u16);  // For quantized features

#[repr(transparent)]
pub struct CategoryIdx(u32);
```

**Pros**:

- Type safety: Can't accidentally mix node index with feature index
- Documentation: Types are self-documenting
- Zero-cost: `#[repr(transparent)]` guarantees same layout as inner type

**Cons**:

- Boilerplate: Need `From`, `Into`, arithmetic ops
- Slice indexing: `slice[idx.0 as usize]` is less ergonomic than `slice[idx]`
- May complicate generic code

**Recommendation**: Defer to a separate RFC. Use type aliases initially (`type NodeIdx = u32`), migrate to newtypes if we find bugs from index confusion.

## Open Questions

1. **Tree traversal methods**: Should `NodeForest` have direct `predict_row` or require explicit traversal visitor? Currently included for convenience but could be removed for purity.

2. **Lazy freezing**: Should `freeze()` be lazy (on first predict) or eager (explicit call)? Currently eager for clarity.

## Alternatives Considered

### Single Unified Type

A single `Forest<Layout>` generic over layout type:

```rust
struct Forest<L: LeafValue, Layout: ForestLayout> {
    layout: Layout,
    metadata: ForestMeta,
}
```

**Rejected**: Adds complexity without clear benefit. The two layouts have very different invariants (mutable vs immutable).

### Copy-on-Write

Use `Cow` for trees to enable lazy conversion:

```rust
enum ForestCow<L> {
    Train(NodeForest<L>),
    Soa(SoAForest<L>),
}
```

**Deferred**: Could be added later if needed for interactive workflows.

## References

- XGBoost `GBTreeModel`: `src/gbm/gbtree_model.h`
- RFC-0002: Tree Data Structures
- [design/analysis/design_challenges_and_tradeoffs.md](../analysis/design_challenges_and_tradeoffs.md) §2
