# RFC-0002: Tree Data Structures

- **Status**: Draft
- **Created**: 2024-11-24
- **Depends on**: RFC-0001 (Forest Data Structures)
- **Scope**: Individual tree representations and node layouts

## Summary

This RFC defines tree-level data structures for training and inference:
1. **`NodeTree`**: Mutable AoS representation with per-node structs
2. **`SoATreeView`**: Immutable SoA representation as array slices
3. **`ArrayTreeLayout`**: Unrolled top-k levels for block traversal

## Motivation

Individual trees are the atomic unit of the ensemble. The tree structure must:
- Support efficient node addition during training (splits)
- Enable fast traversal during inference
- Handle both numerical and categorical splits
- Support scalar and vector leaves

XGBoost's `RegTree::Node` is a compact 20-byte struct. We aim for similar compactness while enabling SoA transformation.

## Design

### Node Representation (AoS)

```rust
/// Node in a tree (AoS layout)
/// 
/// Size: 24 bytes (padded for alignment)
#[repr(C)]
pub struct Node<L: LeafValue> {
    /// Split feature index, OR marker for leaf/deleted
    /// Bits 0-30: feature index
    /// Bit 31: default_left flag
    split_index_flags: u32,
    
    /// Split threshold for numerical, or category set index for categorical
    /// For leaves: unused (leaf value stored separately)
    split_value: f32,
    
    /// Left child index (INVALID_NODE if leaf)
    left: i32,
    
    /// Right child index (INVALID_NODE if leaf)
    right: i32,
    
    /// Parent node index (INVALID_NODE if root)
    parent: i32,
    
    /// Leaf value (only valid if is_leaf)
    /// For vector leaves, this indexes into a separate leaf array
    leaf_value_or_idx: LeafValueStorage<L>,
}

/// Compact storage for leaf values
pub enum LeafValueStorage<L: LeafValue> {
    /// Scalar leaf: value stored inline
    Scalar(f32),
    /// Vector leaf: index into separate storage
    VectorIndex(u32),
    /// Not a leaf
    NotLeaf,
}

impl<L: LeafValue> Node<L> {
    pub const INVALID_NODE: i32 = -1;
    pub const DELETED_MARKER: u32 = u32::MAX;
    
    /// Create a new leaf node
    pub fn new_leaf(value: L, parent: i32) -> Self;
    
    /// Create a new split node
    pub fn new_split(
        feature: u32,
        threshold: f32,
        default_left: bool,
        left: i32,
        right: i32,
        parent: i32,
    ) -> Self;
    
    /// Is this a leaf node?
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left == Self::INVALID_NODE
    }
    
    /// Is this node deleted?
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.split_index_flags == Self::DELETED_MARKER
    }
    
    /// Get split feature index
    #[inline]
    pub fn split_index(&self) -> u32 {
        self.split_index_flags & 0x7FFF_FFFF
    }
    
    /// Get default direction for missing values
    #[inline]
    pub fn default_left(&self) -> bool {
        (self.split_index_flags >> 31) != 0
    }
    
    /// Get split threshold
    #[inline]
    pub fn split_threshold(&self) -> f32 {
        self.split_value
    }
    
    /// Get left child index
    #[inline]
    pub fn left_child(&self) -> i32 {
        self.left
    }
    
    /// Get right child index
    #[inline]
    pub fn right_child(&self) -> i32 {
        self.right
    }
}
```

### Node Memory Layout

```text
Node<ScalarLeaf> (24 bytes)
┌─────────────────────────────────────────────────────────────────────┐
│ split_index_flags │ split_value │   left   │  right   │  parent  │ │
│      (u32)        │   (f32)     │  (i32)   │  (i32)   │  (i32)   │ │
│    4 bytes        │  4 bytes    │ 4 bytes  │ 4 bytes  │ 4 bytes  │ │
├───────────────────┴─────────────┴──────────┴──────────┴──────────┤ │
│ leaf_value_or_idx (4 bytes)                                       │ │
└─────────────────────────────────────────────────────────────────────┘

split_index_flags bit layout:
┌──────────────────────────────────┬───────────────────────────────┐
│ bit 31: default_left             │ bits 0-30: feature_index      │
└──────────────────────────────────┴───────────────────────────────┘
```

### NodeTree (Mutable, AoS)

```rust
/// Mutable tree with explicit node representation
/// 
/// Named "NodeTree" to describe the layout (AoS nodes), not the use case.
pub struct NodeTree<L: LeafValue = ScalarLeaf> {
    /// Nodes stored in breadth-first order
    nodes: Vec<Node<L>>,
    
    /// Vector leaf values (only used if L is vector-valued)
    vector_leaves: Vec<L>,
    
    /// Number of deleted nodes (for compaction heuristics)
    num_deleted: u32,
    
    /// Maximum depth reached
    max_depth: u32,
    
    /// Categorical split data for this tree
    categorical_splits: TreeCategoricalSplits,
    
    /// Training statistics (optional)
    stats: Option<TreeStats>,
}

/// Training statistics per node
pub struct TreeStats {
    /// Per-node statistics
    node_stats: Vec<NodeStat>,
}

pub struct NodeStat {
    /// Loss reduction from this split
    pub loss_change: f32,
    /// Sum of hessians (coverage)
    pub sum_hess: f32,
    /// Base weight before learning rate
    pub base_weight: f32,
}

impl<L: LeafValue> NodeTree<L> {
    /// Create a tree with just a root leaf
    pub fn new(root_value: L) -> Self {
        Self {
            nodes: vec![Node::new_leaf(root_value, Node::INVALID_NODE)],
            vector_leaves: Vec::new(),
            num_deleted: 0,
            max_depth: 0,
            categorical_splits: TreeCategoricalSplits::new(),
            stats: None,
        }
    }
    
    /// Expand a leaf into a split with two new leaves
    pub fn expand_node(
        &mut self,
        node_idx: usize,
        feature: u32,
        threshold: f32,
        default_left: bool,
        left_value: L,
        right_value: L,
    ) -> (usize, usize) {
        debug_assert!(self.nodes[node_idx].is_leaf());
        
        let left_idx = self.nodes.len();
        let right_idx = left_idx + 1;
        
        // Add new leaf nodes
        self.nodes.push(Node::new_leaf(left_value, node_idx as i32));
        self.nodes.push(Node::new_leaf(right_value, node_idx as i32));
        
        // Convert parent to split node
        let parent = &mut self.nodes[node_idx];
        parent.split_index_flags = feature | ((default_left as u32) << 31);
        parent.split_value = threshold;
        parent.left = left_idx as i32;
        parent.right = right_idx as i32;
        
        // Update max depth
        let depth = self.node_depth(node_idx) + 1;
        self.max_depth = self.max_depth.max(depth);
        
        (left_idx, right_idx)
    }
    
    /// Expand with a categorical split
    pub fn expand_categorical(
        &mut self,
        node_idx: usize,
        feature: u32,
        categories_left: &[u32],  // Categories that go left
        default_left: bool,
        left_value: L,
        right_value: L,
    ) -> (usize, usize);
    
    /// Prune a subtree (convert split back to leaf)
    pub fn prune(&mut self, node_idx: usize, new_value: L);
    
    /// Number of nodes (including deleted)
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Number of leaves
    pub fn num_leaves(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf() && !n.is_deleted()).count()
    }
    
    /// Calculate depth of a node
    fn node_depth(&self, node_idx: usize) -> u32 {
        let mut depth = 0;
        let mut idx = node_idx;
        while self.nodes[idx].parent != Node::<L>::INVALID_NODE {
            idx = self.nodes[idx].parent as usize;
            depth += 1;
        }
        depth
    }
    
    /// Access nodes slice
    pub fn nodes(&self) -> &[Node<L>] {
        &self.nodes
    }
}
```

### SoA Tree View

The `SoATree` is not a standalone type but a **view into `SoATreeStorage`** (from RFC-0001):

```rust
/// View into a single tree within SoATreeStorage
pub struct SoATreeView<'a, L: LeafValue> {
    /// Offset of this tree's first node in the global arrays
    node_offset: u32,
    
    /// Number of nodes in this tree
    num_nodes: u32,
    
    /// Reference to the shared storage
    storage: &'a SoATreeStorage<L>,
}

impl<'a, L: LeafValue> SoATreeView<'a, L> {
    /// Get split feature index for a node
    #[inline]
    pub fn split_index(&self, local_idx: u32) -> u32 {
        self.storage.split_index[(self.node_offset + local_idx) as usize]
    }
    
    /// Get split threshold for a node
    #[inline]
    pub fn split_threshold(&self, local_idx: u32) -> f32 {
        self.storage.split_threshold[(self.node_offset + local_idx) as usize]
    }
    
    /// Is this node a leaf?
    #[inline]
    pub fn is_leaf(&self, local_idx: u32) -> bool {
        self.storage.is_leaf[(self.node_offset + local_idx) as usize]
    }
    
    /// Get left child (local index within tree)
    #[inline]
    pub fn left_child(&self, local_idx: u32) -> u32 {
        self.storage.left_child[(self.node_offset + local_idx) as usize]
    }
    
    /// Get right child (local index within tree)
    #[inline]
    pub fn right_child(&self, local_idx: u32) -> u32 {
        self.storage.right_child[(self.node_offset + local_idx) as usize]
    }
    
    /// Get default direction for missing values
    #[inline]
    pub fn default_left(&self, local_idx: u32) -> bool {
        self.storage.default_left[(self.node_offset + local_idx) as usize]
    }
    
    /// Get leaf value for a leaf node
    #[inline]
    pub fn leaf_value(&self, local_idx: u32) -> &L {
        debug_assert!(self.is_leaf(local_idx));
        let leaf_idx = self.storage.node_to_leaf[(self.node_offset + local_idx) as usize];
        self.storage.leaf_values.get(leaf_idx)
    }
    
    /// Traverse to find leaf for given features
    pub fn find_leaf(&self, features: &[f32]) -> &L {
        let mut idx = 0u32;  // Start at root
        
        while !self.is_leaf(idx) {
            let feat_idx = self.split_index(idx) as usize;
            let threshold = self.split_threshold(idx);
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);
            
            idx = if fvalue.is_nan() {
                if self.default_left(idx) {
                    self.left_child(idx)
                } else {
                    self.right_child(idx)
                }
            } else if fvalue < threshold {
                self.left_child(idx)
            } else {
                self.right_child(idx)
            };
        }
        
        self.leaf_value(idx)
    }
}
```

### ArrayTreeLayout (Unrolled Top Levels)

For block-based traversal, we unroll the top K levels into a perfect binary tree array layout:

```rust
/// Unrolled top-K levels of a tree for efficient block traversal
/// 
/// Layout: Perfect binary tree in array form
/// - Node 0: root (level 0)
/// - Nodes 1-2: level 1 (left, right children of root)
/// - Nodes 3-6: level 2
/// - ...
/// - Nodes (2^K - 1) - 1: pointers to subtrees at level K
pub struct ArrayTreeLayout<const DEPTH: usize = 6> {
    /// Number of nodes in unrolled section: 2^DEPTH - 1
    const NUM_NODES: usize = (1 << DEPTH) - 1;
    
    /// Split feature indices (0 for passthrough nodes)
    split_index: [u32; Self::NUM_NODES],
    
    /// Split thresholds (NaN forces right for passthrough)
    split_threshold: [f32; Self::NUM_NODES],
    
    /// Default direction for missing (false = right)
    default_left: [bool; Self::NUM_NODES],
    
    /// Is this a categorical split?
    is_categorical: [bool; Self::NUM_NODES],
    
    /// For nodes at level DEPTH-1, index into original tree
    /// Length: 2^DEPTH (the "exit" nodes)
    subtree_roots: [u32; 1 << DEPTH],
}

impl<const DEPTH: usize> ArrayTreeLayout<DEPTH> {
    /// Build from a SoATreeView
    pub fn from_tree<L: LeafValue>(tree: &SoATreeView<'_, L>) -> Self {
        let mut layout = Self {
            split_index: [0; Self::NUM_NODES],
            split_threshold: [f32::NAN; Self::NUM_NODES],  // NaN → go right
            default_left: [false; Self::NUM_NODES],
            is_categorical: [false; Self::NUM_NODES],
            subtree_roots: [0; 1 << DEPTH],
        };
        
        // Recursively populate from tree
        layout.populate(tree, 0, 0, 0);
        layout
    }
    
    fn populate<L: LeafValue>(
        &mut self,
        tree: &SoATreeView<'_, L>,
        array_idx: usize,   // Index in our array
        tree_idx: u32,      // Index in the tree
        depth: usize,
    ) {
        if depth == DEPTH {
            // Store subtree root for continuation
            self.subtree_roots[array_idx - Self::NUM_NODES] = tree_idx;
            return;
        }
        
        if tree.is_leaf(tree_idx) {
            // Leaf before max depth: fill with passthrough
            // NaN threshold means always go right, which hits same leaf
            self.split_threshold[array_idx] = f32::NAN;
            self.subtree_roots[/* calculate exit index */] = tree_idx;
            // Recursively fill children with same pattern
            self.populate(tree, 2 * array_idx + 1, tree_idx, depth + 1);
            self.populate(tree, 2 * array_idx + 2, tree_idx, depth + 1);
        } else {
            // Copy split info
            self.split_index[array_idx] = tree.split_index(tree_idx);
            self.split_threshold[array_idx] = tree.split_threshold(tree_idx);
            self.default_left[array_idx] = tree.default_left(tree_idx);
            
            // Recurse to children
            self.populate(tree, 2 * array_idx + 1, tree.left_child(tree_idx), depth + 1);
            self.populate(tree, 2 * array_idx + 2, tree.right_child(tree_idx), depth + 1);
        }
    }
    
    /// Process a block of rows through the unrolled levels
    /// Returns: for each row, the subtree root index to continue from
    #[inline]
    pub fn process_block(
        &self,
        features: &[&[f32]],    // features[row][feature]
        results: &mut [u32],    // Output: subtree indices
    ) {
        debug_assert_eq!(features.len(), results.len());
        
        // Initialize all rows at root
        for r in results.iter_mut() {
            *r = 0;
        }
        
        // Process level by level
        for level in 0..DEPTH {
            let level_start = (1 << level) - 1;
            
            for (row_idx, (feats, result)) in features.iter().zip(results.iter_mut()).enumerate() {
                let array_idx = level_start + *result as usize;
                
                let feat_idx = self.split_index[array_idx] as usize;
                let threshold = self.split_threshold[array_idx];
                let fvalue = feats.get(feat_idx).copied().unwrap_or(f32::NAN);
                
                let go_left = if fvalue.is_nan() {
                    self.default_left[array_idx]
                } else {
                    fvalue < threshold
                };
                
                // Update position for next level
                // In perfect binary tree: left = 2*pos, right = 2*pos + 1
                *result = 2 * *result + (!go_left as u32);
            }
        }
        
        // Convert final positions to subtree root indices
        for result in results.iter_mut() {
            *result = self.subtree_roots[*result as usize];
        }
    }
}
```

### Array Layout Visualization

```
Original Tree (depth 4):            ArrayTreeLayout<3> (unroll 3 levels):

        [0]                                    Array indices:
       /   \                                        [0]          Level 0
      /     \                                      /   \
    [1]     [2]                                 [1]     [2]      Level 1  
    / \     / \                                 / \     / \
  [3] [4] [5] [6]                            [3] [4] [5] [6]    Level 2
  /\  /\  /\  /\                              │   │   │   │
 7 8 9 10...                                  ▼   ▼   ▼   ▼
                                          subtree_roots[0..8]:
                                          Points back to nodes 3,4,5,6
                                          (or their children if they exist)

Memory layout (DEPTH=3, 7 nodes + 8 subtree pointers):

split_index:     [f0, f1, f2, f3, f4, f5, f6]
split_threshold: [0.5, 0.3, 0.7, 0.2, 0.4, 0.6, 0.8]
default_left:    [T,   F,   T,   F,   F,   T,   F]
subtree_roots:   [7, 8, 9, 10, 11, 12, 13, 14]  (continue from here)
```

### SIMD-Friendly Operations

```rust
#[cfg(feature = "simd")]
mod simd {
    use std::simd::{f32x8, mask32x8, Simd};
    
    impl<const DEPTH: usize> ArrayTreeLayout<DEPTH> {
        /// Process 8 rows simultaneously through one level
        #[inline]
        pub fn process_level_simd(
            &self,
            level: usize,
            features: &[f32x8],     // features[feat_idx] = 8 rows' values
            positions: &mut [u32; 8],
        ) {
            let level_start = (1 << level) - 1;
            
            // Gather split indices and thresholds for each row's current position
            let mut thresholds = [0.0f32; 8];
            let mut feat_indices = [0u32; 8];
            let mut defaults = [false; 8];
            
            for (i, &pos) in positions.iter().enumerate() {
                let array_idx = level_start + pos as usize;
                thresholds[i] = self.split_threshold[array_idx];
                feat_indices[i] = self.split_index[array_idx];
                defaults[i] = self.default_left[array_idx];
            }
            
            // Gather feature values for the 8 rows
            // (This is the tricky part - features might be different per row)
            // For now, assume same feature index for simplicity
            // Real impl needs scatter-gather
            
            let thresh_simd = f32x8::from_array(thresholds);
            // ... comparison and position update
        }
    }
}
```

## Tree Categorical Splits

```rust
/// Categorical split data for a single tree
pub struct TreeCategoricalSplits {
    /// Which nodes have categorical splits (local node indices)
    nodes: Vec<u32>,
    
    /// Bitset data for all categorical nodes
    bitsets: Vec<u64>,
    
    /// Offsets into bitsets for each node in `nodes`
    offsets: Vec<u32>,
}

impl TreeCategoricalSplits {
    /// Add a categorical split for a node
    pub fn add_split(&mut self, node_idx: u32, categories_left: &[u32]) {
        self.nodes.push(node_idx);
        self.offsets.push(self.bitsets.len() as u32);
        
        // Convert categories to bitset
        let max_cat = categories_left.iter().max().copied().unwrap_or(0);
        let num_words = (max_cat / 64 + 1) as usize;
        
        let start = self.bitsets.len();
        self.bitsets.resize(start + num_words, 0);
        
        for &cat in categories_left {
            let word_idx = (cat / 64) as usize;
            let bit_idx = cat % 64;
            self.bitsets[start + word_idx] |= 1u64 << bit_idx;
        }
    }
    
    /// Check if a category goes left for a node
    pub fn goes_left(&self, node_idx: u32, category: u32) -> Option<bool> {
        let pos = self.nodes.iter().position(|&n| n == node_idx)?;
        let start = self.offsets[pos] as usize;
        let end = self.offsets.get(pos + 1).copied().unwrap_or(self.bitsets.len() as u32) as usize;
        
        let bitset = &self.bitsets[start..end];
        let word_idx = (category / 64) as usize;
        let bit_idx = category % 64;
        
        Some(bitset.get(word_idx).map_or(false, |w| (w >> bit_idx) & 1 == 1))
    }
}
```

## ArrayTreeLayout vs SoATreeView

Both provide tree traversal, but serve different purposes.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                        Tree Representations                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SoATreeView (Dynamic)              ArrayTreeLayout (Static)         │
│  ─────────────────────              ──────────────────────────       │
│                                                                      │
│  • Variable depth trees             • Fixed top-K levels only        │
│  • Full tree access                 • Perfect binary tree array      │
│  • Random node lookup               • Sequential level-by-level      │
│  • Used for: general traversal      • Used for: block processing     │
│                                                                      │
│  Storage: pointers to SoA arrays    Storage: inline fixed arrays     │
│                                                                      │
│       SoATreeView                         ArrayTreeLayout<3>         │
│           │                                                          │
│     ┌─────┴─────┐                         [0][1][2][3][4][5][6]      │
│     │ storage ──┼──▶ SoATreeStorage       └─level─┘└─level─┘└─l3─┘   │
│     │ offset    │                               0       1       2    │
│     │ num_nodes │                                                    │
│     └───────────┘                         Then: subtree_roots[8]     │
│                                           points back to full tree   │
└─────────────────────────────────────────────────────────────────────┘
```

### Comparison

| Aspect | SoATreeView | ArrayTreeLayout |
|--------|-------------|-----------------|
| **Purpose** | General tree access | Block traversal optimization |
| **Storage** | Borrows from `SoATreeStorage` | Owns fixed-size arrays |
| **Depth** | Full tree (any depth) | Top K levels only (e.g., 6) |
| **Memory** | No extra allocation | 2^K - 1 nodes inline |
| **Access pattern** | Random (follow pointers) | Sequential (level-by-level) |
| **Cache behavior** | Depends on tree shape | Predictable, prefetch-friendly |
| **Lifetime** | Ephemeral (created per prediction) | Can be cached |

### When to use ArrayTreeLayout

1. **Block processing**: Process multiple rows through top levels in lockstep
2. **CPU SIMD**: Level-by-level enables vectorized comparisons
3. **Branch prediction**: Predictable access pattern (no data-dependent branches in top levels)

### When to use SoATreeView directly

1. **Deep trees**: When tree depth > K, ArrayTreeLayout only helps partially
2. **Single-row prediction**: Block overhead not justified
3. **Sparse data**: Rows may exit at different levels anyway

### Combined Workflow

```text
Input rows ──▶ ArrayTreeLayout.process_block()
                         │
                         │ (rows reach level K)
                         ▼
              subtree_roots[row_i] = node index
                         │
                         │ (continue from that node)
                         ▼
              SoATreeView.traverse_from(subtree_root, features)
                         │
                         ▼
                    Leaf value
```

### SIMD Optimization Analysis

The main SIMD challenge in tree traversal is the **gather operation**:

```rust
// Each row needs a DIFFERENT feature index
for row in 0..8 {
    let feat_idx = split_index[position[row]];  // Different per row!
    values[row] = features[row][feat_idx];      // Scatter-gather needed
}
```

This requires `_mm256_i32gather_ps` or equivalent, which is slower than sequential loads on current hardware.

**What CAN be vectorized**:

1. The comparison itself: `fval < threshold` across 8 rows
2. Position update: `pos = 2*pos + offset` across 8 rows  
3. Leaf accumulation: Adding 8 leaf values to 8 outputs

**What CANNOT be easily vectorized**:

1. Threshold/index lookup: Different positions → different array indices
2. Feature value lookup: Different feature indices per row
3. Exit conditions: Rows exit tree at different levels

**The gather loop** (bottleneck):

```rust
for (i, &pos) in positions.iter().enumerate() {
    let array_idx = level_start + pos as usize;
    thresholds[i] = self.split_threshold[array_idx];
    feat_indices[i] = self.split_index[array_idx];
}
```

This is **inherently serial** due to different positions. Options:

1. **Accept it**: 8 sequential loads from L1 cache are fast (~4 cycles each)
2. **Hardware gather**: Use `_mm256_i32gather_ps` (benchmark to compare!)
3. **Prefetch**: Hint next level's indices to prefetcher

**Recommendation**: Start with scalar gather loop, benchmark, then try hardware gather. The sequential version often wins for small N due to instruction overhead of gather.

## Design Decisions

This section records architectural decisions with rationale.

### DD-1: Child Index Type — Signed vs Unsigned **[DECIDED]**

**Decision**: Use **unsigned `u32`** for child indices in SoA storage, **signed `i32`** for AoS node storage during construction.

**Rationale**:

For **NodeTree (AoS, mutable)**:

- Signed `i32` allows `-1` as `INVALID_NODE` sentinel (XGBoost pattern)
- Parent pointers need sentinel for root node
- Simplifies node construction logic

For **SoATreeStorage (immutable)**:

- Unsigned `u32` uses all bits effectively (no wasted sign bit)
- Leaf nodes use `is_leaf` bitmap, not sentinel values
- `left_child` and `right_child` are only accessed for non-leaf nodes

```rust
// AoS (mutable, during construction)
pub struct Node<L: LeafValue> {
    left: i32,   // -1 for leaf
    right: i32,  // -1 for leaf  
    parent: i32, // -1 for root
}

// SoA (immutable, for inference)
pub struct SoATreeStorage<L: LeafValue> {
    left_child: Box<[u32]>,  // Only valid where !is_leaf
    right_child: Box<[u32]>, // Only valid where !is_leaf
    is_leaf: BitVec,         // Sentinel replaced by explicit flag
}
```

**Trade-off analysis**:

- Signed: Wastes 1 bit, but convenient sentinel
- Unsigned with `u32::MAX` sentinel: Works but less idiomatic
- Unsigned with separate `is_leaf`: Cleaner, explicit, chosen for SoA

### DD-2: Node Ordering — BFS vs DFS **[DECIDED]**

**Decision**: Use **BFS (breadth-first)** ordering for nodes.

**Rationale**:

- BFS aligns with ArrayTreeLayout (levels are contiguous)
- XGBoost uses BFS ordering
- Enables level-by-level SIMD processing
- Subtree locality matters less than level locality for inference

**Trade-off**: DFS (pre-order) would keep subtrees contiguous, better if we frequently traverse single paths. But block processing dominates inference workload.

### DD-3: Deleted Node Handling **[DECIDED]**

**Decision**: Use **lazy deletion** with compaction on `freeze()`. Optionally track deleted slots for reuse.

**Analysis of XGBoost approach**:
- XGBoost marks deleted nodes but doesn't compact during training
- Compaction happens when serializing or converting to prediction format
- Rationale: Training frequently prunes and re-expands; compaction would be O(n) per prune

**Our approach**:
- `NodeTree`: Track `num_deleted`, mark nodes with `DELETED_MARKER`
- On `freeze()` to `SoAForest`: Compact (skip deleted nodes, renumber indices)
- Inference path never sees deleted nodes

```rust
impl<L: LeafValue> NodeTree<L> {
    pub fn prune(&mut self, node_idx: usize, new_value: L) {
        // Mark subtree as deleted (don't compact)
        self.mark_deleted_recursive(node_idx);
        // Convert node to leaf
        self.nodes[node_idx] = Node::new_leaf(new_value, self.nodes[node_idx].parent);
    }
    
    fn compacted_nodes(&self) -> impl Iterator<Item = &Node<L>> {
        self.nodes.iter().filter(|n| !n.is_deleted())
    }
}
```

### DD-4: Const Generic Depth for ArrayTreeLayout **[DECIDED]**

**Decision**: Use **const generic** with common depths as type aliases.

**Rationale**:
- SIMD unrolling benefits from compile-time known depth
- XGBoost uses depth 6 as default (63 nodes = fits in cache line well)
- Runtime configurability adds branches in hot loop

```rust
pub type ArrayTreeLayout6 = ArrayTreeLayout<6>;  // 63 nodes, default
pub type ArrayTreeLayout4 = ArrayTreeLayout<4>;  // 15 nodes, small trees
pub type ArrayTreeLayout8 = ArrayTreeLayout<8>;  // 255 nodes, deep trees

// Usage determined at forest-load time based on max_depth
fn select_layout(max_depth: u32) -> Box<dyn BlockTraversal> {
    match max_depth {
        0..=4 => Box::new(ArrayTreeLayout4::new()),
        5..=6 => Box::new(ArrayTreeLayout6::new()),
        _ => Box::new(ArrayTreeLayout8::new()),
    }
}
```

**Note**: The layout selection happens once at prediction setup, not per-row.

## Open Questions

1. **Deleted slot reuse**: Should we track deleted node slots for reallocation during training? Could reduce memory churn for prune-then-expand patterns. XGBoost doesn't appear to do this, but it could be a frugal optimization.

2. **Index newtypes**: Should we introduce `NodeIdx`, `LeafIdx` newtypes? (See RFC-0001 DD-5 for discussion.)

## References

- XGBoost `RegTree::Node`: `include/xgboost/tree_model.h:86`
- XGBoost `ArrayTreeLayout`: `src/predictor/array_tree_layout.h`
- RFC-0001: Forest Data Structures
- [design/analysis/design_challenges_and_tradeoffs.md](../analysis/design_challenges_and_tradeoffs.md) §1
