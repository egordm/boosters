# Tree Storage Formats

## Overview

Tree storage formats present a classic **AoS vs SoA** (Array of Structures vs Structure of
Arrays) trade-off:

| Format | Structure | Memory Pattern | Mutation | Cache Behavior |
|--------|-----------|----------------|----------|----------------|
| **AoS** | `Node { feat, thresh, left, right }` | Per-node access | Easy | Good for single-node ops |
| **SoA** | `feat[], thresh[], left[], right[]` | Per-field access | Harder | Good for batch ops |

The fundamental insight:

- **Training** needs frequent mutation (adding nodes, updating splits) → AoS is natural
- **Inference** needs fast batch prediction (same operation on many samples) → SoA enables SIMD/coalescing

## AoS: Array of Structures

### Concept

Each node is a self-contained structure with all its fields:

```text
Node 0: { parent=NULL, left=1, right=2, feature=5, threshold=0.5, leaf_value=N/A }
Node 1: { parent=0, left=NULL, right=NULL, feature=N/A, threshold=N/A, leaf_value=1.5 }
Node 2: { parent=0, left=3, right=4, feature=2, threshold=0.3, leaf_value=N/A }
...
```

### Memory Layout

```text
Memory (contiguous nodes):
+--------+--------+--------+--------+
| Node 0 | Node 1 | Node 2 | Node 3 |
+--------+--------+--------+--------+
   24B      24B      24B      24B

Each node:
+--------+-------+--------+---------+-----------+
| parent | left  | right  | feature | threshold |
| 4B     | 4B    | 4B     | 4B      | 4B        |
+--------+-------+--------+---------+-----------+
```

### XGBoost's Training Tree (AoS)

XGBoost uses AoS for tree construction:

```cpp
// From xgboost/include/xgboost/tree_model.h
class RegTree::Node {
    int32_t parent_;     // Parent index (high bit: is_left_child)
    int32_t cleft_;      // Left child (kInvalidNodeId if leaf)
    int32_t cright_;     // Right child
    uint32_t sindex_;    // Split feature (high bit: default_left)
    union {
        float leaf_value;   // If leaf node
        float split_cond;   // If internal node
    } info_;
};
```

Note the **union**: internal nodes store `split_cond` (threshold), leaf nodes store
`leaf_value`. This saves memory but means you must check `IsLeaf()` before accessing.

> **Reference**: `xgboost/include/xgboost/tree_model.h`, class `RegTree::Node`

### Why AoS for Training

During training, trees are built incrementally:

```text
1. Start with single leaf (root)
2. Find best split for current leaves
3. Split one leaf -> create two children
4. Repeat until stopping criterion
```

Each step modifies individual nodes:

- Add new nodes to the array
- Update parent pointers
- Store split information

With AoS, this is natural—all node data is together:

```text
ALGORITHM: SplitNode(tree, node_id, split)
------------------------------------------
1. left_id <- tree.add_node()
2. right_id <- tree.add_node()
3. tree.nodes[node_id].left <- left_id
4. tree.nodes[node_id].right <- right_id
5. tree.nodes[node_id].feature <- split.feature
6. tree.nodes[node_id].threshold <- split.threshold
7. tree.nodes[left_id].parent <- node_id
8. tree.nodes[right_id].parent <- node_id
```

## SoA: Structure of Arrays

### Concept

Separate arrays for each field, indexed by node ID:

```text
left_child:    [1, -1, 3, -1, -1]    // -1 = leaf (no child)
right_child:   [2, -1, 4, -1, -1]
split_feature: [5, -, 2, -, -]       // - = N/A for leaves
threshold:     [0.5, -, 0.3, -, -]
leaf_value:    [-, 1.5, -, 0.8, 2.1]
```

### Memory Layout

```text
Memory (separate arrays):
left_child:    [1, -1, 3, -1, -1]  <- contiguous
right_child:   [2, -1, 4, -1, -1]  <- contiguous
split_feature: [5, 0, 2, 0, 0]     <- contiguous
threshold:     [0.5, 0, 0.3, 0, 0] <- contiguous
leaf_value:    [0, 1.5, 0, 0.8, 2.1] <- contiguous
```

### LightGBM's Inference Tree (SoA)

LightGBM stores trained trees in SoA format:

```cpp
// From LightGBM/include/LightGBM/tree.h
class Tree {
    // Split node data (size = num_leaves - 1)
    std::vector<int> left_child_;
    std::vector<int> right_child_;
    std::vector<int> split_feature_;
    std::vector<double> threshold_;
    std::vector<int8_t> decision_type_;  // Packed flags
    
    // Leaf data (size = num_leaves)
    std::vector<double> leaf_value_;
    
    int num_leaves_;
    double shrinkage_;
};
```

> **Reference**: `LightGBM/include/LightGBM/tree.h`, class `Tree`

### Why SoA for Inference

During batch prediction, you need to traverse many samples through the tree:

```text
FOR each sample IN batch:
    node <- root
    WHILE node is not leaf:
        IF sample.features[split_feature[node]] < threshold[node]:
            node <- left_child[node]
        ELSE:
            node <- right_child[node]
    output[sample] <- leaf_value[node]
```

With SoA, reading thresholds for many nodes loads contiguous memory:

```text
SoA memory access pattern:
threshold = [t0, t1, t2, t3, t4, ...]
             +--- contiguous reads ---+

AoS memory access pattern:
nodes = [{...t0...}, {...t1...}, {...t2...}]
        +-- stride = 24 bytes between thresholds --+
```

### SIMD Benefits

SoA enables vectorization—comparing multiple values against multiple thresholds
simultaneously:

```text
// Conceptual SIMD operation
thresholds_vec = LOAD_8_FLOATS(threshold[node:node+8])
features_vec = GATHER_8_FLOATS(sample_features, split_feature[node:node+8])
mask = COMPARE_LESS_THAN(features_vec, thresholds_vec)  // 8 comparisons in 1 instruction
```

### GPU Coalescing

For GPU inference, SoA is essential:

```text
Warp of 32 threads at different tree nodes:
  Thread 0 needs threshold[node_0]
  Thread 1 needs threshold[node_1]
  Thread 2 needs threshold[node_2]
  ...

SoA: All reads from contiguous threshold[] array -> coalesced memory access
AoS: Reads scattered across node structs -> uncoalesced, slow
```

## Tree Unrolling (Top-Level Optimization)

For the first few tree levels, we can eliminate pointer chasing entirely by using
**implicit indexing** — storing nodes in level-order so child positions are computed
arithmetically.

### Implicit Indexing

In a complete binary tree stored in level order:

```text
Level 0:     [0]           ← root at index 0
Level 1:   [1] [2]         ← children at 2*0+1, 2*0+2
Level 2: [3][4][5][6]      ← children at 2*1+1, 2*1+2, etc.

For node at index i:
  left_child  = 2*i + 1
  right_child = 2*i + 2
  parent      = (i-1) / 2
```

**Benefit**: No pointer storage, no pointer chasing — just arithmetic.

### Why Partial Unrolling?

Fully unrolling a tree to depth D requires $2^D - 1$ node slots:

| Depth | Nodes Required | Memory (32B/node) |
|-------|----------------|-------------------|
| 6     | 63             | 2 KB              |
| 10    | 1,023          | 32 KB             |
| 15    | 32,767         | 1 MB              |
| 20    | 1,048,575      | 32 MB             |

XGBoost unrolls only the **top 6 levels** because:

1. **Memory explosion**: Sparse trees waste slots (padding to $2^D$)
2. **Gather bottleneck**: Beyond ~6 levels, the random memory access to fetch features
   (gather) dominates, not tree structure cache misses
3. **Sample divergence**: With block size 64, samples diverge to ~64 different subtrees
   after 6 levels anyway
4. **Diminishing returns**: Top levels route samples to $2^6 = 64$ subtrees, covering
   the highest-entropy decisions

### Unrolled Traversal Algorithm

```text
ALGORITHM: TraverseUnrolled(features, tree)
─────────────────────────────────────────
pos ← 0  // Position in level-order array

// Unrolled levels (no pointer chasing)
FOR level FROM 0 TO UNROLL_DEPTH - 1:
    feat ← tree.feature[pos]
    thresh ← tree.threshold[pos]
    
    IF features[feat] < thresh:
        pos ← 2 * pos + 1  // Left child (arithmetic)
    ELSE:
        pos ← 2 * pos + 2  // Right child (arithmetic)

// Remaining levels use standard pointer-following
node ← subtree_roots[pos]
WHILE node is not leaf:
    // Standard traversal...

RETURN leaf_value[node]
```

### Trade-offs

| Aspect | Unrolled | Pointer-Based |
|--------|----------|---------------|
| Memory | Higher (padding) | Lower (sparse OK) |
| Cache misses (structure) | Zero for unrolled levels | One per level |
| Implementation | More complex | Simple |
| Best for | Top ~6 levels | Deep/sparse trees |

## Node Indexing Conventions

Different libraries use different conventions for identifying leaf nodes:

### XGBoost: Sentinel Value

Uses a special value (`kInvalidNodeId = -1`) to mark leaves:

```text
cleft_ == kInvalidNodeId (-1) -> this is a leaf node
Otherwise -> this is an internal node

Tree structure:
        [0]           <- root (internal)
       /   \
     [1]   [2]        <- node 1 is leaf, node 2 is internal
          /   \
        [3]   [4]     <- both leaves
```

### LightGBM: Negative Indices

Uses negative indices (bitwise NOT) for leaves:

```text
Positive index -> internal node
Negative index -> leaf (use ~index to get leaf ID)

left_child_  = [~0, ~1]  // Node 0 -> Leaf 0, Node 1 -> Leaf 1
right_child_ = [1, ~2]   // Node 0 -> Node 1, Node 1 -> Leaf 2
```

This convention separates internal nodes from leaves in storage, allowing different-sized
arrays.

## Decision Type Encoding (LightGBM)

LightGBM packs multiple boolean flags into a single byte:

```text
Bit 0: Categorical flag     (0=numerical, 1=categorical)
Bit 1: Default left flag    (0=missing->right, 1=missing->left)
Bit 2-3: Missing type       (00=None, 01=Zero, 10=NaN)

Example: decision_type = 0b00000110 = 6
  - Bit 0: 0 -> numerical split
  - Bit 1: 1 -> missing values go left
  - Bit 2-3: 01 -> treat 0.0 as missing
```

This saves memory compared to separate boolean fields and improves cache efficiency.

> **Reference**: `LightGBM/include/LightGBM/tree.h`, `kCategoricalMask`, `kDefaultLeftMask`

## Multi-Output (Vector) Leaves

For multi-target regression or multi-class classification, leaves may store multiple values:

### Scalar Leaves (Standard)

```text
leaf_value: [v0, v1, v2, v3]  // 4 leaves, 1 value each
```

### Vector Leaves

```text
// 4 leaves, 3 outputs each
leaf_value: [v0_0, v0_1, v0_2,   // Leaf 0
             v1_0, v1_1, v1_2,   // Leaf 1
             v2_0, v2_1, v2_2,   // Leaf 2
             v3_0, v3_1, v3_2]   // Leaf 3
```

XGBoost supports this via `TreeParam::size_leaf_vector`:

```cpp
// From xgboost/include/xgboost/tree_model.h
struct TreeParam {
    bst_target_t size_leaf_vector;  // 1 for scalar, >1 for vector
};
```

LightGBM typically handles multi-output via separate trees per output rather than vector
leaves.

## Linear Tree Extension

LightGBM supports linear models in leaves (leaf contains a linear function, not just a
constant):

```text
Prediction for leaf i:
  output = leaf_const[i] + sum(leaf_coeff[i][j] * feature[leaf_features[i][j]])
```

This requires additional storage per leaf:

```cpp
// Conceptual (from LightGBM tree.h)
std::vector<double> leaf_const_;              // Intercept per leaf
std::vector<std::vector<double>> leaf_coeff_; // Coefficients per leaf
std::vector<std::vector<int>> leaf_features_; // Feature indices per leaf
```

## Memory Estimates

For a tree with L leaves (L-1 internal nodes, 2L-1 total nodes):

### XGBoost AoS (Training)

```text
Nodes: (2L-1) * 24 bytes = ~48L bytes
Stats: (2L-1) * 16 bytes = ~32L bytes (RTreeNodeStat)
Total: ~80L bytes per tree
```

### LightGBM SoA (Inference)

```text
Internal nodes: (L-1) * (4+4+4+8+1+4) = ~25(L-1) bytes
Leaf nodes:     L * (8+8+4+4) = ~24L bytes
Total: ~49L bytes per tree (more compact)
```

### Forest Memory

For 100 trees with 31 leaves each (depth 5):

```text
XGBoost AoS: 100 * 80 * 31 = ~248 KB
LightGBM SoA: 100 * 49 * 31 = ~152 KB
```

SoA is typically more compact because:

- No padding between fields
- Leaves and internal nodes can use different field sets
- Bit-packing for flags

## When to Use Each Format

### AoS (Training)

**Use when:**

- Building trees incrementally
- Frequent node mutations
- Need parent pointers for backtracking
- Implementing tree updaters/pruners

**Avoid when:**

- Batch inference with many samples
- GPU inference
- SIMD optimization is critical

### SoA (Inference)

**Use when:**

- Batch prediction (many samples)
- GPU inference
- SIMD optimization
- Memory-mapped model loading
- Read-only tree access

**Avoid when:**

- Training (frequent mutations are awkward)
- Single-sample latency-critical paths (AoS may be simpler)

## Summary

| Aspect | Training (AoS) | Inference (SoA) |
|--------|----------------|-----------------|
| **Layout** | Node structs in array | Separate array per field |
| **Mutation** | Easy (modify one node) | Hard (update multiple arrays) |
| **Cache** | Good for single node access | Good for field across many nodes |
| **SIMD** | Limited | Excellent (contiguous data) |
| **GPU** | Poor (uncoalesced access) | Excellent (coalesced access) |
| **Use case** | Tree construction | Batch prediction |

### Key References

- `xgboost/include/xgboost/tree_model.h` — `RegTree::Node` (AoS)
- `LightGBM/include/LightGBM/tree.h` — `Tree` class (SoA)
- "Data-Oriented Design" — General principles of AoS vs SoA trade-offs
