# Tree Storage Formats

## Overview

Tree storage formats present a classic **AoS vs SoA** (Array of Structures vs Structure of Arrays) trade-off:

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

```
Node 0: { parent=NULL, left=1, right=2, feature=5, threshold=0.5, leaf_value=N/A }
Node 1: { parent=0, left=NULL, right=NULL, feature=N/A, threshold=N/A, leaf_value=1.5 }
Node 2: { parent=0, left=3, right=4, feature=2, threshold=0.3, leaf_value=N/A }
...
```

### Memory Layout

```
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

Note the **union**: internal nodes store `split_cond` (threshold), leaf nodes store `leaf_value`. This saves memory but means you must check `IsLeaf()` before accessing.

> **Reference**: `xgboost/include/xgboost/tree_model.h`, class `RegTree::Node`

### Why AoS for Training

During training, trees are built incrementally:

```
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

```
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

```
left_child:    [1, -1, 3, -1, -1]    // -1 = leaf (no child)
right_child:   [2, -1, 4, -1, -1]
split_feature: [5, -, 2, -, -]       // - = N/A for leaves
threshold:     [0.5, -, 0.3, -, -]
leaf_value:    [-, 1.5, -, 0.8, 2.1]
```

### Memory Layout

```
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

```
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

```
SoA memory access pattern:
threshold = [t0, t1, t2, t3, t4, ...]
             +--- contiguous reads ---+

AoS memory access pattern:
nodes = [{...t0...}, {...t1...}, {...t2...}]
        +-- stride = 24 bytes between thresholds --+
```

### SIMD Benefits

SoA enables vectorization—comparing multiple values against multiple thresholds simultaneously:

```
// Conceptual SIMD operation
thresholds_vec = LOAD_8_FLOATS(threshold[node:node+8])
features_vec = GATHER_8_FLOATS(sample_features, split_feature[node:node+8])
mask = COMPARE_LESS_THAN(features_vec, thresholds_vec)  // 8 comparisons in 1 instruction
```

### GPU Coalescing

For GPU inference, SoA is essential:

```
Warp of 32 threads at different tree nodes:
  Thread 0 needs threshold[node_0]
  Thread 1 needs threshold[node_1]
  Thread 2 needs threshold[node_2]
  ...

SoA: All reads from contiguous threshold[] array -> coalesced memory access
AoS: Reads scattered across node structs -> uncoalesced, slow
```

## Node Indexing Conventions

Different libraries use different conventions for identifying leaf nodes:

### XGBoost: Sentinel Value

Uses a special value (`kInvalidNodeId = -1`) to mark leaves:

```
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

```
Positive index -> internal node
Negative index -> leaf (use ~index to get leaf ID)

left_child_  = [~0, ~1]  // Node 0 -> Leaf 0, Node 1 -> Leaf 1
right_child_ = [1, ~2]   // Node 0 -> Node 1, Node 1 -> Leaf 2
```

This convention separates internal nodes from leaves in storage, allowing different-sized arrays.

## Decision Type Encoding (LightGBM)

LightGBM packs multiple boolean flags into a single byte:

```
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

```
leaf_value: [v0, v1, v2, v3]  // 4 leaves, 1 value each
```

### Vector Leaves

```
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

LightGBM typically handles multi-output via separate trees per output rather than vector leaves.

## Linear Tree Extension

LightGBM supports linear models in leaves (leaf contains a linear function, not just a constant):

```
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

## Conversion: AoS to SoA

Models are typically:

1. **Trained** with AoS (mutable, easy to build)
2. **Converted** to SoA after training
3. **Serialized** in SoA format
4. **Loaded** for inference in SoA

Conversion algorithm:

```
ALGORITHM: ConvertAoStoSoA(aos_trees)
-------------------------------------
1. total_nodes <- SUM(tree.num_nodes FOR tree IN aos_trees)
2. 
3. // Allocate SoA arrays
4. left_child <- ALLOCATE(total_nodes)
5. right_child <- ALLOCATE(total_nodes)
6. split_feature <- ALLOCATE(total_nodes)
7. threshold <- ALLOCATE(total_nodes)
8. leaf_value <- ALLOCATE(total_nodes)
9. 
10. // Copy from each tree
11. offset <- 0
12. FOR tree IN aos_trees:
        FOR i FROM 0 TO tree.num_nodes:
            left_child[offset + i] <- tree.nodes[i].left
            right_child[offset + i] <- tree.nodes[i].right
            split_feature[offset + i] <- tree.nodes[i].feature
            threshold[offset + i] <- tree.nodes[i].threshold
            leaf_value[offset + i] <- tree.nodes[i].leaf_value
        offset <- offset + tree.num_nodes
13. 
14. RETURN SoAForest(left_child, right_child, split_feature, threshold, leaf_value)
```

## Memory Estimates

For a tree with L leaves (L-1 internal nodes, 2L-1 total nodes):

### XGBoost AoS (Training)

```
Nodes: (2L-1) * 24 bytes = ~48L bytes
Stats: (2L-1) * 16 bytes = ~32L bytes (RTreeNodeStat)
Total: ~80L bytes per tree
```

### LightGBM SoA (Inference)

```
Internal nodes: (L-1) * (4+4+4+8+1+4) = ~25(L-1) bytes
Leaf nodes:     L * (8+8+4+4) = ~24L bytes
Total: ~49L bytes per tree (more compact)
```

### Forest Memory

For 100 trees with 31 leaves each (depth 5):

```
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
| **Conversion** | -> SoA after training | <- From AoS or serialized |

### Key References

- `xgboost/include/xgboost/tree_model.h` — `RegTree::Node` (AoS)
- `LightGBM/include/LightGBM/tree.h` — `Tree` class (SoA)
- "Data-Oriented Design" — General principles of AoS vs SoA trade-offs
