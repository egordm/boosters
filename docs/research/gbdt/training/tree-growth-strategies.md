# Tree Growth Strategies

Tree growth strategy determines the order in which nodes are expanded during training.
The two main approaches — depth-wise and leaf-wise — have different trade-offs in
terms of efficiency, accuracy, and overfitting risk.

---

## Overview

### ELI5

Imagine you're organizing a tournament bracket. 

**Depth-wise** (level-by-level): Play all first-round games, then all second-round games,
then all third-round games. Very organized, but you do work even on "boring" games.

**Leaf-wise** (best-first): Always play the most exciting game next, wherever it is in
the bracket. More efficient if some branches are clearly decided early.

### ELI13

**Depth-wise** builds balanced trees by splitting all nodes at each level before moving
to the next level. This is the traditional approach.

**Leaf-wise** always splits the node (leaf) with the highest gain, regardless of depth.
This focuses computation on the most informative splits.

```text
Depth-wise:
Level 0:    [root]           ← Split this first
               │
Level 1:  [A]   [B]          ← Split BOTH before going deeper
           │     │
Level 2: [C][D] [E][F]       ← Split ALL four

Leaf-wise:
[root] → [highest-gain leaf] → [highest-gain leaf] → ...
May produce unbalanced trees
```

---

## Depth-Wise Growth

### Algorithm

```text
Algorithm: Depth-Wise Tree Growth
─────────────────────────────────────────
Input: Training data, max_depth, other constraints
Output: Decision tree

1. Create root node with all samples
2. For depth = 0 to max_depth - 1:
   a. For each leaf node at current depth:
      - Build histogram
      - Find best split
      - If gain > 0 and constraints satisfied:
        - Create two child nodes
        - Partition samples
   b. If no splits made, stop
3. Compute leaf weights for all leaves
```

### Characteristics

| Aspect | Depth-Wise |
|--------|-----------|
| Tree shape | Balanced (same depth everywhere) |
| Parallelism | Easy — process entire level together |
| Memory | Predictable — 2^depth nodes at each level |
| Best for | Shallow trees, distributed training |
| Default in | XGBoost |

### Level-Parallel Optimization

All nodes at the same depth are independent, enabling parallelism:

```text
Level 2: [C] [D] [E] [F]
         ↓   ↓   ↓   ↓    ← All can be processed in parallel
         Build histograms, find splits simultaneously
```

This is particularly valuable in distributed settings where histogram aggregation
can be done level-by-level.

---

## Leaf-Wise Growth

### Algorithm

```text
Algorithm: Leaf-Wise Tree Growth
─────────────────────────────────────────
Input: Training data, num_leaves, other constraints
Output: Decision tree

1. Create root node, add to priority queue with gain = ∞
2. While num_nodes < num_leaves:
   a. Pop leaf with highest gain from queue
   b. Build histogram for this leaf
   c. Find best split
   d. If gain > 0 and constraints satisfied:
      - Split into two children
      - Compute potential gains for children
      - Add children to priority queue
   e. Else: remove from queue (won't be split)
3. Compute leaf weights for all leaves
```

### Characteristics

| Aspect | Leaf-Wise |
|--------|-----------|
| Tree shape | Unbalanced (deeper where data is complex) |
| Parallelism | Harder — must track best leaf globally |
| Memory | Variable — priority queue of candidates |
| Best for | Large datasets, when efficiency matters |
| Default in | LightGBM |

### Why Leaf-Wise is More Efficient

Consider 8 samples with this structure:

```text
Depth-wise must split all nodes at each level:
      [8]            ← Split (gain = 10)
     /    \
   [4]    [4]        ← Split BOTH (gains = 8, 2)
   / \    / \
 [2][2] [2][2]       ← Split ALL FOUR

Leaf-wise only splits high-gain nodes:
      [8]            ← Split (gain = 10)
     /    \
   [4]    [4]        ← Only split left (gain = 8)
   / \      \
 [2][2]    [4]       ← Only split [2] with gain > threshold
```

With the same number of leaves, leaf-wise achieves lower loss because it prioritizes
the most impactful splits.

---

## Comparison

| Criterion | Depth-Wise | Leaf-Wise |
|-----------|------------|-----------|
| Accuracy (same #leaves) | Lower | Higher |
| Training speed | Slower | Faster |
| Overfitting risk | Lower | Higher (needs max_depth) |
| Distributed training | Easier | Harder |
| GPU efficiency | Better | Okay |
| Tree interpretability | Balanced structure | Varied depths |

### When to Use Depth-Wise

- Small datasets (< 10k samples) — less overfitting risk
- Distributed training — level-sync is natural
- GPU training — uniform tree structure helps
- When max_depth is small (< 6)

### When to Use Leaf-Wise

- Large datasets (> 100k samples) — efficiency matters
- When training time is a bottleneck
- Deep trees are beneficial
- Single-machine training

---

## Hybrid Approaches

### Leaf-Wise with Max Depth

LightGBM uses leaf-wise but respects `max_depth`:

```text
Parameters:
  num_leaves = 31     # Primary constraint
  max_depth = -1      # No depth limit (default)
  # or
  max_depth = 8       # Additional constraint
```

This gives leaf-wise efficiency with depth-based regularization.

### Loss-Guided in XGBoost

XGBoost's `grow_policy = lossguide` enables leaf-wise:

```text
Parameters:
  grow_policy = 'lossguide'
  max_leaves = 31
```

---

## Stopping Criteria

Both strategies use similar stopping conditions:

| Criterion | Description |
|-----------|-------------|
| max_depth | Don't grow beyond this depth |
| num_leaves / max_leaves | Maximum leaf count |
| min_child_weight | Minimum Hessian sum in child |
| min_split_gain | Minimum gain to accept split |
| min_data_in_leaf | Minimum samples in leaf |

The difference is which splits are **prioritized**, not which are **allowed**.

---

## Implementation Considerations

### Depth-Wise

```text
Simpler implementation:
- Process level by level
- All nodes at same level have same parent depth
- Natural batch processing

Data structures:
- Two arrays: current_level_nodes, next_level_nodes
- Swap arrays each level
```

### Leaf-Wise

```text
More complex implementation:
- Priority queue of candidate leaves
- Track potential gain for each leaf
- May need to recompute gains after sibling splits

Data structures:
- Priority queue (max-heap by gain)
- Gain cache per leaf
```

---

## Theoretical Analysis

### Loss Reduction

For a fixed number of leaves $L$, leaf-wise achieves the globally optimal allocation
of splits (greedily). Depth-wise may waste splits on low-gain regions.

### Overfitting Risk

Leaf-wise can create very deep paths for outliers, leading to overfitting on small
datasets. The `max_depth` constraint mitigates this.

### Computational Complexity

Both strategies: O(samples × features × depth) per tree.

Leaf-wise has lower constants because:
- Histogram subtraction is more effective (larger sibling more common)
- Fewer total nodes visited

---

## Source References

### XGBoost

- `src/tree/updater_quantile_hist.cc` — `DepthWise` and `LossGuide` drivers
- Parameter: `grow_policy = 'depthwise'` (default) or `'lossguide'`

### LightGBM

- `src/treelearner/serial_tree_learner.cpp` — `Train()` method (leaf-wise)
- Parameters: `num_leaves`, `max_depth`
