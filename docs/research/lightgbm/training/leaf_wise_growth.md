# LightGBM Leaf-wise Growth

## Overview

LightGBM's default tree growth strategy is **leaf-wise** (best-first), which differs
fundamentally from XGBoost's default **depth-wise** (level-wise) strategy.

## Comparison

```text
DEPTH-WISE (XGBoost default)          LEAF-WISE (LightGBM default)
                                      
Level 0:     [root]                         [root]
               │                              │
Level 1:    ┌──┴──┐                       ┌───┴───┐
           [L1]  [L2]                   [L1]    [L2] ← highest gain
            │      │                              │
Level 2:  ┌─┴─┐  ┌─┴─┐                        ┌───┴───┐
         ... ... ... ...                    [L3]    [L4] ← highest gain
                                                     │
                                                 ┌───┴───┐
                                               [L5]    [L6]

- Split ALL nodes at each level       - Split BEST leaf regardless of level
- Balanced trees                      - Potentially unbalanced trees
- O(2^depth) leaves                   - More efficient with same #leaves
- Better for smaller datasets         - Can overfit small datasets
```

## Implementation

From `SerialTreeLearner::Train()`:

```cpp
Tree* SerialTreeLearner::Train(...) {
  auto tree = new Tree(config_->num_leaves, ...);
  
  // Root setup
  tree->SetLeafOutput(0, CalculateSplittedLeafOutput(...));
  
  int left_leaf = 0;
  int right_leaf = -1;
  
  // Main loop: grow until max_leaves-1 splits
  for (int split = 0; split < config_->num_leaves - 1; ++split) {
    
    // 1. Find best splits for current candidate leaves
    if (BeforeFindBestSplit(tree, left_leaf, right_leaf)) {
      FindBestSplits(tree);
    }
    
    // 2. KEY: Select leaf with HIGHEST gain (not by level!)
    int best_leaf = ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_);
    
    // 3. Check stopping condition
    const SplitInfo& best = best_split_per_leaf_[best_leaf];
    if (best.gain <= 0.0) {
      break;  // No positive gain possible
    }
    
    // 4. Split the best leaf
    Split(tree, best_leaf, &left_leaf, &right_leaf);
  }
  
  return tree;
}
```

### Key Difference: Leaf Selection

```cpp
// LightGBM: Select leaf with MAX gain
int best_leaf = ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_);

// XGBoost (depth-wise): Process all leaves at current depth
// Then move to next depth level
```

## Gain Calculation

The split gain formula is standard:

$$
\text{Gain} = \frac{1}{2}\left[
  \frac{G_L^2}{H_L + \lambda} +
  \frac{G_R^2}{H_R + \lambda} -
  \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}
\right] - \gamma
$$

Where:

- $G_L, G_R$ = sum of gradients in left/right child
- $H_L, H_R$ = sum of hessians in left/right child
- $\lambda$ = L2 regularization
- $\gamma$ = min_gain_to_split

## Tracking Leaf Candidates

LightGBM maintains split info for ALL leaves:

```cpp
// For each leaf, store best split found
std::vector<SplitInfo> best_split_per_leaf_;  // size = num_leaves

// After each split, both children become candidates
void Split(Tree* tree, int best_leaf, int* left, int* right) {
  // ... perform split ...
  
  // Both children will be evaluated in next iteration
  *left = new_left_leaf_id;
  *right = new_right_leaf_id;
}
```

## Depth Control

Since leaf-wise can create deep trees, `max_depth` is crucial:

```cpp
bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) {
  // Check depth limit
  if (config_->max_depth > 0) {
    if (tree->leaf_depth(left_leaf) >= config_->max_depth) {
      // Mark as non-splittable
      best_split_per_leaf_[left_leaf].gain = kMinScore;
      if (right_leaf >= 0) {
        best_split_per_leaf_[right_leaf].gain = kMinScore;
      }
      return false;
    }
  }
  // ... continue with split finding
}
```

## Trade-offs

### Advantages of Leaf-wise

1. **Lower loss**: With same number of leaves, achieves better training loss
2. **Efficiency**: Resources focus on most impactful splits
3. **Natural early stopping**: Low-gain leaves automatically deprioritized

### Disadvantages

1. **Overfitting risk**: Can create very deep branches on small datasets
2. **Less balanced**: Trees may be highly asymmetric
3. **Harder to parallelize**: Depth-wise allows level-parallelism

## Recommended Usage

```python
# Small dataset: limit depth
params = {
    'num_leaves': 31,
    'max_depth': 6  # Critical for small data
}

# Large dataset: more leaves OK
params = {
    'num_leaves': 255,
    'max_depth': -1  # No limit
}
```

## Comparison with XGBoost

| Aspect | LightGBM (leaf-wise) | XGBoost (depth-wise) |
|--------|---------------------|---------------------|
| Split priority | Highest gain leaf | All leaves at level |
| Tree shape | Asymmetric | Balanced |
| Memory | Lower (fewer nodes) | Higher (full levels) |
| Training time | Often faster | Slower for same leaves |
| Overfitting | More risk on small data | More regularized |
| Parallelism | Feature-parallel | Level-parallel possible |

## Source References

| Component | Source File |
|-----------|-------------|
| Tree learner | `src/treelearner/serial_tree_learner.cpp` |
| Split tracking | `src/treelearner/split_info.hpp` |
| Leaf splits | `src/treelearner/leaf_splits.hpp` |
