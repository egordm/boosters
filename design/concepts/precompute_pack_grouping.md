# Precompute & Pack Grouping

## ELI13
For models that predict multiple outputs (like multi-class), each tree belongs to a group (a class). We record which group every tree belongs to, then precompute this mapping and store it as a simple array. That way, during prediction we know exactly where to add the leaf value (e.g. which slot in the answer array) without searching or computing repeatedly.

## ELI Grad
`tree_info` is the per-tree group index mapping in XGBoost: the group (or target) a tree contributes to. Precomputing and packing grouping means we prepare compact arrays (`tree_group[]`) that map every tree id to a group index, possibly compressed for faster GPU addressing and vectorized reduction. When vector-leaf trees are used, `tree_group` is often 0 and the leaf vector length indicates which group element to add. For scalar-leaf trees, `tree_group[tree_id]` indicates the group destination. The packed grouping can also embed GPU-friendly metadata like memory offsets, weights (for DART), and bitmasks for group subranges to enable fast group-level kernels.

### Example structure
```
tree_info: [ 0, 1, 0, 2, 1 ]  // 5 trees, group per tree
// Accumulate using: out[row * n_groups + tree_info[t]] += leaf_value(t)

For DART weight-drop: weight_drop: [1.0, 0.5, 1.0, 0.9, ...]
apply: out[...] += leaf_value * weight_drop[t]
```

### Packed group layout (GPU-minded)
- `group_offsets[]`: for each group g, the index range `[group_offsets[g], group_offsets[g+1])` of trees contributing to g; repeatedly iterate group partition for addition.
- `tree_to_group[]`: dense `u32` array length `num_trees` mapping to group id.
- `tree_weight[]`: optional per-tree weight (DART) array aligned and packed.

### Optimization note
Grouping packing reduces branch overhead and allows grouped kernels (e.g., process all trees for group `g` in a single pass) offering better memory locality and parallel aggregation strategies.

## Training vs Inference
- Training: `tree_info` (group per tree) is fundamental during training for multi-class / multi-target boosting. Training routines rely on this mapping to produce group-specific gradients or splits and therefore need some version of it during training. That said, `group_offsets` and packed grouping for GPU-friendly grouped kernels are not typical during training and can be generated later.
- Inference: Packing grouping metadata into `group_offsets[]`, `tree_to_group[]` and `tree_weight[]` significantly accelerates prediction on both CPU and GPU by avoiding branches and enabling grouped addition kernels. This is usually done post-training (model finalization or load time) and requires only the `tree_info` mapping from the training artifacts.
- Notes: It is safe to compute packed grouping at model finalization and store it as part of the runtime model; training code can continue to use a simpler `tree_info` layout.
