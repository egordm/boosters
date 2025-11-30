# XGBoost GBTree Research

Research on XGBoost's tree-based booster and related optimizations.

## Contents

### Inference

- [xgboost-inference.md](xgboost-inference.md) — How XGBoost predicts with trees
- [array_tree_layout.md](array_tree_layout.md) — Structure-of-Arrays tree storage
- [block_based_traversal.md](block_based_traversal.md) — Block-based prediction for cache efficiency
- [packed_vector_leaf.md](packed_vector_leaf.md) — Multi-output leaf value packing
- [precompute_pack_grouping.md](precompute_pack_grouping.md) — Tree-to-group mapping optimization

### Training

- [quantized_features.md](quantized_features.md) — Feature quantization for histogram building
- [quantized_data_structures/](quantized_data_structures/) — Detailed data structure research

## Key Concepts

**GBTree** is XGBoost's default booster. It builds an ensemble of decision trees
using gradient boosting — each tree corrects the errors of previous trees.

### Optimization Stack (Inference)

1. **Array Tree Layout** — Store node attributes in contiguous arrays (SoA) instead
   of per-node structs (AoS). Better cache utilization, especially on GPU.

2. **Block Processing** — Process rows in blocks (e.g., 64 rows) rather than one
   at a time. Amortizes setup costs, improves cache reuse.

3. **Unrolled Traversal** — Process all rows at the same tree level together.
   All rows do the same "go left or right" decision, enabling vectorization.

4. **Thread Parallelism** — Parallelize across rows (or blocks of rows).
   Each prediction is independent.

### Optimization Stack (Training)

1. **Quantization** — Convert continuous features to discrete bins (0-255).
   Enables histogram building with integer indexing.

2. **Histogram Building** — Aggregate gradients per bin instead of per row.
   Reduces split-finding from O(n) to O(bins).

3. **Histogram Subtraction** — For a binary split, child = parent - sibling.
   Only build histogram for one child, derive the other for free.

4. **Level-wise Growth** — Grow all nodes at the same depth simultaneously.
   Better parallelization than leaf-wise.
