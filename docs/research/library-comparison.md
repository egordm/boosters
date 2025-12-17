# XGBoost vs LightGBM: Approach Comparison

How XGBoost and LightGBM solve the same problems differently.

This document compares **approaches and trade-offs**, not feature checklists.
Use it to understand why certain design decisions were made.

---

## Tree Growth Strategy

**The problem**: In what order should we split tree nodes?

### XGBoost: Depth-wise (Level-by-Level)

Split all nodes at the current level before moving to the next:

```text
Level 0:    [root]         ← Split first
Level 1:  [A]   [B]        ← Split BOTH before going deeper
Level 2: [a][b] [c][d]     ← Split ALL FOUR
```

**Properties**:
- Produces balanced trees
- Easy to parallelize (entire level at once)
- May waste work on low-gain splits

### LightGBM: Leaf-wise (Best-First)

Always split the leaf with the highest gain:

```text
[root] → [highest-gain] → [highest-gain] → ...
```

**Properties**:
- Lower loss for same number of leaves
- More efficient (skips low-gain regions)
- Risk of overfitting without `max_depth` constraint
- Produces unbalanced trees

### Trade-off Summary

| Criterion | Depth-wise | Leaf-wise |
|-----------|------------|-----------|
| Accuracy (same leaves) | Lower | Higher |
| Overfitting risk | Lower | Higher |
| Parallelization | Easier | Harder |
| Best for | Small data, shallow trees | Large data, efficiency |

---

## Split Finding

**The problem**: How to efficiently find the best split point?

### Both: Histogram-Based

Both libraries quantize features into bins and use histogram aggregation.

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Default bins | 256 | 255 |
| Histogram subtraction | ✅ | ✅ |
| Missing value handling | Learned direction | Learned direction |

### Difference: Histogram Build Strategy

**XGBoost**: Row-wise by default — iterate samples, accumulate to feature histograms.

**LightGBM**: Adaptive — auto-selects row-wise or column-wise based on cache analysis.

---

## Sampling Strategies

**The problem**: How to reduce training data while maintaining quality?

### XGBoost: Random Subsampling

Uniform random selection of rows per tree.

```text
subsample = 0.8  → Use random 80% of rows per tree
```

Simple, unbiased, but doesn't prioritize informative samples.

### LightGBM: GOSS (Gradient-based One-Side Sampling)

Keep samples with large gradients, randomly sample the rest:

```text
1. Keep top 20% by |gradient|  (informative)
2. Random sample 10% of rest   (maintain distribution)
3. Upweight sampled rest       (correct for sampling bias)
```

**Insight**: Large gradient = model is wrong = more to learn.

**Trade-off**: More complex, but better sample efficiency for large datasets.

---

## Categorical Feature Handling

**The problem**: How to split on categorical features efficiently?

### XGBoost: One-Hot or Approximate Partitions

- Traditional: One-hot encode, treat as numerical
- v1.5+: Approximate partition-based splits via `enable_categorical`

### LightGBM: Native Gradient-Sorted Partitions

For high-cardinality categoricals:

1. Sort categories by gradient statistics
2. Find optimal partition in O(k log k) for k categories

**Key difference**: LightGBM finds optimal **binary partitions** (subset vs complement),
not just one-vs-rest splits.

---

## Sparse Data Optimization

**The problem**: How to handle datasets with many zeros/missing values?

### Both: Learned Default Direction

During split finding, try missing values going left AND right, pick better.

### LightGBM Extra: Exclusive Feature Bundling (EFB)

Bundle mutually exclusive features (if A ≠ 0 → B = 0):

```text
Features A, B, C never overlap → Bundle into single feature AB'C
```

Reduces histogram memory and computation for sparse/one-hot data.

---

## Gradient Precision

**The problem**: How to reduce memory for gradient storage?

### XGBoost: GPU-Only Quantization

Full precision (float32/64) on CPU. GPU uses 16-bit packed gradients.

### LightGBM: CPU + GPU Quantization

Adaptive precision based on leaf size:
- Large leaves: 32-bit (grad:16 + hess:16)
- Small leaves: 16-bit (grad:8 + hess:8)

**Trade-off**: Minor accuracy loss, significant memory/bandwidth savings.

---

## Multi-Output Handling

**The problem**: How to handle multi-class or multi-target problems?

### XGBoost: Vector Leaves (Optional)

Single tree can output K values via `size_leaf_vector`:

```text
Leaf stores: [v₀, v₁, ..., vₖ₋₁]
```

### LightGBM: Separate Trees per Output

Train `num_tree_per_iteration = K` trees, one per class:

```text
Iteration i: [Tree_class0, Tree_class1, ..., Tree_classK]
```

**Trade-off**:
- Vector leaves: shared structure, single traversal
- Separate trees: more flexible, independent tree shapes

---

## Model Serialization

**The problem**: How to save and load trained models?

| Format | XGBoost | LightGBM |
|--------|---------|----------|
| Primary | JSON (structured) | Text (simple) |
| Binary | UBJSON | Binary |
| Schema | Fully specified | Implicit |

XGBoost's JSON format has a documented schema, making it easier for third-party
implementations to load models.

---

## Summary: Design Philosophy

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| **Philosophy** | Correctness, compatibility | Speed, efficiency |
| **Tree growth** | Balanced (depth-wise) | Aggressive (leaf-wise) |
| **Sampling** | Simple random | Gradient-aware (GOSS) |
| **Categoricals** | Approximate | Native optimal |
| **Gradients** | Full precision CPU | Quantized CPU+GPU |
| **Documentation** | Extensive | Good |

### What We Learn from Each

**From XGBoost**:
- Clear JSON model format
- Monotonic constraint implementation
- Comprehensive parameter documentation

**From LightGBM**:
- Leaf-wise growth efficiency
- Native categorical handling
- CPU gradient quantization
- GOSS sampling strategy
