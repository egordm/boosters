# GBDT Inference

GBDT inference traverses each tree in the ensemble to accumulate predictions. While
conceptually simple, efficient implementation requires careful attention to memory
access patterns and parallelization strategies.

---

## The Basic Algorithm

### ELI5

To make a prediction, you take your input features and "walk" through each tree:
- Start at the root
- At each node, check: "Is my feature value less than the threshold?"
- Go left or right based on the answer
- When you reach a leaf, record the value
- Add up all the leaf values from all trees

### ELI-Grad

For a single sample $x$ with forest $F = \{T_1, ..., T_M\}$:

$$
\hat{y} = \text{base\_score} + \sum_{m=1}^{M} T_m(x)
$$

Where $T_m(x)$ is the leaf value reached by traversing tree $m$ with input $x$.

For multi-class with $K$ classes, trees are grouped (usually round-robin):
$$
\hat{y}_k = \text{base\_score}_k + \sum_{m: \text{group}(m)=k} T_m(x)
$$

---

## Key Challenges

| Challenge | Why It Matters |
|-----------|----------------|
| **Memory access** | Each split accesses a different feature — random access pattern |
| **Branch prediction** | Different samples take different paths — unpredictable branches |
| **Parallelism** | Need to process many samples and trees efficiently |
| **Missing values** | Must handle NaN inputs with learned default directions |

---

## Optimization Strategies

### 1. Block-Based Traversal

Process samples in blocks (e.g., 64 samples) rather than one at a time. This amortizes
memory access costs and enables better cache utilization.

See [Batch Traversal](batch-traversal.md) for details.

### 2. Tree Layout Optimization

Store trees in formats optimized for inference:
- **SoA layout**: Separate arrays for split features, thresholds, children
- **Top-level unrolling**: Pre-expand first few levels for cache efficiency

See [Tree Storage](../data-structures/tree-storage.md).

### 3. Parallelism

Two main approaches:
- **Data parallelism**: Parallelize across samples (most common)
- **Model parallelism**: Parallelize across trees (useful for small batches)

### 4. SIMD Vectorization

Process multiple samples simultaneously using SIMD:
- Load features for 4/8 samples at once
- Compare against thresholds in parallel
- Accumulate results with vector operations

---

## Contents

| Document | Description |
|----------|-------------|
| [Batch Traversal](batch-traversal.md) | Block-based prediction, cache efficiency, unrolled layouts |
| [Multi-Output](multi-output.md) | Multi-class and multi-target prediction strategies |

---

## Complexity Analysis

| Configuration | Time Complexity | Notes |
|---------------|-----------------|-------|
| Single sample | O(M × D) | M trees, D depth |
| N samples | O(N × M × D) | Embarrassingly parallel |
| With blocks | O(N × M × D / B) amortized | B = block size |

Memory bandwidth is typically the bottleneck, not computation.

---

## Library Implementations

| Library | Key Source Files |
|---------|------------------|
| XGBoost | `src/predictor/cpu_predictor.cc` |
| LightGBM | `src/io/tree.cpp` (prediction methods) |
