# Multi-Output Inference

## ELI5

Normally, a decision tree gives you one answer per prediction (like "the house price is $350,000"). But sometimes you want multiple answers at once:
- "This image shows a cat (90% sure) and might show a dog (10% sure) and definitely not a bird (1%)"
- "Tomorrow's temperature will be 72°F, humidity will be 65%, and wind speed will be 8mph"

**Multi-output trees** give you several numbers at each leaf instead of just one, answering multiple questions simultaneously.

## ELI-Grad

Multi-output prediction extends tree ensembles to produce K output values per sample, supporting:

| Task | Outputs | Description |
|------|---------|-------------|
| **Multi-class classification** | K classes | One logit per class, softmax for probabilities |
| **Multi-target regression** | T targets | One value per target variable |
| **Multi-label classification** | L labels | One logit per label, sigmoid for probabilities |
| **Quantile regression** | Q quantiles | One value per quantile (10th, 50th, 90th percentile) |

There are two fundamental approaches:

1. **Separate trees per output**: Train K independent ensembles
2. **Vector leaves**: Single ensemble where each leaf stores K values

### Mathematical Formulation

For multi-output with K outputs:

**Separate trees**:
$$\hat{y}^{(k)} = \sum_{m=1}^{M_k} f_m^{(k)}(\mathbf{x})$$

Each output $k$ has its own set of $M_k$ trees.

**Vector leaves**:
$$\hat{\mathbf{y}} = \sum_{m=1}^{M} \mathbf{f}_m(\mathbf{x})$$

Each tree $m$ outputs a vector $\mathbf{f}_m(\mathbf{x}) \in \mathbb{R}^K$.

## Output Types

### Scalar Leaves (Single Output)

Standard trees produce one value per leaf:

```
Leaf values: [2.5, -1.3, 0.8, 1.1]  // 4 leaves, 1 value each
Prediction: sum of leaf values from all trees -> single scalar
```

Used for:
- Binary classification (single logit)
- Scalar regression
- Ranking

### Vector Leaves (Multiple Outputs)

Each leaf stores K values:

```
Leaf values (K=3):
  Leaf 0: [0.5, -0.2, 0.3]
  Leaf 1: [0.1, 0.4, -0.5]
  Leaf 2: [-0.3, 0.1, 0.2]
  Leaf 3: [0.2, -0.1, 0.0]

Prediction: sum of leaf vectors from all trees -> vector of K values
```

Used for:
- Multi-class classification (K class logits)
- Multi-target regression (K target values)

## Storage Approaches

### Approach 1: Naive (Inefficient)

Store variable-length vectors per leaf:

```
Leaf {
    values: dynamically allocated array of size K
}
```

Problems:
- Pointer indirection per leaf access
- Cache-unfriendly scattered allocations
- Cannot vectorize accumulation

### Approach 2: Packed Contiguous (Recommended)

Flatten all leaf vectors into a single contiguous buffer:

```
Packed storage for 4 leaves, K=3 outputs:

leaf_values: [L0_O0, L0_O1, L0_O2, L1_O0, L1_O1, L1_O2, L2_O0, L2_O1, L2_O2, L3_O0, L3_O1, L3_O2]
              |---- Leaf 0 ----|  |---- Leaf 1 ----|  |---- Leaf 2 ----|  |---- Leaf 3 ----|

Access leaf i: leaf_values[i * K : (i+1) * K]
```

Benefits:
- Contiguous reads: load entire leaf vector with one cache line
- SIMD-friendly: aligned vectors enable vectorized accumulation
- Memory efficient: no pointer overhead
- GPU-compatible: coalesced memory access patterns

### XGBoost: Vector Leaves

XGBoost supports vector leaves via `TreeParam::size_leaf_vector`:

```cpp
// From xgboost/include/xgboost/tree_model.h
struct TreeParam {
    bst_target_t size_leaf_vector;  // 0 or 1 for scalar, >1 for vector
};
```

When `size_leaf_vector > 1`, leaf values are stored in a separate packed array rather than inline in node structures.

### LightGBM: Separate Trees

LightGBM typically handles multi-class via separate trees per class, stored in `num_tree_per_iteration_`:

```cpp
// Conceptually, from LightGBM boosting
int num_tree_per_iteration_;  // = num_classes for multi-class

// Tree index for class k, iteration i:
// tree_idx = i * num_tree_per_iteration_ + k
```

## Tree Grouping for Multi-Class

### One-vs-All Structure

Train separate trees for each class:

```
Trees organized by class:
  Class 0: [T0_0, T0_1, T0_2, ...]  // All trees for class 0
  Class 1: [T1_0, T1_1, T1_2, ...]  // All trees for class 1
  Class 2: [T2_0, T2_1, T2_2, ...]  // All trees for class 2

Prediction for class k:
  output[k] = sum(tree.predict(x) FOR tree IN trees_for_class[k])
```

### Per-Iteration Grouping

XGBoost groups trees by boosting iteration:

```
Iteration 0: [T_class0, T_class1, T_class2]  // K trees, one per class
Iteration 1: [T_class0, T_class1, T_class2]  // K trees
...

Memory layout:
trees: [T0_c0, T0_c1, T0_c2, T1_c0, T1_c1, T1_c2, T2_c0, ...]
        |-- Iteration 0 --|  |-- Iteration 1 --|  |-- Iter 2...
```

This layout enables **early stopping**: evaluate partial predictions after each iteration and stop when margin is sufficient.

### Cache Implications

| Grouping | Cache Behavior | Best For |
|----------|----------------|----------|
| By class | Same class's trees cached together | Parallel class prediction |
| By iteration | All classes' trees for one iteration cached | Early stopping |

## Accumulation Strategies

### Scalar Accumulation

Simple addition for single-output:

```
output = 0.0
FOR tree IN trees:
    output += tree.predict(features)
```

### Vector Accumulation

Accumulate across all K outputs:

```
output = zeros(K)
FOR tree IN trees:
    leaf_vector = tree.predict_vector(features)
    FOR k FROM 0 TO K:
        output[k] += leaf_vector[k]
```

### SIMD Vector Accumulation

Vectorize the inner loop:

```
ALGORITHM: AccumulateSIMD(output, leaf_vector, K)
-------------------------------------------------
1. // Process 8 values at a time (AVX)
2. FOR i FROM 0 TO K STEP 8:
3.     out_vec <- LOAD_8_FLOATS(output[i:i+8])
4.     leaf_vec <- LOAD_8_FLOATS(leaf_vector[i:i+8])
5.     result <- ADD(out_vec, leaf_vec)
6.     STORE_8_FLOATS(output[i:i+8], result)
7. 
8. // Handle remainder
9. FOR i FROM (K // 8) * 8 TO K:
10.    output[i] += leaf_vector[i]
```

For K=1000 classes, this gives ~125x fewer loop iterations than scalar accumulation.

## Vector Leaves: Training Considerations

During training with vector leaves, each leaf accumulates gradients/hessians for all K outputs:

### Gradient Computation

For multi-class softmax with K classes, the gradient and Hessian for sample i, class k:

$$g_{i,k} = p_{i,k} - y_{i,k}$$
$$h_{i,k} = p_{i,k}(1 - p_{i,k})$$

where $p_{i,k} = \text{softmax}(F_k(x_i))$ and $y_{i,k}$ is the one-hot target.

### Leaf Value Computation

Each leaf stores K values, computed from the accumulated statistics:

$$w_k = -\frac{\sum_{i \in \text{leaf}} g_{i,k}}{\sum_{i \in \text{leaf}} h_{i,k} + \lambda}$$

This is applied independently for each of the K outputs.

## Batch Multi-Output Prediction

### Memory Layout for Batch Output

For B samples and K outputs:

**Row-major** (outputs contiguous per sample):
```
output: [S0_O0, S0_O1, S0_O2, S1_O0, S1_O1, S1_O2, S2_O0, ...]
         |-- Sample 0 ---|  |-- Sample 1 ---|  |-- Sample 2...
```

**Column-major** (samples contiguous per output):
```
output: [S0_O0, S1_O0, S2_O0, S0_O1, S1_O1, S2_O1, S0_O2, ...]
         |-- Output 0 ---|  |-- Output 1 ---|  |-- Output 2...
```

**Recommendation**: 
- Row-major for CPU (cache-friendly per sample, natural for most use cases)
- Column-major for GPU (coalesced writes when threads handle different samples)

### Block Processing for Multi-Output

```
ALGORITHM: PredictBatchMultiOutput(samples, trees, K)
----------------------------------------------------
1. output <- zeros(n_samples, K)
2. 
3. FOR block IN samples.chunks(BLOCK_SIZE):
4.     FOR tree IN trees:
5.         // Get leaf indices for entire block
6.         leaf_indices <- TraverseBlock(tree, block)
7.         
8.         // Accumulate vector leaves
9.         FOR i FROM 0 TO block.size:
10.            sample_idx <- block.start + i
11.            leaf_vector <- tree.get_leaf_vector(leaf_indices[i])
12.            
13.            // Vectorized accumulation
14.            AccumulateSIMD(output[sample_idx], leaf_vector, K)
```

## Output Transformation

After accumulation, transform raw scores to final predictions:

| Task | Transformation | Formula |
|------|----------------|---------|
| Binary classification | Sigmoid | $p = 1/(1 + e^{-s})$ |
| Multi-class | Softmax | $p_k = e^{s_k} / \sum_j e^{s_j}$ |
| Regression | Identity | $y = s$ |
| Multi-label | Element-wise sigmoid | $p_k = 1/(1 + e^{-s_k})$ |

### Softmax Implementation (Numerically Stable)

```
ALGORITHM: Softmax(logits)
--------------------------
1. // Subtract max for numerical stability
2. max_val <- MAX(logits)
3. 
4. // Exp and sum
5. sum <- 0.0
6. FOR k FROM 0 TO K:
7.     logits[k] <- exp(logits[k] - max_val)
8.     sum += logits[k]
9. 
10. // Normalize
11. FOR k FROM 0 TO K:
12.    logits[k] <- logits[k] / sum
```

The max subtraction prevents overflow when logits are large.

## Trade-offs: Separate Trees vs Vector Leaves

### Separate Trees per Output

**Advantages**:
- Simpler implementation (standard scalar trees)
- Each output can have different tree structure
- Parallelizable by output

**Disadvantages**:
- No shared representation learning across outputs
- K times more trees to store and traverse
- Cannot exploit correlations between outputs

### Vector Leaves

**Advantages**:
- Captures correlations between outputs (shared splits)
- More compact model (one tree predicts all outputs)
- Single traversal per tree

**Disadvantages**:
- More complex training (K gradients per sample)
- All outputs must use same tree structure
- Larger leaf storage

### When to Choose Each

| Scenario | Recommendation |
|----------|----------------|
| Outputs are independent | Separate trees |
| Outputs are correlated | Vector leaves |
| Few classes (K < 10) | Either works |
| Many classes (K > 100) | Vector leaves (efficiency) |
| Need per-output complexity control | Separate trees |

## GPU Considerations

### Packed Vector Leaves on GPU

Contiguous storage is critical for GPU performance:

```
GPU kernel conceptually:
  sample_idx <- thread_id
  
  FOR tree IN trees:
      leaf_idx <- traverse(tree, samples[sample_idx])
      leaf_ptr <- leaf_values + leaf_idx * K
      
      // Contiguous read of K values
      FOR k FROM 0 TO K:
          output[sample_idx * K + k] += leaf_ptr[k]
```

### Shared Memory for Leaf Values

Cache frequently accessed leaves in shared memory:

```
// GPU: Load leaf values for current tree into shared memory
__shared__ float shared_leaves[MAX_LEAVES_PER_TREE * K]

// Cooperative load
IF thread_in_block < n_leaves * K:
    shared_leaves[thread_in_block] <- leaf_values[tree_offset + thread_in_block]
BARRIER()

// Access from shared memory (fast)
leaf_ptr <- shared_leaves + leaf_idx * K
```

## Summary

| Aspect | Separate Trees | Vector Leaves |
|--------|----------------|---------------|
| **Storage** | K independent ensembles | Packed K-vectors per leaf |
| **Training** | Standard per-output | K gradients accumulated together |
| **Inference** | K traversals per sample | 1 traversal, K accumulations |
| **Correlation** | Not captured | Captured in shared splits |
| **Flexibility** | Different structure per output | Same structure for all |
| **Implementation** | Simpler | More complex |

### Key References

- `xgboost/include/xgboost/tree_model.h` — `size_leaf_vector` for vector leaves
- `xgboost/src/gbm/gbtree_model.h` — Tree grouping by iteration (`num_parallel_tree`)
- `LightGBM/src/boosting/gbdt.cpp` — `num_tree_per_iteration_` for multi-class
- XGBoost paper Section 2.2 — Multi-class objective gradients
