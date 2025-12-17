# Batch Traversal

## ELI5

Imagine you're a teacher grading tests. You could:
1. Grade one student's entire test, then the next student's, etc.
2. Grade Question 1 for ALL students, then Question 2 for ALL students, etc.

The second way is faster because you keep the answer key for Question 1 in your head while grading everyone's answer—you don't have to keep looking it up!

**Batch traversal** works similarly: instead of running one sample through all trees, we run many samples through each tree together. This keeps tree nodes "in our head" (CPU cache) while processing all samples.

## ELI-Grad

Batch traversal is an optimization that processes multiple samples together through a tree ensemble, improving cache utilization and enabling vectorization. The key insight is that **tree structure is read-only during inference**—if we process many samples simultaneously, they all share access to the same tree nodes.

### The Cache Efficiency Problem

Modern CPUs have a memory hierarchy:

| Level | Size | Latency | What fits |
|-------|------|---------|-----------|
| L1 Cache | 32-64 KB | ~4 cycles | A few tree levels |
| L2 Cache | 256-512 KB | ~12 cycles | Several small trees |
| L3 Cache | 4-32 MB | ~40 cycles | Many trees |
| RAM | GBs | ~200+ cycles | Everything |

**Single-sample processing** problem:

```
FOR sample IN batch:
    FOR tree IN forest:
        output[sample] += traverse(tree, sample)  // Tree loaded from RAM
                                                  // Evicted before next sample uses it
```

Each tree is loaded from RAM, used once, and evicted before the next sample can benefit.

**Batch processing** solution:

```
FOR tree IN forest:
    FOR sample IN batch:
        output[sample] += traverse(tree, sample)  // Tree stays in cache
                                                  // Reused by many samples
```

Tree stays in cache while all samples in the batch traverse it.

## Block-Based Traversal

### The Algorithm

Rather than processing samples one at a time, group them into **blocks**:

```
ALGORITHM: BatchPredict(samples, trees, block_size=64)
------------------------------------------------------
1. FOR block IN samples.chunks(block_size):
2.     // Load features for entire block
3.     block_features <- LoadFeatures(block)
4.     
5.     FOR tree IN trees:
6.         // Traverse all samples in block through this tree
7.         leaf_indices <- TraverseBlock(tree, block_features)
8.         
9.         // Accumulate leaf values
10.        FOR i FROM 0 TO block.size:
11.            output[block.start + i] += tree.leaf_value[leaf_indices[i]]
```

### Why Block Size Matters

| Block Size | Pros | Cons |
|------------|------|------|
| Small (1-16) | Low memory, simple | No vectorization, high loop overhead |
| Medium (32-64) | Cache-friendly, vectorizable | Good balance for most workloads |
| Large (128-256) | Better amortization | Memory pressure, samples diverge quickly |

**Recommended default: 64 samples**

This matches:
- Typical L1 cache line behavior
- Common SIMD register widths (8x8 or 4x16 patterns)
- GPU warp sizes (32 threads, 2 warps per block)

### Per-Thread Buffers

To avoid allocation in the hot path, pre-allocate reusable buffers:

```
ThreadLocalBuffer:
  feature_values: array[BLOCK_SIZE * MAX_FEATURES] of float
  node_positions: array[BLOCK_SIZE] of int
  leaf_outputs:   array[BLOCK_SIZE] of float
```

Allocate once at initialization, reuse across all predictions.

## Level-Synchronized Traversal

### Concept

Instead of each sample following its own path independently, synchronize all samples at each tree level:

```
Level 0: All samples at root
         |
         v
Level 1: Samples split to left/right children
         |
         v
Level 2: Samples split again
         ...
```

### Why This Helps

**Independent traversal**: Each sample has unpredictable branches

```
Sample 0: root -> left -> right -> left -> leaf
Sample 1: root -> right -> left -> left -> leaf
Sample 2: root -> left -> left -> right -> leaf
          (completely different memory access patterns)
```

**Level-synchronized traversal**: All samples access the same tree level simultaneously

```
Level 0: All samples compare against root.threshold
Level 1: All samples compare against level-1 thresholds
         (contiguous memory access, predictable branches)
```

## Unrolled Tree Layouts

### The Concept

Standard trees use pointer-based navigation (store child indices, follow them). This causes random memory access:

```
Standard: nodes[current].left -> nodes[that].left -> nodes[other].left
          (3 pointer chases, 3 potential cache misses)
```

**Unrolled layout** stores the top K levels in position-indexed arrays:

```
Unrolled (depth 3):
  Level 0: [root]           <- position 0
  Level 1: [left, right]    <- positions 0, 1
  Level 2: [LL, LR, RL, RR] <- positions 0, 1, 2, 3

Position in level d: directly indexes into level-d array
No pointer chasing for first K levels!
```

### Implicit Indexing

For a complete binary tree, child positions are computed arithmetically:

```
Parent at position p in level d:
  Left child:  position 2*p     in level d+1
  Right child: position 2*p + 1 in level d+1
```

This eliminates pointer storage and following for the unrolled levels.

### Why Partial Unrolling?

Fully unrolling a tree to depth D requires $2^D - 1$ node slots:

| Depth | Nodes Required | Memory (32B/node) |
|-------|----------------|-------------------|
| 6 | 63 | 2 KB |
| 10 | 1,023 | 32 KB |
| 15 | 32,767 | 1 MB |
| 20 | 1,048,575 | 32 MB |

**XGBoost unrolls only the top 6 levels** because:

1. **Memory explosion**: Deep unrolling wastes memory on padding (sparse trees don't use all positions)
2. **Gather bottleneck**: Beyond ~6 levels, the random memory access (gather) to fetch features dominates, not cache misses for tree structure
3. **Diminishing returns**: Top levels route samples to $2^6 = 64$ subtrees, covering high-entropy decisions
4. **Block amortization**: With block size 64, samples diverge to ~64 different subtrees after 6 levels anyway

> **Reference**: `xgboost/src/predictor/cpu_predictor.cc`, array tree layout

### Traversal with Unrolled Layout

```
ALGORITHM: TraverseUnrolled(features, unrolled_tree)
----------------------------------------------------
1. pos <- 0  // Position within current level
2. 
3. // Unrolled levels (no indirection)
4. FOR level FROM 0 TO UNROLL_DEPTH:
5.     feat_idx <- unrolled_tree.feature[level][pos]
6.     threshold <- unrolled_tree.threshold[level][pos]
7.     fval <- features[feat_idx]
8.     
9.     // Next position: left child at 2*pos, right at 2*pos+1
10.    IF fval <= threshold:
11.        pos <- 2 * pos
12.    ELSE:
13.        pos <- 2 * pos + 1
14.
15. // Continue with standard traversal for deep levels
16. node <- subtree_roots[pos]
17. WHILE node is not leaf:
18.    // Standard pointer-following for remaining levels
19.    ...
20.
21. RETURN leaf_value[node]
```

## Cache Efficiency Techniques

### Feature Staging

Load all features for a sample into a contiguous buffer before traversing trees:

```
WITHOUT staging (bad):
  FOR tree IN trees:
      fval <- data[sample][tree.root.feature]  // Random access
      // ... traverse using random feature accesses

WITH staging (good):
  staged_features <- data[sample, :]  // One contiguous read
  FOR tree IN trees:
      fval <- staged_features[tree.root.feature]  // Cached access
```

Benefits:
- Original data might be column-major (bad for row access)
- Staged buffer is compact and cache-friendly
- Same features reused across many trees

### Prefetching

Explicitly request next-level nodes before they're needed:

```
ALGORITHM: TraverseWithPrefetch(node, features)
-----------------------------------------------
1. PREFETCH(nodes[node.left])   // Request from memory now
2. PREFETCH(nodes[node.right])
3. 
4. // By the time we need children, they're in cache
5. IF features[node.feature] <= node.threshold:
6.     RETURN TraverseWithPrefetch(node.left, features)
7. ELSE:
8.     RETURN TraverseWithPrefetch(node.right, features)
```

Hardware prefetch works well for sequential access but struggles with tree traversal's irregular patterns. Explicit prefetching can help.

## Parallelism Strategies

### Data Parallelism (Across Samples)

Partition samples across threads:

```
PARALLEL FOR block IN data.chunks(BLOCK_SIZE):
    FOR tree IN forest:
        process(tree, block)
```

**Pros**: Simple, good load balancing, linear scaling
**Cons**: Each thread loads all trees (duplicated cache pressure)

### Model Parallelism (Across Trees)

Partition trees across threads:

```
FOR block IN data.chunks(BLOCK_SIZE):
    partial_sums <- PARALLEL FOR tree_chunk IN forest.chunks(TREES_PER_THREAD):
        sum(traverse(tree, block) FOR tree IN tree_chunk)
    output[block] <- sum(partial_sums)
```

**Pros**: Each thread caches subset of trees
**Cons**: Requires reduction, less balanced if trees vary in size

### Hybrid Approach

For large workloads, combine both:

```
// Outer: partition samples across thread groups
// Inner: partition trees within each group

PARALLEL FOR sample_chunk IN data.chunks(SAMPLES_PER_GROUP):
    local_output <- zeros(sample_chunk.size)
    
    PARALLEL FOR tree_chunk IN forest.chunks(TREES_PER_THREAD):
        FOR tree IN tree_chunk:
            accumulate(local_output, traverse(tree, sample_chunk))
    
    output[sample_chunk] <- local_output
```

## SIMD Considerations

### What Vectorizes Well

1. **Threshold comparisons** (same threshold for multiple samples at same node):

```
// 8 samples at same node, comparing against one threshold
thresholds <- BROADCAST(node.threshold)  // [t, t, t, t, t, t, t, t]
features <- LOAD_8_FLOATS(staged_features[0:8])
go_left_mask <- COMPARE_LE(features, thresholds)  // 8 comparisons, 1 instruction
```

2. **Position updates** (predictable arithmetic):

```
positions <- 2 * positions + decisions  // SIMD multiply-add
```

3. **Leaf accumulation** (independent additions):

```
outputs <- outputs + leaf_values  // SIMD add
```

### What Doesn't Vectorize

1. **Feature gathering** (different feature index per sample):

```
// Each sample needs different feature
FOR i FROM 0 TO 8:
    features[i] <- data[sample_idx[i], feat_idx[i]]  // Random access
```

SIMD gather instructions exist but are slower than contiguous loads.

2. **Divergent paths** (samples in different subtrees):

```
// After samples diverge to different nodes, synchronization breaks down
// Cannot vectorize across samples in different tree branches
```

### Practical SIMD Strategy

Focus SIMD on predictable parts:
- **Level-by-level comparison**: All samples at same level, vectorize threshold comparison
- **Leaf accumulation**: Vectorize adding leaf values to outputs
- **Multi-tree parallelism**: Process 4/8 trees simultaneously for one sample

## GPU Considerations

### Why GPU Needs Blocks

GPU efficiency requires thousands of threads executing similar operations. Block-based traversal is essential:

```
GPU Kernel: One thread per sample
__global__ void predict(samples, trees, outputs):
    sample_idx <- blockIdx.x * blockDim.x + threadIdx.x
    
    // Load tree structure to shared memory (block-wide)
    __shared__ TreeNodes shared_tree
    cooperative_load(trees[tree_idx], shared_tree)
    __syncthreads()
    
    // All threads in block traverse same tree (from shared memory)
    output <- traverse(shared_tree, samples[sample_idx])
    outputs[sample_idx] += output
```

### Warp Efficiency

Threads in a warp (32 threads) execute in lockstep. Divergent branches hurt performance:

```
Thread 0: goes left  --+
Thread 1: goes right   |-- Warp divergence (both paths execute, half threads idle)
Thread 2: goes left    |
...                  --+
```

Level-synchronized traversal and proper tree layout minimize divergence by keeping samples at similar tree positions.

## Summary

| Technique | Benefit | Implementation |
|-----------|---------|----------------|
| **Block processing** | Keep tree in cache | Process 64 samples per tree |
| **Level synchronization** | Predictable access | All samples at same depth |
| **Tree unrolling** | No pointer chasing | Implicit indexing for top 6 levels |
| **Feature staging** | Cache-friendly | Pre-load sample features |
| **SIMD comparison** | 8x throughput | Vectorize threshold comparisons |
| **Multi-tree** | Better utilization | Process multiple trees per sample |

### Key References

- `xgboost/src/predictor/cpu_predictor.cc` — Block traversal implementation
- XGBoost array tree layout (6-level unrolling)
- `LightGBM/src/boosting/gbdt.cpp` — Batch prediction with iterators
- "What Every Programmer Should Know About Memory" — Cache hierarchy deep dive
