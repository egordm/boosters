# Block-based Traversal

## ELI13
Instead of checking one sample at a time through the tree, we group several samples into a chunk (block) and process that block together. This helps the computer reuse memory and do many things in parallel quickly.

## ELI Grad
Block-based traversal processes rows in blocks (e.g., 32–128 rows) and traverses those rows through a tree forest using a tiled approach. The implementation typically fills a small working buffer of feature vectors (`FVec`) per thread, then executes a depth-first (or array-layout) traversal for all rows in the block, leveraging cached/loaded node metadata and enabling vectorized or GPU-friendly operations.

### Benefits
- Reduces repeated loads from memory: multiple rows will reuse node metadata loaded into L1/L2 or GPU shared memory.
- Enables batched evaluation and branch elimination: some alternations between rows are handled together.
- Allows using array-tree layout optimizations and vectorization.

### Example sketch
- Block size `B = 64` rows.
- For each block:
  - Load `FVec` for `B` rows into a per-thread local buffer.
  - For each tree (or vectorized set of trees): do `PredValueByOneTree` to compute leaf value for each row in the block, storing results as `out[block_index * n_groups + group]`.

### Data structures used
- `ThreadTmp<B>`: thread-local `Vec<RegTree::FVec>` to avoid reallocation.
- `out_predt`: `n_rows × n_groups` contiguous buffer (linalg::TensorView). Each block updates a slice of rows.

### Example memory view

```text
out_predt: [ r0_g0, r0_g1, r1_g0, r1_g1, r2_g0, r2_g1, ... ]
Block 0 updates first B rows range; Block 1 updates next B rows range.
```

### Implementation note
Block-based traversal is widely used in `cpu_predictor.cc` as well as `gpu_predictor` for high throughput. The block size is an important tuning parameter; too small loses vectorization benefits, too large increases memory footprint and may reduce locality.

## CPU vs GPU: Where Block-Based Traversal Helps

### CPU Benefits

Block-based traversal is **highly beneficial on CPU**:

1. **Cache Locality**: When processing 64 rows through a tree's top levels, the tree nodes stay in L1/L2 cache. Without blocking, each row might evict tree nodes before the next row uses them.

2. **Branch Prediction**: Level-by-level processing with `ArrayTreeLayout` converts data-dependent branches into predictable loops:

   ```rust
   // Without blocking: unpredictable branches per row
   for row in rows {
       let mut idx = 0;
       while !is_leaf[idx] {  // Data-dependent branch
           idx = if features[row][split_idx[idx]] < threshold[idx] {
               left[idx]
           } else {
               right[idx]
           };
       }
   }
   
   // With blocking + ArrayTreeLayout: predictable loop
   for level in 0..DEPTH {  // Fixed iteration count
       for (row_idx, pos) in positions.iter_mut().enumerate() {
           // All rows process same level simultaneously
           *pos = next_position(level, *pos, features[row_idx]);
       }
   }
   ```

3. **SIMD Opportunity**: When all rows are at the same tree level, we can vectorize the comparison:

   ```rust
   // 8 rows, same feature index at this level
   let thresholds = f32x8::splat(tree.threshold[level_node]);
   let features = f32x8::from_slice(&row_features[..8]);
   let go_left = features.simd_lt(thresholds);
   ```

### GPU Benefits

Block-based traversal is **essential on GPU**:

1. **Warp Efficiency**: All 32 threads in a warp should execute the same instruction. Level-by-level traversal keeps threads synchronized.

2. **Shared Memory**: Tree nodes for top levels can be loaded into shared memory once, reused by all threads in a block.

3. **Coalesced Access**: Rows are processed together, enabling coalesced memory reads for features.

### CPU Implementation Value

Having a CPU implementation of block-based traversal is valuable even if targeting GPU:

- **Testing**: Easier to debug correctness without GPU complexity
- **Fallback**: Works on machines without GPU
- **Reference**: Validates GPU results against CPU
- **Profiling**: Isolate algorithmic performance from GPU overhead

## ArrayTreeLayout: SIMD Considerations

The main challenge with SIMD in tree traversal is the **gather operation**: each row may need a different feature index and threshold, requiring scatter-gather patterns that are slower than sequential loads on current hardware.

Key insights:

- **What vectorizes well**: Comparisons, position updates, leaf accumulation
- **What doesn't**: Feature/threshold lookups (different indices per row)
- **Recommendation**: Start with scalar gather loops; hardware gather may or may not help (benchmark!)

For detailed analysis of SIMD optimization in ArrayTreeLayout, see [RFC-0002: Tree Data Structures](../architecture/0002-tree-data-structures.md#simd-optimization-analysis).

## Training vs Inference

- **Training**: Block-based traversal is useful in training for operations like computing predictions across minibatches, caching predictions for gradient computation, or parallelizing updates. However, histogram-based split-finding and node statistics computation often dominate training performance and use different traversal/accumulation patterns.

- **Inference**: Block-based traversal is a primary performance optimization for inference; it enables efficient cache reuse, vectorized execution, and GPU-friendly batched work. Converting to blocked workloads is straightforward at inference time and has minimal effect on model correctness.

- **Notes**: A single `predict` implementation can reuse block-based traversal for both training prediction needs and inference, but training-specific code may require additional bookkeeping (weights, sample masks, gradients).
