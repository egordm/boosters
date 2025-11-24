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
```
out_predt: [ r0_g0, r0_g1, r1_g0, r1_g1, r2_g0, r2_g1, ... ]
Block 0 updates first B rows range; Block 1 updates next B rows range.
```

### Implementation note
Block-based traversal is widely used in `cpu_predictor.cc` as well as `gpu_predictor` for high throughput. The block size is an important tuning parameter; too small loses vectorization benefits, too large increases memory footprint and may reduce locality.

## Training vs Inference
- Training: Block-based traversal is useful in training for operations like computing predictions across minibatches, caching predictions for gradient computation, or parallelizing updates. However, histogram-based split-finding and node statistics computation often dominate training performance and use different traversal/accumulation patterns.
- Inference: Block-based traversal is a primary performance optimization for inference; it enables efficient cache reuse, vectorized execution, and GPU-friendly batched work. Converting to blocked workloads is straightforward at inference time and has minimal effect on model correctness.
- Notes: A single `predict` implementation can reuse block-based traversal for both training prediction needs and inference, but training-specific code may require additional bookkeeping (weights, sample masks, gradients).
