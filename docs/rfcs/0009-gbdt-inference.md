# RFC-0009: GBDT Inference

**Status**: Implemented  
**Created**: 2025-12-15  
**Updated**: 2026-01-02  
**Scope**: Tree ensemble prediction pipeline

## Summary

GBDT inference traverses each tree with input features to accumulate leaf
values. Block-based processing and optional tree unrolling optimize throughput.

## Why Sample-Major for Prediction?

Training uses feature-major layout (features as columns) for histogram building.
Prediction needs sample-major (samples as rows) because:

| Access Pattern | Feature-Major | Sample-Major |
| -------------- | ------------- | ------------ |
| Traversal | Random access per feature | Sequential per sample |
| Cache behavior | Poor (jump between features) | Good (all features together) |

Tree traversal accesses features in split order (unpredictable), so keeping
each sample's features contiguous improves cache hits.

### Block-Based Layout Conversion

Rather than converting the entire dataset upfront (expensive for large data),
we use `buffer_samples()` to convert feature-major to sample-major in blocks:

```rust
// Feature-major input: [n_features, n_samples]
// Sample-major buffer: [block_size, n_features]

let mut buffer = Array2::<f32>::zeros((block_size, n_features));

for block_start in (0..n_samples).step_by(block_size) {
    let samples = dataset.buffer_samples(&mut buffer, block_start);
    // samples is now sample-major, contiguous in memory
    for sample in samples.iter() {
        traverse_all_trees(sample);
    }
}
```

This approach:

- Converts only what fits in L2 cache (~256KB = 64 samples × 1000 features × 4 bytes)
- Amortizes conversion cost across multiple trees
- Reuses the buffer across blocks (no repeated allocation)
- Works with sparse features (fills zeros appropriately)

## Layers

### High Level

Users call `GBDTModel::predict`:

```rust
let predictions = model.predict(&test_data, n_threads);
```

This creates a `Predictor` internally and returns raw predictions.

### Medium Level (Predictor)

```rust
pub struct Predictor<'f, T: TreeTraversal<ScalarLeaf>> {
    forest: &'f Forest<ScalarLeaf>,
    tree_states: Box<[T::TreeState]>,
    block_size: usize,
}

impl<'f, T: TreeTraversal<ScalarLeaf>> Predictor<'f, T> {
    pub fn new(forest: &'f Forest<ScalarLeaf>) -> Self;
    pub fn predict(&self, data: &Dataset, parallelism: Parallelism) -> Array2<f32>;
}
```

The `TreeTraversal` trait parameter allows different traversal strategies.

### Medium Level (TreeTraversal)

```rust
pub trait TreeTraversal<L: LeafValue>: Clone {
    type TreeState: Clone + Send + Sync;
    
    fn build_tree_state(tree: &Tree<L>) -> Self::TreeState;
    fn traverse_tree(tree: &Tree<L>, state: &Self::TreeState, sample: ArrayView1<f32>) -> NodeId;
    fn traverse_block(tree: &Tree<L>, state: &Self::TreeState, data: &SamplesView, out: &mut [NodeId]);
}
```

### Low Level (Traversal Strategies)

```rust
// Simple: no precomputation, traverse nodes one by one
pub struct StandardTraversal;

// Unrolled: precompute flattened layout for first D levels
pub struct UnrolledTraversal<D: UnrollDepth>;
pub type UnrolledTraversal4 = UnrolledTraversal<Depth4>;
pub type UnrolledTraversal6 = UnrolledTraversal<Depth6>;
```

## Block Processing

Rather than traverse one tree per sample, we process blocks of samples
through all trees:

```text
Block of 64 samples
    │
    ├─► Tree 0 ─► 64 leaf indices ─► accumulate
    ├─► Tree 1 ─► 64 leaf indices ─► accumulate
    └─► Tree N ─► 64 leaf indices ─► accumulate
```

Benefits:

- Tree data (splits, thresholds) stays in L1/L2 cache
- Multiple samples amortize cache miss cost
- Predictable memory access for leaf accumulation

Default block size: 64 (matches XGBoost).

## Unrolled Traversal

Traditional traversal:

```rust
loop {
    if is_leaf(node) { return node; }
    node = if feature[split_idx] < threshold { left } else { right };
}
```

Branch misprediction penalty is ~15-20 cycles per miss. Unrolled traversal
precomputes a flattened array for the first D levels:

```rust
pub struct UnrolledTreeLayout<D: UnrollDepth> {
    unrolled: Box<[UnrolledNode]>,    // 2^D - 1 nodes
    subtree_offsets: Box<[NodeId]>,   // Map to original tree
}
```

For depth-6 unrolling (63 nodes), we do ~6 branchless comparisons, then
continue from the subtree root. 2-3× faster for large batches.

## Multi-Output Handling

For K-class classification, the forest contains K trees per round. Output
shape is `[n_samples, n_groups]`. Trees are assigned to groups round-robin,
and predictions accumulate per-group.

```rust
for tree_idx in 0..forest.n_trees() {
    let group = forest.tree_group(tree_idx);
    for sample in 0..n_samples {
        output[[sample, group]] += leaf_value;
    }
}
```

### Prediction Outputs

Raw predictions are logits/scores. The `predict()` method applies the appropriate
transformation based on the objective:

```rust
// predict() applies transformation (sigmoid for binary, softmax for multiclass)
let predictions = model.predict(&data, n_threads);

// predict_raw() returns raw margin scores (no transformation)
let raw = model.predict_raw(&data, n_threads);
```

Note: There is no `predict_proba()` method on `GBDTModel`. The `predict()` method
already applies probability transformations. Use sklearn wrappers if you need
separate `predict()` (class labels) and `predict_proba()` (probabilities).

## Parallelization

With `Parallelism::Parallel(n_threads)`:

```rust
// Parallelize over sample blocks
samples.par_chunks(block_size).for_each(|block| {
    for tree in forest.trees() {
        traverse_block(tree, block, &mut local_output);
    }
});
```

Thread overhead is amortized by block size. For small datasets, sequential
is often faster (no thread spawn cost).

## Files

| Path | Contents |
| ---- | -------- |
| `inference/gbdt/predictor.rs` | `Predictor<T>`, batch/parallel prediction |
| `inference/gbdt/traversal.rs` | `TreeTraversal` trait, `StandardTraversal`, `UnrolledTraversal` |
| `inference/gbdt/unrolled.rs` | `UnrolledTreeLayout`, unrolling logic |
| `inference/gbdt/mod.rs` | Module exports |

## Design Decisions

**DD-1: Block size 64.** Empirically optimal across CPU architectures. Matches
XGBoost default. Can be customized via `with_block_size()`.

**DD-2: Traversal as generic parameter.** Compile-time strategy selection
enables inlining and specialization. No virtual dispatch overhead.

**DD-3: Unroll depth 6.** Covers 63 nodes (first 6 levels). Most trees fit
entirely or have only deep leaves remaining. Depth 8 (255 nodes) showed
minimal additional benefit with larger memory footprint.

**DD-4: Dataset reuse for prediction.** The same `Dataset` type is used for
training and prediction. For training, binning converts to `BinnedDataset`.
For prediction, `buffer_samples()` provides sample-major blocks on-demand.

## Benchmarks

Inference throughput (Covertype, 54 features, 100 trees, depth 6):

| Configuration | Throughput |
| ------------- | ---------- |
| Single-threaded, standard | ~2M samples/sec |
| Single-threaded, unrolled | ~3M samples/sec |
| 8 threads, unrolled | ~15M samples/sec |

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Traversal correctness | Compare standard vs unrolled (same results) |
| XGBoost compatibility | Load XGBoost model, compare predictions |
| LightGBM compatibility | Load LightGBM model, compare predictions |
| Edge cases | Empty forest, single tree, max depth trees |
| Parallelism | Same results with 1, 4, 8 threads |
