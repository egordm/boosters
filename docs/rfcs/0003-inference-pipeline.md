# RFC-0003: Inference Pipeline

**Status**: Implemented

## Summary

The inference pipeline provides batch and single-row prediction for GBDT ensembles using pluggable traversal strategies. The `Predictor<T>` struct orchestrates tree traversal, block processing, and parallel execution via Rayon.

## Design

### Tree Traversal

The `TreeTraversal<L>` trait abstracts how trees are traversed during prediction:

```rust
pub trait TreeTraversal<L: LeafValue>: Clone {
    type TreeState: Clone + Send + Sync;
    const USES_BLOCK_OPTIMIZATION: bool = false;

    fn build_tree_state(tree: &Tree<L>) -> Self::TreeState;
    fn traverse_tree(tree: &Tree<L>, state: &Self::TreeState, features: &[f32]) -> L;
    fn traverse_block(tree, state, feature_buffer, num_features, output, weight);
}
```

**Implementations:**

| Strategy | State | Block Opt | Use Case |
|----------|-------|-----------|----------|
| `StandardTraversal` | `()` | No | Simple node-by-node traversal. Good for single rows or small batches. |
| `UnrolledTraversal<D>` | `UnrolledTreeLayout<D>` | Yes | Pre-computes flat array for top D levels. 2-3x faster for large batches. |
| `SimdTraversal<D>` | `UnrolledTreeLayout<D>` | Yes | Future SIMD optimization placeholder (currently same as unrolled). |

**Unroll Depths:** `Depth4` (15 nodes), `Depth6` (63 nodes, default), `Depth8` (255 nodes).

### Tree Traversal Details

Standard traversal iterates node-by-node from root to leaf:

1. Read feature value at `split_index`
2. Handle missing values via `default_left` flag
3. For numeric splits: `fvalue < threshold` → left
4. For categorical splits: check category bitset membership → right if set
5. Continue until `is_leaf` is true

Unrolled traversal uses `UnrolledTreeLayout`:

1. **Phase 1**: Traverse top D levels using simple index arithmetic (`left = 2*i + 1`, `right = 2*i + 2`)
2. **Phase 2**: Continue from exit node to leaf using standard traversal

Block processing (`traverse_block`) processes all rows through the same tree level together, keeping level data in L1/L2 cache.

### Prediction Orchestration

`Predictor<'f, T>` manages batch prediction:

```rust
pub struct Predictor<'f, T: TreeTraversal<ScalarLeaf>> {
    forest: &'f Forest<ScalarLeaf>,
    tree_states: Box<[T::TreeState]>,  // Pre-computed per tree
    block_size: usize,                  // Default: 64 (matches XGBoost)
}
```

**Methods:**

- `predict(&features)` → batch prediction (chooses simple vs block-optimized based on `T::USES_BLOCK_OPTIMIZATION`)
- `predict_row(&features)` → single-row prediction
- `predict_weighted(&features, &weights)` → DART-style weighted trees
- `par_predict(&features)` → parallel batch prediction using Rayon

**Block Processing Flow:**

1. Split rows into blocks of `block_size` (default 64)
2. Load block features into contiguous buffer
3. For each tree: traverse all block rows, accumulate results per group
4. Scatter accumulated values into output

**Parallel Prediction:**

```rust
let block_outputs: Vec<_> = blocks
    .par_iter()  // Rayon parallel iterator
    .map(|(start, end)| self.process_block_parallel(...))
    .collect();
```

Each block is independent, enabling work-stealing load balancing.

### Output Transformation

`PredictionOutput` stores predictions in row-major layout:

```rust
pub struct PredictionOutput {
    data: Vec<f32>,      // Flat: data[row * num_groups + group]
    num_rows: usize,
    num_groups: usize,   // 1 for regression, K for K-class
}
```

**Transform functions** (in `common` module):

- `sigmoid_inplace(output)` → binary classification probabilities
- `softmax_inplace(row)` → multiclass probabilities (numerically stable: subtract max before exp)
- `softmax_rows(output)` → apply softmax to each row

Output kind is tracked via `OutputKind` enum for dynamic dispatch on transform needs.

## Performance

**Key optimizations implemented:**

1. **Unrolled Tree Layout**: Top tree levels in contiguous array with simple index math. Avoids pointer-chasing, enables cache prefetching.

2. **Block Processing**: Default block size 64 (XGBoost-matching). Features loaded into contiguous buffer. All rows traverse same tree level together → cache locality.

3. **Level-by-Level Traversal**: `UnrolledTreeLayout::process_block()` processes all rows at each level before moving deeper, keeping level data hot in cache.

4. **Stack Allocation for Small Blocks**: Exit indices use `[0usize; 256]` on stack for blocks ≤256 rows, avoiding heap allocation.

5. **Rayon Parallelism**: `par_predict` distributes blocks across threads. Each block is self-contained.

6. **Pre-computed Tree State**: `tree_states` built once at predictor creation, reused across predictions.

**Type aliases for convenience:**

```rust
pub type SimplePredictor<'f> = Predictor<'f, StandardTraversal>;
pub type UnrolledPredictor6<'f> = Predictor<'f, UnrolledTraversal<Depth6>>;
```

**SIMD**: `SimdTraversal<D>` exists as a placeholder for future SIMD optimization. Currently uses the same scalar unrolled traversal for correctness-first implementation. Defines `SIMD_WIDTH = 8` as a hint for batch tuning. True SIMD vectorization (processing 4-8 rows simultaneously) is a future optimization opportunity.

## Future Improvements

1. **SIMD Vectorization**: Implement actual SIMD traversal using `std::simd` or `packed_simd` to process multiple rows in parallel through tree levels.

2. **Thread Pool Control**: `par_predict` currently uses Rayon's global thread pool. Consider adding `par_predict_with_threads(n)` to constrain parallelism.

3. **Training Integration**: Consider exposing batch prediction for use during training/evaluation instead of per-row prediction.
