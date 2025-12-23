# RFC-0003: Inference Pipeline

- **Status**: Implemented
- **Created**: 2024-11-15
- **Updated**: 2025-01-21
- **Depends on**: RFC-0001 (Data Matrix), RFC-0002 (Forest/Trees)
- **Scope**: Batch and single-row prediction for GBDT ensembles

## Summary

The inference pipeline provides batch and single-row prediction for GBDT ensembles using pluggable traversal strategies. The `Predictor<T>` struct orchestrates tree traversal, block processing, and parallel execution via Rayon. Trees are traversed using the `TreeView` trait and `FeatureAccessor` for data access.

## Design Overview

```
┌─────────────────────────────────────────────────────────────┐
│                       Predictor<T>                           │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ forest: &Forest<ScalarLeaf>                              ││
│  │ tree_states: Box<[T::TreeState]>  // Pre-computed        ││
│  │ block_size: usize = 64            // Default             ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ trait TreeTraversal<L>                                   ││
│  │   type TreeState                                         ││
│  │   const USES_BLOCK_OPTIMIZATION: bool                    ││
│  │   fn build_tree_state(tree) → TreeState                  ││
│  │   fn traverse_tree(tree, state, features) → L            ││
│  │   fn traverse_block(tree, state, buffer, ...) → output   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
          ↑                                      ↑
          │                                      │
┌─────────────────────┐            ┌─────────────────────────┐
│ StandardTraversal   │            │ UnrolledTraversal<D>    │
│ State: ()           │            │ State: UnrolledTreeLayout│
│ Block opt: No       │            │ Block opt: Yes          │
└─────────────────────┘            └─────────────────────────┘
```

## Core Types

### Predictor

```rust
pub struct Predictor<'f, T: TreeTraversal<ScalarLeaf>> {
    forest: &'f Forest<ScalarLeaf>,
    tree_states: Box<[T::TreeState]>,  // Pre-computed per tree
    block_size: usize,                  // Default: 64
}

impl<'f, T: TreeTraversal<ScalarLeaf>> Predictor<'f, T> {
    /// Create predictor with pre-computed tree states.
    pub fn new(forest: &'f Forest<ScalarLeaf>) -> Self;
    
    /// Batch prediction using any FeatureAccessor.
    pub fn predict<A: FeatureAccessor + Sync>(
        &self,
        features: &A,
        parallelism: Parallelism,
    ) -> Array2<f32>;
    
    /// Batch prediction into pre-allocated buffer.
    pub fn predict_into<A: FeatureAccessor + Sync>(
        &self,
        features: &A,
        output: &mut Array2<f32>,
        parallelism: Parallelism,
    );
    
    /// Single-row prediction.
    pub fn predict_row(&self, features: &[f32]) -> Vec<f32>;
    
    /// Prediction with per-tree weights (DART-style).
    pub fn predict_weighted<A: FeatureAccessor + Sync>(
        &self,
        features: &A,
        tree_weights: &[f32],
        parallelism: Parallelism,
    ) -> Array2<f32>;
}
```

### Type Aliases

```rust
pub type SimplePredictor<'f> = Predictor<'f, StandardTraversal>;
pub type UnrolledPredictor4<'f> = Predictor<'f, UnrolledTraversal4>;
pub type UnrolledPredictor6<'f> = Predictor<'f, UnrolledTraversal6>;
pub type UnrolledPredictor8<'f> = Predictor<'f, UnrolledTraversal8>;
```

## TreeTraversal Trait

```rust
pub trait TreeTraversal<L: LeafValue>: Clone {
    type TreeState: Clone + Send + Sync;
    
    /// Whether this strategy benefits from block processing.
    const USES_BLOCK_OPTIMIZATION: bool = false;
    
    /// Build state from tree (called once at Predictor creation).
    fn build_tree_state(tree: &Tree<L>) -> Self::TreeState;
    
    /// Traverse a single row to its leaf.
    fn traverse_tree(tree: &Tree<L>, state: &Self::TreeState, features: &[f32]) -> L;
    
    /// Process a block of rows through the tree (optional optimization).
    fn traverse_block(
        tree: &Tree<L>,
        state: &Self::TreeState,
        feature_buffer: &[f32],
        n_features: usize,
        output: &mut [f32],
        weight: f32,
    );
}
```

### Traversal Strategies

| Strategy | State | Block Opt | Use Case |
| -------- | ----- | --------- | -------- |
| `StandardTraversal` | `()` | No | Simple node-by-node. Good for single rows. |
| `UnrolledTraversal<D>` | `UnrolledTreeLayout<D>` | Yes | Pre-computes flat layout for top D levels. 2-3x faster for large batches. |

### Unroll Depths

| Type | Nodes in Flat Array | Notes |
| ---- | ------------------- | ----- |
| `Depth4` | 15 | Small trees |
| `Depth6` | 63 | Default, good balance |
| `Depth8` | 255 | Deep trees |

## Tree Traversal

### Standard Traversal

Uses `TreeView::traverse_to_leaf` (see RFC-0002):

1. Start at root (node 0)
2. Read feature value at `split_index`
3. Handle missing: use `default_left` flag
4. Numeric split: `value < threshold` → left
5. Categorical split: check bitset → right if bit set
6. Continue until `is_leaf == true`

### Unrolled Traversal

Uses `UnrolledTreeLayout` for cache efficiency:

```rust
pub struct UnrolledTreeLayout<D: UnrollDepth> {
    flat: Box<[UnrolledNode]>,  // Top D levels in flat array
    marker: PhantomData<D>,
}

pub struct UnrolledNode {
    split_idx: u32,
    threshold: f32,
    exit_idx: u32,  // Points to Tree node for subtree traversal
}
```

**Two-phase traversal**:

1. **Phase 1**: Traverse top D levels using simple index arithmetic:
   - `left = 2 * i + 1`
   - `right = 2 * i + 2`
   - No pointer chasing, highly cacheable

2. **Phase 2**: Continue from exit node to leaf using standard traversal

**Block processing**: All rows traverse the same tree level together, keeping level data in L1/L2 cache.

## Data Access via FeatureAccessor

Prediction is generic over any `FeatureAccessor` (RFC-0001):

```rust
// Works with SamplesView
let samples = SamplesView::from_array(arr.view());
let output = predictor.predict(&samples, Parallelism::Parallel);

// Works with FeaturesView
let features = FeaturesView::from_array(arr.view());
let output = predictor.predict(&features, Parallelism::Sequential);

// Works with BinnedAccessor (during training)
let accessor = BinnedAccessor::new(&binned_dataset);
let output = predictor.predict(&accessor, Parallelism::Sequential);
```

### BinnedAccessor

For training-time prediction on quantized data:

```rust
pub struct BinnedAccessor<'a> {
    dataset: &'a BinnedDataset,
}

impl FeatureAccessor for BinnedAccessor<'_> {
    fn get_feature(&self, row: usize, feature: usize) -> f32 {
        // Convert bin index to midpoint value
        let bin = self.dataset.bin(row, feature);
        self.dataset.bin_mapper(feature).bin_to_value(bin)
    }
}
```

## Output Format

Predictions are `Array2<f32>` with shape `[n_groups, n_samples]`:

```rust
// Shape: [n_groups, n_samples]
let predictions = predictor.predict(&features, Parallelism::Parallel);

// Access group outputs (contiguous in memory)
let group_0 = predictions.row(0);  // All samples for group 0

// Access sample outputs (requires gather)
let sample_0: Vec<f32> = (0..n_groups)
    .map(|g| predictions[[g, 0]])
    .collect();
```

### Prediction Kind

The `Predictions` wrapper adds semantic meaning:

```rust
pub enum PredictionKind {
    Margin,       // Raw model output
    Value,        // Regression value
    Probability,  // After sigmoid/softmax
    ClassIndex,   // Predicted class
    RankScore,    // Ranking score
}

pub struct Predictions {
    pub kind: PredictionKind,
    pub output: Array2<f32>,
}
```

## Parallelism

Block-based parallel execution:

```rust
match parallelism {
    Parallelism::Sequential => {
        for tree in forest.trees() {
            tree.predict_into(features, predictions, parallelism);
        }
    }
    Parallelism::Parallel => {
        // Rayon parallel iteration over sample blocks
        predictions.axis_chunks_iter_mut(...)
            .par_for_each(|chunk| { ... });
    }
}
```

**Block size**: Default 64 rows per block (matches XGBoost). Features loaded into contiguous buffer per block for cache locality.

## Performance Optimizations

1. **Pre-computed Tree State**: `tree_states` built once at predictor creation, reused across predictions

2. **Unrolled Tree Layout**: Top tree levels in contiguous array with simple index math. Avoids pointer-chasing.

3. **Block Processing**: All rows traverse same tree level together → L1/L2 cache locality

4. **Level-by-Level Traversal**: Process all rows at each level before moving deeper

5. **Stack Allocation**: Exit indices use stack array for blocks ≤256 rows

6. **Rayon Parallelism**: Blocks distributed across threads with work-stealing

## Usage Examples

### Basic Prediction

```rust
use boosters::inference::gbdt::{Predictor, UnrolledTraversal6};
use boosters::data::SamplesView;
use boosters::Parallelism;
use ndarray::Array2;

let forest: Forest<ScalarLeaf> = /* load or train */;
let predictor = Predictor::<UnrolledTraversal6>::new(&forest);

let arr = Array2::from_shape_vec((100, 10), data)?;
let features = SamplesView::from_array(arr.view());

// Parallel prediction
let output = predictor.predict(&features, Parallelism::Parallel);
// output: Array2<f32> with shape [n_groups, 100]
```

### Single Row

```rust
let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
let output = predictor.predict_row(&features);
// output: Vec<f32> with length n_groups
```

### Weighted Trees (DART)

```rust
let tree_weights = vec![1.0; forest.n_trees()];
let output = predictor.predict_weighted(&features, &tree_weights, Parallelism::Parallel);
```

## Design Decisions

### DD-1: Generic over FeatureAccessor

**Context**: How should predictor access feature data?

**Decision**: Generic over `impl FeatureAccessor`.

**Rationale**:
- Works with any layout (SamplesView, FeaturesView, BinnedAccessor)
- Training can use same predictor with binned data
- Zero-cost abstraction via monomorphization

### DD-2: Pre-computed TreeState

**Context**: When to build unrolled tree layouts?

**Decision**: Build at `Predictor::new()`, store in `tree_states`.

**Rationale**:
- One-time cost amortized over many predictions
- State is immutable, can be shared across threads
- `Box<[TreeState]>` matches forest size exactly

### DD-3: Block Size 64

**Context**: How many rows to process per block?

**Decision**: Default 64 rows (matches XGBoost).

**Rationale**:
- Fits in L1 cache with reasonable feature counts
- Good balance between parallelism granularity and overhead
- Configurable via builder pattern if needed

### DD-4: Output Layout [n_groups, n_samples]

**Context**: Prediction output shape?

**Decision**: `[n_groups, n_samples]` (group-major).

**Rationale**:
- Each tree adds to one group's row (contiguous writes)
- Base score initialization per group is contiguous
- Matches gradient layout for consistency

## Integration

| Component | Integration Point |
| --------- | ----------------- |
| RFC-0001 (Data) | `Predictor::predict<A: FeatureAccessor>` |
| RFC-0002 (Trees) | `TreeView::traverse_to_leaf` for traversal |
| RFC-0007 (Growing) | Training uses `BinnedAccessor` for validation |
| RFC-0008 (Objectives) | Transforms margins to predictions |
| RFC-0009 (Metrics) | Evaluates prediction output |

## Future Work

1. **SIMD Vectorization**: Process 4-8 rows simultaneously using `std::simd`
2. **GPU Inference**: CUDA/Metal acceleration for large batches
3. **Streaming**: Process data in chunks for memory efficiency

## Changelog

- 2025-01-21: Major update. Added `FeatureAccessor` integration, updated output format to `Array2`, documented `BinnedAccessor`, updated for `TreeView` trait.
- 2024-11-15: Initial RFC with traversal strategies
