# RFC-0003: Inference Pipeline

- **Status**: Implemented
- **Created**: 2024-11-15
- **Updated**: 2025-01-21
- **Depends on**: RFC-0001 (Data Matrix), RFC-0002 (Forest/Trees)
- **Scope**: Batch and single-row prediction for GBDT ensembles

## Summary

The inference pipeline provides batch and single-row prediction for GBDT ensembles using pluggable traversal strategies. The `Predictor<T>` struct orchestrates tree traversal, block processing, and parallel execution via Rayon. Trees are traversed using `TreeView` and `SampleAccessor` traits for data access.

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
    
    /// Batch prediction (allocating version).
    /// 
    /// Features must be C-contiguous ArrayView2 with shape [n_samples, n_features].
    pub fn predict(
        &self,
        features: ArrayView2<f32>,
        parallelism: Parallelism,
    ) -> Array2<f32>;
    
    /// Batch prediction into pre-allocated buffer.
    /// 
    /// - `features`: [n_samples, n_features], C-contiguous
    /// - `output`: [n_groups, n_samples], mutable view
    /// - `weights`: Optional per-tree weights for DART-style prediction
    pub fn predict_into(
        &self,
        features: ArrayView2<f32>,
        weights: Option<&[f32]>,
        parallelism: Parallelism,
        output: ArrayViewMut2<f32>,
    );
    
    /// Single-row prediction from a slice.
    /// 
    /// Uses slices instead of ndarray for single-row efficiency (avoids view overhead).
    /// This follows the convention that single-row APIs use `&[f32]` while batch APIs
    /// use `ArrayView2<f32>`.
    pub fn predict_row_into(
        &self,
        features: &[f32],
        weights: Option<&[f32]>,
        output: &mut [f32],
    );
}
```

**Note on DataAccessor**: While the `TreeView::traverse_to_leaf` method uses `SampleAccessor` generically, the `Predictor` takes `FeaturesView` directly for efficiency. The predictor internally transposes blocks to sample-major order and passes rows to traversal methods as slices.

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

## Data Access via DataAccessor

Prediction internally uses the `DataAccessor` and `SampleAccessor` traits (RFC-0001), but the `Predictor` API accepts concrete view types for efficiency:

```rust
// FeaturesView for batch prediction (feature-major)
let features = FeaturesView::from_array(arr.view());
let output = predictor.predict(features, Parallelism::Parallel);

// Raw slice for single-row prediction  
let features: &[f32] = &[0.1, 0.2, 0.3];
predictor.predict_row_into(features, None, &mut output);
```

Note: The predictor takes `FeaturesView` (feature-major) for batches because the internal block processing transposes to sample-major per block for cache efficiency. For single rows, raw `&[f32]` slices are used to avoid view overhead.

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

### DD-1: FeaturesView Input with Block Transpose

**Context**: How should predictor access feature data?

**Decision**: Take `FeaturesView` (feature-major) and transpose internally per block.

**Rationale**:

- Feature-major storage is the standard format from training (`BinnedDataset` layout)
- Block processing transposes to sample-major for tree traversal cache efficiency
- Per-block transpose fits in L2 cache (~25KB for 64 rows × 100 features)
- Single-row API uses `&[f32]` to avoid view overhead

**API Layers**: Lower-level representation APIs (e.g., `repr::gblinear::LinearModel`) use `SamplesView` for type safety. Higher-level model APIs (e.g., `model::GBLinearModel`) accept `ArrayView2` for convenience, assuming users provide valid C-contiguous data.

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
| RFC-0001 (Data) | `SamplesView`, `FeaturesView`, `SampleAccessor` |
| RFC-0002 (Trees) | `TreeView::traverse_to_leaf` for traversal |
| RFC-0007 (Growing) | Training uses prediction for validation |
| RFC-0008 (Objectives) | Transforms margins to predictions |
| RFC-0009 (Metrics) | Evaluates prediction output |

## Future Work

1. **SIMD Vectorization**: Process 4-8 rows simultaneously using `std::simd`
2. **GPU Inference**: CUDA/Metal acceleration for large batches
3. **Streaming**: Process data in chunks for memory efficiency

## Changelog

- 2025-01-23: Updated data access section to use `DataAccessor`/`SampleAccessor` and `FeaturesView`. Removed `BinnedAccessor` section (handled internally by training).
- 2025-01-23: Added API layering note to DD-1. Documented single-row slice convention. Clarified contiguity requirement.
- 2025-01-23: Fixed Predictor interface to match implementation (uses `FeaturesView` not `FeatureAccessor`). Updated method signatures.
- 2025-01-21: Major update. Added `FeatureAccessor` integration, updated output format to `Array2`, documented `BinnedAccessor`, updated for `TreeView` trait.
- 2024-11-15: Initial RFC with traversal strategies
