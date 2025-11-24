# RFC-0003: Visitor and Traversal Patterns

- **Status**: Draft
- **Created**: 2024-11-24
- **Depends on**: RFC-0001, RFC-0002
- **Scope**: Tree traversal abstractions and prediction orchestration

## Summary

This RFC defines the visitor pattern for tree traversal, covering:

1. **`Visitor` trait**: Core abstraction for traversal behavior
2. **`Predictor`**: Orchestration of batch prediction with threading
3. **Block traversal**: Processing rows in blocks for cache efficiency
4. **Specialization**: Const-generic dispatch for missing/categorical handling

## Motivation

Tree traversal is the hot path for inference. We need:

- Zero-cost abstraction over different forest layouts
- Efficient batch processing with thread-local buffers
- Specialization to eliminate runtime branches
- Future extensibility for GPU dispatch

XGBoost uses C++ templates for specialization. We use Rust's const generics and traits.

## Design

### Visitor Trait Hierarchy

```text
                    ┌─────────────────────────────────┐
                    │      TreeVisitor<F, L>          │
                    │  (single tree, single row)      │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   ScalarVisitor │    │   VectorVisitor │    │   LeafIndexer   │
    │  (accumulate f32│    │ (accumulate vec)│    │ (return indices)│
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │       BlockVisitor<F, L>        │
                    │   (multiple rows, one tree)     │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │     ForestPredictor<F, L>       │
                    │  (all trees, batch of rows)     │
                    └─────────────────────────────────┘
```

### Core Visitor Trait

```rust
/// Visitor for traversing a single tree with a single row
pub trait TreeVisitor<F: Forest> {
    /// Output type (f32 for scalar, Vec<f32> for vector, u32 for leaf index)
    type Output;
    
    /// Visit a tree with given features, return result
    fn visit_tree(
        &self,
        tree: &SoATreeView<'_, F::Leaf>,
        features: &[f32],
    ) -> Self::Output;
}

/// Visitor that accumulates leaf values across trees
pub trait AccumulatingVisitor<F: Forest>: TreeVisitor<F> {
    /// Accumulator type
    type Accumulator: Default;
    
    /// Add tree result to accumulator
    fn accumulate(
        &self,
        acc: &mut Self::Accumulator,
        tree_result: Self::Output,
        tree_group: u32,
    );
    
    /// Finalize accumulator to output
    fn finalize(&self, acc: Self::Accumulator, base_score: &[f32]) -> Vec<f32>;
}
```

### Scalar Visitor Implementation

```rust
/// Visitor for forests with scalar leaves
pub struct ScalarTreeVisitor<const HAS_MISSING: bool, const HAS_CATEGORICAL: bool>;

impl<F, const HAS_MISSING: bool, const HAS_CATEGORICAL: bool> TreeVisitor<F> 
    for ScalarTreeVisitor<HAS_MISSING, HAS_CATEGORICAL>
where
    F: Forest<Leaf = ScalarLeaf>,
{
    type Output = f32;
    
    #[inline]
    fn visit_tree(
        &self,
        tree: &SoATreeView<'_, ScalarLeaf>,
        features: &[f32],
    ) -> f32 {
        let mut idx = 0u32;
        
        while !tree.is_leaf(idx) {
            let feat_idx = tree.split_index(idx) as usize;
            let threshold = tree.split_threshold(idx);
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);
            
            let go_left = if HAS_CATEGORICAL && tree.is_categorical(idx) {
                // Categorical split: check membership
                tree.category_contains(idx, fvalue as u32)
            } else if HAS_MISSING && fvalue.is_nan() {
                // Missing value: use default direction
                tree.default_left(idx)
            } else {
                // Numerical split
                fvalue < threshold
            };
            
            idx = if go_left {
                tree.left_child(idx)
            } else {
                tree.right_child(idx)
            };
        }
        
        tree.leaf_value(idx).0
    }
}

impl<F, const HAS_MISSING: bool, const HAS_CATEGORICAL: bool> AccumulatingVisitor<F>
    for ScalarTreeVisitor<HAS_MISSING, HAS_CATEGORICAL>
where
    F: Forest<Leaf = ScalarLeaf>,
{
    type Accumulator = Vec<f32>;  // One per group
    
    #[inline]
    fn accumulate(
        &self,
        acc: &mut Self::Accumulator,
        tree_result: f32,
        tree_group: u32,
    ) {
        acc[tree_group as usize] += tree_result;
    }
    
    fn finalize(&self, mut acc: Self::Accumulator, base_score: &[f32]) -> Vec<f32> {
        for (a, &b) in acc.iter_mut().zip(base_score) {
            *a += b;
        }
        acc
    }
}
```

### Block Visitor

```rust
/// Visitor that processes a block of rows through a tree
pub trait BlockVisitor<F: Forest> {
    /// Process a block of rows through one tree
    /// 
    /// # Arguments
    /// * `tree` - The tree to traverse
    /// * `features` - Feature values: features[row][feature]
    /// * `outputs` - Output accumulator: outputs[row][group]
    fn visit_tree_block(
        &self,
        tree: &SoATreeView<'_, F::Leaf>,
        tree_group: u32,
        features: &[&[f32]],
        outputs: &mut [&mut [f32]],
    );
}

/// Block visitor with array tree layout optimization
pub struct ArrayBlockVisitor<const DEPTH: usize, const HAS_MISSING: bool, const HAS_CATEGORICAL: bool> {
    /// Cached array layouts per tree (lazily populated)
    layouts: Vec<Option<ArrayTreeLayout<DEPTH>>>,
}

impl<F, const DEPTH: usize, const HAS_MISSING: bool, const HAS_CATEGORICAL: bool> 
    BlockVisitor<F> for ArrayBlockVisitor<DEPTH, HAS_MISSING, HAS_CATEGORICAL>
where
    F: Forest<Leaf = ScalarLeaf>,
{
    fn visit_tree_block(
        &self,
        tree: &SoATreeView<'_, ScalarLeaf>,
        tree_group: u32,
        features: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) {
        let block_size = features.len();
        
        // Phase 1: Process through unrolled array layout
        let mut subtree_roots = vec![0u32; block_size];
        
        // Note: get_or_build_layout creates an ephemeral ArrayTreeLayout per call,
        // matching XGBoost's approach. For repeated predictions, use CachedPredictor.
        if let Some(layout) = self.get_or_build_layout(tree) {
            layout.process_block(features, &mut subtree_roots);
        }
        
        // Phase 2: Continue from subtree roots to leaves
        for (i, (&root_idx, feats)) in subtree_roots.iter().zip(features.iter()).enumerate() {
            let leaf_value = self.traverse_from::<HAS_MISSING, HAS_CATEGORICAL>(
                tree, root_idx, feats
            );
            outputs[i][tree_group as usize] += leaf_value;
        }
    }
    
    #[inline]
    fn traverse_from<const MISSING: bool, const CATEGORICAL: bool>(
        &self,
        tree: &SoATreeView<'_, ScalarLeaf>,
        start_idx: u32,
        features: &[f32],
    ) -> f32 {
        let mut idx = start_idx;
        
        while !tree.is_leaf(idx) {
            let feat_idx = tree.split_index(idx) as usize;
            let threshold = tree.split_threshold(idx);
            let fvalue = features.get(feat_idx).copied().unwrap_or(f32::NAN);
            
            let go_left = if CATEGORICAL && tree.is_categorical(idx) {
                tree.category_contains(idx, fvalue as u32)
            } else if MISSING && fvalue.is_nan() {
                tree.default_left(idx)
            } else {
                fvalue < threshold
            };
            
            idx = if go_left {
                tree.left_child(idx)
            } else {
                tree.right_child(idx)
            };
        }
        
        tree.leaf_value(idx).0
    }
}
```

### Predictor Orchestration

```rust
/// High-level predictor that manages batching and threading
pub struct Predictor<F: Forest> {
    forest: Arc<F>,
    config: PredictorConfig,
}

pub struct PredictorConfig {
    /// Block size for batch processing
    pub block_size: usize,
    
    /// Number of threads (0 = auto)
    pub num_threads: usize,
    
    /// Density threshold for block vs per-row traversal
    pub block_density_threshold: f64,
    
    /// Use array tree layout optimization
    pub use_array_layout: bool,
    
    /// Array layout depth
    pub array_layout_depth: usize,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            num_threads: 0,  // auto
            block_density_threshold: 0.125,
            use_array_layout: true,
            array_layout_depth: 6,
        }
    }
}

impl<F: Forest + Send + Sync> Predictor<F> 
where
    F::Leaf: Send + Sync,
{
    pub fn new(forest: Arc<F>, config: PredictorConfig) -> Self {
        Self { forest, config }
    }
    
    /// Predict for a batch of rows
    pub fn predict(&self, features: &FeatureMatrix) -> Vec<Vec<f32>> {
        let n_rows = features.num_rows();
        let n_groups = self.forest.num_groups() as usize;
        
        // Decide traversal strategy
        let use_block = self.should_use_block(features);
        let has_missing = features.has_missing();
        let has_categorical = self.forest.has_categorical();
        
        // Dispatch to specialized implementation
        match (use_block, has_missing, has_categorical) {
            (true, true, true) => self.predict_block::<true, true>(features),
            (true, true, false) => self.predict_block::<true, false>(features),
            (true, false, true) => self.predict_block::<false, true>(features),
            (true, false, false) => self.predict_block::<false, false>(features),
            (false, true, true) => self.predict_row::<true, true>(features),
            (false, true, false) => self.predict_row::<true, false>(features),
            (false, false, true) => self.predict_row::<false, true>(features),
            (false, false, false) => self.predict_row::<false, false>(features),
        }
    }
    
    fn should_use_block(&self, features: &FeatureMatrix) -> bool {
        let density = features.density();
        density > self.config.block_density_threshold
    }
    
    fn predict_block<const HAS_MISSING: bool, const HAS_CATEGORICAL: bool>(
        &self,
        features: &FeatureMatrix,
    ) -> Vec<Vec<f32>> {
        let n_rows = features.num_rows();
        let n_groups = self.forest.num_groups() as usize;
        let block_size = self.config.block_size;
        
        // Pre-allocate output
        let mut outputs: Vec<Vec<f32>> = (0..n_rows)
            .map(|_| vec![0.0; n_groups])
            .collect();
        
        // Thread-local buffers
        thread_local! {
            static BUFFERS: RefCell<BlockBuffers> = RefCell::new(BlockBuffers::new());
        }
        
        // Process in parallel blocks
        outputs
            .par_chunks_mut(block_size)
            .enumerate()
            .for_each(|(block_idx, output_block)| {
                let row_start = block_idx * block_size;
                let row_end = (row_start + output_block.len()).min(n_rows);
                
                BUFFERS.with(|buffers| {
                    let mut buffers = buffers.borrow_mut();
                    buffers.ensure_capacity(block_size, self.forest.num_features() as usize);
                    
                    // Load features into buffer
                    for (i, row_idx) in (row_start..row_end).enumerate() {
                        features.copy_row(row_idx, &mut buffers.features[i]);
                    }
                    
                    // Process all trees
                    let visitor = ArrayBlockVisitor::<6, HAS_MISSING, HAS_CATEGORICAL>::new();
                    
                    for tree_idx in 0..self.forest.num_trees() {
                        let tree = self.forest.tree(tree_idx);
                        let tree_group = self.forest.tree_group(tree_idx);
                        
                        visitor.visit_tree_block(
                            &tree,
                            tree_group,
                            &buffers.features[..output_block.len()],
                            output_block,
                        );
                    }
                });
                
                // Add base score
                for output in output_block.iter_mut() {
                    for (o, &b) in output.iter_mut().zip(self.forest.base_score()) {
                        *o += b;
                    }
                }
            });
        
        outputs
    }
    
    fn predict_row<const HAS_MISSING: bool, const HAS_CATEGORICAL: bool>(
        &self,
        features: &FeatureMatrix,
    ) -> Vec<Vec<f32>> {
        let visitor = ScalarTreeVisitor::<HAS_MISSING, HAS_CATEGORICAL>;
        
        (0..features.num_rows())
            .into_par_iter()
            .map(|row_idx| {
                let row_features = features.row(row_idx);
                let mut output = vec![0.0; self.forest.num_groups() as usize];
                
                for tree_idx in 0..self.forest.num_trees() {
                    let tree = self.forest.tree(tree_idx);
                    let tree_group = self.forest.tree_group(tree_idx);
                    let leaf_value = visitor.visit_tree(&tree, row_features);
                    output[tree_group as usize] += leaf_value;
                }
                
                // Add base score
                for (o, &b) in output.iter_mut().zip(self.forest.base_score()) {
                    *o += b;
                }
                
                output
            })
            .collect()
    }
}
```

### Thread-Local Buffers

```rust
/// Per-thread buffers for block processing
struct BlockBuffers {
    /// Feature staging area: features[row][feature]
    features: Vec<Vec<f32>>,
    
    /// Temporary storage for subtree indices
    subtree_indices: Vec<u32>,
    
    /// Capacity tracking
    block_capacity: usize,
    feature_capacity: usize,
}

impl BlockBuffers {
    fn new() -> Self {
        Self {
            features: Vec::new(),
            subtree_indices: Vec::new(),
            block_capacity: 0,
            feature_capacity: 0,
        }
    }
    
    fn ensure_capacity(&mut self, block_size: usize, num_features: usize) {
        if self.block_capacity < block_size || self.feature_capacity < num_features {
            self.features = (0..block_size)
                .map(|_| vec![f32::NAN; num_features])
                .collect();
            self.subtree_indices = vec![0; block_size];
            self.block_capacity = block_size;
            self.feature_capacity = num_features;
        } else {
            // Reset features to NaN (missing)
            for row in &mut self.features[..block_size] {
                row[..num_features].fill(f32::NAN);
            }
        }
    }
}
```

### Data Flow Diagram

```text
                            ┌─────────────────────────────┐
                            │       FeatureMatrix         │
                            │   (input: n_rows × n_feat)  │
                            └─────────────┬───────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    │             Predictor.predict()            │
                    │  ┌─────────────────────────────────────┐  │
                    │  │  1. Check density → block or row?   │  │
                    │  │  2. Check has_missing, has_cat      │  │
                    │  │  3. Dispatch to specialized impl    │  │
                    │  └─────────────────────────────────────┘  │
                    └─────────────────────┬─────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼                             ▼                             ▼
  ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
  │   predict_row<T,F>  │     │  predict_block<T,F> │     │   (GPU future)      │
  │  ScalarTreeVisitor  │     │  ArrayBlockVisitor  │     │                     │
  └──────────┬──────────┘     └──────────┬──────────┘     └─────────────────────┘
             │                           │
             │                           │
             ▼                           ▼
  ┌─────────────────────┐     ┌─────────────────────┐
  │    Per-row loop     │     │  Parallel blocks    │
  │  (rayon par_iter)   │     │ ┌─────────────────┐ │
  │                     │     │ │Thread 0: blk 0  │ │
  │  for row in rows:   │     │ │Thread 1: blk 1  │ │
  │    for tree:        │     │ │    ...          │ │
  │      visit_tree()   │     │ └────────┬────────┘ │
  │      accumulate     │     │          │          │
  └──────────┬──────────┘     │          ▼          │
             │                │ ┌─────────────────┐ │
             │                │ │ ArrayTreeLayout │ │
             │                │ │ process_block() │ │
             │                │ └────────┬────────┘ │
             │                │          │          │
             │                │          ▼          │
             │                │ ┌─────────────────┐ │
             │                │ │traverse_from()  │ │
             │                │ │ (remainder)     │ │
             │                │ └────────┬────────┘ │
             │                └──────────┼──────────┘
             │                           │
             └─────────────┬─────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Add base_score    │
                │   Return outputs    │
                └─────────────────────┘
```

### SIMD Leaf Accumulation

```rust
#[cfg(feature = "simd")]
mod simd_accumulate {
    use std::simd::{f32x8, SimdFloat};
    
    /// Accumulate 8 leaf values into 8 outputs (same group)
    #[inline]
    pub fn accumulate_8(outputs: &mut [f32; 8], leaves: &[f32; 8]) {
        let out_simd = f32x8::from_array(*outputs);
        let leaf_simd = f32x8::from_array(*leaves);
        let result = out_simd + leaf_simd;
        *outputs = result.to_array();
    }
    
    /// Accumulate leaf values for a block
    pub fn accumulate_block(
        outputs: &mut [f32],
        leaves: &[f32],
    ) {
        let chunks = outputs.chunks_exact_mut(8).zip(leaves.chunks_exact(8));
        for (out_chunk, leaf_chunk) in chunks {
            let out_arr: &mut [f32; 8] = out_chunk.try_into().unwrap();
            let leaf_arr: &[f32; 8] = leaf_chunk.try_into().unwrap();
            accumulate_8(out_arr, leaf_arr);
        }
        
        // Remainder
        let remainder_start = (outputs.len() / 8) * 8;
        for (o, &l) in outputs[remainder_start..].iter_mut()
            .zip(&leaves[remainder_start..])
        {
            *o += l;
        }
    }
}
```

## Specialization Strategy

We use const generics to eliminate runtime branches:

```rust
/// Dispatch based on runtime flags to const-generic impl
pub fn dispatch_visitor<F: Forest>(
    has_missing: bool,
    has_categorical: bool,
) -> Box<dyn TreeVisitor<F, Output = f32>>
where
    F::Leaf: LeafValue,
{
    match (has_missing, has_categorical) {
        (true, true) => Box::new(ScalarTreeVisitor::<true, true>),
        (true, false) => Box::new(ScalarTreeVisitor::<true, false>),
        (false, true) => Box::new(ScalarTreeVisitor::<false, true>),
        (false, false) => Box::new(ScalarTreeVisitor::<false, false>),
    }
}
```

This produces 4 specialized implementations at compile time, each with optimally-eliminated branches.

## Design Decisions

This section records architectural decisions with rationale.

### DD-1: Visitor Lifetime — Borrow vs Own **[DECIDED]**

**Decision**: Visitors **borrow** the forest via lifetime parameter.

**Rationale**:

- Lifetimes are Rust's strength; embrace them
- Borrow enables stack-allocated visitors (no `Box` overhead)
- Forest can be shared across multiple visitors
- Clear ownership: caller owns forest, visitor borrows it

**Implementation**:

```rust
pub struct Predictor<'f, F: Forest> {
    forest: &'f F,  // Borrowed, not Arc
    config: PredictorConfig,
}

impl<'f, F: Forest + Sync> Predictor<'f, F> {
    pub fn new(forest: &'f F, config: PredictorConfig) -> Self {
        Self { forest, config }
    }
    
    pub fn predict(&self, features: &FeatureMatrix) -> PredictionOutput {
        // ... borrow forest for duration of predict
    }
}
```

**Escape hatch**: If lifetime complexity becomes problematic, add `PredictorOwned<F>` that holds `Arc<F>`.

**Reversal criteria**: If we find lifetime annotations spreading through too many APIs or blocking async integration, we may add an owned variant.

### DD-2: ArrayTreeLayout Caching — Ephemeral vs Cached **[DECIDED]**

**Decision**: Use **ephemeral** creation like XGBoost, with opt-in caching.

**XGBoost approach analysis**:

- XGBoost creates `ArrayTreeLayout` per prediction call (in `GetArrayTreeLayout`)
- Rationale: Layout is small (~500 bytes for depth 6), creation is cheap
- Memory: Don't hold N×500 bytes for N trees when not predicting

**Our approach**:

```rust
pub struct Predictor<'f, F: Forest> {
    forest: &'f F,
    config: PredictorConfig,
    // NO cached layouts by default
}

impl<'f, F: Forest> Predictor<'f, F> {
    /// Create layout on-demand during prediction
    fn get_layout(&self, tree_idx: usize) -> ArrayTreeLayout<6> {
        let tree = self.forest.tree(tree_idx);
        ArrayTreeLayout::from_tree(&tree)  // ~500 bytes, fast construction
    }
}

/// Alternative: Cached predictor for repeated predictions
pub struct CachedPredictor<'f, F: Forest> {
    forest: &'f F,
    config: PredictorConfig,
    layouts: Vec<ArrayTreeLayout<6>>,  // Pre-built for all trees
}

impl<'f, F: Forest> CachedPredictor<'f, F> {
    pub fn new(forest: &'f F, config: PredictorConfig) -> Self {
        let layouts = (0..forest.num_trees())
            .map(|i| ArrayTreeLayout::from_tree(&forest.tree(i)))
            .collect();
        Self { forest, config, layouts }
    }
}
```

**Trade-off analysis**:

| Scenario | Ephemeral | Cached |
|----------|-----------|--------|
| Single prediction batch | ✓ Optimal | Wasteful setup |
| Repeated predictions | Redundant work | ✓ Amortized |
| Memory (1000 trees) | 0 | ~500KB |
| Construction time | ~1μs/tree | Once at setup |

**Recommendation**: Default to ephemeral. Add `CachedPredictor` as opt-in optimization. Document when to use each.

### DD-3: Output Format **[DECIDED]**

**Decision**: Use **flat `Vec<f32>`** with a zero-cost wrapper type.

**Rationale**:

- Single allocation (no per-row Vec)
- Cache-friendly contiguous memory
- Easy to convert to/from ndarray, nalgebra, etc.
- Wrapper provides ergonomic access

**Implementation**:

```rust
/// Prediction output: flat storage with shape metadata
pub struct PredictionOutput {
    /// Flat data: row-major layout
    /// predictions[row * num_groups + group]
    data: Vec<f32>,
    
    /// Number of rows (samples)
    num_rows: usize,
    
    /// Number of groups (output dimensions)
    num_groups: usize,
}

impl PredictionOutput {
    /// Get prediction for a single row
    #[inline]
    pub fn row(&self, row_idx: usize) -> &[f32] {
        let start = row_idx * self.num_groups;
        &self.data[start..start + self.num_groups]
    }
    
    /// Get mutable prediction for a single row
    #[inline]
    pub fn row_mut(&mut self, row_idx: usize) -> &mut [f32] {
        let start = row_idx * self.num_groups;
        &mut self.data[start..start + self.num_groups]
    }
    
    /// Iterate over rows
    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.data.chunks_exact(self.num_groups)
    }
    
    /// Get raw flat data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    
    /// Consume and return raw data
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }
    
    /// Convert to Vec<Vec<f32>> if needed (allocates)
    pub fn to_nested(&self) -> Vec<Vec<f32>> {
        self.rows().map(|r| r.to_vec()).collect()
    }
    
    /// Shape as (rows, groups)
    pub fn shape(&self) -> (usize, usize) {
        (self.num_rows, self.num_groups)
    }
}

// Optional: ndarray integration
#[cfg(feature = "ndarray")]
impl From<PredictionOutput> for ndarray::Array2<f32> {
    fn from(output: PredictionOutput) -> Self {
        ndarray::Array2::from_shape_vec(
            (output.num_rows, output.num_groups),
            output.data
        ).unwrap()
    }
}
```

**Alternatives considered**:

- `Vec<Vec<f32>>`: Per-row allocation overhead, but more ergonomic for simple cases
- `ndarray::Array2`: External dependency, but richer API
- `[[f32; G]; N]`: Compile-time size, too restrictive

## Open Questions

1. **Async prediction**: Should `predict()` be async-compatible for web services?
   - Current design is sync; async would need `Send + Sync` bounds
   - Deferred until we have concrete async use cases

2. **Streaming prediction**: Should we support predicting rows as they arrive?
   - Current API is batch-oriented
   - Could add `predict_iter()` that yields per-row

## Alternatives Considered

### Trait Object Visitors

Use `Box<dyn TreeVisitor>` for runtime polymorphism:

```rust
fn predict(&self, features: &FeatureMatrix) -> PredictionOutput {
    let visitor: Box<dyn TreeVisitor<_>> = match (has_missing, has_cat) {
        (true, true) => Box::new(ScalarTreeVisitor::<true, true>),
        // ...
    };
    // Use visitor...
}
```

**Rejected for hot path**: Virtual dispatch in inner loop. Kept `dispatch_visitor` for one-time selection, then monomorphized inner loop.

### Separate Visitor per Tree

```rust
trait TreeVisitor {
    fn visit(&self, tree: &SoATreeView, features: &[f32]) -> f32;
}
```

**Current approach**: Visitor is stateless (just const generic params). Could add per-tree state if needed (e.g., for tree-specific categorical lookups).

## References

- XGBoost `cpu_predictor.cc`: Block traversal, `ThreadTmp`
- XGBoost `array_tree_layout.h`: Array layout pattern
- RFC-0001: Forest Data Structures
- RFC-0002: Tree Data Structures
- [design/concepts/block_based_traversal.md](../concepts/block_based_traversal.md)
