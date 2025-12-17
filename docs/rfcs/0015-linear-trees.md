# RFC-0015: Linear Leaves for GBDT

- **Status**: Draft
- **Created**: 2025-12-17
- **Updated**: 2025-12-17
- **Depends on**: RFC-0002 (Forest and Tree Structures), RFC-0007 (Tree Growing), RFC-0014 (GBLinear)
- **Scope**: Linear regression models at tree leaves

## Summary

This RFC extends GBDT to support **linear leaves**—leaf nodes that predict using
linear regression models instead of constant values. Each leaf stores an intercept
plus coefficients for selected features, enabling piece-wise linear predictions.

Key design principles:

1. **Reuse GBLinear infrastructure**: Fit leaf models via coordinate descent,
   avoiding closed-form matrix inversion and its numerical stability issues
2. **Integrate with existing types**: Extend `Tree<L>` with optional coefficient
   storage rather than creating wrapper types
3. **Feature whitelist**: Allow users to control which features get linear
   coefficients, preventing unwanted extrapolation
4. **Clear naming**: Use "linear leaves" terminology to avoid confusion with GBLinear

## Motivation

Standard GBDT produces piece-wise constant predictions. This is suboptimal when:

- **Continuous relationships exist**: Price varies smoothly with square footage
- **Trees are shallow**: Limited splits → coarse partitioning
- **Extrapolation matters**: Constants handle distribution edges poorly

Linear leaves address this by fitting `prediction = const + Σ(coef_j × x_j)` at
each leaf, creating smooth transitions within regions.

LightGBM supports this via `linear_tree=True`. This RFC designs linear leaves for
booste-rs with better numerical stability (coordinate descent vs matrix inversion)
and user control (feature whitelist).

## Design

### Overview

Linear leaves are a **post-processing step** on standard trees:

1. Grow tree structure using existing `TreeGrower` (unchanged)
2. After tree is grown, fit linear models at each leaf using coordinate descent
3. Store coefficients in tree's coefficient storage for inference

### Data Structures

#### LeafCoefficients: Per-Tree Coefficient Storage

Follows the `CategoriesStorage` pattern—flat arrays with offset indexing.

```rust
/// Coefficient storage for linear leaves in a tree.
///
/// Stores per-leaf linear models as:
/// - Feature indices and coefficients (variable-length per leaf)
///
/// The leaf's base prediction (intercept) is stored in the normal
/// `leaf_values` array. Coefficients are additive adjustments.
///
/// When linear leaves are disabled, this storage is empty.
#[derive(Debug, Clone, Default)]
pub struct LeafCoefficients {
    /// Global feature indices (all leaves concatenated).
    feature_indices: Box<[u32]>,
    
    /// Coefficients (all leaves concatenated), parallel to feature_indices.
    coefficients: Box<[f32]>,
    
    /// Per-leaf segment: (start, len) into feature_indices/coefficients.
    /// Indexed by node_idx. Non-leaf nodes have (0, 0).
    segments: Box<[(u32, u16)]>,
}

impl LeafCoefficients {
    /// Empty coefficient storage (no linear leaves).
    pub fn empty() -> Self { Self::default() }
    
    /// Whether storage is empty (no linear coefficients anywhere).
    pub fn is_empty(&self) -> bool { self.segments.is_empty() }
    
    /// Get (feature_indices, coefficients) slice for a leaf node.
    /// Returns empty slices if node has no linear terms.
    pub fn leaf_terms(&self, node_idx: u32) -> (&[u32], &[f32]);
}
```

#### Extended Tree Storage

Add optional coefficient storage to `Tree<L>`:

```rust
/// Tree with linear leaf coefficients.
pub struct Tree<L: LeafValue> {
    // ... existing fields unchanged ...
    
    /// Split feature indices per node.
    split_indices: Box<[u32]>,
    /// Split thresholds per node.
    split_thresholds: Box<[f32]>,
    /// Left child indices per node.
    left_children: Box<[u32]>,
    /// Right child indices per node.
    right_children: Box<[u32]>,
    /// Default direction per node.
    default_left: Box<[bool]>,
    /// Whether each node is a leaf.
    is_leaf: Box<[bool]>,
    /// Leaf values (intercepts for linear leaves).
    leaf_values: Box<[L]>,
    /// Split types per node.
    split_types: Box<[SplitType]>,
    /// Categorical split storage.
    categories: CategoriesStorage,
    
    /// NEW: Linear coefficients for leaves.
    /// When non-empty, leaves use: leaf_value + Σ(coef × feature).
    /// When empty, leaves use constant prediction (standard GBDT).
    /// Not Option—empty state is LeafCoefficients::empty().
    leaf_coefficients: LeafCoefficients,
}
```

Benefits:

- **No new types needed**: Same `Tree<L>`, `Forest<L>`, `Predictor<T>`
- **Backward compatible**: Existing code works unchanged
- **Minimal overhead when disabled**: Empty storage has negligible cost

### Training: Coordinate Descent Fitting

Instead of LightGBM's closed-form matrix solution, we reuse GBLinear's coordinate
descent infrastructure. This provides:

1. **Numerical stability**: No matrix inversion, handles collinearity naturally
2. **Code reuse**: GBLinear's `Updater` already implements weighted CD
3. **Elastic net**: Support for L1 + L2 regularization
4. **Simplicity**: No pseudo-inverse or condition number checks needed

#### LeafLinearTrainer

Follows the library's `Trainer`/`train` naming convention:

```rust
/// Trains linear models for tree leaves using coordinate descent.
///
/// Reuses GBLinear's coordinate descent infrastructure for numerical
/// stability and code reuse.
pub struct LeafLinearTrainer {
    /// L2 regularization on coefficients.
    lambda: f32,
    /// L1 regularization (elastic net).
    alpha: f32,
    /// Number of coordinate descent iterations per leaf.
    n_iters: u32,
    /// Features allowed in linear models (None = path features only).
    feature_whitelist: Option<Box<[u32]>>,
    /// Number of threads (0 = auto, 1 = sequential, >1 = parallel).
    n_threads: usize,
}

impl LeafLinearTrainer {
    /// Train linear coefficients for all leaves in a tree.
    ///
    /// # Arguments
    /// - `tree`: Grown tree (will be modified to add coefficients)
    /// - `dataset`: Dataset with raw feature values
    /// - `partitioner`: Contains row-to-leaf mapping from tree growth
    /// - `gradients`: Gradient/Hessian values
    /// - `output`: Output index
    /// - `learning_rate`: Applied to coefficients
    pub fn train(
        &self,
        tree: &mut Tree<ScalarLeaf>,
        dataset: &BinnedDataset,
        partitioner: &RowPartitioner,
        gradients: &Gradients,
        output: usize,
        learning_rate: f32,
    );
}
```

#### Training Algorithm

```text
Algorithm: TrainLeafLinearModels
──────────────────────────────
Inputs:
  - tree: grown Tree with constant leaves
  - dataset: BinnedDataset + raw feature values
  - partitioner: row-to-leaf assignments
  - gradients: per-row (g, h) pairs
  - config: (λ, α, n_iters, whitelist)

For each leaf ℓ in tree:
  1. Get samples S_ℓ = rows assigned to this leaf
  2. If |S_ℓ| < min_samples: skip (use constant)
  
  3. Determine features F_ℓ:
     a. Collect numerical features on path from root to ℓ
     b. If whitelist provided: F_ℓ = F_ℓ ∩ whitelist
     c. If any sample has NaN in F_ℓ: skip (use constant)
  
  4. Fit via coordinate descent (reusing GBLinear logic):
     - Initialize coefficients c = [0, 0, ..., 0]
     - For iter = 1 to n_iters:
         For each feature j in F_ℓ:
           sum_grad = Σ_{i∈S_ℓ} g_i × x_{i,j} + λ × c_j
           sum_hess = Σ_{i∈S_ℓ} h_i × x_{i,j}² + λ
           c_j -= η × soft_threshold(sum_grad/sum_hess, α/sum_hess)
  
  5. Scale coefficients by learning_rate
  6. Store (F_ℓ, c) in tree.leaf_coefficients

tree.leaf_coefficients = Some(built_coefficients)
```

### Feature Whitelist

Users can specify which features are allowed in linear models:

```rust
/// Parameters for linear leaves.
#[derive(Clone, Debug, Default)]
pub struct LinearLeavesParams {
    /// Enable linear leaves.
    pub enabled: bool,
    
    /// L2 regularization on coefficients.
    pub lambda: f32,
    
    /// L1 regularization (elastic net).
    pub alpha: f32,
    
    /// Coordinate descent iterations per leaf.
    pub n_iters: u32,
    
    /// Feature whitelist: only these features can have linear coefficients.
    /// Features not in whitelist contribute only to the leaf's base value.
    /// None = use all numerical path features (default LightGBM behavior).
    pub feature_whitelist: Option<Vec<u32>>,
    
    /// Minimum samples per leaf for linear fitting.
    pub min_samples: u32,
}
```

**Use case**: Prevent extrapolation on features known to be problematic:

```rust
// Only allow these features in linear models
let params = LinearLeavesParams {
    enabled: true,
    feature_whitelist: Some(vec![0, 1, 5]),  // sqft, bedrooms, year_built
    ..Default::default()
};
// Other path features (e.g., zip_code) contribute only to intercept
```

### Inference

Extended prediction handles linear coefficients when present:

```rust
impl<L: LeafValue> Tree<L> {
    /// Predict for a single row, using linear coefficients if available.
    pub fn predict_row(&self, features: &[f32]) -> L {
        let node = self.traverse_to_leaf(features);
        let base_value = self.leaf_values[node].clone();
        
        // Get linear terms (empty slices if no linear model)
        let (feat_indices, coef_values) = self.leaf_coefficients.leaf_terms(node);
        if feat_indices.is_empty() {
            return base_value;
        }
        
        // Compute linear term
        let mut linear_sum = 0.0f32;
        for (&f_idx, &coef) in feat_indices.iter().zip(coef_values) {
            let x = features[f_idx as usize];
            if x.is_nan() {
                // Fallback to base value on NaN
                return base_value;
            }
            linear_sum += coef * x;
        }
        
        // Return base + linear (for ScalarLeaf, this is addition)
        let mut result = base_value;
        result.accumulate(&ScalarLeaf(linear_sum));
        result
    }
}
```

#### Compatibility with Traversal Strategies

Linear leaves **require `StandardTraversal`**. They are incompatible with
optimized traversal strategies (`UnrolledTraversal`) that pre-compute tree
layouts and assume constant leaf values:

- **UnrolledTraversal**: Caches top 6 levels in flat arrays, expects simple
  leaf value lookup after traversal. Linear leaves need per-leaf feature
  lookups and dot product computation, breaking the pre-computed layout.

**Validation**: Change `TreeTraversal::build_tree_state` to return `Result`:

```rust
trait TreeTraversal<L: LeafValue> {
    fn build_tree_state(tree: &Tree<L>) -> Result<Self::TreeState, String>;
    // ... other methods unchanged
}

impl<D: UnrollDepth> TreeTraversal<ScalarLeaf> for UnrolledTraversal<D> {
    fn build_tree_state(tree: &Tree<ScalarLeaf>) -> Result<Self::TreeState, String> {
        if !tree.leaf_coefficients().is_empty() {
            return Err(
                "Linear leaves are not supported with UnrolledTraversal. \
                 Use Predictor::<StandardTraversal>::new() for forests with linear coefficients."
                    .into(),
            );
        }
        Ok(UnrolledTreeLayout::from_tree(tree))
    }
}
```

The `Predictor` constructor propagates this error, providing a clear message
at predictor creation time (not during prediction).

**Training-time prediction** (`predict_binned_row`) is **unaffected**. It's used
for gradient computation during tree growth, before linear coefficients are
fitted. Linear training is a post-processing step after tree structure is built.

### Integration with GBDTTrainer

Linear leaf fitting integrates into the training loop as a post-processing step:

```rust
impl<O: Objective, M: Metric> GBDTTrainer<O, M> {
    fn train_impl(&self, ...) -> Option<Forest<ScalarLeaf>> {
        // ... existing setup ...
        
        let linear_trainer = self.params.linear_leaves.enabled.then(|| {
            LeafLinearTrainer::new(&self.params.linear_leaves)
        });
        
        for round in 0..self.params.n_trees {
            // ... compute gradients ...
            
            for output in 0..n_outputs {
                // Grow tree (unchanged)
                let mut tree = grower.grow(dataset, &gradients, output, sampled);
                
                // Post-process: train linear coefficients
                // Skip first tree (iteration 0) - matches LightGBM
                if let Some(trainer) = &linear_trainer {
                    if round > 0 {
                        trainer.train(
                            &mut tree,
                            dataset,
                            grower.partitioner(),
                            &gradients,
                            output,
                            self.params.learning_rate,
                        );
                    }
                }
                
                forest.push_tree(tree, output as u32);
            }
        }
        
        Some(forest)
    }
}
```

### Reusing GBLinear Components

The existing `compute_weight_update` function in `training/gblinear/updater.rs`
already implements the core coordinate descent logic. Rather than creating a new
function, we **generalize the existing one** to support both use cases:

```rust
// In training/gblinear/updater.rs (existing function, generalized)

/// Compute weight update for a single feature using elastic net.
///
/// Used by:
/// - GBLinear training (full dataset, column iteration)
/// - Linear leaf training (row subset per leaf)
///
/// The `feature_values` iterator yields (row_index, feature_value) pairs,
/// allowing both column-major iteration and row-subset iteration.
pub fn compute_weight_update<I>(
    feature_values: I,
    grad_hess: &[GradsTuple],  // Use existing GradsTuple type
    current_weight: f32,
    config: &UpdateConfig,     // Reuse existing UpdateConfig
) -> f32
where
    I: Iterator<Item = (usize, f32)>,
{
    let mut sum_grad = 0.0f32;
    let mut sum_hess = 0.0f32;
    
    for (row, value) in feature_values {
        sum_grad += grad_hess[row].grad * value;
        sum_hess += grad_hess[row].hess * value * value;
    }
    
    // L2 regularization
    let grad_l2 = sum_grad + config.lambda * current_weight;
    let hess_l2 = sum_hess + config.lambda;
    
    if hess_l2.abs() < 1e-10 {
        return 0.0;
    }
    
    // L1 soft-thresholding (elastic net)
    let raw_update = -grad_l2 / hess_l2;
    let threshold = config.alpha / hess_l2;
    let thresholded = soft_threshold(raw_update, threshold);
    
    thresholded * config.learning_rate
}
```

**Key insight**: The function already accepts an iterator over (row, value) pairs.
For GBLinear, this comes from `ColMatrix::column()`. For leaf fitting, we provide
an iterator over only the rows assigned to that leaf:

```rust
// In LeafLinearTrainer::train()
for leaf_node in tree.leaves() {
    let rows = partitioner.leaf_rows(leaf_node);
    
    for &feature_idx in &path_features {
        // Iterator over (row, feature_value) for rows in this leaf
        let values = rows.iter().map(|&row| {
            (row as usize, dataset.get(row, feature_idx))
        });
        
        let delta = compute_weight_update(
            values,
            gradients.output_pairs(output),
            current_coef,
            &config,
        );
        coefficients[feature_idx] += delta;
    }
}
```

No new function needed—same `compute_weight_update`, different iterator source.

## Implementation Plan

Phased approach to minimize risk and enable incremental validation:

### Phase 1: Inference Infrastructure

1. Add `LeafCoefficients` storage structure to `src/repr/gbdt/leaf_coefficients.rs`
2. Extend `Tree<L>` with `leaf_coefficients: LeafCoefficients` field
3. Update `Predictor` to handle linear terms in `predict_row`
4. Add unit tests for linear prediction logic

**Validation**: Manual construction of trees with known coefficients, verify predictions

### Phase 2: LightGBM Compatibility

1. Extend LightGBM model loader to parse `linear_tree` coefficients from JSON
2. Load reference models trained with `linear_tree=True` in Python
3. Test inference parity against LightGBM predictions

**Validation**: Integration tests with multiple datasets, verify exact match with LightGBM

### Phase 3: Training Implementation

1. Implement `LeafLinearTrainer` in `src/training/gbdt/leaf_linear.rs`
2. Generalize `compute_weight_update` in GBLinear to accept iterator input
3. Add coordinate descent fitting for single leaf (unit tests)
4. Implement path feature extraction and whitelist filtering

**Validation**: Unit tests with synthetic gradients, verify convergence

### Phase 4: Integration

1. Add `LinearLeavesParams` to `GBDTParams`
2. Integrate `LeafLinearTrainer` into `GBDTTrainer` training loop
3. Add end-to-end training tests (synthetic datasets)
4. Benchmark against LightGBM (quality and performance)

**Validation**: Training tests with known-good datasets, quality metrics vs LightGBM

### Phase 5: Optimization & Polish

1. Add parallelism (shotgun CD and/or parallel leaves)
2. Performance profiling and optimization
3. Documentation and examples
4. Optional: Own serialization format

**Validation**: Performance benchmarks, documentation review

## Design Decisions

### DD-1: Integrate vs Wrapper Types

**Context**: How to add linear coefficient storage?

**Options considered**:

1. **Wrapper types** (`LinearTree`, `LinearForest`, `LinearPredictor`):
   - Pro: Complete separation
   - Con: Code duplication, naming confusion, parallel type hierarchies

2. **Extend existing types** (add `Option<LeafCoefficients>` to `Tree<L>`):
   - Pro: Single set of types, existing predictors work
   - Pro: Zero overhead when disabled
   - Con: Slightly more complex Tree

**Decision**: Option 2 (Extend existing types), but **non-optional**.

Adding a field with empty state (not `Option`) is minimal complexity. Empty
`LeafCoefficients` has three null pointers overhead. Avoids `Option` unwrapping
throughout inference code.

**Consequences**:
- No `LinearTree`, `LinearForest`, `LinearPredictor` types
- `Tree<L>` gains one field (always present, may be empty)
- Inference code calls `leaf_terms()` which returns empty slices when disabled

### DD-2: Coordinate Descent vs Closed-Form Solution

**Context**: How to fit linear models at leaves?

**Options considered**:

1. **Closed-form solution** (LightGBM approach):
   - $\mathbf{c} = -(\mathbf{X}^\top\mathbf{H}\mathbf{X} + \lambda\mathbf{R})^{-1}\mathbf{X}^\top\mathbf{g}$
   - Pro: Single solve per leaf
   - Con: Requires matrix inversion, ill-conditioned with collinear features
   - Con: Needs pseudo-inverse fallback, condition number checks

2. **Coordinate descent** (reuse GBLinear):
   - Pro: No matrix inversion, handles collinearity naturally
   - Pro: Reuses existing, tested code
   - Pro: Supports elastic net (L1+L2)
   - Con: Multiple iterations (but leaves are small)

**Decision**: Option 2 (Coordinate descent).

Numerical stability is more important than theoretical optimality. Leaves are
typically small (dozens to hundreds of samples), so a few CD iterations are fast.
Reusing GBLinear's updater reduces code and bugs.

**Consequences**:
- `LeafLinearTrainer` calls existing `compute_weight_update` with row-subset iterator
- No new function needed—same code, different data source
- No matrix operations, no condition number checks
- Natural L1 support via soft-thresholding

### DD-3: Feature Whitelist

**Context**: Should users control which features get linear coefficients?

**Options considered**:

1. **Path features only** (LightGBM default):
   - Pro: Simple, automatic
   - Con: No control over extrapolation

2. **User whitelist**:
   - Pro: Prevents dangerous extrapolation on known-problematic features
   - Pro: Domain knowledge can improve model
   - Con: Additional parameter

**Decision**: Support both (whitelist is optional).

Default behavior uses path features (LightGBM compatible). When whitelist is
provided, only whitelisted features among path features are used.

**Consequences**:
- `LinearLeavesParams::feature_whitelist: Option<Vec<u32>>`
- Features not in whitelist contribute only to intercept
- Intersection: `final_features = path_features ∩ whitelist`

### DD-4: Naming

**Context**: "Linear" conflicts with GBLinear terminology.

**Options considered**:

1. **LinearTree/LinearForest/LinearPredictor**: Confusing with GBLinear
2. **PiecewiseLinearTree**: Verbose, still has "Linear"
3. **Linear leaves** (leaf-focused naming): Clear distinction from GBLinear (model type)

**Decision**: Use "linear leaves" terminology.

- Parameter struct: `LinearLeavesParams`
- Trainer: `LeafLinearTrainer`
- Storage: `LeafCoefficients`
- Feature flag: `linear_leaves.enabled`

This emphasizes it's a property of leaves within GBDT, not a separate model type.

**Consequences**:
- Clear distinction: GBLinear = booster type, linear leaves = leaf enhancement
- Consistent naming: `LeafXxx` prefix for leaf-specific components

### DD-5: First Tree Handling

**Context**: Should the first tree use linear leaves?

**Decision**: Skip first tree (match LightGBM).

The first tree approximates the mean of the target. Gradients are nearly uniform,
providing poor signal for linear coefficients. Later trees have structured
residuals that benefit from linear fitting.

**Consequences**:

- Check `round > 0` before calling `LeafLinearTrainer`
- First tree uses constant leaves, subsequent trees use linear

### DD-6: Reuse GBLinear Infrastructure

**Context**: Should we reuse `GBLinearTrainer` or create separate training logic?

**Options considered**:

1. **Reuse `GBLinearTrainer` directly**:
   - Pro: Maximum code reuse
   - Con: `GBLinearTrainer` is designed for multi-round training with early stopping,
     evaluation, logging—all unnecessary for one-shot leaf fitting
   - Con: `GBLinearTrainer` uses `ColMatrix` for column iteration; leaves need
     row-subset iteration (only samples in that leaf)
   - Con: Overhead from features we don't need (eval sets, prediction tracking)

2. **Create new `LeafLinearTrainer`, reuse core update logic**:
   - Pro: Right abstraction—one-shot fitting per leaf without multi-round machinery
   - Pro: Direct row-subset iteration via iterator pattern
   - Pro: Reuses tested coordinate descent logic (`compute_weight_update`)
   - Con: New trainer type (but minimal—just calls existing update logic)

**Decision**: Option 2 (New trainer, reuse core logic).

`LeafLinearTrainer` is a focused component for one-shot leaf fitting. It reuses
`compute_weight_update` from GBLinear's `Updater`, which already implements the
core coordinate descent algorithm. We generalize that function to accept an
iterator (instead of only `ColMatrix`), enabling both use cases.

**Consequences**:
- New `LeafLinearTrainer` type in `training/gbdt/leaf_linear.rs`
- `compute_weight_update` generalized to accept iterator input
- No code duplication—both GBLinear and leaf training use same update logic
- Clear separation of concerns (multi-round vs one-shot)

### DD-7: LeafValue vs Separate Storage

**Context**: Should coefficient references/offsets be stored in `LeafValue` or
separately in `Tree`?

**Options considered**:

1. **Extend LeafValue to include coefficient info**:
   ```rust
   struct ScalarLeaf {
       base: f32,
       coef_offset: Option<u32>,  // Into global coefficient storage
       coef_len: u16,
   }
   ```
   - Pro: Coefficient info travels with leaf value
   - Con: Increases `ScalarLeaf` size from 4 bytes to 12+ bytes (even for constant leaves)
   - Con: Breaks `LeafValue` abstraction—trait is about value operations
     (accumulate, scale), not prediction logic
   - Con: Complicates training (leaf values created before coefficients fitted)

2. **Separate `leaf_coefficients` field in Tree** (current design):
   ```rust
   struct Tree<L> {
       leaf_values: Box<[L]>,           // 4 bytes per leaf (ScalarLeaf)
       leaf_coefficients: LeafCoefficients,  // Shared storage
   }
   ```
   - Pro: Clean separation—leaf value is just the intercept
   - Pro: No overhead for constant leaves (4 bytes per leaf)
   - Pro: Coefficients fitted as post-processing, independent of leaf value creation
   - Pro: `LeafValue` trait remains simple (value operations only)
   - Con: Two lookups during prediction (leaf value + coefficient storage)

**Decision**: Option 2 (Separate storage).

The "two lookups" cost is negligible—both are indexed by `node_idx` and are
likely in cache together. The separation keeps `LeafValue` focused on its core
responsibility (value representation and arithmetic) and avoids bloating the
common case (constant leaves).

**Consequences**:
- `ScalarLeaf` remains 4 bytes (just `f32`)
- `LeafValue` trait unchanged—no prediction logic added
- Coefficient storage indexed by node_idx, parallel to leaf_values
- Prediction: `tree.leaf_value(idx) + linear_term(tree.leaf_coefficients, idx, features)`

### DD-8: StandardTraversal Only (Initially)

**Context**: Should linear leaves support `UnrolledTraversal` optimization?

**Options considered**:

1. **Support both StandardTraversal and UnrolledTraversal**:
   - Would require storing coefficients in `UnrolledTreeLayout`
   - At each exit point: `(base_value, coef_offset, coef_len)`
   - Pro: Best theoretical performance (unrolled + linear)
   - Con: Significant complexity (pre-compute coefficient indices/values)
   - Con: Unclear if faster (linear term cost might dominate traversal cost)

2. **StandardTraversal only, validate against UnrolledTraversal**:
   - Linear leaves use standard node-by-node traversal
   - `UnrolledTraversal::build_tree_state()` returns error if coefficients present
   - Pro: Simple, clear separation of concerns
   - Pro: Can optimize later if profiling shows need
   - Con: No unrolled optimization for linear leaves

**Decision**: Option 2 (StandardTraversal only).

Ship initial implementation with `StandardTraversal`. If profiling shows
traversal is a bottleneck (vs linear term computation), we can add unrolled
support later. Premature optimization would add complexity without validated benefit.

**Consequences**:
- Linear leaves require `StandardTraversal`
- `TreeTraversal::build_tree_state` returns `Result<TreeState, String>`
- Error message guides users to StandardTraversal when linear coefficients detected
- Future optimization path: extend UnrolledTreeLayout if needed

## Integration

| Component | Integration Point | Notes |
|-----------|------------------|-------|
| RFC-0002 (Tree Structures) | `Tree<L>` gains `leaf_coefficients` field | Non-optional, empty when disabled |
| RFC-0007 (Tree Growing) | Post-processing after `grow()` | Grower unchanged |
| RFC-0014 (GBLinear) | Reuse `compute_weight_update` | Generalize iterator input |
| Predictor | Call `leaf_terms()` in traversal | Returns empty for constant leaves |

### Module Structure

```text
src/
├── repr/gbdt/
│   ├── tree.rs               # Add LeafCoefficients to Tree<L>
│   ├── leaf_coefficients.rs  # LeafCoefficients storage
│   └── mod.rs
├── training/
│   ├── gbdt/
│   │   ├── leaf_linear.rs    # LeafLinearTrainer
│   │   ├── trainer.rs        # Integrate linear training
│   │   └── mod.rs
│   └── gblinear/
│       └── updater.rs        # Generalize compute_weight_update iterator
├── inference/gbdt/
│   └── traversal.rs          # Handle linear coefficients in predict
└── compat/lightgbm/
    └── loader.rs             # Parse linear tree coefficients
```

## Open Questions

1. ~~**Serialization format**~~: **Resolved**—Focus on LightGBM format first.
   We already support loading LightGBM models. Extend that loader to parse
   linear tree coefficients, then verify inference parity. Own format later.

2. ~~**Multi-output trees**~~: **Resolved**—Out of scope.
   We support one-output-per-tree mode only (XGBoost compatibility). Multi-output
   leaves (`VectorLeaf`) would require per-output coefficients; defer to future.

3. ~~**Parallelism**~~: **Resolved**—Add `n_threads` parameter.
   - `n_threads = 0`: Auto (rayon global pool)
   - `n_threads = 1`: Sequential (no rayon calls)
   - `n_threads > 1`: Parallel with dedicated pool
   
   Parallel strategies:
   - **Shotgun CD**: Update all features in parallel within each leaf
   - **Parallel leaves**: Train multiple leaves concurrently
   
   Must respect user's `n_threads = 1` setting—no rayon at all.

## Future Work

- [ ] LightGBM model import/export with linear leaves
- [ ] Multi-output linear leaves (`VectorLeaf`)
- [ ] Per-leaf regularization (adaptive λ based on leaf size)
- [ ] Feature importance for linear coefficients

## References

- [Linear Trees Research Doc](../research/gbdt/training/linear-trees.md)
- [LightGBM Linear Trees Paper](https://arxiv.org/abs/1802.05640): Shi et al., 2018
- RFC-0014 (GBLinear): Coordinate descent implementation

## Changelog

- 2025-12-17: Major revision based on feedback
  - Changed from wrapper types to extending existing `Tree<L>`
  - Adopted coordinate descent (reuse GBLinear) instead of closed-form
  - Added feature whitelist for extrapolation control
  - Renamed from "Linear Trees" to "Linear Leaves" to avoid GBLinear confusion
