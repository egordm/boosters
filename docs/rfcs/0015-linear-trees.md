# RFC-0015: Linear Leaves for GBDT

- **Status**: Implemented
- **Created**: 2025-12-17
- **Implemented**: 2025-12-18
- **Depends on**: RFC-0002, RFC-0007, RFC-0014, RFC-0016

## Summary

Linear leaves fit `intercept + Σ(coef × feature)` at each leaf instead of constants.
Trained via coordinate descent (reuse GBLinear), requires raw feature access.

## Motivation

Standard GBDT produces piece-wise constant predictions. Linear leaves add smooth
transitions within tree partitions—useful for continuous relationships, shallow
trees, and edge extrapolation.

LightGBM supports this (`linear_tree=True`). We improve with coordinate descent
(numerical stability) and feature whitelist (control extrapolation).

**When NOT to use linear trees:**

- Very deep trees (max_depth > 10): Leaves have few samples, linear fits are unstable
- Pure categorical problems: No linear structure to exploit
- When individual tree interpretability matters: Linear coefficients add complexity
- High-noise domains: Linear fits may overfit noise patterns

## Design

### Integration Point

Linear leaves are configured via `GBDTConfig`:

```rust
let config = GBDTConfig {
    n_trees: 100,
    max_depth: 6,
    learning_rate: 0.1,
    // Enable linear leaves with custom settings
    linear_leaves: Some(LinearLeafConfig {
        lambda: 0.01,
        min_samples: 100,
        ..Default::default()
    }),
    ..Default::default()
};

let model = GBDTTrainer::new(config).train(&dataset)?;
```

### Training Flow

```rust
for round in 0..n_trees {
    let mut tree = grower.grow(dataset, &gradients, output, sampled);
    
    if linear_leaves_enabled && round > 0 {
        // partitioner has row→leaf mapping from growth
        let rows_per_leaf = grower.partitioner();
        linear_trainer.train(&mut tree, &col_matrix, rows_per_leaf, &gradients);
    }
    
    let tree = tree.freeze();  // LeafCoefficients built here
    tree.predict_batch_accumulate(&accessor, &mut predictions);
    forest.push_tree(tree, output);
}
```

### Raw Feature Access with Pre-gathering

Linear leaves need actual f32 values for `coef × feature`. From previous
optimizations, contiguous row access is much faster than scattered gather.

**Buffer allocation**: Once at training start, sized for largest leaf:

```rust
/// Column-major buffer for leaf feature data
/// Allocated once, reused for each leaf
pub struct LeafFeatureBuffer {
    /// Column-major: features[feat_idx * max_rows + row_idx]
    data: Vec<f32>,
    max_rows: usize,
    n_rows: usize,      // current leaf row count
    n_features: usize,  // current feature count  
}

impl LeafFeatureBuffer {
    /// Allocate once at training start
    /// max_rows = max samples per leaf (e.g., n_samples / n_leaves * 2)
    /// max_features = max features used in any path (e.g., max_depth)
    pub fn new(max_rows: usize, max_features: usize) -> Self {
        Self {
            data: vec![0.0; max_rows * max_features],
            max_rows,
            n_rows: 0,
            n_features: 0,
        }
    }
}
```

**Gather order**: Source is column-major (`ColMatrix`), so iterate features-first
for cache-friendly reads. Output is also column-major for contiguous slices:

```rust
impl LeafFeatureBuffer {
    /// Gather from column-major source into column-major buffer
    /// Features-first iteration: cache-friendly read from ColMatrix
    pub fn gather<S: AsRef<[f32]>>(
        &mut self,
        rows: &[u32],
        col_matrix: &ColMatrix<f32, S>,
        features: &[u32],
    ) {
        self.n_rows = rows.len();
        self.n_features = features.len();
        
        // Iterate features first (source is column-major)
        for (feat_idx, &feat) in features.iter().enumerate() {
            let col_offset = feat_idx * self.max_rows;
            for (row_idx, &row) in rows.iter().enumerate() {
                self.data[col_offset + row_idx] = col_matrix.get(row as usize, feat as usize);
            }
        }
    }
    
    /// Get contiguous slice for a feature (column-major layout)
    pub fn feature_slice(&self, feat_idx: usize) -> &[f32] {
        let start = feat_idx * self.max_rows;
        &self.data[start..start + self.n_rows]
    }
}
```

**No gradient gathering**: Gradients are indexed by original row indices. We keep
a `rows: &[u32]` slice to map buffer indices back to gradient indices.

### Linear Model Fitting

Use coordinate descent on the normal equations with fixed GBDT gradients.
This matches LightGBM's approach while avoiding matrix inversion.

#### Symmetric Matrix Indexing

Store only upper triangle of the symmetric matrix X^THX:

```rust
/// Index into upper triangle storage for symmetric matrix
/// For k×k matrix stored as [A[0,0], A[0,1], ..., A[0,k-1], A[1,1], A[1,2], ...]
#[inline]
fn tri_index(i: usize, j: usize, size: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };  // Ensure i <= j
    i * size - i * (i + 1) / 2 + j
}
```

#### Solver Abstraction

Create a reusable solver for weighted least squares problems:

```rust
/// Solver for weighted least squares: min Σ h_i (y_i - x_i^T c)² + λ||c||²
pub struct WeightedLeastSquaresSolver {
    /// Upper triangle of X^T H X matrix
    xthx: Vec<f64>,
    /// X^T g vector  
    xtg: Vec<f64>,
    /// Solution coefficients
    coefficients: Vec<f64>,
    /// Maximum problem size
    max_features: usize,
}

impl WeightedLeastSquaresSolver {
    /// Allocate once with maximum expected feature count
    pub fn new(max_features: usize) -> Self {
        let size = max_features + 1;  // +1 for intercept
        Self {
            xthx: vec![0.0; size * (size + 1) / 2],
            xtg: vec![0.0; size],
            coefficients: vec![0.0; size],
            max_features,
        }
    }
    
    /// Reset for new problem
    pub fn reset(&mut self, n_features: usize) {
        let size = n_features + 1;
        self.xthx[..size * (size + 1) / 2].fill(0.0);
        self.xtg[..size].fill(0.0);
        self.coefficients[..size].fill(0.0);
    }
    
    /// Accumulate one sample into the normal equations
    pub fn accumulate(&mut self, features: &[f32], grad: f32, hess: f32);
    
    /// Add L2 regularization (not on intercept)
    pub fn add_regularization(&mut self, lambda: f64, n_features: usize);
    
    /// Solve via coordinate descent, returns (intercept, coefficients)
    pub fn solve_cd(&mut self, n_features: usize, config: &CDConfig) -> (&f64, &[f64]);
}
```

This solver:
- Allocates once at training start
- Reuses buffers for each leaf
- Encapsulates CD logic for potential replacement with direct solve

#### Configuration

```rust
/// Configuration for linear leaf training
pub struct LinearLeafConfig {
    /// L2 regularization on coefficients (default: 0.01)
    /// Small default prevents overfitting in small leaves.
    pub lambda: f32,
    /// L1 regularization for sparse coefficients (default: 0.0)
    pub alpha: f32,
    /// Maximum CD iterations per leaf (default: 10)
    pub max_iterations: u32,
    /// Convergence tolerance (default: 1e-6)
    pub tolerance: f64,
    /// Minimum samples in leaf to fit linear model (default: 50)
    pub min_samples: usize,
    /// Coefficient threshold—prune if |coef| < threshold (default: 1e-6)
    pub coefficient_threshold: f32,
    /// Feature whitelist—None means use all path features
    /// Can be built from names via `DataMatrix::feature_indices(&["age", "income"])`
    pub feature_whitelist: Option<Box<[u32]>>,
}

impl Default for LinearLeafConfig {
    fn default() -> Self {
        Self {
            lambda: 0.01,  // Light regularization by default
            alpha: 0.0,
            max_iterations: 10,
            tolerance: 1e-6,
            min_samples: 50,
            coefficient_threshold: 1e-6,
            feature_whitelist: None,
        }
    }
}
```

#### Fitting Flow

```rust
/// Fit linear coefficients for one leaf
fn fit_leaf(
    solver: &mut WeightedLeastSquaresSolver,
    feature_buffer: &LeafFeatureBuffer,
    rows: &[u32],
    gradients: &Gradients,
    output: usize,
    config: &LinearLeafConfig,
) -> Option<LeafCoefficients> {
    let n_features = feature_buffer.n_features;
    let n_rows = rows.len();
    solver.reset(n_features);
    
    // Accumulate normal equations (column-wise for cache efficiency)
    for feat_idx in 0..n_features {
        let feat_slice = feature_buffer.feature_slice(feat_idx);
        for (buf_idx, &orig_row) in rows.iter().enumerate() {
            let gh = gradients.get_pair(orig_row as usize, output);
            let x = feat_slice[buf_idx];
            solver.accumulate_column(feat_idx, buf_idx, x, gh.grad, gh.hess);
        }
    }
    
    solver.add_regularization(config.lambda as f64, n_features);
    
    // Solve and check convergence
    let converged = solver.solve_cd(n_features, config.max_iterations, config.tolerance);
    if !converged {
        // Fall back to constant leaf if CD doesn't converge
        return None;
    }
    
    // Prune near-zero coefficients and convert to f32
    let (intercept, coefs) = solver.coefficients(n_features);
    let pruned: Vec<(usize, f32)> = coefs.iter().enumerate()
        .filter(|(_, &c)| c.abs() > config.coefficient_threshold as f64)
        .map(|(i, &c)| (i, c as f32))
        .collect();
    
    Some(LeafCoefficients::new(*intercept as f32, pruned))
}
```

**Why CD over direct solve?**

- No BLAS/LAPACK dependency
- Gracefully handles collinearity (just converges slower)
- Natural L1 support via soft-thresholding
- Early termination when converged
- Reuses GBLinear infrastructure

**Alternative backends**: The solver interface allows future backends:
- Direct solve via `faer` or `nalgebra` for benchmarking
- GPU-accelerated batched solve for many leaves

### LeafLinearTrainer

Orchestrates the full training flow with pre-allocated resources:

```rust
pub struct LeafLinearTrainer {
    config: LinearLeafConfig,
    /// Pre-allocated feature buffer (reused per leaf)
    feature_buffer: LeafFeatureBuffer,
    /// Pre-allocated solver (reused per leaf)  
    solver: WeightedLeastSquaresSolver,
}

impl LeafLinearTrainer {
    pub fn new(config: LinearLeafConfig, max_leaf_samples: usize) -> Self {
        Self {
            feature_buffer: LeafFeatureBuffer::new(max_leaf_samples, config.max_features),
            solver: WeightedLeastSquaresSolver::new(config.max_features),
            config,
        }
    }
    
    /// Train linear models for all leaves (sequential)
    pub fn train<S: AsRef<[f32]>>(
        &mut self,
        tree: &mut MutableTree<ScalarLeaf>,
        col_matrix: &ColMatrix<f32, S>,
        partitioner: &RowPartitioner,
        gradients: &Gradients,
        output: usize,
        learning_rate: f32,
    ) {
        for leaf_node in tree.leaf_nodes() {
            let rows = partitioner.get_leaf_indices(leaf_node);
            if rows.len() < self.config.min_samples { continue; }
            
            let features = self.select_features(tree, leaf_node);
            self.feature_buffer.gather(rows, col_matrix, &features);
            
            if let Some(coefs) = fit_leaf(
                &mut self.solver,
                &self.feature_buffer,
                rows,
                gradients,
                output,
                &self.config,
            ) {
                // Apply learning rate to coefficients (matches constant leaf scaling)
                let scaled = coefs.scale(learning_rate);
                tree.set_leaf_coefficients(leaf_node, features, scaled);
            }
            // If fit_leaf returns None, leaf keeps constant value only
        }
    }
}
```

**Multi-output**: For multi-output regression, call `train()` once per output with
the corresponding `output` index. Each output gets independent coefficients.

**Performance**: Linear leaf training adds ~20-50% overhead vs standard GBDT,
depending on tree depth and leaf sizes. The main cost is the gather + CD solve.

### Parallel Training

For parallel execution, each thread needs its own buffers:

```rust
/// Thread-local resources for parallel leaf fitting
struct LeafFitContext {
    feature_buffer: LeafFeatureBuffer,
    solver: WeightedLeastSquaresSolver,
}

impl LeafLinearTrainer {
    pub fn train_parallel<S: AsRef<[f32]> + Sync>(
        &self,
        tree: &MutableTree<ScalarLeaf>,
        col_matrix: &ColMatrix<f32, S>,
        partitioner: &RowPartitioner,
        gradients: &Gradients,
        output: usize,
    ) -> Vec<(NodeId, Vec<u32>, Vec<f32>)> {
        let leaf_nodes: Vec<_> = tree.leaf_nodes().collect();
        
        // Each thread gets its own context via thread_local or pool
        leaf_nodes.par_iter()
            .filter_map(|&leaf| {
                // Get thread-local context
                let ctx = get_thread_local_context(&self.config);
                
                let rows = partitioner.get_leaf_indices(leaf);
                if rows.len() < self.config.min_samples { return None; }
                
                let features = self.select_features(tree, leaf);
                ctx.feature_buffer.gather(rows, col_matrix, &features);
                
                let coefs = fit_leaf(&mut ctx.solver, ...);
                Some((leaf, features.to_vec(), coefs.to_vec()))
            })
            .collect()
    }
}
```

Returns coefficients to be applied after parallel phase (avoids mutable tree sharing).

### Partitioner Reuse

The grower's `RowPartitioner` already has row→leaf mapping after tree growth:

```rust
impl TreeGrower {
    pub fn partitioner(&self) -> &RowPartitioner { &self.partitioner }
}

// RowPartitioner already has:
pub fn get_leaf_indices(&self, leaf: LeafId) -> &[u32]
```

No new partitioner needed—just expose the existing one.

### Coefficient Storage

Follow the `CategoriesStorage` pattern for `freeze()`:

**During training** (MutableTree): Simple Vec accumulation

```rust
pub struct MutableTree<L: LeafValue> {
    // ... existing fields ...
    /// Linear coefficients: (node_idx, feature_indices, coefficients)
    linear_leaves: Vec<(NodeId, Vec<u32>, Vec<f32>)>,
}

impl MutableTree<L> {
    pub fn set_leaf_coefficients(&mut self, node: NodeId, features: Vec<u32>, coefs: Vec<f32>) {
        self.linear_leaves.push((node, features, coefs));
    }
}
```

**After freeze** (Tree): Flat packed storage

```rust
pub struct LeafCoefficients {
    feature_indices: Box<[u32]>,
    coefficients: Box<[f32]>,
    segments: Box<[(u32, u16)]>,  // (start, len) per node
}
```

The `freeze()` method builds `LeafCoefficients` from the Vec, exactly like it
builds `CategoriesStorage` from `categorical_nodes: Vec<(NodeId, Vec<u32>)>`.

**Empty coefficients**: If all coefficients are pruned (below threshold) or
fitting fails, the leaf has no entry in `LeafCoefficients`. The `leaf_terms()`
method returns `None`, and inference uses the constant base value.

**Serialization**: `LeafCoefficients` is included in tree serialization (JSON,
binary formats). Models with linear leaves can be saved and loaded normally.
Older library versions without linear leaf support will fail to load these models.

**No MutableLeafCoefficients struct**—just a Vec in MutableTree.

### Inference

No new `predict_row_with_linear`. Linear leaves are handled transparently:

```rust
impl Tree<ScalarLeaf> {
    pub fn predict_row(&self, features: &[f32]) -> f32 {
        let node = self.traverse_to_leaf(features);
        let base = self.leaf_value(node).0;
        
        // If no linear coefficients, this returns empty slices
        if let Some((feat_idx, coefs)) = self.leaf_coefficients.leaf_terms(node) {
            // Check for NaN in linear features—fall back to base if found
            for &f in feat_idx {
                if features[f as usize].is_nan() {
                    return base;
                }
            }
            let linear: f32 = feat_idx.iter()
                .zip(coefs)
                .map(|(&f, &c)| c * features[f as usize])
                .sum();
            base + linear
        } else {
            base
        }
    }
}
```

**NaN handling**: If any feature in the linear model is NaN at inference, return
the constant base value only. This matches training behavior (leaves with NaN
fall back to constant).

The `leaf_terms()` method returns `Option` or empty slices for nodes without
coefficients—standard trees have no overhead beyond the check.

**Model inspection**: For debugging and interpretability:

```rust
impl Tree<ScalarLeaf> {
    /// Iterate over all leaves with linear coefficients
    pub fn linear_leaves(&self) -> impl Iterator<Item = (u32, &[u32], &[f32])> { ... }
}
```

### LinearTree Wrapper (Alternative)

Instead of adding `leaf_coefficients` to Tree, we could wrap:

```rust
pub struct LinearTree {
    inner: Tree<ScalarLeaf>,
    coefficients: LeafCoefficients,
}

impl LinearTree {
    pub fn predict_row(&self, features: &[f32]) -> f32 {
        let node = self.inner.traverse_to_leaf(features);
        let base = self.inner.leaf_value(node).0;
        // ... apply coefficients
    }
}
```

**Pros**: Tree stays simple; explicit type distinction.
**Cons**: Forests need `enum { Standard(Tree), Linear(LinearTree) }` or trait objects.

**Decision**: Start with coefficients embedded in Tree. If it complicates the
code, refactor to LinearTree wrapper later.

## Design Decisions

### DD-1: Use Existing ColMatrix

No new RawFeatureView. Caller has ColMatrix for GBLinear anyway. Pass it to
LeafLinearTrainer.

### DD-2: Reuse Grower's Partitioner

`RowPartitioner::get_leaf_indices()` already exists. Just expose it from grower.

### DD-3: No MutableLeafCoefficients

Use `Vec<(NodeId, Vec<u32>, Vec<f32>)>` in MutableTree. Build `LeafCoefficients`
in `freeze()`, like CategoriesStorage.

### DD-4: Column-Major Buffer

LeafFeatureBuffer uses column-major layout for contiguous feature slices. Gather
features-first for cache-friendly ColMatrix reads.

### DD-5: Fixed GBDT Gradients

Use gradient/hessian from current boosting round directly. Single pass to
accumulate X^THX and X^Tg per leaf. Matches LightGBM approach.

Dynamic gradient approach (recalculating during CD) may be added via separate RFC
if needed for complex objectives. See `docs/research/gbdt/training/linear-trees.md`
for design considerations.

### DD-6: Transparent Linear Handling

No new `predict_row_with_linear`. Existing `predict_row` checks for coefficients
and applies them—zero overhead for standard trees (empty check is cheap).

### DD-7: Skip First Tree

First tree has homogeneous gradients: at round 0, all samples have
`gradient = y_i - base_score`. Linear regression on constant pseudo-targets
yields zero coefficients (intercept absorbs everything). Skip linear fitting
for first tree. Match LightGBM.

### DD-8: Start Embedded, Consider Wrapper

Embed coefficients in Tree initially. If complexity grows, refactor to LinearTree
wrapper. Easier to extract than to combine.

### DD-9: Parallel Across Leaves

Leaves are independent. Use rayon for parallel fitting. Returns Vec of coefficients,
apply to MutableTree after parallel phase (avoids mutable sharing).

## Implementation Plan

1. Add `linear_leaves` field to MutableTree (Vec storage)
2. Add `LeafCoefficients` to Tree (built in freeze)
3. Add `LeafFeatureBuffer` for contiguous column-major gathering
4. Add `compute_weight_update_slice` to GBLinear updater
5. Implement `LeafLinearTrainer` with parallel leaf fitting
6. Expose `partitioner()` from TreeGrower
7. Integrate into GBDTTrainer
8. Extend LightGBM loader

## Integration

| Component | Change |
|-----------|--------|
| RFC-0016 | `grow()` returns MutableTree |
| RFC-0014 | Add `compute_weight_update_slice` |
| MutableTree | Add `linear_leaves: Vec<...>` |
| Tree | Add `leaf_coefficients: LeafCoefficients` |
| TreeGrower | Add `partitioner()` getter |
| LeafLinearTrainer | New: `LeafFeatureBuffer` + parallel CD |

---

## Changelog

- **2025-12-18**: Status updated to Implemented
- **2025-12-18**: DD-10 added — LightGBM uses direct matrix solve (Eigen fullPivLu),
  not coordinate descent. Our CD approach is intentional for simplicity and L1 support.
- **2025-12-18**: Implementation note — Prediction overhead is ~5.4x vs standard
  trees (LightGBM achieves ~1.75x). Future optimization story created.
- **2025-12-18**: LightGBM loader completed — Parses `is_linear`, `leaf_const`,
  `leaf_features`, `leaf_coeff` fields. Predictions match exactly.
