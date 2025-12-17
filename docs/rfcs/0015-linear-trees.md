# RFC-0015: Linear Leaves for GBDT

- **Status**: Draft
- **Created**: 2025-12-17
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

## Design

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

#### Fitting Flow

```rust
/// Fit linear coefficients for one leaf
fn fit_leaf(
    solver: &mut WeightedLeastSquaresSolver,
    feature_buffer: &LeafFeatureBuffer,
    rows: &[u32],
    gradients: &Gradients,
    output: usize,
    config: &LinearConfig,
) -> LeafCoefficients {
    let n_features = feature_buffer.n_features;
    solver.reset(n_features);
    
    // Accumulate normal equations (single pass over leaf data)
    for (buf_idx, &orig_row) in rows.iter().enumerate() {
        let gh = gradients.get_pair(orig_row as usize, output);
        
        // Gather feature values for this row
        let row_features: Vec<f32> = (0..n_features)
            .map(|f| feature_buffer.feature_slice(f)[buf_idx])
            .collect();
        
        solver.accumulate(&row_features, gh.grad, gh.hess);
    }
    
    solver.add_regularization(config.lambda as f64, n_features);
    let (intercept, coefs) = solver.solve_cd(n_features, &config.cd_config);
    
    LeafCoefficients::new(*intercept as f32, coefs.iter().map(|&c| c as f32).collect())
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
    ) {
        for leaf_node in tree.leaf_nodes() {
            let rows = partitioner.get_leaf_indices(leaf_node);
            if rows.len() < self.config.min_samples { continue; }
            
            let features = self.select_features(tree, leaf_node);
            self.feature_buffer.gather(rows, col_matrix, &features);
            
            let coefs = fit_leaf(
                &mut self.solver,
                &self.feature_buffer,
                rows,
                gradients,
                output,
                &self.config,
            );
            
            tree.set_leaf_coefficients(leaf_node, features, coefs);
        }
    }
}
```

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

The `leaf_terms()` method returns `Option` or empty slices for nodes without
coefficients—standard trees have no overhead beyond the check.

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

First tree approximates mean—poor signal for linear coefficients. Match LightGBM.

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
