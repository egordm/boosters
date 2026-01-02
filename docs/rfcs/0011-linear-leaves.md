# RFC-0011: Linear Leaves

**Status**: Implemented  
**Created**: 2025-12-15  
**Updated**: 2026-01-02  
**Scope**: Fitting linear models within tree leaves

## Summary

Linear leaves fit `y = intercept + Σ(coef × feature)` at each tree leaf,
providing smoother predictions within each partition. Features used are
those on the root-to-leaf path.

## Why Linear Leaves?

Standard GBDT uses constant leaf values. This creates piecewise-constant
predictions that can be suboptimal when the true relationship is smoother.

| Leaf Type | Prediction | Smoothness |
| --------- | ---------- | ---------- |
| Constant | Single value | Discontinuous |
| Linear | Weighted sum | Continuous within leaf |

Linear leaves are useful when:

- Target function is smooth within tree partitions
- Dataset is small and trees are shallow
- Smooth gradient is needed (e.g., optimization downstream)

Note: This feature is sometimes called "linear trees" / "linear leaves" in other
libraries. We do not target an external serialization format here.

## Layers

### High Level

Users enable via `GBDTConfig`:

```rust
let config = GBDTConfig::builder()
    .linear_leaves(Some(LinearLeafConfig::default()))
    .build()?;
```

### Medium Level (Config)

```rust
pub struct LinearLeafConfig {
    pub lambda: f32,                 // L2 regularization
    pub alpha: f32,                  // L1 regularization (currently not applied by the solver)
    pub max_iterations: u32,         // Coordinate descent iterations
    pub tolerance: f64,              // Convergence tolerance
    pub min_samples: usize,          // Min samples to fit linear model
    pub coefficient_threshold: f32,  // Prune tiny coefficients
    pub max_features: usize,         // Max features per leaf model (path features capped)
}
```

### Medium Level (Trainer)

```rust
pub struct LeafLinearTrainer {
    config: LinearLeafConfig,
    feature_buffer: LeafFeatureBuffer,
    solver: WeightedLeastSquaresSolver,
}

impl LeafLinearTrainer {
    pub fn train(
        &mut self,
        tree: &MutableTree<ScalarLeaf>,
        data: &Dataset,
        partitioner: &RowPartitioner,
        leaf_node_mapping: &[(u32, u32)],
        gradients: &Gradients,
        output: usize,
        learning_rate: f32,
    ) -> Vec<FittedLeaf>;
}
```

### Low Level (Solver)

```rust
pub struct WeightedLeastSquaresSolver {
    max_features: usize,
    // internal buffers
}

impl WeightedLeastSquaresSolver {
    pub fn reset(&mut self, n_features: usize);
    pub fn accumulate_intercept(&mut self, grad_hess: &[GradsTuple]);
    pub fn accumulate_column(&mut self, feat_idx: usize, values: &[f32], grad_hess: &[GradsTuple]);
    pub fn accumulate_cross_term(
        &mut self,
        feat_i: usize,
        feat_j: usize,
        values_i: &[f32],
        values_j: &[f32],
        grad_hess: &[GradsTuple],
    );
    pub fn add_regularization(&mut self, lambda: f64);
    pub fn solve_cd(&mut self, max_iterations: u32, tolerance: f64) -> bool;
    pub fn coefficients(&self) -> (f64, &[f64]);
}
```

The solver builds and solves the normal equations $X^T H X$ / $X^T g$ using
pre-allocated buffers, then solves for coefficients via coordinate descent.

## Feature Selection

Linear leaves use only features on the path from root to leaf:

```text
        [age < 30]        ← feature: age
       /          \
[income < 50k]    ...     ← feature: income
    /     \
  [leaf]  ...

Linear model at leaf uses: age, income
```

Why path features?

- Already proven relevant by tree splits
- Bounded number (max = tree depth)
- Often highly predictive for that partition

Categorical splits are skipped (no linear relationship).

## Weighted Least Squares

Fitting minimizes Hessian-weighted squared error:

$$\min_{\beta} \sum_i h_i \cdot (r_i - X_i \cdot \beta)^2 + \lambda \|\beta\|^2$$

Where:

- $r_i = -g_i / h_i$ (pseudo-response)
- $h_i$ = Hessian (weight)
- $\lambda$ = L2 regularization

Solved by (1) accumulating the normal equations and (2) running coordinate
descent on the coefficient vector.

## Solver Options

Once we have the normal equations for a leaf,

$$A = X^T H X + \lambda I, \quad b = -X^T g$$

we need to solve $A\beta = b$ for the intercept and coefficients.

We considered three practical options.

### Option 1: Direct solve via matrix factorization (Cholesky/LDLT)

**Idea**: treat $A$ as (approximately) symmetric positive definite (SPD) when
`lambda > 0`, factorize once (e.g. Cholesky/LDLT), and solve in one shot.

**Pros**:

- Fast for small-to-medium feature counts (one-shot solve)
- Numerically stable when SPD assumptions hold
- Straightforward stopping behavior (no iteration)

**Cons**:

- Requires an SPD-capable linear algebra backend or custom factorization
- Still needs careful handling of near-singular / ill-conditioned cases
- Harder to extend to L1 regularization (would require different algorithms)

**When it wins**: L2-only (ridge) problems with moderate leaf feature count and
well-conditioned $A$.

### Option 2: Direct solve via general decomposition / explicit inverse (LU)

**Idea**: use a general solver (e.g. pivoted LU) and compute $\beta$.
Some libraries implement this as “invert then multiply” for convenience.

**Pros**:

- Works even when $A$ is not SPD
- Conceptually simple if you already have a linear algebra library

**Cons**:

- “Inverse then multiply” is usually less numerically robust than solving
    directly (and tends to amplify conditioning issues)
- Adds an external dependency (e.g. Eigen/BLAS)
- More sensitive to collinearity unless regularization is strong

**Reference**: LightGBM’s linear trees build $X^T H X$ and then use a pivoted LU
routine to obtain coefficients.

### Option 3 (Chosen): Coordinate descent (iterative)

**Idea**: solve $A\beta=b$ by iteratively updating one coordinate at a time.

**Pros**:

- No BLAS / heavy dependencies
- Naturally supports sparsity-inducing L1-style extensions (future)
- Handles ill-conditioning more gracefully in practice than explicit inversion
    (especially when combined with L2 regularization)

**Cons**:

- Iterative: runtime depends on `max_iterations` and convergence tolerance
- For larger feature counts, naive CD can have higher constant factors than a
    factorization-based solve

**Collinearity note**: coordinate descent does not “magically fix” collinearity.
The key stabilizer is ridge regularization (`lambda > 0`), which improves
conditioning and yields a unique solution.

```rust
// After we have xthx = X^T H X and xtg = X^T g (with sign conventions)
for iter in 0..max_iterations {
    let mut max_delta = 0.0;

    for j in 0..size { // size = 1 + n_features (intercept + features)
        // residual_j = xtg[j] - sum_{k != j} xthx[j,k] * coef[k]
        let mut residual = xtg[j];
        for k in 0..size {
            if k != j {
                residual -= xthx[j,k] * coef[k];
            }
        }
        let new_coef = residual / xthx[j,j];
        max_delta = max_delta.max(abs(new_coef - coef[j]));
        coef[j] = new_coef;
    }

    if max_delta < tolerance {
        break;
    }
}
```

### Why this solver structure?

We intentionally separate the problem into:

1. **Decomposition / accumulation**: compute $X^T H X$ and $X^T g$ from raw
     feature columns and per-row $(g_i, h_i)$.
2. **Optimization**: solve the resulting weighted least squares system.

This matches the actual implementation and makes it easy to reuse the same
infrastructure for different feature layouts (dense/sparse) while keeping the
numerical linear algebra localized.

### Trade-off: one-pass gradients vs iterative gradient recomputation

There are two conceptually different approaches:

- **(Current) Fixed-gradient leaf solve**: use the gradients/hessians from the
    current boosting round once, fit coefficients, and move on. This matches the
    idea of a Newton step for the leaf model.
- **Iterative leaf solve with gradient recomputation**: repeatedly recompute
    gradients/hessians as the leaf model changes (i.e. multiple inner rounds).

We chose the first approach for simplicity and speed; the second can, in
principle, improve fit quality on some objectives but significantly increases
training cost and complicates determinism and early stopping.

## Integration with Tree Growing

```text
TreeGrower.grow(dataset, gradients)
    │
    ├─► Build tree (normal splits)
    │
    ├─► If linear_leaves enabled:
    │       for each leaf:
    │           features ← path features
    │           samples ← partitioner.get_leaf_indices(node)
    │           fit_linear(features, samples, gradients)
    │           store in tree node
    │
    └─► Return Tree with linear coefficients
```

## Tree Storage

Linear coefficients are stored in a packed `LeafCoefficients` structure (in
`repr/gbdt/coefficients.rs`). Conceptually this is:

- flat arrays `feature_indices[]` and `coefficients[]`
- per-node `(start, len)` segment into the flat arrays
- per-node `intercepts[]`

This avoids per-leaf heap allocations and keeps inference reads cache-friendly.

## Inference

Prediction at a linear leaf:

```rust
fn predict_linear_leaf(sample: &[f32], leaf: &LinearLeafData) -> f32 {
    let mut value = leaf.intercept;
    for (i, &feature_idx) in leaf.features.iter().enumerate() {
        value += leaf.coefficients[i] * sample[feature_idx as usize];
    }
    value
}
```

**Learning rate (shrinkage)**: As with constant leaves, the contribution of a
tree is scaled by `learning_rate`. For linear leaves, this is applied to both
the intercept and all feature coefficients at training time, so inference uses
the stored coefficients directly.

Falls back to constant leaf value if linear data is missing.

Also falls back to the constant leaf value if any required feature value is NaN.

## Files

| Path | Contents |
| ---- | -------- |
| `training/gbdt/linear/config.rs` | `LinearLeafConfig` |
| `training/gbdt/linear/trainer.rs` | `LeafLinearTrainer`, `FittedLeaf` |
| `training/gbdt/linear/solver.rs` | `WeightedLeastSquaresSolver` |
| `training/gbdt/linear/buffer.rs` | `LeafFeatureBuffer` |
| `repr/gbdt/tree.rs` | `LinearLeafData` storage |

## Design Decisions

**DD-1: Path features only.** Limits model complexity and ensures features
are relevant. Tree depth bounds feature count.

**DD-2: Skip categorical features.** Linear relationships require numeric
features. Categorical splits don't imply linear effects.

**DD-3: Hessian weighting.** Aligns with gradient boosting objective
(Newton step interpretation). Better than unweighted least squares.

**DD-4: Coordinate descent.** We considered (1) factorization-based direct
solves, (2) general decomposition / explicit inverse-style solves, and (3)
coordinate descent. We chose coordinate descent because it avoids heavy
dependencies, has good numerical behavior with ridge regularization, and keeps
the door open to L1-style extensions.

**DD-5: Fallback to constant.** If linear fit fails (too few samples,
non-convergence), use the constant leaf value. Never worse than baseline.

Fitting fails when:

- Fewer than `min_samples` samples in leaf
- Feature matrix is singular (all same value)
- Coordinate descent doesn't converge in `max_iterations`

## Performance

Linear leaves add ~10-20% training time overhead:

- O(n_leaves × depth × iterations) for fitting
- Negligible inference overhead (few extra multiplications per leaf)

## Out of Scope

This RFC does not define an on-disk model format or any export/import guarantees
for linear leaves.

## Debugging

To verify linear leaves are active:

```rust
let model = GBDTModel::train(...);
let tree = model.forest().tree(0);
println!("Has linear leaves: {}", tree.has_linear_leaves());
```

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Linear fit | Known linear data → correct coefficients |
| Fallback | Too few samples → constant leaf used |
| Path features | Only split features used in linear model |
| Coefficient storage | `LeafCoefficients` packing matches fitted leaf output |
