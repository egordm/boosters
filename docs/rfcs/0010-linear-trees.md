# RFC-0010: Linear Leaves for GBDT

- **Status**: Implemented
- **Created**: 2025-12-17
- **Updated**: 2025-01-25
- **Implemented**: 2025-12-18
- **Depends on**: RFC-0002, RFC-0005, RFC-0009
- **Scope**: Linear model fitting at tree leaves

## Summary

Linear leaves fit `intercept + Σ(coef × feature)` at each leaf instead of constants.
Trained via coordinate descent, requires raw feature access during training.

## Motivation

Standard GBDT produces piece-wise constant predictions. Linear leaves add smooth
transitions within tree partitions—useful for:

- Continuous relationships within partitions
- Better extrapolation at tree edges
- Improved performance with shallow trees

LightGBM supports this (`linear_tree=True`). We use coordinate descent for numerical
stability (vs LightGBM's direct matrix solve).

**When NOT to use**:
- Deep trees (max_depth > 10) — too few samples per leaf
- Pure categorical problems — no linear structure
- High-noise domains — linear fits may overfit

## Design Overview

```
Tree Growth                    Linear Fitting
───────────────────────────────────────────────
grow_tree(data, grads)         
    │                          
    ├─► tree structure         
    │                          
    └─► partitioner ──────────► fit_leaf_models(tree, partitioner)
         (row→leaf mapping)         │
                                    ├─► gather features per leaf
                                    ├─► solve WLS via coord descent
                                    └─► store coefficients in tree
```

### Configuration

```rust
pub struct LinearLeafConfig {
    pub lambda: f32,           // L2 regularization (default: 0.01)
    pub alpha: f32,            // L1 for sparsity (default: 0.0)
    pub max_iterations: u32,   // CD iterations (default: 10)
    pub tolerance: f64,        // Convergence (default: 1e-6)
    pub min_samples: usize,    // Min samples to fit (default: 50)
}
```

Enable in GBDT config:

```rust
GBDTConfig {
    linear_leaves: Some(LinearLeafConfig::default()),
    ..
}
```

### Key Types

| Type | Purpose |
|------|---------|
| `LinearLeafConfig` | Configuration for linear fitting |
| `LeafCoefficients` | Packed storage in frozen `Tree` |
| `LeafLinearTrainer` | Orchestrates fitting with pre-allocated buffers |
| `WeightedLeastSquaresSolver` | CD solver for `min Σ hᵢ(yᵢ - xᵢᵀc)² + λ‖c‖²` |

### Training Flow

```
for each tree (except first):
    1. Grow tree structure via normal tree growing
    2. Get row→leaf mapping from partitioner
    3. For each leaf with >= min_samples:
       a. Gather raw features for leaf rows (column-major buffer)
       b. Build normal equations: X^THX, X^Tg
       c. Solve via coordinate descent
       d. Store coefficients in tree
    4. Freeze tree (builds LeafCoefficients)
```

**Skip first tree**: First tree has homogeneous gradients, linear fit yields zero coefficients.

### Inference

Transparent — existing `predict_row` handles linear coefficients:

```rust
impl Tree<ScalarLeaf> {
    fn predict_row(&self, features: &[f32]) -> f32 {
        let node = self.traverse_to_leaf(features);
        let base = self.leaf_value(node).0;
        
        // Add linear term if present
        if let Some((feat_idx, coefs)) = self.leaf_coefficients.get(node) {
            // NaN in linear features → fall back to base
            if feat_idx.iter().any(|&f| features[f as usize].is_nan()) {
                return base;
            }
            base + feat_idx.iter().zip(coefs).map(|(&f, &c)| c * features[f as usize]).sum::<f32>()
        } else {
            base
        }
    }
}
```

---

## Design Decisions

### DD-1: Coordinate Descent over Direct Solve

CD is simpler (no BLAS dependency), handles collinearity gracefully, supports L1,
and matches GBLinear infrastructure.

### DD-2: Reuse Grower's Partitioner

Tree grower already tracks row→leaf mapping. Expose via `grower.partitioner()`.

### DD-3: Column-Major Feature Buffer

Gather features column-by-column into column-major buffer. Matches `FeaturesView`
layout for cache-friendly reads.

### DD-4: Fixed GBDT Gradients

Use gradients from current boosting round directly (not iterative gradient updates).
Single pass to accumulate normal equations. Matches LightGBM approach.

### DD-5: Embedded Coefficients

Store `LeafCoefficients` in `Tree` rather than separate `LinearTree` wrapper.
Zero overhead for standard trees (empty coefficients check).

### DD-6: Learning Rate Applied to Coefficients

Learning rate scales linear coefficients same as constant leaf values.

---

## Performance

Training overhead: ~20-50% vs standard GBDT (feature gather + CD solve).

Prediction overhead: ~5x vs standard trees (LightGBM achieves ~1.75x).

---

## LightGBM Compatibility

Loader parses `is_linear`, `leaf_const`, `leaf_features`, `leaf_coeff` fields.
Predictions match LightGBM exactly.

---

## Changelog

- 2025-01-25: Simplified RFC — removed implementation details, focus on design
- 2025-12-18: Status updated to Implemented
- 2025-12-18: LightGBM loader completed
