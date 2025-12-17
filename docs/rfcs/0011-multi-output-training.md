# RFC-0011: Multi-Output Training

**Status**: Implemented

## Summary

Multi-output training (multiclass, multi-quantile, etc.) uses a **one-tree-per-output** strategy: for K outputs, each boosting round trains K separate trees with scalar leaves, one per output class/dimension.

## Design Decision

### Why One-Tree-Per-Output Over Vector Leaves

**Vector-leaf approach**: Each tree has K-dimensional leaf values, producing K outputs per traversal.

**One-tree-per-output approach** (implemented): K separate trees, each with scalar leaves, one per output.

We chose one-tree-per-output because:

1. **Simpler training**: Each tree trains on its own gradient slice using the existing scalar-leaf grower unchanged. No special K-dimensional gain computation or leaf weight optimization needed.

2. **Same quality**: Empirically, both approaches achieve equivalent model quality—each output still gets gradients computed from the same loss function.

3. **Memory efficiency**: Scalar leaves avoid allocating K f32 values at every leaf node across all trees.

4. **Inference flexibility**: Trees can be evaluated independently, enabling potential parallelization per output.

The `VectorLeaf` type exists only for loading external models (e.g., XGBoost multi-target regression) that use vector-leaf encoding.

## Implementation

### Training Loop

The trainer iterates outputs per round, training one tree per output:

```rust
// trainer.rs: for each boosting round
for round in 0..n_trees {
    // Compute gradients for ALL outputs at once
    objective.compute_gradients(..., gradients.pairs_mut());
    
    // Grow one tree per output
    for output in 0..n_outputs {
        let grad_hess = gradients.output_pairs_mut(output);
        let sampled = row_sampler.sample(round, grad_hess);
        
        let tree = grower.grow(dataset, &gradients, output, sampled);
        
        // Update predictions for this output
        for row in 0..n_rows {
            predictions[output * n_rows + row] += tree.predict(&row);
        }
        
        forest.push_tree(tree, output as u32);
    }
}
```

### Gradient Buffer (`Gradients`)

Column-major layout optimizes for the per-output training pattern:

```rust
pub struct Gradients {
    data: Vec<GradsTuple>,  // [n_samples * n_outputs] pairs
    n_samples: usize,
    n_outputs: usize,
}
```

Layout: `[output0_sample0, output0_sample1, ..., output1_sample0, ...]`

Key method for training:
```rust
// Zero-copy contiguous slice for output k
fn output_pairs(&self, output: usize) -> &[GradsTuple]
fn output_pairs_mut(&mut self, output: usize) -> &mut [GradsTuple]
```

This layout gives perfect cache locality when building histograms for a single output.

### Tree Grower

The grower is output-agnostic—it receives a gradient slice and grows a scalar-leaf tree:

```rust
// grower.rs
pub fn grow(
    &mut self,
    dataset: &BinnedDataset,
    gradients: &Gradients,
    output: usize,              // Which output's gradients to use
    sampled_rows: Option<&[u32]>,
) -> Tree<ScalarLeaf>
```

The grower reads `gradients.output_pairs(output)` internally.

### Inference (Forest)

Trees are grouped by output via `tree_groups`:

```rust
pub struct Forest<L: LeafValue> {
    trees: Vec<Tree<L>>,
    tree_groups: Vec<u32>,  // tree_groups[i] = which output tree i belongs to
    n_groups: u32,          // K outputs
    base_score: Vec<f32>,   // K base scores
}
```

Prediction accumulates per group:
```rust
fn predict_row(&self, features: &[f32]) -> Vec<f32> {
    let mut output = self.base_score.clone();  // [K]
    
    for (tree, group) in self.trees_with_groups() {
        output[group as usize] += tree.predict_row(features).0;
    }
    
    output
}
```

### Objective Integration

Objectives declare their output count via `n_outputs()`:

```rust
impl Objective for SoftmaxLoss {
    fn n_outputs(&self) -> usize {
        self.num_classes  // K classes = K outputs
    }
}

impl Objective for PinballLoss {
    fn n_outputs(&self) -> usize {
        self.quantiles.len()  // N quantiles = N outputs
    }
}
```

The trainer queries this to determine how many trees to grow per round.

## Key Types

| Type | Purpose |
|------|---------|
| `Gradients` | Column-major gradient buffer with per-output slicing |
| `GradsTuple` | Interleaved gradient/hessian pair |
| `Forest<ScalarLeaf>` | Tree ensemble with group assignments |
| `tree_groups: Vec<u32>` | Maps each tree to its output index |
| `VectorLeaf` | K-dimensional leaf (inference-only, for external model compat) |

## Compatibility

`VectorLeaf` exists in `repr::gbdt` for loading XGBoost/LightGBM models that use vector-leaf encoding (e.g., multi-target regression). The `Forest<VectorLeaf>` type supports these models for inference, but training always produces `Forest<ScalarLeaf>`.
