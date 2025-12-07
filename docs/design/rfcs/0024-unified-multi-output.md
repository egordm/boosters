# RFC-0024: Multi-Output Training Strategy

- **Status**: Accepted
- **Created**: 2024-12-01
- **Updated**: 2024-12-02
- **Depends on**: None
- **Scope**: Training strategy for multiclass and multi-target models

## Summary

This RFC documents our decision to use the "one tree per output" strategy for
multi-output training (multiclass classification, multi-target regression).
This matches the default behavior of both XGBoost and LightGBM.

## Motivation

Multi-output models (multiclass classification, multi-target regression) require
training to produce predictions for K outputs simultaneously. The training
algorithm must decide how to handle K-dimensional gradients and produce trees
that can predict all K outputs.

Two fundamentally different strategies exist:

1. **One Tree Per Output**: Train K separate trees per boosting round
2. **Vector Leaves**: Train 1 tree per round with K-dimensional leaf values

We need to decide which strategy to implement, considering compatibility with
XGBoost/LightGBM models, implementation complexity, and performance.

## Design

### Overview

Booste-rs uses the "one tree per output" strategy:

- Each boosting round trains K separate trees (one per output)
- Each tree has scalar leaf values
- Forest contains K×N trees for N rounds with K outputs
- Tree structure can differ between outputs (different splits optimal per output)

### Architecture

```text
GBTreeTrainer
├── train()           → Single-output, produces N trees
└── train_multiclass() → K outputs, produces K×N trees
                         Internally calls train() K times per round
```

### Data Structures

Gradient buffer supports K outputs:

```rust
pub struct GradientBuffer {
    grads: Box<[f32]>,  // [n_samples * n_outputs]
    hess: Box<[f32]>,   // [n_samples * n_outputs]
    n_samples: u32,
    n_outputs: u16,
}
```

Split info and building nodes use scalar statistics:

```rust
pub struct SplitInfo {
    pub feature: u32,
    pub threshold: f32,
    pub gain: f32,
    pub grad_left: f32,      // Scalar (not K-dimensional)
    pub hess_left: f32,
    pub weight_left: f32,
    // ...
}

pub struct BuildingNode {
    pub weight: f32,         // Scalar (not K-dimensional)
    pub split: Option<SplitInfo>,
    // ...
}
```

Forest tracks group membership:

```rust
pub struct SoAForest<L: LeafValue> {
    trees: Vec<SoATreeStorage<L>>,  // K×N trees for multiclass
    base_scores: Vec<f32>,          // One per output (K values)
    num_groups: u32,                // K for multiclass, 1 for regression
}
```

### API

```rust
// Single-output training
let forest = trainer.train(policy, &quantized, &labels, &cuts, &[]);

// Multiclass training (K trees per round)
let loss = SoftmaxLoss::new(num_classes);
let forest = trainer.train_multiclass(&loss, policy, &quantized, &labels, &cuts, &[]);
```

### Multiclass Training Loop

```rust
for round in 0..num_rounds {
    // Compute gradients for all K outputs
    loss.compute_gradients(&predictions, &labels, &mut grad_buffer);
    
    // Train K trees, one per output
    for output in 0..num_classes {
        let (grads, hess) = grad_buffer.grads_for_output(output);
        let tree = grower.grow_tree(grads, hess);
        forest.push(tree, group = output);
        
        // Update predictions for this output
        update_predictions(&tree, output, &mut predictions);
    }
}
```

### Prediction

```rust
// Tree i belongs to group (i % num_groups)
for (tree_idx, tree) in forest.trees().enumerate() {
    let group = tree_idx % forest.num_groups();
    let leaf = tree.predict_row(features);
    scores[group] += leaf.value();
}
```

## Design Decisions

### DD-1: One Tree Per Output vs Vector Leaves

**Context**: We need to support multi-output training (multiclass, multi-target).
Two strategies exist with different trade-offs.

**Options considered**:

1. **One tree per output** — Train K trees per round, each with scalar leaves
2. **Vector leaves** — Train 1 tree per round with K-dimensional leaves
3. **Both** — Support both strategies via configuration

**Decision**: We chose **Option 1 (one tree per output only)** because:

1. **Industry standard**: Both XGBoost and LightGBM default to this strategy
2. **Simpler implementation**: All tree building code remains single-output
3. **Memory efficient**: Histograms store 2 floats/bin vs 2K floats/bin
4. **Feature complete**: Monotonic constraints, interaction constraints work without modification
5. **No significant use case**: Vector leaves help correlated outputs, which is uncommon

**Consequences**:

- Simpler codebase with single training path
- Cannot train models with shared tree structure across outputs
- Can still load XGBoost models trained with `multi_strategy="multi_output_tree"` (inference only)

### DD-2: Inference Support for Vector Leaves

**Context**: XGBoost supports `multi_strategy="multi_output_tree"` which produces
models with K-dimensional leaf values. We need to decide whether to support
loading these models for inference.

**Options considered**:

1. **No support** — Only load models with scalar leaves
2. **Full support** — Load and predict with vector leaf models

**Decision**: We chose **Option 2 (full support)** because:

1. **Compatibility**: Users may have XGBoost models trained with vector leaves
2. **Minimal cost**: Only adds one `VectorLeaf` type and prediction path
3. **Future-proofing**: If we add vector leaf training later, inference is ready

**Consequences**:

- `LeafValue` trait with `ScalarLeaf` and `VectorLeaf` implementations
- Slightly more complex prediction code (generic over leaf type)
- Full compatibility with all XGBoost models

### DD-3: Gradient Buffer Design

**Context**: Multi-output training requires gradients for all K outputs per sample.
We need to decide how to store and access these gradients.

**Options considered**:

1. **K separate buffers** — One `GradientBuffer` per output
2. **Single interleaved buffer** — All outputs in one buffer, `[g0_k0, g0_k1, ..., g1_k0, ...]`
3. **Single contiguous buffer** — All outputs contiguous, `[all_k0, all_k1, ...]`

**Decision**: We chose **Option 3 (contiguous buffer)** because:

1. **Cache-friendly**: When training tree for output k, all gradients are contiguous
2. **Simple slicing**: `grads_for_output(k)` returns a contiguous slice
3. **XGBoost pattern**: Matches XGBoost's `Slice(linalg::All(), k)` approach

**Consequences**:

- Layout: `grads[sample * n_outputs + output]` for per-sample access
- Layout: `grads[output * n_samples..][sample]` for per-output slicing
- Efficient iteration when training individual trees

## Integration

| Component | Integration Point | Notes |
|-----------|------------------|-------|
| Tree Grower | Uses scalar `SplitInfo` | No changes needed for multi-output |
| Histogram Builder | Single-output histograms | One histogram per tree, not per output |
| Loss Functions | Computes K gradients | `SoftmaxLoss`, `MultiLogLoss` |
| Forest | Stores K×N trees | `num_groups` tracks K |
| Prediction | Groups trees by output | `tree_idx % num_groups` |

## Future Work

If demand arises for vector leaf training:

- [ ] Add `multi_strategy` parameter to `TrainerParams`
- [ ] Implement K-dimensional histogram accumulation  
- [ ] Implement multi-output gain computation (sum across K outputs)
- [ ] Create `MultiOutputTreeGrower`

However, we don't anticipate this need given that:
- XGBoost's `multi_output_tree` is rarely used
- LightGBM doesn't support it at all

## References

- [Multi-Output Training Research](../research/xgboost-gbtree/training/multi_output.md)
- [Tree Growing Strategies](../research/xgboost-gbtree/training/tree_growing.md)
- [XGBoost multi_strategy parameter](https://xgboost.readthedocs.io/en/stable/parameter.html)

## Changelog

- 2024-12-01: Initial draft
- 2024-12-02: Rewrote after research; documented one-tree-per-output decision
- 2024-12-02: Removed multi-output histogram infrastructure; histograms now single-output only
