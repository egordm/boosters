# RFC-0007: Tree Growing

- **Status**: Implemented
- **Created**: 2024-11-01
- **Updated**: 2025-01-21
- **Depends on**: RFC-0005, RFC-0006
- **Scope**: Decision tree construction with depth-wise and leaf-wise strategies

## Summary

Tree growing is orchestrated by `TreeGrower`, which builds decision trees using histogram-based split finding, row partitioning, and the subtraction trick. Two growth strategies are supported: depth-wise (XGBoost style) and leaf-wise (LightGBM style).

## Design

### Growth Strategies

The `GrowthStrategy` enum defines two tree construction approaches:

- **DepthWise** (`max_depth`): Expands all nodes at each depth level before proceeding deeper. Controlled by `max_depth` parameter (default: 6). XGBoost-style growth.

- **LeafWise** (`max_leaves`): Always expands the leaf with highest gain. Controlled by `max_leaves` parameter (default: 31). LightGBM-style growth that typically produces deeper, more accurate trees.

Each strategy initializes a `GrowthState` that manages the expansion queue:
- DepthWise uses level-by-level `Vec` buffers with `advance()` to move between depths
- LeafWise uses a max-heap (`BinaryHeap`) ordered by split gain

### Tree Construction

`TreeGrower::grow()` orchestrates the tree building process:

1. **Initialization**: Reset partitioner, histogram pool, and column sampler. Build root histogram.

2. **Root setup**: Compute gradient sums from histogram, find best split, push root to expansion state.

3. **Expansion loop**: While `state.should_continue()`:
   - Pop next candidates (`pop_next()`)
   - For each candidate, either make leaf (if no valid split) or apply split
   - Build child histograms using subtraction trick (smaller child built, larger = parent - smaller)
   - Find splits for children and push to state
   - Call `state.advance()` for depth-wise

4. **Finalization**: Convert remaining candidates to leaves, apply learning rate to all leaf weights.

The subtraction trick reduces histogram building by ~50%: only the smaller child's histogram is built from data; the larger child's histogram is computed by subtracting from the cached parent histogram.

### Row Partitioning

`RowPartitioner` manages sample assignment to tree nodes:

- Single contiguous `indices` buffer with per-leaf ranges (`leaf_begin`, `leaf_count`)
- Stable in-place partitioning preserves row order for cache efficiency
- Tracks `leaf_sequential_start` to detect contiguous indices (enables O(1) gradient access vs O(n) gather)

Split partitioning uses typed dispatch (`FeatureView::U8`/`U16`) for efficient bin access. Missing values are routed based on `default_left` from the split.

### Regularization

**Learning rate**: Applied to final leaf weights via `tree_builder.apply_learning_rate()`. Also cached per-node for fast prediction updates via `update_predictions_from_last_tree()`.

**Leaf weight computation** (in `GainParams`):
```
weight = -sign(G) × max(0, |G| - α) / (H + λ)
```
Where α = L1 regularization, λ = L2 regularization.

**Split gain** (XGBoost formula):
```
gain = 0.5 × [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G_P²/(H_P + λ)] - γ
```

**Constraints**: `min_child_weight` (minimum hessian sum), `min_samples_leaf`, `min_gain` (γ) threshold.

## Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `GrowthStrategy` | `expansion.rs` | Config enum: `DepthWise { max_depth }` or `LeafWise { max_leaves }` |
| `GrowthState` | `expansion.rs` | Runtime state with `push()`, `pop_next()`, `advance()`, `should_continue()` |
| `NodeCandidate` | `expansion.rs` | Expansion candidate with node IDs, split info, gradient sums |
| `TreeGrower` | `grower.rs` | Main orchestrator owning histogram pool, partitioner, tree builder |
| `GrowerParams` | `grower.rs` | Growth config: `GainParams`, `learning_rate`, `GrowthStrategy`, `ColSamplingParams` |
| `RowPartitioner` | `partition.rs` | Row index management per leaf with stable split partitioning |
| `GainParams` | `split/gain.rs` | Regularization params: `reg_lambda`, `reg_alpha`, `min_gain`, `min_child_weight` |
| `SplitInfo` | `split/types.rs` | Split result: feature, threshold/categories, gain, default_left |
| `HistogramPool` | `histograms/` | LRU-cached histogram slots with subtraction support |

## Changelog

- 2025-01-21: Updated terminology to match refactored implementation; standardized header format
