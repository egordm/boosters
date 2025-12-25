# RFC-0005: Tree Growing

- **Status**: Implemented
- **Created**: 2024-11-01
- **Updated**: 2025-01-24
- **Depends on**: RFC-0004 (Binning and Histograms)
- **Scope**: Split finding and tree construction for GBDT

## Summary

This RFC covers split finding (greedy enumeration of histogram bins) and tree growing (construction of decision trees). The `TreeGrower` orchestrates the process using histogram-based splits, row partitioning, and the subtraction trick. Supports depth-wise and leaf-wise growth strategies.

## 1. Split Finding

### Gain Formula

Split gain uses the XGBoost formula with regularization:

$$
\text{gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G_P^2}{H_P + \lambda} \right] - \gamma
$$

Where:
- $G_L, G_R, G_P$ = gradient sums for left, right, parent
- $H_L, H_R, H_P$ = hessian sums for left, right, parent
- $\lambda$ = L2 regularization (`reg_lambda`)
- $\gamma$ = minimum gain threshold (`min_gain`)

**Optimization**: Parent score is precomputed once per node via `NodeGainContext`, reducing from 3 divisions to 2 per candidate.

### Leaf Weight Formula

$$
w = -\frac{\text{sign}(G) \cdot \max(0, |G| - \alpha)}{H + \lambda}
$$

Where $\alpha$ = L1 regularization (`reg_alpha`). When $\alpha = 0$, simplifies to Newton step $w = -G/(H + \lambda)$.

### Split Enumeration

**Numerical Features** — Bidirectional scanning:
1. **Forward scan**: Accumulate left stats bin-by-bin, missing values go right (`default_left = false`)
2. **Backward scan** (only if `has_missing`): Accumulate right stats, missing values go left (`default_left = true`)
3. Return split with highest gain

**Categorical Features** — Strategy based on cardinality:
- **One-hot** (≤ `max_onehot_cats`, default 4): Try each category as singleton left partition, O(k)
- **Sorted partition** (> `max_onehot_cats`): Sort categories by gradient/hessian ratio, scan for optimal binary partition, O(k log k)

### Constraints

Splits are rejected if they violate:
- `min_child_weight`: Minimum hessian sum per child (default: 1.0)
- `min_samples_leaf`: Minimum sample count per child (default: 1)
- `min_gain`: Minimum split gain threshold (default: 0.0)

Validation check:
```rust
hess_left >= min_child_weight
    && hess_right >= min_child_weight
    && count_left >= min_samples_leaf
    && count_right >= min_samples_leaf
```

### Parallel Split Finding

`GreedySplitter` supports parallel feature evaluation via rayon:
- Self-corrects to sequential when `features.len() < 16`
- Parallel mode evaluates features concurrently, reduces with max gain
- Categorical sorted splits allocate per-thread scratch space

### Split Types

```rust
pub struct SplitInfo {
    pub feature: u32,           // Feature index
    pub gain: f32,              // Split gain (NEG_INFINITY if invalid)
    pub default_left: bool,     // Missing value direction
    pub split_type: SplitType,  // Numerical or Categorical
}

pub enum SplitType {
    Numerical { bin: u16 },                    // Bin threshold (inclusive left)
    Categorical { left_cats: CatBitset },      // Categories going left
}

pub struct GainParams {
    pub reg_lambda: f32,        // L2 regularization (default: 1.0)
    pub reg_alpha: f32,         // L1 regularization (default: 0.0)
    pub min_gain: f32,          // Minimum split gain (default: 0.0)
    pub min_child_weight: f32,  // Min hessian per child (default: 1.0)
    pub min_samples_leaf: u32,  // Min samples per child (default: 1)
}
```

---

## 2. Tree Growing

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

### Subtraction Trick

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

## Key Types Summary

| Type | Purpose |
|------|---------|
| `GrowthStrategy` | Config enum: `DepthWise { max_depth }` or `LeafWise { max_leaves }` |
| `GrowthState` | Runtime state with `push()`, `pop_next()`, `advance()`, `should_continue()` |
| `NodeCandidate` | Expansion candidate with node IDs, split info, gradient sums |
| `TreeGrower` | Main orchestrator owning histogram pool, partitioner, tree builder |
| `GrowerParams` | Growth config: `GainParams`, `learning_rate`, `GrowthStrategy`, `ColSamplingParams` |
| `RowPartitioner` | Row index management per leaf with stable split partitioning |
| `GainParams` | Regularization params: `reg_lambda`, `reg_alpha`, `min_gain`, `min_child_weight` |
| `SplitInfo` | Split result: feature, threshold/categories, gain, default_left |
| `GreedySplitter` | Feature-parallel split search with categorical support |

## Integration

| Component | Integration Point |
| --------- | ----------------- |
| RFC-0004 (Binning) | Histograms provide bin-level gradient sums |
| RFC-0002 (Trees) | `TreeGrower` produces `MutableTree`, freezes to `Tree` |
| RFC-0006 (Training) | Trainer invokes `grow()` per boosting round |

## Changelog

- 2025-01-24: Merged RFC-0006 (Split Finding) and RFC-0007 (Tree Growing) into unified RFC.
- 2025-01-21: Updated terminology to match refactored implementation.
