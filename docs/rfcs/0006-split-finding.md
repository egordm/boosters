# RFC-0006: Split Finding

**Status**: Implemented

## Summary

Greedy split finding algorithm that exhaustively scans histogram bins to find the optimal split point with the highest gain, supporting both numerical and categorical features with L1/L2 regularization.

## Design

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

## Key Types

### `SplitInfo`
```rust
pub struct SplitInfo {
    pub feature: u32,           // Feature index
    pub gain: f32,              // Split gain (NEG_INFINITY if invalid)
    pub default_left: bool,     // Missing value direction
    pub split_type: SplitType,  // Numerical or Categorical
}
```

### `SplitType`
```rust
pub enum SplitType {
    Numerical { bin: u16 },                    // Bin threshold (inclusive left)
    Categorical { left_cats: CatBitset },      // Categories going left
}
```

### `GainParams`
```rust
pub struct GainParams {
    pub reg_lambda: f32,        // L2 regularization (default: 1.0)
    pub reg_alpha: f32,         // L1 regularization (default: 0.0)
    pub min_gain: f32,          // Minimum split gain (default: 0.0)
    pub min_child_weight: f32,  // Min hessian per child (default: 1.0)
    pub min_samples_leaf: u32,  // Min samples per child (default: 1)
}
```

### `GreedySplitter`
```rust
pub struct GreedySplitter {
    gain_params: GainParams,
    max_onehot_cats: u32,           // Threshold for one-hot vs sorted (default: 4)
    parallelism: Parallelism,       // Sequential or parallel feature search
    cat_scratch: Vec<(u32, f64)>,   // Reusable buffer for sorted categorical
}
```

## Implementation Notes

- Located in `src/training/gbdt/split/`
- `NodeGainContext` provides ~7% speedup by precomputing parent score
- Categorical scratch buffer capped at 256 categories to bound memory
- Missing value handling integrated into numerical scan (no separate bin)
