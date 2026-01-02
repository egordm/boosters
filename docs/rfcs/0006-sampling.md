# RFC-0006: Sampling Strategies

**Status**: Implemented  
**Created**: 2025-12-15  
**Updated**: 2026-01-02  
**Scope**: Row and column sampling for regularization

## Summary

Sampling strategies reduce computation and improve generalization. Row sampling
(bagging, GOSS) selects training samples per tree. Column sampling selects
features at tree/level/node granularity.

## Why Sampling?

| Goal | Mechanism |
| ---- | --------- |
| Reduce overfitting | Random subsets prevent memorization |
| Speed up training | Fewer samples/features to process |
| Diversity | Each tree sees different data |

## Row Sampling

### Configuration

```rust
pub enum RowSamplingParams {
    None,
    Uniform { subsample: f32 },
    Goss { top_rate: f32, other_rate: f32 },
}
```

### Uniform Sampling

Standard bagging: randomly select `subsample` fraction of samples.

```rust
// 80% of samples per tree
let sampling = RowSamplingParams::uniform(0.8);
```

Implementation: zero out gradients for unselected samples. No data copying.

### GOSS (Gradient-based One-Side Sampling)

From LightGBM's "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
(Ke et al., 2017). GOSS exploits the insight that samples with small gradients
are already well-modeled and contribute little to the information gain.

**Key insight**: Data instances with different gradients play different roles
in computing information gain. Instances with larger gradients contribute more
to the information gain. To maintain the accuracy of information gain estimation,
we keep all instances with large gradients and only randomly sample from
instances with small gradients.

**Algorithm**:

1. Sort samples by absolute gradient magnitude: $|g_i \cdot h_i|$
2. Keep top $a$ fraction (large gradients, informative samples)
3. Randomly sample $b$ fraction from remaining $(1-a)$ samples
4. Amplify sampled small-gradient instances by factor $\frac{1-a}{b}$

**Mathematical formulation**:

For a dataset with $n$ instances, let $A$ be the top-$a \cdot n$ instances
sorted by gradient magnitude, and $B$ be a random sample of size $b \cdot n$
from the remaining instances. The estimated variance gain is:

$$\tilde{V}_j(d) = \frac{1}{n}\left(\frac{(\sum_{x_i \in A_l} g_i + \frac{1-a}{b}\sum_{x_i \in B_l} g_i)^2}{n_l^j(d)} + \frac{(\sum_{x_i \in A_r} g_i + \frac{1-a}{b}\sum_{x_i \in B_r} g_i)^2}{n_r^j(d)}\right)$$

The amplification factor $\frac{1-a}{b}$ compensates for the underrepresentation
of small-gradient samples, ensuring unbiased gradient sums.

**Typical values**:

- `top_rate = 0.2` (keep top 20% high-gradient samples)
- `other_rate = 0.1` (sample 10% from remaining 80%)
- Effective data usage: 0.2 + 0.8 × 0.1 = 28% of samples

```rust
// Keep top 20%, sample 10% of rest
let sampling = RowSamplingParams::goss(0.2, 0.1);
```

**Warmup period**: GOSS skips the first `⌊1/learning_rate⌋` rounds. Early
iterations have unreliable gradients since predictions are far from targets.
During warmup, all samples are used.

**Implementation details**:

1. Gradient importance computed as `|grad × hess|` for proper weighting
2. Partial sort (quickselect) used to find top-$a$ threshold efficiently: O(n)
3. Gradients modified in-place—no data copying
4. Same RNG seed ensures reproducibility

### Sampler API

```rust
pub struct RowSampler {
    config: RowSamplingParams,
    rng: SmallRng,
}

impl RowSampler {
    pub fn new(config: RowSamplingParams, seed: u64) -> Self;
    
    /// Apply sampling by modifying gradients in-place
    pub fn apply(&mut self, gradients: &mut Gradients, iteration: usize);
    
    /// Check if warmup period is active
    pub fn is_warmup(&self, iteration: usize) -> bool;
}
```

## Column Sampling

### ColSamplingParams

```rust
pub enum ColSamplingParams {
    None,
    ByTree { colsample: f32 },
    ByLevel { colsample: f32 },
    ByNode { colsample: f32 },
}
```

### Granularity

| Level | When Applied | Scope |
| ----- | ------------ | ----- |
| ByTree | Once per tree | All nodes share same features |
| ByLevel | When depth changes | Nodes at same depth share features |
| ByNode | Every split finding | Each node has independent features |

**Cascading**: In XGBoost, levels cascade: `ByTree` × `ByLevel` × `ByNode`.
We implement the primary level only for simplicity.

### ColSampler API

```rust
pub struct ColSampler {
    config: ColSamplingParams,
    n_features: u32,
    rng: SmallRng,
    active_features: Vec<u32>,
}

impl ColSampler {
    pub fn new(config: ColSamplingParams, n_features: u32, seed: u64) -> Self;
    
    /// Resample for new tree
    pub fn resample_tree(&mut self);
    
    /// Resample for new level (ByLevel only)
    pub fn resample_level(&mut self, depth: u32);
    
    /// Resample for new node (ByNode only)
    pub fn resample_node(&mut self);
    
    /// Get currently active features
    pub fn active_features(&self) -> &[u32];
}
```

## Integration with Training

Row sampling in trainer:

```rust
// Before each tree
row_sampler.apply(&mut gradients, tree_idx);
// gradients now has zeros for unsampled rows
```

Column sampling in grower:

```rust
// Before split finding at node
let active = col_sampler.active_features();
splitter.find_split(histogram, parent_stats, active);
```

## Files

| Path | Contents |
| ---- | -------- |
| `training/sampling/mod.rs` | Module exports |
| `training/sampling/row.rs` | `RowSampler`, `RowSamplingParams` |
| `training/sampling/column.rs` | `ColSampler`, `ColSamplingParams` |

## Design Decisions

**DD-1: Zero gradients, don't skip.** Zeroing unsampled gradients is simpler
than maintaining sample indices. Split finding naturally ignores zero-gradient
samples.

**DD-2: GOSS warmup.** Early gradients are noisy (all samples equally wrong).
Skip GOSS until model has learned basic patterns.

**DD-3: Single column sampling level.** XGBoost's cascading is complex.
Most users pick one level. Keep it simple, add cascading if needed.

**DD-4: SmallRng for speed.** Sampling happens every tree. Use fast RNG
(SmallRng) rather than cryptographic quality.

**DD-5: Deterministic with seed.** Same seed = same sampling sequence.
Important for reproducibility.

## Accuracy Impact

| Strategy | Typical Accuracy | Training Speed |
| -------- | ---------------- | -------------- |
| None (100%) | Baseline | 1.0× |
| Uniform 80% | -0.1% to -0.5% | 1.2× |
| Uniform 50% | -0.5% to -2% | 1.8× |
| GOSS (20%, 10%) | -0.1% to -0.5% | 1.5× |

GOSS typically matches uniform sampling quality with less data.

## Recommended Settings

| Use Case | Row Sampling | Column Sampling |
| -------- | ------------ | --------------- |
| Default | None | ByTree(0.8) |
| Large dataset | Uniform(0.8) | ByTree(0.8) |
| Overfitting | GOSS(0.2, 0.1) | ByNode(0.5) |
| Speed priority | Uniform(0.5) | ByTree(0.5) |

## Performance

Sampling overhead is negligible (<1% of training time):

- Uniform: O(n) gradient zeroing
- GOSS: O(n log n) for gradient sorting + O(n) amplification
- Column: O(m) feature selection

## Early Stopping Interaction

Sampling applies per-tree, not per-round. Early stopping evaluates on
validation set (not affected by training sampling). GOSS warmup may delay
early stopping convergence slightly.

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Uniform correctness | Correct fraction selected |
| GOSS correctness | Top samples kept, amplification applied |
| Determinism | Same seed → same samples |
| Column sampling | Correct features selected per granularity |
| Warmup | GOSS skips correct number of rounds |
