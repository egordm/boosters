# RFC-0010: Sampling Strategies

**Status**: Implemented

## Summary

Row and column sampling strategies for regularization and computational efficiency. Row sampling supports uniform (bagging) and GOSS (Gradient-based One-Side Sampling). Column sampling supports three-level cascading: bytree, bylevel, and bynode.

## Design

### Row Sampling

Row sampling modifies gradients in-place by zeroing out unsampled rows.

**Uniform Sampling**: Standard bagging with reservoir sampling for efficient O(n) selection without shuffling the full dataset. Selected rows are tracked via a mask buffer to avoid repeated allocations.

**GOSS Sampling**: Keeps high-gradient samples (most informative) and randomly samples from the rest:
1. Compute gradient magnitude: `|grad × hess|`
2. Keep top `top_rate` fraction (high gradient)
3. Randomly sample `other_rate` fraction from remaining
4. Apply amplification factor to sampled small gradients

### GOSS Details

**Algorithm**:
- Use quickselect (`select_nth_unstable`) to find threshold for top-k
- Adaptive probability sampling for small gradients: `prob = rest_needed / rest_remaining`
- Amplification factor: `(n - top_k) / other_k` corrects distribution bias

**Warmup**: GOSS skips sampling for the first `⌊1/learning_rate⌋` iterations. Early iterations have uninformative gradients since the model hasn't learned patterns yet.

**Parameters**:
- `top_rate`: Fraction of high-gradient samples to keep (e.g., 0.2)
- `other_rate`: Fraction of remaining samples to randomly select (e.g., 0.1)
- Constraint: `top_rate + other_rate ≤ 1.0`

### Column Sampling

Three-level cascading feature sampling (matches XGBoost design):

| Level | When Applied | Scope |
|-------|--------------|-------|
| `colsample_bytree` | Once per tree | Filters features for entire tree |
| `colsample_bylevel` | When depth changes | Further filters tree features |
| `colsample_bynode` | Every split finding | Further filters level features |

**Effective rate**: `bytree × bylevel × bynode`

**Implementation**:
- Partial Fisher-Yates shuffle for O(k) selection
- Features sorted after sampling for cache-friendly histogram access
- Minimum 1 feature guaranteed at each level
- Level sampling only triggers when depth actually changes

## Key Types

```rust
/// Row sampling configuration
pub enum RowSamplingParams {
    None,
    Uniform { subsample: f32 },
    Goss { top_rate: f32, other_rate: f32 },
}

/// Column sampling configuration  
pub enum ColSamplingParams {
    None,
    Sample {
        colsample_bytree: f32,
        colsample_bylevel: f32,
        colsample_bynode: f32,
    },
}

/// Row sampler (stateful, per-iteration)
pub struct RowSampler {
    config: RowSamplingParams,
    rng: SmallRng,
    indices: Vec<u32>,
    uniform_mask: Vec<u8>,
    grad_magnitudes: Vec<f32>,  // GOSS only
    warmup_rounds: usize,       // GOSS only
}

/// Column sampler (stateful, per-tree/level/node)
pub struct ColSampler {
    config: ColSamplingParams,
    n_features: u32,
    rng: SmallRng,
    tree_features: Vec<u32>,
    level_features: Vec<u32>,
    node_features: Vec<u32>,
    current_depth: u16,
}
```

## Usage Pattern

```rust
// Row sampling (each iteration)
let indices = row_sampler.sample(iteration, &mut grad_hess);
// grad_hess is modified in-place (zeros or amplified)

// Column sampling (hierarchical)
col_sampler.sample_tree();           // Start of tree
col_sampler.sample_level(depth);     // At each level
let features = col_sampler.sample_node();  // For split finding
```

## Sample Weight Interaction

Row sampling and sample weights are orthogonal:
- Sample weights affect gradient/hessian computation in objectives
- Row sampling operates on the resulting gradients
- GOSS amplification multiplies both grad and hess, preserving weight ratios
