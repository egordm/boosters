# GOSS (Gradient-based One-Side Sampling)

## Overview

Gradient-based One-Side Sampling (GOSS) is a data sampling technique introduced in LightGBM
that selects training instances based on gradient magnitudes. The key insight is that samples
with larger gradients contribute more to the information gain, so we can focus computational
effort on these informative samples.

## Algorithm

### Single-Output Case

For regression or binary classification (single output per sample):

1. **Sort by gradient magnitude**: Rank all samples by |gradient|
2. **Keep top samples**: Always include top `top_rate` fraction (e.g., 20%)
3. **Random sample remaining**: Sample `other_rate` fraction (e.g., 10%) from remaining
4. **Weight amplification**: Multiply gradients/hessians of sampled rows by `(1-top_rate)/other_rate`

The weight amplification compensates for undersampling the small-gradient population,
ensuring unbiased gradient estimates.

### Multi-Output Case (Multiclass, Multi-Quantile)

For multi-output objectives like softmax or multi-quantile, each sample has K gradient values
(one per output). LightGBM handles this by computing a single importance score per sample:

**LightGBM's approach** (from `src/boosting/goss.hpp`):

```cpp
// Sum of |gradient × hessian| across all outputs
for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration; ++cur_tree_id) {
    tmp_gradients[i] += std::fabs(gradients[idx] * hessians[idx]);
}
```

**Our implementation** (booste-rs):

We use the L2 norm of the gradient vector instead, which is mathematically cleaner:

```rust
// L2 norm of gradient vector: sqrt(sum(grad_k^2))
let magnitude = (0..num_outputs)
    .map(|k| grads.get(row, k).0.powi(2))
    .sum::<f32>()
    .sqrt();
```

Both approaches achieve the same goal: samples with large gradient magnitudes across any
output dimension are considered "important" and always kept.

### Weight Application

Once sampled, the weight amplification factor is applied to **all outputs** for each
sampled (non-top) row:

```rust
for output_idx in 0..num_outputs {
    let (grad, hess) = grads.get(row, output_idx);
    grads.set(row, output_idx, grad * weight, hess * weight);
}
```

This ensures gradient estimates remain unbiased across all output dimensions.

## Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `top_rate` | 0.2 | Fraction of high-gradient samples to always keep |
| `other_rate` | 0.1 | Fraction of remaining samples to randomly select |

The effective sample rate is approximately:

```text
effective_rate = top_rate + other_rate × (1 - top_rate)
```

For `top_rate=0.2, other_rate=0.1`:

```text
effective_rate = 0.2 + 0.1 × 0.8 = 0.28 (28% of data used)
```

## LightGBM Implementation Details

From `src/boosting/goss.hpp`:

1. **Warm-up period**: GOSS is disabled for the first `1/learning_rate` iterations
   to allow gradients to stabilize:

   ```cpp
   if (iter < static_cast<int>(1.0f / config_->learning_rate)) {
       return;  // Don't subsample
   }
   ```

2. **Threshold-based selection**: Uses partial sort (ArgMaxAtK) to find the gradient
   threshold that separates top samples from others.

3. **Online sampling**: The "other" samples are selected via reservoir sampling as
   the data is scanned, not in a separate pass.

## Comparison: GOSS vs Random Subsampling

| Aspect | GOSS | Random Subsample |
|--------|------|------------------|
| Selection | Gradient-based | Uniform random |
| Bias | Unbiased (via weights) | Unbiased |
| Variance | Lower (focuses on informative samples) | Higher |
| Overhead | Sorting gradients | None |
| Multi-output | Sum/L2 norm of gradients | Same indices for all outputs |

## Implementation in booste-rs

### Files

- `src/training/gbtree/sampling.rs`: `GossSampler`, `sample_multioutput()`
- `src/training/gbtree/trainer.rs`: Integration in `train_internal()`

### Usage

```rust
use booste_rs::training::{GBTreeTrainer, LossFunction};
use booste_rs::training::gbtree::RowSampling;

let trainer = GBTreeTrainer::builder()
    .loss(LossFunction::Softmax { num_classes: 3 })
    .row_sampling(RowSampling::Goss { top_rate: 0.2, other_rate: 0.1 })
    .build()
    .unwrap();
```

### Key Design Decisions

1. **L2 norm for multi-output**: We use `sqrt(sum(grad^2))` rather than LightGBM's
   `sum(|grad×hess|)` for mathematical consistency. Both work well in practice.

2. **Same sample for all outputs**: Following LightGBM, we sample once per round
   and use the same row indices for all output trees in that round.

3. **Weight applies to all outputs**: The amplification factor is applied uniformly
   across all output dimensions for sampled rows.

## References

- [LightGBM Paper](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- LightGBM source: `src/boosting/goss.hpp`
