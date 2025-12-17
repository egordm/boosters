# Sampling Strategies in Gradient Boosting

## Overview

Row sampling (also called bagging or subsampling) trains each tree on a subset of the data. This provides:

- **Regularization**: Reduces overfitting by introducing randomness
- **Speed**: Less data to process per tree
- **Variance reduction**: Ensemble of diverse trees

There are two main approaches:

1. **Random Subsampling**: Uniform random selection
2. **GOSS**: Gradient-based One-Side Sampling (prioritizes informative samples)

## Random Subsampling

### Algorithm

```text
function random_subsample(data, subsample_rate, seed):
    n = len(data)
    k = floor(n * subsample_rate)
    
    # Random selection without replacement
    indices = random_choice(range(n), size=k, replace=False, seed=seed)
    
    return data[indices]
```

### Properties

| Property | Value |
|----------|-------|
| Selection | Uniform random |
| Sample size | `subsample_rate × n` |
| Weight adjustment | None needed |
| Bias | Unbiased |

### Parameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `subsample` | 0.5 - 1.0 | Fraction of rows to sample per tree |

### When to Use

- **Large datasets**: Speed up training proportionally
- **Overfitting**: Add regularization through randomness
- **Ensemble diversity**: Each tree sees different data

### Example

```text
Dataset: 1,000,000 rows
subsample = 0.5

Each tree trains on:
  - 500,000 randomly selected rows
  - Different random seed per tree
```

## GOSS (Gradient-based One-Side Sampling)

GOSS is a smarter sampling strategy that keeps samples with large gradients (informative) and randomly samples from the rest.

### Intuition

Not all samples are equally important for learning:

- **Large gradient**: Model is making big mistakes → very informative
- **Small gradient**: Model is already accurate → less informative

GOSS keeps all the "hard" examples and subsamples the "easy" ones.

### Algorithm

```text
function goss_sample(gradients, top_rate, other_rate):
    n = len(gradients)
    
    # 1. Compute importance scores
    scores = |gradients|  # Absolute gradient magnitude
    
    # 2. Find top-k samples
    top_k = floor(n * top_rate)
    top_indices = argtopk(scores, top_k)
    
    # 3. Random sample from rest
    other_indices = set(range(n)) - set(top_indices)
    other_k = floor(len(other_indices) * other_rate)
    sampled_other = random_choice(other_indices, size=other_k)
    
    # 4. Weight amplification for sampled-other
    weight = (1 - top_rate) / other_rate
    for i in sampled_other:
        gradients[i] *= weight
        hessians[i] *= weight
    
    return top_indices + sampled_other
```

### Weight Amplification

The sampled "other" rows need weight adjustment to maintain unbiased gradient estimates:

```text
Population of "other" rows: (1 - top_rate) × n
Sampled "other" rows:       other_rate × (1 - top_rate) × n

Sampling rate for "other": other_rate
Weight multiplier:         1 / other_rate × (1 - top_rate) = (1 - top_rate) / other_rate
```

### Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `top_rate` | 0.2 | Fraction of high-gradient samples to keep |
| `other_rate` | 0.1 | Fraction of remaining samples to randomly select |

### Effective Sample Rate

```text
effective_rate = top_rate + other_rate × (1 - top_rate)

Example: top_rate=0.2, other_rate=0.1
         = 0.2 + 0.1 × 0.8
         = 0.2 + 0.08
         = 0.28 (28% of data)
```

### Multi-Output Case

For multi-output objectives (multiclass, multi-quantile), each sample has K gradients. We need a single importance score per sample:

**Option 1: Sum of products (LightGBM)**
$$
\text{score}_i = \sum_{k=1}^{K} |g_{i,k} \times h_{i,k}|
$$

**Option 2: L2 norm of gradients**
$$
\text{score}_i = \sqrt{\sum_{k=1}^{K} g_{i,k}^2}
$$

Both capture the overall "importance" of a sample across all outputs.

### Warm-up Period

LightGBM disables GOSS for the first few iterations:

```text
warm_up_rounds = 1 / learning_rate

if iteration < warm_up_rounds:
    use all data (no GOSS)
else:
    apply GOSS
```

Rationale: Early gradients are noisy and don't reflect true sample importance.

## Comparison: Random vs GOSS

```text
Random Subsampling (50%)           GOSS (top=20%, other=10%)

Sample Selection:                   Sample Selection:
  ○○●○●○○●●○○●○○●○                   ●●●●○○●○○○○●○○○○ (top 20%)
  (random 50%)                       ○○●○○○○○●○○○○○○○ (random 10% of rest)
                                     
  ● = selected                       Weight:
  ○ = not selected                   top samples: weight = 1
                                     other samples: weight = 4× (amplified)

Effective data: 50%                 Effective data: 28%
Gradient bias: none                 Gradient bias: none (due to weighting)
Variance: higher                    Variance: lower (keeps important samples)
```

### When to Use Each

| Scenario | Recommendation |
|----------|----------------|
| Baseline | Random subsampling |
| Large dataset, accuracy critical | GOSS |
| Highly imbalanced data | GOSS (keeps minority samples with large gradients) |
| Very large dataset (>10M rows) | GOSS (bigger speedup) |
| Small dataset | Random (or no sampling) |
| Debugging/reproducibility | Random (simpler to reason about) |

## Column Sampling

In addition to row sampling, gradient boosting supports column (feature) sampling:

### Levels of Column Sampling

```text
colsample_bytree:  Sample features once per tree
colsample_bylevel: Sample features once per depth level
colsample_bynode:  Sample features once per node
```

### Combined Effect

The sampling rates multiply:

```text
effective_feature_rate = colsample_bytree × colsample_bylevel × colsample_bynode

Example: 0.8 × 0.8 × 0.8 = 0.512 (51.2% of features per node)
```

### Implementation

```text
function get_features_for_node(tree, node, seed):
    available = range(num_features)
    
    # Tree-level sampling
    tree_seed = seed + tree.id
    available = sample(available, colsample_bytree, tree_seed)
    
    # Level-level sampling
    level_seed = seed + tree.id + node.depth
    available = sample(available, colsample_bylevel, level_seed)
    
    # Node-level sampling
    node_seed = seed + tree.id + node.id
    available = sample(available, colsample_bynode, node_seed)
    
    return available
```

## Stochastic Gradient Boosting

Combining row and column sampling creates **Stochastic Gradient Boosting**:

```text
for each tree:
    rows = sample_rows(data, subsample)      # or GOSS
    cols = sample_columns(features, colsample_bytree)
    
    for each node:
        node_cols = sample_columns(cols, colsample_bynode)
        histogram = build_histogram(rows, node_cols)
        split = find_best_split(histogram)
```

This provides multiple sources of randomness:

1. Which rows each tree sees
2. Which features each tree/level/node considers
3. Bootstrap-like variance reduction

## Performance Considerations

### Memory

| Strategy | Memory Impact |
|----------|---------------|
| Random subsample | Indices only (n × 4 bytes) |
| GOSS | Indices + weights (n × 8 bytes) |
| Column sample | Minimal (feature mask) |

### Computation

| Strategy | Speedup |
|----------|---------|
| Random 50% | ~2× faster histogram building |
| GOSS 28% | ~3.5× faster histogram building |
| Column 50% | ~2× fewer features to evaluate |

### GOSS Overhead

GOSS has additional overhead compared to random sampling:

1. Compute gradient magnitudes: O(n)
2. Partial sort for top-k: O(n)
3. Weight multiplication: O(n × other_rate)

For large datasets, this overhead is small compared to histogram building savings.

## Practical Recommendations

### Starting Point

```python
# Moderate regularization
params = {
    'subsample': 0.8,           # 80% of rows
    'colsample_bytree': 0.8,    # 80% of features per tree
    'colsample_bylevel': 1.0,   # All features per level
    'colsample_bynode': 1.0,    # All features per node
}
```

### For Large Datasets (>1M rows)

```python
# Aggressive subsampling for speed
params = {
    'boosting_type': 'goss',  # LightGBM
    'top_rate': 0.2,
    'other_rate': 0.1,
    # or for XGBoost:
    'subsample': 0.5,
}
```

### For Overfitting Problems

```python
# More randomness = more regularization
params = {
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'colsample_bynode': 0.8,
}
```

## References

- [LightGBM Paper](https://papers.nips.cc/paper/6907) - Section 4 (GOSS)
- XGBoost: Stochastic gradient boosting via `subsample` parameter
- LightGBM: `src/boosting/goss.hpp`
- [Stochastic Gradient Boosting](https://statweb.stanford.edu/~jhf/ftp/stobst.pdf) - Friedman, 2002
