# LightGBM Categorical Feature Handling

## Overview

LightGBM has native support for categorical features without one-hot encoding.
This is a significant advantage over XGBoost, which requires manual encoding.

## Two Strategies

LightGBM uses two strategies based on the number of categories:

| Categories | Strategy | Complexity |
|-----------|----------|------------|
| ≤ `max_cat_to_onehot` (default: 4) | One-hot split | O(k) |
| > `max_cat_to_onehot` | Gradient-sorted split | O(k log k) |

## One-Hot Strategy (Few Categories)

For features with few categories, test each category individually:

```cpp
// From FindBestThresholdCategoricalInner
if (use_onehot) {
  for (int t = bin_start; t < bin_end; ++t) {
    const auto grad = GET_GRAD(data_, t);
    const auto hess = GET_HESS(data_, t);
    
    // Split: {category t} vs {all other categories}
    double current_gain = GetSplitGains(
        sum_gradient - grad,     // left: all except t
        sum_hessian - hess,
        grad,                     // right: only t
        hess + kEpsilon,
        ...
    );
    
    if (current_gain > best_gain) {
      best_threshold = t;
      best_gain = current_gain;
    }
  }
}
```

**Result**: Split is `{one category}` vs `{rest}`

## Gradient-Sorted Strategy (Many Categories)

For features with many categories, LightGBM uses a clever sorting trick:

### Key Insight

For optimal binary split of categories:

1. **Compute per-category ratio**: `sum_gradient / (sum_hessian + smooth)`
2. **Sort categories by this ratio**
3. **Find optimal split point in sorted order** (like numerical feature)

This reduces O(2^k) possible splits to O(k log k).

### Implementation

```cpp
// 1. Filter categories with enough data
for (int i = bin_start; i < bin_end; ++i) {
  if (Common::RoundInt(GET_HESS(data_, i) * cnt_factor) >= config_->cat_smooth) {
    sorted_idx.push_back(i);
  }
}

// 2. Sort by gradient/hessian ratio
auto ctr_fun = [this](double sum_grad, double sum_hess) {
  return sum_grad / (sum_hess + config_->cat_smooth);
};

std::stable_sort(sorted_idx.begin(), sorted_idx.end(), 
    [this, &ctr_fun](int i, int j) {
      return ctr_fun(GET_GRAD(data_, i), GET_HESS(data_, i)) <
             ctr_fun(GET_GRAD(data_, j), GET_HESS(data_, j));
    });

// 3. Scan sorted order to find best split
// Try both directions: low-to-high and high-to-low
std::vector<int> find_direction = {1, -1};

for (auto dir : find_direction) {
  double sum_left_gradient = 0.0;
  double sum_left_hessian = kEpsilon;
  
  for (int i = 0; i < max_num_cat; ++i) {
    auto t = sorted_idx[start_pos];
    start_pos += dir;
    
    // Accumulate into left group
    sum_left_gradient += GET_GRAD(data_, t);
    sum_left_hessian += GET_HESS(data_, t);
    
    // Compute gain
    double current_gain = GetSplitGains(
        sum_left_gradient, sum_left_hessian,
        sum_gradient - sum_left_gradient,
        sum_hessian - sum_left_hessian,
        ...
    );
    
    if (current_gain > best_gain) {
      best_threshold = i;  // First i categories in sorted order
      best_gain = current_gain;
      best_dir = dir;
    }
  }
}
```

### Why This Works

For regression with squared loss, the optimal split of categories can be found by
sorting on their mean target value. For gradient boosting:

- Categories with similar `gradient/hessian` ratio should be grouped
- This ratio approximates the optimal leaf value
- Sorting creates natural groupings

### Output Format

The split stores which categories go left:

```cpp
if (is_splittable_) {
  output->num_cat_threshold = best_threshold + 1;
  output->cat_threshold = std::vector<uint32_t>(output->num_cat_threshold);
  
  if (best_dir == 1) {
    // Categories at sorted positions [0, best_threshold] go left
    for (int i = 0; i < output->num_cat_threshold; ++i) {
      output->cat_threshold[i] = sorted_idx[i] + offset;
    }
  } else {
    // Categories at sorted positions [used_bin-1, used_bin-1-best_threshold] go left
    for (int i = 0; i < output->num_cat_threshold; ++i) {
      output->cat_threshold[i] = sorted_idx[used_bin - 1 - i] + offset;
    }
  }
}
```

## Regularization for Categorical

Extra regularization parameters for categorical splits:

```cpp
// Extra L2 for categorical (prevents overfitting)
l2 += config_->cat_l2;  // default: 10.0

// Smoothing in ratio calculation
ctr = sum_grad / (sum_hess + config_->cat_smooth);  // default: 10.0

// Minimum data per category
if (cnt < config_->cat_smooth) {
  // Skip this category
}

// Maximum categories to consider
const int max_num_cat = std::min(config_->max_cat_threshold, (used_bin + 1) / 2);
```

## Configuration

```python
# Declare categorical features
params = {
    'categorical_feature': [0, 3, 7],  # Column indices
    # OR use names
    'categorical_feature': 'name:cat_col1,cat_col2',
    
    # Tuning
    'max_cat_to_onehot': 4,       # One-hot threshold
    'cat_l2': 10.0,               # Extra L2 for categorical splits
    'cat_smooth': 10.0,           # Smoothing in ratio calculation
    'max_cat_threshold': 32,      # Max categories per split
    'min_data_per_group': 100,    # Min data for category group
}
```

## Comparison with XGBoost

| Feature | LightGBM | XGBoost |
|---------|---------|---------|
| Native support | ✅ Yes | ❌ No (need encoding) |
| One-hot encoding | Not required | Required |
| Cardinality limit | No (sorted strategy) | High cardinality expensive |
| Split type | Category set | Threshold on encoded value |
| Memory | Efficient | Expands with one-hot |

### XGBoost Alternative

XGBoost typically requires:

```python
# XGBoost requires one-hot or label encoding
import pandas as pd
data = pd.get_dummies(data, columns=['cat_col1', 'cat_col2'])
# This explodes feature count!
```

## Advantages of LightGBM's Approach

1. **No feature explosion**: Categories stay as single feature
2. **Optimal splits**: Gradient-sorting finds near-optimal binary splits
3. **Memory efficient**: No one-hot expansion
4. **Handles high cardinality**: O(k log k) scaling

## Disadvantages

1. **Sorting overhead**: O(k log k) per split evaluation
2. **Limited multi-category splits**: Only binary splits in sorted order
3. **May miss complex interactions**: True optimal might need arbitrary subsets

## Mathematical Justification

For squared loss, Fisher's natural split theorem says:

> The optimal binary split of k categories can be found by sorting categories
> by their mean target value and choosing a split point in that order.

LightGBM extends this to gradient boosting by using `gradient/hessian` as
the proxy for mean target value.

## Source References

| Component | Source File |
|-----------|-------------|
| Categorical split finding | `src/treelearner/feature_histogram.cpp` |
| Bin mapper (categorical) | `src/io/bin.cpp` |
| Configuration | `include/LightGBM/config.h` |
