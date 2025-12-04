# LightGBM Histogram Building

## Overview

LightGBM's histogram building is similar to XGBoost's `hist` method, with some
key optimizations and different implementation choices.

## Histogram Structure

Each histogram bin stores:

```cpp
// 64-bit entry per bin (standard)
struct HistEntry {
  double sum_gradients;   // 32-bit float in practice
  double sum_hessians;    // 32-bit float
};

// With quantized gradients: 16-bit or 32-bit packed
// grad (16 bits) | hess (16 bits) = 32-bit entry
// grad (32 bits) | hess (32 bits) = 64-bit entry
```

## Building Strategies

LightGBM chooses between column-wise and row-wise based on data characteristics:

### Column-wise (Feature-major)

```cpp
// For each feature, iterate over rows in leaf
for (int feature = 0; feature < num_features; ++feature) {
  for (data_size_t i = 0; i < num_data_in_leaf; ++i) {
    int row = data_indices[i];
    int bin = get_bin(feature, row);
    histogram[feature][bin].grad += gradients[row];
    histogram[feature][bin].hess += hessians[row];
  }
}
```

**Better when**: Few features, small histograms fit in cache

### Row-wise (Sample-major)

```cpp
// For each row, iterate over features
for (data_size_t i = 0; i < num_data_in_leaf; ++i) {
  int row = data_indices[i];
  double grad = gradients[row];
  double hess = hessians[row];
  
  for (int feature = 0; feature < num_features; ++feature) {
    int bin = get_bin(feature, row);
    histogram[feature][bin].grad += grad;
    histogram[feature][bin].hess += hess;
  }
}
```

**Better when**: Many features, want to prefetch gradients

### Selection Heuristic

```cpp
// From GetShareStates
bool is_col_wise = (force_col_wise ||
    (!force_row_wise && 
     num_features * sizeof(HistEntry) * max_bins < L2_CACHE_SIZE));
```

## Histogram Subtraction

Same optimization as XGBoost - derive larger child from parent:

```cpp
void FindBestSplitsFromHistograms(..., bool use_subtract, ...) {
  // Build histogram for smaller leaf (always)
  train_data_->ConstructHistograms(
      is_feature_used,
      smaller_leaf_splits_->data_indices(),
      smaller_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_,
      smaller_leaf_histogram_array_
  );
  
  // For larger leaf: use subtraction if parent available
  if (use_subtract) {
    // Subtraction happens during split finding
    larger_hist[bin] = parent_hist[bin] - smaller_hist[bin];
  } else {
    // Build from scratch
    train_data_->ConstructHistograms(..., larger_leaf_histogram_array_);
  }
}
```

### Subtraction Implementation

```cpp
template <bool USE_DIST_GRAD, ...>
void Subtract(const FeatureHistogram& other, ...) {
  if (USE_DIST_GRAD) {
    // Quantized gradient subtraction
    for (int i = 0; i < num_bin; ++i) {
      result_data[i] = this_data[i] - other_data[i];
    }
  } else {
    // Standard float subtraction
    for (int i = 0; i < (num_bin) * 2; ++i) {
      data_[i] -= other.data_[i];
    }
  }
}
```

## Gradient Discretization

LightGBM supports quantized gradients for faster histogram building:

```cpp
class GradientDiscretizer {
  void DiscretizeGradients(data_size_t num_data,
                           const score_t* gradients,
                           const score_t* hessians) {
    // Find min/max
    double max_grad = ..., min_grad = ...;
    double max_hess = ..., min_hess = ...;
    
    // Compute scale factors
    grad_scale_ = (max_grad - min_grad) / (num_bins - 1);
    hess_scale_ = max_hess / (num_bins - 1);
    
    // Quantize
    for (data_size_t i = 0; i < num_data; ++i) {
      int grad_bin = (gradients[i] - min_grad) / grad_scale_;
      int hess_bin = hessians[i] / hess_scale_;
      
      // Pack into single integer
      discretized_[i] = (grad_bin << 16) | hess_bin;
    }
  }
};
```

### Bit Widths

LightGBM adaptively chooses bit width:

```cpp
// Based on leaf size and gradient range
uint8_t GetHistBitsInLeaf(int leaf_index) {
  // Smaller leaves can use fewer bits
  if (leaf_count < threshold_16bit) {
    return 16;  // 8-bit grad + 8-bit hess packed
  } else {
    return 32;  // 16-bit each
  }
}
```

## Histogram Pool

LightGBM caches histograms for reuse:

```cpp
class HistogramPool {
  // Pool of allocated histograms
  std::vector<std::unique_ptr<FeatureHistogram[]>> pool_;
  
  // Mapping: leaf_id → pool_slot
  std::vector<int> mapper_;
  
  // LRU tracking
  std::vector<int> last_used_time_;
  
  // Get histogram for leaf, possibly evicting old one
  bool Get(int idx, FeatureHistogram** out) {
    if (is_enough_) {
      *out = pool_[idx].get();
      return true;
    } else if (mapper_[idx] >= 0) {
      // Already in cache
      int slot = mapper_[idx];
      *out = pool_[slot].get();
      last_used_time_[slot] = ++cur_time_;
      return true;
    } else {
      // Evict LRU and allocate
      int slot = ArrayArgs<int>::ArgMin(last_used_time_);
      *out = pool_[slot].get();
      // Update mappings...
      return false;  // Indicates data needs to be computed
    }
  }
};
```

## Missing Value Handling

LightGBM tracks missing values specially:

```cpp
enum class MissingType {
  None,   // No missing values
  Zero,   // Missing encoded as 0 (skip default bin)
  NaN     // Explicit NaN handling
};

void FindBestThreshold(...) {
  if (meta_->missing_type == MissingType::Zero) {
    // Skip default bin during scan
    SKIP_DEFAULT_BIN = true;
  } else if (meta_->missing_type == MissingType::NaN) {
    // Include NaN bin at end
    NA_AS_MISSING = true;
  }
}
```

## Parallel Histogram Building

LightGBM parallelizes across features:

```cpp
void FindBestSplits(const Tree* tree, ...) {
  std::vector<int8_t> is_feature_used(num_features_, 0);
  
  #pragma omp parallel for schedule(static, 256) if (num_features_ >= 512)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!col_sampler_.is_feature_used_bytree()[feature_index]) continue;
    is_feature_used[feature_index] = 1;
  }
  
  // Parallel histogram construction
  ConstructHistograms(is_feature_used, use_subtract);
  
  // Parallel split finding
  FindBestSplitsFromHistograms(is_feature_used, use_subtract, tree);
}
```

## Comparison with XGBoost

| Aspect | LightGBM | XGBoost |
|--------|---------|---------|
| Histogram format | float64 or packed int | float64 |
| Subtraction | ✅ Yes | ✅ Yes |
| Gradient quantization | ✅ 8/16/32-bit | ✅ (GPU only) |
| Build strategy | Adaptive col/row | Row-wise default |
| Parallelism | Feature-level | Row-level |
| Caching | LRU pool | Per-node arrays |

## Key Differences from XGBoost

1. **Adaptive strategy**: LightGBM chooses col/row based on cache analysis
2. **Quantization on CPU**: LightGBM supports gradient quantization on CPU
3. **Pool management**: LRU caching vs. per-node allocation

## Performance Considerations

### Memory Layout

```cpp
// Contiguous histogram for all features
std::vector<hist_t> data_;  // size = total_bins * 2

// Feature i starts at offset[i]
pool_[i][j].Init(data_.data() + offsets[j] * 2, &feature_metas_[j]);
```

### Cache Optimization

- Column-wise: Features accessed sequentially, better prefetching
- Row-wise: Gradient/hessian reused across features, better locality

## Source References

| Component | Source File |
|-----------|-------------|
| Histogram class | `src/treelearner/feature_histogram.hpp` |
| Histogram pool | `src/treelearner/feature_histogram.hpp` (HistogramPool) |
| Construction | `src/treelearner/serial_tree_learner.cpp` |
| Share states | `include/LightGBM/train_share_states.h` |
| Gradient discretizer | `src/treelearner/gradient_discretizer.hpp` |
