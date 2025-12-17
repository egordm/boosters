# LightGBM Training Overview

## Introduction

LightGBM uses histogram-based gradient boosting similar to XGBoost's `hist` method, but with
several key differences in strategy and optimization.

## Training Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Create BinMappers: Greedy binning per feature                       │
│  2. Build Dataset: Store binned features (dense or sparse)              │
│  3. Initialize TreeLearner                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      FOR EACH BOOSTING ROUND                            │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Compute Gradients: g_i = ∂L/∂ŷ_i, h_i = ∂²L/∂ŷ_i²                  │
│  2. [Optional] GOSS Sampling:                                           │
│     - Keep top_rate samples with highest |g × h|                        │
│     - Randomly sample other_rate from rest                              │
│     - Multiply sampled gradients by (1-top_rate)/other_rate            │
│  3. Initialize root leaf with all (sampled) data                        │
│  4. Grow tree LEAF-WISE:                                                │
│     while (num_leaves < max_leaves):                                    │
│       a. Select leaf with highest gain potential                        │
│       b. Build histogram for selected leaf                              │
│       c. Find best split across all features                            │
│       d. If gain > 0:                                                   │
│          - Split leaf, partition data                                   │
│          - Add children to candidates                                   │
│       e. Else: stop                                                     │
│  5. Update predictions: ŷ += learning_rate × tree(x)                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. BinMapper (Quantization)

LightGBM uses greedy binning instead of quantile sketching:

```cpp
// From bin.cpp: GreedyFindBin
// Goal: Find bin boundaries that respect min_data_in_bin
std::vector<double> GreedyFindBin(
    const double* distinct_values,
    const int* counts,
    int num_distinct_values,
    int max_bin,
    size_t total_cnt,
    int min_data_in_bin
);
```

**Key difference from XGBoost**: LightGBM uses a greedy algorithm that considers
data distribution directly, rather than approximate quantile sketching.

See: [bin_mapper.md](../data_structures/bin_mapper.md)

### 2. Dataset Storage

LightGBM supports multiple bin storage formats:

- **DenseBin**: Contiguous array of bin indices
- **SparseBin**: CSR-like format for sparse features
- **MultiValBin**: For features grouped together (EFB)

```cpp
// Dense storage: simple array
template <typename VAL_T>
class DenseBin {
  std::vector<VAL_T> data_;  // data_[row] = bin_idx
};

// Sparse storage: CSR format
template <typename VAL_T>
class SparseBin {
  std::vector<VAL_T> vals_;  // bin values
  std::vector<data_size_t> deltas_;  // row deltas (compressed)
};
```

### 3. SerialTreeLearner

The main tree learning class in LightGBM:

```cpp
class SerialTreeLearner {
  const Config* config_;
  const Dataset* train_data_;
  
  // Histogram management
  HistogramPool histogram_pool_;
  FeatureHistogram* smaller_leaf_histogram_array_;
  FeatureHistogram* larger_leaf_histogram_array_;
  
  // Split tracking
  std::vector<SplitInfo> best_split_per_leaf_;
  
  // Data partition
  std::unique_ptr<DataPartition> data_partition_;
  
  // Leaf info
  std::unique_ptr<LeafSplits> smaller_leaf_splits_;
  std::unique_ptr<LeafSplits> larger_leaf_splits_;
};
```

### 4. Training Loop

From `SerialTreeLearner::Train`:

```cpp
Tree* Train(const score_t* gradients, const score_t* hessians) {
  // 1. Setup
  BeforeTrain();
  auto tree = new Tree(config_->num_leaves);
  
  // 2. Set root value
  tree->SetLeafOutput(0, CalculateLeafOutput(...));
  
  // 3. Leaf-wise growth
  for (int split = 0; split < config_->num_leaves - 1; ++split) {
    // Find splits for current leaves
    if (BeforeFindBestSplit(tree, left_leaf, right_leaf)) {
      FindBestSplits(tree);
    }
    
    // Get leaf with max gain
    int best_leaf = ArgMax(best_split_per_leaf_);
    
    // Check if we should stop
    if (best_split_per_leaf_[best_leaf].gain <= 0.0) {
      break;
    }
    
    // Split the best leaf
    Split(tree, best_leaf, &left_leaf, &right_leaf);
  }
  
  return tree;
}
```

### 5. Histogram Building

LightGBM builds histograms per-feature:

```cpp
void ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                         bool use_subtract) {
  // Build histogram for smaller leaf (always)
  train_data_->ConstructHistograms(
      is_feature_used,
      smaller_leaf_splits_->data_indices(),
      smaller_leaf_splits_->num_data_in_leaf(),
      gradients_, hessians_,
      smaller_leaf_histogram_array_
  );
  
  // For larger leaf: build or subtract
  if (!use_subtract) {
    train_data_->ConstructHistograms(..., larger_leaf_histogram_array_);
  }
  // Subtraction happens in FindBestSplitsFromHistograms
}
```

### 6. Split Finding

For each feature, scan bins to find best split:

```cpp
void FindBestSplitsFromHistograms(...) {
  #pragma omp parallel for
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    // Fix histogram (add missing bin)
    train_data_->FixHistogram(feature_index, sum_gradients, sum_hessians, hist);
    
    // Find best threshold for this feature
    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
        sum_gradient, sum_hessian, num_data,
        constraints, parent_output,
        &smaller_best[tid]
    );
    
    // Subtract for larger leaf
    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(
          smaller_leaf_histogram_array_[feature_index]
      );
    }
    
    // Find best for larger leaf
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(...);
  }
  
  // Reduce best splits across threads
  for (thread : threads) {
    best_split_per_leaf_[smaller].Update(smaller_best[thread]);
    best_split_per_leaf_[larger].Update(larger_best[thread]);
  }
}
```

## Key Optimizations

### 1. Leaf-wise Growth

Unlike XGBoost's default depth-wise growth, LightGBM grows leaf-wise:

- Always split the leaf with highest gain
- Can achieve lower loss with same number of leaves
- Risk of overfitting on small datasets (use `max_depth` to control)

### 2. Histogram Subtraction

Same optimization as XGBoost:

```text
larger_hist = parent_hist - smaller_hist
```

Build histogram for smaller leaf, derive larger by subtraction.

### 3. Column-wise vs Row-wise

LightGBM chooses between column-wise and row-wise histogram building:

```cpp
// Decision based on cache considerations
bool is_col_wise = /* based on data characteristics */;

if (is_col_wise) {
  // For each feature, iterate over rows
  // Better when #features is small, bins fit in cache
} else {
  // For each row, iterate over features
  // Better when #rows is small relative to #features
}
```

### 4. Gradient Quantization

LightGBM supports quantized gradients (int8/int16):

```cpp
class GradientDiscretizer {
  // Discretize gradients for faster histogram building
  void DiscretizeGradients(data_size_t num_data,
                           const score_t* gradients,
                           const score_t* hessians);
  
  // Recover original values
  double grad_scale_;
  double hess_scale_;
};
```

### 5. GOSS (Gradient-based One-Side Sampling)

Keep samples with large gradients, randomly sample from small:

```cpp
// From goss.hpp
void Bagging(int iter, ...) {
  // Find top_k samples with largest |g × h|
  ArrayArgs<score_t>::ArgMaxAtK(&gradients, 0, cnt, top_k - 1);
  
  // Randomly sample other_k from rest
  for (sample : rest) {
    if (rand() < prob) {
      // Include and multiply gradients
      gradients[sample] *= multiply;  // = (1-top_rate)/other_rate
    }
  }
}
```

## Threading Model

LightGBM parallelizes at multiple levels:

1. **Feature-level**: Different threads evaluate different features
2. **Data-level**: Histogram building can be parallelized
3. **Row partition**: Parallel data partitioning

```cpp
#pragma omp parallel for schedule(static) num_threads(num_threads_)
for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
  // Thread-local best split
  // Reduction at the end
}
```

## Complexity Analysis

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Binning (one-time) | O(n × d × log(max_bin)) | Greedy binning |
| Histogram Build | O(n_leaf × d) | n_leaf = rows in leaf |
| Split Finding | O(d × bins) | Scan all bins per feature |
| Data Partition | O(n_leaf) | Assign each row once |
| Total per tree | O(n × d) | Dominated by histogram building |

## Source Code References

| Component | LightGBM Source |
|-----------|-----------------|
| Main booster | `src/boosting/gbdt.{h,cpp}` |
| Tree learner | `src/treelearner/serial_tree_learner.{h,cpp}` |
| Histogram | `src/treelearner/feature_histogram.hpp` |
| Data partition | `src/treelearner/data_partition.hpp` |
| Split info | `src/treelearner/split_info.hpp` |
| Leaf splits | `src/treelearner/leaf_splits.hpp` |
