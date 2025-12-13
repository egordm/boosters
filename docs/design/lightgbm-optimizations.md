# LightGBM Performance Optimizations Analysis

This document details the key performance optimizations found in LightGBM's codebase that make it one of the fastest gradient boosting implementations. These techniques are essential references for building a competitive Rust implementation.

## Table of Contents
1. [Histogram Building](#1-histogram-building)
2. [GOSS (Gradient-based One-Side Sampling)](#2-goss-gradient-based-one-side-sampling)
3. [EFB (Exclusive Feature Bundling)](#3-efb-exclusive-feature-bundling)
4. [Leaf-wise Growth](#4-leaf-wise-growth)
5. [Memory Layout](#5-memory-layout)
6. [Parallelization](#6-parallelization)
7. [Categorical Feature Handling](#7-categorical-feature-handling)
8. [Quantization](#8-quantization)

---

## 1. Histogram Building

**Key files:** `src/treelearner/feature_histogram.hpp`, `src/io/dense_bin.hpp`, `src/io/sparse_bin.hpp`

### 1.1 Core Histogram Structure

LightGBM uses a highly optimized histogram layout:

```cpp
// Histogram entry: gradient and hessian packed together
typedef double hist_t;  // or int32_t for quantized
const size_t kHistEntrySize = 2 * sizeof(hist_t);  // 16 bytes per bin

#define GET_GRAD(hist, i) hist[(i) << 1]
#define GET_HESS(hist, i) hist[((i) << 1) + 1]
```

**Key Optimizations:**

1. **Interleaved grad/hess storage**: Gradient and hessian for the same bin are adjacent in memory, maximizing cache utilization during split finding.

2. **Aligned memory allocation**: Uses `Common::AlignmentAllocator<hist_t, kAlignedSize>` (typically 32-byte alignment) for SIMD efficiency.

3. **Histogram subtraction trick**: Instead of building histograms for both children, builds only the smaller child's histogram and subtracts from parent:
   ```cpp
   void Subtract(const FeatureHistogram& other) {
     for (int i = 0; i < (meta_->num_bin - meta_->offset) * 2; ++i) {
       data_[i] -= other.data_[i];
     }
   }
   ```

### 1.2 Dense Bin Histogram Construction

From `dense_bin.hpp`:

```cpp
template <bool USE_INDICES, bool USE_PREFETCH, bool USE_HESSIAN>
void ConstructHistogramInner(...) {
  data_size_t i = start;
  hist_t* grad = out;
  hist_t* hess = out + 1;
  
  if (USE_PREFETCH) {
    const data_size_t pf_offset = 64 / sizeof(VAL_T);  // Prefetch distance
    const data_size_t pf_end = end - pf_offset;
    for (; i < pf_end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto pf_idx = USE_INDICES ? data_indices[i + pf_offset] : i + pf_offset;
      PREFETCH_T0(data_.data() + pf_idx);  // Software prefetching
      
      const auto ti = static_cast<uint32_t>(data(idx)) << 1;
      grad[ti] += ordered_gradients[i];
      hess[ti] += ordered_hessians[i];
    }
  }
}
```

**Key Optimizations:**
- **Software prefetching**: `PREFETCH_T0` hint loads data ahead of use
- **Conditional compilation via templates**: Eliminates runtime branching
- **4-bit packing support**: `IS_4BIT` template parameter enables storing 2 bin values per byte

### 1.3 Histogram Pool with LRU Caching

```cpp
class HistogramPool {
  // LRU cache for histogram reuse
  std::vector<int> mapper_;         // leaf_idx -> cache_slot
  std::vector<int> inverse_mapper_; // cache_slot -> leaf_idx
  std::vector<int> last_used_time_;
  
  bool Get(int idx, FeatureHistogram** out) {
    if (is_enough_) {
      *out = pool_[idx].get();
      return true;
    } else if (mapper_[idx] >= 0) {
      int slot = mapper_[idx];
      *out = pool_[slot].get();
      last_used_time_[slot] = ++cur_time_;
      return true;  // Cache hit
    } else {
      // LRU eviction
      int slot = ArrayArgs<int>::ArgMin(last_used_time_);
      // ... evict and reuse
      return false;  // Cache miss - needs rebuild
    }
  }
};
```

---

## 2. GOSS (Gradient-based One-Side Sampling)

**Key file:** `src/boosting/goss.hpp`

GOSS reduces the number of data instances used for histogram building while maintaining accuracy by keeping all large-gradient instances and randomly sampling small-gradient instances.

### 2.1 Core Algorithm

```cpp
data_size_t Helper(data_size_t start, data_size_t cnt, data_size_t* buffer,
                   score_t* gradients, score_t* hessians) {
  // Step 1: Compute |gradient * hessian| for each instance
  std::vector<score_t> tmp_gradients(cnt, 0.0f);
  for (data_size_t i = 0; i < cnt; ++i) {
    for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
      size_t idx = static_cast<size_t>(cur_tree_id) * num_data_ + start + i;
      tmp_gradients[i] += std::fabs(gradients[idx] * hessians[idx]);
    }
  }
  
  // Step 2: Find top-k by gradient magnitude
  data_size_t top_k = static_cast<data_size_t>(cnt * config_->top_rate);
  data_size_t other_k = static_cast<data_size_t>(cnt * config_->other_rate);
  ArrayArgs<score_t>::ArgMaxAtK(&tmp_gradients, 0, tmp_gradients.size(), top_k - 1);
  score_t threshold = tmp_gradients[top_k - 1];
  
  // Step 3: Weight amplification for sampled small-gradient instances
  score_t multiply = static_cast<score_t>(cnt - top_k) / other_k;
  
  // Step 4: Select instances
  for (data_size_t i = 0; i < cnt; ++i) {
    if (grad >= threshold) {
      buffer[cur_left_cnt++] = cur_idx;  // Keep all large gradients
    } else {
      // Reservoir sampling for small gradients
      double prob = (rest_need) / static_cast<double>(rest_all);
      if (random.NextFloat() < prob) {
        buffer[cur_left_cnt++] = cur_idx;
        // Amplify gradients/hessians to correct for sampling bias
        gradients[idx] *= multiply;
        hessians[idx] *= multiply;
      }
    }
  }
}
```

**Key Parameters:**
- `top_rate`: Fraction of large-gradient instances to keep (default: 0.2)
- `other_rate`: Fraction of small-gradient instances to sample (default: 0.1)
- Only activates after `1/learning_rate` iterations (warm-up period)

---

## 3. EFB (Exclusive Feature Bundling)

**Key files:** `include/LightGBM/feature_group.h`, `src/io/multi_val_dense_bin.hpp`

EFB bundles mutually exclusive sparse features together, reducing the effective number of features and improving cache efficiency.

### 3.1 Feature Group Structure

```cpp
class FeatureGroup {
  // Single bin for all features in group (when not multi-val)
  std::unique_ptr<Bin> bin_data_;
  
  // Separate bins per feature (when multi-val / EFB)
  std::vector<std::unique_ptr<Bin>> multi_bin_data_;
  
  // Bin offsets for features within the group
  std::vector<uint32_t> bin_offsets_;  // [0, num_bins_f0, num_bins_f0+f1, ...]
  
  bool is_multi_val_;       // True for EFB bundled features
  bool is_dense_multi_val_; // Dense vs sparse multi-val storage
};
```

### 3.2 Multi-Val Dense Bin (Row-wise Feature Bundle)

From `multi_val_dense_bin.hpp`:

```cpp
template <typename VAL_T>
class MultiValDenseBin : public MultiValBin {
  // Row-major storage: all bundled feature values for a row are contiguous
  std::vector<VAL_T> data_;  // Size: num_data * num_feature_in_bundle
  
  void ConstructHistogramInner(...) {
    for (data_size_t i = start; i < end; ++i) {
      const auto idx = USE_INDICES ? data_indices[i] : i;
      const auto j_start = RowPtr(idx);  // idx * num_feature_
      const VAL_T* data_ptr = data_.data() + j_start;
      
      // Process all features in bundle for this row
      for (int j = 0; j < num_feature_; ++j) {
        const uint32_t bin = static_cast<uint32_t>(data_ptr[j]);
        const auto ti = (bin + offsets_[j]) << 1;
        grad[ti] += gradient;
        hess[ti] += hessian;
      }
    }
  }
};
```

**Key insight:** By storing bundled features row-wise, a single pass through the data builds histograms for all bundled features, dramatically improving memory bandwidth utilization.

### 3.3 Sparsity Detection and Bundling Decision

```cpp
const double kSparseThreshold = 0.7;  // 70% zeros triggers sparse storage

// In FeatureGroup constructor:
double sum_sparse_rate = 0.0f;
for (int i = 0; i < num_feature_; ++i) {
  sum_sparse_rate += bin_mappers_[i]->sparse_rate();
}
sum_sparse_rate /= num_feature_;

// Use dense multi-val bin if features are not too sparse
if (sum_sparse_rate < MultiValBin::multi_val_bin_sparse_threshold) {
  is_dense_multi_val_ = true;  // threshold = 0.25
}
```

---

## 4. Leaf-wise Growth

**Key file:** `src/treelearner/serial_tree_learner.cpp`

### 4.1 Best-First (Leaf-wise) Split Selection

```cpp
Tree* SerialTreeLearner::Train(...) {
  // Initialize root
  BeforeTrain();
  int left_leaf = 0;
  
  for (int split = 0; split < config_->num_leaves - 1; ++split) {
    // Find best splits for current leaves
    if (BeforeFindBestSplit(tree_ptr, left_leaf, right_leaf)) {
      FindBestSplits(tree_ptr);
    }
    
    // Select leaf with MAXIMUM gain (best-first / leaf-wise)
    int best_leaf = ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_);
    const SplitInfo& best_split = best_split_per_leaf_[best_leaf];
    
    if (best_split.gain <= 0.0) break;  // No beneficial split
    
    // Split the best leaf
    Split(tree_ptr, best_leaf, &left_leaf, &right_leaf);
  }
}
```

### 4.2 Smaller/Larger Leaf Optimization

```cpp
bool SerialTreeLearner::BeforeFindBestSplit(...) {
  // Determine which child is smaller
  if (num_data_in_left_child < num_data_in_right_child) {
    // Build histogram only for smaller leaf
    histogram_pool_.Get(left_leaf, &smaller_leaf_histogram_array_);
    // Reuse parent histogram for larger leaf (will subtract)
    parent_leaf_histogram_array_ = larger_leaf_histogram_array_;
    histogram_pool_.Move(left_leaf, right_leaf);
  }
  // ... symmetric case
}
```

This ensures histogram building complexity is O(min(left_count, right_count)) instead of O(total_count).

---

## 5. Memory Layout

### 5.1 Data Partition (Contiguous Index Storage)

**Key file:** `src/treelearner/data_partition.hpp`

```cpp
class DataPartition {
  // Indices stored contiguously per leaf: [leaf0_indices..., leaf1_indices..., ...]
  std::vector<data_size_t> indices_;
  std::vector<data_size_t> leaf_begin_;  // Start position for each leaf
  std::vector<data_size_t> leaf_count_;  // Count for each leaf
  
  void Split(int leaf, ..., int right_leaf) {
    // Partition in-place using parallel runner
    const data_size_t begin = leaf_begin_[leaf];
    auto left_start = indices_.data() + begin;
    
    const auto left_cnt = runner_.Run<false>(
      cnt,
      [=](..., data_size_t* left, data_size_t* right) {
        return dataset->Split(feature, threshold, ..., left, right);
      },
      left_start);
    
    // Update metadata
    leaf_count_[leaf] = left_cnt;
    leaf_begin_[right_leaf] = left_cnt + begin;
    leaf_count_[right_leaf] = cnt - left_cnt;
  }
};
```

### 5.2 Ordered Gradients for Cache Efficiency

```cpp
// Gradients are reordered to match data_indices order
std::vector<score_t> ordered_gradients_;
std::vector<score_t> ordered_hessians_;

// During histogram building, access is sequential:
// ordered_gradients[i] corresponds to data_indices[i]
// This is cache-optimal vs gradients[data_indices[i]] which has random access
```

### 5.3 Bin Storage Types

```cpp
// Dense bins: Direct array lookup
template <typename VAL_T, bool IS_4BIT>
class DenseBin {
  std::vector<VAL_T> data_;  // VAL_T = uint8_t, uint16_t, or uint32_t
  
  // 4-bit packing for bins <= 15
  inline VAL_T data(data_size_t idx) const {
    if (IS_4BIT) {
      return (data_[idx >> 1] >> ((idx & 1) << 2)) & 0xf;
    } else {
      return data_[idx];
    }
  }
};

// Sparse bins: Delta-encoded positions
template <typename VAL_T>
class SparseBin {
  std::vector<uint8_t> deltas_;  // Delta from previous non-zero
  std::vector<VAL_T> vals_;      // Bin values
  std::vector<std::pair<data_size_t, data_size_t>> fast_index_;  // Skip index
};
```

---

## 6. Parallelization

### 6.1 Feature-Parallel Histogram Building

```cpp
void SerialTreeLearner::FindBestSplitsFromHistograms(...) {
  std::vector<SplitInfo> smaller_best(share_state_->num_threads);
  std::vector<SplitInfo> larger_best(share_state_->num_threads);
  
  #pragma omp parallel for schedule(static) num_threads(num_threads)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    const int tid = omp_get_thread_num();
    
    // Fix histogram (handle most frequent bin)
    train_data_->FixHistogram(feature_index, sum_gradients, sum_hessians,
                              histogram_array[feature_index].RawData());
    
    // Compute histogram subtraction (for larger leaf)
    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(
          smaller_leaf_histogram_array_[feature_index]);
    }
    
    // Find best split for this feature
    ComputeBestSplitForFeature(histogram_array, feature_index, ...,
                               &smaller_best[tid], ...);
  }
  
  // Reduce across threads
  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_best);
  best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx];
}
```

### 6.2 Block-Parallel Histogram Construction

From `train_share_states.h`:

```cpp
template <bool USE_INDICES, bool ORDERED, bool USE_QUANT_GRAD, int HIST_BITS>
void ConstructHistograms(...) {
  // Divide data into blocks
  Threading::BlockInfo<data_size_t>(num_threads_, num_data, min_block_size_,
                                    &n_data_block_, &data_block_size_);
  
  #pragma omp parallel for schedule(static) num_threads(num_threads_)
  for (int block_id = 0; block_id < n_data_block_; ++block_id) {
    data_size_t start = block_id * data_block_size_;
    data_size_t end = std::min(start + data_block_size_, num_data);
    
    // Each thread builds partial histogram
    ConstructHistogramsForBlock<...>(start, end, ..., block_id, hist_buf);
  }
  
  // Merge partial histograms
  HistMerge<...>(hist_buf);
}
```

### 6.3 Parallel Partition Runner

```cpp
template <typename INDEX_T, bool TWO_BUFFER>
class ParallelPartitionRunner {
  // Used for parallel data splitting during tree growth
  template <bool FORCE_ORDERED>
  INDEX_T Run(INDEX_T num_data,
              const std::function<INDEX_T(int, INDEX_T, INDEX_T, INDEX_T*, INDEX_T*)>& func,
              INDEX_T* buffer) {
    // Parallel prefix sum approach for load-balanced partitioning
  }
};
```

---

## 7. Categorical Feature Handling

**Key file:** `src/treelearner/feature_histogram.cpp`

### 7.1 One-hot vs Many-vs-Many Split

```cpp
void FindBestThresholdCategoricalInner(...) {
  bool use_onehot = meta_->num_bin <= meta_->config->max_cat_to_onehot;
  
  if (use_onehot) {
    // O(k) algorithm: try each category as a split
    for (int t = bin_start; t < bin_end; ++t) {
      // Split: category t vs all others
      double current_gain = GetSplitGains(sum_other_gradient, sum_other_hessian,
                                          grad, hess, ...);
    }
  } else {
    // Many-vs-many: sort by gradient statistics
    auto ctr_fun = [](double sum_grad, double sum_hess) {
      return sum_grad / (sum_hess + cat_smooth);
    };
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                     [&](int i, int j) { return ctr_fun(i) < ctr_fun(j); });
    
    // Try cumulative splits from both directions
    for (size_t dir : {1, -1}) {
      double sum_left_gradient = 0;
      for (int i = 0; i < max_num_cat; ++i) {
        // Accumulate categories in sorted order
        sum_left_gradient += grad[sorted_idx[i]];
        // ... evaluate split
      }
    }
  }
}
```

### 7.2 Categorical Split Representation

```cpp
struct SplitInfo {
  int num_cat_threshold;           // Number of categories on left
  std::vector<uint32_t> cat_threshold;  // Category indices on left
};

// During data split: use bitset for O(1) membership test
std::vector<uint32_t> cat_bitset = Common::ConstructBitset(threshold, num_threshold);
// Check: Common::FindInBitset(cat_bitset.data(), num_threshold, bin)
```

---

## 8. Quantization

**Key file:** `src/treelearner/gradient_discretizer.hpp`

### 8.1 Gradient Discretization

```cpp
class GradientDiscretizer {
  int num_grad_quant_bins_;  // Typically 16-256 bins
  
  std::vector<int8_t> discretized_gradients_and_hessians_vector_;
  double gradient_scale_;   // Scale factor: max_gradient / num_bins
  double hessian_scale_;
  
  void DiscretizeGradients(const score_t* gradients, const score_t* hessians) {
    // Find max absolute values
    max_gradient_abs_ = max(|gradients|);
    max_hessian_abs_ = max(|hessians|);
    
    // Compute scale factors
    gradient_scale_ = max_gradient_abs_ / (num_grad_quant_bins_ / 2 - 1);
    hessian_scale_ = max_hessian_abs_ / (num_grad_quant_bins_ - 1);
    
    // Quantize with optional stochastic rounding
    for (data_size_t i = 0; i < num_data; ++i) {
      int8_t grad_bin = round(gradients[i] / gradient_scale_);
      int8_t hess_bin = round(hessians[i] / hessian_scale_);
      // Pack into int16: [grad_8bit][hess_8bit]
    }
  }
};
```

### 8.2 Packed Histogram Types

```cpp
// 16-bit histogram: 8-bit grad + 8-bit hess packed into int16
// 32-bit histogram: 16-bit grad + 16-bit hess packed into int32
// 64-bit histogram: 32-bit grad + 32-bit hess packed into int64

template <typename PACKED_HIST_T, int HIST_BITS>
void ConstructHistogramIntInner(...) {
  PACKED_HIST_T* out_ptr = reinterpret_cast<PACKED_HIST_T*>(out);
  const int16_t* gradients_ptr = reinterpret_cast<const int16_t*>(ordered_gradients);
  
  for (data_size_t i = start; i < end; ++i) {
    const auto ti = static_cast<uint32_t>(data(idx));
    const int16_t gradient_16 = gradients_ptr[i];
    // Unpack and accumulate
    const PACKED_HIST_T gradient_packed = 
      (static_cast<PACKED_HIST_T>(gradient_16 >> 8) << HIST_BITS) | (gradient_16 & 0xff);
    out_ptr[ti] += gradient_packed;
  }
}
```

### 8.3 Adaptive Histogram Bit Width

```cpp
template <bool IS_GLOBAL>
void SetNumBitsInHistogramBin(int left_leaf, int right_leaf,
                              data_size_t left_count, data_size_t right_count) {
  // Choose histogram precision based on leaf size
  // Smaller leaves can use fewer bits without overflow
  if (left_count * num_grad_quant_bins_ < 256) {
    leaf_num_bits_[left_leaf] = 8;
  } else if (left_count * num_grad_quant_bins_ < 65536) {
    leaf_num_bits_[left_leaf] = 16;
  } else {
    leaf_num_bits_[left_leaf] = 32;
  }
}
```

---

## Summary: Key Takeaways for Rust Implementation

### Must-Have Optimizations

1. **Histogram subtraction**: Build histogram only for smaller child, subtract from parent for larger
2. **Ordered gradients**: Pre-reorder gradients to match data indices for sequential memory access
3. **Feature-parallel split finding**: Each thread evaluates different features
4. **Block-parallel histogram building**: Divide data into blocks, merge partial histograms
5. **Aligned memory**: 32-byte alignment for SIMD operations

### High-Impact Optimizations

1. **4-bit bin packing** for features with ≤15 bins
2. **LRU histogram cache** to avoid redundant computation
3. **Software prefetching** in histogram building loops
4. **Compile-time template specialization** to eliminate runtime branches

### Algorithm-Level Optimizations

1. **GOSS sampling** for large datasets (reduce data size by ~70%)
2. **EFB bundling** for sparse features (reduce feature count)
3. **Many-vs-many categorical splits** with gradient-sorted ordering
4. **Gradient quantization** for reduced memory bandwidth (8/16-bit vs 64-bit)

### Memory Layout Principles

1. **Interleaved grad/hess** in histograms (cache line utilization)
2. **Contiguous indices per leaf** in data partition
3. **Row-wise feature bundles** for EFB (single pass over data)
4. **Delta encoding** for sparse bins (memory efficiency)
