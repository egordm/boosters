# HistogramCuts: Bin Boundaries for Quantization

## Overview

`HistogramCuts` stores the bin boundaries (cut points) computed during quantization.
It maps continuous feature values to discrete bin indices.

## Structure

```cpp
class HistogramCuts {
  // Cut points for each feature, concatenated
  // cut_ptrs_[f] to cut_ptrs_[f+1] gives cuts for feature f
  std::vector<float> cut_values_;
  
  // Pointers into cut_values_ for each feature
  // cut_ptrs_[f] = start index of feature f's cuts
  std::vector<uint32_t> cut_ptrs_;
  
  // Minimum value per feature (for values below first cut)
  std::vector<float> min_vals_;
};
```

## Example

For 3 features with [2, 3, 2] bins:

```text
Feature 0: bins at [0.5, 1.5]       → values in [−∞, 0.5), [0.5, 1.5), [1.5, +∞)
Feature 1: bins at [10, 20, 30]     → values in [−∞, 10), [10, 20), [20, 30), [30, +∞)
Feature 2: bins at [0.0, 0.5]       → values in [−∞, 0.0), [0.0, 0.5), [0.5, +∞)

cut_values_ = [0.5, 1.5, 10, 20, 30, 0.0, 0.5]
cut_ptrs_   = [0, 2, 5, 7]
min_vals_   = [0.0, 5.0, -1.0]  // Minimum observed values
```

## Operations

### SearchBin: Find Bin for Value

```cpp
uint32_t SearchBin(float value, uint32_t feature_idx) const {
  auto begin = cut_values_.begin() + cut_ptrs_[feature_idx];
  auto end = cut_values_.begin() + cut_ptrs_[feature_idx + 1];
  
  // Binary search for first cut >= value
  auto it = std::lower_bound(begin, end, value);
  
  // Return bin index (0-based within feature)
  return std::distance(begin, it);
}
```

### GetBinCount: Bins per Feature

```cpp
uint32_t GetBinCount(uint32_t feature_idx) const {
  return cut_ptrs_[feature_idx + 1] - cut_ptrs_[feature_idx] + 1;
  // +1 because n cut points define n+1 bins
}

uint32_t TotalBins() const {
  return cut_values_.size() + num_features_;
  // Each feature has (n_cuts + 1) bins
}
```

### FeatureOffset: Global Bin Index

```cpp
// For histogram indexing, we need global bin index:
// global_bin = feature_offset[f] + local_bin
std::vector<uint32_t> feature_offset_;

// feature_offset[f] = sum of bins for features 0..f-1
// feature_offset[0] = 0
// feature_offset[1] = bins(feature 0)
// feature_offset[2] = bins(feature 0) + bins(feature 1)
// etc.
```

## Building HistogramCuts

### From Sorted Data (Exact)

```cpp
void BuildFromSorted(std::vector<float> const& sorted_values,
                     std::vector<size_t> const& feature_counts,
                     size_t max_bins) {
  for (size_t f = 0; f < num_features; f++) {
    size_t n = feature_counts[f];
    size_t step = n / max_bins;
    
    for (size_t i = step; i < n; i += step) {
      float cut = sorted_values[feature_start[f] + i];
      if (cut != cut_values_.back()) {  // Skip duplicates
        cut_values_.push_back(cut);
      }
    }
    cut_ptrs_.push_back(cut_values_.size());
  }
}
```

### From Quantile Sketch (Approximate)

```cpp
void BuildFromSketch(WQSummary<float> const& sketch,
                     size_t max_bins) {
  // Query sketch for quantiles at 1/max_bins intervals
  for (size_t i = 1; i < max_bins; i++) {
    float q = static_cast<float>(i) / max_bins;
    float cut = sketch.Query(q);
    cut_values_.push_back(cut);
  }
}
```

## booste-rs Design

```rust
/// Bin boundaries for all features
pub struct HistogramCuts {
    /// Concatenated cut points for all features
    cut_values: Vec<f32>,
    
    /// cut_ptrs[f]..cut_ptrs[f+1] = range in cut_values for feature f
    cut_ptrs: Vec<u32>,
    
    /// Minimum value observed per feature
    min_vals: Vec<f32>,
    
    /// Precomputed global bin offset per feature
    feature_offsets: Vec<u32>,
}

impl HistogramCuts {
    /// Find bin index for a value in a feature
    pub fn search_bin(&self, value: f32, feature: usize) -> u32 {
        let start = self.cut_ptrs[feature] as usize;
        let end = self.cut_ptrs[feature + 1] as usize;
        let cuts = &self.cut_values[start..end];
        
        // Binary search
        match cuts.binary_search_by(|c| c.partial_cmp(&value).unwrap()) {
            Ok(i) => i as u32 + 1,  // Exact match, goes to right bin
            Err(i) => i as u32,      // Insert position = bin index
        }
    }
    
    /// Global bin index (for histogram indexing)
    pub fn global_bin(&self, feature: usize, local_bin: u32) -> u32 {
        self.feature_offsets[feature] + local_bin
    }
    
    /// Number of bins for a feature
    pub fn num_bins(&self, feature: usize) -> u32 {
        self.cut_ptrs[feature + 1] - self.cut_ptrs[feature] + 1
    }
    
    /// Total bins across all features
    pub fn total_bins(&self) -> u32 {
        self.feature_offsets.last().copied().unwrap_or(0)
    }
}
```

## Missing Value Handling

Missing values (NaN) need special treatment:

1. **During quantization**: Track missing count per feature
2. **During binning**: Map NaN to a special bin (typically 0 or max)
3. **During split evaluation**: Missing values can go left or right

```rust
impl HistogramCuts {
    pub fn search_bin_with_missing(&self, value: f32, feature: usize) -> Option<u32> {
        if value.is_nan() {
            None  // Missing value, direction determined by split
        } else {
            Some(self.search_bin(value, feature))
        }
    }
}
```

## Source Code References

| Component | XGBoost Source |
|-----------|----------------|
| HistogramCuts | `include/xgboost/data.h` |
| Cut building | `src/common/hist_util.cc` |
| SearchBin | `src/common/hist_util.h` |
