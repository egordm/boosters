# LightGBM Inference

> Source: `include/LightGBM/tree.h`, `src/io/tree.cpp`, `src/boosting/gbdt_prediction.cpp`

## Overview

LightGBM's inference pipeline is designed for flexibility (single-row, batch, sparse) while maintaining performance through:

1. **Dual prediction paths**: Binned (fast, training) vs raw values (flexible, inference)
2. **Linear tree support**: Leaf-level linear models
3. **Multiple output modes**: Raw scores, transformed, leaf indices, SHAP values
4. **Parallel batch prediction**: Threading for large datasets

## Tree Traversal

### Standard Tree (Constant Leaves)

```cpp
// From tree.h - GetLeaf()
inline int Tree::GetLeaf(const double* feature_values) const {
  int node = 0;
  if (num_cat_ > 0) {
    while (node >= 0) {
      node = Decision(feature_values[split_feature_[node]], node);
    }
  } else {
    // Fast path: numerical only
    while (node >= 0) {
      node = NumericalDecision(feature_values[split_feature_[node]], node);
    }
  }
  return ~node;  // Leaf indices are stored as ~leaf_idx
}
```

**Key insight**: LightGBM uses **negative node indices** to represent leaves. When `node < 0`, `~node` gives the leaf index.

### Numerical Decision

```cpp
inline int NumericalDecision(double fval, int node) const {
  uint8_t missing_type = GetMissingType(decision_type_[node]);
  
  // Handle NaN → 0 conversion if not NaN-aware
  if (std::isnan(fval) && missing_type != MissingType::NaN) {
    fval = 0.0f;
  }
  
  // Handle missing values (Zero or NaN)
  if ((missing_type == MissingType::Zero && IsZero(fval))
      || (missing_type == MissingType::NaN && std::isnan(fval))) {
    if (GetDecisionType(decision_type_[node], kDefaultLeftMask)) {
      return left_child_[node];
    } else {
      return right_child_[node];
    }
  }
  
  // Standard comparison
  if (fval <= threshold_[node]) {
    return left_child_[node];
  } else {
    return right_child_[node];
  }
}
```

### Categorical Decision

```cpp
inline int CategoricalDecision(double fval, int node) const {
  // NaN → right child
  if (std::isnan(fval)) {
    return right_child_[node];
  }
  
  int int_fval = static_cast<int>(fval);
  // Negative → right child
  if (int_fval < 0) {
    return right_child_[node];
  }
  
  // Check if category is in the left set (stored as bitset)
  int cat_idx = static_cast<int>(threshold_[node]);
  if (Common::FindInBitset(cat_threshold_.data() + cat_boundaries_[cat_idx],
                           cat_boundaries_[cat_idx + 1] - cat_boundaries_[cat_idx], 
                           int_fval)) {
    return left_child_[node];
  }
  return right_child_[node];
}
```

## Linear Tree Prediction

When `is_linear_` is true, leaves contain linear models:

```cpp
inline double Tree::Predict(const double* feature_values) const {
  if (is_linear_) {
    int leaf = (num_leaves_ > 1) ? GetLeaf(feature_values) : 0;
    
    // Start with intercept (constant term)
    double output = leaf_const_[leaf];
    
    // Add linear terms
    bool nan_found = false;
    for (size_t i = 0; i < leaf_features_[leaf].size(); ++i) {
      int feat_raw = leaf_features_[leaf][i];
      double feat_val = feature_values[feat_raw];
      
      if (std::isnan(feat_val)) {
        nan_found = true;
        break;
      } else {
        output += leaf_coeff_[leaf][i] * feat_val;
      }
    }
    
    // Fallback to constant if NaN encountered
    if (nan_found) {
      return LeafOutput(leaf);
    }
    return output;
  } else {
    // Standard constant leaf
    if (num_leaves_ > 1) {
      int leaf = GetLeaf(feature_values);
      return LeafOutput(leaf);
    } else {
      return leaf_value_[0];
    }
  }
}
```

## Ensemble Prediction

### Single Row

```cpp
// From gbdt_prediction.cpp
void GBDT::PredictRaw(const double* features, double* output, 
                      const PredictionEarlyStopInstance* early_stop) const {
  int early_stop_round_counter = 0;
  
  // Initialize output to zero
  std::memset(output, 0, sizeof(double) * num_tree_per_iteration_);
  
  const int end_iteration = start_iteration_for_pred_ + num_iteration_for_pred_;
  for (int i = start_iteration_for_pred_; i < end_iteration; ++i) {
    // Predict all trees for one iteration (handles multi-class)
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] += models_[i * num_tree_per_iteration_ + k]->Predict(features);
    }
    
    // Check early stopping
    ++early_stop_round_counter;
    if (early_stop->round_period == early_stop_round_counter) {
      if (early_stop->callback_function(output, num_tree_per_iteration_)) {
        return;
      }
      early_stop_round_counter = 0;
    }
  }
}
```

### Batch Prediction (Dataset)

For batch prediction with Dataset (binned features), LightGBM uses **BinIterator** for efficient access:

```cpp
// From tree.cpp - AddPredictionToScore (simplified)
void Tree::AddPredictionToScore(const Dataset* data, data_size_t num_data, 
                                 double* score) const {
  // Get default and max bin values for each split node
  std::vector<uint32_t> default_bins(num_leaves_ - 1);
  std::vector<uint32_t> max_bins(num_leaves_ - 1);
  for (int i = 0; i < num_leaves_ - 1; ++i) {
    const int fidx = split_feature_inner_[i];
    auto bin_mapper = data->FeatureBinMapper(fidx);
    default_bins[i] = bin_mapper->GetDefaultBin();
    max_bins[i] = bin_mapper->num_bin() - 1;
  }
  
  // Process in parallel chunks
  Threading::For<data_size_t>(0, num_data, 512, [&](int, data_size_t start, data_size_t end) {
    // Create bin iterators
    std::vector<std::unique_ptr<BinIterator>> iter(num_leaves_ - 1);
    for (int i = 0; i < num_leaves_ - 1; ++i) {
      iter[i].reset(data->FeatureIterator(split_feature_inner_[i]));
      iter[i]->Reset(start);
    }
    
    // Traverse each sample
    for (data_size_t i = start; i < end; ++i) {
      int node = 0;
      while (node >= 0) {
        node = NumericalDecisionInner(iter[node]->Get(i), node,
                                      default_bins[node], max_bins[node]);
      }
      score[i] += leaf_value_[~node];
    }
  });
}
```

## Decision Type Encoding

LightGBM packs multiple flags into a single `int8_t`:

```cpp
// Bit layout of decision_type_
// Bit 0: kCategoricalMask (1 = categorical split)
// Bit 1: kDefaultLeftMask (1 = missing goes left)
// Bits 2-3: MissingType (0=None, 1=Zero, 2=NaN)

#define kCategoricalMask (1)
#define kDefaultLeftMask (2)

inline static bool GetDecisionType(int8_t decision_type, int8_t mask) {
  return (decision_type & mask) > 0;
}

inline static int8_t GetMissingType(int8_t decision_type) {
  return (decision_type >> 2) & 3;
}
```

## Output Transformation

After raw prediction, objective-specific transformation:

```cpp
void GBDT::Predict(const double* features, double* output, 
                   const PredictionEarlyStopInstance* early_stop) const {
  // Get raw scores
  PredictRaw(features, output, early_stop);
  
  // Average if needed (e.g., random forest mode)
  if (average_output_) {
    for (int k = 0; k < num_tree_per_iteration_; ++k) {
      output[k] /= num_iteration_for_pred_;
    }
  }
  
  // Apply objective transformation (sigmoid, softmax, exp, etc.)
  if (objective_function_ != nullptr) {
    objective_function_->ConvertOutput(output, output);
  }
}
```

## Comparison with XGBoost

| Aspect | LightGBM | XGBoost |
|--------|----------|---------|
| **Leaf encoding** | Negative index (`~leaf`) | Separate IsLeaf check |
| **Node structure** | Separate arrays (SoA) | Union Node struct (AoS) |
| **Missing handling** | 3 types (None, Zero, NaN) | default_left only |
| **Categorical** | Bitset in separate array | Bitset in node |
| **Linear trees** | Native support | Not supported |
| **Batch prediction** | BinIterator abstraction | FVec abstraction |
| **Threading** | Custom Threading::For | OpenMP parallel for |

### XGBoost Tree Node (AoS)

```cpp
// XGBoost Node (24 bytes, tightly packed)
class Node {
  int32_t parent_;   // Parent index + is_left_child flag
  int32_t cleft_;    // Left child
  int32_t cright_;   // Right child
  uint32_t sindex_;  // Split feature + default_left flag
  union Info {
    float leaf_value;
    float split_cond;
  } info_;
};
```

### LightGBM Tree (SoA)

```cpp
// LightGBM uses separate arrays
std::vector<int> left_child_;
std::vector<int> right_child_;
std::vector<int> split_feature_;
std::vector<double> threshold_;
std::vector<int8_t> decision_type_;
std::vector<double> leaf_value_;
// ... and more
```

## Performance Considerations

### Iterator Strategy Selection

LightGBM chooses between two iteration strategies:

```cpp
// From tree.cpp
if (data->num_features() > num_leaves_ - 1) {
  // More features than splits: iterate by node
  // Create iterator per split node
  PredictionFun(num_leaves_ - 1, split_feature_inner_[i], ...)
} else {
  // More splits than features: iterate by feature
  // Create iterator per feature
  PredictionFun(data->num_features(), i, ...)
}
```

### Early Stopping

For applications like ranking where early cutoff is possible:

```cpp
struct PredictionEarlyStopInstance {
  int round_period;
  std::function<bool(const double*, int)> callback_function;
};
```

### SHAP Values (TreeSHAP)

LightGBM implements polynomial-time TreeSHAP:

```cpp
void Tree::PredictContrib(const double* feature_values, int num_features, 
                          double* output) {
  output[num_features] += ExpectedValue();  // Base value
  if (num_leaves_ > 1) {
    const int max_path_len = max_depth_ + 1;
    std::vector<PathElement> unique_path_data(max_path_len*(max_path_len + 1) / 2);
    TreeSHAP(feature_values, output, 0, 0, unique_path_data.data(), 1, 1, -1);
  }
}
```

## booste-rs Implications

### Key Design Points

1. **Leaf encoding**: Consider using negative indices (simple, one comparison)
2. **Decision type packing**: Single byte for all flags is efficient
3. **Missing type enum**: More explicit than XGBoost's approach
4. **Linear tree support**: Plan for optional leaf coefficients

### Recommended Implementation

```rust
// Tree traversal with negative leaf encoding
fn get_leaf(&self, features: &[f32]) -> i32 {
    let mut node = 0i32;
    while node >= 0 {
        let split_feat = self.split_feature[node as usize];
        let fval = features[split_feat as usize];
        node = self.decision(fval, node as usize);
    }
    !node  // Bitwise NOT to get leaf index
}

// Decision with packed flags
fn decision(&self, fval: f32, node: usize) -> i32 {
    let decision_type = self.decision_type[node];
    let is_categorical = (decision_type & 0x1) != 0;
    let default_left = (decision_type & 0x2) != 0;
    let missing_type = (decision_type >> 2) & 0x3;
    
    // Handle missing
    if self.is_missing(fval, missing_type) {
        return if default_left { 
            self.left_child[node] 
        } else { 
            self.right_child[node] 
        };
    }
    
    // Standard split
    if is_categorical {
        self.categorical_decision(fval as i32, node)
    } else {
        if fval <= self.threshold[node] {
            self.left_child[node]
        } else {
            self.right_child[node]
        }
    }
}
```

### Linear Tree Support

```rust
struct LinearTreeLeaf {
    constant: f32,
    coefficients: Vec<f32>,
    feature_indices: Vec<u32>,
}

fn predict_linear_leaf(&self, features: &[f32], leaf: &LinearTreeLeaf) -> f32 {
    let mut output = leaf.constant;
    for (i, &feat_idx) in leaf.feature_indices.iter().enumerate() {
        let fval = features[feat_idx as usize];
        if fval.is_nan() {
            // Fallback to constant
            return self.leaf_value[leaf_idx];
        }
        output += leaf.coefficients[i] * fval;
    }
    output
}
```
