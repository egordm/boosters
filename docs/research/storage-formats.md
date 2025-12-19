# Storage Format Research

This document evaluates existing tree ensemble serialization formats and their suitability for booste-rs.

## Treelite Evaluation

### Overview

[Treelite](https://treelite.readthedocs.io/) is a universal tree ensemble format designed for model deployment. It provides a common representation that can import from XGBoost, LightGBM, and scikit-learn.

### Capabilities

**Supported Import Sources**:
- XGBoost (JSON, UBJSON, legacy binary)
- LightGBM (text format)
- scikit-learn (RandomForest, GradientBoosting, HistGradientBoosting, etc.)

**Supported Model Features**:
- Tree structure (numerical and categorical splits)
- Multiple output groups (multiclass classification)
- Post-processors (sigmoid, softmax, exponential_standard_ratio)
- Base scores
- Tree statistics (hessian sums, data counts, gain)
- Comparison operators (LT, LE, GT, GE, EQ)

**Treelite v4 Format Structure**:
```
Header:
  - major_ver, minor_ver, patch_ver
  - threshold_type (float32 or float64)
  - leaf_output_type (float32 or float64)
  - num_tree
  
Header 2:
  - num_feature
  - task_type: kBinaryClf, kRegressor, kMultiClf, kLearningToRank, kIsolationForest
  - num_target, num_class, leaf_vector_shape
  - target_id, class_id per tree
  - postprocessor, sigmoid_alpha, ratio_c, base_scores
  
Per Tree:
  - num_nodes, has_categorical_split
  - node_type, cleft, cright, split_index
  - default_left, leaf_value, threshold, cmp
  - category_list (for categorical splits)
  - leaf_vector (for multi-output leaves)
  - data_count, sum_hess, gain (optional statistics)
```

### Critical Limitations for booste-rs

After thorough evaluation, Treelite has fundamental limitations that prevent its use as our primary format:

#### 1. No GBLinear Support

Treelite is fundamentally a **tree-only** format. XGBoost's `gblinear` booster stores a weight matrix and bias vector—not trees. There is no mechanism in Treelite to represent:

```
GBLinear Model:
  weights: [num_features × num_groups]
  bias: [num_groups]
```

The `TaskType` enum only includes tree-based tasks:
- `kBinaryClf`, `kRegressor`, `kMultiClf`, `kLearningToRank`, `kIsolationForest`

No `kLinear` or similar exists.

#### 2. No Linear Leaves Support

LightGBM supports `linear_tree=True` which fits a linear model in each leaf:

```cpp
// From LightGBM tree.h
bool is_linear_;
std::vector<std::vector<double>> leaf_coeff_;   // coefficients per leaf
std::vector<double> leaf_const_;                 // intercept per leaf
std::vector<std::vector<int>> leaf_features_;   // features used per leaf
```

Prediction with linear leaves:
```cpp
double output = leaf_const_[leaf];
for (size_t i = 0; i < leaf_features_[leaf].size(); ++i) {
    output += leaf_coeff_[leaf][i] * feature_values[leaf_features_[leaf][i]];
}
```

Treelite's leaf representation is limited to:
- `leaf_value`: scalar float
- `leaf_vector`: array of floats (for multi-output)

There is no provision for linear coefficients, feature indices, or intercepts within leaves.

#### 3. DART Weights Not Preserved

While Treelite can store DART trees structurally, the per-tree dropout weights used during DART prediction are not part of the format. DART requires:

```rust
struct DartModel {
    trees: Vec<Tree>,
    tree_weights: Vec<f32>,  // Not in Treelite
}
```

### Treelite as Export Target

Despite limitations for our primary format, Treelite could serve as an **export-only** target for standard GBDT models:

**Benefits**:
- Interoperability with TL2cgen (compiled model deployment)
- ONNX conversion via Treelite ecosystem
- Common format for model sharing

**Constraints**:
- Only for models without linear leaves
- Only for GBDT, not GBLinear
- Export-only (not for loading back into booste-rs)

### Recommendation

**Do not use Treelite as the primary storage format** for booste-rs.

Instead:
1. Define a native format that supports all booste-rs features
2. Consider Treelite export as a future interoperability feature for standard GBDT models

---

## XGBoost JSON Format

### Overview

XGBoost uses JSON (and UBJSON for binary efficiency) as its primary model format since v1.0.

### Structure

```json
{
  "version": [2, 1, 0],
  "learner": {
    "learner_model_param": {
      "base_score": "0.5",
      "num_feature": "10",
      "num_class": "0"
    },
    "gradient_booster": {
      "name": "gbtree",
      "model": {
        "gbtree_model_param": { "num_trees": "100" },
        "trees": [
          {
            "tree_param": { "num_nodes": "15" },
            "split_indices": [...],
            "split_conditions": [...],
            "left_children": [...],
            "right_children": [...],
            "default_left": [...],
            "categories": [...],  // optional
            "split_type": [...]   // 0=numerical, 1=categorical
          }
        ]
      }
    },
    "objective": { "name": "binary:logistic", ... }
  }
}
```

### Considerations

**Pros**:
- Human readable
- Well documented
- Supports GBLinear (different structure)
- Supports DART (with tree weights)

**Cons**:
- Verbose (large file sizes)
- Frequent format changes between XGBoost versions
- Quirks (base_score as string, array, or bracketed string)
- No linear leaves (XGBoost doesn't support them)

---

## LightGBM Text Format

### Overview

LightGBM uses a line-based text format for model serialization.

### Structure

```
tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=9
objective=binary sigmoid:1

parameters:
[boosting: gbdt]
...
end of parameters

Tree=0
num_leaves=31
num_cat=0
split_feature=3 7 2 ...
threshold=0.5 1.2 0.8 ...
decision_type=2 2 2 ...
left_child=-1 -2 0 ...
right_child=1 2 -3 ...
leaf_value=0.1 -0.2 0.05 ...

Tree=1
...
```

### Linear Tree Extension

```
is_linear=1
num_features_per_leaf=3 2 4 ...
leaf_features=0:2:5 1:3 0:1:2:4 ...
leaf_coeff=0.1:0.2:-0.1 0.3:0.4 ...
leaf_const=0.5 0.3 0.1 ...
```

### Considerations

**Pros**:
- Supports linear trees
- Relatively stable format
- Human readable

**Cons**:
- Text parsing overhead
- No GBLinear equivalent
- Format quirks (threshold adjustment for `<=` vs `<`)

---

## Conclusion

Neither XGBoost nor LightGBM formats are ideal for booste-rs:

| Feature | XGBoost JSON | LightGBM Text | Treelite v4 |
|---------|--------------|---------------|-------------|
| GBLinear | ✅ | ❌ | ❌ |
| Linear leaves | ❌ | ✅ | ❌ |
| DART weights | ✅ | ❌ | ❌ |
| Categorical | ✅ | ✅ | ✅ |
| Human readable | ✅ | ✅ | ❌ |
| Compact | ❌ | ❌ | ✅ |

**Recommendation**: Define a native booste-rs format that unifies all features.
