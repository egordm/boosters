# Explainability Research

This document covers the theory and implementation approaches for model explainability in gradient boosting systems.

## Overview

Model explainability in GBDT/GBLinear systems operates at two levels:

1. **Global Explanations** - Understanding the overall model behavior
   - Feature importance (gain, split count, cover)
   - Feature interactions

2. **Local Explanations** - Understanding individual predictions
   - SHAP values (Shapley additive explanations)
   - Feature contributions per prediction

## Feature Importance

### Types of Feature Importance

| Type | Description | Use Case |
|------|-------------|----------|
| **Split count** | Number of times a feature is used for splitting | Simple, fast, but can be misleading for high-cardinality features |
| **Gain** | Total/average gain from splits using this feature | Better indicator of predictive power |
| **Cover** | Number of samples covered by splits on this feature | Indicates breadth of feature influence |
| **Permutation** | Loss increase when feature is randomly shuffled | Most reliable but computationally expensive |

### XGBoost Implementation

XGBoost computes feature importance via `XGBoosterFeatureScore`:

```cpp
// Types: weight, gain, cover, total_gain, total_cover
// For tree models: walks tree structure
// For linear models: returns normalized coefficients
```

### LightGBM Implementation

LightGBM computes via `LGBM_BoosterFeatureImportance`:
- `importance_type = 0`: Split count
- `importance_type = 1`: Gain

### Categorical Feature Handling

For categorical features with one-hot encoding or native handling:
- Sum importance across all categories
- Provide per-category breakdown as optional

### Linear Model Feature Importance

For GBLinear and linear leaves:
- Weight magnitude: `|w_i|` (absolute coefficient value)
- Standardized weight: `|w_i| * std(x_i)` (accounts for feature scale)

## SHAP Values (Shapley Additive Explanations)

### Theory

SHAP values are based on game theory's Shapley values. For a prediction:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

Where:
- $\phi_0$ = expected model output (base value)
- $\phi_i$ = contribution of feature $i$ to the prediction

The Shapley value for feature $i$ is:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(M-|S|-1)!}{M!} [f_{S \cup \{i\}}(x) - f_S(x)]$$

### TreeSHAP Algorithm

For tree ensembles, TreeSHAP provides an efficient O(TLD²) algorithm where:
- T = number of trees
- L = max leaves per tree
- D = max depth

The key insight is that we can compute exact Shapley values by tracking paths through trees.

#### Path-Based Algorithm (Lundberg et al.)

```
TREESHAP(tree, x):
    φ = [0] * num_features
    
    function recurse(node, path, zero_weight):
        if node is leaf:
            for each unique feature in path:
                # Compute contribution based on path weights
                φ[feature] += contribution
        else:
            # Split path based on feature
            left_path = extend_path(path, go_left(x, node))
            right_path = extend_path(path, go_right(x, node))
            
            recurse(node.left, left_path, ...)
            recurse(node.right, right_path, ...)
    
    recurse(root, empty_path, 1.0)
    return φ
```

### Interventional vs Conditional SHAP

1. **Conditional (TreeSHAP)**: Uses tree structure to marginalize features
   - Faster, leverages tree paths
   - Can produce counterintuitive results for correlated features

2. **Interventional**: Treats features as independent
   - More causally meaningful
   - Requires different algorithm (GPUTreeSHAP supports this)

### GPUTreeSHAP Approach

The GPUTreeSHAP library (used by XGBoost) uses:
- Path extraction: Convert tree to list of `PathElement` structures
- Parallel processing: One thread per (row, path) pair
- Efficient memory layout for coalesced access

```cpp
struct PathElement {
    size_t path_idx;        // Unique path index
    int64_t feature_idx;    // Feature for this split (-1 for root)
    int group;              // Output group (multiclass)
    SplitCondition split;   // Threshold info
    double zero_fraction;   // P(taking this path when feature is "unknown")
    float v;                // Leaf value
};
```

## SHAP for Linear Models

For linear models, SHAP values are exact and have closed form:

$$\phi_i = w_i (x_i - E[X_i])$$

Where:
- $w_i$ = coefficient for feature $i$
- $x_i$ = feature value for this prediction
- $E[X_i]$ = mean of feature $i$ over training data

### Linear Leaves in GBDT

For trees with linear leaves:
1. First compute tree contribution (which leaf was reached)
2. Within that leaf, compute linear SHAP values

Total contribution = tree path contribution + linear leaf contribution

## Multi-Output Models

For multiclass or multi-output regression:
- Compute SHAP values per output
- Return shape: `(n_samples, n_features + 1, n_outputs)`
- The +1 is for the expected value

## Implementation Strategy

### Phase 1: Feature Importance

1. **Tree-based models**
   - Walk tree structure during/after training
   - Accumulate gain/split-count per feature
   - Handle categorical features (sum across categories or report separately)

2. **Linear models**
   - Return coefficient magnitudes
   - Optionally standardize by feature scale

### Phase 2: SHAP Values

1. **Tree-based models (CPU)**
   - Implement path-based TreeSHAP algorithm
   - Support both conditional and interventional modes
   - Handle categoricals, missing values, default directions

2. **Linear models**
   - Implement closed-form solution
   - Requires storing feature means from training

3. **Linear leaves**
   - Hybrid approach: tree path + linear contribution

### GPU Consideration (Future)

For GPU acceleration:
- Extract paths to contiguous memory
- Use GPUTreeSHAP algorithm
- Batch rows for efficient kernel execution

## References

1. Lundberg, Scott M., and Su-In Lee. "A unified approach to interpreting model predictions." NeurIPS 2017.

2. Lundberg, Scott M., Gabriel G. Erion, and Su-In Lee. "Consistent individualized feature attribution for tree ensembles." arXiv:1802.03888 (2018).

3. Mitchell, R., Frank, E., & Holmes, G. "GPUTreeShap: Massively parallel exact calculation of SHAP scores for tree ensembles." arXiv:2010.13972 (2022).

4. XGBoost SHAP implementation: `xgboost/src/tree/tree_shap.cc`

5. GPUTreeSHAP: `xgboost/gputreeshap/GPUTreeShap/gpu_treeshap.h`

6. LightGBM feature importance: `LightGBM/src/boosting/gbdt.cpp`
