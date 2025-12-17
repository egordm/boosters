# LightGBM Linear Trees

> Source: `src/treelearner/linear_tree_learner.cpp`, `src/treelearner/linear_tree_learner.h`

## Overview

LightGBM's **linear trees** feature fits a **linear regression model within each leaf** instead of using a constant value. This hybrid approach combines the partitioning power of decision trees with the local smoothness of linear models.

This is inspired by the paper: [Local Linear Forests](https://arxiv.org/abs/1802.05640).

## Key Concept

Standard decision trees:
- Each leaf predicts a **constant value** (e.g., mean of targets)
- Prediction is discontinuous across leaf boundaries

Linear trees:
- Each leaf predicts: `output = constant + Σ(coeff_i × feature_i)`
- Features used are those along the **branch path** to that leaf
- Provides **smooth** predictions within each partition

```
Standard Tree Leaf:
    y = c (constant)

Linear Tree Leaf:
    y = c + β₁x₁ + β₂x₂ + ... + βₖxₖ
    (where x₁..xₖ are features used in splits leading to this leaf)
```

## Algorithm

### 1. Tree Structure Phase (Standard)

First, grow the tree normally using leaf-wise or depth-wise strategy:
- Build histograms
- Find best splits
- Create tree structure

### 2. Linear Coefficient Fitting Phase

After tree structure is fixed, fit linear models in each leaf:

```cpp
// From linear_tree_learner.cpp
// calculate coefficients using the method described in Eq 3 of https://arxiv.org/abs/1802.05640
// the coefficients vector is given by
// - (X_T * H * X + lambda) ^ (-1) * (X_T * g)
// where:
// X is the matrix where columns are feature values plus a constant column
// H is the diagonal matrix of the hessian
// lambda is the diagonal matrix with L2 regularization (linear_lambda)
// g is the vector of gradients
```

**Key steps:**

1. **Identify features**: For each leaf, gather numerical features used in splits along the branch path
2. **Build design matrix X**: Feature values + intercept column  
3. **Compute X^T H X**: Weighted by hessians (second-order info)
4. **Add regularization**: `linear_lambda` on diagonal
5. **Solve linear system**: Using Eigen's `fullPivLu().inverse()`
6. **Filter coefficients**: Remove near-zero coefficients

### 3. NaN Handling

Linear trees require raw feature values (not binned), so NaN handling is explicit:

```cpp
// From linear_tree_learner.cpp
// identify features containing nans
for (int feat = 0; feat < train_data->num_features(); ++feat) {
    auto bin_mapper = this->train_data_->FeatureBinMapper(feat);
    if (bin_mapper->bin_type() == BinType::NumericalBin) {
        const float* feat_ptr = this->train_data_->raw_index(feat);
        for (int i = 0; i < train_data->num_data(); ++i) {
            if (std::isnan(feat_ptr[i])) {
                contains_nan_[feat] = 1;
                break;
            }
        }
    }
}
```

When NaN is encountered:
- **Training**: Skip sample in coefficient calculation
- **Inference**: Fall back to constant leaf output

## Data Structures

### Per-Leaf Storage

```cpp
// Stored in Tree object
std::vector<double> leaf_const_;           // Intercept for each leaf
std::vector<std::vector<double>> leaf_coeff_;  // Coefficients per leaf
std::vector<std::vector<int>> leaf_features_;  // Feature indices per leaf
```

### Temporary Matrices (During Training)

```cpp
// Upper triangular of X^T H X stored in row-major order
std::vector<std::vector<double>> XTHX_;
// X^T g vector
std::vector<std::vector<double>> XTg_;
// Per-thread copies for parallel accumulation
std::vector<std::vector<std::vector<double>>> XTHX_by_thread_;
std::vector<std::vector<std::vector<double>>> XTg_by_thread_;
```

## Prediction

```cpp
// From linear_tree_learner.h AddPredictionToScoreInner
for (int i = 0; i < num_data; ++i) {
    int leaf_num = leaf_map_[i];
    double output = leaf_const[leaf_num];  // Start with intercept
    
    for (int feat_ind = 0; feat_ind < leaf_num_features[leaf_num]; ++feat_ind) {
        float val = feat_ptr[leaf_num][feat_ind][i];
        if (std::isnan(val)) {
            // Fall back to constant leaf output
            output = leaf_output[leaf_num];
            break;
        }
        output += val * leaf_coeff[leaf_num][feat_ind];
    }
    out_score[i] += output;
}
```

## Configuration

```python
params = {
    'linear_tree': True,      # Enable linear trees
    'linear_lambda': 0.0,     # L2 regularization on linear coefficients
}
```

**Important notes:**
- Requires storing raw feature values (increased memory)
- Only uses **numerical features** in linear models
- Categorical features are used for splits but not in leaf models

## Comparison: Linear Trees vs Standard Trees

| Aspect | Standard Tree | Linear Tree |
|--------|--------------|-------------|
| Leaf Output | Constant | Linear function |
| Prediction | `c` | `c + Σ βᵢxᵢ` |
| Memory | Low | High (raw features) |
| Speed | Fast | Slower |
| Smoothness | Discontinuous | Smooth within partitions |
| Best For | Tabular data | Data with local linear structure |

## When to Use Linear Trees

**Good for:**
- Data with local linear relationships
- Smooth target functions
- When interpretability of leaf models matters
- Time series with trends

**Avoid for:**
- Pure categorical data
- Highly non-linear local structure  
- Memory-constrained environments
- Very large datasets (memory overhead)

## Implementation Notes

### Dependencies
- Uses **Eigen** library for matrix operations
- Requires raw feature storage (not just binned)

### Memory Overhead
```cpp
// From config.h
// Note: setting linear_tree=true significantly increases the memory use of LightGBM
```

### GPU Support
Linear trees work with GPU tree learner but coefficient fitting is done on CPU:
```cpp
template void LinearTreeLearner<GPUTreeLearner>::Init(...);
template Tree* LinearTreeLearner<GPUTreeLearner>::Train(...);
```

## booste-rs Implications

### For Inference
- Need to store: `leaf_const`, `leaf_coeff`, `leaf_features` per leaf
- Prediction requires raw feature values (not binned)
- NaN handling with fallback to constant

### For Training (Future)
- Would require Eigen or similar linear algebra
- Significant memory for raw feature storage
- Consider making optional feature

### Recommended Approach
1. **Inference**: Support loading and predicting linear tree models
2. **Training**: Low priority - complex with limited use cases

## References

- Paper: [Friedman, Popescu - Local Linear Forests (2018)](https://arxiv.org/abs/1802.05640)
- LightGBM docs: [Parameters - linear_tree](https://lightgbm.readthedocs.io/en/latest/Parameters.html#linear_tree)
