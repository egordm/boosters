# Multi-Output Training Strategies

## Overview

Gradient boosting frameworks support multiple output targets for:
- **Multiclass classification**: K classes → K outputs
- **Multi-target regression**: K targets → K outputs
- **Multi-quantile regression**: K quantiles → K outputs

There are two fundamentally different strategies for handling multiple outputs.

## Strategy 1: One Tree Per Output

**Approach**: Train K separate trees per boosting round, where each tree handles one output.

```text
Round 1: Tree_0 (output 0), Tree_1 (output 1), Tree_2 (output 2)
Round 2: Tree_3 (output 0), Tree_4 (output 1), Tree_5 (output 2)
...
Total: K × num_rounds trees
```

### How It Works

1. Compute gradients for all K outputs: `grad[sample][output]`, `hess[sample][output]`
2. For each output k:
   - Extract `grad[:, k]` and `hess[:, k]`
   - Train a single-output tree using standard histogram-based split finding
   - Tree stores scalar leaf values
3. Forest metadata tracks which trees belong to which output (group)

### Tree Storage

Each tree has scalar leaf values:
```cpp
// XGBoost RegTree (single-output)
float LeafValue(bst_node_t nidx) const;  // Returns single float
```

### Advantages

- **Simple**: All tree building code is single-output only
- **Flexible**: Different trees can have different structures per output
- **Memory-efficient**: Histograms store 2 floats per bin (grad, hess)
- **Well-tested**: Default in both XGBoost and LightGBM

### Disadvantages

- **More trees**: K × N instead of N trees
- **Redundant structure**: Same splits often beneficial across outputs
- **Cannot share**: Split decisions are independent per output

## Strategy 2: Multi-Output Trees (Vector Leaves)

**Approach**: Train one tree per round with K-dimensional leaf values.

```text
Round 1: Tree_0 with leaves = [w_0, w_1, w_2] (K values per leaf)
Round 2: Tree_1 with leaves = [w_0, w_1, w_2]
...
Total: num_rounds trees (each with K-dimensional leaves)
```

### How It Works

1. Histograms store K gradients and K hessians per bin
2. Split gain = sum of gains across all K outputs
3. Leaf weight = K-dimensional vector
4. Single tree structure shared across all outputs

### Tree Storage

Each tree has vector leaf values:
```cpp
// XGBoost MultiTargetTree
linalg::VectorView<float const> LeafValue(bst_node_t nidx) const;  // Returns K floats
```

### Gain Computation

For multi-output, the split gain sums across all K outputs:

```cpp
float ComputeGain(GradStats left[K], GradStats right[K], float lambda, float gamma) {
    float total = 0.0;
    for (int k = 0; k < K; k++) {
        float parent_obj = sq(left[k].grad + right[k].grad) / (left[k].hess + right[k].hess + lambda);
        float left_obj = sq(left[k].grad) / (left[k].hess + lambda);
        float right_obj = sq(right[k].grad) / (right[k].hess + lambda);
        total += 0.5 * (left_obj + right_obj - parent_obj);
    }
    return total - gamma;  // Single gamma penalty
}
```

### Advantages

- **Fewer trees**: N instead of K × N
- **Shared structure**: One split benefits all outputs
- **Better for correlated outputs**: Exploits output correlations
- **Smaller model size**: Fewer nodes total

### Disadvantages

- **Complex histograms**: K × 2 floats per bin vs 2 floats
- **More memory**: Histograms are K times larger
- **May force suboptimal splits**: A split good for average may be bad for some outputs
- **Limited feature support**: Some features (e.g., monotonic constraints) harder to implement

## XGBoost Implementation

XGBoost supports both strategies via the `multi_strategy` parameter:

```cpp
enum class MultiStrategy : std::int32_t {
  kOneOutputPerTree = 0,  // Default
  kMultiOutputTree = 1,
};
```

### Configuration

```python
# Strategy 1: One tree per output (default)
xgb.XGBClassifier(multi_strategy="one_output_per_tree")

# Strategy 2: Multi-output trees
xgb.XGBClassifier(multi_strategy="multi_output_tree")
```

### Data Structures

**Strategy 1** uses `RegTree`:
```cpp
struct RegTree {
    std::vector<Node> nodes_;  // Node stores single leaf value
    // ...
};

struct Node {
    float leaf_value() const { return info_.leaf_value; }
};
```

**Strategy 2** uses `MultiTargetTree`:
```cpp
class MultiTargetTree {
    TreeParam const* param_;
    HostDeviceVector<float> weights_;  // [num_nodes * size_leaf_vector]
    // ...
    
    linalg::VectorView<float const> LeafValue(bst_node_t nidx) const {
        auto beg = nidx * NumTargets();
        return weights_.subspan(beg, NumTargets());
    }
};
```

### Histogram Storage

**Strategy 1**: Standard single-output histogram
```cpp
struct GradientPairPrecise {
    double grad_;
    double hess_;
};
// Per bin: 1 GradientPairPrecise = 16 bytes
```

**Strategy 2**: Multi-output histogram

XGBoost uses an interesting approach for multi-output histograms: rather than a single
K-dimensional histogram structure, it maintains **K separate single-output histogram builders**:

```cpp
// histogram.h - MultiHistogramBuilder class
class MultiHistogramBuilder {
private:
    std::vector<HistogramBuilder> target_builders_;  // K separate builders

public:
    void BuildHistogram(common::BlockedSpace2d const& space,
                       GHistIndexMatrix const& gidx,
                       linalg::MatrixView<GradientPairPrecise const> gpair,
                       ...) {
        bst_target_t n_targets = target_builders_.size();
        for (bst_target_t t = 0; t < n_targets; ++t) {
            // Slice gradients for target t: gpair[:, t]
            auto t_gpair = gpair.Slice(linalg::All(), t);
            
            // Build histogram for this target using existing single-output code
            target_builders_[t].BuildHist(t_gpair, ...);
        }
    }
};
```

**Key insight**: This design reuses all existing single-output histogram code. Each target
gets its own `HistogramBuilder` that operates on sliced gradients `gpair[:, t]`.

Memory layout ends up equivalent (K × 16 bytes per bin), but the implementation is simpler
than creating a new K-dimensional histogram structure.

```cpp
// Effective layout (conceptual)
// Per bin: K × GradientPairPrecise = K × 16 bytes
// Stored as K separate histograms, each with 1 GradientPairPrecise per bin
```

### Training Loop Difference

**Strategy 1** (one_output_per_tree):
```cpp
// For each boosting round
for (bst_target_t gid = 0; gid < n_groups; ++gid) {
    // Extract gradients for this output
    auto gpair_for_group = GetGradientSlice(gid);
    
    // Train single-output tree
    auto tree = TreeLearner::Train(gpair_for_group);
    trees.push_back(tree);
}
```

**Strategy 2** (multi_output_tree):
```cpp
// For each boosting round
// Use all K gradients together
auto tree = MultiOutputTreeLearner::Train(all_gpairs);  // K-dimensional
trees.push_back(tree);
```

## LightGBM Implementation

LightGBM only supports Strategy 1 (one tree per output):

```cpp
// gbdt.cpp
num_tree_per_iteration_ = num_class_;

for (int cur_tree_id = 0; cur_tree_id < num_tree_per_iteration_; ++cur_tree_id) {
    const size_t offset = static_cast<size_t>(cur_tree_id) * num_data_;
    auto grad = gradients + offset;
    auto hess = hessians + offset;
    new_tree.reset(tree_learner_->Train(grad, hess, is_first_tree));
}
```

LightGBM does not have a `multi_output_tree` equivalent.

## Memory Comparison

For a dataset with 256 bins, 10 classes (K=10), and 1000 features:

### Strategy 1 (per-tree histograms)
- Per-feature histogram: 256 bins × 16 bytes = 4 KB
- Per-node histogram: 1000 features × 4 KB = 4 MB
- Histograms reused across K trees within a round
- **Peak memory**: ~4 MB per node being built

### Strategy 2 (K-output histograms)
- Per-feature histogram: 256 bins × 10 outputs × 16 bytes = 40 KB
- Per-node histogram: 1000 features × 40 KB = 40 MB
- **Peak memory**: ~40 MB per node being built

Strategy 1 uses 10× less histogram memory.

## When to Use Each Strategy

### Use Strategy 1 (One Tree Per Output) When:
- Outputs have different optimal tree structures
- Memory is constrained
- Outputs are independent or weakly correlated
- You need maximum feature support (monotonic constraints, etc.)
- Debugging/interpretability is important (can examine per-output trees)

### Use Strategy 2 (Multi-Output Trees) When:
- Outputs are highly correlated
- Model size is critical (fewer trees)
- Inference speed matters (one tree traversal vs K)
- Outputs should share the same split structure

## Gain Computation Details

### Strategy 1: Per-Output Gain

Each output k has independent gain:

```text
Gain_k = 0.5 * (G_L^2 / (H_L + λ) + G_R^2 / (H_R + λ) - (G_L+G_R)^2 / (H_L+H_R + λ)) - γ
```

Trees for different outputs may split on different features at the same depth.

### Strategy 2: Summed Gain

Gain summed across outputs:

```text
Gain = Σ_k [ 0.5 * (G_L[k]^2 / (H_L[k] + λ) + G_R[k]^2 / (H_R[k] + λ) 
           - (G_L[k]+G_R[k])^2 / (H_L[k]+H_R[k] + λ)) ] - γ
```

A split is chosen if it improves the sum, even if it hurts some outputs.

### XGBoost Multi-Target Gain Implementation

XGBoost implements multi-target gain calculation in `src/tree/hist/evaluate_splits.h`:

```cpp
// MultiCalcSplitGain (evaluate_splits.h:503)
XGBOOST_DEVICE auto MultiCalcSplitGain(
    TrainParam const& param,
    linalg::VectorView<GradientPairPrecise const> left_sum,
    linalg::VectorView<GradientPairPrecise const> right_sum
) {
    float gain{0.0f};
    
    // Compute weight and gain for each target on left child
    for (bst_target_t t = 0; t < n_targets; ++t) {
        auto w = CalcWeight(param, left_sum(t).GetGrad(), left_sum(t).GetHess());
        gain += CalcGainGivenWeight(param, left_sum(t).GetGrad(), 
                                    left_sum(t).GetHess(), w);
    }
    
    // Compute weight and gain for each target on right child
    for (bst_target_t t = 0; t < n_targets; ++t) {
        auto w = CalcWeight(param, right_sum(t).GetGrad(), right_sum(t).GetHess());
        gain += CalcGainGivenWeight(param, right_sum(t).GetGrad(), 
                                    right_sum(t).GetHess(), w);
    }
    
    return gain;
}
```

The underlying scalar functions (from `src/tree/param.h`):

```cpp
// CalcWeight: Compute optimal leaf weight
template <typename T>
T CalcWeight(TrainingParams const& p, T sum_grad, T sum_hess) {
    if (sum_hess < p.min_child_weight || sum_hess <= 0.0) {
        return 0.0;
    }
    T dw = -ThresholdL1(sum_grad, p.reg_alpha) / (sum_hess + p.reg_lambda);
    if (p.max_delta_step != 0.0f && std::abs(dw) > p.max_delta_step) {
        dw = std::copysign(p.max_delta_step, dw);
    }
    return dw;
}

// CalcGainGivenWeight: Compute gain for a node given its weight
template <typename T>
T CalcGainGivenWeight(TrainingParams const& p, T sum_grad, T sum_hess, T w) {
    return -(2.0 * sum_grad * w + (sum_hess + p.reg_lambda) * w * w);
}
```

**Key insight**: Multi-target gain is computed by simply summing the per-target gains.
There's no special interaction between targets—each target contributes independently.
This means multi-output trees optimize for "average improvement across all targets".

## Source Code References

### XGBoost

| Component | File | Description |
|-----------|------|-------------|
| MultiStrategy enum | `include/xgboost/learner.h` | Strategy selection |
| MultiTargetTree | `include/xgboost/multi_target_tree_model.h` | Vector leaf tree |
| MultiTargetTree impl | `src/tree/multi_target_tree_model.cc` | Tree operations |
| RegTree | `include/xgboost/tree_model.h` | Standard scalar tree |
| Training loop | `src/gbm/gbtree.cc` | Per-group iteration |
| Histogram | `src/tree/hist/histogram.h` | Gradient accumulation |

### LightGBM

| Component | File | Description |
|-----------|------|-------------|
| Training loop | `src/boosting/gbdt.cpp` | `num_tree_per_iteration_` |
| Tree learner | `src/treelearner/serial_tree_learner.cpp` | Single-output only |
| Histogram | `src/treelearner/feature_histogram.hpp` | Single-output only |

## Summary

| Aspect | Strategy 1 (Per-Output) | Strategy 2 (Vector Leaves) |
|--------|------------------------|---------------------------|
| Trees per round | K | 1 |
| Total trees | K × N | N |
| Histogram memory | O(bins × features) | O(K × bins × features) |
| Split structure | Independent per output | Shared across outputs |
| Supported in XGBoost | ✅ Yes (default) | ✅ Yes (opt-in) |
| Supported in LightGBM | ✅ Yes (only option) | ❌ No |
| Best for | Independent outputs | Correlated outputs |
