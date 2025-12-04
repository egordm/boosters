# Split Finding and Evaluation

## Overview

After building a histogram for a node, we need to find the best split. This involves
scanning histogram bins for each feature and computing the gain for each potential
split point.

## Split Gain Formula

The split gain measures how much a split reduces the objective function:

```text
Gain = ½ × [G_L²/(H_L + λ) + G_R²/(H_R + λ) - G²/(H + λ)] - γ

where:
  G_L, H_L = sum of gradients/hessians in left child
  G_R, H_R = sum of gradients/hessians in right child
  G, H     = G_L + G_R, H_L + H_R (parent sums)
  λ        = reg_lambda (L2 regularization)
  γ        = min_split_loss (complexity penalty)
```

### Intuition

The gain compares:

- **Term 1**: G_L²/(H_L + λ) — how well left child explains its gradients
- **Term 2**: G_R²/(H_R + λ) — how well right child explains its gradients  
- **Term 3**: G²/(H + λ) — how well parent explains all gradients (subtracted)
- **Term 4**: γ — penalty for adding complexity (subtracted)

A split is beneficial when children explain gradients better than parent.

## Split Enumeration

For each feature, scan the histogram bins and track cumulative sums:

```text
Feature histogram: bins 0, 1, 2, 3, ...

Forward scan:
  Split at bin 0: left = {0},     right = {1,2,3,...}
  Split at bin 1: left = {0,1},   right = {2,3,...}
  Split at bin 2: left = {0,1,2}, right = {3,...}
  ...
```

**Key insight**: Cumulative sums allow O(1) gain computation per split:

```cpp
// Forward enumeration
GradStats left_sum;
for (bin in bins) {
  left_sum += hist[bin];
  right_sum = parent_sum - left_sum;
  gain = CalcSplitGain(left_sum, right_sum);
  if (gain > best_gain) {
    best_gain = gain;
    best_split = bin;
  }
}
```

## Handling Missing Values

XGBoost assigns missing values to either left or right child, choosing the direction
that maximizes gain. This requires scanning in both directions:

```text
Forward scan:  missing goes RIGHT (with right child)
Backward scan: missing goes LEFT (with left child)
```

### Implementation

From `evaluate_splits.h`:

```cpp
// Forward enumeration: missing goes right
auto grad_stats = EnumerateSplit<+1>(cut, hist, fidx, nidx, evaluator, &best);

// Check if feature has missing values
if (SplitContainsMissingValues(grad_stats, parent_stats)) {
  // Backward enumeration: missing goes left
  EnumerateSplit<-1>(cut, hist, fidx, nidx, evaluator, &best);
}
```

The `SplitContainsMissingValues` check:

```cpp
bool SplitContainsMissingValues(GradStats e, NodeEntry parent) {
  // If histogram sum != parent sum, some values are missing
  return !(e.grad == parent.stats.grad && e.hess == parent.stats.hess);
}
```

## XGBoost Split Entry

```cpp
template<typename GradientT>
struct SplitEntryContainer {
  bst_float loss_chg;      // Gain of this split
  bst_feature_t sindex;    // Feature index | (default_left << 31)
  bst_float split_value;   // Split threshold
  bool is_cat;             // Categorical split?
  std::vector<uint32_t> cat_bits;  // Category bitmap (if categorical)
  
  GradientT left_sum;      // Gradient sum for left child
  GradientT right_sum;     // Gradient sum for right child
  
  // Check if new split is better
  bool NeedReplace(bst_float new_loss_chg, unsigned split_index) {
    if (std::isinf(new_loss_chg)) return false;
    if (this->SplitIndex() <= split_index) {
      return new_loss_chg > this->loss_chg;
    } else {
      return !(this->loss_chg > new_loss_chg);
    }
  }
};
```

## Parallel Split Evaluation

XGBoost evaluates splits in parallel across features:

```cpp
// Space: (nodes × features)
common::ParallelFor2d(space, n_threads, [&](node_in_set, feature_range) {
  auto tidx = omp_get_thread_num();
  auto entry = &tloc_candidates[n_threads * node_in_set + tidx];
  auto best = &entry->split;
  auto histogram = hist[entry->nid];
  
  for (fidx in feature_range) {
    EnumerateSplit(histogram, fidx, &best);
  }
});

// Reduce across threads
for (node : nodes) {
  for (tid : threads) {
    entries[node].split.Update(tloc_candidates[tid][node].split);
  }
}
```

## Constraints and Regularization

### Minimum Child Weight

A split is invalid if either child has insufficient hessian:

```cpp
bool IsValid(GradStats left, GradStats right) {
  return left.hess >= min_child_weight && 
         right.hess >= min_child_weight;
}
```

### Monotonic Constraints

Some features may be constrained to be monotonically increasing or decreasing:

```cpp
// constraint = +1: feature must increase output
// constraint = -1: feature must decrease output
// constraint = 0:  no constraint

// During split evaluation, check if split respects monotonicity
bool SatisfiesConstraint(left_weight, right_weight, constraint) {
  if (constraint == 0) return true;
  if (constraint > 0) return left_weight <= right_weight;
  return left_weight >= right_weight;
}
```

### Interaction Constraints

Limit which features can interact (appear together in paths):

```cpp
// If node was split on feature A, constrain descendants
// to only use features in the same interaction group
bool Query(bst_node_t nid, bst_feature_t fidx) {
  return allowed_features_[nid].contains(fidx);
}
```

## Categorical Feature Splits

XGBoost supports two strategies for categorical features:

### One-Hot Encoding (few categories)

Each category becomes a separate split candidate:

```text
Category ∈ {A, B, C, D}

Possible splits:
  {A} vs {B, C, D}
  {B} vs {A, C, D}
  {C} vs {A, B, D}
  {D} vs {A, B, C}
```

### Partition-Based (many categories)

Sort categories by gradient statistic, find optimal partition:

```text
Categories sorted by gradient: [C, A, D, B]

Scan partitions:
  {C} vs {A, D, B}
  {C, A} vs {D, B}
  {C, A, D} vs {B}
```

This is O(k log k) instead of O(2^k) for k categories.

## Weight Calculation

Once a split is chosen, compute leaf weights:

```cpp
// Optimal weight minimizes objective
weight = -G / (H + λ)

// With L1 regularization (alpha)
weight = -ThresholdL1(G, α) / (H + λ)

// ThresholdL1: soft thresholding
T ThresholdL1(T w, T alpha) {
  if (w > +alpha) return w - alpha;
  if (w < -alpha) return w + alpha;
  return 0.0;
}
```

## Considerations for booste-rs

### What We Need

1. **SplitEntry struct**: Store best split per node
2. **EnumerateSplit function**: Scan bins, compute gains
3. **Forward/backward scans**: Handle missing values
4. **Gain calculation**: With regularization terms

### Potential Simplifications

1. **Skip categorical splits initially**: Focus on numeric features
2. **Skip interaction constraints initially**: Add if needed
3. **Simple min_child_weight**: Ignore advanced constraints first

### Potential Improvements

1. **SIMD gain calculation**: Vectorize across multiple split points
2. **Early stopping**: Skip feature if remaining gain can't beat current best
3. **Feature importance tracking**: Accumulate gain per feature

## Source Code References

| Component | XGBoost Source |
|-----------|----------------|
| HistEvaluator | `src/tree/hist/evaluate_splits.h` |
| EnumerateSplit | `src/tree/hist/evaluate_splits.h` |
| CalcGain/CalcWeight | `src/tree/param.h` |
| SplitEntry | `src/tree/param.h` |
| TreeEvaluator | `src/tree/split_evaluator.h` |
