# RFC-0013: Split Finding & Gain Computation

- **Status**: Draft
- **Created**: 2024-11-30
- **Updated**: 2024-11-30
- **Depends on**: RFC-0012 (Histogram Building)
- **Scope**: Finding optimal splits from histograms and computing gain

## Summary

This RFC defines how the best split is found for a tree node given its gradient histograms:

1. **Gain computation**: The objective reduction formula
2. **Split enumeration**: Iterating candidate splits per feature
3. **Split selection**: Choosing the globally best split across features
4. **Regularization**: L1/L2 penalties and constraints

## Motivation

Split finding determines model quality. For each node, we must:

1. Enumerate all possible splits (one per bin boundary per feature)
2. Compute gain for each candidate split
3. Select the split with maximum gain (if above threshold)
4. Handle missing values (default direction)

This is called O(trees × nodes) times. Each call scans O(features × bins) candidates.

## Design

### Overview

```
Split finding pipeline:
━━━━━━━━━━━━━━━━━━━━━━

NodeHistogram                    SplitCandidates              BestSplit
┌────────────────┐              ┌──────────────────┐         ┌──────────────┐
│ Feature 0:     │  enumerate   │ (f0, bin3, 0.5)  │  max    │ feature: 2   │
│   bins[0..255] │  ─────────▶  │ (f0, bin7, 0.3)  │  ────▶  │ threshold:   │
│ Feature 1:     │              │ (f1, bin2, 0.8)  │         │   1.23       │
│   bins[0..255] │              │ (f2, bin5, 1.2)  │  ◀──    │ gain: 1.2    │
│ ...            │              │ ...              │         │ left_weight  │
└────────────────┘              └──────────────────┘         │ right_weight │
                                                             └──────────────┘
```

### Gain Computation

The gain from a split measures how much the objective function decreases:

```rust
/// Parameters for gain computation
#[derive(Clone, Copy)]
pub struct GainParams {
    /// L2 regularization on leaf weights (lambda)
    pub lambda: f32,
    /// L1 regularization on leaf weights (alpha)
    pub alpha: f32,
    /// Minimum loss reduction to make a split (gamma)
    pub min_split_gain: f32,
    /// Minimum sum of hessians in a child (min_child_weight)
    pub min_child_weight: f32,
}

impl Default for GainParams {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            alpha: 0.0,
            min_split_gain: 0.0,
            min_child_weight: 1.0,
        }
    }
}

/// Compute the objective value for a leaf node
/// 
/// obj = -0.5 * G² / (H + λ) + α|w*|
/// where w* = -G / (H + λ) is the optimal leaf weight
#[inline]
pub fn leaf_objective(sum_grad: f32, sum_hess: f32, params: &GainParams) -> f32 {
    let g = sum_grad;
    let h = sum_hess + params.lambda;
    
    if params.alpha > 0.0 {
        // L1 regularized case (soft thresholding)
        let w = soft_threshold(g, params.alpha) / h;
        -0.5 * g * g / h + params.alpha * w.abs()
    } else {
        // Simple L2 case
        -0.5 * g * g / h
    }
}

/// Soft thresholding for L1 regularization
#[inline]
fn soft_threshold(g: f32, alpha: f32) -> f32 {
    if g > alpha {
        g - alpha
    } else if g < -alpha {
        g + alpha
    } else {
        0.0
    }
}

/// Compute optimal leaf weight
#[inline]
pub fn leaf_weight(sum_grad: f32, sum_hess: f32, params: &GainParams) -> f32 {
    let h = sum_hess + params.lambda;
    if params.alpha > 0.0 {
        -soft_threshold(sum_grad, params.alpha) / h
    } else {
        -sum_grad / h
    }
}

/// Compute gain from splitting a node
/// 
/// gain = obj(left) + obj(right) - obj(parent) - gamma
///      = 0.5 * [G_L² / (H_L + λ) + G_R² / (H_R + λ) - G² / (H + λ)] - gamma
#[inline]
pub fn split_gain(
    grad_left: f32, hess_left: f32,
    grad_right: f32, hess_right: f32,
    grad_parent: f32, hess_parent: f32,
    params: &GainParams,
) -> f32 {
    let obj_left = leaf_objective(grad_left, hess_left, params);
    let obj_right = leaf_objective(grad_right, hess_right, params);
    let obj_parent = leaf_objective(grad_parent, hess_parent, params);
    
    // Gain is reduction in objective (negative objectives, so subtract)
    let gain = obj_parent - obj_left - obj_right - params.min_split_gain;
    
    gain.max(0.0)  // Negative gain means no improvement
}
```

### Split Information

```rust
/// Complete information about a split decision
#[derive(Clone, Debug)]
pub struct SplitInfo {
    /// Feature index to split on
    pub feature: u32,
    /// Split threshold (go left if value < threshold)
    pub threshold: f32,
    /// Gain from this split
    pub gain: f32,
    /// Sum of gradients in left child
    pub grad_left: f32,
    /// Sum of hessians in left child
    pub hess_left: f32,
    /// Sum of gradients in right child
    pub grad_right: f32,
    /// Sum of hessians in right child
    pub hess_right: f32,
    /// Optimal weight for left leaf
    pub weight_left: f32,
    /// Optimal weight for right leaf
    pub weight_right: f32,
    /// Default direction for missing values
    pub default_left: bool,
    /// Whether this is a categorical split
    pub is_categorical: bool,
    /// For categorical: which categories go left (empty for numerical)
    pub categories_left: Vec<u32>,
}

impl SplitInfo {
    /// A null split (no valid split found)
    pub fn none() -> Self {
        Self {
            feature: u32::MAX,
            threshold: f32::NAN,
            gain: f32::NEG_INFINITY,
            grad_left: 0.0,
            hess_left: 0.0,
            grad_right: 0.0,
            hess_right: 0.0,
            weight_left: 0.0,
            weight_right: 0.0,
            default_left: true,
            is_categorical: false,
            categories_left: Vec::new(),
        }
    }
    
    /// Check if this is a valid split
    pub fn is_valid(&self) -> bool {
        self.gain > 0.0 && self.feature != u32::MAX
    }
}
```

### Split Finder

```rust
/// Strategy for finding splits
pub trait SplitFinder {
    /// Find the best split for a node given its histogram
    fn find_best_split(
        &self,
        histogram: &NodeHistogram,
        cuts: &BinCuts,
        params: &GainParams,
    ) -> SplitInfo;
}

/// Standard greedy split finder
/// 
/// Enumerates all bin boundaries for all features,
/// computes gain for each, returns the best.
pub struct GreedySplitFinder {
    /// Features to consider (None = all features)
    pub feature_subset: Option<Vec<u32>>,
}

impl SplitFinder for GreedySplitFinder {
    fn find_best_split(
        &self,
        histogram: &NodeHistogram,
        cuts: &BinCuts,
        params: &GainParams,
    ) -> SplitInfo {
        let mut best = SplitInfo::none();
        
        let features = self.feature_subset.as_ref()
            .map(|f| f.as_slice())
            .unwrap_or_else(|| &(0..cuts.num_features).collect::<Vec<_>>());
        
        let parent_grad = histogram.total_grad;
        let parent_hess = histogram.total_hess;
        
        for &feat in features {
            let feat_hist = histogram.feature(feat);
            let feat_cuts = cuts.feature_cuts(feat);
            
            // Try each bin boundary as a split point
            let split = self.find_best_split_for_feature(
                feat,
                feat_hist,
                feat_cuts,
                parent_grad,
                parent_hess,
                params,
            );
            
            if split.gain > best.gain {
                best = split;
            }
        }
        
        best
    }
}

impl GreedySplitFinder {
    /// Find best split for a single feature
    fn find_best_split_for_feature(
        &self,
        feature: u32,
        hist: &FeatureHistogram,
        cuts: &[f32],
        parent_grad: f32,
        parent_hess: f32,
        params: &GainParams,
    ) -> SplitInfo {
        let mut best = SplitInfo::none();
        best.feature = feature;
        
        // Cumulative sums from left
        let mut grad_left = 0.0f32;
        let mut hess_left = 0.0f32;
        
        // Scan bins left to right
        for bin in 0..(hist.num_bins - 1) {
            let (g, h, _) = hist.bin_stats(bin);
            grad_left += g;
            hess_left += h;
            
            let grad_right = parent_grad - grad_left;
            let hess_right = parent_hess - hess_left;
            
            // Check min_child_weight constraint
            if hess_left < params.min_child_weight || hess_right < params.min_child_weight {
                continue;
            }
            
            let gain = split_gain(
                grad_left, hess_left,
                grad_right, hess_right,
                parent_grad, parent_hess,
                params,
            );
            
            if gain > best.gain {
                best.gain = gain;
                best.threshold = cuts[bin as usize + 1];  // Upper bound of bin
                best.grad_left = grad_left;
                best.hess_left = hess_left;
                best.grad_right = grad_right;
                best.hess_right = hess_right;
                best.weight_left = leaf_weight(grad_left, hess_left, params);
                best.weight_right = leaf_weight(grad_right, hess_right, params);
            }
        }
        
        // Also consider missing values going left vs right
        best.default_left = self.choose_default_direction(&best, hist, params);
        
        best
    }
    
    /// Choose default direction for missing values
    fn choose_default_direction(
        &self,
        split: &SplitInfo,
        hist: &FeatureHistogram,
        params: &GainParams,
    ) -> bool {
        // Missing values are in bin 0 (by convention from RFC-0011)
        let (missing_grad, missing_hess, missing_count) = hist.bin_stats(0);
        
        if missing_count == 0 {
            return true;  // No missing values, doesn't matter
        }
        
        // Compare gain with missing going left vs right
        let gain_left = split_gain(
            split.grad_left + missing_grad,
            split.hess_left + missing_hess,
            split.grad_right,
            split.hess_right,
            split.grad_left + split.grad_right + missing_grad,
            split.hess_left + split.hess_right + missing_hess,
            params,
        );
        
        let gain_right = split_gain(
            split.grad_left,
            split.hess_left,
            split.grad_right + missing_grad,
            split.hess_right + missing_hess,
            split.grad_left + split.grad_right + missing_grad,
            split.hess_left + split.hess_right + missing_hess,
            params,
        );
        
        gain_left >= gain_right
    }
}
```

### Categorical Split Finder

```rust
/// Split finder for categorical features
/// 
/// Uses gradient-sorted partition: sort categories by grad/hess ratio,
/// then binary search for optimal split point.
pub struct CategoricalSplitFinder;

impl CategoricalSplitFinder {
    /// Find best categorical split for a feature
    pub fn find_categorical_split(
        &self,
        feature: u32,
        hist: &FeatureHistogram,
        parent_grad: f32,
        parent_hess: f32,
        params: &GainParams,
    ) -> SplitInfo {
        // Collect non-empty categories with their gradient stats
        let mut categories: Vec<(u32, f32, f32)> = Vec::new();
        
        for bin in 0..hist.num_bins {
            let (g, h, count) = hist.bin_stats(bin);
            if count > 0 {
                categories.push((bin as u32, g, h));
            }
        }
        
        if categories.len() < 2 {
            return SplitInfo::none();
        }
        
        // Sort by gradient / hessian ratio (optimal ordering for binary partition)
        categories.sort_by(|a, b| {
            let ratio_a = a.1 / (a.2 + params.lambda);
            let ratio_b = b.1 / (b.2 + params.lambda);
            ratio_a.partial_cmp(&ratio_b).unwrap()
        });
        
        // Scan for best split point
        let mut best = SplitInfo::none();
        best.feature = feature;
        best.is_categorical = true;
        
        let mut grad_left = 0.0f32;
        let mut hess_left = 0.0f32;
        let mut cats_left = Vec::new();
        
        for i in 0..(categories.len() - 1) {
            let (cat, g, h) = categories[i];
            grad_left += g;
            hess_left += h;
            cats_left.push(cat);
            
            let grad_right = parent_grad - grad_left;
            let hess_right = parent_hess - hess_left;
            
            if hess_left < params.min_child_weight || hess_right < params.min_child_weight {
                continue;
            }
            
            let gain = split_gain(
                grad_left, hess_left,
                grad_right, hess_right,
                parent_grad, parent_hess,
                params,
            );
            
            if gain > best.gain {
                best.gain = gain;
                best.grad_left = grad_left;
                best.hess_left = hess_left;
                best.grad_right = grad_right;
                best.hess_right = hess_right;
                best.weight_left = leaf_weight(grad_left, hess_left, params);
                best.weight_right = leaf_weight(grad_right, hess_right, params);
                best.categories_left = cats_left.clone();
            }
        }
        
        best
    }
}
```

### Parallel Split Finding

```rust
impl GreedySplitFinder {
    /// Find best split with parallel feature evaluation
    pub fn find_best_split_parallel(
        &self,
        histogram: &NodeHistogram,
        cuts: &BinCuts,
        params: &GainParams,
    ) -> SplitInfo {
        let parent_grad = histogram.total_grad;
        let parent_hess = histogram.total_hess;
        let num_features = cuts.num_features;
        
        // Parallel map-reduce over features
        (0..num_features)
            .into_par_iter()
            .map(|feat| {
                let feat_hist = histogram.feature(feat);
                let feat_cuts = cuts.feature_cuts(feat);
                self.find_best_split_for_feature(
                    feat,
                    feat_hist,
                    feat_cuts,
                    parent_grad,
                    parent_hess,
                    params,
                )
            })
            .reduce(SplitInfo::none, |best, split| {
                if split.gain > best.gain { split } else { best }
            })
    }
}
```

### Multi-Output Split Finding

```rust
/// Split finding for multi-class problems
/// 
/// Gain is computed by summing across all classes.
pub struct MultiOutputSplitFinder<const NUM_CLASSES: usize>;

impl<const NUM_CLASSES: usize> MultiOutputSplitFinder<NUM_CLASSES> {
    /// Compute gain for multi-class split
    pub fn multi_class_gain(
        grad_left: &[f32; NUM_CLASSES],
        hess_left: &[f32; NUM_CLASSES],
        grad_right: &[f32; NUM_CLASSES],
        hess_right: &[f32; NUM_CLASSES],
        grad_parent: &[f32; NUM_CLASSES],
        hess_parent: &[f32; NUM_CLASSES],
        params: &GainParams,
    ) -> f32 {
        let mut total_gain = 0.0;
        
        for c in 0..NUM_CLASSES {
            let gain_c = split_gain(
                grad_left[c], hess_left[c],
                grad_right[c], hess_right[c],
                grad_parent[c], hess_parent[c],
                params,
            );
            total_gain += gain_c;
        }
        
        total_gain
    }
    
    /// Find best split for multi-output histogram
    pub fn find_best_split(
        &self,
        histogram: &MultiOutputNodeHistogram<NUM_CLASSES>,
        cuts: &BinCuts,
        params: &GainParams,
    ) -> SplitInfo {
        // Similar to single-output, but accumulates vectors
        todo!("Implement multi-output split finding")
    }
}
```

### Constraints Support

```rust
/// Monotonic constraints for features
#[derive(Clone, Copy, PartialEq)]
pub enum MonotonicConstraint {
    /// No constraint
    None,
    /// Feature must have non-negative effect
    Increasing,
    /// Feature must have non-positive effect  
    Decreasing,
}

/// Split finder with constraints
pub struct ConstrainedSplitFinder {
    /// Monotonic constraints per feature (None = no constraint)
    pub monotonic: Option<Vec<MonotonicConstraint>>,
    /// Maximum depth of tree
    pub max_depth: Option<u32>,
    /// Maximum number of leaves
    pub max_leaves: Option<u32>,
}

impl ConstrainedSplitFinder {
    /// Check if a split satisfies monotonic constraint
    fn satisfies_monotonic(
        &self,
        feature: u32,
        weight_left: f32,
        weight_right: f32,
    ) -> bool {
        match self.monotonic.as_ref().and_then(|m| m.get(feature as usize)) {
            Some(MonotonicConstraint::Increasing) => weight_left <= weight_right,
            Some(MonotonicConstraint::Decreasing) => weight_left >= weight_right,
            _ => true,
        }
    }
}
```

## Design Decisions

### DD-1: Gain Formula Variant

**Context**: Multiple equivalent gain formulations exist.

**Options considered**:

1. **XGBoost formula**: `0.5 * [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ`
2. **LightGBM formula**: Similar but with different regularization handling
3. **Simplified**: Just `G²/H` without explicit regularization

**Decision**: Use XGBoost formula with full L1/L2 support.

**Rationale**:

- Well-understood and proven
- Regularization is important for generalization
- Direct compatibility with XGBoost models
- L1 enables sparse leaf weights

### DD-2: Missing Value Default Direction

**Context**: Where should missing values go during splits?

**Options considered**:

1. **Learn from data**: Try both directions, pick better gain
2. **Always left**: Simpler but suboptimal
3. **Configurable**: User specifies default

**Decision**: Learn from data (XGBoost approach).

**Rationale**:

- Automatically handles sparse data optimally
- Small overhead (compute gain twice at split finding)
- XGBoost and LightGBM both do this

### DD-3: Categorical Split Algorithm

**Context**: How to find optimal binary partition of categories.

**Options considered**:

1. **Exhaustive**: Try all 2^(k-1) partitions — O(2^k)
2. **Gradient-sorted**: Sort by g/h ratio, linear scan — O(k log k)
3. **One-vs-rest**: Each category as separate split — O(k) splits

**Decision**: Gradient-sorted partition (LightGBM approach).

**Rationale**:

- Provably finds optimal binary partition for convex losses
- O(k log k) is tractable for large cardinality
- Matches LightGBM's proven approach

### DD-4: Parallelization Granularity

**Context**: How to parallelize split finding.

**Options considered**:

1. **Per-node**: Each node finds splits in parallel (across nodes)
2. **Per-feature**: Within a node, evaluate features in parallel
3. **Both**: Nested parallelism

**Decision**: Per-feature parallelism within node, sequential across nodes.

**Rationale**:

- Per-feature is embarrassingly parallel (no synchronization)
- Per-node parallelism conflicts with tree growth strategy (leaf-wise needs sequential)
- Nested parallelism adds complexity with unclear benefit

## Integration

| Component | Integration Point | Notes |
|-----------|-------------------|-------|
| RFC-0012 (Histograms) | `NodeHistogram` | Input to split finder |
| RFC-0011 (Quantization) | `BinCuts` | Bin boundaries for thresholds |
| RFC-0014 (Row Partition) | Split application | Uses `SplitInfo` to partition |
| RFC-0015 (Tree Growing) | Split selection | Coordinates which nodes to split |
| RFC-0023 (Constraints) | `ConstrainedSplitFinder` | Monotonic/interaction constraints |

### Integration with Existing Code

- **`src/training/loss.rs`**: Provides gradient/hessian computation (already implemented)
- **`src/trees/node.rs`**: Existing `SplitCondition` can inform `SplitInfo` design
- **New module**: `src/training/split.rs` for `SplitInfo`, `GainParams`, `SplitFinder` trait

## Open Questions

1. **Interaction constraints**: XGBoost uses `FeatureInteractionConstraintHost` which maintains per-node allowed feature sets. When a split is made on feature F, children inherit only features that can interact with F according to constraint groups. **Implementation**: Store `node_constraints: Vec<BitSet>` tracking allowed features per node.

2. **Approximate split finding**: For very wide data (10k+ features), histogram sampling can help. **Decision**: Not needed initially — per-feature parallelism handles this well.

3. **Linear tree splits**: LightGBM's `linear_tree` mode fits a linear model at each leaf during training. Leaf value becomes `w₀ + Σ wᵢ × xᵢ` instead of constant. **Defer** to Linear Trees RFC-0021.

## Future Work

- [ ] Monotonic constraint enforcement
- [ ] Interaction constraint enforcement
- [ ] Approximate split finding for high-dimensional data
- [ ] Linear tree split integration

## References

- [XGBoost split_evaluator.h](https://github.com/dmlc/xgboost/blob/master/src/tree/split_evaluator.h)
- [LightGBM feature_histogram.hpp](https://github.com/microsoft/LightGBM/blob/master/src/treelearner/feature_histogram.hpp)
- [XGBoost paper](https://arxiv.org/abs/1603.02754) - Section 2.2 on regularized objective
- [Feature Overview](../FEATURE_OVERVIEW.md) - Priority and design context

## Changelog

- 2024-11-30: Initial draft
