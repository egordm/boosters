# RFC-0020: Explainability

- **Status**: Draft
- **Created**: 2025-12-19
- **Updated**: 2025-12-19
- **Depends on**: RFC-0002, RFC-0014, RFC-0015, RFC-0019
- **Scope**: Feature importance and SHAP value computation

## Summary

This RFC defines the explainability infrastructure for boosters, providing:

1. **Feature importance** - Static model analysis (gain, split count, cover)
2. **SHAP values** - Per-prediction feature contributions
3. **Interaction values** - Feature interaction analysis (future)

Support is provided for all model types: GBDT, GBLinear, GBDT with linear leaves, and categorical features.

## Motivation

### Use Cases

1. **Model debugging** - Understanding why a model makes certain predictions
2. **Feature selection** - Identifying important vs. redundant features
3. **Regulatory compliance** - Explaining individual predictions (GDPR, FCRA)
4. **Trust building** - Helping users understand model behavior

### Current Gap

boosters has complete training and inference but no explainability. Users must:

- Export to XGBoost/LightGBM format for SHAP
- Implement feature importance from scratch
- Cannot explain linear leaves or native categoricals

### Goals

1. **Unified API** - Same interface for all model types
2. **Fast** - Optimized implementations (CPU first, GPU later)
3. **Complete** - Handle all features: categoricals, linear leaves, missing values
4. **Compatible** - Output matches XGBoost/SHAP library format

## Design

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       Explainability API                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Model                                                        ││
│  │  .feature_importance(type) → ImportanceMap                   ││
│  │  .shap_values(data) → ShapValues                            ││
│  │  .shap_interaction_values(data) → InteractionMatrix (future)││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ TreeExplainer   │  │ LinearExplainer │  │ HybridExplainer │
│ (GBDT)          │  │ (GBLinear)      │  │ (Linear Leaves) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Data Structures

#### Feature Importance

```rust
/// Types of feature importance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportanceType {
    /// Number of times feature is used in splits.
    /// For linear models: non-zero weights count.
    Split,
    
    /// Total gain from splits using this feature.
    /// For linear models: |weight| * std(feature).
    Gain,
    
    /// Average gain per split.
    AverageGain,
    
    /// Total samples covered by splits on this feature.
    Cover,
    
    /// Average samples per split.
    AverageCover,
}

/// Feature importance result.
#[derive(Debug, Clone)]
pub struct FeatureImportance {
    /// Importance values indexed by feature index.
    values: Vec<f64>,
    /// Importance type used.
    importance_type: ImportanceType,
    /// Feature names (if available).
    feature_names: Option<Vec<String>>,
}

impl FeatureImportance {
    /// Get importance by feature index.
    pub fn get(&self, feature_idx: usize) -> f64 {
        self.values.get(feature_idx).copied().unwrap_or(0.0)
    }

    /// Get importance by feature name.
    pub fn get_by_name(&self, name: &str) -> Option<f64> {
        self.feature_names.as_ref().and_then(|names| {
            names.iter().position(|n| n == name).map(|i| self.values[i])
        })
    }

    /// Get sorted (feature_idx, importance) pairs, descending.
    pub fn sorted(&self) -> Vec<(usize, f64)> {
        let mut pairs: Vec<_> = self.values.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs
    }

    /// Get top-k features by importance.
    pub fn top_k(&self, k: usize) -> Vec<(usize, f64)> {
        self.sorted().into_iter().take(k).collect()
    }

    /// Convert to HashMap (feature_name → importance).
    pub fn to_map(&self) -> HashMap<String, f64> {
        let names = self.feature_names.clone()
            .unwrap_or_else(|| (0..self.values.len()).map(|i| format!("f{}", i)).collect());
        names.into_iter().zip(self.values.iter().copied()).collect()
    }
}
```

#### SHAP Values

```rust
/// SHAP values for a batch of predictions.
///
/// Shape: `[n_samples, n_features + 1, n_outputs]`
/// The last feature dimension is the base value (expected output).
#[derive(Debug, Clone)]
pub struct ShapValues {
    /// Flat storage in row-major order.
    /// Layout: [sample, feature, output]
    values: Vec<f64>,
    /// Number of samples.
    n_samples: usize,
    /// Number of features (not including base value).
    n_features: usize,
    /// Number of outputs.
    n_outputs: usize,
}

impl ShapValues {
    /// Get SHAP value for a specific sample, feature, and output.
    pub fn get(&self, sample: usize, feature: usize, output: usize) -> f64 {
        let idx = (sample * (self.n_features + 1) + feature) * self.n_outputs + output;
        self.values[idx]
    }

    /// Get base value (expected output) for a sample and output.
    pub fn base_value(&self, sample: usize, output: usize) -> f64 {
        self.get(sample, self.n_features, output)
    }

    /// Get all SHAP values for a single sample.
    /// Returns `[n_features + 1, n_outputs]` slice.
    pub fn sample(&self, sample: usize) -> &[f64] {
        let start = sample * (self.n_features + 1) * self.n_outputs;
        let end = start + (self.n_features + 1) * self.n_outputs;
        &self.values[start..end]
    }

    /// Verify SHAP values sum to prediction.
    /// 
    /// For each sample: sum(shap_values) ≈ prediction
    pub fn verify(&self, predictions: &[f64], tolerance: f64) -> bool {
        for sample in 0..self.n_samples {
            for output in 0..self.n_outputs {
                let sum: f64 = (0..=self.n_features)
                    .map(|f| self.get(sample, f, output))
                    .sum();
                let pred = predictions[sample * self.n_outputs + output];
                if (sum - pred).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Convert to 3D array (for Python numpy).
    pub fn to_3d_array(&self) -> Vec<Vec<Vec<f64>>> {
        (0..self.n_samples)
            .map(|s| {
                (0..=self.n_features)
                    .map(|f| {
                        (0..self.n_outputs)
                            .map(|o| self.get(s, f, o))
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}
```

### Algorithms

#### Feature Importance for Trees

**Note on Node Statistics**: The current `Tree` structure in boosters stores split information (feature, threshold, children) but does NOT store per-node `gain` and `cover` values. This is a design decision to minimize memory for inference. 

To support gain-based and cover-based importance, we have two options:

1. **Extend Tree struct** - Add optional `gains: Box<[f32]>` and `covers: Box<[f32]>` to `Tree`
2. **Compute on-demand** - For split count (always available), and defer gain/cover to format conversion

**Recommended approach**: Add optional statistics that can be populated during training or model loading (XGBoost/LightGBM models include these values).

```rust
/// Extended tree with optional statistics for explainability.
impl<L: LeafValue> Tree<L> {
    /// Optional per-node gain values (populated during training or import).
    gains: Option<Box<[f32]>>,
    /// Optional per-node cover values (sum of hessians).
    covers: Option<Box<[f32]>>,
}

impl Forest<ScalarLeaf> {
    /// Compute feature importance by traversing all trees.
    /// 
    /// # Errors
    /// Returns `ExplainError::MissingNodeStats` if gain/cover importance is
    /// requested but node statistics are not available.
    pub fn feature_importance(
        &self, 
        imp_type: ImportanceType
    ) -> Result<FeatureImportance, ExplainError> {
        let n_features = self.max_feature_idx() + 1;
        let mut gain_sum = vec![0.0f64; n_features];
        let mut split_count = vec![0u64; n_features];
        let mut cover_sum = vec![0.0f64; n_features];

        // Check if we need node statistics
        let needs_gain = matches!(imp_type, 
            ImportanceType::Gain | ImportanceType::AverageGain);
        let needs_cover = matches!(imp_type, 
            ImportanceType::Cover | ImportanceType::AverageCover);

        for tree in self.trees() {
            if needs_gain && tree.gains.is_none() {
                return Err(ExplainError::MissingNodeStats("gain"));
            }
            if needs_cover && tree.covers.is_none() {
                return Err(ExplainError::MissingNodeStats("cover"));
            }
            
            Self::accumulate_importance(
                tree,
                &mut gain_sum,
                &mut split_count,
                &mut cover_sum,
            );
        }

        let values = match imp_type {
            ImportanceType::Split => {
                split_count.iter().map(|&c| c as f64).collect()
            }
            ImportanceType::Gain => gain_sum,
            ImportanceType::AverageGain => {
                gain_sum.iter().zip(&split_count)
                    .map(|(&g, &c)| if c > 0 { g / c as f64 } else { 0.0 })
                    .collect()
            }
            ImportanceType::Cover => cover_sum,
            ImportanceType::AverageCover => {
                cover_sum.iter().zip(&split_count)
                    .map(|(&c, &n)| if n > 0 { c / n as f64 } else { 0.0 })
                    .collect()
            }
        };

        Ok(FeatureImportance {
            values,
            importance_type: imp_type,
            feature_names: None,
        })
    }

    fn accumulate_importance(
        tree: &Tree<ScalarLeaf>,
        gain_sum: &mut [f64],
        split_count: &mut [u64],
        cover_sum: &mut [f64],
    ) {
        for node_idx in 0..tree.n_nodes() {
            if tree.is_leaf(node_idx as u32) {
                continue;
            }
            
            let feature = tree.split_index(node_idx as u32) as usize;
            
            if feature < gain_sum.len() {
                split_count[feature] += 1;
                
                // Gain and cover are optional
                if let Some(gains) = &tree.gains {
                    gain_sum[feature] += gains[node_idx] as f64;
                }
                if let Some(covers) = &tree.covers {
                    cover_sum[feature] += covers[node_idx] as f64;
                }
            }
        }
    }
}
```

**Training Integration**: During tree building, the `TreeGrower` already computes `SplitInfo::gain` and tracks hessian sums. To enable gain/cover importance:

```rust
// In TreeBuilder::apply_split(), store the gain and cover:
impl TreeBuilder {
    pub fn apply_split_with_stats(
        &mut self,
        node: NodeId,
        split: &SplitInfo,
        hess_sum: f64,  // Cover value
        depth: u16,
    ) -> (NodeId, NodeId) {
        let (left, right) = self.apply_split(node, split, depth);
        
        // Store statistics
        self.gains.push((node, split.gain));
        self.covers.push((node, hess_sum as f32));
        
        (left, right)
    }
}
```

#### Feature Importance for Linear Models

```rust
impl LinearModel {
    /// Compute feature importance for linear model.
    ///
    /// For linear models, importance is based on coefficient magnitude.
    /// If feature statistics are provided, uses standardized importance.
    pub fn feature_importance(
        &self,
        imp_type: ImportanceType,
        feature_stats: Option<&FeatureStats>,
    ) -> FeatureImportance {
        let values: Vec<f64> = match imp_type {
            ImportanceType::Split => {
                // Count of non-zero weights per feature
                (0..self.n_features())
                    .map(|f| {
                        let non_zero = (0..self.n_groups())
                            .filter(|&g| self.weight(f, g).abs() > 1e-10)
                            .count();
                        non_zero as f64
                    })
                    .collect()
            }
            ImportanceType::Gain | ImportanceType::AverageGain => {
                // |weight| or |weight| * std(feature)
                (0..self.n_features())
                    .map(|f| {
                        let w_sum: f64 = (0..self.n_groups())
                            .map(|g| self.weight(f, g).abs() as f64)
                            .sum();
                        
                        if let Some(stats) = feature_stats {
                            w_sum * stats.std(f)
                        } else {
                            w_sum
                        }
                    })
                    .collect()
            }
            _ => {
                // Cover not meaningful for linear models
                vec![0.0; self.n_features()]
            }
        };

        FeatureImportance {
            values,
            importance_type: imp_type,
            feature_names: None,
        }
    }
}
```

#### TreeSHAP Algorithm

The TreeSHAP algorithm computes exact SHAP values by tracking contribution through tree paths.

**Important**: TreeSHAP requires per-node `cover` values (sum of hessians for samples reaching that node) to compute the zero_fraction for each child. This is why the Tree struct needs optional `covers: Option<Box<[f32]>>` as described in the Feature Importance section above.

If cover values are not available, SHAP computation will return `ExplainError::MissingNodeStats("cover")`.

```rust
/// TreeSHAP explainer for tree ensembles.
pub struct TreeExplainer<'a> {
    forest: &'a Forest<ScalarLeaf>,
    /// Background data for interventional SHAP (optional)
    background: Option<&'a [f32]>,
}

impl<'a> TreeExplainer<'a> {
    pub fn new(forest: &'a Forest<ScalarLeaf>) -> Self {
        Self { forest, background: None }
    }

    /// Set background data for interventional SHAP.
    pub fn with_background(mut self, data: &'a [f32]) -> Self {
        self.background = Some(data);
        self
    }

    /// Compute SHAP values for a batch of samples.
    pub fn shap_values<M: DataMatrix<Element = f32>>(
        &self,
        data: &M,
    ) -> ShapValues {
        let n_samples = data.num_rows();
        let n_features = data.num_features();
        let n_outputs = self.forest.n_groups() as usize;

        let mut values = vec![0.0f64; n_samples * (n_features + 1) * n_outputs];

        // Process each sample
        for sample_idx in 0..n_samples {
            self.shap_for_sample(data, sample_idx, &mut values, n_features, n_outputs);
        }

        ShapValues {
            values,
            n_samples,
            n_features,
            n_outputs,
        }
    }

    fn shap_for_sample<M: DataMatrix<Element = f32>>(
        &self,
        data: &M,
        sample_idx: usize,
        values: &mut [f64],
        n_features: usize,
        n_outputs: usize,
    ) {
        // Add base values
        let base_offset = sample_idx * (n_features + 1) * n_outputs + n_features * n_outputs;
        for (g, &base) in self.forest.base_score().iter().enumerate() {
            values[base_offset + g] = base as f64;
        }

        // Process each tree
        for (tree, group) in self.forest.trees_with_groups() {
            let group_idx = group as usize;
            let sample_offset = sample_idx * (n_features + 1) * n_outputs;

            // Initialize path
            let mut path = PathState::new(n_features);

            // Recursive tree traversal with SHAP computation
            self.tree_shap_recursive(
                tree,
                data,
                sample_idx,
                0, // root node
                &mut path,
                1.0, // one_fraction
                1.0, // zero_fraction
                -1,  // parent feature
                &mut values[sample_offset..],
                n_features,
                group_idx,
                n_outputs,
            );
        }
    }

    fn tree_shap_recursive<M: DataMatrix<Element = f32>>(
        &self,
        tree: &Tree<ScalarLeaf>,
        data: &M,
        sample_idx: usize,
        node_idx: u32,
        path: &mut PathState,
        one_fraction: f64,
        zero_fraction: f64,
        parent_feature_idx: i32,
        values: &mut [f64],
        n_features: usize,
        group_idx: usize,
        n_outputs: usize,
    ) {
        // Extend path with current node
        path.extend(parent_feature_idx, zero_fraction, one_fraction);

        if tree.is_leaf(node_idx) {
            // Leaf node: compute contributions
            let leaf_value = tree.leaf_value(node_idx).0 as f64;
            
            // Handle linear leaf if present
            let linear_contrib = if tree.has_linear_leaves() {
                self.linear_leaf_shap(tree, node_idx, data, sample_idx)
            } else {
                vec![0.0; n_features]
            };

            // Compute SHAP contributions from path
            for i in 1..path.len() {
                let feature = path.feature(i);
                if feature >= 0 {
                    let f = feature as usize;
                    let contrib = leaf_value * path.unwound_sum(i);
                    let total_contrib = contrib + linear_contrib.get(f).copied().unwrap_or(0.0);
                    values[f * n_outputs + group_idx] += total_contrib;
                }
            }
        } else {
            // Internal node: recurse
            let node = tree.node(node_idx);
            let feature = node.feature() as usize;
            let feature_val = data.get(sample_idx, feature).unwrap_or(f32::NAN);

            let (hot_idx, cold_idx) = if node.goes_left(feature_val) {
                (node.left_child(), node.right_child())
            } else {
                (node.right_child(), node.left_child())
            };

            // Compute fractions based on node covers
            let hot_cover = tree.cover(hot_idx);
            let cold_cover = tree.cover(cold_idx);
            let parent_cover = hot_cover + cold_cover;

            let hot_zero_fraction = hot_cover as f64 / parent_cover as f64;
            let cold_zero_fraction = cold_cover as f64 / parent_cover as f64;

            // Recurse into hot path (path taken by this sample)
            self.tree_shap_recursive(
                tree, data, sample_idx, hot_idx,
                path, 1.0, hot_zero_fraction,
                feature as i32,
                values, n_features, group_idx, n_outputs,
            );

            // Recurse into cold path (path not taken)
            self.tree_shap_recursive(
                tree, data, sample_idx, cold_idx,
                path, 0.0, cold_zero_fraction,
                feature as i32,
                values, n_features, group_idx, n_outputs,
            );
        }

        // Unwind path
        path.unwind();
    }

    /// Compute SHAP values for linear terms in a leaf.
    fn linear_leaf_shap<M: DataMatrix<Element = f32>>(
        &self,
        tree: &Tree<ScalarLeaf>,
        leaf_idx: u32,
        data: &M,
        sample_idx: usize,
    ) -> Vec<f64> {
        let n_features = data.num_features();
        let mut contrib = vec![0.0; n_features];

        if let Some((feat_indices, coeffs)) = tree.leaf_terms(leaf_idx) {
            // For linear models: SHAP = coef * (x - E[x])
            // We need feature means - either from background or assume 0
            for (&f, &c) in feat_indices.iter().zip(coeffs.iter()) {
                let f = f as usize;
                let x = data.get(sample_idx, f).unwrap_or(0.0) as f64;
                let mean = self.background
                    .map(|bg| bg[f] as f64)  // Use background mean
                    .unwrap_or(0.0);         // Or assume 0
                
                contrib[f] = c as f64 * (x - mean);
            }
        }

        contrib
    }
}

/// Path state tracking for TreeSHAP algorithm.
struct PathState {
    /// Feature indices along the path (-1 for root)
    features: Vec<i32>,
    /// Zero fractions (P(path | feature unknown))
    zero_fractions: Vec<f64>,
    /// One fractions (P(path | feature known))
    one_fractions: Vec<f64>,
    /// Partial weights for SHAP computation
    weights: Vec<f64>,
}

impl PathState {
    fn new(n_features: usize) -> Self {
        Self {
            features: Vec::with_capacity(n_features),
            zero_fractions: Vec::with_capacity(n_features),
            one_fractions: Vec::with_capacity(n_features),
            weights: Vec::with_capacity(n_features),
        }
    }

    fn len(&self) -> usize {
        self.features.len()
    }

    fn feature(&self, i: usize) -> i32 {
        self.features[i]
    }

    fn extend(&mut self, feature: i32, zero_fraction: f64, one_fraction: f64) {
        self.features.push(feature);
        self.zero_fractions.push(zero_fraction);
        self.one_fractions.push(one_fraction);
        
        // Update weights using the SHAP path algorithm
        let n = self.features.len();
        self.weights.push(if n == 1 { 1.0 } else { 0.0 });
        
        for i in (1..n).rev() {
            let w = self.weights[i];
            self.weights[i] = w * (i as f64 / n as f64) * one_fraction
                + self.weights[i - 1] * ((n - i) as f64 / n as f64) * zero_fraction;
        }
        self.weights[0] *= zero_fraction;
    }

    fn unwind(&mut self) {
        self.features.pop();
        self.zero_fractions.pop();
        self.one_fractions.pop();
        self.weights.pop();
    }

    fn unwound_sum(&self, target_idx: usize) -> f64 {
        // Compute contribution for unwinding feature at target_idx
        // This is the core SHAP computation
        let n = self.features.len();
        if n == 0 {
            return 0.0;
        }

        let one_fraction = self.one_fractions[target_idx];
        let zero_fraction = self.zero_fractions[target_idx];

        let mut total = 0.0;
        for i in 0..n {
            let next = if i + 1 < n {
                self.weights[i + 1]
            } else {
                0.0
            };
            
            let w = if one_fraction != 0.0 {
                (self.weights[i] - next * zero_fraction) / one_fraction
            } else {
                next * (n as f64 - i as f64) / (i as f64 + 1.0)
            };
            
            total += w * (one_fraction - zero_fraction);
        }
        total
    }
}
```

#### Linear SHAP

```rust
/// Explainer for linear models.
pub struct LinearExplainer<'a> {
    model: &'a LinearModel,
    /// Feature means from training data
    feature_means: Vec<f64>,
}

impl<'a> LinearExplainer<'a> {
    pub fn new(model: &'a LinearModel, feature_means: Vec<f64>) -> Self {
        debug_assert_eq!(feature_means.len(), model.n_features());
        Self { model, feature_means }
    }

    /// Compute SHAP values for linear model.
    ///
    /// For linear models, SHAP values have a closed form:
    /// φ_i = w_i * (x_i - E[x_i])
    pub fn shap_values<M: DataMatrix<Element = f32>>(&self, data: &M) -> ShapValues {
        let n_samples = data.num_rows();
        let n_features = self.model.n_features();
        let n_groups = self.model.n_groups();

        let mut values = vec![0.0f64; n_samples * (n_features + 1) * n_groups];

        for sample in 0..n_samples {
            let sample_offset = sample * (n_features + 1) * n_groups;

            for feature in 0..n_features {
                let x = data.get(sample, feature).unwrap_or(0.0) as f64;
                let mean = self.feature_means[feature];
                let deviation = x - mean;

                for group in 0..n_groups {
                    let w = self.model.weight(feature, group) as f64;
                    let contrib = w * deviation;
                    values[sample_offset + feature * n_groups + group] = contrib;
                }
            }

            // Base value = E[prediction] = sum(w_i * mean_i) + bias
            for group in 0..n_groups {
                let base: f64 = (0..n_features)
                    .map(|f| self.model.weight(f, group) as f64 * self.feature_means[f])
                    .sum::<f64>()
                    + self.model.bias(group) as f64;
                
                values[sample_offset + n_features * n_groups + group] = base;
            }
        }

        ShapValues {
            values,
            n_samples,
            n_features,
            n_outputs: n_groups,
        }
    }
}
```

### Missing Values Handling

Missing values (NaN) require special handling in both feature importance and SHAP:

**For Feature Importance:**

Missing values don't directly affect split-based importance counting. When a feature has missing values during training:

- The tree still uses the feature for splits
- `default_left` determines which direction missing values go
- Split count and gain are tracked as normal

**For SHAP Values:**

For SHAP computation, missing values use the `default_left` direction:

```rust
// When feature value is NaN, use default direction
let goes_left = if feature_val.is_nan() {
    tree.default_left(node_idx)
} else {
    // Normal comparison
    feature_val < threshold
};
```

The SHAP contribution for a missing feature is based on which path it takes (hot/cold) according to the default direction learned during training.

**Key point**: Missing values still have SHAP contributions - they contribute based on the default path behavior, not zero contribution.

### Categorical Feature Handling

```rust
/// Options for handling categorical features in explanations.
pub enum CategoricalExplanation {
    /// Report importance/SHAP per category (one-hot style)
    PerCategory,
    /// Aggregate to single value per categorical feature
    Aggregated,
}

impl FeatureImportance {
    /// Aggregate categorical feature importance.
    pub fn aggregate_categorical(
        &self,
        cat_features: &[(usize, Vec<usize>)], // (original_idx, category_indices)
    ) -> FeatureImportance {
        let mut new_values = self.values.clone();
        
        for (original_idx, category_indices) in cat_features {
            // Sum importance across categories
            let total: f64 = category_indices.iter()
                .map(|&i| self.values.get(i).copied().unwrap_or(0.0))
                .sum();
            
            // Set aggregated value at original index
            if *original_idx < new_values.len() {
                new_values[*original_idx] = total;
            }
            
            // Zero out individual category values
            for &cat_idx in category_indices {
                if cat_idx < new_values.len() && cat_idx != *original_idx {
                    new_values[cat_idx] = 0.0;
                }
            }
        }
        
        FeatureImportance {
            values: new_values,
            importance_type: self.importance_type,
            feature_names: self.feature_names.clone(),
        }
    }
}
```

### Model API Integration

```rust
impl Model {
    /// Compute feature importance.
    pub fn feature_importance(&self, imp_type: ImportanceType) -> FeatureImportance {
        let mut importance = match &self.core {
            ModelCore::GBDT { forest, .. } => {
                forest.feature_importance(imp_type)
            }
            ModelCore::GBLinear { model } => {
                model.feature_importance(imp_type, self.feature_stats.as_ref())
            }
        };
        
        importance.feature_names = self.meta.feature_names.clone();
        importance
    }

    /// Compute SHAP values for given data.
    ///
    /// Returns `[n_samples, n_features + 1, n_outputs]` where the last
    /// feature dimension is the base value.
    pub fn shap_values<M: DataMatrix<Element = f32>>(
        &self,
        data: &M,
    ) -> Result<ShapValues, ExplainError> {
        match &self.core {
            ModelCore::GBDT { forest, .. } => {
                let explainer = TreeExplainer::new(forest);
                Ok(explainer.shap_values(data))
            }
            ModelCore::GBLinear { model } => {
                let means = self.feature_stats
                    .as_ref()
                    .map(|s| s.means().to_vec())
                    .ok_or(ExplainError::MissingFeatureStats)?;
                
                let explainer = LinearExplainer::new(model, means);
                Ok(explainer.shap_values(data))
            }
        }
    }

    /// Compute SHAP values with options.
    pub fn shap_values_with_options<M: DataMatrix<Element = f32>>(
        &self,
        data: &M,
        options: ShapOptions,
    ) -> Result<ShapValues, ExplainError> {
        /* ... */
    }
}

/// SHAP computation options.
pub struct ShapOptions {
    /// Method for handling conditional vs interventional
    pub method: ShapMethod,
    /// How to handle categorical features
    pub categorical: CategoricalExplanation,
    /// Approximate computation for speed (future)
    pub approximate: bool,
    /// Number of threads (0 = auto)
    pub n_threads: usize,
}

/// SHAP computation method.
pub enum ShapMethod {
    /// TreeSHAP (conditional, uses tree structure)
    TreePath,
    /// Interventional (treats features as independent)
    Interventional { background_samples: usize },
}
```

### Error Types

```rust
/// Errors during explainability computation.
#[derive(Debug, thiserror::Error)]
pub enum ExplainError {
    #[error("Feature statistics required for linear SHAP but not available")]
    MissingFeatureStats,
    
    #[error("Node statistics ({0}) required but not available - model may need re-training or import with stats")]
    MissingNodeStats(&'static str),
    
    #[error("Background data required for interventional SHAP")]
    MissingBackground,
    
    #[error("Feature count mismatch: model has {expected}, data has {got}")]
    FeatureMismatch { expected: usize, got: usize },
    
    #[error("Invalid importance type {0:?} for this model type")]
    InvalidImportanceType(ImportanceType),
}
```

## Design Decisions

### DD-1: f64 for SHAP Accumulators

**Context**: Should SHAP values use f32 or f64 precision?

**Options**:

1. **f32** - Faster, smaller memory footprint
2. **f64** - More accurate accumulation

**Decision**: f64 for internal computation, convert to f32 for output.

**Rationale**: SHAP values involve many additions that can accumulate error. XGBoost uses f64 internally as well.

### DD-2: TreeSHAP vs Interventional Default

**Context**: Which SHAP method should be the default?

**Options**:

1. **TreeSHAP (conditional)** - Fast, leverages tree structure
2. **Interventional** - More causally meaningful

**Decision**: TreeSHAP as default, interventional as option.

**Rationale**: 
- TreeSHAP is O(TLD²) vs O(TLD² × 2^M) for interventional
- Most users expect TreeSHAP (matches `shap` library)
- Interventional available via `ShapOptions`

### DD-3: Linear Leaf SHAP Integration

**Context**: How to handle SHAP for trees with linear leaves?

**Options**:

1. **Ignore linear terms** - Treat as constant leaf
2. **Hybrid** - Tree path + linear SHAP
3. **Fully linear** - Decompose everything to linear contributions

**Decision**: Hybrid approach:
- Tree path contribution computed normally
- Linear terms within reached leaf add: `w_i * (x_i - E[x_i])`

**Rationale**: Captures both tree structure decisions and linear fine-tuning.

### DD-4: Feature Statistics Storage

**Context**: Linear SHAP needs feature means. Where to store?

**Options**:

1. **In Model** - Store FeatureStats with model
2. **User provides** - Require background data
3. **Optional** - Store if available, require otherwise

**Decision**: Store in Model (optional), provide during training:
- `BinnedDatasetBuilder` can compute statistics
- Store in `Model::feature_stats`
- Error if needed but not available

## Integration

| Component | Integration Point | Notes |
|-----------|------------------|-------|
| RFC-0002 (Forest) | `Forest::feature_importance()` | Core method |
| RFC-0014 (GBLinear) | `LinearModel::feature_importance()` | Weight-based |
| RFC-0015 (Linear Leaves) | `TreeExplainer::linear_leaf_shap()` | Hybrid method |
| RFC-0019 (Python) | `Booster.get_score()`, `predict_contributions()` | Python API |
| RFC-0018 (Categoricals) | `aggregate_categorical()` | Grouping |

## Open Questions

1. **Parallel SHAP**: How to efficiently parallelize TreeSHAP?
   - Per-sample parallelism (simple, good for many samples)
   - Per-tree parallelism (more complex, good for large forests)
   - **Tentative**: Per-sample with rayon

2. **Approximate SHAP**: Should we support sampling-based approximation?
   - Pro: Much faster for large models
   - Con: Adds complexity, less accurate
   - **Tentative**: Future work, not in initial implementation

3. **Interaction values**: Implementation priority?
   - Full interaction matrix is O(TLD² × M²)
   - **Tentative**: Defer to post-1.0

## Future Work

- [ ] GPU-accelerated SHAP (GPUTreeSHAP integration)
- [ ] SHAP interaction values
- [ ] Approximate SHAP (sampling-based)
- [ ] Permutation importance
- [ ] Partial dependence plots
- [ ] Global surrogate explanations

## References

- [Research: Explainability](../research/explainability.md)
- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- Lundberg et al. (2018). "Consistent Individualized Feature Attribution for Tree Ensembles"
- [XGBoost SHAP](https://github.com/dmlc/xgboost/blob/master/src/tree/tree_shap.cc)
- [GPUTreeSHAP](https://github.com/rapidsai/gputreeshap)
- [SHAP library](https://github.com/slundberg/shap)

## Changelog

- 2025-12-19: Initial draft
