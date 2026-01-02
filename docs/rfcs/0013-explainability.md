# RFC-0013: Explainability

**Status**: Implemented  
**Created**: 2025-12-20  
**Updated**: 2026-01-02  
**Scope**: Feature importance and SHAP values

## Summary

Explainability features help users understand model behavior: which features
matter (importance) and how they affect individual predictions (SHAP).

## Why Explainability?

| Need | Solution |
| ---- | -------- |
| "Which features matter?" | Feature importance |
| "Why this prediction?" | SHAP values |
| "Model debugging" | Both |
| "Regulatory compliance" | Both |

## Feature Importance

### ImportanceType Enum

```rust
pub enum ImportanceType {
    /// Number of times each feature is used in splits.
    Split,          // Default
    /// Total gain from splits using each feature.
    Gain,           // Requires gain storage
    /// Average gain per split (Gain / Split count).
    AverageGain,    // Requires gain storage
    /// Total cover (hessian sum) at nodes splitting on each feature.
    Cover,          // Requires cover storage
    /// Average cover per split (Cover / Split count).
    AverageCover,   // Requires cover storage
}
```

### FeatureImportance Container

The `FeatureImportance` struct wraps raw importance values with utility methods:

```rust
impl FeatureImportance {
    pub fn values(&self) -> &[f32];                    // Raw values
    pub fn normalized(&self) -> Self;                  // Sum to 1.0
    pub fn sorted_indices(&self) -> Vec<usize>;        // Descending order
    pub fn top_k(&self, k: usize) -> Vec<(usize, Option<String>, f32)>;
    pub fn get_by_name(&self, name: &str) -> Option<f32>;
}
```

### GBDTModel API

```rust
impl GBDTModel {
    pub fn feature_importance(
        &self,
        importance_type: ImportanceType,
    ) -> Result<FeatureImportance, ExplainError>;
}
```

Returns `ExplainError::MissingNodeStats` if gain/cover types are requested
but the model lacks those statistics.

## SHAP Values

### Shapley Value Formula

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{j\}) - f(S)]$$

### ShapValues Container

SHAP output is stored in a 3D array:

```rust
pub struct ShapValues(Array3<f32>);  // [n_samples, n_features + 1, n_outputs]

impl ShapValues {
    pub fn get(&self, sample: usize, feature: usize, output: usize) -> f32;
    pub fn base_value(&self, sample: usize, output: usize) -> f32;
    pub fn feature_shap(&self, sample_idx: usize, output: usize) -> Vec<f32>;
    pub fn verify(&self, predictions: &[f32], tolerance: f32) -> bool;
}
```

The `verify()` method checks the sum property: `sum(SHAP) + base ≈ prediction`.

## TreeSHAP

The `TreeExplainer` computes exact SHAP values for tree ensembles in
polynomial time using the algorithm from Lundberg et al. (2020).

### API

```rust
pub struct TreeExplainer<'a> {
    forest: &'a Forest<ScalarLeaf>,
    base_value: f64,
    block_size: usize,
    max_depth: usize,
}

impl<'a> TreeExplainer<'a> {
    pub fn new(forest: &'a Forest<ScalarLeaf>) -> Result<Self, ExplainError>;
    pub fn base_value(&self) -> f64;
    pub fn shap_values(&self, dataset: &Dataset, parallelism: Parallelism) -> ShapValues;
}
```

**Requires cover statistics** — returns `ExplainError::MissingNodeStats` if trees
don't have covers. Models trained with booste-rs automatically have covers.

### Algorithm

TreeSHAP tracks a path state through the tree:

1. **PathState**: Tracks features seen, zero fractions, one fractions
2. **Recursive traversal**: At each node, extend path and recurse both branches
3. **At leaves**: Compute contributions for all features in path
4. **Unwound sum**: Weight contributions by path weights

```rust
fn tree_shap(&self, tree: &Tree, sample: ArrayView1, path: &mut PathState, node: u32) {
    if tree.is_leaf(node) {
        self.compute_contributions(output, path, leaf_value);
        return;
    }
    
    // Hot path (sample's direction)
    path.extend(feature, zero_fraction, one_fraction);
    self.tree_shap(tree, sample, path, hot_child);
    path.unwind();
    
    // Cold path (other direction)
    path.extend(feature, cold_zero_fraction, 0.0);
    self.tree_shap(tree, sample, path, cold_child);
    path.unwind();
}
```

### TreeSHAP Complexity

- **Per sample**: O(T × L × D²) where T=trees, L=leaves, D=depth
- **Parallelization**: Over samples via `Parallelism::Parallel`
- **Memory**: O(D) path state per thread

### GBDTModel.shap_values

```rust
impl GBDTModel {
    pub fn shap_values(&self, data: &Dataset) -> Result<ShapValues, ExplainError>;
}
```

## Linear SHAP

For `LinearModel`, SHAP values have a closed-form solution:

$$\phi_i = w_i \times (x_i - \mu_i)$$

Where $w_i$ is the feature weight and $\mu_i$ is the background mean.

### LinearExplainer API

```rust
pub struct LinearExplainer<'a> {
    model: &'a LinearModel,
    feature_means: Vec<f64>,
}

impl<'a> LinearExplainer<'a> {
    pub fn new(model: &'a LinearModel, feature_means: Vec<f64>) -> Result<Self, ExplainError>;
    pub fn with_zero_means(model: &'a LinearModel) -> Self;
    pub fn base_value(&self, output: usize) -> f64;
    pub fn shap_values(&self, dataset: &Dataset) -> ShapValues;
}
```

### Base Value

The expected prediction given the background distribution:

$$E[f(x)] = \sum_i w_i \mu_i + \text{bias}$$

### Linear SHAP Complexity

- O(n_features × n_samples) — linear in data size
- No additional statistics required

## File Structure

| Path | Contents |
| ---- | -------- |
| `explainability/mod.rs` | Module exports |
| `explainability/importance.rs` | Feature importance types and computation |
| `explainability/shap/mod.rs` | SHAP submodule |
| `explainability/shap/values.rs` | ShapValues container |
| `explainability/shap/path.rs` | PathState for TreeSHAP |
| `explainability/shap/tree_explainer.rs` | TreeSHAP implementation |
| `explainability/shap/linear_explainer.rs` | Linear SHAP implementation |

## Design Decisions

**DD-1: Separate from inference.** Explainability is optional and adds
computation. Keep it in dedicated module, not integrated into prediction.

**DD-2: f64 for intermediate, f32 for output.** TreeSHAP uses f64 internally
for numerical stability, but stores results as f32 for consistency with
predictions and memory efficiency.

**DD-3: Base value stored per sample.** The base value is stored in
`ShapValues` at index `n_features` for each sample/output, enabling
the `verify()` method to check correctness.

**DD-4: Tree-path SHAP (not interventional).** We implement the faster
tree-path SHAP variant. Interventional SHAP is more theoretically sound
but O(2^d) expensive.

**DD-5: Require covers for TreeSHAP.** Rather than estimating covers,
we require them to be present. Models trained with booste-rs automatically
store covers during training.

**DD-6: Block-based sample buffering.** TreeExplainer uses the same
block buffering approach as prediction (buffer_samples) for cache efficiency
when converting feature-major to sample-major layout.

## Error Handling

```rust
pub enum ExplainError {
    /// Node statistics (gains/covers) are required but not available.
    MissingNodeStats(&'static str),
    /// Feature statistics (means) are required but not available.
    MissingFeatureStats,
    /// Empty model (no trees).
    EmptyModel,
}
```

## Testing Strategy

| Category | Tests |
| -------- | ----- |
| Sum property | `shap.verify(predictions, tol)` returns true |
| Splitting feature | Non-zero contribution for features that split |
| Linear exactness | Linear SHAP matches closed-form |
| Multi-output | Correct shape for multiclass |
| Missing stats | Proper error when covers unavailable |
| Importance types | All five types compute correctly |

## Future Work

- **SHAP interaction values**: Feature × feature interactions
- **Approximate SHAP**: Sampling-based for faster large-model computation
- **Examples and API polish**: Add end-to-end Python examples and stabilize the public surface (without changing the underlying algorithms)
