# RFC-0013: Explainability

- **Status**: Draft
- **Created**: 2025-12-19
- **Updated**: 2025-01-25
- **Depends on**: RFC-0002, RFC-0009, RFC-0010
- **Scope**: Feature importance and SHAP value computation

## Summary

Explainability infrastructure for boosters:

1. **Feature importance** - Static model analysis (gain, split count, cover)
2. **SHAP values** - Per-prediction feature contributions
3. **Interaction values** - Feature interaction analysis (future)

## Motivation

- **Model debugging**: Understanding why predictions are made
- **Feature selection**: Identifying important vs. redundant features
- **Regulatory compliance**: Explaining predictions (GDPR, FCRA)

## Design Overview

```
Model
  .feature_importance(type) → FeatureImportance
  .shap_values(data) → ShapValues
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
TreeExplainer  LinearExplainer  HybridExplainer
   (GBDT)       (GBLinear)     (Linear Leaves)
```

---

## Feature Importance

### Importance Types

| Type | Description |
|------|-------------|
| `Split` | Number of times feature is used in splits |
| `Gain` | Total gain from splits using this feature |
| `AverageGain` | Gain / split count |
| `Cover` | Total samples covered by feature splits |
| `AverageCover` | Cover / split count |

### Data Structures

```rust
pub enum ImportanceType { Split, Gain, AverageGain, Cover, AverageCover }

pub struct FeatureImportance {
    values: Vec<f64>,
    importance_type: ImportanceType,
    feature_names: Option<Vec<String>>,
}
```

Key methods: `get(idx)`, `get_by_name(name)`, `top_k(k)`, `normalized()`, `to_map()`

### Algorithm (Trees)

```
for each tree in forest:
    for each internal node:
        feature = node.split_feature
        split_count[feature] += 1
        gain_sum[feature] += node.gain       // requires node stats
        cover_sum[feature] += node.cover     // requires node stats
```

**Note**: Gain/cover importance requires per-node statistics (`gains`, `covers` arrays in `Tree`). Trees without stats only support `Split` importance.

### Algorithm (Linear Models)

For linear models, importance is based on coefficient magnitude:
- `Split`: Count of non-zero weights
- `Gain`: `|weight| × std(feature)` (requires feature statistics)

---

## SHAP Values

SHAP (SHapley Additive exPlanations) values explain individual predictions by computing each feature's contribution.

### Data Structures

```rust
pub struct ShapValues {
    values: Array3<f64>,  // [n_samples, n_features + 1, n_outputs]
    // Last feature slot stores base value (expected output)
}
```

Key methods: `get(sample, feature, output)`, `base_value(sample, output)`, `verify(predictions, tolerance)`

### TreeSHAP Algorithm

TreeSHAP computes exact SHAP values in polynomial time by tracking contributions through tree paths.

**Complexity**: O(TLD²) per sample where T = trees, L = leaves, D = depth.

**Key insight**: At each node, track two quantities:
- `one_fraction`: Probability of reaching node when feature is in coalition
- `zero_fraction`: Probability when feature is NOT in coalition

```
function tree_shap(node, path_state):
    if node is leaf:
        compute_contributions(path_state, leaf_value)
        return
    
    feature = node.split_feature
    hot_child = child sample goes to
    cold_child = other child
    
    // Recurse hot path (feature in coalition)
    path_state.extend(feature, zero_frac=cover_ratio, one_frac=1.0)
    tree_shap(hot_child, path_state)
    path_state.unwind()
    
    // Recurse cold path (feature not in coalition)  
    path_state.extend(feature, zero_frac=cold_cover_ratio, one_frac=0.0)
    tree_shap(cold_child, path_state)
    path_state.unwind()
```

**Requirement**: TreeSHAP needs per-node `cover` values to compute path fractions. Models without covers will return an error.

### Linear SHAP

For linear models, SHAP has a closed form:

```
φᵢ = wᵢ × (xᵢ - E[xᵢ])
```

Requires feature means from training data.

### Hybrid SHAP (Linear Leaves)

For trees with linear leaves:
1. Compute tree path contributions via TreeSHAP
2. At each leaf, add linear term contributions: `wᵢ × (xᵢ - mean)`

---

## Missing Values

**Feature Importance**: Missing values don't affect importance counting. Trees still use features for splits regardless of missing values.

**SHAP Values**: Missing features use `default_left` direction from training. SHAP contribution is based on which path is taken.

---

## Error Handling

```rust
pub enum ExplainError {
    MissingNodeStats(&'static str),  // Need gains/covers
    MissingFeatureStats,              // Need feature means for linear SHAP
    EmptyModel,
}
```

---

## Usage Examples

### Python

```python
# Feature importance
importance = model.feature_importance("gain")
for name, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
    print(f"{name}: {score:.4f}")

# SHAP values (compatible with shap library)
shap_values = model.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### Rust

```rust
let importance = model.feature_importance(ImportanceType::Gain)?;
for (idx, score) in importance.top_k(10) {
    println!("{}: {:.4}", idx, score);
}

let explainer = TreeExplainer::new(&forest)?;
let shap = explainer.shap_values(features_view);
assert!(shap.verify(&predictions, 1e-6));
```

---

## Design Decisions

### DD-1: f64 for SHAP Accumulators

SHAP involves many additions that accumulate error. Use f64 internally.

### DD-2: TreeSHAP as Default

TreeSHAP is O(TLD²) and leverages tree structure. Interventional SHAP available as option for causal interpretation.

### DD-3: Embedded Node Stats

Store optional `gains` and `covers` in `Tree`. Populated during training or model loading.

---

## Integration

| Component | Integration Point |
|-----------|------------------|
| RFC-0002 (Trees) | `Tree::gains()`, `Tree::covers()` |
| RFC-0009 (GBLinear) | `LinearModel` weight access |
| RFC-0010 (Linear Leaves) | Hybrid SHAP for linear terms |

## Future Work

- GPU-accelerated SHAP
- SHAP interaction values
- Approximate SHAP (sampling-based)
- Permutation importance

## References

- Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
- Lundberg et al. (2018). "Consistent Individualized Feature Attribution for Tree Ensembles"

## Changelog

- 2025-01-25: Simplified RFC - removed excessive implementation detail, added pseudocode
- 2025-12-19: Initial draft
