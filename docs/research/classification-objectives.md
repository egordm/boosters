# Classification Objective Research

Comparing binary and multiclass classification objectives between booste-rs, LightGBM, and XGBoost to identify differences that may explain performance gaps on small datasets.

## Key Findings Summary

| Factor | booste-rs | LightGBM | XGBoost | Impact |
|--------|-----------|----------|---------|--------|
| `min_data_in_leaf` default | 1 | **20** | 1 | Large on small datasets |
| `min_child_weight` default | 1.0 | 1e-3 | 1.0 | Minimal |
| Binary label format | {0, 1} | **{-1, +1}** | {0, 1} | Different gradient formula |
| Binary sigmoid param | N/A | 1.0 (configurable) | N/A | Gradient scaling |
| Multiclass hessian factor | 1.0 | **K/(K-1)** | **2.0** | Affects step size |
| Class imbalance handling | None | `is_unbalance`, `scale_pos_weight` | `scale_pos_weight` | Important for skewed classes |

## Detailed Analysis

### 1. min_data_in_leaf Default Difference (HIGH IMPACT)

**LightGBM**: `min_data_in_leaf = 20` by default
**booste-rs**: `min_samples_leaf = 1` by default

This is the most significant difference for small datasets like iris (150 samples) and breast_cancer (569 samples). LightGBM's higher default prevents overfitting by requiring at least 20 samples per leaf.

**Recommendation**: Consider increasing default to 10-20 for better out-of-box performance on small datasets.

### 2. Binary Classification: Label Format & Gradient Formula

**LightGBM uses {-1, +1} labels** with a scaled sigmoid:

```cpp
// LightGBM binary gradient (label ∈ {-1, +1})
const double response = -label * sigmoid_ / (1.0 + exp(label * sigmoid_ * score));
const double abs_response = fabs(response);
gradient = response * label_weight;
hessian = abs_response * (sigmoid_ - abs_response) * label_weight;
```

This is mathematically equivalent to the {0, 1} formulation but may have different numerical properties.

**booste-rs and XGBoost use {0, 1} labels**:

```rust
// booste-rs/XGBoost binary gradient (label ∈ {0, 1})
let p = sigmoid(pred);
grad = p - label;
hess = p * (1 - p);
```

### 3. Multiclass Hessian Scaling (MEDIUM IMPACT)

The second-order gradient (hessian) for softmax cross-entropy should technically be a matrix, but libraries use a diagonal approximation. The scaling factor differs:

**LightGBM**:
```cpp
factor_ = num_class_ / (num_class_ - 1.0);  // 1.5 for 3 classes
hessians[idx] = factor_ * p * (1.0f - p);
```

**XGBoost**:
```cpp
float h = 2.0f * p * (1.0f - p);  // constant factor of 2
```

**booste-rs**:
```rust
let hess = p * (1.0 - p);  // no factor
```

The factor affects the Newton step size: larger hessian → smaller steps → more conservative updates.

| n_classes | LightGBM factor | XGBoost factor | booste-rs factor |
|-----------|-----------------|----------------|------------------|
| 2 | 2.0 | 2.0 | 1.0 |
| 3 | 1.5 | 2.0 | 1.0 |
| 5 | 1.25 | 2.0 | 1.0 |
| 10 | 1.11 | 2.0 | 1.0 |

This means booste-rs takes **2x larger steps** than XGBoost and **1.5-2x larger steps** than LightGBM for multiclass. On small datasets, this could lead to overfitting.

### 4. Class Imbalance Handling

**LightGBM** has sophisticated imbalance handling:
- `is_unbalance`: Automatically reweights classes
- `scale_pos_weight`: Manual positive class scaling
- `pos_bagging_fraction` / `neg_bagging_fraction`: Balanced sampling

**XGBoost**:
- `scale_pos_weight`: Manual positive class scaling

**booste-rs**:
- User must provide custom weights (no automatic handling)

For datasets like breast_cancer with class imbalance, LightGBM's automatic handling could improve performance.

### 5. BoostFromScore / Base Score

Both LightGBM and XGBoost compute sophisticated initial predictions:

**LightGBM binary** (`BoostFromScore`):
```cpp
double pavg = suml / sumw;  // weighted mean
double initscore = log(pavg / (1.0 - pavg)) / sigmoid_;
```

**booste-rs** does compute base scores similarly - this is likely correct.

### 6. Regularization Defaults

**LightGBM**:
- `lambda_l1 = 0.0`
- `lambda_l2 = 0.0`
- `min_gain_to_split = 0.0`

**booste-rs**:
- `l1_reg = 0.0`
- `l2_reg = 1.0` ← Different! We have L2 reg by default
- `min_gain = 0.0`

Our default L2 reg of 1.0 might actually help, but the lack of min_data_in_leaf is more impactful.

## Recommendations

### Priority 1: Increase min_samples_leaf Default (Easy Win)

Change default from 1 to 20 (matching LightGBM):

```rust
#[builder(default = 20)]
pub min_samples_leaf: u32,
```

This alone could significantly improve small dataset performance.

### Priority 2: Add Multiclass Hessian Scaling Factor

Add the K/(K-1) factor from Friedman's original paper:

```rust
// In SoftmaxLoss::compute_gradients_into
let factor = n_classes as f32 / (n_classes as f32 - 1.0);
// ...
gh.hess = (w * factor * p * (1.0 - p)).max(HESS_MIN);
```

### Priority 3: Add scale_pos_weight for Binary Classification

Add optional class weighting to `LogisticLoss`:

```rust
pub struct LogisticLoss {
    pub scale_pos_weight: f32,  // default: 1.0
}
```

### Priority 4: Consider LightGBM-style Binary Gradient (Low Priority)

The {-1, +1} formulation with sigmoid scaling might have better numerical properties, but is mathematically equivalent. Only consider if other changes don't resolve the gap.

## Benchmarking Plan

1. Run current benchmarks as baseline
2. Change `min_samples_leaf` default to 20, re-run
3. Add multiclass hessian factor, re-run
4. Add `scale_pos_weight` option, test on imbalanced datasets

## Experimental Results

### Experiment 1: Multiclass Hessian Scaling Factor (K/(K-1))

**Hypothesis**: Adding the K/(K-1) factor would improve multiclass performance by making updates more conservative.

**Result**: NEGATIVE. Performance degraded on all multiclass datasets.

| Dataset | Before | After (K/(K-1)) | Delta |
|---------|--------|-----------------|-------|
| covertype | 0.4137 | 0.4215 | +0.0078 ❌ |
| iris | 0.1738 | 0.1930 | +0.0192 ❌ |
| synthetic_multi | 0.6704 | 0.6807 | +0.0103 ❌ |

**Conclusion**: The hessian scaling factor is NOT the differentiating factor. Our current implementation without scaling appears correct. The gaps on small datasets are due to other factors (likely `min_data_in_leaf`).

### Experiment 2: Constant Hessian Factor of 2.0 (XGBoost style)

**Hypothesis**: XGBoost uses 2.0 constant, maybe that helps?

**Result**: VERY NEGATIVE. Much worse performance, especially on multiclass.

| Dataset | Before | After (2.0) | Delta |
|---------|--------|-------------|-------|
| covertype | 0.4137 | 0.5096 | +0.0959 ❌❌ |
| iris | 0.1738 | 0.4743 | +0.3005 ❌❌❌ |

**Conclusion**: Higher hessian factors make steps too conservative for our learning rate. Not recommended.

### Analysis: Why LightGBM Wins on Small Datasets

The remaining differences are:

1. **min_data_in_leaf = 20**: LightGBM prevents deep trees on small datasets
2. **Class imbalance handling**: `is_unbalance` option auto-reweights
3. **Bagging/sampling defaults**: May provide additional regularization
4. **Possibly different histogram binning** on small datasets

Since the objective function is correct, the gap is likely due to tree-building constraints rather than gradient computation.

### Next Steps

1. **Do NOT change hessian scaling** - current implementation is optimal
2. Consider adding `is_unbalance` / `scale_pos_weight` options
3. Focus on tree-building constraints for small datasets (outside objective scope)

## References

- LightGBM config.h: `include/LightGBM/config.h`
- LightGBM binary_objective.hpp: `src/objective/binary_objective.hpp`
- LightGBM multiclass_objective.hpp: `src/objective/multiclass_objective.hpp`
- XGBoost regression_loss.h: `src/objective/regression_loss.h`
- XGBoost multiclass_obj.cu: `src/objective/multiclass_obj.cu`
- Friedman GBDT paper (2001): Mentions the K/(K-1) rescaling
