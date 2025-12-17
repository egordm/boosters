# Linear Trees

Linear Trees enhance gradient boosting by replacing constant leaf values with linear
regression models. Instead of predicting a single value per leaf, each leaf learns
coefficients for a subset of features, enabling piece-wise linear approximations.

---

## Background and Motivation

Standard GBDT uses constant leaf values:

$$
\text{prediction} = \sum_{t=1}^{T} \text{const}_{\text{leaf}(x, t)}
$$

Each leaf contains a single scalar that all samples falling into that leaf share.
This creates **piece-wise constant** predictions—the function is a step function
with discontinuities at split boundaries.

Linear Trees extend this by fitting a linear model at each leaf:

$$
\text{prediction} = \sum_{t=1}^{T} \left(\text{const}_{\ell} + \sum_{j \in F_\ell} c_j \cdot x_j \right)
$$

Where:

- $\text{const}_\ell$ is the intercept for leaf $\ell$
- $F_\ell$ is the set of features used in splits along the path to leaf $\ell$
- $c_j$ are the learned coefficients for each feature

This creates **piece-wise linear** predictions—smoother and often more accurate
when the underlying function has linear structure within regions.

### Intuition

Consider modeling house prices. A standard GBDT might split on `bedrooms < 3`,
giving separate constant predictions for small and large houses. With Linear Trees,
each leaf can also model `price ~ β₁×sqft + β₀`, capturing that price increases
with square footage even within the "small house" leaf.

The key insight: features used for splitting are likely relevant for prediction
within each leaf region. The split `bedrooms < 3` suggests `bedrooms` matters for
pricing, so including it in the leaf's linear model often improves fit.

### When Linear Trees Help

- **Continuous relationships**: Target varies smoothly with features
- **Interaction effects**: Different slopes in different regions of feature space
- **Limited tree depth**: Linear models compensate for coarse partitioning
- **Extrapolation**: Better behavior at distribution edges vs step functions

### When They May Hurt

- **Pure categorical targets**: No linear structure to exploit
- **High noise**: Linear fits may overfit noise patterns
- **Very deep trees**: Leaves have few samples, unstable coefficient estimates
- **Many features per leaf**: Overfitting risk with high-dimensional linear models

---

## Algorithm Overview

Linear Trees fit is a **post-processing step** after tree structure is determined:

```text
Algorithm: Train Linear Tree
────────────────────────────
1. Grow tree structure using standard histogram-based splits
2. For each leaf ℓ:
   a. Collect samples S_ℓ falling into this leaf
   b. Identify features F_ℓ used in splits on path from root to ℓ
   c. Filter to numerical features only (categoricals excluded)
   d. Fit linear model: y_ℓ ~ X_{S_ℓ, F_ℓ}
   e. Store (const_ℓ, coefficients_ℓ, feature_indices_ℓ)
3. Return tree with linear leaf models
```

This approach:

- Uses the **same tree structure** as standard GBDT (no split-finding changes)
- Exploits **path features** which are likely predictive in that region
- Limits model complexity to features relevant for the region

---

## Linear Model Fitting

At each leaf, we solve a weighted least squares problem in gradient boosting's
second-order approximation framework.

### Problem Formulation

Given samples in leaf $\ell$ with gradients $g_i$ and Hessians $h_i$:

$$
\min_{\mathbf{c}} \sum_{i \in S_\ell} h_i \left( \frac{g_i}{h_i} - \mathbf{x}_i^\top \mathbf{c} \right)^2 + \lambda \|\mathbf{c}_{1:}\|_2^2
$$

Where:

- $\mathbf{x}_i$ is the feature vector for sample $i$ (restricted to path features)
- $\mathbf{c} = [c_0, c_1, \ldots, c_k]$ are coefficients (intercept + features)
- $\lambda$ is L2 regularization (only on non-intercept coefficients)

### Approach 1: Closed-Form Solution (LightGBM)

LightGBM uses a direct matrix solution. Define:

- $\mathbf{X}$: Design matrix with rows $\mathbf{x}_i$, column 0 is all-ones (intercept)
- $\mathbf{H}$: Diagonal matrix with $H_{ii} = h_i$ (Hessian weights)
- $\mathbf{g}$: Gradient vector
- $\mathbf{R}$: Regularization matrix (identity except $R_{00} = 0$ for intercept)

The solution is:

$$
\mathbf{c} = -(\mathbf{X}^\top \mathbf{H} \mathbf{X} + \lambda \mathbf{R})^{-1} \mathbf{X}^\top \mathbf{g}
$$

The negative sign comes from the Newton step direction.

**Complexity**: $O(nk^2 + k^3)$ per leaf (matrix multiply + solve).

**Limitations**:

- Requires matrix inversion—ill-conditioned when features are collinear
- Must check condition number and fall back to constant if singular
- Needs robust solver (pseudo-inverse or QR decomposition)

### Approach 2: Coordinate Descent (Alternative)

Instead of closed-form solution, we can reuse **coordinate descent** infrastructure
(same as GBLinear training). This iteratively updates each coefficient:

```text
Algorithm: Fit Linear Model via Coordinate Descent
───────────────────────────────────────────────────
Inputs:
  - X: n × k feature matrix (samples in leaf, restricted features)
  - g: n × 1 gradient vector
  - H: n × 1 Hessian vector
  - λ: L2 regularization, α: L1 regularization
  - n_iters: number of passes

Initialize: c = [0, 0, ..., 0]

for iter = 1 to n_iters:
    # Update bias (no regularization)
    c[0] -= learning_rate × Σg / ΣH
    
    # Update each coefficient
    for j = 1 to k:
        sum_grad = Σ(g × X[:,j]) + λ × c[j]
        sum_hess = Σ(H × X[:,j]²) + λ
        c[j] -= learning_rate × soft_threshold(sum_grad/sum_hess, α/sum_hess)

return c
```

**Advantages over closed-form**:

1. **No matrix inversion**: Avoids ill-conditioning issues entirely
2. **Handles collinearity naturally**: Coordinate descent converges even with
   correlated features (may just take more iterations)
3. **L1 regularization**: Can use elastic net for sparse coefficients
4. **Reuses existing code**: GBLinear's `Updater` already implements this
5. **Early termination**: Can stop when coefficients stabilize

**Trade-offs**:

- Multiple iterations vs single solve
- But leaf sizes are small (dozens to hundreds of samples), so either is fast

### Regularization

The regularization parameter $\lambda$ (often called `linear_lambda`) prevents
overfitting when leaves have few samples or many features:

- **$\lambda = 0$**: No regularization
- **$\lambda > 0$**: Ridge regression (L2), shrinks coefficients toward zero
- **$\alpha > 0$**: Lasso regularization (L1), encourages sparse coefficients
- **Large $\lambda$**: Coefficients approach zero, leaf degenerates to constant

In practice, LightGBM defaults to `linear_lambda = 0.0`, as the tree structure
already provides regularization (depth limits, min samples per leaf).

### Numerical Stability

With closed-form solution, the system can be ill-conditioned when:

- Features are nearly collinear
- Leaf has very few samples
- Hessians are very small (flat loss region)

Coordinate descent avoids most of these issues, but may converge slowly with
highly correlated features. Adding small λ helps in both approaches.

---

## Feature Selection for Leaves

### Path Feature Extraction

The feature set for each leaf is determined by the splits on the path from root:

```text
Example Tree:
                    [feature 0 < 0.5]
                   /                 \
        [feature 2 < 0.3]         [feature 1 < 0.8]
         /           \             /           \
      leaf A       leaf B       leaf C       leaf D

Leaf A path features: {0, 2}
Leaf B path features: {0, 2}
Leaf C path features: {0, 1}
Leaf D path features: {0, 1}
```

### Categorical Feature Handling

Categorical features in splits are **excluded** from linear models:

- Linear coefficients don't make sense for categorical variables
- One-hot encoding would explode feature count
- The split already captures categorical effect

Only **numerical features** from the path are included.

### Feature Whitelist (Extrapolation Control)

Users may want to control which features get linear coefficients:

- **Extrapolation concern**: Some features (e.g., IDs, rare categories) shouldn't
  extrapolate linearly beyond training range
- **Domain knowledge**: Known non-linear relationships shouldn't be linearized

A **feature whitelist** allows users to specify which features can have linear
coefficients. Features not on the whitelist contribute only to the intercept:

```text
whitelist = {sqft, bedrooms, bathrooms}

For leaf with path features {sqft, year_built, lot_id}:
  linear_features = {sqft}  # Only whitelisted features
  intercept absorbs effect of year_built, lot_id
```

This gives finer control than path-only selection, preventing extrapolation on
features known to be problematic.

### Feature Subsetting Benefits

Using only path features (optionally filtered by whitelist) provides:

1. **Reduced complexity**: k << total features, smaller matrices
2. **Relevance**: Path features are known-useful for this region
3. **Regularization**: Implicit feature selection prevents overfitting
4. **Interpretability**: Each leaf uses a small, meaningful feature set
5. **Extrapolation control**: Whitelist prevents dangerous extrapolation

---

## Missing Value Handling

Linear models cannot directly handle NaN values. LightGBM uses a fallback strategy:

### Detection and Fallback

```text
For each leaf ℓ:
  X_ℓ = features for samples in leaf (restricted to path features)
  
  if any(isnan(X_ℓ)):
    # Fall back to constant prediction
    output = weighted_mean(y_ℓ, weights=H_ℓ)
    coefficients = []
    features = []
  else:
    # Fit linear model
    (const, coefficients, features) = fit_linear(X_ℓ, g_ℓ, H_ℓ)
```

### Implications

- Leaves with any NaN in path features degrade to constants
- This is conservative: entire leaf loses linear benefit, even for non-NaN rows
- In practice, NaN often falls on one side of splits, limiting affected leaves

### Alternative Strategies (Not in LightGBM)

- **Mean imputation**: Replace NaN with feature mean
- **Per-row fallback**: Use linear for non-NaN rows, constant for NaN rows
- **Separate intercept**: Track NaN-adjusted predictions per row

---

## First Tree Special Case

The first tree in an ensemble often has special handling:

### Why First Tree is Different

- No residuals yet: Predictions are just the base score
- Gradients are homogeneous: All samples have similar gradient/Hessian ratios
- Linear fit provides little benefit: Intercept dominates

### LightGBM Behavior

In LightGBM's implementation, the first tree always uses **constant leaves**,
even when `linear_tree=True`:

```cpp
// From LightGBM linear_tree_learner.cpp
if (train_data_->iteration() == 0) {
  // First tree: constant leaves only
}
```

This is a pragmatic choice:

- Avoids ill-conditioned systems when gradients are uniform
- First tree typically just approximates the mean
- Subsequent trees have more structure to exploit

---

## Inference

At inference time, linear trees require feature values to compute predictions:

### Prediction Algorithm

```text
Algorithm: Predict with Linear Tree
───────────────────────────────────
Input: feature vector x, tree T

# Traverse to leaf
node = root
while not is_leaf(node):
    if x[split_feature(node)] < threshold(node):
        node = left_child(node)
    else:
        node = right_child(node)

# Linear prediction at leaf
leaf_features = T.features[node]  # Feature indices for this leaf
coefficients = T.coefficients[node]  # Coefficients for these features
const = T.const[node]  # Intercept

prediction = const
for (i, f) in enumerate(leaf_features):
    prediction += coefficients[i] * x[f]

return prediction
```

### Complexity

- **Traversal**: O(depth), same as standard trees
- **Linear combination**: O(k) where k = number of path features

For deep trees, k can approach depth (one feature per split level), so total
inference is O(depth²) in the worst case. In practice, features are often reused,
keeping k small.

### NaN at Inference

If any feature in `leaf_features` is NaN at inference time:

- LightGBM returns just the constant (ignores linear part)
- This matches training behavior (NaN leaves use constant)

```text
for (i, f) in enumerate(leaf_features):
    if isnan(x[f]):
        return const  # Early exit, use constant only
    prediction += coefficients[i] * x[f]
```

---

## Learning Rate Application

Learning rate (shrinkage) is applied uniformly to the entire leaf prediction:

$$
\text{scaled\_output} = \eta \cdot (\text{const} + \sum_j c_j x_j)
$$

This is equivalent to scaling both the intercept and coefficients:

$$
\text{const}' = \eta \cdot \text{const}, \quad c_j' = \eta \cdot c_j
$$

LightGBM applies shrinkage during training (post linear-fit), storing the scaled
values. This matches how constant leaves are handled.

---

## Data Structures

### Per-Leaf Storage

Each leaf stores:

| Field | Type | Description |
|-------|------|-------------|
| `const` | f32 | Intercept term (scaled by learning rate) |
| `n_features` | u8/u16 | Number of features in linear model |
| `feature_indices` | Vec<u32> | Global feature indices |
| `coefficients` | Vec<f32> | Coefficients (scaled by learning rate) |

### Tree-Level Storage

Options for organizing leaf data:

**Option A: Inline per-leaf**

```rust
struct LinearLeaf {
    const_: f32,
    features: Box<[u32]>,
    coefficients: Box<[f32]>,
}
```

Simple but variable-sized, requires allocation per leaf.

**Option B: SoA with offset indexing**

```rust
struct LinearTree {
    // Per-leaf scalars
    leaf_consts: Vec<f32>,      // [n_leaves]
    
    // Coefficient storage (concatenated)
    all_coefficients: Vec<f32>, // [sum of all leaf feature counts]
    all_features: Vec<u32>,     // [sum of all leaf feature counts]
    
    // Indexing into coefficient storage
    coef_offsets: Vec<u32>,     // [n_leaves + 1] - start offset per leaf
}
```

Cache-friendly for inference, similar to how categories are stored in booste-rs.

---

## LightGBM Implementation Details

LightGBM's implementation is in `linear_tree_learner.cpp`:

### Key Parameters

- `linear_tree` (bool): Enable linear tree leaves
- `linear_lambda` (f64): L2 regularization on coefficients (not intercept)

### Data Structures

```cpp
// In Tree class
std::vector<int> leaf_features_;     // Feature indices (all leaves concatenated)
std::vector<double> leaf_coeff_;     // Coefficients (all leaves concatenated)
std::vector<double> leaf_const_;     // Intercepts (one per leaf)
std::vector<int> leaf_features_offset_; // Start index per leaf
```

### Training Flow

1. Standard tree structure is grown first
2. `LinearTreeLearner::CalculateLinearTree()` is called
3. For each leaf:
   - Extract path features
   - Build Gram matrix and gradient projection
   - Solve linear system (Cholesky decomposition)
   - Prune near-zero coefficients
   - Apply shrinkage (learning rate)
4. Store coefficients in tree structure

### Coefficient Pruning

LightGBM prunes coefficients with absolute value below a threshold:

```cpp
const double kZeroThreshold = 1e-6;
if (std::fabs(coeff[j]) < kZeroThreshold) {
    // Omit this coefficient
}
```

This reduces storage and inference cost for effectively zero weights.

---

## References

- **LightGBM Paper (Linear Trees)**: Shi, Y., et al. (2018).
  "Gradient Boosting with Piece-Wise Linear Regression Trees." arXiv:1802.05640.
  https://arxiv.org/abs/1802.05640

- **LightGBM Implementation**: 
  `src/treelearner/linear_tree_learner.cpp`,
  `include/LightGBM/tree.h`

- **XGBoost**: As of 2024, XGBoost does not support linear tree leaves natively.
  Linear models are available via `booster='gblinear'` (separate model type).

---

## Summary

Linear Trees extend GBDT by fitting linear models at leaves instead of constants:

1. **Tree structure unchanged**: Standard histogram-based growing
2. **Post-processing step**: Fit linear models after structure is fixed
3. **Path features**: Use only features from splits leading to each leaf
4. **Weighted least squares**: Solve $\mathbf{c} = -(\mathbf{X}^\top\mathbf{H}\mathbf{X} + \lambda\mathbf{R})^{-1}\mathbf{X}^\top\mathbf{g}$
5. **Fallback for NaN**: Use constant prediction when any path feature is NaN
6. **First tree special case**: Always use constant leaves
7. **Learning rate applies uniformly**: Both intercept and coefficients scaled
