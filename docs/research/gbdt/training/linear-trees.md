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

> **Deep Dive**: See [Appendix: Fitting Strategies Deep Dive](#appendix-fitting-strategies-deep-dive)
> for comprehensive analysis of solving strategies, gradient recalculation,
> multi-class considerations, and alternative approaches.

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

## Summary

Linear Trees extend GBDT by fitting linear models at leaves instead of constants:

1. **Tree structure unchanged**: Standard histogram-based growing
2. **Post-processing step**: Fit linear models after structure is fixed
3. **Path features**: Use only features from splits leading to each leaf
4. **Weighted least squares**: Solve $\mathbf{c} = -(\mathbf{X}^\top\mathbf{H}\mathbf{X} + \lambda\mathbf{R})^{-1}\mathbf{X}^\top\mathbf{g}$
5. **Fallback for NaN**: Use constant prediction when any path feature is NaN
6. **First tree special case**: Always use constant leaves
7. **Learning rate applies uniformly**: Both intercept and coefficients scaled

---

## References

- **LightGBM Paper (Linear Trees)**: Shi, Y., et al. (2018).
  "Gradient Boosting with Piece-Wise Linear Regression Trees." arXiv:1802.05640.
  https://arxiv.org/abs/1802.05640

- **LightGBM Implementation**: 
  `src/treelearner/linear_tree_learner.cpp`,
  `include/LightGBM/tree.h`

- **XGBoost**: As of 2024, XGBoost does not support linear tree leaves natively.
  Linear models are available via `booster='gblinear'` (separate model type.

---

## Appendix: Fitting Strategies Deep Dive

The core question in linear leaf training is: **how do we solve the weighted
least squares problem at each leaf?** This section explores the design space,
from simple closed-form solutions to iterative methods, and explains when each
approach is appropriate.

### The Fundamental Problem

At each leaf, we want coefficients $\mathbf{c}$ that minimize the second-order
approximation of the loss:

$$
\mathcal{L}(\mathbf{c}) = \sum_{i \in S_\ell} \left[ g_i \cdot f_i + \frac{1}{2} h_i \cdot f_i^2 \right] + \frac{\lambda}{2} \|\mathbf{c}_{1:}\|_2^2
$$

where $f_i = \mathbf{x}_i^\top \mathbf{c}$ is the linear prediction for sample $i$.

Taking the gradient and setting to zero gives the normal equations:

$$
(\mathbf{X}^\top \mathbf{H} \mathbf{X} + \lambda \mathbf{R}) \mathbf{c} = -\mathbf{X}^\top \mathbf{g}
$$

This is a $(k+1) \times (k+1)$ linear system where $k$ is the number of features.

### Strategy 1: Direct Matrix Solve (LightGBM)

The most straightforward approach: build the matrix, invert, multiply.

```text
Algorithm: Direct Solve for Linear Leaf
───────────────────────────────────────
Input: X[n,k+1] (design matrix with intercept column)
       g[n], H[n] (gradients and hessians)
       λ (regularization)

# Build normal equations
A = X^T × diag(H) × X        # (k+1) × (k+1) matrix
A[1:,1:] += λ × I            # Add regularization (not on intercept)
b = -X^T × g                 # (k+1) vector

# Solve linear system
c = solve(A, b)              # Cholesky, LU, or pseudo-inverse

return c
```

**Pros:**
- Single pass to build system, then O(k³) to solve
- Exact solution (up to numerical precision)
- Well-understood numerical properties

**Cons:**
- Requires matrix inversion—fails on singular/ill-conditioned systems
- No natural L1 (lasso) support
- Needs robust solver with fallback for numerical issues

**Intuition:** This is the "textbook" weighted least squares solution. It's fast
when k is small (typical: 3-10 features per leaf) and the system is well-conditioned.

### Strategy 2: Coordinate Descent on Normal Equations

Instead of matrix inversion, iterate over coefficients one at a time:

```text
Algorithm: CD on Normal Equations
─────────────────────────────────
Input: X[n,k+1], g[n], H[n], λ, α, max_iters, tol

# Precompute column statistics (done once)
for j = 0 to k:
    col_H[j] = Σ(H × X[:,j]²)     # Hessian-weighted column norm

# Initialize
c = [0, 0, ..., 0]

for iter = 1 to max_iters:
    max_delta = 0
    
    for j = 0 to k:
        # Compute gradient for this coefficient
        grad_j = Σ(g × X[:,j]) + Σ(H × X[:,j] × (X × c - X[:,j] × c[j]))
        
        # Simplified: using residual tracking
        grad_j = Σ((g + H × residual) × X[:,j])
        
        # Add L2 regularization (skip intercept j=0)
        if j > 0: grad_j += λ × c[j]
        hess_j = col_H[j] + (λ if j > 0 else 0)
        
        # Update with soft-thresholding for L1
        old_c = c[j]
        c[j] = soft_threshold(-grad_j / hess_j, α / hess_j)
        delta = c[j] - old_c
        
        # Update residual
        residual += delta × X[:,j]
        max_delta = max(max_delta, |delta|)
    
    if max_delta < tol: break

return c
```

**Pros:**
- No matrix inversion—never fails numerically
- Handles collinearity gracefully (just converges slower)
- Natural L1 support via soft-thresholding
- Early termination when converged

**Cons:**
- Multiple iterations (typically 5-20 for convergence)
- Slightly more bookkeeping (residual tracking)

**Intuition:** CD solves the same problem as direct solve, just iteratively.
It's like adjusting one knob at a time until the system settles. Collinear
features cause oscillation, but regularization and small learning rate dampen it.

### Strategy 3: True CD with Gradient Recalculation (GBLinear-style)

The strategies above use **fixed gradients** from the boosting round. This strategy
recalculates gradients based on current predictions:

```text
Algorithm: True CD for Linear Leaf
──────────────────────────────────
Input: X[n,k+1], labels[n], weights[n], objective, output_idx
       λ, α, max_iters, tol

# Initialize predictions at leaf's base value
base = objective.base_score(labels, weights, output_idx)
preds = [base, base, ..., base]   # n values
c = [base, 0, 0, ..., 0]          # intercept = base, others = 0

for iter = 1 to max_iters:
    # Recalculate gradients from CURRENT predictions
    (g, H) = objective.compute_gradients(preds, labels, weights, output_idx)
    
    max_delta = 0
    for j = 0 to k:
        # Standard CD update with fresh gradients
        grad_j = Σ(g × X[:,j])
        if j > 0: grad_j += λ × c[j]
        hess_j = Σ(H × X[:,j]²) + (λ if j > 0 else 0)
        
        old_c = c[j]
        c[j] = soft_threshold(-grad_j / hess_j, α / hess_j)
        delta = c[j] - old_c
        
        # Update predictions incrementally
        preds += delta × X[:,j]
        max_delta = max(max_delta, |delta|)
    
    if max_delta < tol: break

return c
```

**Pros:**
- Gradients reflect current state—no stale approximation
- Works with any objective, even non-smooth ones
- Exact match to how GBLinear trains globally

**Cons:**
- Expensive: O(n) gradient computation per iteration
- Requires access to objective, labels, weights
- More complex interface

**Intuition:** This is "training a tiny GBLinear model inside each leaf." The
gradients update as predictions change, so the optimization follows the true
loss surface rather than a quadratic approximation.

### When Does Gradient Recalculation Matter?

The key insight: fixed gradients assume the **Newton approximation is accurate**
within the leaf. This holds when:

1. **Loss is locally quadratic**: Squared loss is exactly quadratic, so fixed
   gradients give the exact answer. Logistic loss is well-approximated.

2. **Predictions don't move far**: If coefficients are small (heavy regularization),
   predictions stay near the initial point where gradients were computed.

3. **Objective is smooth**: Discontinuous gradients (quantile loss) violate the
   quadratic assumption everywhere.

**When fixed gradients fail:**

- **Quantile/Pinball loss**: Gradient is $\pm 1$, hessian is 0 (or smoothed).
  The quadratic approximation is flat—it provides no curvature information.
  
- **Huber loss at transition**: Near the δ threshold, behavior changes abruptly.

- **Large coefficient updates**: If the linear model predicts values far from
  the initial leaf value, the original gradients become stale.

### Multi-Output and Multi-Class Considerations

**Multi-output regression (e.g., multi-quantile):**

Each output has independent gradients. Train linear models independently:

```text
for output_idx = 0 to num_outputs - 1:
    (g, H) = gradients_for_output[output_idx]
    c[output_idx] = fit_linear(X, g, H, config)
```

With fixed gradients (Strategy 1 or 2), this is straightforward—just use
gradients for each output. With recalculation (Strategy 3), each output
still fits independently since outputs don't interact.

**Multi-class classification (softmax):**

This is more complex. Softmax gradients for class $k$ depend on predictions
for **all classes**:

$$
g_{ik} = p_{ik} - \mathbf{1}[y_i = k], \quad p_{ik} = \frac{e^{f_{ik}}}{\sum_j e^{f_{ij}}}
$$

**Important**: The true Hessian for softmax is **not diagonal**—it's the covariance
matrix of class probabilities: $H_{kl} = p_k(\mathbf{1}[k=l] - p_l)$. The fixed-gradient
approach implicitly uses only the diagonal terms $H_{kk} = p_k(1-p_k)$, which is a
valid Newton approximation but ignores cross-class correlations.

**With fixed gradients:** Use gradients computed at round start. Each class's
trees are independent—the gradient for class $k$ was computed before any linear
fitting in this round. This is simple and matches LightGBM.

**With gradient recalculation:** Would need to track predictions for ALL classes
simultaneously and recompute softmax after each coefficient update. This couples
all class trees together—complex and expensive.

```text
# Pseudocode: Recalculated gradients for softmax (NOT recommended)
for iter = 1 to max_iters:
    # Compute softmax probabilities from ALL class predictions
    for i = 1 to n:
        probs[i,:] = softmax(all_class_preds[i,:])
    
    # Update gradients for this class
    g = probs[:,this_class] - (labels == this_class)
    H = probs[:,this_class] × (1 - probs[:,this_class])
    
    # CD update...
    # Problem: changing this class's preds affects OTHER classes' gradients!
```

**Recommendation:** Always use fixed gradients for multi-class. The complexity
of gradient recalculation is not worth it.

**Binary classification (logistic):**

Falls between regression and multi-class. Fixed gradients work well because:
- Logistic loss is smooth and well-approximated by quadratic
- Only one output to track

Gradient recalculation is feasible but rarely needed.

### Other Approaches to Consider

Beyond the three main strategies, there are other methods worth noting:

**Conjugate Gradient (CG):**

For large sparse systems, CG is efficient:
- O(n·k) per iteration (matrix-vector products, no full matrix)
- Converges in at most k iterations for k×k system
- Good when k is large and matrix is sparse

For linear leaves, k is typically small (< 20), so CG offers little advantage
over direct solve. More relevant for very deep trees with many path features.

**Stochastic/Mini-batch CD:**

For very large leaves, don't use all samples:

```text
for iter = 1 to max_iters:
    batch = random_sample(rows, batch_size)
    g_batch, H_batch, X_batch = gather(batch)
    # CD update on batch only
```

Adds noise but reduces per-iteration cost. Relevant if leaves have millions of
samples—unusual in practice.

**Proximal Gradient Methods:**

For L1 regularization, proximal gradient descent can be more efficient than CD:
- Update: $c^{(t+1)} = \text{prox}_{\alpha}(c^{(t)} - \eta \nabla f(c^{(t)}))$
- Prox operator is soft-thresholding
- Can use acceleration (FISTA) for faster convergence

CD is simpler and works well for small k, but proximal methods scale better.

**Hybrid: Fixed Gradients + Multiple CD Iterations:**

A middle ground between Strategy 1 and 3:
- Use fixed gradients (no recalculation)
- But run multiple CD iterations to handle collinearity
- Simpler than true CD (no objective access) but more robust than direct solve

This is essentially Strategy 2, and is the recommended approach.

### Comparison Summary

| Strategy | Complexity | Robustness | L1 Support | Objective Access |
|----------|-----------|------------|------------|------------------|
| 1. Direct Solve | O(nk² + k³) | Low (ill-conditioning) | No | No |
| 2. CD on Normal Eq | O(nk × iters) | High | Yes | No |
| 3. True CD | O(nk × iters) | Highest | Yes | Yes |
| CG | O(nk × k) | Medium | No | No |
| Proximal | O(nk × iters) | High | Yes | No |

### Recommended Approach

**Primary: Strategy 2 (CD on Normal Equations)**

- Robust to collinearity without special handling
- Supports elastic net (L1 + L2) regularization
- No objective access needed—simpler interface
- Matches expected behavior for standard losses
- Can reuse GBLinear's CD infrastructure

**Fallback: Strategy 1 (Direct Solve)**

- For benchmarking/validation against LightGBM
- Faster for very small k when system is well-conditioned
- Useful as optional backend for advanced users

**Future: Strategy 3 (True CD)**

- If users report issues with complex objectives
- Requires separate RFC due to interface changes
- Most beneficial for quantile regression and custom losses
