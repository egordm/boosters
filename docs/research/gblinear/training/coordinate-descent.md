# Coordinate Descent Training

How linear models learn feature weights using coordinate descent optimization.

## Overview

Linear boosters use **coordinate descent** — instead of updating all weights at once (like gradient descent), we update one weight at a time while holding others fixed.

### ELI5

Imagine tuning a guitar. Gradient descent tries to adjust all strings at once. Coordinate descent adjusts one string, listens, adjusts the next string, listens, and keeps cycling through. Simpler, and for certain problems, just as effective.

### Why Coordinate Descent?

For convex problems with separable regularization (like elastic net):

1. **Closed-form updates** — Each coordinate update has an analytical solution (no step size tuning)
2. **Convergence guaranteed** — Provably reaches the global optimum
3. **Efficient for sparse data** — Only touches non-zero feature values
4. **Natural for L1** — Soft thresholding integrates cleanly

---

## Elastic Net Regularization

The objective function is:

$$
\text{minimize:} \quad \mathcal{L}(\mathbf{w}) + \lambda_1 \sum_i |w_i| + \lambda_2 \sum_i w_i^2
$$

Where:
- $\mathcal{L}(\mathbf{w})$ — Loss function (squared error, logistic, etc.)
- $\lambda_1$ (`reg_alpha`) — L1 penalty, encourages sparse weights
- $\lambda_2$ (`reg_lambda`) — L2 penalty, encourages small weights

### Why Elastic Net?

| Regularization | Pros | Cons |
|---------------|------|------|
| L1 alone (Lasso) | Feature selection, sparse | Unstable with correlated features |
| L2 alone (Ridge) | Stable, handles correlation | Keeps all features |
| Elastic Net | Sparse AND stable | Two hyperparameters |

Elastic net gets the best of both: it can zero out irrelevant features (like L1) while remaining stable when features are correlated (like L2).

---

## The Coordinate Update Formula

For a single weight $w_j$, the update proceeds as follows:

### Step 1: Compute Gradient Statistics

Sum over all training examples:

$$
G_j = \sum_i g_i \cdot x_{ij} \qquad H_j = \sum_i h_i \cdot x_{ij}^2
$$

Where:
- $g_i$ = gradient of loss for example $i$
- $h_i$ = hessian of loss for example $i$
- $x_{ij}$ = value of feature $j$ for example $i$

### Step 2: Add L2 Regularization

$$
G_j^{L2} = G_j + \lambda_2 \cdot w_j \qquad H_j^{L2} = H_j + \lambda_2
$$

### Step 3: Apply Soft Thresholding (L1)

The soft thresholding operator handles L1 regularization:

$$
\delta = \begin{cases}
\max\left(-\frac{G_j^{L2} + \lambda_1}{H_j^{L2}}, -w_j\right) & \text{if } w_j - \frac{G_j^{L2}}{H_j^{L2}} \geq 0 \\[1em]
\min\left(-\frac{G_j^{L2} - \lambda_1}{H_j^{L2}}, -w_j\right) & \text{otherwise}
\end{cases}
$$

### Step 4: Apply Update

$$
w_j \leftarrow w_j + \eta \cdot \delta
$$

Where $\eta$ is the learning rate.

### Pseudocode

```python
def coordinate_update(w_j, feature_column, gradients, hessians, lambda1, lambda2, eta):
    # Step 1: Gradient stats
    grad_sum = sum(g * x for g, x in zip(gradients, feature_column))
    hess_sum = sum(h * x * x for h, x in zip(hessians, feature_column))
    
    # Step 2: L2 adjustment
    grad_L2 = grad_sum + lambda2 * w_j
    hess_L2 = hess_sum + lambda2
    
    # Step 3: Soft thresholding for L1
    tmp = w_j - grad_L2 / hess_L2
    if tmp >= 0:
        delta = max(-(grad_L2 + lambda1) / hess_L2, -w_j)
    else:
        delta = min(-(grad_L2 - lambda1) / hess_L2, -w_j)
    
    # Step 4: Apply with learning rate
    return w_j + eta * delta
```

### Bias Update

The bias term has **no regularization** — just plain gradient descent:

$$
b \leftarrow b + \eta \cdot \left(-\frac{\sum_i g_i}{\sum_i h_i}\right)
$$

---

## Updaters: Parallel vs Sequential

XGBoost provides two coordinate descent implementations:

### Shotgun Updater (Default)

Updates features **in parallel** — different threads update different weights simultaneously.

```
Thread 0: update w₀    ┐
Thread 1: update w₁    ├── all at the same time
Thread 2: update w₂    │
Thread 3: update w₃    ┘
```

**How it works**:
- All threads read the same gradient vector
- Each thread updates its assigned weights
- Race conditions in gradient reads are tolerated
- Uses lock-free writes

**Trade-offs**:
| Pros | Cons |
|------|------|
| Significant speedup on multi-core | Approximate gradients (stale reads) |
| Good scaling | Only supports `cyclic` and `shuffle` selectors |
| Works well with small learning rates | Convergence is approximate |

**When to use**: Default choice for most use cases. Works well in practice.

### Coord_descent Updater (Sequential)

Updates features **one at a time**, updating residuals after each change.

```
update w₀ → refresh gradients
        ↓
update w₁ → refresh gradients
        ↓
update w₂ → refresh gradients
        ↓
...
```

**Trade-offs**:
| Pros | Cons |
|------|------|
| Exact gradients | Slower (sequential) |
| Supports all feature selectors | Can't parallelize feature updates |
| Better for greedy/thrifty selection | |

**When to use**: When using advanced feature selectors, or when shotgun isn't converging well.

---

## Feature Selectors

How to choose which feature to update next?

| Selector | Strategy | Complexity | Shotgun? |
|----------|----------|------------|----------|
| **cyclic** | Round-robin: 0, 1, 2, ..., n, 0, 1, ... | O(n) | ✅ |
| **shuffle** | Random permutation each round | O(n) | ✅ |
| **random** | Random with replacement | O(n) | ❌ |
| **thrifty** | Sort by gradient magnitude, update important first | O(n log n) | ❌ |
| **greedy** | Always pick feature with largest gradient | O(n²) | ❌ |

### Why Feature Selection Matters

For high-dimensional sparse data, most features may have zero gradient at any given time. Smarter selection strategies can converge faster by focusing on features that actually matter.

### Selector Details

**Cyclic** (default): Simple and fast. Processes features in order. No overhead.

**Shuffle**: Like cyclic, but randomizes order each round. Can break patterns and improve convergence for some problems.

**Random**: Picks features randomly with replacement. Some features may be updated multiple times per round, others not at all.

**Thrifty**: Pre-sorts features by gradient magnitude once per round, then processes in that order. Good balance — focuses on important features without the O(n²) cost of greedy.

**Greedy**: Recomputes gradients after each update and always picks the feature with the largest gradient. Best convergence but O(n²) per round — only practical for small feature sets.

### Top-K Pruning

For `thrifty` and `greedy`, the `top_k` parameter limits updates to the k most important features:

```python
params = {
    'feature_selector': 'thrifty',
    'top_k': 100,  # Only update top 100 features per round
}
```

Reduces computation when most features are irrelevant.

---

## Convergence Criteria

XGBoost tracks the largest weight change each round:

$$
\Delta_{\max} = \max_j |w_j^{\text{new}} - w_j^{\text{old}}|
$$

Training stops early if:

$$
\Delta_{\max} \leq \texttt{tolerance}
$$

**Parameters**:
- `tolerance = 0` (default): Disabled, run all rounds
- `tolerance = 1e-4`: Stop when weights stabilize

### Practical Notes

- With strong regularization, convergence is faster
- Shotgun may oscillate slightly due to stale gradients
- If not converging, try:
  - Reducing learning rate
  - Switching to `coord_descent` updater
  - Increasing regularization

---

## Data Format: CSC

Training uses **CSC (Column-Sparse-Compressed)** format for efficient feature-wise access:

```
Column 0: [(row_2, 0.5), (row_7, 1.2), (row_9, 0.8)]
Column 1: [(row_0, 0.3), (row_5, 2.1)]
Column 2: [(row_1, 0.7), (row_3, 1.5), (row_6, 0.2), (row_8, 0.9)]
...
```

### Why CSC?

Coordinate descent iterates over features (columns):

| Format | Access pattern | Cost per feature |
|--------|---------------|------------------|
| Row-major (CSR) | Scan all rows | O(total_nnz) |
| Column-major (CSC) | Direct column access | O(nnz in column) |

CSC is essential for efficient coordinate descent on sparse data.

---

## Threading Model

Training parallelism happens at multiple levels:

### 1. Gradient Accumulation (parallel over rows)

For each feature, the gradient sum is computed in parallel:

```cpp
// Parallel reduction with thread-local accumulators
parallel_for(rows, [&](row) {
    auto tid = thread_id();
    sum_grad_local[tid] += gradient[row] * feature_value[row];
    sum_hess_local[tid] += hessian[row] * feature_value[row]²;
});
// Single-thread reduce
total_grad = sum(sum_grad_local);
total_hess = sum(sum_hess_local);
```

Uses thread-local buffers to avoid atomics.

### 2. Feature Updates (shotgun only)

The shotgun updater parallelizes across features:

```cpp
parallel_for(features, [&](feature) {
    // Each thread updates different features
    // Gradients may be slightly stale
    update_weight(feature);
});
```

### 3. Residual Updates (sequential updater)

After each weight change, residuals must be updated sequentially — this is why `coord_descent` is slower than `shotgun`.

---

## Summary

| Aspect | Description |
|--------|-------------|
| Algorithm | Coordinate descent with elastic net |
| Update | Closed-form with soft thresholding |
| Parallelism | Shotgun (features) or sequential + parallel gradient sums |
| Data format | CSC for efficient column access |
| Convergence | Guaranteed for convex objectives |
| Key params | `lambda` (L2), `alpha` (L1), `eta` (learning rate) |
