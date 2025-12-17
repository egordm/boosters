# GBLinear Training

How GBLinear learns feature weights using coordinate descent.

## Overview

GBLinear uses **coordinate descent** — instead of updating all weights at once
(like gradient descent), it updates one weight at a time while holding others fixed.

### ELI5

Imagine tuning a guitar. Gradient descent tries to adjust all strings at once.
Coordinate descent adjusts one string, listens, adjusts the next string, listens,
and keeps cycling through. Simpler, and for certain problems, just as effective.

### ELI-Grad

Coordinate descent iteratively minimizes along each coordinate axis. For convex
problems with separable regularization (like elastic net), it converges to the
global optimum. Each coordinate update has a closed-form solution.

## Elastic Net Regularization

The objective function is:

```
minimize: L(w) + λ₁ Σ|wᵢ| + λ₂ Σwᵢ²
```

Where:
- `L(w)` — Loss function (squared error, logistic, etc.)
- `λ₁` (reg_alpha) — L1 penalty, encourages sparse weights (some become exactly 0)
- `λ₂` (reg_lambda) — L2 penalty, encourages small weights (weight decay)

**Why elastic net?**
- L1 alone (Lasso): Feature selection, but unstable with correlated features
- L2 alone (Ridge): Stable, but keeps all features
- Elastic net: Best of both — sparse, stable, handles correlation

## The Coordinate Update

For a single weight `w_j`, the update rule is:

```python
# Compute gradient stats
grad_sum = Σᵢ (gradient_i × feature_value_ij)
hess_sum = Σᵢ (hessian_i × feature_value_ij²)

# L2-adjusted quantities
grad_L2 = grad_sum + λ₂ × w_j
hess_L2 = hess_sum + λ₂

# Compute delta (with soft thresholding for L1)
tmp = w_j - grad_L2 / hess_L2
if tmp >= 0:
    delta = max(-(grad_L2 + λ₁) / hess_L2, -w_j)
else:
    delta = min(-(grad_L2 - λ₁) / hess_L2, -w_j)

w_j += learning_rate × delta
```

The soft thresholding is the **proximal operator** for L1 — it pushes small weights
toward zero and can make them exactly zero.

## Updaters

XGBoost has two coordinate descent updaters:

### Shotgun (Default)

Updates features **in parallel** — different threads update different weights simultaneously.

```
Thread 1: update w₀
Thread 2: update w₁     (all at the same time)
Thread 3: update w₂
...
```

**Trade-off**: Introduces race conditions in gradient updates, but:
- Works well in practice with small learning rates
- Significant speedup on multi-core systems
- Convergence still guaranteed (approximately)

**Restrictions**: Only supports `cyclic` and `shuffle` feature selectors.

### Coord_descent (Sequential)

Updates features **one at a time**, updating residuals after each change.

```
update w₀ → update gradients
update w₁ → update gradients
update w₂ → update gradients
...
```

**Trade-off**: Slower but more accurate gradients. Supports all feature selectors.

## Feature Selectors

How to choose which feature to update next?

| Selector | Description | Complexity |
|----------|-------------|------------|
| **cyclic** | Round-robin: 0, 1, 2, ..., n, 0, 1, ... | O(n) |
| **shuffle** | Random permutation each round | O(n) |
| **random** | Random with replacement | O(n) |
| **thrifty** | Sort by gradient magnitude, update important features first | O(n log n) |
| **greedy** | Always pick feature with largest gradient | O(n²) |

### Why Feature Selection Matters

For high-dimensional sparse data, most features may have zero gradient at any time.
Smarter selection (thrifty, greedy) can converge faster by focusing on features
that actually matter.

**Thrifty** is a good balance — pre-sorts features by importance once per round,
then processes in that order. Cheaper than greedy's O(n²) per-feature selection.

## Convergence

XGBoost tracks the largest weight change each round:

```python
largest_delta = max(|w_new - w_old| for all weights)
if largest_delta <= tolerance:
    stop_training()  # Converged
```

With `tolerance = 0` (default), training continues for all rounds.

## Bias Update

The bias term has no regularization — it's updated with simple gradient descent:

```python
grad_sum = Σ gradient_i
hess_sum = Σ hessian_i
bias += learning_rate × (-grad_sum / hess_sum)
```

## Data Format

Training uses **CSC (column-sparse-compressed)** format for efficient feature-wise access.
Each column stores (row_index, value) pairs for non-zero entries.

Why CSC?
- Coordinate descent iterates over features (columns)
- CSC gives O(nnz) access to all non-zeros in a column
- Row-major formats would require scanning all data per feature
