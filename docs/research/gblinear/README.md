# GBLinear

GBLinear is a **linear booster** — an alternative to tree-based boosting that trains a
generalized linear model (GLM) with **elastic net regularization** (L1 + L2).

This is a simpler model than GBDT, but can be effective for problems with linear
relationships or as a quick baseline.

---

## Contents

| Folder | Description |
|--------|-------------|
| [training/](training/coordinate-descent.md) | Coordinate descent optimization |
| [inference/](inference/prediction.md) | Linear prediction |

---

## The Core Idea

A linear model predicts by summing weighted features:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \ldots + w_n x_n = \mathbf{w}^T \mathbf{x} + b$$

GBLinear minimizes the elastic net objective:

$$
\min_{\mathbf{w}} \quad \mathcal{L}(\mathbf{w}) + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2
$$

Where:
- $\mathcal{L}(\mathbf{w})$ is the loss (squared error, logistic, etc.)
- $\lambda_1$ controls L1 sparsity (feature selection)
- $\lambda_2$ controls L2 weight decay (stability)

---

## When to Use Linear vs Tree Models

| Use Case | Linear | Tree |
|----------|--------|------|
| Data has linear relationships | ✅ Good | Overkill |
| Need interpretable coefficients | ✅ Good | Limited |
| High-dimensional sparse data | ✅ Often better | Can overfit |
| Complex feature interactions | ❌ Can't capture | ✅ Excels |
| Baseline comparison | ✅ Quick sanity check | — |
| Production latency matters | ✅ Simpler | More complex |

### Linear Model Strengths

- **Speed**: Training and inference are very fast (just multiply-add)
- **Interpretability**: Coefficients directly show feature importance
- **Sparsity**: L1 regularization can zero out irrelevant features
- **Memory**: Minimal — just a weight vector

### Linear Model Limitations

- **No categorical features** — Requires numerical inputs only
- **No feature interactions** — Can't capture XOR-like patterns
- **No tree-specific features** — No leaf indices, prediction layers, etc.
- **SHAP interactions** — Always zero (linear models have no interactions)

---

## Key Concepts

### Elastic Net Regularization

Combines L1 (Lasso) and L2 (Ridge) regularization:

| Type | Penalty | Effect |
|------|---------|--------|
| L1 (Lasso) | $\sum \|w_i\|$ | Sparse weights — some become exactly 0 |
| L2 (Ridge) | $\sum w_i^2$ | Small weights — stable, keeps all features |
| Elastic Net | Both | Best of both — sparse, stable, handles correlation |

### Coordinate Descent

Instead of updating all weights at once (gradient descent), coordinate descent updates **one weight at a time** while holding others fixed. For convex problems with separable regularization, each update has a closed-form solution.

### Soft Thresholding

The L1 penalty creates a "dead zone" around zero. The soft thresholding operator (proximal operator for L1) pushes small weights toward zero and can make them exactly zero — this is how L1 achieves sparsity.

---

## Model Structure

The model is a weight matrix of shape `(num_features + 1) × num_output_groups`:

```text
┌──────────────────────────────────────────┐
│  Feature 0:  [w₀₀, w₀₁, ..., w₀ₙ]       │  ← weights for each output group
│  Feature 1:  [w₁₀, w₁₁, ..., w₁ₙ]       │
│  ...                                      │
│  Feature k:  [wₖ₀, wₖ₁, ..., wₖₙ]       │
│  BIAS:       [b₀,  b₁,  ..., bₙ]         │  ← last row is bias (no regularization)
└──────────────────────────────────────────┘
```

| Configuration | Output Groups |
|---------------|---------------|
| Binary classification | 1 |
| K-class classification | K |
| Regression | 1 |

---

## XGBoost Source Files

| File | Purpose |
|------|---------|
| `src/gbm/gblinear.cc` | Main booster implementation |
| `src/gbm/gblinear_model.h` | Weight matrix structure |
| `src/linear/param.h` | Training parameters |
| `src/linear/coordinate_common.h` | Coordinate update algorithms |
| `src/linear/updater_shotgun.cc` | Parallel (lock-free) updater |
| `src/linear/updater_coordinate.cc` | Sequential updater |

---

## Comparison: Linear vs Tree Booster

| Aspect | Linear | Tree |
|--------|--------|------|
| Model structure | Weight vector | Tree ensemble |
| Categorical features | ❌ Not supported | ✅ Supported |
| Feature interactions | ❌ None (linear) | ✅ Captures |
| Interpretability | ✅ Direct coefficients | Limited |
| Training speed | ✅ Very fast | Slower |
| Prediction speed | ✅ O(features) | O(trees × depth) |
| Memory usage | ✅ Minimal | Larger |
| Hyperparameters | Few (just regularization) | Many (tree structure) |
