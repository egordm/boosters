# GBLinear Overview

## What is GBLinear?

GBLinear is XGBoost's **linear booster** — an alternative to the tree-based GBTree.
It trains a generalized linear model (GLM) with **elastic net regularization** (L1 + L2).

### ELI5

A linear model is like a recipe: "prediction = 2×feature1 + 3×feature2 - 1×feature3 + bias".
You just multiply each feature by a weight and add them up. Training finds the best weights.

### ELI-Grad

GBLinear implements coordinate descent optimization for the elastic net objective:

```
minimize: L(w) + λ₁‖w‖₁ + λ₂‖w‖₂²
```

Where L(w) is the loss function defined by the objective (squared error, logistic, etc.),
λ₁ controls L1 sparsity, and λ₂ controls L2 weight decay.

## When to Use GBLinear

| Use Case | GBLinear | GBTree |
|----------|----------|--------|
| Data has linear relationships | ✅ Good | Overkill |
| Need interpretable coefficients | ✅ Good | Limited |
| High-dimensional sparse data | ✅ Often better | Can overfit |
| Complex feature interactions | ❌ Can't capture | ✅ Excels |
| Baseline comparison | ✅ Quick sanity check | — |
| Production latency matters | ✅ Simpler | More complex |

## Model Structure

The model is a weight matrix of shape `(num_features + 1) × num_output_groups`:

```
┌──────────────────────────────────────────┐
│  Feature 0:  [w₀₀, w₀₁, ..., w₀ₙ]       │  ← weights for each group
│  Feature 1:  [w₁₀, w₁₁, ..., w₁ₙ]       │
│  ...                                      │
│  Feature k:  [wₖ₀, wₖ₁, ..., wₖₙ]       │
│  BIAS:       [b₀,  b₁,  ..., bₙ]         │  ← last row is bias
└──────────────────────────────────────────┘
```

For binary classification, `num_output_groups = 1`.
For K-class classification, `num_output_groups = K`.

## Key Differences from GBTree

| Aspect | GBLinear | GBTree |
|--------|----------|--------|
| Model structure | Weight vector | Tree ensemble |
| Categorical features | ❌ Not supported | ✅ Supported |
| Feature interactions | ❌ None (linear) | ✅ Captures |
| Interpretability | ✅ Direct coefficients | Limited |
| Training speed | ✅ Very fast | Slower |
| Prediction speed | ✅ O(features) | O(trees × depth) |
| Memory usage | ✅ Minimal | Larger |

## Limitations

1. **No categorical features** — GBLinear requires numerical features only
2. **No prediction range** — Can't do layer slicing like GBTree
3. **No leaf indices** — Doesn't apply (no trees)
4. **No feature interactions** — SHAP interactions are all zeros
5. **Feature importance** — Only "weight" type (coefficient magnitude)
