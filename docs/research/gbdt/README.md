# GBDT (Gradient Boosted Decision Trees)

GBDT uses **decision trees** as the weak learners in gradient boosting. Each tree is
trained to predict the pseudo-residuals (negative gradients) of the current ensemble,
progressively reducing the overall loss.

This is the most widely-used form of gradient boosting, implemented in XGBoost, LightGBM,
CatBoost, and other popular libraries.

---

## The Core Idea

GBDT builds an ensemble of trees sequentially:

```text
Prediction = Tree₁(x) + Tree₂(x) + Tree₃(x) + ... + Treeₘ(x)
```

Each tree is trained on the **gradient** of the loss function with respect to the current
predictions. For squared error loss, this is simply the residual (actual - predicted).
For other losses, it's the gradient that tells us "which direction should we move the
prediction to reduce error?"

Modern implementations (XGBoost, LightGBM) use a second-order Taylor expansion. For each
sample $i$ at a given tree node, we compute:

- Gradient: $g_i = \frac{\partial L(y_i, \hat{y}_i)}{\partial \hat{y}_i}$
- Hessian: $h_i = \frac{\partial^2 L(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$

The optimal leaf weight for a set of samples $I$ is:

$$w^* = -\frac{\sum_{i \in I} g_i}{\sum_{i \in I} h_i + \lambda}$$

Where $\lambda$ is L2 regularization.

---

## Why Trees?

Decision trees have properties that make them excellent weak learners:

| Property | Benefit |
|----------|---------|
| Non-linear | Can model complex relationships |
| Automatic feature selection | Splits on most informative features |
| Handle mixed types | Numerical and categorical (with encoding) |
| Missing value handling | Can learn default directions |
| Fast inference | Just follow decision rules |
| Interpretable splits | "If age > 30 AND income > 50k" |

The combination of many weak trees (high bias, low variance individually) creates a
strong learner with low bias and controlled variance.

---

## Training Overview

GBDT training follows this high-level loop:

```
Algorithm: GBDT Training
─────────────────────────────────────────
Input: Training data (X, y), loss function L, num_trees M
Output: Ensemble of trees {T₁, T₂, ..., Tₘ}

1. Initialize predictions: F₀(x) = base_score
2. For m = 1 to M:
   a. Compute gradients: gᵢ = ∂L/∂F for each sample
   b. Compute hessians:  hᵢ = ∂²L/∂F² for each sample
   c. Build tree Tₘ by:
      - Finding best splits using gradient statistics
      - Computing leaf weights from gradients/hessians
   d. Update predictions: Fₘ = Fₘ₋₁ + η · Tₘ
3. Return ensemble {T₁, ..., Tₘ}
```

See [Training](training/) for detailed algorithms.

---

## Modern Optimizations

Modern GBDT implementations use several key optimizations:

### Histogram-Based Training

Instead of sorting features to find splits (O(n log n)), modern implementations:
1. Quantize features into discrete bins
2. Build gradient histograms per bin
3. Find splits by scanning histograms (O(bins))

See [Histogram Method](training/histogram-training.md).

### Histogram Subtraction

For binary tree nodes: `child = parent - sibling`. This nearly halves the work by
only building histograms for the smaller child.

### Tree Growth Strategies

- **Depth-wise** (XGBoost default): Grow all nodes at the same level
- **Leaf-wise** (LightGBM default): Always split the best leaf

See [Tree Growth Strategies](training/tree-growth-strategies.md).

### Sampling

- **Row subsampling**: Use fraction of rows per tree
- **GOSS**: Keep large-gradient samples, sample small-gradient samples

See [Sampling Strategies](training/sampling-strategies.md).

---

## Contents

### Training

Algorithms for building trees efficiently:

| Document | Description |
|----------|-------------|
| [Histogram Method](training/histogram-training.md) | Core histogram-based split finding |
| [Quantization](training/quantization.md) | Feature binning and discretization |
| [Split Finding](training/split-finding.md) | Gain formula and split enumeration |
| [Tree Growth](training/tree-growth-strategies.md) | Depth-wise vs leaf-wise strategies |
| [Sampling](training/sampling-strategies.md) | GOSS and subsampling |

### Inference

Efficient prediction with trained models:

| Document | Description |
|----------|-------------|
| [Batch Traversal](inference/batch-traversal.md) | Block-based prediction, cache efficiency |
| [Multi-Output](inference/multi-output.md) | Multi-class and multi-target |

### Data Structures

Key representations for training and inference:

| Document | Description |
|----------|-------------|
| [Histogram Cuts](data-structures/histogram-cuts.md) | Bin boundaries and quantile sketches |
| [Quantized Data](data-structures/quantized-matrix.md) | Binned feature storage |
| [Tree Storage](data-structures/tree-storage.md) | AoS vs SoA tree layouts |

---

## Library Implementations

| Library | Source Files |
|---------|--------------|
| XGBoost | `src/tree/updater_quantile_hist.cc`, `src/gbm/gbtree.cc` |
| LightGBM | `src/treelearner/serial_tree_learner.cpp`, `src/boosting/gbdt.cpp` |

See [Library Comparison](../library-comparison.md) for detailed feature comparison.
