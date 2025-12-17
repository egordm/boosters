# Gradient Boosting

Gradient boosting is a **method** for building predictive models by iteratively adding
weak learners that correct the errors of the ensemble. It is not an algorithm itself,
but a framework that can be instantiated with different base learners:

- **GBDT** (Gradient Boosted Decision Trees): Uses decision trees as base learners
- **GBLinear**: Uses linear models as base learners

This document explains the theoretical foundations common to all gradient boosting variants.

---

## The Core Idea

### ELI5

Imagine you're trying to guess someone's age from their photo. Your first guess might be
"30 years old" — but you're off by 5 years. So you make a second guess that tries to
correct that 5-year error. Then a third guess to correct any remaining error. And so on.

Gradient boosting works the same way: each new model focuses on fixing what the previous
models got wrong.

### ELI13

Gradient boosting builds a prediction model as a **sum of weak learners**:

```
F(x) = f₁(x) + f₂(x) + f₃(x) + ... + fₘ(x)
```

Each weak learner `fᵢ` is trained to predict the **residual errors** of all previous
learners combined. The key insight is that we can use gradient descent in function space:
instead of adjusting parameters, we add entire functions that point in the direction of
steepest descent.

### ELI-Grad

#### Functional Gradient Descent

Consider minimizing an expected loss over a training set:

$$\mathcal{L}(F) = \sum_{i=1}^{n} L(y_i, F(x_i))$$

In standard gradient descent, we update parameters: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$

In **functional gradient descent**, we update the function itself:

$$F \leftarrow F - \eta \cdot \nabla_F \mathcal{L}$$

The functional gradient at each point $x_i$ is:

$$-\nabla_F \mathcal{L} \big|_{x_i} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$$

This is exactly the **negative gradient** (or pseudo-residual) at each training point.

#### The Boosting Algorithm

At each iteration $m$:

1. Compute pseudo-residuals: $r_i^{(m)} = -\frac{\partial L(y_i, F^{(m-1)}(x_i))}{\partial F^{(m-1)}(x_i)}$

2. Fit a weak learner $h_m$ to the pseudo-residuals: $h_m \approx \arg\min_h \sum_i (r_i^{(m)} - h(x_i))^2$

3. Update the ensemble: $F^{(m)} = F^{(m-1)} + \eta \cdot h_m$

The learning rate $\eta$ controls how much each weak learner contributes.

---

## Loss Functions and Gradients

Different loss functions lead to different gradient boosting behaviors:

| Task | Loss Function | Gradient (Pseudo-Residual) |
|------|---------------|---------------------------|
| Regression | $\frac{1}{2}(y - F)^2$ | $y - F$ (residual) |
| Classification | $\log(1 + e^{-yF})$ | $y \cdot \sigma(-yF)$ |
| Ranking | LambdaRank | Pairwise gradients |

### Second-Order Approximation

XGBoost and LightGBM use a **second-order Taylor expansion** of the loss:

$$L(y, F + h) \approx L(y, F) + g \cdot h + \frac{1}{2} H \cdot h^2$$

Where:
- $g = \frac{\partial L}{\partial F}$ is the gradient
- $H = \frac{\partial^2 L}{\partial F^2}$ is the Hessian (second derivative)

Using the Hessian allows better approximation and enables Newton-like updates instead
of pure gradient descent. This is why modern implementations track both gradient and
Hessian for each sample.

---

## Regularization

Gradient boosting is prone to overfitting. Common regularization techniques:

### Shrinkage (Learning Rate)

Reduce the contribution of each weak learner:

$$F^{(m)} = F^{(m-1)} + \eta \cdot h_m, \quad \eta \in (0, 1]$$

Smaller $\eta$ requires more iterations but often achieves better generalization.

### Subsampling

Train each weak learner on a random subset of data:
- **Row subsampling**: Random fraction of training examples
- **Column subsampling**: Random fraction of features

### Structural Regularization

For tree-based learners:
- Limit tree depth
- Limit number of leaves
- Minimum samples per leaf
- L1/L2 regularization on leaf weights

---

## GBDT vs GBLinear

The gradient boosting framework can use any differentiable weak learner:

| Aspect | GBDT | GBLinear |
|--------|------|----------|
| Base learner | Decision tree | Linear model |
| Non-linearity | Captured by trees | None (linear) |
| Feature interactions | Automatic via splits | None |
| Interpretability | Lower | Higher |
| Speed | Slower | Faster |
| Best for | Complex patterns | Linear relationships |

### When to Use Each

**GBDT** excels when:
- Data has non-linear relationships
- Feature interactions matter
- You have sufficient training data

**GBLinear** excels when:
- Relationships are approximately linear
- You need fast training/inference
- Interpretability is important
- Data is very high-dimensional and sparse

---

## Historical Context

Gradient boosting was introduced by Friedman (2001) as a generalization of AdaBoost.
Key developments:

| Year | Development |
|------|-------------|
| 1997 | AdaBoost (Freund & Schapire) |
| 2001 | Gradient Boosting Machines (Friedman) |
| 2014 | XGBoost (Chen & Guestrin) — second-order, regularization |
| 2017 | LightGBM (Ke et al.) — histogram-based, leaf-wise |
| 2017 | CatBoost (Prokhorenkova et al.) — categorical handling |

Modern implementations (XGBoost, LightGBM) add:
- Second-order approximation (Hessian)
- Histogram-based split finding
- Native missing value handling
- Regularization in the objective

---

## Further Reading

### In This Documentation

- [GBDT](gbdt/) — Decision tree-based gradient boosting
- [GBLinear](gblinear/) — Linear model-based gradient boosting

### Academic Papers

- Friedman, J.H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

### Source Code

- XGBoost: `src/gbm/gbtree.cc`, `src/gbm/gblinear.cc`
- LightGBM: `src/boosting/gbdt.cpp`
