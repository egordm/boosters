# RFC-0009: GBLinear Training

- **Status**: Implemented
- **Created**: 2024-11-29
- **Depends on**: RFC-0008 (GBLinear Inference)
- **Scope**: Linear booster training via coordinate descent

## Summary

Add training support for GBLinear using coordinate descent optimization with
elastic net regularization (L1 + L2). This provides a simple training path
that validates gradient/objective infrastructure before tackling tree training.

## Motivation

- **Training infrastructure validation**: Test gradients and objectives before tree training
- **Complete GBLinear support**: Users can train and predict with linear models
- **Simpler than tree training**: No histograms, no split finding, no tree growing

## Design

### Overview

GBLinear training uses **coordinate descent**: update one weight at a time.
Each weight update has a closed-form solution with elastic net regularization.

Two variants:

- **Parallel** (default): Update all features with stale gradients
- **Sequential**: Update features one at a time with stale gradients

### Training Loop

```text
for round in 0..num_rounds:
    1. Compute predictions → gradients (grad, hess) from objective
    2. Update bias term (no regularization)
    3. Update feature weights via coordinate descent
    4. Check convergence (optional)
```

### Coordinate Descent Update

For weight `w_j`, the update uses soft thresholding for L1:

```text
grad_l2 = Σ(gradient × feature) + λ × w
hess_l2 = Σ(hessian × feature²) + λ
delta   = soft_threshold(-grad_l2 / hess_l2, α / hess_l2) × learning_rate
```

### Data Format: CSC

Training requires column-wise access for efficient per-feature gradients.
Convert from row-major input internally.

### Feature Selectors

Control feature update order. Start with:

- **Cyclic**: Simple sequential order
- **Shuffle**: Random permutation each round

XGBoost's greedy/thrifty selectors add complexity with marginal benefit.

## Design Decisions

### DD-1: CSC Format for Training

Coordinate descent iterates over features (columns). CSC gives O(nnz_in_column)
access. XGBoost uses CSC too.

### DD-2: Stale Gradient Updates (Differs from XGBoost)

**Our approach**: Compute gradients once per round, then update all features
using these "stale" gradients. No residual updates between features.

**XGBoost `coord_descent`**: Updates residuals after each feature, giving
exact gradients for subsequent features within the same round.

**Why we differ**:

1. **Performance**: No residual updates = faster per-round execution
2. **Simplicity**: Single gradient computation, parallel-friendly
3. **Empirically validated**: Achieves similar or better test RMSE than XGBoost

This is essentially "shotgun CD applied sequentially" — a valid optimization
used in many ML libraries. The stale gradients act as implicit momentum,
leading to a different but equally valid convergence path.

**Validation results** (from Story 5):

- Weight correlation with XGBoost: 0.91-0.95 (high)
- Test RMSE often better than XGBoost's sequential approach
- Binary classification produces reasonable logits

### DD-3: Simplified Feature Selectors

Start with cyclic and shuffle only. Greedy/thrifty can be added later if needed.

### DD-4: No GPU Training

XGBoost deprecated GPU coordinate descent. Linear models are memory-bound;
GPU transfer overhead exceeds benefit.

### DD-5: Reusable Objective Trait

Design the objective/gradient interface to be reusable for GBTree training.
Core trait computes `(grad, hess)` pairs from predictions and labels.
Evaluation metrics are separate from loss functions.

### DD-6: Training Infrastructure

Include from the start:

- **Early stopping**: Monitor validation metric, stop when no improvement
- **Evaluation**: Compute metrics on train/validation sets each round
- **Logging**: Verbosity levels (silent, warning, info, debug)

These apply to both GBLinear and future GBTree training.

## References

- Friedman et al. (2010) "Regularization Paths for GLMs via Coordinate Descent"

## Changelog

- 2024-11-29: Initial RFC approved
- 2025-11-29: DD-2 updated to document algorithmic difference from XGBoost
