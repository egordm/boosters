# Loss Functions: LightGBM vs XGBoost vs booste-rs

This document compares objective/loss functions across LightGBM, XGBoost, and booste-rs (current implementation).

## Regression Objectives

| Loss Function | XGBoost | LightGBM | booste-rs | Notes |
|---------------|---------|----------|-----------|-------|
| **L2 (MSE)** | `reg:squarederror` | `regression` | `SquaredError` | Standard regression |
| **L1 (MAE)** | `reg:absoluteerror` | `regression_l1` | `AbsoluteError` | Median regression |
| **Huber** | `reg:pseudohubererror` | `huber` | ❌ | Robust regression |
| **Fair** | ❌ | `fair` | ❌ | Robust, less sensitive than L1 |
| **Poisson** | `count:poisson` | `poisson` | `Poisson` | Count data |
| **Gamma** | `reg:gamma` | `gamma` | `Gamma` | Positive continuous |
| **Tweedie** | `reg:tweedie` | `tweedie` | `Tweedie` | Compound Poisson-Gamma |
| **Quantile** | `reg:quantileerror` | `quantile` | ❌ | Percentile regression |
| **MAPE** | ❌ | `mape` | ❌ | Mean absolute percentage |
| **Log-Squared** | `reg:squaredlogerror` | ❌ | ❌ | For positive targets |

## Binary Classification

| Loss Function | XGBoost | LightGBM | booste-rs | Notes |
|---------------|---------|----------|-----------|-------|
| **Logistic** | `binary:logistic` | `binary` | `BinaryLogistic` | Returns probability |
| **Logit Raw** | `binary:logitraw` | ❌ | `BinaryLogitRaw` | Returns raw score |
| **Cross-Entropy** | ❌ | `cross_entropy` | ❌ | Labels in [0,1] |
| **Cross-Entropy Lambda** | ❌ | `cross_entropy_lambda` | ❌ | Alternative parameterization |
| **Hinge** | `binary:hinge` | ❌ | ❌ | SVM-style |

## Multiclass Classification

| Loss Function | XGBoost | LightGBM | booste-rs | Notes |
|---------------|---------|----------|-----------|-------|
| **Softmax** | `multi:softmax` | `multiclass` | `MultiSoftmax` | Returns class |
| **Softprob** | `multi:softprob` | `multiclassova` | `MultiSoftprob` | Returns probabilities |

## Ranking

| Loss Function | XGBoost | LightGBM | booste-rs | Notes |
|---------------|---------|----------|-----------|-------|
| **Pairwise** | `rank:pairwise` | `lambdarank` | `RankPairwise` | LambdaRank |
| **NDCG** | `rank:ndcg` | `lambdarank` | `RankNdcg` | NDCG optimization |
| **MAP** | `rank:map` | `lambdarank` | `RankMap` | MAP optimization |
| **XE-NDCG** | ❌ | `rank_xendcg` | ❌ | Softmax NDCG |

## Survival Analysis

| Loss Function | XGBoost | LightGBM | booste-rs | Notes |
|---------------|---------|----------|-----------|-------|
| **Cox** | `survival:cox` | ❌ | `SurvivalCox` | Proportional hazards |
| **AFT** | `survival:aft` | ❌ | ❌ | Accelerated failure time |

---

## Detailed Gradient/Hessian Comparison

### L2 (Squared Error)

Both implementations are identical:

```
gradient = prediction - label
hessian = 1.0
```

### L1 (Absolute Error)

**XGBoost:**
```cpp
gradient = sign(pred - label)
hessian = 1.0
```

**LightGBM:**
```cpp
gradient = sign(pred - label) * weight
hessian = weight
// Plus: RenewTreeOutput using weighted median
```

LightGBM additionally uses **leaf output renewal** with weighted percentile.

### Huber Loss

**LightGBM only** (`huber`):

```cpp
// α = config.alpha (delta threshold)
diff = pred - label
if |diff| <= α:
    gradient = diff
else:
    gradient = sign(diff) * α
hessian = 1.0
```

**XGBoost** uses **Pseudo-Huber** instead:
```cpp
// Smooth approximation
gradient = diff / sqrt(1 + (diff/δ)²)
hessian = 1 / (1 + (diff/δ)²)^(3/2)
```

### Fair Loss (LightGBM only)

```cpp
// c = config.fair_c
x = pred - label
gradient = c * x / (|x| + c)
hessian = c² / (|x| + c)²
```

Provides smooth transition between L2 (small errors) and L1 (large errors).

### Quantile Regression

**XGBoost:**
```cpp
if (pred - label) >= 0:
    gradient = (1 - α) * weight
else:
    gradient = -α * weight
hessian = weight
```

**LightGBM:**
```cpp
// Same gradient formula
// Plus: RenewTreeOutput using weighted percentile at α
```

### Binary Classification

**XGBoost:**
```cpp
p = sigmoid(pred)
gradient = p - label
hessian = max(p * (1 - p), eps)
```

**LightGBM** (`binary`):
```cpp
// With sigmoid parameter
response = -label * sigmoid / (1 + exp(label * sigmoid * pred))
gradient = response * label_weight
hessian = |response| * (sigmoid - |response|) * label_weight
```

**LightGBM** (`cross_entropy`):
```cpp
// More numerically stable implementation
if pred > -37:
    exp_tmp = exp(-pred)
    gradient = ((1 - label) - label * exp_tmp) / (1 + exp_tmp)
    hessian = exp_tmp / (1 + exp_tmp)²
else:
    exp_tmp = exp(pred)
    gradient = exp_tmp - label
    hessian = exp_tmp
```

### LambdaRank (Ranking)

Both implementations use similar NDCG-based lambda gradients:

```cpp
for each pair (i, j) where label[i] > label[j]:
    delta_score = score[i] - score[j]
    delta_ndcg = |DCG_swap| * inverse_max_dcg
    
    p_lambda = sigmoid(delta_score) * delta_ndcg
    p_hessian = sigmoid' * delta_ndcg
    
    gradient[i] += p_lambda
    gradient[j] -= p_lambda
```

**LightGBM additions:**
- Position bias correction
- Normalization option (`lambdarank_norm`)
- Sigmoid table caching for speed

### XE-NDCG (LightGBM only)

Uses **cross-entropy over softmax** for smoother gradients:

```cpp
// Softmax over scores
rho = softmax(scores)
// Ground truth distribution
phi[i] = 2^label[i] - random()

// Approximate gradients using Taylor expansion
gradient = -phi/sum(phi) + rho + second_order_terms + third_order_terms
hessian = rho * (1 - rho)
```

---

## Key Differences Summary

### Numerical Stability

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Sigmoid | Basic | Branch-free stable implementation |
| Exp overflow | Clamping | Cutoff at -37 |
| Hessian floor | eps = 1e-16 | Varies by objective |

### Leaf Output Optimization

**LightGBM** supports **leaf output renewal** for certain objectives:
- L1/Quantile: Recompute leaf value as weighted percentile
- Provides better convergence for median-based losses

**XGBoost** uses adaptive tree leaf update for quantile regression.

### Weight Handling

Both support sample weights, but LightGBM has additional:
- `scale_pos_weight` in binary classification
- `is_unbalance` auto-weighting
- Position-dependent weights in ranking

---

## booste-rs Recommendations

### Current Coverage ✅

The `Objective` enum covers main use cases:
- Regression: SquaredError, AbsoluteError, Poisson, Gamma, Tweedie
- Classification: BinaryLogistic, BinaryLogitRaw, MultiSoftmax, MultiSoftprob
- Ranking: RankPairwise, RankNdcg, RankMap
- Survival: SurvivalCox

### Missing (Consider Adding)

**High Priority:**
1. **Huber/Pseudo-Huber** - Common robust regression
2. **Quantile** - Important for uncertainty estimation

**Medium Priority:**
3. **Fair** - Useful for outlier-robust regression
4. **MAPE** - Common in forecasting

**Low Priority:**
5. Cross-Entropy variants (niche)
6. XE-NDCG (specialized ranking)
7. AFT survival (specialized)

### Implementation Notes

For **training**, need to implement:
- `GetGradients(preds, labels) -> (gradients, hessians)`
- `BoostFromScore(labels) -> initial_score`

For **inference only**, current `Objective::transform()` is sufficient.

### Suggested Additions

```rust
pub enum Objective {
    // ... existing ...
    
    // NEW: Robust regression
    Huber { delta: f32 },
    PseudoHuber { delta: f32 },
    Fair { c: f32 },
    
    // NEW: Percentile regression  
    Quantile { alpha: f32 },
    
    // NEW: Error metrics
    MAPE,
}
```
