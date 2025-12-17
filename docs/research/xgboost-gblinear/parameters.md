# GBLinear Parameters

Configuration options for XGBoost's linear booster.

## Booster-Level Parameters

Set via `booster='gblinear'` or in the parameters dict.

### `updater`

**Default**: `"shotgun"`

Which coordinate descent algorithm to use:

| Value | Description |
|-------|-------------|
| `shotgun` | Parallel updates across features. Fast, slight approximation. |
| `coord_descent` | Sequential updates. Slower, exact gradients. |

**When to use coord_descent**: When using greedy/thrifty feature selection, or if
shotgun isn't converging well.

### `tolerance`

**Default**: `0.0`

Convergence threshold. If the largest weight change in a round is smaller than this,
training stops early.

- `0.0` = Disabled, run all rounds
- `1e-4` = Stop when weights stabilize

### `feature_selector`

**Default**: `"cyclic"`

How to choose which feature to update next:

| Value | Description | Best For |
|-------|-------------|----------|
| `cyclic` | Sequential: 0, 1, 2, ... | General use |
| `shuffle` | Random permutation each round | Breaking patterns |
| `random` | Random with replacement | ? |
| `thrifty` | Sort by gradient magnitude | Sparse data, faster convergence |
| `greedy` | Always pick highest gradient | Small feature sets, best convergence |

**Note**: `shotgun` updater only supports `cyclic` and `shuffle`.

### `top_k`

**Default**: `0` (all features)

For `thrifty` and `greedy` selectors, limit to the top-k features by gradient magnitude.
Reduces computation when most features are irrelevant.

## Regularization Parameters

### `lambda` (reg_lambda)

**Default**: `0.0`

L2 regularization strength. Encourages small weights (weight decay).

- Higher values → smaller weights, less overfitting
- Too high → underfitting

### `alpha` (reg_alpha)

**Default**: `0.0`

L1 regularization strength. Encourages sparse weights (feature selection).

- Higher values → more weights become exactly zero
- Useful for feature selection

### `learning_rate` (eta)

**Default**: `0.5`

Shrinkage factor applied to weight updates.

- Lower values → more conservative updates, may need more rounds
- Higher values → faster training, risk of instability

**Note**: GBLinear's default (0.5) is much higher than GBTree's default (0.3).
Linear models are simpler and can tolerate larger steps.

## Comparison: GBLinear vs GBTree Regularization

| Parameter | GBLinear | GBTree |
|-----------|----------|--------|
| `lambda` | L2 on weights | L2 on leaf weights |
| `alpha` | L1 on weights | L1 on leaf weights |
| `gamma` | N/A | Min split loss |
| `max_depth` | N/A | Tree depth limit |
| `min_child_weight` | N/A | Min hessian sum |

GBLinear has fewer hyperparameters — complexity is controlled only through regularization.

## Typical Configurations

### Fast Iteration (Default)

```python
params = {
    'booster': 'gblinear',
    'updater': 'shotgun',
    'feature_selector': 'cyclic',
    'lambda': 0,
    'alpha': 0,
    'eta': 0.5,
}
```

### Feature Selection

```python
params = {
    'booster': 'gblinear',
    'updater': 'coord_descent',
    'feature_selector': 'thrifty',
    'lambda': 0.1,
    'alpha': 1.0,  # Strong L1 for sparsity
    'eta': 0.3,
}
```

### Stability Focus

```python
params = {
    'booster': 'gblinear',
    'updater': 'coord_descent',
    'feature_selector': 'shuffle',
    'lambda': 1.0,  # Strong L2
    'alpha': 0.1,
    'eta': 0.1,
    'tolerance': 1e-5,
}
```
