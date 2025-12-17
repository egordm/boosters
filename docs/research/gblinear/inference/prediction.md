# Prediction & Optimization

How inference works for linear models, and why there are fewer optimizations than tree methods.

## Inference Algorithm

Prediction is a weighted sum — the simplest possible model evaluation:

$$
\text{output}[g] = \text{bias}[g] + \sum_{j} x_j \cdot w_{j,g}
$$

Where:
- $x_j$ = feature value
- $w_{j,g}$ = weight for feature $j$, output group $g$
- $\text{bias}[g]$ = bias term for output group $g$

### Dense Features

```python
def predict_linear(features, weights, bias, num_groups):
    outputs = []
    for group in range(num_groups):
        total = bias[group]
        for feat_idx, value in enumerate(features):
            total += value * weights[feat_idx, group]
        outputs.append(total)
    return outputs
```

### Sparse Features

For sparse data, only non-zero features contribute:

```python
def predict_sparse(indices, values, weights, bias, num_groups):
    outputs = [bias[g] for g in range(num_groups)]
    for idx, value in zip(indices, values):
        for group in range(num_groups):
            outputs[group] += value * weights[idx, group]
    return outputs
```

---

## Base Score and Base Margin

Like tree models, linear models add a base score to predictions:

```text
output = bias + base_score + Σ(feature × weight)
```

If per-row `base_margin` values are provided, those override the global base score:

```text
output = bias + base_margin[row] + Σ(feature × weight)
```

---

## Feature Contributions

Since the model is linear, feature contributions are trivial to compute:

$$
\text{contribution}[j] = x_j \cdot w_j
$$

$$
\text{contribution}[\text{bias}] = \text{bias} + \text{base\_score}
$$

**Note**: Feature interactions are always zero. Linear models have no interaction terms — each feature contributes independently.

---

## Complexity

| Operation | Complexity |
|-----------|------------|
| Single row (dense) | O(features × groups) |
| Single row (sparse, K non-zeros) | O(K × groups) |
| Batch of N rows | O(N × features × groups) |

Compare to tree models: O(trees × depth) per row.

For most practical cases, linear prediction is **faster** because:
- `features` is typically less than `trees × depth`
- Memory access is sequential (no tree pointer chasing)
- Sparse data skips zero features entirely

---

## Threading

### Batch Parallelism

XGBoost parallelizes over **rows** during batch prediction:

```cpp
parallel_for(batch.size(), num_threads, [&](row_idx) {
    for (group in 0..num_groups) {
        preds[row_idx * num_groups + group] = predict(row[row_idx], group);
    }
});
```

Each row's prediction is independent, so this parallelizes perfectly with no synchronization.

### Within-Row Parallelism?

XGBoost does **not** parallelize the feature sum within a single row. Why?

1. **Too little work** — A single dot product is fast; threading overhead dominates
2. **Memory-bound** — The bottleneck is loading weights, not computing sums
3. **Good enough** — Row-level parallelism saturates cores for batch prediction

---

## Why Fewer Optimizations Than Trees?

Linear models have significantly fewer optimizations than tree methods. Here's why:

### 1. Inherent Simplicity

The prediction loop is just:

```cpp
for (auto& entry : sparse_row) {
    sum += entry.fvalue * weights[entry.index];
}
```

There's not much to optimize. The compiler handles it well.

### 2. Memory-Bound Workload

| Operation | Compute | Memory |
|-----------|---------|--------|
| Load feature value | 0 | 1 read (4-8 bytes) |
| Load weight | 0 | 1 read (4-8 bytes) |
| Multiply-add | 2 ops | 0 |

**Arithmetic intensity**: ~2 ops per 8-16 bytes ≈ 0.125-0.25 ops/byte

Modern CPUs can sustain 10+ ops per byte of memory bandwidth. This workload is **heavily memory-bound** — faster compute doesn't help.

### 3. Sparse Data Issues

Most GBLinear use cases involve sparse features. This creates problems for:

**SIMD**: Requires contiguous data, but sparse iteration jumps around randomly. Gather instructions exist but are slow.

**Prefetching**: Access patterns are data-dependent and unpredictable.

**Caching**: Random access patterns defeat cache locality.

---

## Optimization Comparison: Linear vs Tree

| Optimization | Tree Models | Linear Models | Why? |
|--------------|-------------|---------------|------|
| Thread parallelism | ✅ Yes | ✅ Yes | Both benefit from row parallelism |
| SIMD | ❌ Limited | ❌ No | Gather operations too slow |
| GPU inference | ✅ Yes | ❌ No | Data transfer > compute time |
| GPU training | ✅ Yes | ⚠️ Deprecated | Same reason |
| Block processing | ✅ Yes | ❌ No | Not beneficial for dot products |
| Histogram tricks | ✅ Yes | N/A | Training only, trees only |
| SoA layout | ✅ Yes | N/A | Already minimal structure |

### GPU: Why Not?

Linear models on GPU face fundamental issues:

1. **Data transfer overhead**: Copying sparse features to GPU costs more than the computation
2. **Low arithmetic intensity**: Just one multiply-add per non-zero. GPUs need high compute/memory ratios.
3. **Small models**: Linear models are tiny (just weights). No benefit from massive parallelism.

XGBoost's GPU training for linear models (`gpu_coord_descent`) is deprecated as of v2.0.

---

## Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `booster` | — | Set to `'gblinear'` for linear model |
| `eta` (learning_rate) | 0.5 | Shrinkage factor for weight updates |
| `lambda` (reg_lambda) | 0.0 | L2 regularization strength |
| `alpha` (reg_alpha) | 0.0 | L1 regularization strength |

**Note**: GBLinear's default learning rate (0.5) is higher than GBTree's (0.3). Linear models are simpler and tolerate larger steps.

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `updater` | `'shotgun'` | `'shotgun'` (parallel) or `'coord_descent'` (sequential) |
| `feature_selector` | `'cyclic'` | How to order feature updates |
| `top_k` | 0 | Limit to top-k features (thrifty/greedy only) |
| `tolerance` | 0.0 | Early stop if max weight change < tolerance |

### Parameter Effects

**Learning rate (`eta`)**:
- Lower → more conservative, may need more rounds
- Higher → faster training, risk of instability

**L2 regularization (`lambda`)**:
- Higher → smaller weights, less overfitting
- Too high → underfitting

**L1 regularization (`alpha`)**:
- Higher → more weights become exactly zero
- Useful for automatic feature selection

**Updater**:
- `shotgun` → faster, parallel, slight approximation
- `coord_descent` → slower, exact, supports all selectors

---

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

Good for initial experimentation.

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

Use when you want the model to automatically select features.

### Maximum Stability

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

Use when shotgun isn't converging or you need reproducibility.

### High-Dimensional Sparse Data

```python
params = {
    'booster': 'gblinear',
    'updater': 'coord_descent',
    'feature_selector': 'greedy',  # or 'thrifty' for speed
    'top_k': 1000,
    'lambda': 0.01,
    'alpha': 0.5,
    'eta': 0.3,
}
```

Focus computation on the most important features.

---

## Summary

| Aspect | Details |
|--------|---------|
| Prediction | Simple dot product: $\sum x_j \cdot w_j + \text{bias}$ |
| Complexity | O(features) dense, O(nnz) sparse |
| Threading | Row-level parallelism only |
| SIMD | Not used (sparse data, memory-bound) |
| GPU | Not supported (overhead > benefit) |
| Bottleneck | Memory bandwidth, not compute |

Linear models are simple, fast, and interpretable — but this simplicity means there's less room for clever optimizations. The best approach is straightforward: parallelize across rows and use efficient sparse data structures.
