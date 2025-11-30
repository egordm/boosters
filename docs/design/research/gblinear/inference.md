# GBLinear Inference

How prediction works in GBLinear — it's just weighted sums.

## The Algorithm

For a single row of features, prediction is:

```
output[group] = bias[group] + Σ(feature_value × coefficient[feature, group])
```

That's it. No trees, no comparisons, no branching — just multiply-add.

### Pseudocode

```python
def predict_linear(features, weights, bias, num_groups):
    outputs = []
    for group in range(num_groups):
        total = bias[group]
        for feat_idx, value in enumerate(features):
            total += value * weights[feat_idx][group]
        outputs.append(total)
    return outputs
```

## Sparse Features

For sparse data, only non-zero features contribute:

```python
def predict_sparse(indices, values, weights, bias, num_groups):
    outputs = [bias[g] for g in range(num_groups)]
    for idx, value in zip(indices, values):
        for group in range(num_groups):
            outputs[group] += value * weights[idx][group]
    return outputs
```

## Batch Prediction

XGBoost parallelizes over rows — each row's prediction is independent:

```python
def predict_batch(data, weights, bias):
    outputs = []
    parallel_for row in data:  # Can run in parallel
        outputs.append(predict_linear(row, weights, bias))
    return outputs
```

## Base Score / Base Margin

Like GBTree, GBLinear adds a `base_score` to predictions. If per-row `base_margin`
values are provided (via DMatrix), those are used instead:

```python
output = bias + base_score  # or base_margin[row] if provided
output += Σ(feature × weight)
```

## Feature Contributions

Since GBLinear is linear, feature contributions are trivial:

```python
contribution[feature] = feature_value × coefficient[feature, group]
contribution[bias_idx] = bias[group] + base_score
```

**Note**: Feature interactions are always zero — linear models have no interactions.

## Complexity

| Operation | Complexity |
|-----------|------------|
| Single row | O(features × groups) |
| Batch of N rows | O(N × features × groups) |
| Sparse row with K non-zeros | O(K × groups) |

Compare to GBTree: O(trees × depth) per row.
