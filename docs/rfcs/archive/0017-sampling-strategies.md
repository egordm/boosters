# RFC-0017: Sampling Strategies

- **Status**: Draft
- **Created**: 2024-12-01
- **Updated**: 2024-12-01
- **Depends on**: RFC-0011 (GBTree Core), RFC-0014 (Split Finding)
- **Scope**: Row and column sampling during training

## Summary

This RFC defines sampling strategies for gradient boosting training:

1. **Random Row Sampling** — Subsample rows per iteration (`subsample`)
2. **Column Sampling** — Sample features at tree/level/node level (`colsample_*`)
3. **GOSS** — Gradient-based One-Side Sampling (LightGBM)

These techniques reduce training time and can improve generalization.

## Motivation

Full histogram building scans all rows × all features. Sampling reduces this:

- Row subsampling: `subsample=0.8` → ~20% fewer rows per iteration
- Column subsampling: `colsample_bytree=0.8` → ~20% fewer features
- GOSS: Keep top gradients + sample rest → focus on informative rows

Both XGBoost and LightGBM support these parameters.

## Design

### Row Sampling

**Random row subsampling**: Sample `subsample` fraction of rows at start of each iteration.

Key design points:

- Sort sampled indices for cache-friendly access
- All trees in multi-tree round see same sample
- Empty weight vector means uniform weights (avoid allocation)

### Column Sampling (Cascading)

Column sampling cascades: `bytree` → `bylevel` → `bynode`:

```text
Tree-level:  Select 90% of features → [f0, f1, f3, f4, f6, f8, f9]
Level-level: For depth 0, select 80% of tree features → [f0, f3, f4, f6, f9]
Node-level:  For each split, select 70% → varies per node
```

**Integration**: `GreedySplitFinder` receives allowed feature indices and only evaluates those.

### GOSS (Gradient-based One-Side Sampling)

GOSS samples rows based on gradient magnitude:

1. Keep top `top_rate` fraction by |gradient| (always included)
2. Randomly sample `other_rate` fraction of remainder
3. Weight sampled rows by `(1 - top_rate) / other_rate` to compensate

**Weighted histogram accumulation**: When weights are non-uniform, multiply gradient/hessian
by sample weight during histogram building.

### Integration Points

| Component | Integration |
|-----------|-------------|
| `GBTreeParams` | Add `subsample`, `ColumnSampling`, `SamplingStrategy` |
| `HistogramBuilder` | Add weighted accumulation variant |
| `GreedySplitFinder` | Accept feature index filter |
| Tree building loop | Sample rows at iteration start |

## Design Decisions

### DD-1: Sort Sampled Indices

**Decision**: Always sort sampled row indices before iteration.

**Rationale**: Sequential memory access is much faster than random. O(n log n) sort
cost is typically recovered in cache efficiency.

### DD-2: Weight Multiplier for GOSS

**Decision**: Weight remaining samples by `(1 - top_rate) / other_rate`.

**Rationale**: Compensates for sampling bias. If we keep 20% top + sample 10% of
remaining 80%, sampled rows represent 8% of data, so weight = 8.0.

### DD-3: Sample Per Iteration, Not Per Tree

**Decision**: Row sampling occurs at iteration start, not per tree.

**Rationale**: Matches XGBoost/LightGBM behavior. For multi-class with multiple trees
per iteration, all trees see the same sample.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `subsample` | 1.0 | Row subsample ratio (0, 1] |
| `colsample_bytree` | 1.0 | Column subsample per tree |
| `colsample_bylevel` | 1.0 | Column subsample per level |
| `colsample_bynode` | 1.0 | Column subsample per node |
| `top_rate` (GOSS) | 0.2 | Fraction to keep by gradient |
| `other_rate` (GOSS) | 0.1 | Fraction to sample from rest |

## Design Decisions (Continued)

### DD-4: No Sampling With Replacement

**Decision**: Sample without replacement only.

**Rationale**: Neither XGBoost nor LightGBM support sampling with replacement for
`subsample`/`bagging_fraction`. LightGBM explicitly states "randomly select part of
data without resampling." If established libraries don't implement it, there's
likely good reason (e.g., efficiency, boosting theory).

### DD-5: No Stratified Sampling (For Now)

**Decision**: Defer stratified sampling to future work.

**Rationale**: Neither XGBoost nor LightGBM implement general stratified sampling.
LightGBM has `pos_bagging_fraction`/`neg_bagging_fraction` for binary imbalance,
but this is not full stratification. If established libraries haven't prioritized
this, we can evaluate later based on user demand.

## Testing Strategy

### Unit Tests

- Row sampling produces correct fraction of rows
- Column sampling cascades correctly (bytree → bylevel → bynode)
- GOSS selects top gradient rows correctly
- Weight amplification computed correctly for GOSS
- Weighted histogram accumulation is accurate

### Integration Tests

- Subsampled training converges to reasonable accuracy
- GOSS produces valid models on standard datasets
- Seeds produce reproducible results

### Performance Tests

- Measure training time: subsampled vs full data
- Expected: ~linear speedup with subsample ratio
- Document any deviations from expected speedup

### Validation Tests

- Compare accuracy of sampled training vs full training
- Tolerance: accuracy drop < 2% with `subsample=0.8`
- Compare against XGBoost/LightGBM with same parameters
- If accuracy differs significantly, investigate source code

### Qualitative Tests

- Train on overfitting-prone dataset with/without subsampling
- Verify subsampling reduces overfitting (test > train gap)
- Set expectations before training; investigate if not met

## References

- [LightGBM GOSS paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [XGBoost sampling parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)
