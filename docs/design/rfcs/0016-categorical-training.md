# RFC-0016: Categorical Feature Training

- **Status**: Draft
- **Created**: 2024-12-01
- **Updated**: 2024-12-01
- **Depends on**: RFC-0011 (Quantization), RFC-0012 (Histograms), RFC-0013 (Split Finding)
- **Scope**: Native categorical feature support during GBTree training

## Summary

This RFC defines how categorical features are handled during training without one-hot encoding.
It implements LightGBM-style gradient-sorted categorical splits that find optimal binary
partitions in O(k log k) time for k categories.

**Key point**: This reuses existing categorical **inference** infrastructure:

- `SplitType::Categorical` and bitset storage in `SoATreeStorage`
- `SplitInfo.is_categorical` and `categories_left` (already present)
- `RowPartitioner` already handles categorical partitioning

The addition is the **split-finding algorithm** that determines which categories go left vs right.

## Motivation

One-hot encoding categorical features is problematic:

1. **Memory explosion**: A feature with 1000 categories becomes 1000 binary features
2. **Split dilution**: Tree must use many splits to capture category effects
3. **Histogram inefficiency**: Sparse features have mostly empty bins

### XGBoost vs LightGBM Approach

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Method | One-vs-all | Gradient-sorted binary partition |
| Complexity | O(k) splits per node | O(k log k) for optimal partition |
| Typical use | Low cardinality (< 10) | Any cardinality |

**Decision**: LightGBM-style gradient-sorted approach for better accuracy on high-cardinality.

## Design

### Overview

```
Categorical Histogram                     Sorted by gradient/hess ratio
┌────────────────────────┐               ┌─────────────────────────────┐
│ cat  │ grad  │ hess   │               │ cat  │ grad  │ hess │ ratio │
│──────│───────│────────│    sort by    │──────│───────│──────│───────│
│  A   │ -2.0  │  1.0   │   grad/hess   │  C   │ -1.0  │  2.0 │ -0.50 │
│  B   │  3.0  │  2.0   │ ──────────▶   │  A   │ -2.0  │  1.0 │ -2.00 │
│  C   │ -1.0  │  2.0   │               │  B   │  3.0  │  2.0 │  1.50 │
│  D   │  1.5  │  1.5   │               │  D   │  1.5  │  1.5 │  1.00 │
└────────────────────────┘               └─────────────────────────────┘
                                                      │
                                         Linear scan for best split
                                                      ▼
                                         Best split: {C, A} vs {B, D}
```

### Algorithm

1. **Extract** non-empty categories from histogram (already bin-indexed)
2. **Sort** categories by `grad_sum / hess_sum` ratio
3. **Scan** sorted order to find optimal split point (maximize gain)
4. **Output** `SplitInfo` with `is_categorical = true` and `categories_left` populated

### Integration Points

| Component | What Exists | What to Add |
|-----------|-------------|-------------|
| `BinCuts` | Bin storage | `is_categorical` flag per feature |
| `SplitInfo` | `is_categorical`, `categories_left` fields | Already complete |
| `GreedySplitFinder` | Numerical split finding | Route to categorical finder |
| `RowPartitioner` | Categorical partitioning via bitset | Already complete |
| `SoATreeStorage` | Bitset storage for categorical | Already complete |

**GreedySplitFinder integration**: Route to categorical split finder when feature is categorical.
The categorical finder returns a `SplitInfo` with `is_categorical = true`.

**Tree Conversion**: `categories_left` → bitset conversion already exists (`categories_to_bitset()`
in trainer.rs).

### Missing Value Handling

Missing values (NaN → bin 0) use **default direction**. During split finding, compute gain
for missing going left vs right and choose the better option. Consistent with numerical features.

### High-Cardinality Cutoff

For features with many categories:

- `max_categories` (default 32): Only consider top categories by sample count
- `max_cat_to_onehot` (default 4): Use one-hot style if ≤ threshold categories

## Design Decisions

### DD-1: Gradient-Sorted vs One-vs-All

**Decision**: Gradient-sorted approach (LightGBM-style).

**Rationale**: Finds globally optimal binary partition for convex loss. Single split
captures complex category effects, better for high-cardinality.

### DD-2: Missing Value Direction

**Decision**: Learn default direction based on gain comparison.

**Rationale**: Consistent with numerical features. Missing values can be grouped
with similar categories if beneficial.

### DD-3: Reuse Existing Infrastructure

**Decision**: Reuse `SplitInfo.is_categorical`, `categories_left`, and bitset storage.

**Rationale**: Inference already works. Only need to add training-side split finding.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_categories` | 32 | Max categories for gradient-sorted split |
| `max_cat_to_onehot` | 4 | Use one-hot if ≤ this many categories |

## Design Decisions (Continued)

### DD-4: Category Smoothing

**Decision**: Implement category smoothing (like LightGBM's `cat_smooth`).

**Rationale**: Categories with few samples can produce noisy gradient estimates.
Smoothing regularizes by blending category statistics with global statistics.
Adds robustness without significant complexity.

### DD-5: Unseen Categories at Inference

**Decision**: Use default direction (same as missing values).

**Rationale**: Unseen categories are semantically similar to missing — we have no
information about them. Default direction is simple, consistent, and robust.

## Testing Strategy

### Unit Tests

- Gradient accumulation by category produces correct sums
- Gradient-sorted partition finding selects optimal split
- Bitset generation for categories-left is correct
- Category smoothing applies regularization correctly

### Integration Tests

- Categorical splits integrate with tree growing
- Trained models produce reasonable predictions on categorical data
- Output compatible with existing inference code

### Validation Tests

- Compare trained model predictions against XGBoost/LightGBM baselines
- Tolerance: predictions within 1e-2 for same hyperparameters
- If deviations exceed tolerance, investigate source code differences

### Qualitative Tests

- Train on datasets with known categorical structure
- Verify model learns expected patterns (e.g., category A > category B)
- Set accuracy expectations before training; investigate if not met

## References

- [LightGBM Categorical Features](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support)
- [Phase 2 Research Notes](../research/phase2-notes.md)
