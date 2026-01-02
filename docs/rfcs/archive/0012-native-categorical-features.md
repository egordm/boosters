# RFC-0012: Native Categorical Feature Support

- **Status**: Implemented
- **Created**: 2025-12-18
- **Updated**: 2025-01-25
- **Depends on**: RFC-0004 (Binning and Histograms)
- **Scope**: Native categorical feature handling with optimal partitioning

## Summary

Native categorical support enables optimal split finding without one-hot encoding.
Eliminates 10-100× overhead of one-hot encoding and improves model quality on
high-cardinality categoricals.

## Motivation

One-hot encoding categorical features has severe limitations:

- **Memory**: 100 categories → 100 features → 100× histogram memory
- **Quality**: Treats categories independently, can't group similar ones
- **Compute**: Build 100 histograms vs 1

### When to Use

| Scenario | Approach |
| -------- | -------- |
| Raw categorical data | Native categorical (this RFC) |
| Pre-encoded one-hot | Bundle them (RFC-0011) |
| High cardinality (>10) | Native categorical |
| Very low cardinality (<5) | Either works |

---

## Design Overview

```text
                    ┌─────────────────────────┐
                    │ Categorical Feature     │
                    │ color ∈ {R, G, B, Y}    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Histogram Building      │
                    │ One bin per category    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
     ┌────────────────┐                   ┌────────────────┐
     │ One-vs-Rest    │                   │ Partition-Based│
     │ (cardinality   │                   │ (high card)    │
     │  <= 32)        │                   │                │
     └───────┬────────┘                   └───────┬────────┘
             │                                     │
             ▼                                     ▼
     Split: color == R             Split: color ∈ {R, G}
```

### Configuration

```rust
pub struct CategoricalConfig {
    pub max_cat_one_hot: u32,      // One-vs-rest threshold (default: 32)
    pub max_cat_threshold: u32,    // Max categories to search (default: 64)
    pub cat_smooth: f32,           // Smoothing factor (default: 10.0)
    pub min_data_per_group: u32,   // Min samples per category (default: 10)
}
```

### Split Types

| Type | Condition | When |
| ---- | --------- | ---- |
| Numerical | `feature <= threshold` | Continuous features |
| CategoricalOneHot | `feature == category` | Low cardinality |
| CategoricalPartition | `feature ∈ {subset}` | High cardinality |

### Algorithms

**One-vs-Rest** (cardinality ≤ 32):
Test each category against all others. O(n_categories).

**Partition-Based** (high cardinality):

```text
1. Compute score = gradient / (hessian + smooth) per category
2. Sort categories by score (groups similar categories)
3. Search for best split point in sorted order
4. Store left categories as bitset
```

The key insight: categories with similar `g/h` ratios would benefit from the same
leaf value, so sorting by this metric naturally groups them.

### API

```rust
// Mark columns as categorical
let dataset = BinnedDatasetBuilder::from_matrix(&matrix, 256)
    .with_categorical_columns(vec![0, 3, 7])
    .build()?;

// Categorical splits handled automatically in training
let model = GBDTTrainer::new(config).train(&dataset)?;
```

---

## Design Decisions

### DD-1: Sorting Metric

Use `gradient / (hessian + smooth)` for partition search. Approximates optimal leaf value.
Matches LightGBM.

### DD-2: Split Storage

Use bitset (1 bit per category) for O(1) inference lookup.

### DD-3: Unknown Categories

Map to bin 0 (grouped with rare categories). Default behavior matches training.

### DD-4: Category Encoding

Remap raw categories to contiguous 0..n indices internally.

---

## Integration

| Component | Change |
| --------- | ------ |
| Tree | Store `SplitType` + `CategoriesStorage` (SoA layout) |
| Split Finding | `find_onehot_split()`, `find_sorted_split()` |
| Inference | Handle categorical split conditions |

---

## Quality Results (Adult)

| Library | Accuracy | AUC |
| ------- | -------- | --- |
| boosters | 86.57% | 0.926 |
| LightGBM | 86.6% | 0.927 |
| XGBoost | 86.5% | 0.926 |

---

## Implementation Notes

- Uses SoA design: `split_types: Box<[SplitType]>` + `CategoriesStorage` (not per-node enum)
- `CatBitset`: 64-bit inline + overflow for larger sets
- Global `lambda` used for regularization (not separate `cat_l2`)

---

## Changelog

- 2025-01-25: Simplified RFC — removed verbose implementation details
- 2025-12-19: Implementation complete
- 2025-12-18: Initial draft

## References

- [LightGBM Categorical Features](https://lightgbm.readthedocs.io/en/latest/Features.html#optimal-split-for-categorical-features)
