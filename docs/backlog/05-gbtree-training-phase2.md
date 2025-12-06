# Epic 5: GBTree Training Phase 2

**Status**: Planned  
**Priority**: High  
**Depends on**: Epic 4 (GBTree Phase 1 — Complete)

## Overview

Phase 2 extends the GBTree training implementation with advanced features:

- Categorical feature handling
- Sampling strategies (GOSS, row/column subsampling)
- Multi-output tree support
- Exclusive feature bundling
- Monotonic and interaction constraints

---

## Testing Philosophy

Testing is a core part of Phase 2 development. Each story includes:

1. **Unit Tests**: Test individual functions and components in isolation
2. **Integration Tests**: Test feature end-to-end within training pipeline
3. **Validation Tests**: Compare output against XGBoost/LightGBM baselines
4. **Performance Tests**: Measure speed/memory against expectations
5. **Qualitative Tests**: Verify trained models are accurate and sensible

### Validation Test Approach

- Set tolerance thresholds **before** implementation (typically 1e-2 for predictions)
- If results deviate beyond tolerance, investigate:
  1. Check parameter mapping matches other library
  2. Review algorithm implementation against source code
  3. Document intentional differences if found
- Use `tests/test-cases/` for storing reference models and expected outputs

### Performance Test Approach

- Set expected speedup/memory targets before implementation
- If performance deviates significantly, investigate:
  1. Profile to identify bottlenecks
  2. Compare algorithm complexity with reference implementation
  3. Document findings and any tradeoffs made

---

## RFCs

| RFC | Title | Status |
|-----|-------|--------|
| RFC-0016 | Categorical Feature Training | Draft |
| RFC-0017 | Sampling Strategies | Draft |
| RFC-0018 | Multi-output Trees | Draft |
| RFC-0019 | Exclusive Feature Bundling | Draft |
| RFC-0023 | Training Constraints | Draft |

---

## Story 1: Categorical Feature Training

**Goal**: Train trees that can split on categorical features

**RFCs**: RFC-0016

### Tasks

- [ ] 1.1: Add `ColumnType::Categorical { cardinality }` to data structures
- [ ] 1.2: Implement gradient summation by category in histogram builder
- [ ] 1.3: Implement gradient-sorted partition finding (O(k log k))
- [ ] 1.4: Generate bitset for categories going left
- [ ] 1.5: Integrate with existing SplitInfo categorical fields
- [ ] 1.6: Tests with synthetic categorical data
- [ ] 1.7: Validation against XGBoost categorical splits

### Acceptance Criteria

- Categorical splits produce correct partitions
- Output bitsets compatible with existing inference code
- Performance reasonable for high-cardinality features

---

## Story 2: GOSS Sampling

**Goal**: Gradient-based One-Side Sampling for faster training

**RFCs**: RFC-0017

### Tasks

- [ ] 2.1: Add GOSS parameters to GBTreeParams (top_rate, other_rate)
- [ ] 2.2: Implement gradient magnitude computation
- [ ] 2.3: Implement top-gradient selection (keep top_rate)
- [ ] 2.4: Implement random sampling of remaining (other_rate)
- [ ] 2.5: Apply weight amplification to small gradients
- [ ] 2.6: Create SampledDataView that wraps full data with indices
- [ ] 2.7: Ensure downstream code uses sample weights correctly
- [ ] 2.8: Benchmark training speed vs accuracy tradeoff

### Acceptance Criteria

- Training speed improved with large datasets
- Accuracy within acceptable tolerance of full data training
- Weight amplification correctly applied

---

## Story 3: Row/Column Subsampling

**Goal**: Bootstrap and feature subsampling for regularization

**RFCs**: RFC-0017

### Tasks

- [ ] 3.1: Add subsample (row) parameter to GBTreeParams
- [ ] 3.2: Add colsample_bytree parameter to GBTreeParams
- [ ] 3.3: Implement row sampling per tree (with seed control)
- [ ] 3.4: Implement column sampling per tree
- [ ] 3.5: Ensure sampling is reproducible with fixed seed
- [ ] 3.6: Tests verifying reduced overfitting on synthetic data

### Acceptance Criteria

- Row/column subsampling improves generalization
- Results reproducible with same seed
- Sampling overhead minimal

---

## Story 4: Multi-output Support (One Output Per Tree)

**Goal**: Train separate forest per output (simple multi-output)

**RFCs**: RFC-0018

### Tasks

- [ ] 4.1: Add MultiStrategy::OneOutputPerTree to GBTreeParams
- [ ] 4.2: Implement output-sliced gradient view
- [ ] 4.3: Loop over outputs, train K independent forests
- [ ] 4.4: Combine predictions from K forests
- [ ] 4.5: Tests with multi-class classification
- [ ] 4.6: Validation against XGBoost multi:softmax

### Acceptance Criteria

- Multi-class classification working
- Predictions match XGBoost with same strategy
- Memory usage scales linearly with outputs

---

## Story 5: Multi-output Support (Multi-output Tree)

**Goal**: Trees with vector-valued leaves

**RFCs**: RFC-0018

### Tasks

- [ ] 5.1: Add MultiStrategy::MultiOutputTree option
- [ ] 5.2: Modify GradientHistogram to store n_outputs per bin
- [ ] 5.3: Implement aggregated gain computation across outputs
- [ ] 5.4: Implement vector leaf weight computation
- [ ] 5.5: Extend tree storage for vector leaves
- [ ] 5.6: Tests with multi-output regression
- [ ] 5.7: Benchmark vs OneOutputPerTree memory and speed

### Acceptance Criteria

- Vector leaves correctly computed
- Memory more efficient than K separate forests
- Training speed acceptable

---

## Story 6: Exclusive Feature Bundling

**Goal**: Bundle mutually exclusive sparse features

**RFCs**: RFC-0019

### Tasks

- [ ] 6.1: Implement conflict graph construction
- [ ] 6.2: Implement greedy graph coloring for bundling
- [ ] 6.3: Implement bundle offset encoding
- [ ] 6.4: Create bundle metadata (feature ranges per bundle)
- [ ] 6.5: Modify histogram builder to decode bundles
- [ ] 6.6: Implement split decoding (bundle bin → original feature)
- [ ] 6.7: Benchmark on sparse datasets (CTR, one-hot encoded)

### Acceptance Criteria

- Histogram memory reduced proportionally
- Training speed improved on sparse data
- Predictions match non-bundled training

---

## Story 7: Monotonic Constraints

**Goal**: Enforce monotonic relationships

**RFCs**: RFC-0023

### Tasks

- [ ] 7.1: Add monotone_constraints parameter to GBTreeParams
- [ ] 7.2: Add bounds tracking to BuildingNode
- [ ] 7.3: Implement monotonicity check after split finding
- [ ] 7.4: Implement bounds propagation to children
- [ ] 7.5: Implement leaf value clamping to bounds
- [ ] 7.6: Tests verifying monotonicity is enforced
- [ ] 7.7: Validation against XGBoost monotonic output

### Acceptance Criteria

- Increasing/decreasing constraints enforced
- Leaf predictions respect bounds
- Reasonable accuracy impact

---

## Story 8: Interaction Constraints

**Goal**: Limit feature interactions

**RFCs**: RFC-0023

### Tasks

- [ ] 8.1: Add interaction_constraints parameter to GBTreeParams
- [ ] 8.2: Add path feature tracking to BuildingNode
- [ ] 8.3: Implement allowed-feature computation per node
- [ ] 8.4: Filter candidate features in split finder
- [ ] 8.5: Tests verifying interaction limits enforced
- [ ] 8.6: Validation against XGBoost interaction constraints

### Acceptance Criteria

- Features only interact within allowed groups
- Tree structure respects constraints
- Performance overhead minimal

---

## Dependencies

```
Story 1 (Categorical) ─────────────────────────────┐
Story 2 (GOSS) ────────────────────────────────────┤
Story 3 (Row/Col Sampling) ────────────────────────┤
                                                   ├──► Phase 2 Complete
Story 4 (Multi-output Per Tree) ───────────────────┤
Story 5 (Multi-output Tree) ───────────────────────┤
Story 6 (Feature Bundling) ────────────────────────┤
Story 7 (Monotonic) ───────────────────────────────┤
Story 8 (Interaction) ─────────────────────────────┘
```

Stories 1-8 are independent and can be worked in any order.

## Estimated Effort

| Story | Effort | Notes |
|-------|--------|-------|
| 1. Categorical | Medium | Integration with existing inference |
| 2. GOSS | Medium | Weight handling is tricky |
| 3. Row/Col Sampling | Small | Straightforward sampling |
| 4. Multi-output Per Tree | Medium | Loop over outputs |
| 5. Multi-output Tree | Large | Vector histogram/leaves |
| 6. Feature Bundling | Large | Graph algorithms + decoding |
| 7. Monotonic | Medium | Bounds tracking/propagation |
| 8. Interaction | Medium | Path tracking + filtering |

Total: ~4-6 weeks
