# RFC-0019: Exclusive Feature Bundling (EFB)

- **Status**: Draft
- **Created**: 2024-12-01
- **Updated**: 2024-12-01
- **Depends on**: RFC-0011 (Quantization), RFC-0013 (Histogram Building)
- **Scope**: Efficient handling of sparse/one-hot encoded features

## Summary

Exclusive Feature Bundling (EFB) reduces effective feature count by bundling mutually
exclusive sparse features. If features A and B are never both non-zero in the same row,
they can share a histogram using offset encoding.

## Motivation

Many datasets have sparse features:

- One-hot encoded categoricals (100 categories → 100 sparse features)
- Text TF-IDF features (95-99% zeros)
- Indicator variables

Building histograms for each sparse feature is wasteful. EFB can achieve 5-50x compression.

### Example

```text
Feature A: [0, 0, 2, 0, 1, 0]  → bins [0, 0, 2, 0, 1, 0]
Feature B: [1, 3, 0, 0, 0, 2]  → bins [1, 3, 0, 0, 0, 2]

Bundled (offset = num_bins_A = 3):
Bundle:    [4, 6, 2, 0, 1, 5]  → A_bin + (B > 0 ? B + offset : 0)
```

One histogram instead of two, same information preserved.

## Design

### Bundle Detection

Build conflict graph: features that have non-zero values together are "conflicting".
Use `max_conflict_rate` threshold (default 0) to allow small conflicts.

### Bundle Assignment

Greedy graph coloring:

1. Sort features by conflict count (descending)
2. Assign each feature to first non-conflicting bundle
3. Create new bundle if no existing bundle works

### Offset Encoding

Within a bundle, each feature gets an offset equal to the cumulative bin count of
preceding features:

```text
Bundle {F0, F1, F2}
F0: bins 0-9,    offset = 0
F1: bins 10-19,  offset = 10
F2: bins 20-29,  offset = 20
```

### Split Unbundling

When a split is found on a bundle, convert back to original feature:

```text
Split: Bundle_0 < 15
→ Threshold 15 is in F1's range (10-19)
→ Original: Feature F1 < (15 - 10) = 5
```

## Design Decisions

### DD-1: Avoid New Matrix Type

**Context**: Original draft proposed `BundledQuantizedMatrix` as separate type.

**Decision**: Integrate bundling into existing `QuantizedMatrix` via feature mapping.

**Rationale**: Avoid proliferating matrix types. Bundle info can be metadata:

- `bundle_map: Vec<(bundle_idx, offset)>` per original feature
- `bundles: Vec<Vec<usize>>` — features per bundle
- Histogram building uses bundle indices internally
- Final splits unbundled before storage

This keeps the interface clean: training sees bundled features, inference sees original features.

### DD-2: First Non-Zero Wins

**Decision**: When features in a bundle conflict (both non-zero), keep the first one.

**Rationale**: Simple, deterministic. With low conflict rate, this rarely happens.

### DD-3: Build Bundles Once

**Decision**: Compute bundles once before training, not per iteration.

**Rationale**: Bundle structure depends on data sparsity, not gradients.
Amortize O(features²) conflict detection across all iterations.

## Integration

| Component | Change |
|-----------|--------|
| `BinCuts` | Add optional `BundleInfo` with feature → bundle mapping |
| `QuantizedMatrix` | Store bundled bins when EFB enabled |
| `HistogramBuilder` | Build histograms over bundles (larger bin counts) |
| `SplitInfo` | Store bundle index during search, unbundle before tree storage |
| Tree storage | Unchanged — sees original feature indices |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_bundle` | false | Enable EFB |
| `max_conflict_rate` | 0.0 | Allow conflicts up to this fraction |

## Design Decisions (Continued)

### DD-4: Auto-disable for Dense Data

**Decision**: Check sparsity before bundling; skip if data is mostly dense.

**Rationale**: EFB overhead (conflict graph construction) isn't worth it for dense data.
Heuristic: skip if < 50% of values are zero across features.

### DD-5: No Incremental Bundling

**Decision**: Bundles computed once before training, no updates.

**Rationale**: Streaming/online learning would require expensive rebundling.
Not worth the complexity for batch training use case.

### DD-6: Missing in Bundle is Global Missing

**Decision**: Missing (bin 0) in any bundled feature maps to bundle bin 0.

**Rationale**: Simple and consistent. If all features in a bundle are missing/zero,
the bundle value is zero. Default direction handles missing at split time.

## Testing Strategy

### Unit Tests

- Conflict graph correctly identifies mutually exclusive features
- Greedy coloring produces valid bundles (no conflicts)
- Offset encoding preserves bin information
- Split unbundling recovers original feature and threshold

### Integration Tests

- EFB-trained model produces valid predictions
- Tree structure uses original feature indices (not bundle indices)
- Training converges on sparse datasets (one-hot encoded)

### Performance Tests

- Measure histogram memory: bundled vs unbundled
- Expected: ~K× reduction for K bundled features
- Measure training time on sparse datasets
- Document when EFB overhead exceeds benefit

### Validation Tests

- Compare predictions: EFB enabled vs disabled
- Tolerance: predictions should match closely (same model capacity)
- Compare against LightGBM with `enable_bundle=true`
- If deviations occur, investigate bundling algorithm differences

### Qualitative Tests

- Train on CTR dataset (highly sparse)
- Verify model quality comparable to non-EFB baseline
- Set accuracy expectations before training; investigate if not met

## References

- [LightGBM EFB paper section](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- Graph coloring heuristics
