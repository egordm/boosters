# RFC-0011: Feature Bundling and Sparse Optimization

- **Status**: Implemented
- **Created**: 2025-12-18
- **Updated**: 2025-01-25
- **Depends on**: RFC-0004 (Binning and Histograms)
- **Scope**: Dataset construction, histogram building

## Summary

Exclusive Feature Bundling (EFB) reduces memory and computation for sparse datasets,
particularly those with one-hot encoded categoricals. Mutually exclusive features
(features that are rarely non-zero together) are packed into single "bundle" columns.

## Motivation

**Example: Adult Dataset**

| Metric | Before | After | Improvement |
| ------ | ------ | ----- | ----------- |
| Features | 105 (one-hot) | 14 bundles | 7.5× fewer |
| Binned storage | 5.04 MB | 0.67 MB | 7.5× less |
| Histograms/leaf | 105 × 256 | 14 × 256 | 7.5× fewer |

EFB is especially effective when:

- Many one-hot encoded categorical features
- High sparsity (> 90% zeros)
- Pre-encoded features from external pipelines

## Design Overview

```text
Raw Features ──► Feature Analysis ──► Bundle Planning ──► Bundle Encoding
                 (detect binary,       (conflict graph,    (offset scheme)
                  compute sparsity)    greedy assign)
                                                                  │
                                                                  ▼
                                                           BinnedDataset
                                                           (m << n columns)
```

### Configuration

```rust
pub struct BundlingConfig {
    pub enable_bundling: bool,         // Default: true
    pub max_conflict_rate: f32,        // Default: 0.0001 (0.01%)
    pub min_sparsity: f32,             // Default: 0.9
    pub max_bundle_size: usize,        // Default: 256 (fits u8)
    pub bundle_hints: Option<Vec<Vec<usize>>>,  // Skip conflict detection
}

impl BundlingConfig {
    fn auto() -> Self { Self::default() }           // Good default
    fn disabled() -> Self { /* bundling off */ }    // For debugging
    fn aggressive() -> Self { /* 0.1% conflicts, 0.8 sparsity */ }
}
```

### Key Types

| Type | Purpose |
| ---- | ------- |
| `FeatureInfo` | Metadata: sparsity, is_binary, is_trivial |
| `FeatureBundle` | Group of features with bin offsets |
| `BundlePlan` | Full mapping: bundles + standalone features |
| `FeatureLocation` | Where a feature ended up (Bundled/Standalone/Skipped) |
| `BundlingStats` | Summary with `is_effective()`, `reduction_ratio()` |

### Algorithms

**Phase 1: Feature Analysis** — Single-pass O(n×m), parallelizable

For each feature, track: min, max, non_zero_count, first two distinct values.
Detect binary (exactly 2 values) and trivial (constant) features.

**Phase 2: Conflict Graph** — O(sample × n_sparse²)

Build bitsets of non-zero row indices per sparse feature.
Count pairwise conflicts (intersection of bitsets).
Skip if > 1000 sparse features. Sample 10K rows for large datasets.

**Phase 3: Greedy Bundling** — O(n_sparse² × n_bundles)

```text
Sort features by density (denser first)
For each feature:
    Find bundle with lowest conflict increase
    If total conflicts ≤ threshold and bundle not full:
        Add to bundle
    Else:
        Create new bundle
```

**Phase 4: Bundle Encoding**

Each bundle uses offset scheme: `bundle_bin = offset[i] + feature_bin[i]`
Bin 0 = "all features zero", bins 1..N encode active feature + its bin.

### API

```rust
// Default: bundling enabled
let dataset = BinnedDatasetBuilder::new()
    .with_bundling(BundlingConfig::auto())
    .build(&matrix, &labels)?;

// Check effectiveness
let stats = dataset.bundling_stats().unwrap();
if stats.is_effective() {
    println!("Bundled {} → {} columns", stats.original_features, stats.after_bundling);
}
```

---

## Design Decisions

### DD-1: Greedy vs Optimal Bundling

Optimal bundling is NP-hard (graph coloring). Greedy is O(n²) and usually near-optimal.
For one-hot features from same categorical, greedy is optimal.

### DD-2: Conflict Tolerance

Default 0.01% conflicts tolerated. Creates negligible gradient noise (~0.01% label noise equivalent).
User can set 0.0 for strict exclusivity.

### DD-3: Binary Feature Detection

Skip expensive quantile binning for binary features. Just use 2 bins.

### DD-4: Column Sampling Interaction

`colsample_bytree` samples original features. Bundle is active if any member selected.

### DD-5: Feature Importance on Original Features

Track splits per original feature, not per bundle. Decode bundle splits during importance.

---

## Integration

| Component | How |
| --------- | --- |
| Histogram Building | Build on bundled columns (fewer histograms) |
| Split Finding | Decode bundle bin → original feature + bin |
| Model Export | Store splits on original features (bundling transparent) |
| Native Categoricals | Bundling for one-hot; RFC-0012 for raw categoricals |

---

## Measured Performance

| Dataset | Features | Bundled Cols | Memory Reduction |
| ------- | -------- | ------------ | ---------------- |
| small_sparse (10K×32) | 32 | 5 | 84.4% |
| medium_sparse (50K×105) | 105 | 10 | 90.5% |
| high_sparse (20K×502) | 502 | 12 | 97.6% |

**Note**: Training integration is in RFC-0017. When integrated, EFB provides:

- ~3-4× fewer histogram passes per node (fewer columns to process)
- ~2-3× faster split finding (fewer bins to scan)
- Overall ~3× training speedup on sparse datasets like covertype

Bundling overhead: 4-19% one-time binning cost for analysis.

---

## Limitations

1. **Implicit regularization** — Fewer effective features may reduce overfitting capacity
2. **Conflict rate estimation** — Row sampling has ~1% error for rare conflicts
3. **Dense datasets** — Automatically skipped if no sparse features (> 10% non-zero)

---

## Changelog

- 2025-01-25: Simplified RFC — removed verbose implementation details
- 2025-12-19: Implementation complete with benchmarks
- 2025-12-18: Initial draft

## References

- [LightGBM Paper, Section 3.1](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
