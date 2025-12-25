# RFC-0004: Quantization and Binning

- **Status**: Implemented
- **Created**: 2024-11-01
- **Updated**: 2025-01-21
- **Depends on**: RFC-0001
- **Scope**: Feature discretization for histogram-based GBDT

## Summary

Feature quantization (binning) discretizes continuous feature values into a fixed number of bins, enabling histogram-based gradient boosting with O(bins) instead of O(samples) complexity per split.

## Motivation

Histogram-based GBDT algorithms (LightGBM, XGBoost "hist") require binning for:
- **Speed**: Build histograms over bins (~256) instead of sorting samples (~millions)
- **Memory**: Store bins as u8/u16 instead of f64 (8x reduction for u8)
- **Cache efficiency**: Small histograms fit in L1/L2 cache

## Design

### Bin Computation

`BinMapper` stores bin boundaries and handles value-to-bin mapping:

```rust
pub struct BinMapper {
    bin_upper_bounds: Box<[f64]>,  // Upper bound for each bin
    n_bins: u32,                    // Total bins (including missing bin)
    missing_type: MissingType,      // None | Zero | NaN
    default_bin: u32,               // Bin for missing/default values
    most_freq_bin: u32,             // For histogram subtraction optimization
    feature_type: FeatureType,      // Numerical | Categorical
    // ...
}
```

For numerical features, values are mapped via binary search:
- Value v → first bin where `v <= bin_upper_bounds[bin]`
- Last bound is typically `f64::MAX` to catch all values

For categorical features, a `HashMap<i32, u32>` maps category → bin.

### Quantized Storage

`BinnedDataset` organizes quantized features into groups with flexible layouts:

```
BinnedDataset
├── FeatureGroup 0 (dense, column-major, u8)
│   ├── Feature 0
│   └── Feature 1
├── FeatureGroup 1 (dense, column-major, u16)
│   └── Feature 2 (wide: >256 bins)
└── FeatureGroup 2 (sparse, column-major, u8)
    └── Feature 3
```

**Storage types** (`BinStorage`):
- `DenseU8` / `DenseU16`: Contiguous bin arrays
- `SparseU8` / `SparseU16`: CSR-like (row_indices, bin_values) for >90% sparse features

**Layouts** (`GroupLayout`):
- `ColumnMajor`: `[f0_row0, f0_row1, ..., f0_rowN, f1_row0, ...]` — contiguous per-feature access for histogram building (default)
- `RowMajor`: `[row0_f0, row0_f1, ..., row0_fK, row1_f0, ...]` — sequential row access

Column-major is preferred for training (13% speedup in benchmarks) as each feature's bins are contiguous, enabling efficient histogram accumulation.

### Missing Values

`MissingType` enum specifies handling:
- `None`: No missing values
- `NaN`: NaN values get a dedicated bin (typically last bin: `n_bins - 1`)
- `Zero`: Zeros treated as missing (for sparse data)

The `default_bin` field specifies which bin receives missing values during binning.

## Key Types

| Type | Purpose |
|------|---------|
| `BinMapper` | Computes and stores bin boundaries; maps values ↔ bins |
| `BinnedDataset` | Main quantized dataset with feature groups |
| `FeatureGroup` | Storage for related features (shared layout/bin type) |
| `BinnedFeatureInfo` | Per-feature metadata (mapper, group_index, index_in_group) |
| `BinStorage` | Enum of storage formats (DenseU8/U16, SparseU8/U16) |
| `FeatureView` | Zero-cost slice view for accessing feature bins |
| `BinnedDatasetBuilder` | Fluent API for dataset construction |

## Implementation Notes

- **Max bins**: 256 for u8 (typical), 65536 for u16 (wide features)
- **Auto-grouping**: `GroupStrategy::Auto` separates features by bin count and sparsity
- **Global bin offsets**: Pre-computed for flat histogram indexing across all features
- **Bin type selection**: `BinType::for_max_bins(n)` selects smallest type that fits

## Changelog

- 2025-01-23: Renamed `FeatureMeta` to `BinnedFeatureInfo` to clarify purpose and avoid confusion with `schema::FeatureMeta`.
- 2025-01-21: Updated terminology to match refactored implementation (`n_bins`, `n_features`, `n_samples`)
