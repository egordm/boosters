# RFC-0001: Data Types and Layout

- **Status**: Implemented
- **Created**: 2024-11-15
- **Updated**: 2025-12-25
- **Scope**: Feature matrices, datasets, and data access patterns

## Summary

This RFC defines the data abstraction layer for boosters:

1. **Layout Convention**: All matrices are C-contiguous (row-major in memory)
2. **View Types**: `SamplesView` and `FeaturesView` provide semantic clarity over raw ndarray
3. **Accessor Traits**: `SampleAccessor` and `DataAccessor` enable generic tree traversal
4. **Dataset**: User-facing container with features, targets, and weights
5. **Binned Data**: Quantized representation for histogram-based training

## Why C-Contiguous Only

ndarray supports both C-order (row-major) and Fortran-order (column-major), but the layout semantics are implicit and hard to reason about. We enforce **C-contiguous only** for simplicity:

| Reason | Explanation |
| ------ | ----------- |
| **Predictable** | `as_slice()` always succeeds; no surprise failures |
| **Cache-friendly** | Row iteration is contiguous memory access |
| **SIMD-friendly** | Contiguous data enables vectorization |
| **NumPy default** | NumPy defaults to C-order; zero-copy FFI |
| **Simpler code** | No need to handle both layouts everywhere |

**Enforcement**: `debug_assert!(view.is_standard_layout())` in view constructors.

## Terminology

| Term | Meaning |
| ---- | ------- |
| `n_samples` | Number of training/inference samples |
| `n_features` | Number of input features |
| `n_groups` | Number of output groups (1=regression, K=multiclass) |
| Sample-major | Shape `[n_samples, n_features]` - each row is a sample |
| Feature-major | Shape `[n_features, n_samples]` - each row is a feature |

## View Types

Two semantic wrappers over `ArrayView2<f32>`:

```rust
/// Sample-major view: [n_samples, n_features]
/// Each sample's features are contiguous in memory.
/// Used for: inference (row-by-row tree traversal)
pub struct SamplesView<'a>(ArrayView2<'a, f32>);

/// Feature-major view: [n_features, n_samples]  
/// Each feature's values across samples are contiguous.
/// Used for: training (histogram building, coordinate descent)
pub struct FeaturesView<'a>(ArrayView2<'a, f32>);
```

**Why wrappers?** The axis semantics depend on context. `SamplesView` and `FeaturesView` make the layout explicit and prevent accidental misuse.

## Accessor Traits

For generic tree traversal over different data sources:

```rust
/// Access features for a single sample.
pub trait SampleAccessor {
    fn feature(&self, index: usize) -> f32;
    fn n_features(&self) -> usize;
}

/// Access samples from a multi-sample dataset.
pub trait DataAccessor {
    type Sample<'a>: SampleAccessor where Self: 'a;
    fn sample(&self, index: usize) -> Self::Sample<'_>;
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
}
```

**Implementors**: `SamplesView`, `BinnedDataset`, and their sample types.

## Dataset

The user-facing data container:

```rust
pub struct Dataset {
    /// Feature data [n_features, n_samples] - feature-major for training
    features: Array2<f32>,
    /// Feature metadata
    schema: DatasetSchema,
    /// Target values [n_groups, n_samples]
    targets: Option<Array2<f32>>,
    /// Sample weights [n_samples]
    weights: Option<Array1<f32>>,
}

pub struct DatasetSchema {
    features: Vec<FeatureMeta>,
}

pub struct FeatureMeta {
    pub name: Option<String>,
    pub feature_type: FeatureType,
}

pub enum FeatureType {
    Numeric,      // Continuous values, NaN = missing
    Categorical,  // Integer categories as floats
}
```

### Categorical Values

Following XGBoost/LightGBM, categorical features are stored as floats:

| Value | Interpretation |
| ----- | -------------- |
| `0.0, 1.0, 2.0, ...` | Valid category IDs (cast to i32) |
| `NaN` | Missing value |
| Negative values | Treated as missing |

### Construction

```rust
// Simple: from arrays
let dataset = Dataset::new(features.view(), Some(targets.view()), None);

// Builder for complex cases
let dataset = Dataset::builder()
    .add_feature("age", age_values.view())
    .add_categorical("category", cat_values.view())
    .targets(targets.view())
    .weights(weights.view())
    .build()?;
```

### View Access

```rust
impl Dataset {
    pub fn features(&self) -> FeaturesView<'_>;
    pub fn targets(&self) -> Option<TargetsView<'_>>;
    pub fn weights(&self) -> WeightsView<'_>;
    pub fn n_samples(&self) -> usize;
    pub fn n_features(&self) -> usize;
}
```

## Binned Data

Quantized features for histogram-based GBDT training:

```rust
pub struct BinnedDataset {
    /// Feature groups with quantized bins
    groups: Vec<FeatureGroup>,
    /// Per-feature metadata (bin mapper, group location)
    features: Box<[BinnedFeatureInfo]>,
    /// Global bin offsets for histogram allocation
    global_bin_offsets: Box<[u32]>,
}

pub struct BinnedFeatureInfo {
    pub bin_mapper: BinMapper,
    pub group_index: u32,
    pub index_in_group: u32,
}
```

**Storage types**: `DenseU8`, `DenseU16`, `SparseU8`, `SparseU16`
**Layouts**: Column-major (optimal for histogram building)

### Conversion

```rust
let binned = BinnedDatasetBuilder::new(BinningConfig::default())
    .add_features(dataset.features(), Parallelism::Parallel)
    .build()?;
```

## Prediction Layout

Predictions use shape `[n_groups, n_samples]`:

```rust
let predictions = model.predict(&dataset, parallelism);
// predictions: Array2<f32> with shape [n_groups, n_samples]
```

**Why group-major?**

- Base score initialization is contiguous per group
- Tree accumulation writes contiguously per group
- Matches gradient layout for training

## Missing Values

Represented as `f32::NAN`:

```rust
// Detection
fn is_missing(value: f32) -> bool {
    value.is_nan()
}

// Tree traversal handles NaN via default_left
if value.is_nan() {
    if node.default_left { go_left() } else { go_right() }
}
```

## Key Types Summary

| Type | Shape/Purpose |
| ---- | ------------- |
| `SamplesView` | `[n_samples, n_features]` - inference |
| `FeaturesView` | `[n_features, n_samples]` - training |
| `Dataset` | User-facing container with schema |
| `BinnedDataset` | Quantized bins for GBDT |
| `TargetsView` | `[n_groups, n_samples]` - target access |
| `WeightsView` | `[n_samples]` or uniform |

## Integration

| Component | Integration Point |
| --------- | ----------------- |
| RFC-0002 (Trees) | `TreeView::traverse_to_leaf<A: SampleAccessor>` |
| RFC-0003 (Inference) | `Predictor::predict(FeaturesView, ...)` |
| RFC-0004 (Binning) | `BinnedDataset` built from `Dataset` |
| RFC-0009 (GBLinear) | `FeaturesView` for coordinate descent |

## Changelog

- 2025-12-25: Restructured RFC to reflect the current data layer (views/accessors) after the 2025-12 refactors.
- 2025-01-23: Updated accessor traits to `SampleAccessor`/`DataAccessor`.
- 2025-01-21: Major rewrite for ndarray migration.
- 2024-11-15: Initial RFC.
