# RFC-0018: BinnedDataset Redesign with Raw Feature Storage

**Status**: Draft  
**Created**: 2025-01-28  
**Revised**: 2025-01-28  
**Authors**: Team

## Summary

A comprehensive redesign of `BinnedDataset` storage that:

1. **Removes `GroupLayout`** - everything is column-major, eliminating strided access code
2. **Removes separate `Dataset` type** - `BinnedDataset` becomes the single source of truth
3. **Introduces typed storage** - `NumericStorage` vs `CategoricalStorage` encode semantics
4. **Enforces homogeneous groups** - each group is all-numeric OR all-categorical
5. **Makes EFB first-class** - dedicated `Bundle` storage for exclusive feature bundles
6. **Removes `BinType` enum** - `BinData` already encodes U8 vs U16
7. **Bundles don't store raw values** - EFB encoding is lossless for sparse categorical features
8. **Optimizes for histogram hot path** - zero-cost access with single match dispatch

**Priority**: IMMEDIATE - blocks linear tree quality improvement + opportunity for architectural cleanup.

## Usage Example

### Batch Ingestion (Recommended)

```rust
use boosters::data::{DatasetBuilder, BinnedDataset, BinningConfig, FeatureMetadata};
use ndarray::Array2;

// Full matrix: 1000 samples, 50 features
let data: Array2<f32> = load_my_data();
let labels: Array1<f32> = load_my_labels();

// Auto-detect feature types and optimal bin widths
// Result: e.g., 30 numeric (U8), 5 numeric (U16), 15 categorical (U8)
let dataset = DatasetBuilder::from_array(data.view(), &BinningConfig::default())?
    .set_labels(labels.view())
    .build()?;

// With user-provided metadata
let metadata = FeatureMetadata::default()
    .names(feature_names.clone())                  // Feature names
    .categorical(vec![10, 15, 20, 25, 30])         // Columns 10,15,20,25,30 are categorical
    .max_bins_for(0, 64)                           // Feature 0: fewer bins
    .max_bins_for(5, 1024);                        // Feature 5: more bins (U16)

let dataset = DatasetBuilder::from_array_with_metadata(data.view(), &metadata, &BinningConfig::default())?
    .set_labels(labels.view())
    .build()?;

println!("{}", dataset.describe());
// Output: BinnedDataset { samples: 1000, features: 50, groups: 4, 
//         numeric_u8: 29, numeric_u16: 6, categorical: 15, bundled: 0, memory: 245KB }
```

### Example: Single-Feature API

```rust
use boosters::data::{DatasetBuilder, BinnedDataset, FeatureStorageType};
use ndarray::array;

// Build dataset with explicit feature types
let dataset: BinnedDataset = DatasetBuilder::new()
    .add_numeric("age", array![25.0, 30.0, 35.0].view())
    .add_numeric("income", array![50000.0, 75000.0, 100000.0].view())
    .add_categorical("gender", array![0.0, 1.0, 0.0].view())
    .set_labels(array![0.0, 1.0, 0.0].view())
    .build()?;

// Introspection
assert!(dataset.has_raw_values());
assert_eq!(dataset.feature_storage_type(0), FeatureStorageType::Numeric);
assert_eq!(dataset.feature_storage_type(2), FeatureStorageType::Categorical);

// Raw access for linear trees
let age_raw = dataset.raw_feature_slice(0).unwrap();
assert_eq!(age_raw, &[25.0, 30.0, 35.0]);
```

### Prediction with Row Blocks

```rust
use boosters::utils::Parallelism;

// Load dataset (column-major internally)
let dataset = BinnedDataset::load("test_data.bin")?;
let model = Model::load("model.bin")?;

// Parallel prediction using row blocks (2x faster than column access)
let parallelism = Parallelism::from_threads(n_threads);
let predictions: Vec<f32> = dataset.row_blocks(256).flat_map_with(parallelism, |_, block| {
    block.rows()
        .into_iter()
        .map(|row| model.predict_one(row.as_slice().unwrap()))
        .collect()
});
```

### Serialization

```rust
// Save with raw values (default)
dataset.save("model.bin")?;
let loaded = BinnedDataset::load("model.bin")?;
assert!(loaded.has_raw_values()); // Raw values preserved

// Save without raw values (smaller file)
dataset.save_without_raw("model_small.bin")?;
```

## Motivation

### Problem Statement: Linear Trees Quality

Linear trees combine decision tree splits with linear regression at leaf nodes. This requires access to **raw feature values** for computing regression coefficients. Currently, `BinnedDataset` only stores quantized bin indices, and recovering original values via `bin_to_midpoint()` produces approximations.

**Example failure case**: For binary 0/1 features:

- Original values: `[0.0, 1.0]`  
- Bin boundaries: `[0.5, inf]` → bins `[0, 1]`
- `bin_to_midpoint()` returns: `[0.25, 0.75]`

This destroys the true relationship `y = β₀ + β₁x` where `x ∈ {0, 1}`.

### Problem Statement: Dual Dataset Types

Currently we have both `Dataset` (raw f32) and `BinnedDataset` (quantized). With raw values stored in `BinnedDataset`, the separate `Dataset` type becomes redundant:

- **For GBDT**: Uses binned data + raw values for linear trees
- **For gblinear**: Would use raw values directly (binning happens but is ignored)
- **For users**: Single type to understand, more control over feature specification

### Problem Statement: Architectural Complexity

The current architecture has accumulated complexity:

1. **Unused `GroupLayout::RowMajor`** - everything is column-major in practice
2. **Strided access in histogram kernels** - dead code for row-major layout
3. **EFB handled externally** - `BundlePlan` and `BundledColumns` live outside groups
4. **Mixed feature types in groups** - complicates raw value storage semantics
5. **Redundant `BinType` enum** - `BinData::U8/U16` already encodes this

### Impact

On covertype dataset with linear trees:

- **Current (midpoints)**: mlogloss ~0.80
- **Expected (raw values)**: mlogloss ~0.45
- **LightGBM reference**: mlogloss ~0.40

### Scope

1. **Primary**: Raw value storage for linear tree training
2. **Primary**: Remove `Dataset` type - `BinnedDataset` as single source of truth
3. **Secondary**: Remove `GroupLayout`, eliminate strided access
4. **Secondary**: First-class EFB support in groups
5. **Secondary**: Homogeneous groups (numeric-only or categorical-only)

## Background

### Current Hot Path Analysis

The histogram building kernel is the performance-critical path. Current code:

```rust
fn build_feature_gathered(
    histogram: &mut [HistogramBin],
    ordered_grad_hess: &[GradsTuple],
    indices: &[u32],
    view: &FeatureView<'_>,
) {
    match view {
        FeatureView::U8 { bins, stride: 1 } => {
            // Zero-cost: direct slice access
            for i in 0..indices.len() {
                let sample = indices[i] as usize;
                let bin = bins[sample] as usize;
                histogram[bin].0 += grad;
            }
        }
        FeatureView::U8 { bins, stride } => {
            // Strided access (row-major) - NEVER USED
            let bin = bins[sample * stride];
        }
        // ... U16 variants
    }
}
```

**Key observation**: The `stride > 1` paths are dead code. Removing `GroupLayout` eliminates them.

### LightGBM Reference

LightGBM stores feature type (`NumericalBin`/`CategoricalBin`) per feature in `BinMapper`, allowing mixed groups. They track numeric features separately via `numeric_feature_map_`.

Our design differs: we enforce homogeneous groups, making the type a group-level property encoded in the storage enum. This simplifies:

- Raw value storage (numeric groups always have it, categorical never)
- EFB logic (bundles are for sparse features, typically categorical)
- Auto-detection (binary detection already per-feature, grouping is straightforward)

### Current Automatic Detection

We already detect feature properties during analysis (`feature_analysis.rs`):

- **Binary features**: Exactly 2 unique values
- **Trivial features**: All zeros/missing (skipped)
- **Sparse features**: High fraction of zeros

This can be extended to auto-detect categorical vs numeric based on:

- User-specified `FeatureType::Categorical`
- Integer values with low cardinality (heuristic)
- Binary features (could go either way based on config)

## Proposed Design

### Core Principles

1. **Single source of truth**: `BinnedDataset` contains both binned and raw data
2. **Zero-cost access**: Single match dispatch, then direct array access
3. **Remove dead code**: No `GroupLayout`, no strided access, no `BinType` enum
4. **Typed storage**: Enum variants encode numeric vs categorical semantics
5. **Homogeneous groups**: Each group is either all-numeric or all-categorical
6. **Bundles are lossless**: No raw values needed - decode bin → original value

### New Storage Hierarchy

```rust
/// Bin data container. The variant encodes the bin width.
/// Replaces the separate `BinType` enum.
pub enum BinData {
    U8(Box<[u8]>),   // max 256 bins per feature
    U16(Box<[u16]>), // max 65536 bins per feature
}

impl BinData {
    /// Whether this is U8 or U16 (for pre-allocation decisions).
    pub fn is_u8(&self) -> bool { matches!(self, BinData::U8(_)) }
    pub fn is_u16(&self) -> bool { matches!(self, BinData::U16(_)) }
}

/// Feature storage with bins and optional raw values.
/// All storage is column-major (feature values are contiguous per feature).
///
/// Groups are homogeneous: all features in a group share the same type.
pub enum FeatureStorage {
    /// Numeric features (dense): bins + raw values.
    Numeric(NumericStorage),
    
    /// Categorical features (dense): bins only (lossless).
    Categorical(CategoricalStorage),
    
    /// Sparse numeric features: CSC-like storage + raw values.
    SparseNumeric(SparseNumericStorage),
    
    /// Sparse categorical features: CSC-like storage (lossless).
    SparseCategorical(SparseCategoricalStorage),
    
    /// EFB bundle: multiple sparse features encoded into one column.
    /// Bundles are for sparse categorical features (lossless).
    /// Linear trees skip bundled features for regression.
    Bundle(BundleStorage),
}
```

### Numeric Storage (Dense)

```rust
/// Dense numeric storage: [n_features × n_samples], column-major.
/// For feature f at sample s: bins[f * n_samples + s]
/// 
/// Raw values store actual f32 values including NaN for missing.
/// Missing handling semantics are defined by BinMapper.
pub struct NumericStorage {
    /// Bin values.
    bins: BinData,
    /// Raw values: [n_features × n_samples], column-major.
    /// Always present for numeric features.
    raw_values: Box<[f32]>,
}
```

**Access pattern** (O(1)):

```rust
impl NumericStorage {
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> u32 {
        let idx = feature_in_group * n_samples + sample;
        match &self.bins {
            BinData::U8(data) => data[idx] as u32,
            BinData::U16(data) => data[idx] as u32,
        }
    }
    
    #[inline]
    pub fn raw(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> f32 {
        self.raw_values[feature_in_group * n_samples + sample]
    }
    
    /// Get contiguous raw values for a feature (efficient for linear trees).
    #[inline]
    pub fn raw_slice(&self, feature_in_group: usize, n_samples: usize) -> &[f32] {
        let start = feature_in_group * n_samples;
        &self.raw_values[start..start + n_samples]
    }
}
```

### Categorical Storage (Dense)

```rust
/// Dense categorical storage: [n_features × n_samples], column-major.
/// No raw values - bin = category ID (lossless).
pub struct CategoricalStorage {
    /// Bin values (bin index = category ID).
    bins: BinData,
}
```

**Access pattern** (O(1)):

```rust
impl CategoricalStorage {
    #[inline]
    pub fn bin(&self, sample: usize, feature_in_group: usize, n_samples: usize) -> u32 {
        let idx = feature_in_group * n_samples + sample;
        match &self.bins {
            BinData::U8(data) => data[idx] as u32,
            BinData::U16(data) => data[idx] as u32,
        }
    }
    
    // No raw() method - categorical features are lossless
}
```

### Sparse Numeric Storage

```rust
/// Sparse numeric storage: CSC-like, single feature.
/// Non-zero samples only.
///
/// **Important**: Sparse storage assumes zeros are meaningful values, not missing.
/// Samples not in `sample_indices` have implicit bin=0, raw=0.0.
/// For features where missing should be NaN, use dense storage instead.
pub struct SparseNumericStorage {
    /// Sample indices of non-zero entries (sorted).
    sample_indices: Box<[u32]>,
    /// Bin values for non-zero entries (parallel to sample_indices).
    bins: BinData,
    /// Raw values for non-zero entries (parallel to bins).
    raw_values: Box<[f32]>,
    /// Total number of samples.
    n_samples: usize,
}
```

**Semantics**:

- Samples in `sample_indices` have explicit bin and raw values
- Samples NOT in `sample_indices` have implicit bin=0, raw=0.0

**Access pattern** (O(log nnz) for random access, O(1) for iteration):

```rust
impl SparseNumericStorage {
    /// Binary search for sample. Returns (bin, raw) or (0, 0.0).
    #[inline]
    pub fn get(&self, sample: usize) -> (u32, f32) {
        match self.sample_indices.binary_search(&(sample as u32)) {
            Ok(pos) => {
                let bin = match &self.bins {
                    BinData::U8(d) => d[pos] as u32,
                    BinData::U16(d) => d[pos] as u32,
                };
                (bin, self.raw_values[pos])
            }
            Err(_) => (0, 0.0),
        }
    }
}
```

### Sparse Categorical Storage

```rust
/// Sparse categorical storage: CSC-like, single feature.
/// Non-zero samples only. No raw values (lossless).
pub struct SparseCategoricalStorage {
    /// Sample indices of non-zero entries (sorted).
    sample_indices: Box<[u32]>,
    /// Bin values for non-zero entries (parallel to sample_indices).
    bins: BinData,
    /// Total number of samples.
    n_samples: usize,
}
```

### Bundle Storage (EFB)

```rust
/// EFB bundle: multiple sparse features encoded into one column.
/// Bundles are lossless - no raw values needed.
/// Linear trees skip bundled features for regression (use only for splits).
pub struct BundleStorage {
    /// Encoded bins: [n_samples], always U16.
    /// Bin 0 = all features have default value.
    /// Bin k = offset[i] + original_bin for active feature i.
    encoded_bins: Box<[u16]>,
    
    /// Original feature indices in this bundle.
    feature_indices: Box<[u32]>,
    
    /// Bin offset for each feature in the bundle.
    bin_offsets: Box<[u32]>,
    
    /// Number of bins per feature.
    feature_n_bins: Box<[u32]>,
    
    /// Total bins in this bundle.
    total_bins: u32,
    
    /// Default bin for each feature (bin when value is "zero").
    default_bins: Box<[u32]>,
    
    /// Number of samples.
    n_samples: usize,
}

impl BundleStorage {
    /// Decode encoded bin to (original_feature_position, original_bin).
    ///
    /// Implementation note: For bundles with ≤4 features, consider using
    /// linear scan instead of binary search for better cache performance.
    #[inline]
    pub fn decode(&self, encoded_bin: u16) -> Option<(usize, u32)> {
        if encoded_bin == 0 {
            return None; // All features at default
        }
        // Binary search in bin_offsets to find which feature
        // (or linear scan for small bundles - see implementation note)
        let encoded = encoded_bin as u32;
        match self.bin_offsets.binary_search(&encoded) {
            Ok(i) => Some((i, 0)), // Exactly at offset = bin 0 of feature i
            Err(i) => {
                if i == 0 { return None; }
                Some((i - 1, encoded - self.bin_offsets[i - 1]))
            }
        }
    }
    
    // No raw values - bundles are for categorical features (lossless)
}
```

**Why no raw values in bundles?**

EFB (Exclusive Feature Bundling) encodes multiple sparse features into one column. Features are bundled when they rarely have non-zero values simultaneously. This typically applies to:

- One-hot encoded categoricals (mutually exclusive by definition)
- Sparse indicator features

For these, `bin → value` is lossless via `BinMapper::bin_to_value()`. Linear regression in linear trees can skip bundled features and only use direct numeric features.

### Simplified FeatureGroup

```rust
pub struct FeatureGroup {
    /// Global feature indices in this group.
    feature_indices: Box<[u32]>,
    /// Number of samples.
    n_samples: usize,
    /// Storage (bins + optional raw values).
    /// The storage variant determines the feature type (numeric/categorical).
    storage: FeatureStorage,
    /// Per-feature bin counts.
    bin_counts: Box<[u32]>,
    /// Cumulative bin offsets within group histogram.
    bin_offsets: Box<[u32]>,
}

impl FeatureGroup {
    /// Whether this group has raw values (numeric features).
    pub fn has_raw_values(&self) -> bool {
        matches!(
            self.storage,
            FeatureStorage::Numeric(_) | FeatureStorage::SparseNumeric(_)
        )
    }
    
    /// Whether this group contains categorical features.
    pub fn is_categorical(&self) -> bool {
        !self.has_raw_values()
    }
}
```

### Simplified FeatureView

```rust
/// Zero-cost view into feature bins.
/// No stride - everything is column-major, contiguous.
pub enum FeatureView<'a> {
    /// Dense bins, contiguous per feature.
    U8(&'a [u8]),
    U16(&'a [u16]),
    /// Sparse bins.
    SparseU8 { sample_indices: &'a [u32], bin_values: &'a [u8] },
    SparseU16 { sample_indices: &'a [u32], bin_values: &'a [u16] },
}

// No stride field - always 1
// Renamed: row_indices → sample_indices
```

### Unified BinnedDataset (Replaces Dataset + BinnedDataset)

```rust
/// The unified dataset type for training and inference.
/// Contains both binned data (for tree splits) and raw data (for linear regression).
///
/// This replaces the previous separate `Dataset` and `BinnedDataset` types.
pub struct BinnedDataset {
    /// Number of samples.
    n_samples: usize,
    /// Per-feature metadata (name, bin mapper, location).
    features: Box<[BinnedFeatureInfo]>,
    /// Feature groups (actual storage).
    groups: Vec<FeatureGroup>,
    /// Global bin offsets for histogram allocation.
    global_bin_offsets: Box<[u32]>,
    /// Where each original feature's data lives.
    feature_locations: Box<[FeatureLocation]>,
}

/// Metadata for a single feature.
pub struct BinnedFeatureInfo {
    pub name: Option<String>,
    pub bin_mapper: BinMapper,
    /// Where this feature's data lives.
    pub location: FeatureLocation,
}

/// Where a feature's data lives.
pub enum FeatureLocation {
    /// Feature in a regular (Dense or Sparse) group.
    Direct { group_idx: u32, idx_in_group: u32 },
    /// Feature bundled into a Bundle group.
    Bundled { bundle_group_idx: u32, position_in_bundle: u32 },
    /// Feature was skipped (trivial, constant value).
    Skipped,
}
```

### Access API

```rust
impl BinnedDataset {
    // =========================================================================
    // Basic accessors
    // =========================================================================
    
    /// Get bin value for a sample/feature.
    pub fn bin(&self, sample: usize, feature: usize) -> u32;
    
    /// Get raw value for numeric features.
    /// Returns None for categorical features.
    pub fn raw_value(&self, sample: usize, feature: usize) -> Option<f32>;
    
    /// Get contiguous raw values for a feature (efficient for linear trees).
    /// Returns None for categorical features or bundled features.
    pub fn raw_feature_slice(&self, feature: usize) -> Option<&[f32]>;
    
    // =========================================================================
    // Feature introspection
    // =========================================================================
    
    /// Get the storage type for a feature.
    pub fn feature_storage_type(&self, feature: usize) -> FeatureStorageType;
    
    /// Get original feature indices in a bundle group.
    /// Panics if group is not a bundle.
    pub fn bundle_features(&self, bundle_group_idx: usize) -> &[u32];
    
    /// Get number of groups.
    pub fn n_groups(&self) -> usize;
    
    // =========================================================================
    // Histogram building (training hot path)
    // =========================================================================
    
    /// Get feature views for histogram building.
    /// For bundled datasets, returns bundled views.
    /// This is the primary API for training.
    pub fn feature_views(&self) -> Vec<FeatureView<'_>>;
    
    /// Get view for a single original feature (unbundled).
    /// Use only when you need access to a specific original feature.
    pub fn original_feature_view(&self, feature: usize) -> FeatureView<'_>;
    
    /// Decode a split from bundled feature index to original feature.
    pub fn decode_split(&self, bundled_feature: usize, bin: u32) -> (usize, u32);
    
    // =========================================================================
    // Linear trees support
    // =========================================================================
    
    /// Check if any feature has raw values (for linear trees).
    /// True if there's at least one numeric group.
    pub fn has_raw_values(&self) -> bool {
        self.groups.iter().any(|g| g.has_raw_values())
    }
    
    /// Get indices of numeric features (for linear tree feature selection).
    /// Linear trees use this to identify which features to include in regression.
    /// Features with `FeatureStorageType::Bundled` are excluded (splits only, no regression).
    pub fn numeric_feature_indices(&self) -> impl Iterator<Item = usize> + '_;
    
    // =========================================================================
    // Bulk raw access (for gblinear)
    // =========================================================================
    
    /// Get raw values for all numeric features as a matrix view.
    /// Returns None if no numeric features exist.
    /// Layout: [n_numeric_features, n_samples], feature-major (Fortran order).
    ///
    /// **Note**: If numeric features are scattered across multiple groups,
    /// this allocates and copies into a contiguous array. Use `raw_feature_iter()`
    /// for zero-allocation access when you don't need a contiguous matrix.
    pub fn raw_numeric_matrix(&self) -> Option<CowArray<'_, f32, Ix2>>;
    
    /// Iterator over (feature_index, raw_slice) for all numeric features.
    /// Zero-allocation access to raw values.
    pub fn raw_feature_iter(&self) -> impl Iterator<Item = (usize, &[f32])> + '_;
    
    // =========================================================================
    // Serialization
    // =========================================================================
    
    /// Save dataset to file. Raw values are included by default.
    /// Returns `DatasetError` on I/O failure.
    pub fn save(&self, path: &Path) -> Result<(), DatasetError>;
    
    /// Save dataset without raw values (smaller file, but linear trees
    /// won't work after reload).
    pub fn save_without_raw(&self, path: &Path) -> Result<(), DatasetError>;
    
    /// Load dataset from file.
    /// Files saved without raw values will have `has_raw_values() = false`.
    pub fn load(path: &Path) -> Result<Self, DatasetError>;
    
    // =========================================================================
    // Debugging
    // =========================================================================
    
    /// Print dataset summary: n_samples, n_features, n_numeric, n_categorical,
    /// n_bundled, memory usage.
    pub fn describe(&self) -> DatasetSummary;
    
    // =========================================================================
    // Memory
    // =========================================================================
    
    /// Memory used by raw storage.
    pub fn raw_storage_bytes(&self) -> usize;
}

/// How a feature is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureStorageType {
    /// Dense numeric (has raw values).
    Numeric,
    /// Sparse numeric (has raw values).
    SparseNumeric,
    /// Dense categorical (lossless, no raw values).
    Categorical,
    /// Sparse categorical (lossless, no raw values).
    SparseCategorical,
    /// Bundled with other features (lossless, no raw values).
    Bundled,
    /// Skipped (trivial/constant feature).
    Skipped,
}

// Note: Use ndarray's CowArray for raw_numeric_matrix() return type.
// CowArray<'a, f32, Ix2> is either a borrowed view or an owned array,
// matching the semantics we need (zero-copy when contiguous, copy when scattered).
```

**Note**: Removed `supports_linear_trees()` - use `has_raw_values()` instead. More general and descriptive.

### Zero-Cost Histogram Building

The new design eliminates strided access:

```rust
fn build_feature_gathered(
    histogram: &mut [HistogramBin],
    ordered_grad_hess: &[GradsTuple],
    indices: &[u32],
    view: &FeatureView<'_>,
) {
    match view {
        // Contiguous access - no stride multiplication
        FeatureView::U8(bins) => {
            for i in 0..indices.len() {
                let sample = unsafe { *indices.get_unchecked(i) } as usize;
                let bin = unsafe { *bins.get_unchecked(sample) } as usize;
                let slot = unsafe { histogram.get_unchecked_mut(bin) };
                let gh = unsafe { *ordered_grad_hess.get_unchecked(i) };
                slot.0 += gh.grad as f64;
                slot.1 += gh.hess as f64;
            }
        }
        FeatureView::U16(bins) => { /* same pattern */ }
        FeatureView::SparseU8 { .. } => { /* sparse kernel */ }
        FeatureView::SparseU16 { .. } => { /* sparse kernel */ }
    }
}
```

**Fewer match arms** (4 vs 6): No strided variants.

### Homogeneous Group Building

Groups are built by feature type:

```rust
/// Group features by type for homogeneous storage.
fn group_features(
    features: &[FeatureMeta],
    feature_infos: &[FeatureInfo],
    config: &BinningConfig,
) -> Vec<GroupSpec> {
    let mut numeric_dense = Vec::new();
    let mut numeric_sparse = Vec::new();
    let mut categorical_dense = Vec::new();
    let mut categorical_sparse = Vec::new();
    
    for (idx, (meta, info)) in features.iter().zip(feature_infos).enumerate() {
        if info.is_trivial {
            continue; // Skip trivial features
        }
        
        let is_sparse = info.is_sparse(config.sparsity_threshold);
        
        match meta.feature_type {
            FeatureType::Numeric => {
                if is_sparse {
                    numeric_sparse.push(idx);
                } else {
                    numeric_dense.push(idx);
                }
            }
            FeatureType::Categorical => {
                if is_sparse {
                    categorical_sparse.push(idx);
                } else {
                    categorical_dense.push(idx);
                }
            }
        }
    }
    
    // Build groups...
    // Dense numeric features → one NumericStorage group
    // Dense categorical features → one CategoricalStorage group
    // Each sparse feature → its own Sparse* group
    // Sparse categorical features may be bundled → BundleStorage
}
```

### Dataset Building: Batch vs Single-Feature APIs

The builder supports two usage patterns:

1. **Batch ingestion** (recommended): Provide a full matrix, let the builder auto-detect types and optimize grouping
2. **Single-feature**: Add features one-by-one when you need fine-grained control

#### Builder: Batch Ingestion (Recommended)

```rust
impl DatasetBuilder {
    /// Build from a 2D array with auto-detection.
    /// Detects numeric vs categorical features, optimal bin widths (U8/U16),
    /// and groups features homogeneously for optimal storage.
    ///
    /// - Analyzes all n features
    /// - Detects n₁ numeric, n₂ categorical features
    /// - Within numeric: detects n₁₁ that fit in U8, n₁₂ that need U16
    /// - Creates homogeneous groups: one for U8 numeric, one for U16 numeric,
    ///   one for U8 categorical, etc.
    pub fn from_array(
        data: ArrayView2<f32>,
        config: &BinningConfig,
    ) -> Result<Self, DatasetError>;
    
    /// Build from array with feature metadata.
    /// User provides optional metadata for each feature:
    /// - Feature names
    /// - Which columns are categorical vs numeric
    /// - Per-feature max_bins overrides
    pub fn from_array_with_metadata(
        data: ArrayView2<f32>,
        metadata: &FeatureMetadata,
        config: &BinningConfig,
    ) -> Result<Self, DatasetError>;
}

/// Metadata for features in a matrix.
/// All fields are optional - unspecified features use auto-detection.
#[derive(Default, Clone)]
pub struct FeatureMetadata {
    /// Feature names (length must match n_features or be empty).
    pub names: Vec<String>,
    /// Indices of categorical features. All others are numeric.
    /// If empty, auto-detect based on cardinality.
    pub categorical_features: Vec<usize>,
    /// Per-feature max_bins overrides. Key = feature index.
    pub max_bins: HashMap<usize, u32>,
}

impl FeatureMetadata {
    /// Create with just feature names.
    pub fn with_names(names: Vec<String>) -> Self {
        Self { names, ..Default::default() }
    }
    
    /// Create with categorical feature indices.
    pub fn with_categorical(categorical_features: Vec<usize>) -> Self {
        Self { categorical_features, ..Default::default() }
    }
    
    /// Builder pattern: set names.
    pub fn names(mut self, names: Vec<String>) -> Self {
        self.names = names;
        self
    }
    
    /// Builder pattern: set categorical features.
    pub fn categorical(mut self, indices: Vec<usize>) -> Self {
        self.categorical_features = indices;
        self
    }
    
    /// Builder pattern: set max_bins for a feature.
    pub fn max_bins_for(mut self, feature: usize, bins: u32) -> Self {
        self.max_bins.insert(feature, bins);
        self
    }
}
```

**Grouping strategy**:

```text
Input: n features with mixed types and bin requirements

Step 1: Analyze all features
  - Detect numeric vs categorical (auto or user-specified)
  - Compute required bins per feature
  - Detect sparsity

Step 2: Partition by type and bin width
  - Numeric dense, ≤256 bins → one NumericStorage group (U8)
  - Numeric dense, >256 bins → one NumericStorage group (U16)
  - Categorical dense, ≤256 bins → one CategoricalStorage group (U8)
  - Categorical dense, >256 bins → one CategoricalStorage group (U16)
  - Sparse features → individual Sparse* groups
  - Sparse categorical → candidates for EFB bundling

Step 3: Build homogeneous groups
  - Each group has uniform BinData type (all U8 or all U16)
  - Feature order within groups optimized for cache locality
```

#### Builder: Single-Feature API

For cases requiring fine-grained control:

```rust
impl DatasetBuilder {
    /// Add a numeric feature (default).
    pub fn add_feature(self, name: &str, values: ArrayView1<f32>) -> Self;
    
    /// Add a numeric feature explicitly (same as add_feature).
    pub fn add_numeric(self, name: &str, values: ArrayView1<f32>) -> Self;
    
    /// Add a categorical feature explicitly.
    pub fn add_categorical(self, name: &str, values: ArrayView1<f32>) -> Self;
    
    /// Add a feature with full specification.
    pub fn add_feature_with_spec(
        self,
        values: ArrayView1<f32>,
        spec: FeatureSpec,
    ) -> Self;
}
```

**Note**: Single-feature API defers grouping until `build()`. At build time, features are still grouped homogeneously by type and bin width.

#### Auto-Detection Heuristics

1. User-specified type always takes precedence
2. Integer values with cardinality ≤ `max_categorical_cardinality` → Categorical
3. Binary features (exactly 2 values) → Numeric (unless user specifies Categorical)
4. Everything else → Numeric

**Bin width detection**:

- Feature needs ≤256 bins → U8
- Feature needs >256 bins → U16

### Configuration

```rust
pub struct BinningConfig {
    /// Maximum bins per feature (global default).
    /// Can be overridden per-feature via `FeatureMetadata::max_bins`.
    pub max_bins: u32,
    /// Sparsity threshold (fraction of zeros to use sparse storage).
    pub sparsity_threshold: f32,
    /// Enable EFB bundling.
    pub enable_bundling: bool,
    /// Max categories to auto-detect as categorical.
    pub max_categorical_cardinality: u32,
}

impl Default for BinningConfig {
    fn default() -> Self {
        Self {
            max_bins: 256,
            sparsity_threshold: 0.9,
            enable_bundling: true,
            max_categorical_cardinality: 256,
        }
    }
}
```

**Per-feature max_bins**:

```rust
// Global: 256 bins
let config = BinningConfig::default();

// Override for specific features via metadata
let metadata = FeatureMetadata::default()
    .max_bins_for(0, 64)    // Feature 0: 64 bins
    .max_bins_for(2, 1024); // Feature 2: 1024 bins (U16)
    // Feature 1: uses global (256)

let dataset = DatasetBuilder::from_array_with_metadata(data.view(), &metadata, &config)?.build()?;
```

**Note**: Removed `store_raw_values: bool`. Raw values are always stored for numeric features - the overhead is acceptable and simplifies the API.

## Implementation Plan

### Phase 1: Storage Types

1. Create `BinData` enum (remove `BinType`)
2. Create `NumericStorage`, `CategoricalStorage`
3. Create `SparseNumericStorage`, `SparseCategoricalStorage`
4. Create `BundleStorage` (consolidate from `bundling.rs`)
5. Create `FeatureStorage` enum

### Phase 2: FeatureGroup Migration

1. Update `FeatureGroup` to use `FeatureStorage`
2. Remove `layout` field (everything column-major)
3. Add `has_raw_values()` / `is_categorical()` helpers
4. Update `FeatureView` (remove stride)
5. Add runtime assertions for homogeneous group invariant (panic on violation)

### Phase 3: Builder Unification

1. Create single `DatasetBuilder` that produces `BinnedDataset`
2. Add `add_feature()`, `add_numeric()`, `add_categorical()`, `add_feature_as()` methods
3. Implement homogeneous group building with validation
4. Error on attempt to bundle numeric features
5. Store raw values during feature addition

### Phase 4: Dataset Migration

1. Deprecate `Dataset` type with `#[deprecated]` attribute
2. Add `FeatureLocation` to `BinnedFeatureInfo`
3. Rename `effective_feature_views()` → `feature_views()`
4. Rename `feature_view()` → `original_feature_view()`
5. Implement `raw_value()`, `raw_feature_slice()`, `raw_numeric_matrix()`, `raw_feature_iter()`
6. Add `describe()` for debugging

### Phase 5: Serialization

1. Update serialization to include raw values by default
2. Add `save_without_raw()` for space-conscious users
3. Add version field to support future format changes

### Phase 6: Cleanup

1. Remove `GroupLayout` enum
2. Remove `BinType` enum
3. Remove strided histogram kernel variants
4. Remove `BundledColumns`, `BundlePlan` (consolidated into `BundleStorage`)
5. Remove deprecated `Dataset` type (next major version)
6. Update all callers

### Phase 7: Validation

1. Verify covertype mlogloss improvement with linear trees
2. Ensure no performance regression in histogram building
3. Benchmark memory usage
4. Test categorical feature handling
5. Test serialization roundtrip

## Code Removal Summary

| Removed                              | Lines | Reason                              |
| ------------------------------------ | ----- | ----------------------------------- |
| `Dataset` struct (or becomes alias)  | ~200  | Merged into `BinnedDataset`         |
| `GroupLayout` enum                   | ~50   | Unused (always ColumnMajor)         |
| `BinType` enum                       | ~20   | `BinData` encodes this              |
| Strided `FeatureView` variants       | ~40   | No row-major layout                 |
| Strided histogram kernels            | ~60   | Dead code                           |
| `BundledColumns` struct              | ~100  | Moved into `BundleStorage`          |
| `bundle_plan`, `bundled_columns`     | ~50   | Consolidated into groups            |
| `supports_linear_trees()` method     | ~10   | Use `has_numeric_features()` instead|

**Estimated**: -530 lines removed, +400 lines added = **-130 lines net**

## Alternatives Considered

### Alternative A: Keep Separate Dataset Type

Keep `Dataset` for raw data, `BinnedDataset` for binned.

**Rejected**: Duplication. With raw values in `BinnedDataset`, the separation is unnecessary. gblinear can use `BinnedDataset::raw_feature_slice()`.

### Alternative B: Mixed Feature Groups

Allow numeric + categorical features in the same group.

**Rejected**: Complicates storage semantics. With homogeneous groups, the storage variant encodes the type cleanly (`NumericStorage` always has raw, `CategoricalStorage` never does).

### Alternative C: Store Raw Values in Bundles

Store raw values for bundled features.

**Rejected**: EFB is lossless - `bin_to_value()` recovers exact values for categorical features. Linear trees can skip bundled features for regression (they're typically sparse categorical).

### Alternative D: Keep BinType Enum

Keep `BinType` separate from `BinData`.

**Rejected**: Redundant. `BinData::U8` vs `BinData::U16` already encodes this. Add a helper `BinData::is_u8()` if needed for config decisions.

## Testing Considerations

### Unit Tests

1. `NumericStorage`, `CategoricalStorage` construction and access
2. `SparseNumericStorage`, `SparseCategoricalStorage` construction and access  
3. `BundleStorage` encoding/decoding (lossless verification)
4. Homogeneous group building with runtime assertions
5. Feature type detection utility (opt-in)
6. Memory accounting
7. `raw_numeric_matrix()` for gblinear access pattern

### Edge Cases

1. Empty dataset (0 samples, 0 features)
2. Single sample dataset
3. All features trivial (all skipped)
4. All features bundled (no direct features)
5. All features categorical (no raw storage)
6. Mixed NaN patterns across features
7. Feature with all NaN values
8. Bundled numeric feature detection (should error/warn)

### Integration Tests

1. End-to-end training with mixed feature types
2. Linear trees on covertype
3. Categorical splits
4. EFB bundling
5. Serialization roundtrip (with and without raw values)
6. Loading old format files (backward compatibility)

### Performance Tests (Benchmarks)

1. `bench_histogram_building`: Compare storage access patterns (should be no regression)
2. `bench_training_covertype`: End-to-end training time
3. `bench_memory_usage`: Measure raw value overhead vs baseline
4. `bench_serialization`: Save/load times with and without raw values

**CI Integration**: Benchmarks run in CI with criterion. Performance regressions >5% fail the build.

### Test Datasets

- **covertype**: Primary test for linear trees quality improvement
- **higgs**: Large dataset for EFB bundling verification
- **synthetic**: Edge cases (all categorical, all bundled, empty, single sample)

### Property-Based Tests (proptest)

1. Roundtrip: build → serialize → deserialize → same data
2. Invariant: numeric groups always have `raw_values.len() == n_features * n_samples`
3. Invariant: categorical groups have no raw_values access
4. Invariant: `feature_views().len()` equals number of training features (including bundles)

### Quality Gates

| Metric                            | Baseline | Target        |
| --------------------------------- | -------- | ------------- |
| Covertype mlogloss (linear trees) | ~0.80    | ~0.45         |
| Memory overhead (% of binned)     | 0%       | <50%          |
| Histogram build time              | baseline | no regression |

### Documentation Requirements

1. API docs for all public types and methods
2. Migration guide: `Dataset` → `BinnedDataset`
3. Module-level examples for common use cases
4. Performance notes (f64 accumulators for linear trees)

## Open Questions

1. **Phased rollout**: Single PR or multiple?
   - Recommendation: Multiple PRs with dependency graph:
   
   ```text
   PR1: Storage types + FeatureGroup refactor (~400 LOC)
    ↓
   PR2: Builder unification + Dataset deprecation (~300 LOC)
    ↓
   PR3: Serialization + debugging APIs (~200 LOC)
    ↓
   PR4: Cleanup (GroupLayout, BinType, strided kernels) (~-300 LOC net)
   ```

2. **Binary feature handling**: Numeric or Categorical by default?
   - Recommendation: Numeric (benefits from linear regression), unless user specifies Categorical

3. **gblinear migration**: How to handle?
   - Use `BinnedDataset::raw_feature_iter()` for zero-allocation access
   - Accept minimal binning overhead (cheap compared to training)

4. **Version compatibility**: Loading old serialized files without raw values?
   - Recommendation: Load succeeds with `has_raw_values() = false`. Document that linear trees require re-binning from source data.

5. **Feature flag during migration?**
   - Recommendation: No feature flag. Changes are internal storage reorganization. API remains compatible via deprecation warnings.

## Design Notes

### Thread Safety

`BinnedDataset` is `Send + Sync`. All fields are immutable boxed slices or `Vec<T>` where `T` is `Send + Sync`. Add compile-time assertion:

```rust
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<BinnedDataset>();
    }
};
```

### Precision in Linear Trees

Raw values are stored as f32 for memory efficiency. Linear tree implementations SHOULD use f64 accumulators for numerical stability during regression coefficient computation. This is consistent with gradient/hessian handling elsewhere in the codebase.

### Prediction Use Case

`BinnedDataset` is used for both training and prediction (GBDT and gblinear). For prediction, binning is not strictly necessary - we only need raw values. However, using `BinnedDataset` for prediction is acceptable:

**Overhead analysis**:

- Binning cost: O(n_samples × n_features × log(n_bins)) - negligible compared to tree traversal
- Memory overhead: ~1-2 bytes per sample per feature for bins - acceptable
- Benefit: Unified API, single code path, can use same dataset for train/predict

**Recommendation**: Accept the small overhead. The simplicity of a single dataset type outweighs the minor performance cost.

### Row Block Iterator for Prediction

Prediction typically processes samples row-by-row, but `BinnedDataset` stores data column-major. We observed a **2x speedup** by buffering row blocks before prediction rather than random column access.

This iterator is a **reusable component** that should be extracted from prediction code and live in the dataset module.

```rust
impl BinnedDataset {
    /// Create a buffered row block iterator for prediction.
    /// Buffers `block_size` rows at a time into row-major layout.
    /// This provides ~2x speedup for prediction vs random column access.
    ///
    /// Parallelism controls whether blocks are processed in parallel.
    pub fn row_blocks(&self, block_size: usize) -> RowBlocks<'_>;
    
    /// Get a single row's raw values (allocates).
    /// Use `row_blocks()` for batch prediction.
    pub fn get_row(&self, sample: usize) -> Vec<f32>;
}

/// Buffered row block iterator with parallel support.
/// Transposes column-major storage to row-major blocks on demand.
pub struct RowBlocks<'a> {
    dataset: &'a BinnedDataset,
    block_size: usize,
}

impl<'a> RowBlocks<'a> {
    /// Number of blocks.
    pub fn n_blocks(&self) -> usize;
    
    /// Process blocks sequentially, calling `f` for each block.
    /// `f` receives (start_sample_idx, block: ArrayView2<f32>).
    /// Block is row-major: [block_size, n_features].
    pub fn for_each<F>(&self, f: F)
    where
        F: FnMut(usize, ArrayView2<f32>);
    
    /// Process blocks with parallelism control.
    /// When `Parallelism::Parallel`, blocks are processed in parallel using rayon.
    /// When `Parallelism::Sequential`, blocks are processed sequentially.
    ///
    /// `f` receives (start_sample_idx, block: ArrayView2<f32>).
    pub fn for_each_with<F>(&self, parallelism: Parallelism, f: F)
    where
        F: Fn(usize, ArrayView2<f32>) + Sync + Send;
    
    /// Collect results from each block with parallelism control.
    /// `f` receives (start_sample_idx, block: ArrayView2<f32>) -> Vec<T>.
    /// Results are concatenated in order.
    pub fn flat_map_with<T, F>(&self, parallelism: Parallelism, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize, ArrayView2<f32>) -> Vec<T> + Sync + Send;
    
    /// Get a specific block by index (useful for parallel iteration).
    /// Returns (start_sample_idx, block: Array2<f32>).
    pub fn block(&self, block_idx: usize) -> (usize, Array2<f32>);
    
    /// Iterate over block indices (for external parallel iteration).
    pub fn block_indices(&self) -> impl Iterator<Item = usize>;
}
```

**Usage for prediction**:

```rust
let dataset = BinnedDataset::load("data.bin")?;
let model = Model::load("model.bin")?;

// Sequential prediction
let mut predictions = Vec::with_capacity(dataset.n_samples());
dataset.row_blocks(256).for_each(|start_idx, block| {
    for row in block.rows() {
        predictions.push(model.predict_one(row.as_slice().unwrap()));
    }
});

// Parallel prediction (uses rayon when Parallelism::Parallel)
let parallelism = Parallelism::from_threads(n_threads);
let predictions: Vec<f32> = dataset.row_blocks(256).flat_map_with(parallelism, |start_idx, block| {
    block.rows()
        .into_iter()
        .map(|row| model.predict_one(row.as_slice().unwrap()))
        .collect()
});

// External parallel iteration (when you need more control)
use rayon::prelude::*;
let blocks = dataset.row_blocks(256);
let predictions: Vec<f32> = blocks.block_indices()
    .into_par_iter()
    .flat_map(|block_idx| {
        let (start, block) = blocks.block(block_idx);
        block.rows().into_iter().map(|row| model.predict_one(row.as_slice().unwrap())).collect::<Vec<_>>()
    })
    .collect();
```

**Why this works**:

- Column-major storage is optimal for training (feature iteration)
- Row-major access is optimal for prediction (sample iteration)
- Buffering amortizes the column→row transpose cost
- Block size of 256-1024 fits in L2 cache
- Parallelism via `Parallelism` enum integrates with existing infrastructure

**Cleanup**: Prediction code should delegate to `RowBlocks` rather than implementing its own transpose logic.

### Future Considerations

1. **SIMD-aligned storage**: Current `Box<[f32]>` may not be SIMD-aligned. Future optimization could use aligned allocations.
2. **Feature scaling**: Storing per-feature scale/offset for standardization could benefit linear trees. Out of scope for this RFC.
3. **Ordinal features**: Features that are ordered categories (e.g., education level). Currently treat as numeric. May warrant dedicated handling in future.
4. **Prediction-only dataset**: A lighter-weight type without bins for pure prediction. Deferred - overhead is acceptable.

## References

- LightGBM `dataset.h`: `raw_data_`, `numeric_feature_map_`, `BinType` enum
- RFC-0010: Linear Trees design
- RFC-0017: EFB Training Integration
- Current `feature_analysis.rs`: Binary/trivial/sparse detection
