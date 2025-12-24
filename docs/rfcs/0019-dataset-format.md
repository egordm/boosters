# RFC-0019: Dataset Format

- **Status**: Draft
- **Created**: 2025-12-23
- **Updated**: 2025-12-23
- **Depends on**: RFC-0001 (Data Matrix), RFC-0004 (Quantization and Binning)
- **Related**: RFC-0020 (Data Access Layer)
- **Scope**: User-facing dataset abstraction for model-level APIs

## Summary

This RFC defines `Dataset`, a unified user-facing data container that serves as the standard input for all boosters model types (GBDT, GBLinear). The current `dataset.rs` will be deleted and replaced entirely.

## Motivation

### Goals

| Goal | Priority | Description |
| ---- | -------- | ----------- |
| **Unified input** | Must | Single `Dataset` type for GBDT, GBLinear, and future boosters |
| **Feature names** | Must | String identifiers for features |
| **Targets & weights** | Must | Built-in support for target column(s) and sample weights |
| **Categorical support** | Must | Either as floats with metadata OR as integers - see Open Question 1 |
| **Python construction** | Must | From pandas, numpy, other dataframe formats |
| **Zero-copy** | Should | When constructing from external sources |
| **CSR sparse** | Should | Optional sparse column representation |
| **Dtype preservation** | Nice | Timestamps, original types as metadata |
| **Memory efficiency** | Nice | u8 for binary, etc. |

### Non-Goals

- **Serialization/storage**: We only handle conversion FROM pandas/numpy/dataframes. No file formats.
- **Low/mid-level usage**: Dataset is for model-level APIs. Algorithms use specialized views after conversion.
- **Performance degradation**: Changes must not slow down normal code paths.

## Research

### How Other Libraries Handle This

#### XGBoost DMatrix

- Stores categorical values as **floats**, casts to `int32` during binning
- Uses `FeatureType` enum: `kNumerical = 0, kCategorical = 1`
- Primary storage is row-major CSR, converts to column-major for histogram building
- Supports both row and column major via adapters

```cpp
// XGBoost base types
using bst_cat_t = std::int32_t;  // Category type is int32
using bst_float = float;         // Feature values are float

// From categorical.h - cast float to category
bst_cat_t AsCat(float v) { return static_cast<bst_cat_t>(v); }

// Invalid category check (too large for float precision)
bool InvalidCat(float cat) {
    return cat < 0 || cat >= 16777216;  // 2^24 (float mantissa limit)
}
```

#### LightGBM Dataset

- Also stores values as **floats** (via `double` in input)
- Casts to int during `ValueToBin()` for categoricals
- Uses `BinType::CategoricalBin` to mark categorical features
- Native column-major storage per feature

```cpp
// From bin.cpp - categorical binning
int int_value = static_cast<int>(value);
if (int_value < 0) {
    // Negative = missing (NaN bin)
    return 0;
}
if (categorical_2_bin_.count(int_value)) {
    return categorical_2_bin_.at(int_value);
}
```

### Layout Performance (Existing Benchmark)

From our existing benchmark (commit b72721e):

```
=== Layout Benchmark (50k samples, 100 features, 50 trees) ===
RowMajor avg:    597.517 ms, stride=100
ColumnMajor avg: 519.071 ms, stride=1
Speedup: 1.15x (13% faster)
```

**Conclusion**: Column-major (feature-major) is better for histogram building. We adopt column-major internally for consistency.

## Migration from Current Implementation

The current `Dataset` in `src/data/dataset.rs` uses a different approach:

| Aspect | Current | New (RFC-0019) |
| ------ | ------- | -------------- |
| Category storage | `Vec<i32>` in `FeatureColumn::Categorical` | `f32` with `FeatureType::Categorical` metadata |
| Feature storage | `Vec<FeatureColumn>` (owned, per-column) | `Vec<Column>` (dense or sparse per-column) |
| Target shape | `Vec<f32>` (single-output only) | `Array2<f32>` [n_samples, n_outputs] |
| Binned conversion | `to_binned()` method on Dataset | `BinnedFeatures::from_dataset()` |

**Why change to float categories?**

1. Matches XGBoost/LightGBM internal representation
2. Enables uniform storage (all f32)
3. Simplifies prediction path (no type branching per feature)
4. Internally, we already cast categories to float for GBDT prediction

**Migration path:**

The old `Dataset` will be deleted completely. Code using it will update to the new API:

```rust
// Old:
let features = vec![
    FeatureColumn::Categorical { name: None, values: vec![1, 2, 1] },
];
let ds = Dataset::new(features, targets)?;

// New:
let ds = Dataset::builder()
    .add_categorical("feature_0", array![1.0, 2.0, 1.0].view())
    .targets(targets.view())
    .build()?;
```

## Design

### Core Principles

1. **Categories as floats with metadata** - following XGBoost/LightGBM approach
2. **No deprecated methods** - clean API only
3. **Multi-output targets as Array2** - shape `(n_samples, n_outputs)`, single-output has shape `(n_samples, 1)`
4. **Layout flexibility** - support both row and column major via ndarray views

### Category Value Semantics

| Value | Interpretation |
| ----- | -------------- |
| `0.0, 1.0, 2.0, ...` | Valid category IDs (cast to i32) |
| `NaN` | Missing value |
| Negative values | Treated as missing (like XGBoost) |
| Values ≥ 2^24 | Warning logged, may lose precision |
| Fractional values | Warning logged, truncated to int |

### Core Types

```rust
/// The unified dataset container for all boosters models.
pub struct Dataset {
    /// Feature data
    features: FeatureStorage,
    
    /// Feature metadata (names, types)
    schema: DatasetSchema,
    
    /// Target values: [n_samples, n_outputs] (n_outputs=1 for single-output)
    targets: Option<Array2<f32>>,
    
    /// Sample weights [n_samples]
    weights: Option<Array1<f32>>,
    
    /// Number of samples
    n_samples: usize,
}

/// Schema describing the dataset structure.
pub struct DatasetSchema {
    /// Per-feature metadata
    features: Vec<FeatureMeta>,
    
    /// Feature name → index mapping (optional)
    name_index: Option<HashMap<String, usize>>,
}

/// Metadata for a single feature.
pub struct FeatureMeta {
    /// Feature name (optional)
    pub name: Option<String>,
    
    /// Feature type
    pub feature_type: FeatureType,
}

/// Logical feature types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureType {
    /// Continuous numeric feature.
    /// Missing values: f32::NAN
    Numeric,
    
    /// Categorical feature stored as float, interpreted as integer category ID.
    /// Missing values: f32::NAN (or negative values)
    /// Valid categories: 0.0, 1.0, 2.0, ..., n_categories-1.0
    /// 
    /// Why floats? 
    /// - XGBoost and LightGBM both use this approach
    /// - Enables uniform storage (all f32)
    /// - Cast to int during binning: `category_id = value as i32`
    Categorical,
}
```

### Feature Storage

```rust
/// Feature storage is simply a vector of columns.
/// 
/// Dataset is always column-major (feature-major) internally.
/// This provides consistency across all use cases:
/// - Training uses column-major directly (optimal for histograms)
/// - Prediction buffers to row-major in blocks (handled by predictor)
/// 
/// No enum needed - just Vec<Column> for uniform storage.
type FeatureStorage = Vec<Column>;

/// Single feature column.
pub enum Column {
    /// Dense array
    Dense(Array1<f32>),
    
    /// Sparse CSR-like column
    Sparse(SparseColumn),
}

/// Sparse column storage.
pub struct SparseColumn {
    /// Non-zero row indices (sorted)
    indices: Vec<u32>,
    
    /// Values at those indices
    values: Vec<f32>,
    
    /// Total number of samples
    n_samples: usize,
    
    /// Default value for unspecified entries
    default: f32,
}
```

### Construction API

```rust
impl Dataset {
    /// Create from dense feature matrix and targets.
    /// 
    /// # Arguments
    /// * `features` - Feature matrix `[n_samples, n_features]`
    /// * `targets` - Target values `[n_samples, n_outputs]`
    pub fn new(
        features: ArrayView2<f32>,
        targets: ArrayView2<f32>,
    ) -> Self;
    
    /// Builder for complex construction.
    pub fn builder() -> DatasetBuilder;
}

/// Builder for complex dataset construction.
pub struct DatasetBuilder {
    // ...
}

impl DatasetBuilder {
    /// Add a numeric feature column.
    pub fn add_feature(self, name: &str, values: ArrayView1<f32>) -> Self;
    
    /// Add a categorical feature column.
    /// Values should be non-negative integers encoded as floats.
    pub fn add_categorical(self, name: &str, values: ArrayView1<f32>) -> Self;
    
    /// Add a sparse feature column.
    pub fn add_sparse(
        self, 
        name: &str, 
        indices: &[u32], 
        values: &[f32],
        n_samples: usize,
    ) -> Self;
    
    /// Set all feature names at once.
    pub fn feature_names(self, names: &[&str]) -> Self;
    
    /// Set targets.
    pub fn targets(self, targets: ArrayView2<f32>) -> Self;
    
    /// Set sample weights.
    pub fn weights(self, weights: ArrayView1<f32>) -> Self;
    
    /// Build the dataset.
    pub fn build(self) -> Result<Dataset, DatasetError>;
}

/// Dataset construction errors with helpful diagnostics.
#[derive(Debug, Clone)]
pub enum DatasetError {
    /// Shape mismatch between components.
    ShapeMismatch {
        expected: usize,
        got: usize,
        field: &'static str,
    },
    
    /// Invalid category value detected.
    InvalidCategory {
        feature_idx: usize,
        value: f32,
        reason: &'static str,
    },
    
    /// Sparse column indices not sorted.
    UnsortedSparseIndices {
        feature_idx: usize,
    },
    
    /// Duplicate sparse indices.
    DuplicateSparseIndices {
        feature_idx: usize,
        index: u32,
    },
    
    /// No features provided.
    EmptyFeatures,
}
```

### Python Interop

Zero-copy construction from numpy is a primary goal. The Python bindings will use:

```rust
// In boosters-python crate
use numpy::{PyArray2, PyReadonlyArray2};

#[pyclass]
struct PyDataset(Dataset);

#[pymethods]
impl PyDataset {
    /// Create from numpy arrays.
    /// 
    /// Zero-copy when arrays are:
    /// - f32 dtype
    /// - C-contiguous or F-contiguous
    /// - Properly aligned
    #[new]
    fn new(
        py: Python<'_>,
        features: PyReadonlyArray2<f32>,
        targets: PyReadonlyArray2<f32>,
    ) -> PyResult<Self> {
        // PyReadonlyArray2 provides ArrayView2 without copy
        let features_view = features.as_array();
        let targets_view = targets.as_array();
        
        // If we need to own the data, copy here
        // Otherwise, we need lifetime management via Arc or similar
        // 
        // Decision: Copy at construction for now (simple, safe)
        // Future: Investigate Cow<[f32]> for zero-copy
        let dataset = Dataset::new(features_view, targets_view);
        Ok(PyDataset(dataset))
    }
}
```

**Current approach**: Copy at Python boundary (simple, safe)
**Future optimization**: `Cow<'a, [f32]>` or `Arc<[f32]>` for zero-copy

**Internal storage**: `[n_features, n_samples]` in C-contiguous layout. Each feature is a contiguous slice.

**Input handling**: User typically provides `[n_samples, n_features]` (numpy default). We transpose once at construction. This is acceptable one-time overhead.

### Algorithm Integration

#### GBDT Training

```rust
impl GBDTModel {
    pub fn train(dataset: &Dataset, config: &GBDTConfig) -> Result<Self, TrainError> {
        // 1. Convert to binned format
        let binned = BinnedDataset::from_dataset(dataset, config.max_bins)?;
        
        // 2. Train on binned data
        // Categories handled during binning - BinMapper knows feature types
    }
}
```

The BinMapper handles categorical conversion:

```rust
impl BinMapper {
    fn map_value(&self, value: f32, feature_type: FeatureType) -> u8 {
        match feature_type {
            FeatureType::Numeric => self.map_numeric(value),
            FeatureType::Categorical => {
                if value.is_nan() || value < 0.0 {
                    self.missing_bin
                } else {
                    let cat_id = value as i32;
                    self.category_to_bin.get(&cat_id)
                        .copied()
                        .unwrap_or(self.missing_bin)
                }
            }
        }
    }
}
```

#### GBDT Prediction

```rust
impl GBDTModel {
    pub fn predict(&self, dataset: &Dataset) -> Array2<f32> {
        let features = dataset.features_view();
        
        // Tree traversal uses float values directly
        // Categorical splits compare: `value as i32 == split_category`
    }
}
```

#### GBLinear Training

```rust
impl GBLinearModel {
    pub fn train(dataset: &Dataset, config: &GBLinearConfig) -> Result<Self, TrainError> {
        // Validate no categorical features (or require one-hot encoding)
        if dataset.has_categorical() {
            return Err(TrainError::CategoricalNotSupported);
        }
        
        let features = dataset.features_view();
        // Coordinate descent on raw float values
        // Column-major is optimal for coordinate descent (one feature at a time)
    }
}
```

#### GBLinear Prediction

GBLinear prediction works efficiently with column-major data by iterating over features rather than samples. Benchmarked at only 21% overhead vs optimal row-major - see RFC-0020 for implementation details and benchmark data.

### Dataset Methods for View Access

```rust
impl Dataset {
    /// Get a feature-major view for algorithms.
    /// 
    /// Returns a view with shape [n_features, n_samples].
    /// Optimal for column-wise access (histograms, coordinate descent).
    /// 
    /// For prediction (which needs row-major access), the predictor
    /// handles block buffering internally - no view method needed here.
    pub fn features_view(&self) -> FeaturesView<'_>;
    
    /// Get a view of the target data.
    pub fn targets_view(&self) -> Option<TargetsView<'_>>;
    
    /// Get sample weights.
    pub fn weights(&self) -> Option<ArrayView1<'_, f32>>;
    
    /// Number of samples.
    pub fn n_samples(&self) -> usize;
    
    /// Number of features.
    pub fn n_features(&self) -> usize;
}
```

**Note**: No `to_samples_view()` method. The predictor handles row-major buffering internally,
so Dataset doesn't need to expose a samples-view. This keeps the API focused.

## Open Questions

### 1. Categorical Storage: Floats vs Integers?

**Decision: Floats with metadata** (following XGBoost/LightGBM)

- Store all values as f32
- Mark categorical features in schema via `FeatureType::Categorical`
- Cast to int during binning/splitting
- Matches industry standard
- Simpler uniform storage

### 2. Owned vs View Storage

**Decision: Start with owned, track zero-copy as future work**

For MVP, `Dataset` owns all data (copy at construction). This is simple and safe.

Future optimization: `Cow<'a, [f32]>` or `Arc<[f32]>` for zero-copy from numpy.

### 3. Layout Handling

**Decision: Dataset is always column-major (feature-major)**

This provides consistency and simplifies the API:

1. **Training**: Uses column-major directly (optimal for histogram building)
2. **Prediction**: Predictor buffers to row-major in blocks (~5% overhead, benchmarked)
3. **User input**: If user provides row-major, we transpose once at Dataset construction

Benefits:

- Consistent internal representation
- Simpler API (no layout choice for users)
- Optimal training path (the expensive part)
- Prediction overhead is acceptable and handled internally

**View types for internal use**:

- `FeaturesView` - Feature-major [n_features, n_samples] - for training algorithms
- `SamplesView` - Sample-major [n_samples, n_features] - used internally by predictor

```rust
// Existing types in src/data/ndarray.rs
pub struct FeaturesView<'a>(ArrayView2<'a, f32>); // [n_features, n_samples] C-order
pub struct SamplesView<'a>(ArrayView2<'a, f32>);  // [n_samples, n_features] C-order
```

**Layout requirements by algorithm**:

- **GBDT Training (histograms)**: Column-major via `FeaturesView` - optimal
- **GBDT Prediction (tree traversal)**: Predictor buffers internally
- **GBLinear Training (coordinate descent)**: Column-major via `FeaturesView` - optimal
- **GBLinear Prediction**: Column-major works (iterate by feature)

## Integration

| Component | Integration Point | Notes |
| --------- | ----------------- | ----- |
| RFC-0001 (Data Matrix) | `features_view()` | Returns ndarray view |
| RFC-0004 (Binning) | `BinnedDataset::from_dataset()` | Converts for GBDT training |
| RFC-0020 (Data Access) | View types | Algorithms access via views |
| Python bindings | `from_numpy()`, `from_pandas()` | Zero-copy where possible |

## Testing Scenarios

| Scenario | Expected Behavior |
| -------- | ----------------- |
| Empty dataset (0 samples) | Valid empty Dataset |
| Single sample | Works without divide-by-zero |
| All categorical features | Schema marks all as Categorical |
| Mixed dense/sparse | Both column types work |
| Multi-output targets | Shape `(n_samples, n_outputs)` |
| Category value = 16777215 | Valid, no warning |
| Category value = 16777216 | Warning logged, value used |
| Negative category values | Treated as missing |
| Thread safety | Dataset is Send + Sync |
| Large dataset (1M samples) | No overflow in indices |
| Input [n_samples, n_features] C-order | Transpose to [n_features, n_samples] |
| Input [n_features, n_samples] C-order | Accept directly (features contiguous) |

## Acceptance Criteria

1. All existing tests pass after migration
2. No regression in training speed (< 5% slowdown)
3. No regression in prediction speed (< 5% slowdown)
4. Python bindings work with numpy arrays
5. All testing scenarios pass

## Module Organization

```text
boosters/
├── src/
│   ├── dataset/
│   │   ├── mod.rs           # pub use exports
│   │   ├── dataset.rs       # Dataset, DatasetBuilder
│   │   ├── schema.rs        # DatasetSchema, FeatureMeta, FeatureType
│   │   ├── storage.rs       # Column, SparseColumn
│   │   └── error.rs         # DatasetError variants
│   ├── views/
│   │   ├── mod.rs           # pub use exports
│   │   ├── features.rs      # FeaturesView
│   │   ├── targets.rs       # TargetsView
│   │   └── binned.rs        # BinnedFeatures (or in binning module)
│   └── lib.rs
```

## Validation Requirements

| Validation | When | Timing | Behavior |
| ---------- | ---- | ------ | -------- |
| Category precision | Construction | Lazy (per feature on first use) | Warn if any value ≥ 2^24 |
| Sparse indices sorted | Construction | Eager | Error if not strictly ascending |
| Sparse no duplicates | Construction | Eager (during sorted check) | Error if duplicate indices |
| Shape consistency | Construction | Eager | Error if targets.n_samples != features.n_samples |
| Feature count match | Construction | Eager | Error if schema.len() != n_features |

**Precision edge cases:**
- Value = 16777215.0 (2^24 - 1): Valid, no warning
- Value = 16777216.0 (2^24): Warning logged, value used as-is

## Future Work

- [ ] Memory-mapped features for out-of-core training
- [ ] GPU storage variant
- [ ] Streaming construction
- [ ] Automatic one-hot encoding option for GBLinear
- [ ] Cow<[f32]> storage for zero-copy from numpy

## See Also

- **RFC-0020: Data Access Layer** - How algorithms access data from Dataset

## Changelog

- 2025-12-23: Initial draft based on stakeholder feedback and XGBoost/LightGBM research
- 2025-12-23: Added module organization, validation requirements, Python interop section
- 2025-12-23: Added category value semantics table and edge case handling
- 2025-12-23: Added migration section explaining change from `Vec<i32>` to floats
- 2025-12-23: Marked open questions as decided based on design review
- 2025-12-23: Added acceptance criteria and expanded testing scenarios
- 2025-12-24: Rounds 7-10: Added view access methods, added Python performance tip
- 2025-12-24: Layout decision: Dataset is always column-major (feature-major)
- 2025-12-24: Removed `to_samples_view()` - predictor handles buffering internally
- 2025-12-24: Design review complete - ready for implementation
- 2025-12-24: Removed FeatureStorage enum - just Vec<Column> now
- 2025-12-24: Added layout validation - error on row-major input
- 2025-12-24: Added GBLinear prediction section showing column-major compatibility
- 2025-12-24: Simplified GBLinear section - moved implementation to RFC-0020
- 2025-12-24: Added layout validation testing scenarios
