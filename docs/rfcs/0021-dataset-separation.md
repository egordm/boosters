# RFC-0021: Data Module Restructuring and Dataset Separation

**Status**: Draft  
**Created**: 2025-12-30  
**Updated**: 2025-12-30  
**Author**: Team  
**Supersedes**: RFC-0018, RFC-0008 (io-parquet)  
**Related**: RFC-0019 (updated to work on Dataset)

## Summary

This RFC proposes a comprehensive restructuring of the data module with clear architectural separation between raw data (`Dataset`) and binned data (`BinnedDataset`). The current approach has grown complex because:

1. Complex storage types (`NumericStorage`, `SparseNumericStorage`, etc.) that track both bin indices and raw values
2. Difficult-to-test APIs because constructing a `BinnedDataset` requires going through the full binning pipeline
3. Confusion about which type to use where
4. Memory duplication (raw values stored in both `Dataset` and `BinnedDataset`)

**Proposed solution**: Two distinct types with clear, separate responsibilities:

- **`Dataset`**: Contains raw feature data (dense or sparse), targets, and weights. Used for prediction, inference, SHAP values, eval sets, and anywhere raw feature values are needed. This is the **public API**.
- **`BinnedDataset`**: Contains ONLY binned data (bin indices, bin mappers). Used exclusively for histogram building during training. This is an **internal API** not exposed to users.

**Key architectural principles**:

1. **Models** (high-level API) work only with `Dataset`. They internally create `BinnedDataset` when needed.
2. **Trainers** receive `Dataset` (mandatory) and `BinnedDataset` (mandatory). Prediction uses `Dataset`.
3. **Eval sets** are just `Dataset` (no `BinnedDataset` needed - they're only used for prediction/metrics).
4. **BinnedDataset** is internal - no convenience methods, no public exposure.
5. **Feature grouping** is handled by `BinnedDataset` internally, not by users in `Dataset`.
6. **No Arc references** - just copy data when needed. Keep it simple.

## Motivation

### Current Problems

**1. BinnedDataset is Hard to Test**

To create a `BinnedDataset` for unit tests, you must:
```rust
// Current: Go through entire builder pipeline
let built = DatasetBuilder::from_array(data.view(), &config)
    .unwrap()
    .set_targets_1d(labels.view())
    .build_groups()
    .unwrap();
let dataset = BinnedDataset::from_built_groups(built);
```

There's no way to directly construct a `BinnedDataset` with known bin values for testing.

**2. Raw Data Duplication**

When training with linear trees or GBLinear:
- User provides raw data → stored in `Dataset` 
- During binning, raw values are copied again → stored in `NumericStorage`
- Memory usage is 2x what's needed

**3. Storage Type Explosion**

We have 5 storage types because each must track both bins AND raw values:
- `NumericStorage` (bins + raw)
- `SparseNumericStorage` (sparse bins + sparse raw)
- `CategoricalStorage` (bins only)
- `SparseCategoricalStorage` (sparse bins only)
- `BundleStorage` (encoded bins, no raw)

If we separate concerns, storage becomes simpler:
- `DenseBins` (just u8/u16 bins)
- `SparseBins` (sparse u8/u16 bins)
- `BundleBins` (encoded u16 for EFB)

**4. RFC-0019 Complexity**

RFC-0019 proposes `for_each_feature_value()` and `gather_feature_values()` on `BinnedDataset`. But these methods only make sense for numeric features with raw storage. For bundled or categorical features, they panic. This is a design smell—we're adding APIs that only work for some storage types.

### What Components Need What Data

| Component | Needs Bins | Needs Raw | Needs Targets | Needs Weights |
|-----------|------------|-----------|---------------|---------------|
| GBDT histogram building | ✅ | ❌ | ❌ (uses grads) | ❌ |
| GBDT split finding | ✅ | ❌ | ❌ | ❌ |
| GBDT prediction | ❌ | ✅ (split thresholds) | ❌ | ❌ |
| GBLinear training | ❌ | ✅ | ✅ | ✅ |
| GBLinear prediction | ❌ | ✅ | ❌ | ❌ |
| Linear tree fitting | ❌ | ✅ | ✅ | ✅ |
| Linear SHAP | ❌ | ✅ | ❌ | ❌ |
| Tree SHAP | ❌ | ✅ | ❌ | ❌ |
| Gradient computation | ❌ | ❌ | ✅ | ✅ |

**Key insight**: Histogram building only needs bins. Everything else needs raw values. These are naturally separate concerns.

## Scope and Key Changes

This RFC covers four major changes:

### 1. Remove io/io-parquet Feature

The `io-parquet` feature and `data/io/` module are unused and add complexity. Arrow/Parquet I/O was a nice-to-have that we never shipped. Remove:

- `data/io/` module (parquet.rs, record_batches.rs, error.rs)
- `io-parquet` feature from Cargo.toml
- Arrow/Parquet dependencies
- DataSource::Parquet and real_world_configs() from quality_benchmark.rs

### 2. Module Structure Cleanup

Current structure is messy:

```text
data/
├── binned/
│   ├── sample_blocks.rs    ← WRONG: belongs with Dataset
│   └── ...
├── types/
│   ├── dataset.rs          ← "types" is vague
│   ├── views.rs
│   └── ...
└── mod.rs
```

Proposed structure with clear separation:

```text
data/
├── raw/                    ← Raw dataset module (public API)
│   ├── mod.rs
│   ├── dataset.rs          ← Dataset struct
│   ├── builder.rs          ← DatasetBuilder (simple)
│   ├── views.rs            ← TargetsView, WeightsView, FeaturesView
│   ├── sample_blocks.rs    ← SampleBlocks for prediction
│   ├── feature.rs          ← Feature enum (Dense/Sparse)
│   └── schema.rs           ← DatasetSchema, FeatureMeta
├── binned/                 ← Binned dataset module (internal)
│   ├── mod.rs
│   ├── dataset.rs          ← BinnedDataset struct
│   ├── builder.rs          ← BinnedDatasetBuilder (internal)
│   ├── storage.rs          ← BinStorage, NumericStorage, etc.
│   ├── group.rs            ← FeatureGroup
│   ├── view.rs             ← BinnedFeatureView
│   ├── bundling.rs         ← EFB bundling logic
│   └── bin_mapper.rs       ← BinMapper
└── mod.rs                  ← Re-exports Dataset publicly, BinnedDataset pub(crate)
```

Key principle: **Everything related to raw datasets goes in `raw/`**, everything related to binned datasets goes in `binned/`. No cross-contamination.

### 3. Single Validation Set

Current API is overly complex:

```rust
// Current: Multiple named eval sets
let eval_sets = vec![
    EvalSet::new("train", &train_ds),
    EvalSet::new("valid", &valid_ds),
];
model.fit(&train_ds, &eval_sets, config)?;
```

Proposed: Just one optional validation set:

```rust
// Proposed: Simple optional validation set
model.fit(&train_ds, Some(&valid_ds), config)?;
// or
model.fit(&train_ds, None, config)?;
```

Rationale:
- If users need multiple evaluation sets, they evaluate after training
- Training needs at most one validation set for early stopping
- Removes `EvalSet` struct complexity
- Simpler Python bindings (`val_set: Dataset | None`)

> Note that early stopping can only work when a validation set is supplied, otherwise it makes no sense.

### 4. BinnedDataset Simplification

**Remove original_feature vs effective_feature distinction:**

Current API is confusing:
```rust
dataset.effective_feature_views()  // Views with bundles
dataset.original_feature_view(idx) // Single feature view
```

Proposed:
```rust
dataset.feature_views()            // Views for training (bundles + standalone)
dataset.n_features()               // Number of training features
dataset.bin_mapper(idx)            // Bin mapper for feature
```

The `original_feature` concept only mattered for translating split results back to user-facing feature indices. With the split decoding handled internally by the grower, callers don't need to think about this.

**Remove effective_ prefix:**

All methods that have `effective_` prefix should drop it. There's no "original" vs "effective" - there's just features as BinnedDataset sees them.

**Remove BinnedDatasetBuilder entirely:**

Since we have `BinnedDataset::from_dataset()`, the builder pattern is unnecessary. The current `BinnedDatasetBuilder` (~700 lines) exists to support multiple input paths, but `BinnedDataset` is internal and ALWAYS created from `Dataset`. The builder's internal helper functions move into the `from_dataset()` implementation.

```rust
// Current: Complex builder with multiple entry points
let builder = BinnedDatasetBuilder::from_array(data, &config)?
    .set_targets_1d(labels.view())
    .build_groups()?;
let dataset = BinnedDataset::from_built_groups(built);

// Proposed: Single factory method
let binned = BinnedDataset::from_dataset(&dataset, &config)?;
```

**What gets deleted:**

- `BinnedDatasetBuilder` struct and all public methods
- `from_array()`, `from_array_with_metadata()`, `new()` entry points
- `add_numeric()`, `add_categorical()`, `add_sparse()` methods
- `set_targets()`, `set_weights()` methods (targets/weights come from Dataset)
- `build_groups()`, `build()` methods

**What gets preserved (moved into from_dataset implementation):**

- `create_bin_mappers()` helper → private function
- `build_feature_group()` helper → private function
- `build_numeric_dense()`, `build_categorical_dense()` → private functions
- `build_sparse_numeric()`, `build_sparse_categorical()` → private functions
- `build_bundle()` for EFB → private function

## Design

### Core Principle: Clear Separation of Concerns

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           HIGH-LEVEL API (Public)                       │
│                                                                         │
│   Model.fit(dataset, val_set) ──────────────────────────────────────────│
│         │                                                               │
│         ▼ (internally creates BinnedDataset for train only)             │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                         TRAINER (Internal)                      │   │
│   │   Receives: Dataset + BinnedDataset (for train)                 │   │
│   │             val_set: Option<&Dataset> (just Dataset, no bins!)  │   │
│   │                                                                 │   │
│   │   Uses BinnedDataset for: histogram building only               │   │
│   │   Uses Dataset for: prediction, eval, linear trees, SHAP       │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   Model.predict(dataset) ───────────────────────────────────────────────│
│         │                                                               │
│         ▼ (uses Dataset directly via SampleBlocks)                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

  PUBLIC: Dataset                    INTERNAL: BinnedDataset
  ├── features: Feature[]            ├── bins: BinStorage[]
  │   (dense or sparse)              ├── bin_mappers: BinMapper[]
  ├── targets: [n_outputs]           ├── feature_groups: (EFB, grouping)
  ├── weights: [n_samples]           └── n_features, n_samples
  └── schema: DatasetSchema
                                     (Groups features internally - user
  Methods (RFC-0019 on Dataset):      doesn't need to worry about this)
  ├── for_each_feature_value()
  ├── gather_feature_values()
  └── SampleBlocks (in raw/ module)
```

### Feature Grouping: Dataset vs BinnedDataset

**Question**: Should users provide grouped data in `Dataset`, or should `BinnedDataset` handle grouping?

**Answer**: `BinnedDataset` handles all grouping internally. Users provide simple per-feature columns.

**Rationale**:

- **User simplicity**: Users don't need to understand EFB, sparse grouping, etc.
- **Dataset stays simple**: Each column = one feature. Dense or sparse.
- **BinnedDataset optimizes**: During binning, it analyzes features and groups them optimally (EFB bundles, sparse groups, etc.)
- **Sparse data**: When a user provides sparse features, Dataset stores them as `FeatureColumn::Sparse`. BinnedDataset then decides how to group/bundle them for efficient histogram building.

```rust
// User provides simple columns - doesn't worry about grouping
let dataset = Dataset::builder()
    .add_dense("age", ages)
    .add_dense("income", incomes)
    .add_sparse("rare_feature", indices, values, default)
    .add_categorical("gender", genders)
    .build()?;

// BinnedDataset handles grouping internally
let binned = BinnedDataset::from_dataset(&dataset, &config)?;
// Internally: groups sparse features, applies EFB, etc.
```

### Dataset (Raw Data Container)

```rust
/// Container for raw feature data, targets, and weights.
///
/// This is the user-facing type for all data input. It's used for:
/// - Input to training (passed to trainers, which bin it internally)
/// - Prediction (passed to models directly)
/// - GBLinear and Linear SHAP (which only need raw values)
pub struct Dataset {
    /// Feature storage - [n_features], each contains [n_samples] values
    features: Box<[Feature]>,
    /// Per-feature metadata
    schema: DatasetSchema,
    /// Target values [n_outputs, n_samples]
    targets: Option<Array2<f32>>,
    /// Sample weights [n_samples]
    weights: Option<Array1<f32>>,
    /// Number of samples
    n_samples: usize,
}

/// A single feature's data storage.
pub enum Feature {
    /// Dense storage: contiguous f32 values [n_samples]
    Dense(Box<[f32]>),
    /// Sparse storage: CSC-like (indices, values, default)
    Sparse {
        indices: Box<[u32]>,  // Sample indices with non-default values (sorted)
        values: Box<[f32]>,   // Values at those indices
        default: f32,         // Default value for samples not in indices
    },
}

impl Dataset {
    /// Iterate over raw values for a feature.
    ///
    /// This is the zero-cost pattern from RFC-0019, but on Dataset
    /// where it naturally belongs.
    #[inline]
    pub fn for_each_feature_value<F>(&self, feature: usize, f: F)
    where
        F: FnMut(usize, f32),
    {
        match &self.features[feature] {
            Feature::Dense(values) => {
                for (idx, &val) in values.iter().enumerate() {
                    f(idx, val);
                }
            }
            Feature::Sparse { indices, values, .. } => {
                // Iterate only stored (non-default) values
                for (&idx, &val) in indices.iter().zip(values.iter()) {
                    f(idx as usize, val);
                }
            }
        }
    }
    
    /// Gather raw values at specific indices.
    #[inline]
    pub fn gather_feature_values(
        &self,
        feature: usize,
        sample_indices: &[u32],
        buffer: &mut [f32],
    ) {
        // Efficient gather from dense or sparse storage
    }
    
    /// Get a contiguous slice of raw values (only for dense features).
    #[inline]
    pub fn feature_slice(&self, feature: usize) -> Option<&[f32]> {
        match &self.features[feature] {
            Feature::Dense(values) => Some(values),
            Feature::Sparse { .. } => None,
        }
    }
}
```

### BinnedDataset (Internal: Binned Data for Histograms)

```rust
/// Internal binned feature data for histogram-based GBDT training.
///
/// Contains ONLY bin indices—no raw values, no reference to source.
/// This type is used exclusively for histogram building. When raw values 
/// are needed (prediction, linear trees, SHAP), use the Dataset directly.
///
/// # Visibility
///
/// This type is pub(crate)—NOT part of the public API. Users never see
/// or interact with BinnedDataset. Models create it internally from Dataset.
///
/// # Feature Grouping
///
/// BinnedDataset handles all feature grouping internally:
/// - Applies EFB (Exclusive Feature Bundling) for sparse categorical features
/// - Groups sparse features efficiently
/// - Users provide simple per-feature columns in Dataset; we optimize here
///
/// # Memory Layout
///
/// Bins are stored densely for most features, but EFB bundles combine
/// multiple sparse categorical features into a single encoded column.
pub(crate) struct BinnedDataset {
    /// Number of samples
    n_samples: usize,
    /// Per-feature bin mappers (thresholds for numeric, categories for categorical)
    bin_mappers: Box<[BinMapper]>,
    /// Feature groups (EFB bundles, sparse groups, etc.)
    groups: Vec<BinnedGroup>,
    /// Where each feature lives (which group, index in group, or skipped)
    feature_locations: Box<[FeatureLocation]>,
    /// Global bin offsets for histogram allocation
    global_bin_offsets: Box<[u32]>,
    // NOTE: No source reference! BinnedDataset is pure binned data.
    // Trainer holds both Dataset and BinnedDataset separately.
}

/// A group of features with homogeneous bin storage.
pub struct BinnedGroup {
    /// Global feature indices in this group
    feature_indices: Box<[u32]>,
    /// Bin storage (just bins, no raw values)
    bins: BinStorage,
    /// Per-feature bin counts
    bin_counts: Box<[u32]>,
}

/// Bin storage types—simpler than current because no raw values.
pub enum BinStorage {
    /// Dense bins, small bin counts (≤256)
    Dense8(Box<[u8]>),
    /// Dense bins, large bin counts (>256)  
    Dense16(Box<[u16]>),
    /// Sparse bins (for sparse features)
    Sparse8 {
        indices: Box<[u32]>,
        bins: Box<[u8]>,
        default_bin: u8,
    },
    /// EFB bundle (encoded sparse categoricals)
    Bundle {
        encoded: Box<[u16]>,
        offsets: Box<[u16]>,  // Per-feature offsets in encoded space
    },
}

impl BinnedDataset {
    /// Create from a Dataset.
    ///
    /// Bins the data according to config. Does NOT retain reference to source.
    pub fn from_dataset(
        dataset: &Dataset,
        config: &BinningConfig,
    ) -> Result<Self, BuildError> {
        // 1. Analyze features
        // 2. Create bin mappers
        // 3. Bin the data
        // 4. Apply EFB if configured
        // 5. Return pure binned data (no source reference)
    }
    
    /// Access bins for histogram building.
    #[inline]
    pub fn bin(&self, sample: usize, feature: usize) -> u32 {
        // Fast path for dense groups
    }
    
    /// Get feature bin view for histogram building.
    pub fn feature_bin_view(&self, feature: usize) -> FeatureBinView<'_> {
        // Returns view into bin storage for efficient histogram building
    }
    
    /// Get bin mapper for a feature (for split threshold lookup).
    pub fn bin_mapper(&self, feature: usize) -> &BinMapper {
        &self.bin_mappers[feature]
    }
}
```

### Schema Handling

**Decision**: Schema sharing between `Dataset` and `BinnedDataset` is **optional**—only do it if convenient.

EFB (Exclusive Feature Bundling) complicates schema sharing because:
- `Dataset` has N original features
- `BinnedDataset` may have fewer "effective" features due to bundling
- Bundled features have different bin counts than original features

**Practical approach**: 

- `Dataset` owns `DatasetSchema` with user-facing feature metadata
- `BinnedDataset` tracks its own internal feature mapping (groups, offsets, bin counts)
- No shared schema type—each type has what it needs

```rust
// Dataset schema - user-facing
pub struct DatasetSchema {
    features: Box<[FeatureMeta]>,
    name_index: OnceCell<HashMap<String, usize>>,
}

// BinnedDataset tracks its own internal structure
pub(crate) struct BinnedDataset {
    n_samples: usize,
    bin_mappers: Box<[BinMapper]>,           // One per ORIGINAL feature
    groups: Vec<BinnedGroup>,                 // Effective groups (may have bundles)
    feature_locations: Box<[FeatureLocation]>, // Where each original feature lives
    // No schema reference needed - internal type
}
```

If we later find a convenient way to share, we can add it. But don't force it.

### Construction Patterns

**High-level: User creates Dataset, Model handles everything:**

```rust
// User code - only works with Dataset
let dataset = Dataset::builder()
    .add_column("age", Column::Dense(ages))
    .add_column("gender", Column::Dense(genders))
    .set_categorical(&["gender"])
    .targets(targets)
    .weights(weights)
    .build()?;

// Model.fit() internally creates BinnedDataset
let model = GBDTModel::fit(&dataset, &config)?;

// Prediction uses Dataset directly
let predictions = model.predict(&dataset)?;
```

**Mid-level: Trainer receives both Dataset and BinnedDataset:**

```rust
// Model or internal code creates both
let dataset = Dataset::builder()...build()?;
let binned = BinnedDataset::from_dataset(&dataset, &binning_config)?;

// Trainer receives both (mandatory for training data)
// val_set is just Option<&Dataset> - only needs prediction, not histograms!
let trainer = GBDTTrainer::new(&training_config)?;
let model = trainer.train(
    &dataset,       // For prediction during training, linear trees, SHAP
    &binned,        // For histogram building only (internal)
    val_set,        // Option<&Dataset> - used for validation metrics
)?;
```

**Low-level: Direct feature access via abstractions:**

```rust
// Raw value access (from Dataset)
dataset.for_each_feature_value(feature_idx, |sample, value| {
    // Used by: prediction, linear models, SHAP
});

// Bin access (from BinnedDataset)  
let bin_view = binned.feature_bin_view(feature_idx);
bin_view.for_each_bin(|sample, bin| {
    // Used by: histogram building only
});
```

**Direct BinnedDataset construction for tests:**

```rust
// Test code - can directly construct with known bins
let binned = BinnedDataset::test_builder()
    .n_samples(5)
    .add_bins(0, vec![0, 1, 2, 3, 4])  // Feature 0 bins
    .add_bins(1, vec![0, 0, 1, 1, 2])  // Feature 1 bins
    .add_bin_mapper(0, BinMapper::numeric(vec![0.5, 1.5, 2.5, 3.5]))
    .add_bin_mapper(1, BinMapper::categorical(3))
    .build();

// Now we can test histogram building with known bins!
let hist = build_histogram(&binned, feature=0, &gradients);
assert_eq!(hist[0], expected_bin_0);
```

### Memory Comparison

**Current (BinnedDataset stores raw values):**
```
Dataset:       features [n_features × n_samples × 4 bytes]  = 40 MB (10K samples × 1K features)
BinnedDataset: bins     [n_features × n_samples × 1 byte]   = 10 MB
               raw      [n_features × n_samples × 4 bytes]  = 40 MB (duplicated!)
Total: 90 MB
```

**Proposed (BinnedDataset references Dataset):**
```
Dataset:       features [n_features × n_samples × 4 bytes]  = 40 MB
BinnedDataset: bins     [n_features × n_samples × 1 byte]   = 10 MB
               source   Arc<Dataset> (shared reference)     = 0 MB
Total: 50 MB (44% reduction)
```

### API Changes

**Before (confusing):**

```rust
// Model takes Dataset, bins internally
GBDTModel::train(dataset: &Dataset, ...)

// Or takes BinnedDataset with raw values inside
GBDTModel::train_binned(dataset: &BinnedDataset, ...)

// Prediction takes Dataset
model.predict(&Dataset) -> predictions

// Linear model accesses raw values from BinnedDataset (??)
binned_dataset.for_each_feature_value(...)  // panics for categorical!
```

**After (clear hierarchy):**

```rust
// HIGH-LEVEL: Model works only with Dataset
let model = GBDTModel::fit(&dataset, &config)?;  // Internally creates BinnedDataset
let predictions = model.predict(&dataset)?;      // Uses Dataset directly

// MID-LEVEL: Trainer receives both (explicit control)
let trainer = GBDTTrainer::new(&config)?;
let model = trainer.train(&dataset, &binned, val_set)?;

// LOW-LEVEL: Abstractions for specific access patterns
// Raw values from Dataset:
dataset.for_each_feature_value(feat, |idx, val| ...);
dataset.gather_feature_values(feat, indices, buffer);

// Bins from BinnedDataset:
binned.feature_bin_view(feat).for_each_bin(|idx, bin| ...);
```

### What Gets Removed (Comprehensive Inventory)

This section documents every file, struct, trait, and method that will be removed after this RFC is implemented.

#### Files to Delete Entirely

| File | Lines | Description |
|------|-------|-------------|
| `data/io/mod.rs` | ~50 | I/O module root (parquet) |
| `data/io/parquet.rs` | ~200 | Parquet I/O implementation |
| `data/io/record_batches.rs` | ~150 | Arrow RecordBatch handling |
| `data/io/error.rs` | ~50 | I/O error types |
| `data/types/accessor.rs` | ~130 | DataAccessor, SampleAccessor traits |
| `data/binned/builder.rs` | ~700 | BinnedDatasetBuilder (logic moves to from_dataset) |

**Total: ~1,280 lines deleted**

#### Files to Move/Rename

| From | To |
|------|-----|
| `data/types/dataset.rs` | `data/raw/dataset.rs` |
| `data/types/views.rs` | `data/raw/views.rs` |
| `data/types/column.rs` | `data/raw/column.rs` |
| `data/types/schema.rs` | `data/raw/schema.rs` |
| `data/types/mod.rs` | DELETE (replaced by `data/raw/mod.rs`) |
| `data/binned/sample_blocks.rs` | `data/raw/sample_blocks.rs` |

#### Structs to Delete

| Struct | Location | Replacement |
|--------|----------|-------------|
| `EvalSet` | `training/eval.rs` | `Option<&Dataset>` parameter |
| `PyEvalSet` | `boosters-python/src/data.rs` | `val_set: Dataset \| None` |
| `BinnedDatasetBuilder` | `data/binned/builder.rs` | `BinnedDataset::from_dataset()` |
| `BuiltGroups` | `data/binned/builder.rs` | Internal to from_dataset |
| `DataSource::Parquet` | `quality_benchmark.rs` | Remove variant entirely |

#### Traits to Delete

| Trait | Location | Replacement |
|-------|----------|-------------|
| `DataAccessor` | `data/types/accessor.rs` | `Dataset::for_each_feature_value()` |
| `SampleAccessor` | `data/types/accessor.rs` | `Dataset::gather_feature_values()` |

#### Methods to Delete/Rename

| Current Method | On Type | Action |
|----------------|---------|--------|
| `effective_feature_views()` | `BinnedDataset` | Rename to `feature_views()` |
| `effective_feature_count()` | `BinnedDataset` | Rename to `n_features()` |
| `original_feature_view()` | `BinnedDataset` | DELETE |
| `original_feature_count()` | `BinnedDataset` | DELETE (use schema if needed) |
| `from_built_groups()` | `BinnedDataset` | DELETE (internal to from_dataset) |
| `from_array()` | `BinnedDatasetBuilder` | DELETE |
| `from_array_with_metadata()` | `BinnedDatasetBuilder` | DELETE |
| `build_groups()` | `BinnedDatasetBuilder` | DELETE |

#### Cargo.toml Changes

```toml
# Features to remove
[features]
io-parquet = ["arrow", "parquet"]  # DELETE

# Dependencies to remove
[dependencies]
arrow = { ... }      # DELETE
parquet = { ... }    # DELETE
```

#### Python API Changes

```python
# Before (to remove)
from boosters import EvalSet
eval_set = EvalSet("valid", valid_data)
model.fit(train, eval_set=[eval_set])

# After
model.fit(train, val_set=valid_data)  # or val_set=None
```

#### Storage Type Simplification

| Current Type | Contains | After |
|--------------|----------|-------|
| `NumericStorage` | bins + raw_values | bins only (raw in Dataset) |
| `SparseNumericStorage` | bins + raw_values | bins only |
| `CategoricalStorage` | bins only | unchanged |
| `SparseCategoricalStorage` | bins only | unchanged |
| `BundleStorage` | encoded bins | unchanged |

#### Constants to Remove

| Constant | Location |
|----------|----------|
| `CALIFORNIA_HOUSING_PATH` | `quality_benchmark.rs` |
| `ADULT_PATH` | `quality_benchmark.rs` |
| `COVERTYPE_PATH` | `quality_benchmark.rs` |

### Migration Path

**Phase 0: Remove io-parquet (Independent)**

1. Delete `data/io/` module entirely
2. Remove `io-parquet` feature from Cargo.toml
3. Remove arrow/parquet dependencies from Cargo.toml
4. Update `quality_benchmark.rs`:
   - Remove `#[cfg(feature = "io-parquet")]` imports
   - Remove `DataSource::Parquet` variant
   - Remove `real_world_configs()` function
   - Remove `CALIFORNIA_HOUSING_PATH`, `ADULT_PATH`, `COVERTYPE_PATH` constants
   - Update docstrings (remove `--features io-parquet` from usage examples)

**Phase 1: Module Restructuring**

1. Create `data/raw/` directory
2. Move from `data/types/`: `dataset.rs`, `views.rs`, `column.rs`, `schema.rs` → `data/raw/`
3. Move `data/binned/sample_blocks.rs` → `data/raw/sample_blocks.rs`
4. Delete `data/types/accessor.rs` (DataAccessor, SampleAccessor traits removed)
5. Delete `data/types/` directory
6. Update `data/mod.rs` re-exports

**Phase 2: API Simplification**

1. Change `eval_sets: &[EvalSet<'_>]` to `val_set: Option<&Dataset>` in all APIs
2. Remove `EvalSet` struct from `training/eval.rs`
3. Remove `PyEvalSet` class from Python bindings
4. Rename `effective_feature_views()` → `feature_views()`
5. Remove `effective_` prefix from all BinnedDataset methods
6. Remove `original_feature_view()` method
7. Python API: `eval_set=[...]` → `val_set=...`

**Phase 3: BinnedDataset Simplification**

1. Delete `BinnedDatasetBuilder` struct entirely
2. Delete `data/binned/builder.rs` file 
3. Move builder helper functions into `BinnedDataset::from_dataset()` implementation
4. Add `BinnedDataset::test_builder()` for unit tests (simple, ~50 lines)
5. Simplify storage types (no raw values in BinnedDataset)
6. Remove raw value storage from `NumericStorage`, `SparseNumericStorage`

**Phase 4: Cleanup**

1. Remove deprecated old `Dataset` type in `data/types/`
2. Delete `DataAccessor` and `SampleAccessor` traits
3. Update all callers (GBLinear, linear trees, SHAP)
4. Final Python bindings update

## Design Decisions

### Sparse Column Representation

For `FeatureColumn::Sparse`, we use parallel arrays rather than CSC format:

```rust
pub enum FeatureColumn {
    Dense(Box<[f32]>),
    Sparse {
        indices: Box<[u32]>,  // Sample indices with non-default values
        values: Box<[f32]>,   // Values at those indices
        default: f32,         // Value for samples not in indices
    },
}
```

**Rationale**: Simpler than CSC, matches our existing `SparseNumericStorage` pattern, and sufficient for column-wise iteration. We don't need CSC's efficient column slicing since we always iterate full columns.

### Python API Naming

The validation set parameter in Python will be named `val_set`:

```python
# Before
model.fit(train, eval_set=[EvalSet("valid", valid)])

# After  
model.fit(train, val_set=valid)
```

**Rationale**: Short, clear, consistent with common conventions. Matches Rust API naming.

### Split Decoding for EFB

When features are bundled (EFB), the grower works on bundle indices, not original feature indices. Split decoding happens in `BinnedGroup::decode_split()`:

```rust
impl BinnedGroup {
    /// Decode a bundle split to (original_feature_index, original_bin).
    pub fn decode_split(&self, bundle_bin: u32) -> (usize, u32) {
        // Binary search in bin_offsets to find which feature
    }
}
```

This replaces the old `EffectiveViews::effective_to_original()` pattern. The logic lives with the data (BinnedGroup) rather than in a separate struct.

## Alternatives Considered

### Alternative 1: Keep Current Design, Add Test Helpers

Just add a `BinnedDataset::for_testing()` constructor without changing storage.

**Rejected because:**

- Still has memory duplication
- Still has complex storage types
- Still has APIs that panic for some storage types

### Alternative 2: BinnedDataset References Dataset

Make `BinnedDataset` contain an `Option<Arc<Dataset>>` reference to its source.

**Rejected because:**

- Conflates concerns: BinnedDataset should only know about bins
- Trainer naturally holds both, no need for BinnedDataset to reference Dataset
- Simpler mental model: Dataset is for raw access, BinnedDataset is for bins

### Alternative 3: Unified Dataset with Optional Bins

Have one `Dataset` type with optional `bins` field that's populated on demand.

**Rejected because:**

- Still conflates two concerns
- "Optional bins" would need complex lifecycle management
- Doesn't solve the storage type complexity

## Open Questions

1. ~~**Eval set handling**~~ **RESOLVED: Single validation set**
   
   Changed from `&[EvalSet<'_>]` to `val_set: Option<&Dataset>`. If users need
   multiple evaluation sets, they evaluate after training. Removes EvalSet struct
   and PyEvalSet class entirely.

2. **Sparse feature handling in Dataset**
   
   The `FeatureColumn::Sparse` variant needs design for efficient iteration and gathering. Should we use CSC format? Parallel arrays? Something else?

3. ~~**Schema sharing**~~ **RESOLVED: Optional, not forced**
   
   Schema sharing is optional. `Dataset` owns its schema. `BinnedDataset` tracks
   its own internal structure (groups, bin counts, feature locations). Don't force
   a shared schema type if EFB complicates things. Each type has what it needs.

4. ~~**SampleBlocks porting**~~ **RESOLVED**
   
   `SampleBlocks` moves to `data/raw/sample_blocks.rs` and works on Dataset.
   The pattern generalizes to both dense and sparse columns.

5. ~~**io-parquet removal**~~ **RESOLVED**
   
   Remove entirely. Not used in production. Quality benchmark uses Python-based
   benchmark suite instead.

## Success Criteria

- [ ] `io-parquet` feature and `data/io/` module removed
- [ ] Module structure reorganized: `data/raw/` and `data/binned/`
- [ ] `SampleBlocks` moved from `binned/` to `raw/` module
- [ ] `Dataset` type in `data/raw/dataset.rs`
- [ ] `for_each_feature_value()` and `gather_feature_values()` on `Dataset`
- [ ] `BinnedDataset` is `pub(crate)` (internal API, not exposed to users)
- [ ] `BinnedDataset` created via `BinnedDataset::from_dataset()` only
- [ ] `BinnedDatasetBuilder` struct deleted entirely
- [ ] `data/binned/builder.rs` deleted (logic moved to from_dataset)
- [ ] `effective_` prefix removed from all BinnedDataset methods
- [ ] `original_feature` distinction removed
- [ ] `eval_sets: &[EvalSet<'_>]` changed to `val_set: Option<&Dataset>`
- [ ] `EvalSet` struct removed from public API
- [ ] `PyEvalSet` class removed from Python bindings
- [ ] `DataAccessor` and `SampleAccessor` traits deleted
- [ ] `data/types/accessor.rs` deleted
- [ ] `BinStorage` enum simplified (no raw values)
- [ ] `BinnedDataset::test_builder()` for direct construction in tests
- [ ] GBLinear works with `&Dataset` directly
- [ ] Linear tree fitting receives both `&Dataset` and `&BinnedDataset`
- [ ] All tests pass
- [ ] Python bindings updated
- [ ] `quality_benchmark.rs` updated (remove parquet, remove DataSource::Parquet)
