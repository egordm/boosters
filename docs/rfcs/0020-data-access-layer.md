# RFC-0020: Data Access Layer

- **Status**: Draft
- **Created**: 2025-12-23
- **Updated**: 2025-12-23
- **Depends on**: RFC-0019 (Dataset Format), RFC-0004 (Quantization and Binning)
- **Scope**: Internal data access abstractions for algorithms

## Summary

This RFC defines the internal data access layer that bridges user-facing `Dataset` to algorithm implementations. It specifies view types and access patterns that algorithms use to work with data efficiently.

## Motivation

### The Problem

Different algorithms need different data views:

| Algorithm | Training Needs | Prediction Needs |
| --------- | -------------- | ---------------- |
| GBDT | Binned data (u8), column-wise access for histograms | Raw floats (f32), row-wise for tree traversal |
| GBLinear | Raw floats (f32), row-wise for dot products | Raw floats (f32), row-wise for dot products |

The data access layer provides these views without dictating a single storage format.

### Goals

| Goal | Priority | Description |
| ---- | -------- | ----------- |
| **Algorithm-agnostic views** | Must | Same Dataset works for GBDT and GBLinear |
| **Layout flexibility** | Must | Views work with row-major and column-major storage |
| **Zero overhead** | Must | Views compile away to direct array access |
| **Binned data path** | Must | Efficient u8 binned representation for GBDT |
| **Categorical awareness** | Should | Views carry feature type information |

### Non-Goals

- **I/O or persistence**: That's outside our scope entirely
- **GPU acceleration**: Future work
- **Parallel iteration**: Handled by callers with rayon

## Research

### XGBoost Approach

XGBoost uses adapter pattern with page-based access:

```cpp
// SparsePage holds a subset of rows
class SparsePage {
    HostDeviceVector<Entry> data;  // (feature_idx, value) pairs
    std::vector<bst_row_t> offset; // Row offsets
};

// Column access via ColumnMatrix
class ColumnMatrix {
    template <typename BinIdxType>
    void GetFeature(bst_feature_t fidx, auto* result);
};
```

### LightGBM Approach

LightGBM uses per-feature bin containers:

```cpp
// Each feature has its own bin container
template <typename BIN_TYPE>
class DenseBin {
    std::vector<BIN_TYPE> data_;  // One bin per sample
    
    BIN_TYPE get(data_size_t idx) { return data_[idx]; }
};
```

### Key Insight

Both libraries separate:

1. **User input** (dense/sparse, row/column major, various dtypes)
2. **Training format** (binned, typically u8 or u16)
3. **Prediction format** (original floats, row-wise access)

We should do the same.

### Sparse Feature Handling During Prediction

**Research question**: How do XGBoost/LightGBM handle sparse features during prediction?

**XGBoost approach** (always densify):

- Uses `FVec` (Feature Vector) abstraction - a dense buffer initialized with NaN
- Sparse input (CSR/CSC) is scattered into the dense buffer before traversal
- Missing features naturally become NaN (the buffer's initial value)
- Uses density threshold (12.5%) to choose between block-based and row-based prediction

**LightGBM approach** (adaptive):

- **Dense path**: Thread-local `predict_buf_` (vector of doubles)
- **Sparse path**: `std::unordered_map<int, double>` for extremely sparse data
- Decision: If `n_features > 100k AND nnz < 1%` → use map-based prediction
- Map-based: features not in map default to 0.0, handled by `MissingType::Zero`

**Recommendation for booste-rs**:

Follow XGBoost's simpler approach - always densify sparse columns into the block buffer:

1. Sparse columns are expanded when filling the transpose buffer
2. Missing (absent) values become NaN
3. Tree traversal handles NaN via `default_left` direction
4. Same buffer infrastructure handles both dense and sparse input

This is simpler than LightGBM's dual-path approach and works well for typical sparsity levels.

### Unseen Category Handling

How do XGBoost/LightGBM handle categories seen at prediction but not training?

- **XGBoost**: Maps unseen categories to the "missing" bin (bin 0 typically)
- **LightGBM**: Similar, uses a default bin for out-of-vocabulary categories

We will follow this pattern: unseen categories at prediction time map to the missing/default bin.

### Inference Layout Requirements

**Research question**: Would column-major storage hurt GBDT inference performance?

**Benchmark findings** (10,000 samples, bench_medium model):

| Strategy | Time | Throughput | vs Row-Major |
|----------|------|------------|---------------|
| **Row-major (optimized)** | 11.8 ms | 846 Kelem/s | baseline |
| **Column-major (per-row gather)** | 29.8 ms | 335 Kelem/s | **2.5x slower** |
| **Column-major (block buffer 64)** | 12.4 ms | 808 Kelem/s | **~5% slower** |
| **Column-major (block buffer 256)** | 12.5 ms | 801 Kelem/s | **~5% slower** |

**Transpose cost** (10k × 100 matrix = 1M elements):
- Full transpose: ~100 µs (negligible vs 12ms prediction)
- Block transpose (64 rows): ~1.3 ms

**Key insights**:

1. **Per-row gathering is 2.5x slower** - confirms cache locality matters
2. **Block buffering adds only ~5% overhead** - transpose cost amortizes well
3. **Viable strategy**: Store column-major, buffer to row-major in blocks for prediction

**Existing code requirements**:
- Block traversal uses `feature_buffer[row_offset..][..n_features]` (row-major stride)
- `SamplesView` asserts `is_standard_layout()` (C-order)

**Conclusion**: Column-major storage IS viable with block buffering strategy. The 5% overhead is acceptable for the usability benefit of not requiring users to manage layouts.

**Decision: Dataset is always column-major**

This provides consistency across all use cases:

- Training uses column-major directly (optimal)
- Prediction buffers to row-major in blocks (handled internally)
- Users don't need to think about layouts

**Where buffering happens**: The predictor handles block buffering. Buffer acquisition happens in `predict_into` near the parallel iteration closure, and the buffer is passed to `predict_block_into`.

**Buffer management pattern**:

```rust
impl GBDTModel {
    /// Predict with pre-allocated output buffer.
    pub fn predict_into(&self, dataset: &Dataset, output: &mut [f32]) {
        let features = dataset.features_view();
        let n_samples = dataset.n_samples();
        let n_features = dataset.n_features();
        
        // Parallel iteration over sample blocks
        output
            .par_chunks_mut(BLOCK_SIZE)
            .enumerate()
            .for_each(|(block_idx, output_block)| {
                // Acquire thread-local buffer near the closure
                TRANSPOSE_BUFFER.with(|buf| {
                    let mut buf = buf.borrow_mut();
                    let block_start = block_idx * BLOCK_SIZE;
                    let block_size = output_block.len();
                    
                    // Resize buffer if needed (amortized)
                    buf.resize(block_size * n_features, 0.0);
                    
                    // Extract and transpose this block
                    let block_features = features.slice_block(block_start, block_size);
                    transpose_block(block_features, &mut buf, n_features);
                    
                    // Pass buffer to block prediction
                    self.predict_block_into(&buf, n_features, output_block);
                });
            });
    }
    
    /// Predict a single block. Buffer is passed in, not acquired here.
    fn predict_block_into(&self, buffer: &[f32], n_features: usize, output: &mut [f32]) {
        // Buffer is already row-major: [block_size, n_features]
        for (row_idx, out) in output.iter_mut().enumerate() {
            let row_start = row_idx * n_features;
            let row = &buffer[row_start..row_start + n_features];
            *out = self.traverse_row(row);
        }
    }
}

thread_local! {
    /// Per-thread transpose buffer, lazily allocated on first use
    static TRANSPOSE_BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}
```

**Why this pattern?** 
- Buffer acquisition happens at the parallel iteration boundary (in `predict_into`)
- `predict_block_into` receives the buffer as a parameter - no hidden state
- Makes testing easier: can test `predict_block_into` with any buffer
- Clear ownership: the closure owns the buffer borrow

**Why thread-local?** Rayon's parallel iterators spawn tasks on a thread pool. Each thread needs exclusive access to its buffer. Thread-local storage provides this without locks.

**Performance notes**:

- Contiguous column-major: ~5% overhead (transpose cost amortized)
- Strided arrays: May have higher overhead due to cache misses during transpose
- Contiguous row-major: No overhead (direct use)

**Layout summary**:

| Operation | Required Layout | Notes |
| --------- | --------------- | ----- |
| Histogram building | Column-major | Feature values contiguous |
| GBDT prediction | Row-major | Buffer from column-major if needed |
| GBLinear training | Column-major | Coordinate descent |
| GBLinear prediction | Column-major | Per-feature iteration (see below) |

### GBLinear Prediction Strategy

**Benchmark results** (50k samples × 100 features, single output):

| Strategy | Time | vs Row-major |
| -------- | ---- | ------------ |
| Row-major per-sample (baseline) | 3.33 ms | 1.00x |
| **Column-major per-feature, col output** | 4.02 ms | **1.21x** |
| Column-major buffered (like GBDT) | 6.51 ms | 1.95x |
| Column-major per-feature, row output | 9.53 ms | 2.86x |
| Column-major gather | 14.85 ms | 4.46x |

**Decision**: Use per-feature iteration with column-major output layout. Only 21% overhead vs optimal row-major, and avoids any transpose/buffering.

```rust
/// Predict with column-major output: output[group * n_samples + sample]
fn predict_col_major_output(
    model: &LinearModel,
    data: &Array2<f32>, // [n_features, n_samples] feature-major
    output: &mut [f32], // [n_groups * n_samples] column-major output
) {
    let n_samples = data.ncols();
    let n_features = model.n_features();
    let n_groups = model.n_groups();

    // Initialize with bias
    for g in 0..n_groups {
        output[g * n_samples..(g + 1) * n_samples].fill(model.bias(g));
    }

    // Iterate over features - inner loop over samples is contiguous in both arrays
    for f in 0..n_features {
        let feature_samples = data.row(f); // Row in [n_features, n_samples] = all samples for feature f
        for g in 0..n_groups {
            let weight = model.weight(f, g);
            for (i, &value) in feature_samples.iter().enumerate() {
                output[g * n_samples + i] += value * weight;
            }
        }
    }
}
```

**Why this works well:**

- Feature samples are contiguous (a row in our [n_features, n_samples] storage)
- Output slice per group is contiguous
- Inner loop accesses both arrays sequentially → cache-friendly
- No transpose or buffering needed

### Category Missing Detection

A helper function encapsulates the "is this category value missing?" logic:

```rust
/// Check if a categorical value should be treated as missing.
/// 
/// Missing categories:
/// - NaN values
/// - Negative values (following XGBoost convention)
/// 
/// NOT missing:
/// - `-0.0` (IEEE: `-0.0 < 0.0` is false)
/// - Non-negative values including 0.0
#[inline]
pub fn is_category_missing(value: f32) -> bool {
    value.is_nan() || value < 0.0
}

// Edge case behavior:
// - is_category_missing(f32::NAN) => true
// - is_category_missing(-1.0) => true
// - is_category_missing(-0.0) => false (IEEE semantics)
// - is_category_missing(0.0) => false
// - is_category_missing(f32::NEG_INFINITY) => true
```

This is used in both binning and prediction:

## Design

### Core Traits

To enable extensibility (GPU, sparse, etc.), we define core access traits:

```rust
/// Core trait for feature data access.
pub trait FeatureData {
    /// Number of samples.
    fn n_samples(&self) -> usize;
    
    /// Number of features.
    fn n_features(&self) -> usize;
    
    /// Get a single value.
    fn get(&self, sample: usize, feature: usize) -> f32;
    
    /// Feature type for a given feature index.
    fn feature_type(&self, feature: usize) -> FeatureType;
}

/// Trait for dense row-oriented access.
pub trait RowAccess: FeatureData {
    /// Get a row as a contiguous slice (if layout permits).
    fn row(&self, sample: usize) -> ArrayView1<'_, f32>;
}

/// Trait for dense column-oriented access.
pub trait ColumnAccess: FeatureData {
    /// Get a column as a contiguous slice (if layout permits).
    fn column(&self, feature: usize) -> ArrayView1<'_, f32>;
}
```

### Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                           User Layer                                │
│                                                                     │
│   Dataset::new(features, targets)                                   │
│   Dataset::builder().add_feature(...).build()                       │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        View Layer (this RFC)                        │
│                                                                     │
│   FeaturesView      - Feature-major (column-major) for training     │
│   BinnedDataset     - Binned features for GBDT training             │
│   TargetsView       - Read-only access to targets                   │
│   (SamplesView)     - Internal only, used by predictor buffering    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                       Algorithm Layer                               │
│                                                                     │
│   GBDTTrainer::train(binned: &BinnedDataset, ...)                   │
│   Predictor::predict(dataset: &Dataset, ...)  // buffers internally │
│   GBLinearTrainer::train(features: FeaturesView, ...)               │
│   GBLinearModel::predict(dataset: &Dataset, ...)                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Core View Types

#### FeaturesView - Raw Feature Access

```rust
/// Read-only view into feature data.
/// 
/// Internal storage is feature-major: [n_features, n_samples].
/// This means:
/// - data.row(f) returns all samples for feature f (contiguous)
/// - data.column(s) returns all features for sample s (strided)
/// 
/// The API uses conceptual terms (sample, feature) not array terms (row, col).
#[derive(Clone, Copy)]
pub struct FeaturesView<'a> {
    /// Shape: [n_features, n_samples] - feature-major
    data: ArrayView2<'a, f32>,
    schema: &'a DatasetSchema,
}

impl<'a> FeaturesView<'a> {
    /// Number of samples (second dimension).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.data.ncols()  // samples are columns in feature-major storage
    }
    
    /// Number of features (first dimension).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.data.nrows()  // features are rows in feature-major storage
    }
    
    /// Get a single value.
    #[inline]
    pub fn get(&self, sample: usize, feature: usize) -> f32 {
        self.data[[feature, sample]]  // [feature, sample] indexing
    }
    
    /// Get all values for a feature (contiguous slice).
    /// Returns [n_samples] values.
    #[inline]
    pub fn feature(&self, feature: usize) -> ArrayView1<'a, f32> {
        self.data.row(feature)  // row in storage = feature
    }
    
    /// Get all features for a sample (strided access).
    /// Returns [n_features] values.
    /// Note: Not contiguous - use with care in hot loops.
    #[inline]
    pub fn sample(&self, sample: usize) -> ArrayView1<'a, f32> {
        self.data.column(sample)  // column in storage = sample
    }
    
    /// Feature type for a given feature index.
    #[inline]
    pub fn feature_type(&self, feature: usize) -> FeatureType {
        self.schema.features[feature].feature_type
    }
    
    /// Raw array view for bulk operations.
    #[inline]
    pub fn as_array(&self) -> ArrayView2<'a, f32> {
        self.data
    }
}
```

#### BinnedFeatures - GBDT Training Data

The existing `BinnedDataset` in `src/data/binned/dataset.rs` is a sophisticated implementation with:

- Feature groups (for bundling optimization)
- Multiple storage types (u8/u16 based on bin count)
- Row-major and column-major layouts
- Sparse storage support
- Global bin offsets for histogram allocation

**RFC-0020 does NOT propose replacing `BinnedDataset`.** Instead, we clarify how `Dataset` integrates with it:

```rust
impl Dataset {
    /// Convert to binned format for GBDT training.
    /// 
    /// This uses the existing BinnedDatasetBuilder infrastructure.
    pub fn to_binned(&self, config: &BinningConfig) -> Result<BinnedDataset, BuildError> {
        let mut builder = BinnedDatasetBuilder::new()
            .group_strategy(config.group_strategy);
        
        for (idx, meta) in self.schema.features.iter().enumerate() {
            let values = self.features.column(idx);
            match meta.feature_type {
                FeatureType::Numeric => {
                    builder = builder.add_numeric(&values, config.max_bins, meta.name.clone());
                }
                FeatureType::Categorical => {
                    // Convert float categories to i32 for existing bin_categorical()
                    let int_cats: Vec<i32> = values.iter()
                        .map(|&v| if is_category_missing(v) { -1 } else { v as i32 })
                        .collect();
                    builder = builder.add_categorical(&int_cats, meta.name.clone());
                }
            }
        }
        
        builder.build()
    }
}
```

The `BinnedFeatures` type in this RFC is a **conceptual simplification** for documentation purposes. The actual implementation uses `BinnedDataset` unchanged.

#### TargetsView - Target Access

```rust
/// Read-only view into target values.
#[derive(Clone, Copy)]
pub struct TargetsView<'a> {
    data: ArrayView2<'a, f32>,
}

impl<'a> TargetsView<'a> {
    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }
    
    /// Number of outputs.
    #[inline]
    pub fn n_outputs(&self) -> usize {
        self.data.ncols()
    }
    
    /// Get single-output targets as 1D view.
    /// Panics if n_outputs != 1.
    #[inline]
    pub fn as_single_output(&self) -> ArrayView1<'a, f32> {
        assert_eq!(self.n_outputs(), 1);
        self.data.column(0)
    }
    
    /// Get all targets as 2D view.
    #[inline]
    pub fn as_array(&self) -> ArrayView2<'a, f32> {
        self.data
    }
}
```

### Dataset Methods for View Access

```rust
impl Dataset {
    /// Get a view of the feature data.
    pub fn features_view(&self) -> FeaturesView<'_> {
        FeaturesView {
            data: self.features.as_array_view(),
            schema: &self.schema,
        }
    }
    
    /// Get a view of the target data.
    pub fn targets_view(&self) -> Option<TargetsView<'_>> {
        self.targets.as_ref().map(|t| TargetsView { 
            data: t.view() 
        })
    }
    
    /// Get sample weights.
    pub fn weights(&self) -> Option<ArrayView1<'_, f32>> {
        self.weights.as_ref().map(|w| w.view())
    }
}
```

### Algorithm Integration Patterns

#### GBDT Training

```rust
pub fn train_gbdt(
    dataset: &Dataset,
    config: &GBDTConfig,
) -> Result<GBDTModel, TrainError> {
    // 1. Convert to binned format once
    let binned = BinnedFeatures::from_dataset(dataset, config.max_bins)?;
    
    // 2. Get targets
    let targets = dataset.targets_view()
        .ok_or(TrainError::NoTargets)?;
    
    // 3. Get weights
    let weights = dataset.weights();
    
    // 4. Train using binned features
    let mut trainer = GBDTTrainer::new(&binned, &targets, weights.as_ref(), config);
    
    for round in 0..config.n_rounds {
        // Histogram building uses binned.column(feature)
        // Split finding uses binned.bin_mapper(feature)
        trainer.boost_round()?;
    }
    
    trainer.into_model()
}
```

#### GBDT Prediction

```rust
impl GBDTModel {
    pub fn predict(&self, dataset: &Dataset) -> Array2<f32> {
        let features = dataset.features_view();
        let n_samples = features.n_samples();
        let n_outputs = self.n_outputs();
        
        let mut predictions = Array2::zeros((n_samples, n_outputs));
        
        // Predict row by row
        for sample in 0..n_samples {
            let row = features.row(sample);
            for (out_idx, tree_group) in self.trees_by_output().enumerate() {
                let mut sum = 0.0;
                for tree in tree_group {
                    sum += tree.predict_row(&row);
                }
                predictions[[sample, out_idx]] = sum;
            }
        }
        
        predictions
    }
}
```

### Categorical Split Evaluation

For categorical splits during tree traversal, we need proper missing-value handling:

```rust
/// Evaluate a categorical split.
/// 
/// # Arguments
/// * `value` - Feature value from the sample
/// * `split_category` - Category ID for the split
/// * `default_left` - Direction for missing values
/// 
/// # Returns
/// * `true` if sample goes left, `false` if right
#[inline]
pub fn evaluate_categorical_split(
    value: f32, 
    split_category: i32, 
    default_left: bool,
) -> bool {
    if is_category_missing(value) {
        default_left
    } else {
        (value as i32) == split_category
    }
}
```

**Note**: Multi-category splits using bitsets are already implemented in `src/repr/gbdt/mutable_tree.rs` via `apply_categorical_split()` with `category_bitset: Vec<u32>`. The above is for single-category splits only.

#### GBLinear Training

```rust
pub fn train_gblinear(
    dataset: &Dataset,
    config: &GBLinearConfig,
) -> Result<GBLinearModel, TrainError> {
    // GBLinear needs raw floats, not binned
    let features = dataset.features_view();
    let targets = dataset.targets_view()
        .ok_or(TrainError::NoTargets)?;
    
    // Check no categoricals (or handle via one-hot)
    for f in 0..features.n_features() {
        if features.feature_type(f) == FeatureType::Categorical {
            return Err(TrainError::CategoricalNotSupported(
                "GBLinear requires one-hot encoded categoricals"
            ));
        }
    }
    
    // Coordinate descent training
    let mut trainer = GBLinearTrainer::new(&features, &targets, config);
    trainer.train()?;
    trainer.into_model()
}
```

#### GBLinear Prediction

```rust
impl GBLinearModel {
    /// Predict into a preallocated buffer (allocation-free hot path).
    pub fn predict_into(&self, features: &SamplesView<'_>, output: &mut ArrayViewMut2<f32>) {
        // Uses predict_row_into internally - see src/repr/gblinear/model.rs
        // Avoids .dot() allocation by using explicit accumulation
    }
    
    /// Convenience method that allocates output buffer.
    pub fn predict(&self, features: &SamplesView<'_>) -> Array2<f32>;
}
```

**Design note**: Avoid ndarray's `.dot()` which allocates. Use `predict_into()` pattern for allocation-free prediction.

### Layout Handling

Views abstract away the storage layout. Performance differs based on access pattern:

| Access Pattern | Row-Major Storage | Column-Major Storage |
| -------------- | ----------------- | -------------------- |
| `row(i)` | Contiguous (fast) | Strided (slower) |
| `column(j)` | Strided (slower) | Contiguous (fast) |
| `get(i, j)` | Same | Same |

For performance-critical loops:

```rust
// Histogram building - column access preferred
for feature in 0..n_features {
    let column = binned.column(feature);  // Contiguous with column-major
    for (sample, &bin) in column.iter().enumerate() {
        histogram[bin as usize] += gradients[sample];
    }
}

// Prediction - row access preferred
for sample in 0..n_samples {
    let row = features.row(sample);  // Contiguous with row-major
    predictions[sample] = tree.predict_one(&row);
}
```

The storage layout should be chosen based on the dominant access pattern:

- **Training**: Column-major for histogram building
- **Prediction**: Row-major for tree traversal

This can be configured at Dataset construction or handled via internal conversion.

## Open Questions

### 1. BinnedDataset Layout

**Decision: Column-major only for BinnedDataset**

The existing `BinnedDataset` uses column-major layout (feature values contiguous) because:
- It's only used for histogram building during training
- Histogram building iterates over samples for each feature
- Column-major gives contiguous access for this pattern

This is already the case in the current implementation.

### 2. Sparse Feature Handling in Views

How should sparse columns present in views?

1. Expand to dense on view creation
2. Separate `SparseColumn` view type
3. Iterator-based access for sparse

**Recommendation**: Start with expand-to-dense. Add sparse views later if memory becomes an issue.

### 3. Thread Safety

Should views be `Send + Sync`?

- `FeaturesView<'a>`: Yes - immutable view
- `SamplesView<'a>`: Yes - immutable view
- `BinnedDataset`: Yes - immutable after construction
- Mutable gradient/hessian arrays: Handled separately

**Decision**: All view types should be `Send + Sync`. Mutable state (gradients) lives outside views.

### 4. Parallel Mutable View Splitting

For parallel iteration over mutable data (e.g., gradients), use ndarray's built-in chunking:

```rust
use ndarray::Array1;
use crate::data::axis;
use rayon::prelude::*;

let mut gradients: Array1<f32> = Array1::zeros(n_samples);

// Split into disjoint chunks for parallel processing
gradients
    .axis_chunks_iter_mut(axis::SAMPLES, chunk_size)
    .into_par_iter()
    .for_each(|mut chunk| {
        // Each thread has exclusive access to its chunk
        for g in chunk.iter_mut() {
            *g = compute_gradient();
        }
    });
```

**Note**: Use semantic axis constants (`axis::SAMPLES`, `axis::FEATURES`) instead of raw `Axis(0)` to make intent clear and avoid layout confusion.

This provides parallel access to disjoint views on one array without data races.

## Integration

| Component | Integration Point | Notes |
| --------- | ----------------- | ----- |
| RFC-0019 (Dataset) | `Dataset::features_view()` | Creates FeaturesView |
| RFC-0004 (Binning) | `BinnedFeatures::from_dataset()` | Creates binned representation |
| RFC-0005 (Histograms) | `BinnedFeatures::column()` | Column access for histogram building |
| RFC-0006 (Split Finding) | `BinMapper` | Split threshold recovery |

## Testing Scenarios

| Scenario | Expected Behavior |
| -------- | ----------------- |
| Row-major Dataset | Views work, column access slower |
| Column-major Dataset | Views work, row access slower |
| Mixed sparse/dense | Dense view, sparse expanded |
| Multi-output targets | TargetsView has n_outputs > 1 |
| Empty dataset | Views have n_samples = 0 |

## Future Work

- [ ] Sparse-aware views for memory efficiency
- [ ] GPU views (CudaFeaturesView, etc.)
- [ ] SIMD-optimized accessors
- [ ] Out-of-core views for datasets larger than memory

## See Also

- **RFC-0019: Dataset Format** - User-facing data container
- **RFC-0004: Quantization and Binning** - Binning implementation details

## Changelog

- 2025-12-23: Initial draft
- 2025-12-23: Added core traits for extensibility (FeatureData, RowAccess, ColumnAccess)
- 2025-12-23: Added unseen category handling and is_category_missing() helper
- 2025-12-23: Clarified BinnedFeatures is conceptual; actual impl uses existing BinnedDataset
- 2025-12-23: Added categorical split evaluation function with missing-value handling
- 2025-12-23: Added inference layout research - row-major required for GBDT prediction
- 2025-12-23: Updated architecture to show SamplesView/FeaturesView distinction
- 2025-12-23: Changed predict_one to predict_row per library convention
- 2025-12-23: Removed multi-category bitset from future work (already implemented)
- 2025-12-23: Fixed GBLinear prediction to use predict_into pattern (avoid allocation)
- 2025-12-23: Added parallel view splitting via ndarray chunking
- 2025-12-24: Added benchmark data proving column-major + block buffering is viable (~5% overhead)
- 2025-12-24: Simplified GBLinear prediction example (removed inline implementation)
- 2025-12-24: Use semantic axis names (axis::SAMPLES) instead of raw Axis(0)
- 2025-12-24: Specified predictor-side buffering with pre-allocated transpose buffer
- 2025-12-24: GBLinear prediction is layout-agnostic (can use column-major directly)
- 2025-12-24: Dataset is always column-major - simplified decision
- 2025-12-24: Thread-local transpose buffers for parallel safety
- 2025-12-24: Added sparse feature handling research (XGBoost/LightGBM comparison)
- 2025-12-24: Updated architecture diagram - SamplesView marked as internal-only
- 2025-12-24: Design review complete - ready for implementation
- 2025-12-24: Refactored buffer pattern - predict_block_into receives buffer as parameter
- 2025-12-24: Buffer acquisition happens in predict_into near parallel closure
- 2025-12-24: GBLinear benchmark: per-feature with col-major output is 21% slower than row-major (best option)
- 2025-12-24: Fixed FeaturesView API to match feature-major storage
- 2025-12-24: Renamed row/column methods to feature/sample for clarity
- 2025-12-24: Final review complete - ready for implementation

