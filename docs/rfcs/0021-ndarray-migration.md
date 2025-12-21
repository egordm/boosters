# RFC-0021: ndarray Migration and Interface Simplification

- **Status**: Ready for Review
- **Created**: 2025-12-21
- **Updated**: 2025-12-21
- **Depends on**: RFC-0001 (Data Matrix), RFC-0003 (Inference Pipeline), RFC-0014 (GBLinear)
- **Scope**: Data types, interfaces, code simplification across entire crate

## Summary

Migrate from custom matrix types (`DenseMatrix`, `RowMatrix`, `ColMatrix`) to the `ndarray` crate, adopt domain-specific terminology (samples/features/groups vs rows/cols), and simplify interfaces by reducing convenience method proliferation while leveraging ndarray's built-in capabilities for chunking, parallel iteration, and views.

## Implementation Prerequisites

Before starting implementation:

1. Capture benchmark baselines: `cargo bench > baseline.txt`
2. Tag current commit: `git tag pre-ndarray-migration`
3. Create tracking issue for each phase
4. Ensure CI is green

## Motivation

### Current State Pain Points

1. **Custom Matrix Maintenance Burden**: Our `DenseMatrix<T, L, S>` with Layout trait adds ~600 lines of code duplicating ndarray functionality. The sealed trait pattern, strided iterators, and layout conversion logic are complex and error-prone.

2. **Vague Interface Contracts**: Functions accepting `slice: &[f32], rows: usize, cols: usize` don't encode whether input is row-major or column-major, leading to subtle bugs and unclear APIs.

3. **Method Proliferation**: The current codebase has many convenience variants:
   - `predict()`, `predict_row()`, `predict_weighted()`, `par_predict()`, `par_predict_weighted()`
   - Similar patterns in training, evaluation, and metrics
   - Each variant requires testing and maintenance

4. **Custom Utilities for Common Operations**: We have custom code for:
   - Chunking (`chunks_exact`, block processing)
   - Parallel/sequential dispatch (`Parallelism::maybe_par_*`)
   - Strided iteration over matrices

5. **Terminology Confusion**: "rows" and "columns" are ambiguous in ML contexts. Different components use different conventions.

### Benefits of ndarray

1. **Rich Built-in Functionality**:
   - `axis_chunks_iter()` / `axis_chunks_iter_mut()` for block processing
   - `par_azip!` and rayon integration for parallel iteration
   - Zero-copy views (`ArrayView`, `ArrayViewMut`)
   - Broadcasting operations

2. **Explicit Layout in Type System**: `Array2<f32>` with `.view().reversed_axes()` makes layout explicit

3. **Standard Ecosystem Type**: Compatible with numpy (via numpy), polars, and other Rust ML libraries

4. **Reduced Code Surface**: Delete custom matrix code, iterator implementations, and conversion logic

### Design Philosophy

- **Fewer, more powerful interfaces** over many convenience methods
- **Explicit parameters with sensible defaults** over implicit behavior
- **Domain terminology** (samples, features, groups) over generic terms (rows, cols)
- **Zero-cost where possible** - ndarray views are zero-copy

## Design

### Terminology Convention

Adopt ML-domain terminology throughout:

| Old Term | New Term | Meaning |
|----------|----------|---------|
| `num_rows` | `n_samples` | Number of training/inference samples |
| `num_cols` | `n_features` | Number of input features |
| `num_groups` | `n_groups` | Number of output groups (1=regression, K=multiclass) |
| `output_idx` | `group_idx` | Which output group |
| row-major data | sample-major | Samples contiguous: `[s0_f0, s0_f1, ..., s1_f0, ...]` |
| col-major data | feature-major | Features contiguous: `[f0_s0, f0_s1, ..., f1_s0, ...]` |

### Core Type Aliases

```rust
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

/// Semantic axis constants
pub mod axis {
    use ndarray::Axis;
    
    /// Sample axis (axis 0 in sample-major layout)
    pub const SAMPLES: Axis = Axis(0);
    /// Feature axis (axis 1 in sample-major layout)  
    pub const FEATURES: Axis = Axis(1);
    /// Group axis for multi-output arrays
    pub const GROUPS: Axis = Axis(0);
}

// Type aliases for common use cases (all C-order / row-major by default)
pub type Features<'a> = ArrayView2<'a, f32>;      // [n_samples, n_features] 
pub type FeaturesMut<'a> = ArrayViewMut2<'a, f32>;
pub type Targets<'a> = ArrayView1<'a, f32>;       // [n_samples]
pub type Weights<'a> = Option<ArrayView1<'a, f32>>;

// Output layout: [n_groups, n_samples] - groups as rows for contiguous group access
pub type Predictions<'a> = ArrayView2<'a, f32>;   
pub type PredictionsMut<'a> = ArrayViewMut2<'a, f32>;
```

### Standard Data Layouts

**Input Features**: Always `[n_samples, n_features]` in C-order (sample-major)
- Samples are contiguous rows
- ndarray default, compatible with numpy

**Predictions/Outputs**: `[n_groups, n_samples]` in C-order
- Each group's predictions are a contiguous row
- Efficient for: base score init, tree accumulation, metric computation
- Access pattern: `output.row(group_idx)` gives all samples for that group

**Gradients**: `[n_groups, n_samples]` in C-order
- Matches prediction layout
- `gradients.row(group_idx)` for histogram building per output

### Deleted Types

The following custom types are replaced by ndarray:

| Deleted Type | Replacement |
| ------------ | ----------- |
| `DenseMatrix<T, L, S>` | `Array2<T>` or views |
| `RowMatrix<T>` | `Array2<T>` (C-order default) |
| `ColMatrix<T>` | `Array2<T>` (transpose view when needed) |
| `Layout` trait | Not needed - ndarray handles internally |
| `StridedIter` | `ndarray::iter::Lanes` |
| `DenseRowView`, `DenseRowIter` | `ArrayView1` |
| `StridedRowView`, `StridedRowIter` | Not needed |
| `DenseColumnIter` | `ndarray::iter::Lanes` |

### Library Re-exports

Re-export common ndarray types for user convenience:

```rust
// crates/boosters/src/lib.rs
pub use ndarray::{
    Array1, Array2, ArrayView1, ArrayView2, 
    ArrayViewMut1, ArrayViewMut2, Axis,
};

// Users can use: boosters::Array2 instead of ndarray::Array2
```

### Simplified Trait: FeatureAccessor

Keep a minimal trait for tree traversal, now with ndarray:

```rust
/// Uniform feature access for tree traversal.
/// 
/// Returns `f32::NAN` for missing values.
pub trait FeatureAccessor {
    fn get_feature(&self, sample: usize, feature: usize) -> f32;
    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
}

// Implementation for ndarray views
impl FeatureAccessor for ArrayView2<'_, f32> {
    #[inline]
    fn get_feature(&self, sample: usize, feature: usize) -> f32 {
        self[[sample, feature]]
    }
    
    fn n_samples(&self) -> usize { self.nrows() }
    fn n_features(&self) -> usize { self.ncols() }
}

// Also for owned arrays
impl FeatureAccessor for Array2<f32> {
    #[inline]
    fn get_feature(&self, sample: usize, feature: usize) -> f32 {
        self[[sample, feature]]
    }
    
    fn n_samples(&self) -> usize { self.nrows() }
    fn n_features(&self) -> usize { self.ncols() }
}
```

**Note**: The `DataMatrix` trait is deleted. Its functionality is replaced by:

- Row access: `arr.row(i)` → `ArrayView1`
- Element access: `arr[[row, col]]`
- Missing detection: `arr.iter().any(|x| x.is_nan())`
- Density: Free function or remove (rarely used)

### Interface Consolidation

#### Before: 12 Prediction Methods

```rust
impl Predictor {
    fn predict(&self, features: &RowMatrix) -> PredictionOutput;
    fn predict_row(&self, features: &[f32]) -> Vec<f32>;
    fn predict_weighted(&self, features: &RowMatrix, weights: &[f32]) -> PredictionOutput;
    fn predict_row_weighted(&self, features: &[f32], weights: &[f32]) -> Vec<f32>;
    fn par_predict(&self, features: &RowMatrix, n_threads: usize) -> PredictionOutput;
    fn par_predict_weighted(&self, features: &RowMatrix, weights: &[f32], n_threads: usize) -> PredictionOutput;
    // ... internal methods
    fn predict_into(&self, ...);
    fn predict_block_into(&self, ...);
}
```

#### After: 3 Core Methods

```rust
impl Predictor {
    /// Predict for a single sample.
    fn predict_row_into(
        &self,
        features: &[f32],
        tree_weights: Option<&[f32]>,
        output: &mut [f32],  // length = n_groups
    );
    
    /// Batch prediction into pre-allocated output.
    /// 
    /// Features: [n_samples, n_features] C-order
    /// Output: [n_groups, n_samples] C-order
    fn predict_into(
        &self,
        features: Features<'_>,
        tree_weights: Option<&[f32]>,
        parallelism: Parallelism,
        output: PredictionsMut<'_>,
    );
    
    /// Convenience: allocates output array.
    fn predict(
        &self,
        features: Features<'_>,
        tree_weights: Option<&[f32]>,
        parallelism: Parallelism,
    ) -> Array2<f32> {
        let mut output = Array2::zeros((self.n_groups(), features.nrows()));
        self.predict_into(features, tree_weights, parallelism, output.view_mut());
        output
    }
}
```

### Parallelism Simplification

#### Before: Custom Dispatch

```rust
pub enum Parallelism { Sequential, Parallel }

impl Parallelism {
    fn maybe_par_for_each<T, I, F>(self, iter: I, f: F) { ... }
    fn maybe_par_map<T, B, I, F>(self, iter: I, f: F) -> Vec<B> { ... }
    fn maybe_par_bridge_for_each<T, I, F>(self, iter: I, f: F) { ... }
}
```

#### After: Direct Usage

```rust
use ndarray::parallel::prelude::*;

// In hot paths, use ndarray's parallel iteration directly
match parallelism {
    Parallelism::Parallel => {
        ndarray::Zip::from(output.axis_iter_mut(axis::GROUPS))
            .and(input.axis_iter(axis::SAMPLES))
            .par_for_each(|mut out, inp| { /* ... */ });
    }
    Parallelism::Sequential => {
        ndarray::Zip::from(output.axis_iter_mut(axis::GROUPS))
            .and(input.axis_iter(axis::SAMPLES))
            .for_each(|mut out, inp| { /* ... */ });
    }
}
```

### Block Processing with ndarray

#### Before: Custom Chunking

```rust
// Custom block iteration
let chunks_iter = features.axis_chunks_iter(...).zip(output.axis_chunks_iter_mut(...));
parallelism.maybe_par_bridge_for_each(chunks_iter, |(feat_chunk, out_chunk)| {
    self.predict_block_into(feat_chunk, weights, out_chunk);
});
```

#### After: ndarray Native

```rust
// ndarray's axis_chunks_iter returns ArrayView slices
let block_size = self.block_size;
let feat_chunks = features.axis_chunks_iter(axis::SAMPLES, block_size);
let out_chunks = output.axis_chunks_iter_mut(axis::GROUPS, block_size);

for (feat_block, mut out_block) in feat_chunks.zip(out_chunks) {
    self.predict_block_into(feat_block, tree_weights, out_block);
}
```

### Gradients Refactor

#### Before: Custom GradsTuple and Gradients

```rust
#[repr(C)]
pub struct GradsTuple { pub grad: f32, pub hess: f32 }

pub struct Gradients {
    data: Vec<GradsTuple>,
    n_samples: usize,
    n_outputs: usize,
}
```

#### After: Paired Arrays

```rust
/// Gradient and Hessian storage.
/// 
/// Both arrays have shape [n_groups, n_samples] for efficient per-group access.
pub struct Gradients {
    grad: Array2<f32>,  // [n_groups, n_samples]
    hess: Array2<f32>,  // [n_groups, n_samples]
}

impl Gradients {
    pub fn new(n_samples: usize, n_groups: usize) -> Self {
        Self {
            grad: Array2::zeros((n_groups, n_samples)),
            hess: Array2::zeros((n_groups, n_samples)),
        }
    }
    
    /// Get gradient slice for a group (contiguous).
    pub fn grad_for_group(&self, group: usize) -> ArrayView1<f32> {
        self.grad.row(group)
    }
    
    /// Get mutable gradient/hessian for a group.
    pub fn group_mut(&mut self, group: usize) -> (ArrayViewMut1<f32>, ArrayViewMut1<f32>) {
        (self.grad.row_mut(group), self.hess.row_mut(group))
    }
}
```

### PredictionOutput Migration

#### Before: Custom Type

```rust
pub struct PredictionOutput {
    data: Vec<f32>,
    num_rows: usize,
    num_groups: usize,
}
```

#### After: Just ndarray

```rust
// Output is now Array2<f32> with shape [n_groups, n_samples]
// Methods that returned PredictionOutput now return Array2<f32>

// For semantic wrapper if needed:
pub struct Predictions {
    pub kind: PredictionKind,
    pub data: Array2<f32>,  // [n_groups, n_samples]
}
```

### Dataset and BinnedDataset

The `Dataset` and `BinnedDataset` types remain largely unchanged but their feature access methods return ndarray views:

```rust
impl Dataset {
    /// Create from ndarray (primary constructor for numeric data).
    pub fn from_arrays(
        features: ArrayView2<f32>,
        targets: ArrayView1<f32>,
    ) -> Result<Self, DatasetError>;
    
    /// Create with sample weights.
    pub fn from_arrays_with_weights(
        features: ArrayView2<f32>,
        targets: ArrayView1<f32>,
        weights: ArrayView1<f32>,
    ) -> Result<Self, DatasetError>;
    
    /// Get features as ndarray view [n_samples, n_features].
    pub fn features(&self) -> ArrayView2<f32>;
    
    /// Get targets as ndarray view [n_samples].
    pub fn targets(&self) -> ArrayView1<f32>;
    
    /// Get weights (if present) as ndarray view.
    pub fn weights(&self) -> Option<ArrayView1<f32>>;
}
```

### Migration Path

#### Phase 1: Core Types (Breaking)

1. **Phase 1a**: Add ndarray types alongside existing
2. **Phase 1b**: Migrate internal code to use new types
3. **Phase 1c**: Update public API signatures
4. **Phase 1d**: Remove deprecated old types

#### Phase 2: Inference Pipeline

1. Consolidate `Predictor` methods to 3 core methods
2. Update traversal code to use ndarray views
3. Remove `PredictionOutput`, return `Array2<f32>`

#### Phase 3: Training Pipeline

1. Refactor `Gradients` (pending SoA vs AoS benchmark decision)
2. Update objective functions for ndarray input
3. Simplify histogram building to use ndarray slices
4. Update `GBDTTrainer` and `GBLinearTrainer`

#### Phase 4: Linear Model

1. Update `LinearModel` prediction methods
2. Consolidate to `predict_into` + convenience wrapper
3. Use ndarray for weight storage

#### Phase 5: Cleanup

1. Remove `Parallelism::maybe_par_*` methods (inline directly)
2. Remove custom slice utilities that ndarray provides
3. Update all tests
4. Update documentation

### Testing & Verification

After each significant change, run:

1. `cargo test` - All unit and integration tests must pass
2. `cargo clippy` - No new warnings allowed
3. Spot-check critical benchmarks (5% prediction, 10% training tolerance)

**Per-phase verification**:

- Phase 1: All existing tests pass with new types
- Phase 2: Inference tests pass, prediction values match baseline
- Phase 3: Training tests pass, model quality within tolerance
- Phase 4: Linear model tests pass
- Phase 5: Full test suite green, benchmarks within tolerance

### Usage Examples

#### Creating Features from Raw Data

```rust
use ndarray::{Array2, ArrayView2};

// From a Vec (takes ownership)
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let features = Array2::from_shape_vec((2, 3), data)
    .expect("Shape must match data length");
// features is now:
// [[1, 2, 3],
//  [4, 5, 6]]

// From a slice (zero-copy view)
let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
let features: ArrayView2<f32> = ArrayView2::from_shape((2, 3), &data)
    .expect("Shape must match data length");
```

#### Making Predictions

```rust
use boosters::model::GBDTModel;
use boosters::Parallelism;
use ndarray::Array2;

// Train or load model
let model: GBDTModel = /* ... */;

// Create input features [n_samples, n_features]
let features = Array2::from_shape_vec((100, 10), feature_data)?;

// Predict (sequential)
let predictions = model.predict(features.view(), None, Parallelism::Sequential);
// predictions: Array2<f32> with shape [n_groups, n_samples]

// Predict (parallel with auto thread count)
let predictions = model.predict(features.view(), None, Parallelism::Parallel);

// Access results
let class_0_scores = predictions.row(0);  // All samples for group 0
let sample_0_outputs = predictions.column(0);  // All groups for sample 0
```

#### Pre-allocated Output (Advanced)

```rust
use ndarray::Array2;
use boosters::inference::gbdt::Predictor;

let predictor = Predictor::new(&forest);
let n_samples = features.nrows();
let n_groups = predictor.n_groups();

// Pre-allocate output buffer
let mut output = Array2::zeros((n_groups, n_samples));

// Predict into buffer (no allocation)
predictor.predict_into(
    features.view(),
    None,  // no tree weights
    Parallelism::Parallel,
    output.view_mut(),
);
```

#### Working with Gradients

```rust
use boosters::training::Gradients;

// Create gradient buffer for multiclass (3 classes)
let mut gradients = Gradients::new(1000, 3);  // 1000 samples, 3 groups

// Get gradient slice for histogram building (contiguous!)
let class_0_grads = gradients.grad_for_group(0);  // ArrayView1 of length 1000

// Accumulate for histogram bin (f64 for precision)
let bin_grad_sum: f64 = class_0_grads.iter().map(|&g| g as f64).sum();
```

#### Training Workflow

```rust
use boosters::{GBDTModel, GBDTConfig, Dataset};
use boosters::training::{Objective, Metric};
use ndarray::{Array1, Array2};

// Load or create data as ndarray
let features = Array2::from_shape_vec((1000, 10), feature_data)?;
let targets = Array1::from_vec(target_data);

// Create dataset from arrays
let dataset = Dataset::from_arrays(features.view(), targets.view())?;

// Configure and train
let config = GBDTConfig::builder()
    .objective(Objective::squared_error())
    .metric(Metric::rmse())
    .n_trees(100)
    .learning_rate(0.1)
    .build()?;

let model = GBDTModel::train(&dataset, config)?;

// Predict on new data
let test_features = Array2::from_shape_vec((100, 10), test_data)?;
let predictions = model.predict(test_features.view(), None, Parallelism::Parallel);
```

### Numerical Precision Invariants

The following precision rules are maintained throughout:

| Operation | Precision | Rationale |
|-----------|-----------|-----------|
| Feature values | f32 | Standard ML precision, memory efficient |
| Predictions | f32 | Output type matches input |
| Histogram bins | f64 | Accumulation over thousands of samples |
| Split gain calculation | f64 | Avoid cancellation errors |
| Gradient sums | f64 | Same as histograms |
| Final weight/leaf update | f32 | Cast back after f64 computation |
| Softmax computation | f32 with max-subtraction | Numerical stability |
| Gradient clipping | f32 | Applied before accumulation |

**Missing values**: Represented as `f32::NAN`. The library propagates NaN through computations. Users should ensure their data has NaN only where intended as missing values.

Accumulation pattern:
```rust
// Correct: accumulate in f64
let sum: f64 = gradients.iter().map(|&g| g as f64).sum();

// Incorrect: accumulate in f32 (precision loss)
let sum: f32 = gradients.iter().sum();
```

## Design Decisions

### DD-1: Output Layout [n_groups, n_samples]

**Context**: Need to choose between `[n_samples, n_groups]` (sample-major) and `[n_groups, n_samples]` (group-major) for predictions.

**Options**:
1. `[n_samples, n_groups]` - Each sample's outputs are contiguous
2. `[n_groups, n_samples]` - Each group's outputs are contiguous

**Decision**: `[n_groups, n_samples]` because:
- Base score initialization: `output.row_mut(g).fill(base_score[g])` is contiguous
- Tree accumulation: Adding to all samples of a group is contiguous
- Metric computation: Operating on a group's values is contiguous
- Gradient layout matches (for training loop efficiency)

**Consequences**: Row access for "all outputs of sample i" requires gather, but this is rare.

### DD-2: Keep FeatureAccessor Trait

**Context**: Whether to keep a trait for feature access or use ndarray directly everywhere.

**Options**:
1. Delete trait, use `ArrayView2<f32>` everywhere
2. Keep minimal trait for BinnedAccessor use case

**Decision**: Keep minimal trait because:
- `BinnedAccessor` needs to convert bin indices to midpoint values
- Allows testing with mock data sources
- Type bounds stay simple: `impl FeatureAccessor` vs complex ndarray generics

**Consequences**: One small trait to maintain, but simplifies downstream code.

### DD-3: Gradients as Paired Arrays vs AoS

**Context**: Store gradients as `Array2<GradsTuple>` (AoS) or paired `(Array2<grad>, Array2<hess>)` (SoA).

**Options**:
1. AoS: `Array2<(f32, f32)>` or keep `Vec<GradsTuple>` with `#[repr(C)]` - grad and hess together
2. SoA: Two `Array2<f32>` arrays

**Decision**: **Pending benchmark** - currently leaning AoS because:
- Histogram building accesses `(grad, hess)` pairs together for each sample
- Current `#[repr(C)]` enables SIMD on interleaved pairs
- Sum operations need both values anyway

The SoA approach would benefit:
- Operations that need only gradients (rare in our code)
- Cleaner ndarray integration

**Action**: Benchmark histogram building with both layouts before finalizing.

**Consequences**: If AoS wins, keep current `GradsTuple` approach with ndarray just for indexing helpers.

### DD-4: Parallelism Enum vs Direct Dispatch

**Context**: Whether to keep `Parallelism` enum with helper methods or inline dispatch.

**Options**:

1. Keep enum with `maybe_par_*` helper methods
2. Keep enum, inline dispatch at call sites
3. Remove enum, always use thread count

**Decision**: Keep enum, inline dispatch because:

- Enum clearly expresses intent (`Parallelism::Sequential` vs `n_threads == 1`)
- Helper methods add indirection that prevents optimization
- ndarray's parallel iteration is idiomatic to use directly

**Consequences**: Slightly more verbose at call sites, but clearer and faster.

### DD-5: Delete DataMatrix Trait

**Context**: Whether to keep `DataMatrix` trait after ndarray migration.

**Options**:

1. Keep `DataMatrix` as compatibility layer
2. Delete trait, use ndarray types directly

**Decision**: Delete trait because:

- ndarray provides all functionality: `nrows()`, `ncols()`, indexing, iteration
- Fewer abstractions = easier to understand
- `FeatureAccessor` covers the specialized need (BinnedAccessor)

**Consequences**: 

- Breaking change for any external code using `DataMatrix`
- Code is simpler and more idiomatic
- `has_missing()` and `density()` become free functions

### DD-6: LinearModel Weight Layout

**Context**: How to store linear model weights with ndarray.

**Options**:

1. Separate bias array: `weights: Array2<f32>` [n_features, n_groups], `bias: Array1<f32>` [n_groups]
2. Combined: `weights: Array2<f32>` [n_features + 1, n_groups] where last row is bias

**Decision**: Combined layout because:

- Single allocation
- Column slice gives all parameters for one group
- Matches XGBoost's internal layout

**Layout**:

```rust
pub struct LinearModel {
    /// Shape: [n_features + 1, n_groups]
    /// weights[0..n_features, g] = feature weights for group g
    /// weights[n_features, g] = bias for group g
    weights: Array2<f32>,
    n_features: usize,  // Stored to avoid off-by-one
}
```

## Integration

| Component | Change | Notes |
|-----------|--------|-------|
| `data/` | Major: Delete DenseMatrix, add ndarray aliases | Breaking |
| `inference/gbdt/` | Medium: Simplify Predictor to 3 methods | Breaking API |
| `inference/gblinear/` | Medium: Update LinearModelPredict | Breaking API |
| `training/` | Medium: Gradients refactor, trainer updates | Internal |
| `training/gbdt/` | Low: Use ndarray views in grower/histograms | Internal |
| `training/gblinear/` | Medium: Use ndarray for column access | Internal |
| `explainability/` | Low: Update SHAP input types | Breaking API |
| `model/` | Low: Update public predict methods | Breaking API |
| `compat/` | Low: Update conversion to ndarray | Internal |

## API Changes Summary

### Removed Types

| Type | Replacement |
|------|-------------|
| `DenseMatrix<T, L, S>` | `ndarray::Array2<T>` |
| `RowMatrix<T>` | `ndarray::Array2<T>` |
| `ColMatrix<T>` | `ndarray::Array2<T>` |
| `PredictionOutput` | `ndarray::Array2<f32>` |
| `DataMatrix` trait | Use ndarray methods directly |

### Changed Signatures

```rust
// GBDTModel
// Before: fn predict(&self, features: &RowMatrix) -> PredictionOutput
// After:
fn predict(
    &self, 
    features: ArrayView2<f32>,
    tree_weights: Option<&[f32]>,
    parallelism: Parallelism,
) -> Array2<f32>;

// Dataset
// Before: fn new(columns: Vec<FeatureColumn>, targets: Vec<f32>) -> Result<Self>
// After (additional constructor):
fn from_arrays(
    features: ArrayView2<f32>,
    targets: ArrayView1<f32>,
) -> Result<Self, DatasetError>;

// TreeExplainer
// Before: fn shap_values(&self, data: &[f32], n_samples: usize, n_features: usize) -> ShapValues
// After:
fn shap_values(&self, features: ArrayView2<f32>) -> ShapValues;
```

## Migration Guide

### Basic Prediction

```rust
// ===== BEFORE =====
use boosters::data::RowMatrix;
use boosters::model::GBDTModel;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let features = RowMatrix::from_vec(data, 2, 3);
let predictions = model.predict(&features);
let first_row = predictions.row_vec(0);

// ===== AFTER =====
use boosters::model::GBDTModel;
use boosters::Parallelism;
use ndarray::Array2;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let features = Array2::from_shape_vec((2, 3), data).unwrap();
let predictions = model.predict(features.view(), None, Parallelism::Sequential);
// predictions shape: [n_groups, n_samples]
let group_0 = predictions.row(0);  // All samples for group 0
```

### Training

```rust
// ===== BEFORE =====
use boosters::data::{Dataset, FeatureColumn, RowMatrix};

let features = RowMatrix::from_vec(feature_data, n_samples, n_features);
let dataset = Dataset::from_numeric(&features, targets)?;

// ===== AFTER =====
use boosters::data::Dataset;
use ndarray::{Array1, Array2};

let features = Array2::from_shape_vec((n_samples, n_features), feature_data)?;
let targets = Array1::from_vec(target_data);
let dataset = Dataset::from_arrays(features.view(), targets.view())?;
```

### Working with Slices

```rust
// ===== BEFORE =====
let row = matrix.row_slice(i);           // &[f32]
let col_iter = matrix.col_iter(j);       // StridedIter

// ===== AFTER =====
let row = features.row(i);               // ArrayView1<f32>
let col = features.column(j);            // ArrayView1<f32>

// If you need a &[f32]:
let row_slice = row.as_slice().unwrap(); // Only works for contiguous
```

### Output Layout Change

```rust
// ===== BEFORE =====
// PredictionOutput was column-major: data[group * n_rows + row]
let value = output.get(row_idx, group_idx);

// ===== AFTER =====
// Array2 shape: [n_groups, n_samples]
let value = predictions[[group_idx, sample_idx]];

// Get all predictions for a sample (requires gather):
let sample_outputs: Vec<f32> = (0..n_groups)
    .map(|g| predictions[[g, sample_idx]])
    .collect();

// Get all predictions for a group (contiguous):
let group_outputs = predictions.row(group_idx);
```

## Open Questions

1. **Dataset FeatureColumn**: Keep `FeatureColumn` enum or move to Arrow-based input?
   - Pro Arrow: Modern, standard, handles missing values well
   - Pro FeatureColumn: Simple for basic use cases
   - **Tentative**: Keep FeatureColumn for now, Arrow as opt-in

2. **BinnedDataset internal layout**: Keep current packed storage or use ndarray?
   - Current: Highly optimized packed bins, feature groups
   - ndarray: Simpler but may lose packing optimizations
   - **Tentative**: Keep current for hot path, but provide helper to get midpoint values as ndarray

3. **SoA vs AoS for Gradients**: Requires benchmark before decision
   - Benchmark histogram building loop with both layouts
   - Measure training throughput end-to-end
   - Decision criteria: <5% regression acceptable for cleaner code

4. **Test data helpers**: How to handle test case loading?
   - Option: Keep using `RowMatrix` alias that maps to `Array2`
   - **Tentative**: Type alias for compatibility

5. **Chunking strategy**: Verify that input-only chunking (not output) is correct pattern
   - Input `[n_samples, n_features]` chunks along axis 0 → contiguous blocks
   - Output `[n_groups, n_samples]` NOT chunked; trees write to contiguous group rows

## Future Work

- [ ] Arrow RecordBatch as primary input format
- [ ] GPU acceleration via ndarray-cuda or similar
- [ ] SIMD-optimized histogram kernels using ndarray's raw pointers
- [ ] Streaming/incremental training with chunked input
- [ ] Consider `nalgebra` for small fixed-size matrices in linear leaves

## References

- [ndarray crate documentation](https://docs.rs/ndarray)
- [ndarray parallelism guide](https://docs.rs/ndarray/latest/ndarray/parallel/index.html)
- RFC-0001: Data Matrix (being superseded)
- RFC-0003: Inference Pipeline
- RFC-0014: GBLinear

## Changelog

- 2025-12-21: Initial draft
- 2025-12-21: Round 1-5 review: Added examples, precision invariants, DD-5, DD-6, migration guide
- 2025-12-21: Round 6 review (final): Added prerequisites, re-exports, finalized document
