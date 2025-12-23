# RFC-0001: Data Matrix

- **Status**: Implemented
- **Created**: 2024-11-15
- **Updated**: 2025-01-21
- **Scope**: Feature matrix types and access patterns

## Summary

This RFC defines the data abstraction layer for feature matrices in boosters. The design uses `ndarray` for matrix storage with semantic wrapper types (`SamplesView`, `FeaturesView`) that make layout semantics explicit. A minimal `FeatureAccessor` trait provides uniform access for tree traversal.

## Motivation

GBDT systems require different access patterns at different stages:

- **Inference**: Row-by-row traversal through trees → sample-contiguous optimal
- **Training**: Column iteration for histogram building → feature-contiguous optimal

Rather than implementing custom matrix types, we leverage `ndarray` (the standard Rust matrix library) and provide thin semantic wrappers that clarify axis meanings.

### Design Goals

1. **Zero-copy views**: Wrapper types hold `ArrayView2`, not owned data
2. **Explicit semantics**: Type names (`SamplesView`, `FeaturesView`) make layout clear
3. **Minimal abstraction**: Only `FeatureAccessor` trait, not a full matrix trait
4. **ndarray ecosystem**: Compatible with Arrow, polars, and Python via numpy

## Design

### Memory Layout: C-Contiguous Requirement

All matrices in boosters **must be C-contiguous (row-major)**. This is enforced at construction time via `debug_assert!` checks:

```rust
// From SamplesView::from_array()
debug_assert!(view.is_standard_layout(), "Array must be in C-order");
```

**Why C-contiguous?**

| Reason | Explanation |
| ------ | ----------- |
| **Cache efficiency** | Row iteration is contiguous memory access |
| **SIMD-friendly** | Contiguous data enables vectorization |
| **NumPy compatibility** | NumPy defaults to C-order; zero-copy FFI |
| **Predictable slicing** | `as_slice()` always succeeds for contiguous data |
| **Parallel safety** | Block processing assumes contiguous row chunks |

ndarray supports both C-order (row-major) and Fortran-order (column-major), but **Fortran-order arrays will fail validation**. Use `transpose_to_c_order()` to convert if needed.

### Terminology Convention

| Term | Meaning |
| ---- | ------- |
| `n_samples` | Number of training/inference samples |
| `n_features` | Number of input features |
| `n_groups` | Number of output groups (1=regression, K=multiclass) |
| Sample-major | Samples on axis 0: `[n_samples, n_features]` |
| Feature-major | Features on axis 0: `[n_features, n_samples]` |
| C-contiguous | Row-major memory layout (standard layout) |

### Core Types

The `data` module provides these types:

```rust
use ndarray::{ArrayView2, ArrayView1};

/// Sample-major view: [n_samples, n_features]
/// Each sample's features are contiguous in memory.
pub struct SamplesView<'a>(ArrayView2<'a, f32>);

/// Feature-major view: [n_features, n_samples]
/// Each feature's values across samples are contiguous.
pub struct FeaturesView<'a>(ArrayView2<'a, f32>);

/// Uniform access trait for tree traversal.
pub trait FeatureAccessor {
    fn get_feature(&self, row: usize, feature: usize) -> f32;
    fn n_rows(&self) -> usize;
    fn n_features(&self) -> usize;
}
```

### SamplesView

Sample-major layout with shape `[n_samples, n_features]`. This is the standard layout for inference where we iterate over samples:

```rust
impl<'a> SamplesView<'a> {
    /// Create from a slice in sample-major order (zero-copy).
    /// Returns None if `data.len() != n_samples * n_features`.
    pub fn from_slice(data: &'a [f32], n_samples: usize, n_features: usize) -> Option<Self>;
    
    /// Create from an ndarray view.
    /// Debug-asserts that the array is C-contiguous.
    pub fn from_array(view: ArrayView2<'a, f32>) -> Self;
    
    /// Get a sample's features as a contiguous slice.
    pub fn sample(&self, idx: usize) -> ArrayView1<'_, f32>;
    
    /// Access (sample, feature) element.
    pub fn get(&self, sample: usize, feature: usize) -> f32;
}
```

### FeaturesView

Feature-major layout with shape `[n_features, n_samples]`. This is optimal for training operations that iterate over features (histogram building, coordinate descent):

```rust
impl<'a> FeaturesView<'a> {
    /// Create from a slice in feature-major order (zero-copy).
    pub fn from_slice(data: &'a [f32], n_samples: usize, n_features: usize) -> Option<Self>;
    
    /// Create from an ndarray view.
    pub fn from_array(view: ArrayView2<'a, f32>) -> Self;
    
    /// Get a feature's values across all samples as a contiguous slice.
    pub fn feature(&self, idx: usize) -> ArrayView1<'_, f32>;
    
    /// Access (sample, feature) element.
    pub fn get(&self, sample: usize, feature: usize) -> f32;
}
```

### FeatureAccessor Trait

The minimal trait for tree traversal. Implemented for both wrapper types and `BinnedAccessor`:

```rust
pub trait FeatureAccessor {
    /// Get feature value at (row, feature_index).
    /// Returns f32::NAN for missing values.
    fn get_feature(&self, row: usize, feature: usize) -> f32;
    
    /// Number of rows (samples).
    fn n_rows(&self) -> usize;
    
    /// Number of features.
    fn n_features(&self) -> usize;
}

// Implemented for:
impl FeatureAccessor for SamplesView<'_> { ... }
impl FeatureAccessor for FeaturesView<'_> { ... }
impl FeatureAccessor for BinnedAccessor<'_> { ... }
```

**Design note**: We intentionally do NOT implement `FeatureAccessor` for raw `Array2<f32>` because the axis semantics (samples vs features) depend on context. The wrapper types make the layout explicit.

### Layout Conversion

Use `transpose_to_c_order` to convert between layouts:

```rust
/// Transpose a 2D array and return an owned C-order Array2.
pub fn transpose_to_c_order<S>(arr: ArrayBase<S, Ix2>) -> Array2<f32>
where
    S: Data<Elem = f32>;
```

Example:

```rust
use boosters::data::{SamplesView, FeaturesView, transpose_to_c_order};
use ndarray::Array2;

// Feature-major data [n_features, n_samples]
let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();

// Transpose to sample-major [n_samples, n_features]
let samples = transpose_to_c_order(features.view());
assert_eq!(samples.shape(), &[3, 2]);
assert!(samples.is_standard_layout());
```

### Missing Value Handling

Missing values are represented as `f32::NAN`. This is the modern standard used by XGBoost and other libraries.

```rust
// Detection uses NaN self-inequality
fn has_missing(data: &[f32]) -> bool {
    data.iter().any(|&x| x.is_nan())
}

// Tree traversal handles NaN via default_left
if fvalue.is_nan() {
    if tree.default_left(node) { left } else { right }
}
```

### Prediction Layout

Predictions use shape `[n_groups, n_samples]` with each group's values contiguous. This optimizes:

- Base score initialization: `output.row_mut(g).fill(base_score[g])`
- Tree accumulation: Adding to group values is contiguous
- Metric computation: Operating on a group is contiguous

```rust
/// Initialize predictions with base scores.
pub fn init_predictions(base_scores: &[f32], n_samples: usize) -> Array2<f32> {
    let n_groups = base_scores.len();
    let mut predictions = Array2::zeros((n_groups, n_samples));
    for (group, &base_score) in base_scores.iter().enumerate() {
        predictions.row_mut(group).fill(base_score);
    }
    predictions
}
```

## Key Types Summary

| Type | Shape | Use Case |
| ---- | ----- | -------- |
| `SamplesView<'a>` | `[n_samples, n_features]` | Inference, row iteration |
| `FeaturesView<'a>` | `[n_features, n_samples]` | Training, column iteration |
| `FeatureAccessor` | trait | Uniform tree traversal |
| `Array2<f32>` predictions | `[n_groups, n_samples]` | Model output |
| `BinnedDataset` | packed bins | Histogram-based training |
| `Dataset` | high-level | User-facing with targets, weights |

## Usage Examples

### Basic Matrix Access

```rust
use boosters::data::SamplesView;
use ndarray::Array2;

// Create from ndarray (common path)
let arr = Array2::from_shape_vec((100, 10), data).unwrap();
let features = SamplesView::from_array(arr.view());

// Access a sample's features
let sample_5 = features.sample(5); // ArrayView1<f32>

// Use with tree prediction
tree.predict_into(&features, &mut predictions, Parallelism::Sequential);
```

### Feature-Major for Training

```rust
use boosters::data::FeaturesView;
use ndarray::Array2;

// Feature-major layout for GBLinear coordinate descent
let arr = Array2::from_shape_vec((10, 100), data).unwrap(); // [n_features, n_samples]
let features = FeaturesView::from_array(arr.view());

// Iterate features efficiently
for f in 0..features.n_features() {
    let values = features.feature(f); // Contiguous slice
    // ... histogram building or coordinate descent ...
}
```

### From Raw Slice (Zero-Copy)

```rust
use boosters::data::SamplesView;

// Data in sample-major order
let data: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let features = SamplesView::from_slice(data, 2, 3).unwrap();
// 2 samples × 3 features, zero-copy view
```

### Dataset for Training

```rust
use boosters::data::{Dataset, FeatureColumn};

// High-level API with metadata
let dataset = Dataset::new(
    vec![
        FeatureColumn::Numeric { name: Some("age".into()), values: vec![25.0, 30.0] },
        FeatureColumn::Numeric { name: Some("income".into()), values: vec![50000.0, 75000.0] },
    ],
    vec![0.0, 1.0],  // targets
)?;

// Convert to binned for GBDT training
let binned = BinnedDatasetBuilder::default()
    .max_bins(256)
    .build(&dataset)?;
```

## Design Decisions

### DD-1: ndarray Over Custom Types

**Context**: Should we use custom `DenseMatrix<T, L, S>` or standard `ndarray`?

**Decision**: Use `ndarray` with semantic wrappers.

**Rationale**:

- ndarray is the standard Rust matrix library
- Provides parallel iteration (`par_azip!`), views, slicing
- Compatible with numpy for Python bindings
- Reduces ~600 lines of custom matrix code
- Wrappers add semantic clarity without overhead

### DD-2: Non-Allocating API Convention (`_into` suffix)

**Context**: How should APIs handle output allocation?

**Decision**: Provide both allocating and non-allocating variants using the `_into` suffix pattern.

**Rationale**:

- Follows Rust standard library conventions (e.g., `extend_from_slice`)
- Allows callers to reuse buffers in hot paths
- Non-allocating variants take `ArrayViewMut2` or `&mut [T]` as output
- Allocating variants return `Array2<T>` or `Vec<T>`

**Example pattern**:

```rust
// Allocating (convenience)
fn predict(&self, features: ArrayView2<f32>) -> Array2<f32>;

// Non-allocating (performance)
fn predict_into(&self, features: ArrayView2<f32>, output: ArrayViewMut2<f32>);
```

This pattern applies across all prediction and transformation APIs.

### DD-3: Wrapper Types vs Raw ndarray

**Context**: Use raw `ArrayView2` everywhere, or provide wrapper types?

**Decision**: Provide `SamplesView` and `FeaturesView` wrappers.

**Rationale**:

- Type names clarify axis semantics
- Prevents accidental misuse (passing feature-major where sample-major expected)
- Wrappers are zero-cost (`#[repr(transparent)]` pattern)
- `FeatureAccessor` trait only implemented for explicit types

### DD-4: Minimal FeatureAccessor Trait

**Context**: Should we have a full `DataMatrix` trait or minimal accessor?

**Decision**: Minimal `FeatureAccessor` trait.

**Rationale**:

- Only tree traversal needs a trait (generic over data source)
- Other operations use concrete types with type-specific methods
- Fewer methods = simpler implementations
- `BinnedAccessor` only needs `get_feature`, not full matrix API

### DD-5: Prediction Layout [n_groups, n_samples]

**Context**: Should predictions be `[n_samples, n_groups]` or `[n_groups, n_samples]`?

**Decision**: `[n_groups, n_samples]` (group-major).

**Rationale**:

- Base score init is contiguous per group
- Tree accumulation is contiguous per group
- Matches gradient layout for training loop efficiency
- Metrics operate on one group at a time

### DD-6: C-Contiguous Layout Enforcement

**Context**: Should we support both C-order and Fortran-order arrays?

**Decision**: Require C-contiguous (row-major) layout for all matrices.

**Rationale**:

- Simplifies code: `as_slice()` always works on contiguous data
- Block processing in `Predictor` assumes contiguous row chunks
- NumPy uses C-order by default, enabling zero-copy Python bindings
- SIMD optimization requires predictable memory layout
- Debug assertions catch layout issues early in development

**Enforcement**: `debug_assert!(view.is_standard_layout())` in wrapper constructors. This is debug-only to avoid runtime overhead in release builds. Users must ensure their data is C-contiguous.

## Integration

| Component | Integration Point |
| --------- | ----------------- |
| RFC-0002 (Trees) | `TreeView::traverse_to_leaf<A: FeatureAccessor>` |
| RFC-0003 (Inference) | `Predictor::predict_into(features: &impl FeatureAccessor, ...)` |
| RFC-0004 (Binning) | `BinnedDataset` built from `Dataset` |
| RFC-0014 (GBLinear) | `FeaturesView` for column-wise coordinate descent |

## Non-Goals

- **Sparse matrices**: This RFC covers dense feature matrices only. Sparse data should be densified before use, or handled via feature bundling (RFC-0017).
- **GPU tensors**: ndarray is CPU-only. GPU support would require additional abstractions.

## Changelog

- 2025-01-23: Added DD-2 (non-allocating `_into` convention), renumbered DDs to DD-3 through DD-6. Clarified from_slice failure conditions. Clarified debug-only enforcement in DD-6.
- 2025-01-23: Added DD-5 documenting C-contiguous layout enforcement. Expanded layout documentation.
- 2025-01-21: Major rewrite for ndarray migration. Replaced `DenseMatrix`, `RowMatrix`, `ColMatrix` with `SamplesView`, `FeaturesView`. Replaced `DataMatrix` trait with minimal `FeatureAccessor`. Absorbed content from RFC-0021. Added Non-Goals section.
- 2024-11-15: Initial RFC with custom matrix types
