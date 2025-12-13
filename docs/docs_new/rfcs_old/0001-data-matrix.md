# RFC-0001: Data Matrix

- **Status**: Draft
- **Created**: 2024-12-04
- **Updated**: 2024-12-05
- **Depends on**: None
- **Scope**: Core data structures for training and inference input

## Summary

The data matrix is the foundation of the gradient boosting pipeline. All training data—features, labels, and sample weights—flows through these structures. This RFC establishes column-major layout as the default to optimize for the training-dominant access patterns: histogram building iterates features column-by-column, while gradients are stored one column per output.

## Overview

### Component Hierarchy

```text
Dataset
├── features: &ColMatrix<f32>     ← Feature values (col-major)
├── labels: &ColMatrix<f32>       ← Ground truth (col-major)
└── weights: Option<&ColMatrix>   ← Sample importance (col-major)

ColMatrix<T, S> = DenseMatrix<T, ColMajor, S>
RowMatrix<T, S> = DenseMatrix<T, RowMajor, S>
```

### Data Flow

```text
User Data (any format)
        │
        ▼
┌───────────────────┐
│   ColMatrix<f32>  │  ← Features: n_samples × n_features
│   (col-major)     │
└───────────────────┘
        │
        ├──► Quantizer (RFC-0008)  ──► QuantizedMatrix
        │
        └──► LinearModel (RFC-0005) ──► Predictions
```

### Why Column-Major?

Training is dominated by **column iteration**:

| Operation | Access Pattern | Col-Major Benefit |
|-----------|---------------|-------------------|
| Histogram building | Feature-by-feature | Each column contiguous |
| Gradient computation | Output-by-output | Each gradient column contiguous |
| Quantization | Feature-by-feature | Each column contiguous |
| Split finding | Per-feature histograms | See RFC-0009 |

Row-major would require strided access for every training operation.

## Components

### DenseMatrix

The core matrix type with configurable layout and storage:

```rust
/// Dense matrix with configurable layout and storage.
///
/// # Type Parameters
/// * `T` - Element type (typically f32)
/// * `L` - Memory layout (RowMajor or ColMajor)
/// * `S` - Storage type implementing AsRef<[T]>
pub struct DenseMatrix<T, L: Layout = ColMajor, S: AsRef<[T]> = Box<[T]>> {
    data: S,
    n_rows: usize,
    n_cols: usize,
    _marker: PhantomData<(T, L)>,
}

/// Memory layout marker types (zero-sized)
pub struct RowMajor;
pub struct ColMajor;

pub trait Layout {
    fn index(row: usize, col: usize, n_rows: usize, n_cols: usize) -> usize;
}
```

### Type Aliases

```rust
/// Column-major matrix (default for training)
pub type ColMatrix<T = f32, S = Box<[T]>> = DenseMatrix<T, ColMajor, S>;

/// Row-major matrix (for inference-focused use cases)
pub type RowMatrix<T = f32, S = Box<[T]>> = DenseMatrix<T, RowMajor, S>;

/// Zero-copy borrowed views
pub type ColMatrixView<'a, T = f32> = ColMatrix<T, &'a [T]>;
pub type RowMatrixView<'a, T = f32> = RowMatrix<T, &'a [T]>;
```

### Memory Layout

For a 4×3 matrix (4 rows, 3 columns):

```text
Col-Major (ColMatrix):              Row-Major (RowMatrix):
┌─────────────────────┐             ┌─────────────────────┐
│ [0,0] [1,0] [2,0] [3,0]│ col 0    │ [0,0] [0,1] [0,2] │ row 0
│ [0,1] [1,1] [2,1] [3,1]│ col 1    │ [1,0] [1,1] [1,2] │ row 1
│ [0,2] [1,2] [2,2] [3,2]│ col 2    │ [2,0] [2,1] [2,2] │ row 2
└─────────────────────┘             │ [3,0] [3,1] [3,2] │ row 3
                                    └─────────────────────┘
index = row + col * n_rows          index = col + row * n_cols
```

### Column Access (ColMatrix)

```rust
impl<T, S: AsRef<[T]>> ColMatrix<T, S> {
    /// Get column as contiguous slice. O(1).
    pub fn col_slice(&self, col: usize) -> &[T];
    
    /// Iterate over all columns as contiguous slices.
    pub fn col_slices(&self) -> impl Iterator<Item = &[T]>;
    
    /// Mutable column access (when S: AsMut<[T]>).
    pub fn col_slice_mut(&mut self, col: usize) -> &mut [T];
}
```

### Row Access (RowMatrix)

```rust
impl<T, S: AsRef<[T]>> RowMatrix<T, S> {
    /// Get row as contiguous slice. O(1).
    pub fn row_slice(&self, row: usize) -> &[T];
    
    /// Get contiguous slice of multiple rows. O(1).
    pub fn rows_slice(&self, start: usize, count: usize) -> &[T];
}
```

### Dataset

Bundles features, labels, and weights for training/evaluation:

```rust
/// Complete dataset for training/evaluation.
pub struct Dataset<'a, S: AsRef<[f32]> = Box<[f32]>> {
    pub features: &'a ColMatrix<f32, S>,
    pub labels: &'a ColMatrix<f32, S>,
    pub weights: Option<&'a ColMatrix<f32, S>>,
}

impl<'a, S: AsRef<[f32]>> Dataset<'a, S> {
    pub fn new(
        features: &'a ColMatrix<f32, S>,
        labels: &'a ColMatrix<f32, S>,
        weights: Option<&'a ColMatrix<f32, S>>,
    ) -> Self;
    
    pub fn n_samples(&self) -> usize { self.features.n_rows() }
    pub fn n_features(&self) -> usize { self.features.n_cols() }
    pub fn n_labels(&self) -> usize { self.labels.n_cols() }
}
```

### Labels vs Outputs

The number of label columns (`n_labels`) may differ from output columns (`n_outputs`):

| Task | n_labels | n_outputs | Labels Format |
|------|----------|-----------|---------------|
| Regression | 1 | 1 | Target value |
| Multi-target | k | k | k target values |
| Binary classification | 1 | 1 | 0 or 1 |
| Multiclass (K classes) | 1 | K | Class index 0..K-1 |
| Quantile (Q quantiles) | 1 or Q | Q | Target value(s) |

The **Objective** (RFC-0002) determines `n_outputs` from `n_labels`.

## Design Decisions

### DD-1: Column-Major as Default

**Context**: Training and inference have different access patterns. Training iterates features; inference iterates samples.

**Decision**: Column-major (ColMatrix) as the default.

**Rationale**:

- **Training dominates development time**: We optimize for the common case
- **Histogram building**: O(n_samples × n_features) feature-column iterations
- **Quantization**: Per-feature bin lookup (column access)
- **Gradients**: One column per output, computed and accumulated column-wise
- **Inference alternative**: For inference-heavy workloads, `RowMatrix` is available

The entire pipeline (RFC-0002 through RFC-0009) assumes column-major layout. This consistency eliminates layout conversions within the training loop.

### DD-2: Generic Storage Type

**Context**: Data may be owned, borrowed, or memory-mapped.

**Decision**: Generic `S: AsRef<[T]>` parameter for storage.

**Rationale**:

| Storage | Use Case | Zero-Copy |
|---------|----------|-----------|
| `Box<[T]>` | Owned, heap-allocated (default) | No |
| `&[T]` | Borrowed slice | Yes |
| `Vec<T>` | Construction, resizable | No |
| `Mmap` | Memory-mapped files (future) | Yes |

The generic parameter allows zero-copy views (`ColMatrixView`) for slicing without allocation, while owned data uses `Box<[T]>` for predictable memory layout.

### DD-3: Labels as ColMatrix

**Context**: Single-output is common, but multi-output (multiclass, multi-quantile) must be first-class.

**Decision**: Labels are `ColMatrix<f32, S>`, same as features.

**Rationale**:

- **Multi-output native**: Multiclass softmax has 1 label column (class indices) but K output columns
- **Consistent interface**: Features and labels share the same type
- **Single-output**: Just a matrix with `n_cols = 1`
- **No special cases**: All code paths handle multi-column labels naturally

### DD-4: Optional Weights

**Context**: Most datasets are unweighted; weighted datasets need per-sample importance.

**Decision**: `Option<&ColMatrix<f32, S>>` for weights.

**Rationale**:

- **Unweighted**: `None` avoids allocating a uniform-1 weight vector
- **Weighted**: Matrix allows future extension to per-output weights
- **Broadcast**: Weights have `n_cols = 1`, broadcast to all outputs
- **Consistent**: Same matrix type as features/labels

### DD-5: Borrowed References in Dataset

**Context**: Should Dataset own or borrow its components?

**Decision**: Dataset holds `&'a ColMatrix` references.

**Rationale**:

- **Zero-copy**: No data duplication when creating Dataset
- **Ownership clarity**: User owns the matrices, Dataset borrows them
- **Validation sets**: Multiple Datasets can reference the same feature matrix
- **Lifetime safety**: Rust's borrow checker ensures validity

## Special Values

### Missing Values

`f32::NAN` represents missing values:

```text
Feature value = NaN
       │
       ▼
Quantizer assigns dedicated "missing bin" (RFC-0008)
       │
       ▼
Split finding evaluates both left/right directions (RFC-0009)
```

No sentinel values (like -999) needed. NaN propagates correctly through the pipeline.

### Categorical Features

Categorical features are stored as `f32` indices:

```text
Category "red" = 0.0
Category "green" = 1.0
Category "blue" = 2.0
```

During histogram building, these are cast to integer bin indices. This avoids a separate categorical matrix type while enabling category-aware splits.

## Integration

| Component | How Dataset is Used |
|-----------|---------------------|
| RFC-0002 (Objectives) | `labels`, `weights` for gradient computation |
| RFC-0003 (Metrics) | `labels`, `weights` for evaluation |
| RFC-0008 (Quantization) | `features` quantized to QuantizedMatrix |
| RFC-0004 (GBLinear) | `features` for coordinate descent |
| RFC-0007 (GBTree) | `features` for prediction, `QuantizedMatrix` for training |

## Memory Layout Examples

### Single-Output Regression

```text
features: ColMatrix [1000 × 50]    = 200 KB (f32)
labels: ColMatrix [1000 × 1]       = 4 KB (f32)
weights: None

Total: ~204 KB
```

### Multiclass Classification (10 classes)

```text
features: ColMatrix [1000 × 50]    = 200 KB (f32)
labels: ColMatrix [1000 × 1]       = 4 KB (class indices)
weights: None
predictions: ColMatrix [1000 × 10] = 40 KB (one per class)
gradients: ColMatrix [1000 × 10]   = 40 KB (one per class)

Total: ~284 KB
```

## Future Work

- [ ] Sparse matrix types (CSR for row iteration, CSC for column iteration)
- [ ] Memory-mapped storage backend for out-of-core training
- [ ] Arrow/Parquet integration for data interchange
- [ ] GPU-compatible layouts (may require row-major variants)

## References

- [XGBoost DMatrix](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.DMatrix)
- [LightGBM Dataset](https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.Dataset)
