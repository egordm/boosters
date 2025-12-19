# RFC-0001: Data Matrix

**Status**: Implemented

## Summary

A unified abstraction for feature matrix access that supports multiple memory layouts (row-major, column-major), flexible storage backends, and clean integration with both inference and training pipelines.

## Motivation

GBDT systems require different access patterns at different stages:
- **Inference**: Row-by-row traversal through trees → row-major optimal
- **Training**: Column iteration for histogram building → column-major optimal

Rather than forcing one layout, we parameterize storage layout at the type level, enabling zero-overhead abstraction where the compiler monomorphizes all layout-dependent code. Missing values use `f32::NAN`, the modern standard shared with XGBoost.

## Design

### Core Trait: `DataMatrix`

The `DataMatrix` trait provides a uniform interface regardless of underlying storage:

```rust
pub trait DataMatrix {
    type Element: Copy;
    type Row<'a>: RowView<Element = Self::Element> where Self: 'a;

    fn num_rows(&self) -> usize;
    fn num_features(&self) -> usize;
    fn row(&self, i: usize) -> Self::Row<'_>;
    fn get(&self, row: usize, col: usize) -> Option<Self::Element>;
    fn is_dense(&self) -> bool;
    fn copy_row(&self, i: usize, buf: &mut [Self::Element]);
    fn has_missing(&self) -> bool where Self::Element: PartialEq;
    fn density(&self) -> f64 where Self::Element: PartialEq;
}
```

### Layout Abstraction

Layout is a sealed trait with two implementations:

```rust
pub trait Layout: sealed::Sealed + Copy + Default + 'static {
    fn index(row: usize, col: usize, num_rows: usize, num_cols: usize) -> usize;
    fn stride(num_rows: usize, num_cols: usize) -> usize;
    fn contiguous_len(num_rows: usize, num_cols: usize) -> usize;
}

pub struct RowMajor;  // index = row * num_cols + col
pub struct ColMajor;  // index = col * num_rows + row
```

### Dense Matrix

`DenseMatrix<T, L, S>` is generic over:
- `T`: Element type (default `f32`)
- `L`: Layout (default `RowMajor`)
- `S`: Storage backend implementing `AsRef<[T]>` (default `Box<[T]>`)

```rust
pub struct DenseMatrix<T = f32, L: Layout = RowMajor, S: AsRef<[T]> = Box<[T]>> {
    data: S,
    num_rows: usize,
    num_cols: usize,
    _marker: PhantomData<(T, L)>,
}
```

Layout-specific methods are available only on the appropriate type:
- `RowMajor`: `row_slice()` O(1), `col_iter()` strided
- `ColMajor`: `col_slice()` O(1), `row_iter()` strided

### Type Aliases

```rust
pub type RowMatrix<T = f32, S = Box<[T]>> = DenseMatrix<T, RowMajor, S>;
pub type ColMatrix<T = f32, S = Box<[T]>> = DenseMatrix<T, ColMajor, S>;
```

## Key Types

| Type | Description |
|------|-------------|
| `DataMatrix` | Core trait for uniform feature access |
| `RowView` | View of a single row, iterable over `(feature_idx, value)` |
| `DenseMatrix<T, L, S>` | Dense storage with configurable layout |
| `RowMajor` / `ColMajor` | Layout types determining memory order |
| `RowMatrix<T>` | Alias for `DenseMatrix<T, RowMajor>` |
| `ColMatrix<T>` | Alias for `DenseMatrix<T, ColMajor>` |
| `StridedIter<T>` | Iterator for non-contiguous dimension access |
| `Dataset` | User-facing wrapper with targets, weights, feature names |
| `BinnedDataset` | Quantized features for histogram-based training |

### Missing Value Handling

Missing values are represented as `f32::NAN`. Detection uses `x != x` (NaN self-inequality):

```rust
fn has_missing(&self) -> bool {
    self.data.as_ref().iter().any(|&x| x != x)
}

fn density(&self) -> f64 {
    let non_missing = self.data.iter().filter(|&&x| x == x).count();
    non_missing as f64 / total as f64
}
```

### Layout Conversion

Matrices can be converted between layouts in O(n):

```rust
let row_major = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let col_major: ColMatrix = row_major.to_layout();
// Or via From trait
let col_major: ColMatrix = (&row_major).into();
```

## Usage Example

```rust
use boosters::data::{DataMatrix, RowMatrix, ColMatrix, Dataset, FeatureColumn};

// Create row-major matrix (optimal for inference)
let features = RowMatrix::from_vec(vec![
    1.0, 2.0, 3.0,   // row 0
    4.0, 5.0, 6.0,   // row 1
], 2, 3);

// Access rows efficiently
assert_eq!(features.row_slice(0), &[1.0, 2.0, 3.0]);

// Convert to column-major for training
let col_features: ColMatrix = features.to_layout();
assert_eq!(col_features.col_slice(0), &[1.0, 4.0]);

// Use DataMatrix trait for generic code
fn count_missing<M: DataMatrix<Element = f32>>(m: &M) -> usize {
    (0..m.num_rows())
        .flat_map(|r| m.row(r).iter())
        .filter(|(_, v)| v.is_nan())
        .count()
}

// High-level Dataset API with feature columns
let dataset = Dataset::new(
    vec![
        FeatureColumn::Numeric { name: Some("age".into()), values: vec![25.0, 30.0] },
        FeatureColumn::Numeric { name: Some("income".into()), values: vec![50000.0, 75000.0] },
    ],
    vec![0.0, 1.0],  // targets
)?;

// Convert to binned for tree training
let binned = dataset.to_binned(256)?;
```

## Integration with Training Pipeline

1. **User creates `Dataset`** with `FeatureColumn`s (numeric or categorical)
2. **For tree training**: `dataset.to_binned(max_bins)` → `BinnedDataset`
3. **For linear training**: `dataset.for_gblinear()` → `ColMatrix`
4. **For inference**: Use `RowMatrix` with tree traversal (optimal cache locality)

The `BinnedDataset` organizes features into groups with optimal layouts per group, supporting both row-parallel histogram building (row-major groups) and column-based access (column-major groups).
