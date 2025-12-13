# RFC-0010: Matrix Layout Abstraction

**Status**: Accepted  
**Created**: 2024-11-29  
**Accepted**: 2024-11-29

## Summary

Add column-major layout support to `DenseMatrix` for efficient column iteration.

## Motivation

GBLinear coordinate descent iterates over columns (features). Row-major layout
requires strided access (poor cache performance) or conversion to CSC.

**Current code** (`trainer.rs`) converts row-major → CSC before training:

```rust
let csc = self.to_csc(train_data);  // Full matrix copy
```

With column-major support, we can skip this conversion when users provide
column-major data, or convert row→column (simpler than row→CSC).

## Design

### Layout Trait

```rust
pub trait Layout: sealed::Sealed + Copy + Default {
    fn index(row: usize, col: usize, num_rows: usize, num_cols: usize) -> usize;
    fn stride(major_dim: usize) -> usize;  // Distance between elements in minor dim
}

pub struct RowMajor;   // Rows contiguous, col stride = 1, row stride = num_cols
pub struct ColMajor;   // Cols contiguous, row stride = 1, col stride = num_rows
```

### Generic DenseMatrix

```rust
pub struct DenseMatrix<T = f32, L: Layout = RowMajor, S: AsRef<[T]> = Box<[T]>> {
    data: S,
    num_rows: usize,
    num_cols: usize,
    _marker: PhantomData<(T, L)>,
}
```

Default is `RowMajor` for backward compatibility.

### Slice-Based Access

Layout-specific methods provide O(1) contiguous access:

```rust
impl<T, S: AsRef<[T]>> DenseMatrix<T, RowMajor, S> {
    fn row_slice(&self, row: usize) -> &[T];  // O(1) contiguous
}

impl<T, S: AsRef<[T]>> DenseMatrix<T, ColMajor, S> {
    fn col_slice(&self, col: usize) -> &[T];  // O(1) contiguous
}
```

For the non-contiguous dimension, provide strided iterators:

```rust
impl<T, S: AsRef<[T]>> DenseMatrix<T, RowMajor, S> {
    fn col_iter(&self, col: usize) -> StridedIter<'_, T>;  // Strided
}

impl<T, S: AsRef<[T]>> DenseMatrix<T, ColMajor, S> {
    fn row_iter(&self, row: usize) -> StridedIter<'_, T>;  // Strided
}
```

### DataMatrix Trait

Keep `DataMatrix` element-based (via `RowView`) for sparse matrix compatibility.
Slice access is layout-specific and lives on `DenseMatrix` directly.

Existing usage works unchanged:

- `copy_row()` — already works, used by `traverse_block()`
- `row().get(idx)` — already works, used by training prediction

### Layout Conversion

```rust
impl<T: Copy, L: Layout, S: AsRef<[T]>> DenseMatrix<T, L, S> {
    fn to_layout<L2: Layout>(&self) -> DenseMatrix<T, L2, Box<[T]>>;
}
```

O(n) copy, done once at training start.

## Usage

### GBTree Inference (unchanged)

```rust
// predictor.rs — already uses copy_row() which works with any layout
features.copy_row(row_idx, &mut feature_buffer[offset..][..num_features]);
```

### GBLinear Training (improved)

**Before**: Convert row-major to column-major

```rust
let row_matrix = RowMatrix::from_vec(data, num_rows, num_cols);
let col_matrix: ColMatrix = row_matrix.to_layout();  // Row → Col conversion
for col in selector.select(num_features) {
    let col_data = col_matrix.col_slice(col);  // O(1), contiguous
}
```

**Training API**: Users should provide `ColMatrix` directly for training:

```rust
let col_data: ColMatrix = (&row_data).into();  // Convert once
let model = trainer.train(&col_data, &labels, None, &[]);
```

## Open Questions

~~1. Should we expose `stride()` for the non-contiguous dimension?~~
**Yes** — useful for SIMD-aware code and manual optimization.

~~2. Should `DataMatrix` trait require slice access?~~
**No** — keep it element-based for sparse compatibility. Slice access is a
`DenseMatrix` feature.

## Future Optimizations

### Zero-Copy Block Access for Inference

Currently, `predictor.rs` copies rows into a contiguous buffer:

```rust
let mut feature_buffer = vec![f32::NAN; block_size * num_features];
for i in 0..block_size {
    features.copy_row(block_start + i, &mut feature_buffer[..]);
}
```

For `DenseMatrix<f32, RowMajor>`, this copy is unnecessary — the data is already
contiguous. Add a `rows_slice()` method:

```rust
impl<T, S: AsRef<[T]>> DenseMatrix<T, RowMajor, S> {
    /// Contiguous slice of multiple rows. O(1), zero-copy.
    fn rows_slice(&self, row_start: usize, row_count: usize) -> &[T];
}
```

Predictor can then use a fast path when input is row-major dense:

```rust
// Zero-copy for row-major dense
let feature_buffer: &[f32] = dense_rm.rows_slice(block_start, block_size);
```

This is an optimization, not core to the layout refactor. Implement after
Story 4 if benchmarks show the copy is a bottleneck.

## Implementation Plan

1. Add `Layout` trait and `ColMajor` marker type
2. Make `DenseMatrix` generic over `L: Layout`
3. Add layout-specific `row_slice()` / `col_slice()` impls
4. Add `to_layout()` conversion
5. Update `LinearTrainer` to use column-major
6. ~~Keep CSC for actually sparse data (future)~~ — Delayed, see RFC-0009

## Changelog

- 2024-11-29: Initial RFC accepted
- 2024-12-04: CSC support delayed, training APIs now use ColMatrix directly
