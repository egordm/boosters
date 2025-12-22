//! Dense matrix implementation with configurable layout.
//!
//! Supports both row-major and column-major storage via the [`Layout`] trait.
//! Default is row-major for backward compatibility and optimal inference performance.
//!
//! # Layouts
//!
//! Layout is a type parameter, so all layout-dependent code is monomorphized.
//! There is no runtime overhead for layout dispatch.
//!
//! - [`RowMajor`]: Rows are contiguous. `index = row * num_cols + col`
//! - [`ColMajor`]: Columns are contiguous. `index = col * num_rows + row`
//!
//! # Example
//!
//! ```
//! use boosters::data::{DenseMatrix, RowMajor, ColMajor};
//!
//! // Row-major (default): rows are contiguous
//! let rm = DenseMatrix::<f32, RowMajor>::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//! assert_eq!(rm.row_slice(0), &[1.0, 2.0]);  // O(1) contiguous access
//!
//! // Column-major: columns are contiguous
//! let cm = DenseMatrix::<f32, ColMajor>::from_vec(vec![1.0, 3.0, 2.0, 4.0], 2, 2);
//! assert_eq!(cm.col_slice(0), &[1.0, 3.0]);  // O(1) contiguous access
//! ```
//!
//! See RFC-0010 for design rationale.

use super::traits::{DataMatrix, RowView};
use std::iter::FusedIterator;
use std::marker::PhantomData;

// =============================================================================
// Layout trait and types (merged from layout.rs)
// =============================================================================

// Sealed trait pattern to prevent external implementations
mod sealed {
    pub trait Sealed {}
}

/// Matrix memory layout.
///
/// Determines how 2D indices map to linear memory offsets.
/// Sealed to prevent external implementations.
pub trait Layout: sealed::Sealed + Copy + Default + std::fmt::Debug + 'static {
    /// Convert (row, col) to linear index.
    fn index(row: usize, col: usize, num_rows: usize, num_cols: usize) -> usize;

    /// Stride between consecutive elements in the "slow" (non-contiguous) dimension.
    ///
    /// - For RowMajor: stride between consecutive rows in the same column = num_cols
    /// - For ColMajor: stride between consecutive cols in the same row = num_rows
    fn stride(num_rows: usize, num_cols: usize) -> usize;

    /// Size of the contiguous dimension.
    ///
    /// - For RowMajor: num_cols (row length)
    /// - For ColMajor: num_rows (column length)
    fn contiguous_len(num_rows: usize, num_cols: usize) -> usize;
}

/// Row-major layout: rows are stored contiguously.
///
/// Memory layout for 2x3 matrix:
/// ```text
/// Logical:     Memory:
/// [a b c]      [a b c d e f]
/// [d e f]       ^row0^ ^row1^
/// ```
///
/// - `row_slice()` is O(1), returns contiguous slice
/// - `col_iter()` is strided (stride = num_cols)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowMajor;

impl sealed::Sealed for RowMajor {}

impl Layout for RowMajor {
    #[inline]
    fn index(row: usize, col: usize, _num_rows: usize, num_cols: usize) -> usize {
        row * num_cols + col
    }

    #[inline]
    fn stride(_num_rows: usize, num_cols: usize) -> usize {
        num_cols
    }

    #[inline]
    fn contiguous_len(_num_rows: usize, num_cols: usize) -> usize {
        num_cols
    }
}

/// Column-major layout: columns are stored contiguously.
///
/// Memory layout for 2x3 matrix:
/// ```text
/// Logical:     Memory:
/// [a b c]      [a d b e c f]
/// [d e f]       ^c0 ^c1 ^c2
/// ```
///
/// - `col_slice()` is O(1), returns contiguous slice
/// - `row_iter()` is strided (stride = num_rows)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColMajor;

impl sealed::Sealed for ColMajor {}

impl Layout for ColMajor {
    #[inline]
    fn index(row: usize, col: usize, num_rows: usize, _num_cols: usize) -> usize {
        col * num_rows + row
    }

    #[inline]
    fn stride(num_rows: usize, _num_cols: usize) -> usize {
        num_rows
    }

    #[inline]
    fn contiguous_len(num_rows: usize, _num_cols: usize) -> usize {
        num_rows
    }
}

/// Iterator over elements with a fixed stride.
///
/// Used for accessing the non-contiguous dimension of a matrix
/// (columns in row-major, rows in column-major).
#[derive(Debug, Clone)]
pub struct StridedIter<'a, T> {
    data: &'a [T],
    pos: usize,
    stride: usize,
    remaining: usize,
}

impl<'a, T> StridedIter<'a, T> {
    /// Create a new strided iterator.
    ///
    /// # Arguments
    /// - `data`: The underlying data slice
    /// - `start`: Starting index in the slice
    /// - `stride`: Distance between consecutive elements
    /// - `count`: Number of elements to iterate
    #[inline]
    pub fn new(data: &'a [T], start: usize, stride: usize, count: usize) -> Self {
        Self {
            data,
            pos: start,
            stride,
            remaining: count,
        }
    }
}

impl<'a, T> Iterator for StridedIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let item = &self.data[self.pos];
        self.pos += self.stride;
        self.remaining -= 1;
        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T> ExactSizeIterator for StridedIter<'_, T> {}
impl<T> FusedIterator for StridedIter<'_, T> {}

// =============================================================================
// DenseMatrix
// =============================================================================

/// Dense matrix with configurable memory layout.
///
/// Stores all elements contiguously with layout determined by the `L` type parameter:
/// - [`RowMajor`] (default): rows are contiguous, optimal for row-based access
/// - [`ColMajor`]: columns are contiguous, optimal for column-based access
///
/// # Generic Parameters
///
/// - `T`: Element type (default `f32`)
/// - `L`: Memory layout (default [`RowMajor`])
/// - `S`: Storage type implementing `AsRef<[T]>` (default `Box<[T]>`)
///
/// The storage generic allows zero-copy views from borrowed slices,
/// memory-mapped files, or owned allocations.
///
/// # Missing Values
///
/// Missing values are represented as `f32::NAN` (or `T::NAN` for float types).
///
/// # Example
///
/// ```
/// use boosters::data::{DenseMatrix, DataMatrix, RowMajor, ColMajor, RowMatrix};
///
/// // Row-major (default): optimal for tree inference
/// let rm = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
/// assert_eq!(rm.row_slice(0), &[1.0, 2.0, 3.0]);
///
/// // Column-major: optimal for training (column iteration)
/// let cm = DenseMatrix::<f32, ColMajor>::from_vec(
///     vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3
/// );
/// assert_eq!(cm.col_slice(0), &[1.0, 4.0]);
/// ```
#[derive(Debug, Clone)]
pub struct DenseMatrix<T = f32, L: Layout = RowMajor, S: AsRef<[T]> = Box<[T]>> {
    data: S,
    n_rows: usize,
    n_cols: usize,
    _marker: PhantomData<(T, L)>,
}

// =============================================================================
// Constructors (layout-generic)
// =============================================================================

impl<T, L: Layout> DenseMatrix<T, L, Box<[T]>> {
    /// Create a dense matrix from a Vec, taking ownership.
    ///
    /// Data should be in the layout specified by `L`:
    /// - `RowMajor`: `[row0_col0, row0_col1, ..., row1_col0, ...]`
    /// - `ColMajor`: `[col0_row0, col0_row1, ..., col1_row0, ...]`
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != num_rows * num_cols`.
    pub fn from_vec(data: Vec<T>, num_rows: usize, num_cols: usize) -> Self {
        assert_eq!(
            data.len(),
            num_rows * num_cols,
            "Data length {} does not match dimensions {}x{}",
            data.len(),
            num_rows,
            num_cols
        );
        Self {
            data: data.into_boxed_slice(),
            n_rows: num_rows,
            n_cols: num_cols,
            _marker: PhantomData,
        }
    }
}

impl<T, L: Layout, S: AsRef<[T]>> DenseMatrix<T, L, S> {
    /// Create a dense matrix from storage.
    ///
    /// # Panics
    ///
    /// Panics if `storage.as_ref().len() != num_rows * num_cols`.
    pub fn new(storage: S, num_rows: usize, num_cols: usize) -> Self {
        assert_eq!(
            storage.as_ref().len(),
            num_rows * num_cols,
            "Storage length {} does not match dimensions {}x{}",
            storage.as_ref().len(),
            num_rows,
            num_cols
        );
        Self {
            data: storage,
            n_rows: num_rows,
            n_cols: num_cols,
            _marker: PhantomData,
        }
    }

    /// Get the underlying data as a slice.
    ///
    /// The ordering depends on the layout:
    /// - `RowMajor`: row 0, then row 1, etc.
    /// - `ColMajor`: col 0, then col 1, etc.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_ref()
    }

    /// Number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns (features).
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Get element at (row, col).
    ///
    /// Returns `None` if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        let idx = L::index(row, col, self.n_rows, self.n_cols);
        Some(&self.data.as_ref()[idx])
    }
}

impl<T, L: Layout, S: AsRef<[T]> + AsMut<[T]>> DenseMatrix<T, L, S> {
    /// Get mutable element at (row, col).
    ///
    /// Returns `None` if out of bounds.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        let idx = L::index(row, col, self.n_rows, self.n_cols);
        Some(&mut self.data.as_mut()[idx])
    }

    /// Get the underlying data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
}

impl<T, L: Layout> DenseMatrix<T, L, &[T]> {
    /// Create a borrowed view of a dense matrix from a slice.
    ///
    /// This allows zero-copy matrix creation from existing data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != num_rows * num_cols`.
    pub fn from_slice(data: &[T], num_rows: usize, num_cols: usize) -> DenseMatrix<T, L, &[T]> {
        assert_eq!(
            data.len(),
            num_rows * num_cols,
            "Data length {} does not match dimensions {}x{}",
            data.len(),
            num_rows,
            num_cols
        );
        DenseMatrix {
            data,
            n_rows: num_rows,
            n_cols: num_cols,
            _marker: PhantomData,
        }
    }
}

// =============================================================================
// Layout conversion
// =============================================================================

impl<T: Copy, L: Layout, S: AsRef<[T]>> DenseMatrix<T, L, S> {
    /// Convert to a different layout.
    ///
    /// This creates a new matrix with data rearranged for the target layout.
    /// O(n) where n = num_rows × num_cols.
    ///
    /// # Example
    ///
    /// ```
    /// use boosters::data::{DenseMatrix, RowMajor, ColMajor, RowMatrix};
    ///
    /// let rm = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    /// let cm: DenseMatrix<f32, ColMajor> = rm.to_layout();
    /// assert_eq!(cm.col_slice(0), &[1.0, 3.0]);
    /// ```
    pub fn to_layout<L2: Layout>(&self) -> DenseMatrix<T, L2, Box<[T]>> {
        let mut data = Vec::with_capacity(self.n_rows * self.n_cols);

        // Iterate in target layout order
        for i in 0..(self.n_rows * self.n_cols) {
            // Find (row, col) for position i in target layout
            let (row, col) = if std::any::TypeId::of::<L2>() == std::any::TypeId::of::<RowMajor>() {
                (i / self.n_cols, i % self.n_cols)
            } else {
                (i % self.n_rows, i / self.n_rows)
            };

            // Get from source using source layout indexing
            let src_idx = L::index(row, col, self.n_rows, self.n_cols);
            data.push(self.data.as_ref()[src_idx]);
        }

        DenseMatrix {
            data: data.into_boxed_slice(),
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            _marker: PhantomData,
        }
    }
}

// =============================================================================
// From implementations for layout conversion
// =============================================================================

/// Convert from RowMajor to ColMajor.
impl<T: Copy, S: AsRef<[T]>> From<&DenseMatrix<T, RowMajor, S>> for DenseMatrix<T, ColMajor, Box<[T]>> {
    fn from(source: &DenseMatrix<T, RowMajor, S>) -> Self {
        source.to_layout()
    }
}

/// Convert from ColMajor to RowMajor.
impl<T: Copy, S: AsRef<[T]>> From<&DenseMatrix<T, ColMajor, S>> for DenseMatrix<T, RowMajor, Box<[T]>> {
    fn from(source: &DenseMatrix<T, ColMajor, S>) -> Self {
        source.to_layout()
    }
}

// =============================================================================
// Row-major specific methods
// =============================================================================

impl<T, S: AsRef<[T]>> DenseMatrix<T, RowMajor, S> {
    /// Get a row as a contiguous slice. O(1).
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows`.
    #[inline]
    pub fn row_slice(&self, row: usize) -> &[T] {
        assert!(row < self.n_rows, "Row index {} out of bounds", row);
        let start = row * self.n_cols;
        let end = start + self.n_cols;
        &self.data.as_ref()[start..end]
    }

    /// Iterate over a column (strided access).
    ///
    /// This is slower than `row_slice()` due to non-contiguous memory access.
    #[inline]
    pub fn col_iter(&self, col: usize) -> StridedIter<'_, T> {
        assert!(col < self.n_cols, "Column index {} out of bounds", col);
        StridedIter::new(self.data.as_ref(), col, self.n_cols, self.n_rows)
    }

    /// Get a contiguous slice of multiple rows. O(1).
    ///
    /// Returns `num_rows × num_cols` elements starting from `row_start`.
    /// Useful for zero-copy block access in inference.
    ///
    /// # Panics
    ///
    /// Panics if `row_start + row_count > self.num_rows`.
    #[inline]
    pub fn rows_slice(&self, row_start: usize, row_count: usize) -> &[T] {
        assert!(
            row_start + row_count <= self.n_rows,
            "Row range {}..{} out of bounds for {} rows",
            row_start,
            row_start + row_count,
            self.n_rows
        );
        let start = row_start * self.n_cols;
        let len = row_count * self.n_cols;
        &self.data.as_ref()[start..start + len]
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>> DenseMatrix<T, RowMajor, S> {
    /// Get a mutable row slice. O(1).
    #[inline]
    pub fn row_slice_mut(&mut self, row: usize) -> &mut [T] {
        assert!(row < self.n_rows, "Row index {} out of bounds", row);
        let start = row * self.n_cols;
        let end = start + self.n_cols;
        &mut self.data.as_mut()[start..end]
    }
}

// =============================================================================
// Column-major specific methods
// =============================================================================

impl<T: Copy + Default> DenseMatrix<T, ColMajor, Box<[T]>> {
    /// Create a column-major matrix from any [`DataMatrix`].
    ///
    /// This is the recommended way to convert arbitrary input formats
    /// for coordinate descent training, which requires column iteration.
    ///
    /// # Example
    ///
    /// ```
    /// use boosters::data::{ColMatrix, RowMatrix};
    ///
    /// let row_major = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    /// let col_major = ColMatrix::from_data_matrix(&row_major);
    /// ```
    pub fn from_data_matrix<M: DataMatrix<Element = T>>(source: &M) -> Self {
        let num_rows = source.num_rows();
        let num_cols = source.num_features();
        let mut data = vec![T::default(); num_rows * num_cols];

        // Fill column by column for optimal cache usage
        for col in 0..num_cols {
            for row in 0..num_rows {
                let idx = col * num_rows + row;
                data[idx] = source.get(row, col).unwrap_or_default();
            }
        }

        DenseMatrix {
            data: data.into_boxed_slice(),
            n_rows: num_rows,
            n_cols: num_cols,
            _marker: PhantomData,
        }
    }
}

impl<T, S: AsRef<[T]>> DenseMatrix<T, ColMajor, S> {
    /// Get a column as a contiguous slice. O(1).
    ///
    /// # Panics
    ///
    /// Panics if `col >= num_cols`.
    #[inline]
    pub fn col_slice(&self, col: usize) -> &[T] {
        assert!(col < self.n_cols, "Column index {} out of bounds", col);
        let start = col * self.n_rows;
        let end = start + self.n_rows;
        &self.data.as_ref()[start..end]
    }

    /// Iterate over a row (strided access).
    ///
    /// This is slower than `col_slice()` due to non-contiguous memory access.
    #[inline]
    pub fn row_iter(&self, row: usize) -> StridedIter<'_, T> {
        assert!(row < self.n_rows, "Row index {} out of bounds", row);
        StridedIter::new(self.data.as_ref(), row, self.n_rows, self.n_cols)
    }
}

impl<T, S: AsRef<[T]> + AsMut<[T]>> DenseMatrix<T, ColMajor, S> {
    /// Get a mutable column slice. O(1).
    #[inline]
    pub fn col_slice_mut(&mut self, col: usize) -> &mut [T] {
        assert!(col < self.n_cols, "Column index {} out of bounds", col);
        let start = col * self.n_rows;
        let end = start + self.n_rows;
        &mut self.data.as_mut()[start..end]
    }
}

// =============================================================================
// DataMatrix implementation (row-major only for backward compatibility)
// =============================================================================

impl<T: Copy, S: AsRef<[T]>> DataMatrix for DenseMatrix<T, RowMajor, S> {
    type Element = T;
    type Row<'a>
        = DenseRowView<'a, T>
    where
        Self: 'a;

    #[inline]
    fn num_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    fn num_features(&self) -> usize {
        self.n_cols
    }

    #[inline]
    fn row(&self, i: usize) -> Self::Row<'_> {
        DenseRowView {
            data: self.row_slice(i),
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        Some(self.data.as_ref()[row * self.n_cols + col])
    }

    #[inline]
    fn is_dense(&self) -> bool {
        true
    }

    fn copy_row(&self, i: usize, buf: &mut [T]) {
        assert!(
            buf.len() >= self.n_cols,
            "Buffer too small: {} < {}",
            buf.len(),
            self.n_cols
        );
        buf[..self.n_cols].copy_from_slice(self.row_slice(i));
    }

    #[allow(clippy::eq_op)]
    fn has_missing(&self) -> bool
    where
        Self::Element: PartialEq,
    {
        self.data.as_ref().iter().any(|&x| x != x)
    }

    #[allow(clippy::eq_op)]
    fn density(&self) -> f64
    where
        Self::Element: PartialEq,
    {
        if self.n_rows == 0 || self.n_cols == 0 {
            return 1.0;
        }
        let total = self.n_rows * self.n_cols;
        let non_missing = self.data.as_ref().iter().filter(|&&x| x == x).count();
        non_missing as f64 / total as f64
    }
}

// Also implement for ColMajor (needed for training prediction)
impl<T: Copy, S: AsRef<[T]>> DataMatrix for DenseMatrix<T, ColMajor, S> {
    type Element = T;
    type Row<'a>
        = StridedRowView<'a, T>
    where
        Self: 'a;

    #[inline]
    fn num_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    fn num_features(&self) -> usize {
        self.n_cols
    }

    #[inline]
    fn row(&self, i: usize) -> Self::Row<'_> {
        assert!(i < self.n_rows, "Row index {} out of bounds", i);
        StridedRowView {
            data: self.data.as_ref(),
            start: i,
            stride: self.n_rows,
            len: self.n_cols,
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        let idx = ColMajor::index(row, col, self.n_rows, self.n_cols);
        Some(self.data.as_ref()[idx])
    }

    #[inline]
    fn is_dense(&self) -> bool {
        true
    }

    fn copy_row(&self, i: usize, buf: &mut [T]) {
        assert!(i < self.n_rows, "Row index {} out of bounds", i);
        assert!(
            buf.len() >= self.n_cols,
            "Buffer too small: {} < {}",
            buf.len(),
            self.n_cols
        );
        // Strided copy
        for (col, dst) in buf[..self.n_cols].iter_mut().enumerate() {
            let idx = ColMajor::index(i, col, self.n_rows, self.n_cols);
            *dst = self.data.as_ref()[idx];
        }
    }

    #[allow(clippy::eq_op)]
    fn has_missing(&self) -> bool
    where
        Self::Element: PartialEq,
    {
        self.data.as_ref().iter().any(|&x| x != x)
    }

    #[allow(clippy::eq_op)]
    fn density(&self) -> f64
    where
        Self::Element: PartialEq,
    {
        if self.n_rows == 0 || self.n_cols == 0 {
            return 1.0;
        }
        let total = self.n_rows * self.n_cols;
        let non_missing = self.data.as_ref().iter().filter(|&&x| x == x).count();
        non_missing as f64 / total as f64
    }
}

// =============================================================================
// FeatureAccessor implementations
// =============================================================================

use super::traits::FeatureAccessor;

impl<S: AsRef<[f32]>> FeatureAccessor for DenseMatrix<f32, RowMajor, S> {
    #[inline]
    fn get_feature(&self, row: usize, feature: usize) -> f32 {
        self.get(row, feature).copied().unwrap_or(f32::NAN)
    }

    #[inline]
    fn num_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    fn num_features(&self) -> usize {
        self.n_cols
    }
}

impl<S: AsRef<[f32]>> FeatureAccessor for DenseMatrix<f32, ColMajor, S> {
    #[inline]
    fn get_feature(&self, row: usize, feature: usize) -> f32 {
        self.get(row, feature).copied().unwrap_or(f32::NAN)
    }

    #[inline]
    fn num_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    fn num_features(&self) -> usize {
        self.n_cols
    }
}

// =============================================================================
// Row view types
// =============================================================================

/// View of a single row in a row-major dense matrix.
#[derive(Debug, Clone, Copy)]
pub struct DenseRowView<'a, T> {
    data: &'a [T],
}

impl<'a, T> DenseRowView<'a, T> {
    /// Get the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }
}

impl<T: Copy> RowView for DenseRowView<'_, T> {
    type Element = T;
    type Iter<'a>
        = DenseRowIter<'a, T>
    where
        Self: 'a;

    #[inline]
    fn nnz(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn get(&self, feature_idx: usize) -> Option<T> {
        self.data.get(feature_idx).copied()
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        DenseRowIter {
            data: self.data,
            pos: 0,
        }
    }
}

/// Iterator over (feature_index, value) pairs in a dense row.
#[derive(Debug, Clone)]
pub struct DenseRowIter<'a, T> {
    data: &'a [T],
    pos: usize,
}

impl<T: Copy> Iterator for DenseRowIter<'_, T> {
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.data.len() {
            let idx = self.pos;
            self.pos += 1;
            Some((idx, self.data[idx]))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> ExactSizeIterator for DenseRowIter<'_, T> {}
impl<T: Copy> FusedIterator for DenseRowIter<'_, T> {}

/// View of a single row in a column-major dense matrix (strided access).
#[derive(Debug, Clone, Copy)]
pub struct StridedRowView<'a, T> {
    data: &'a [T],
    start: usize,
    stride: usize,
    len: usize,
}

impl<T: Copy> RowView for StridedRowView<'_, T> {
    type Element = T;
    type Iter<'a>
        = StridedRowIter<'a, T>
    where
        Self: 'a;

    #[inline]
    fn nnz(&self) -> usize {
        self.len
    }

    #[inline]
    fn get(&self, feature_idx: usize) -> Option<T> {
        if feature_idx >= self.len {
            return None;
        }
        Some(self.data[self.start + feature_idx * self.stride])
    }

    #[inline]
    fn iter(&self) -> Self::Iter<'_> {
        StridedRowIter {
            data: self.data,
            pos: self.start,
            stride: self.stride,
            remaining: self.len,
            feature_idx: 0,
        }
    }
}

/// Iterator over (feature_index, value) pairs in a strided row.
#[derive(Debug, Clone)]
pub struct StridedRowIter<'a, T> {
    data: &'a [T],
    pos: usize,
    stride: usize,
    remaining: usize,
    feature_idx: usize,
}

impl<T: Copy> Iterator for StridedRowIter<'_, T> {
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let feature_idx = self.feature_idx;
        let value = self.data[self.pos];
        self.pos += self.stride;
        self.remaining -= 1;
        self.feature_idx += 1;
        Some((feature_idx, value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T: Copy> ExactSizeIterator for StridedRowIter<'_, T> {}
impl<T: Copy> FusedIterator for StridedRowIter<'_, T> {}

// =============================================================================
// Column iteration for ColMajor
// =============================================================================

/// Iterator over (row_index, value) pairs in a column of a ColMajor matrix.
///
/// Used for coordinate descent training where we iterate over all values
/// in a feature column.
#[derive(Debug, Clone)]
pub struct DenseColumnIter<'a, T> {
    data: &'a [T],
    row_idx: usize,
}

impl<T: Copy> Iterator for DenseColumnIter<'_, T> {
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.row_idx < self.data.len() {
            let idx = self.row_idx;
            let val = self.data[idx];
            self.row_idx += 1;
            Some((idx, val))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len() - self.row_idx;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> ExactSizeIterator for DenseColumnIter<'_, T> {}
impl<T: Copy> FusedIterator for DenseColumnIter<'_, T> {}

impl<T: Copy, S: AsRef<[T]>> DenseMatrix<T, ColMajor, S> {
    /// Number of columns (features).
    #[inline]
    pub fn num_columns(&self) -> usize {
        self.n_cols
    }

    /// Iterate over (row_index, value) pairs in the given column.
    ///
    /// This is efficient for ColMajor matrices since columns are contiguous.
    #[inline]
    pub fn column(&self, col: usize) -> DenseColumnIter<'_, T> {
        DenseColumnIter {
            data: self.col_slice(col),
            row_idx: 0,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Type alias for convenience in tests
    type RowMatrix<T = f32, S = Box<[T]>> = DenseMatrix<T, RowMajor, S>;

    // =========================================================================
    // Row-major tests (backward compatibility)
    // =========================================================================

    #[test]
    fn create_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = RowMatrix::from_vec(data, 2, 3);

        assert_eq!(matrix.n_rows(), 2);
        assert_eq!(matrix.n_cols(), 3);
    }

    #[test]
    fn create_from_slice() {
        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix: DenseMatrix<f32, RowMajor, &[f32]> = DenseMatrix::from_slice(&data, 2, 3);

        assert_eq!(matrix.n_rows(), 2);
        assert_eq!(matrix.n_cols(), 3);
    }

    #[test]
    #[should_panic(expected = "does not match dimensions")]
    fn create_wrong_size_panics() {
        let data = vec![1.0, 2.0, 3.0];
        DenseMatrix::<f32>::from_vec(data, 2, 3);
    }

    #[test]
    fn get_element() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = RowMatrix::from_vec(data, 2, 3);

        assert_eq!(matrix.get(0, 0), Some(&1.0));
        assert_eq!(matrix.get(0, 2), Some(&3.0));
        assert_eq!(matrix.get(1, 0), Some(&4.0));
        assert_eq!(matrix.get(1, 2), Some(&6.0));
        assert_eq!(matrix.get(2, 0), None);
        assert_eq!(matrix.get(0, 3), None);
    }

    #[test]
    fn row_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = RowMatrix::from_vec(data, 2, 3);

        assert_eq!(matrix.row_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.row_slice(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn rows_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = RowMatrix::from_vec(data, 2, 3);

        assert_eq!(matrix.rows_slice(0, 2), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix.rows_slice(0, 1), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.rows_slice(1, 1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn col_iter_row_major() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = RowMatrix::from_vec(data, 2, 3);

        let col0: Vec<_> = matrix.col_iter(0).copied().collect();
        let col1: Vec<_> = matrix.col_iter(1).copied().collect();
        let col2: Vec<_> = matrix.col_iter(2).copied().collect();

        assert_eq!(col0, vec![1.0, 4.0]);
        assert_eq!(col1, vec![2.0, 5.0]);
        assert_eq!(col2, vec![3.0, 6.0]);
    }

    #[test]
    fn row_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = RowMatrix::from_vec(data, 2, 3);

        let row0 = matrix.row(0);
        assert_eq!(row0.nnz(), 3);
        assert_eq!(row0.get(0), Some(1.0));
        assert_eq!(row0.get(2), Some(3.0));
        assert_eq!(row0.get(3), None);
    }

    #[test]
    fn row_iteration() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = RowMatrix::from_vec(data, 1, 3);

        let row = matrix.row(0);
        let pairs: Vec<_> = row.iter().collect();

        assert_eq!(pairs, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
    }

    #[test]
    fn copy_row() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = RowMatrix::from_vec(data, 2, 3);

        let mut buf = [0.0f32; 5];
        matrix.copy_row(1, &mut buf);

        assert_eq!(&buf[..3], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn is_dense() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = RowMatrix::from_vec(data, 1, 3);

        assert!(matrix.is_dense());
    }

    #[test]
    fn has_missing_false() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = RowMatrix::from_vec(data, 1, 3);

        assert!(!matrix.has_missing());
    }

    #[test]
    fn has_missing_true() {
        let data = vec![1.0, f32::NAN, 3.0];
        let matrix = RowMatrix::from_vec(data, 1, 3);

        assert!(matrix.has_missing());
    }

    #[test]
    fn density_no_missing() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = RowMatrix::from_vec(data, 2, 2);

        assert_eq!(matrix.density(), 1.0);
    }

    #[test]
    fn density_with_missing() {
        let data = vec![1.0, f32::NAN, 3.0, 4.0];
        let matrix = RowMatrix::from_vec(data, 2, 2);

        assert_eq!(matrix.density(), 0.75);
    }

    #[test]
    fn density_empty() {
        let data: Vec<f32> = vec![];
        let matrix = RowMatrix::from_vec(data, 0, 0);

        assert_eq!(matrix.density(), 1.0);
    }

    // =========================================================================
    // Column-major tests
    // =========================================================================

    #[test]
    fn col_major_create() {
        // 2x3 matrix in column-major: columns are contiguous
        // Logical:  [1 2 3]    Memory: [1, 4, 2, 5, 3, 6] (col0, col1, col2)
        //           [4 5 6]
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let matrix = DenseMatrix::<f32, ColMajor>::from_vec(data, 2, 3);

        assert_eq!(matrix.n_rows(), 2);
        assert_eq!(matrix.n_cols(), 3);
    }

    #[test]
    fn col_major_col_slice() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let matrix = DenseMatrix::<f32, ColMajor>::from_vec(data, 2, 3);

        assert_eq!(matrix.col_slice(0), &[1.0, 4.0]);
        assert_eq!(matrix.col_slice(1), &[2.0, 5.0]);
        assert_eq!(matrix.col_slice(2), &[3.0, 6.0]);
    }

    #[test]
    fn col_major_get() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let matrix = DenseMatrix::<f32, ColMajor>::from_vec(data, 2, 3);

        assert_eq!(matrix.get(0, 0), Some(&1.0));
        assert_eq!(matrix.get(1, 0), Some(&4.0));
        assert_eq!(matrix.get(0, 1), Some(&2.0));
        assert_eq!(matrix.get(1, 1), Some(&5.0));
        assert_eq!(matrix.get(0, 2), Some(&3.0));
        assert_eq!(matrix.get(1, 2), Some(&6.0));
    }

    #[test]
    fn col_major_row_iter() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let matrix = DenseMatrix::<f32, ColMajor>::from_vec(data, 2, 3);

        let row0: Vec<_> = matrix.row_iter(0).copied().collect();
        let row1: Vec<_> = matrix.row_iter(1).copied().collect();

        assert_eq!(row0, vec![1.0, 2.0, 3.0]);
        assert_eq!(row1, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn col_major_data_matrix() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let matrix = DenseMatrix::<f32, ColMajor>::from_vec(data, 2, 3);

        // DataMatrix trait methods
        assert_eq!(DataMatrix::num_rows(&matrix), 2);
        assert_eq!(DataMatrix::num_features(&matrix), 3);

        let row = matrix.row(0);
        assert_eq!(row.get(0), Some(1.0));
        assert_eq!(row.get(1), Some(2.0));
        assert_eq!(row.get(2), Some(3.0));
    }

    #[test]
    fn col_major_copy_row() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let matrix = DenseMatrix::<f32, ColMajor>::from_vec(data, 2, 3);

        let mut buf = [0.0f32; 3];
        matrix.copy_row(0, &mut buf);
        assert_eq!(buf, [1.0, 2.0, 3.0]);

        matrix.copy_row(1, &mut buf);
        assert_eq!(buf, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn col_major_row_view_iteration() {
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let matrix = DenseMatrix::<f32, ColMajor>::from_vec(data, 2, 3);

        let row = matrix.row(0);
        let pairs: Vec<_> = row.iter().collect();
        assert_eq!(pairs, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);

        let row1 = matrix.row(1);
        let pairs1: Vec<_> = row1.iter().collect();
        assert_eq!(pairs1, vec![(0, 4.0), (1, 5.0), (2, 6.0)]);
    }

    // =========================================================================
    // Layout conversion tests
    // =========================================================================

    #[test]
    fn convert_row_to_col_major() {
        // Row-major: [1 2 3] stored as [1, 2, 3, 4, 5, 6]
        //            [4 5 6]
        let rm = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

        let cm: DenseMatrix<f32, ColMajor> = rm.to_layout();

        // Check column slices
        assert_eq!(cm.col_slice(0), &[1.0, 4.0]);
        assert_eq!(cm.col_slice(1), &[2.0, 5.0]);
        assert_eq!(cm.col_slice(2), &[3.0, 6.0]);

        // Check element access matches original
        for row in 0..2 {
            for col in 0..3 {
                assert_eq!(rm.get(row, col), cm.get(row, col));
            }
        }
    }

    #[test]
    fn convert_col_to_row_major() {
        // Column-major
        let cm = DenseMatrix::<f32, ColMajor>::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2, 3);

        let rm: DenseMatrix<f32, RowMajor> = cm.to_layout();

        // Check row slices
        assert_eq!(rm.row_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(rm.row_slice(1), &[4.0, 5.0, 6.0]);

        // Check element access matches original
        for row in 0..2 {
            for col in 0..3 {
                assert_eq!(cm.get(row, col), rm.get(row, col));
            }
        }
    }

    #[test]
    fn roundtrip_conversion() {
        let original = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

        let cm: DenseMatrix<f32, ColMajor> = original.to_layout();
        let back: DenseMatrix<f32, RowMajor> = cm.to_layout();

        assert_eq!(original.as_slice(), back.as_slice());
    }

    // =========================================================================
    // Column iteration tests
    // =========================================================================
    
    #[test]
    fn column_colmajor() {
        // Row-major conceptual data: row0=[0, 1], row1=[2, 3], row2=[4, 5]
        // As ColMajor storage: col0=[0, 2, 4], col1=[1, 3, 5]
        let col_data = vec![0.0_f32, 2.0, 4.0, 1.0, 3.0, 5.0];
        let cm: DenseMatrix<f32, ColMajor> = DenseMatrix::from_vec(col_data, 3, 2);
        
        // Column 0 should yield (0, 0.0), (1, 2.0), (2, 4.0)
        let col0: Vec<_> = cm.column(0).collect();
        assert_eq!(col0, vec![(0, 0.0), (1, 2.0), (2, 4.0)], "ColMajor column 0 incorrect");
        
        // Column 1 should yield (0, 1.0), (1, 3.0), (2, 5.0)
        let col1: Vec<_> = cm.column(1).collect();
        assert_eq!(col1, vec![(0, 1.0), (1, 3.0), (2, 5.0)], "ColMajor column 1 incorrect");
    }
    
    #[test]
    fn rowmajor_to_colmajor_column() {
        // Data stored in row-major order: row0=[0, 1], row1=[2, 3], row2=[4, 5]
        let row_data = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let rm = RowMatrix::from_vec(row_data.clone(), 3, 2);
        
        // Convert to ColMajor to use column iteration
        let cm: DenseMatrix<f32, ColMajor> = (&rm).into();
        
        // Column 0 should be [0, 2, 4] (first column)
        let col0: Vec<_> = cm.column(0).collect();
        assert_eq!(col0, vec![(0, 0.0), (1, 2.0), (2, 4.0)], "Column 0 after conversion incorrect");
        
        // Column 1 should be [1, 3, 5] (second column)
        let col1: Vec<_> = cm.column(1).collect();
        assert_eq!(col1, vec![(0, 1.0), (1, 3.0), (2, 5.0)], "Column 1 after conversion incorrect");
    }
    
    /// Demonstrates that raw data layout matters: if you create a ColMajor matrix
    /// with row-major ordered data (without conversion), the columns will be wrong.
    /// This test documents the expected (incorrect) behavior to prevent confusion.
    #[test]
    fn colmajor_with_rowmajor_data_gives_wrong_columns() {
        // Row-major data: row0=[0, 1], row1=[2, 3], row2=[4, 5] -> [0, 1, 2, 3, 4, 5]
        // If this data is used directly as ColMajor storage:
        //   col0=[0, 1, 2], col1=[3, 4, 5]  <- This is what ColMajor sees
        // But conceptually the data represents:
        //   col0=[0, 2, 4], col1=[1, 3, 5]  <- What we actually want
        //
        // The fix is to use layout conversion: (&row_matrix).into() or .to_layout()
        
        let row_data = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Create ColMajor matrix directly from row-major ordered data (no conversion)
        let wrong_cm = DenseMatrix::<f32, ColMajor>::new(row_data.into_boxed_slice(), 3, 2);
        
        let col0: Vec<_> = wrong_cm.column(0).collect();
        let col1: Vec<_> = wrong_cm.column(1).collect();
        
        // ColMajor interprets contiguous chunks as columns, so we get wrong values
        assert_eq!(col0, vec![(0, 0.0), (1, 1.0), (2, 2.0)]);
        assert_eq!(col1, vec![(0, 3.0), (1, 4.0), (2, 5.0)]);
    }

    // =========================================================================
    // Layout tests (merged from layout.rs)
    // =========================================================================

    #[test]
    fn row_major_indexing() {
        // 2x3 matrix
        assert_eq!(RowMajor::index(0, 0, 2, 3), 0);
        assert_eq!(RowMajor::index(0, 1, 2, 3), 1);
        assert_eq!(RowMajor::index(0, 2, 2, 3), 2);
        assert_eq!(RowMajor::index(1, 0, 2, 3), 3);
        assert_eq!(RowMajor::index(1, 1, 2, 3), 4);
        assert_eq!(RowMajor::index(1, 2, 2, 3), 5);
    }

    #[test]
    fn col_major_indexing() {
        // 2x3 matrix
        assert_eq!(ColMajor::index(0, 0, 2, 3), 0);
        assert_eq!(ColMajor::index(1, 0, 2, 3), 1);
        assert_eq!(ColMajor::index(0, 1, 2, 3), 2);
        assert_eq!(ColMajor::index(1, 1, 2, 3), 3);
        assert_eq!(ColMajor::index(0, 2, 2, 3), 4);
        assert_eq!(ColMajor::index(1, 2, 2, 3), 5);
    }

    #[test]
    fn row_major_stride() {
        assert_eq!(RowMajor::stride(2, 3), 3); // stride = num_cols
        assert_eq!(RowMajor::contiguous_len(2, 3), 3); // row length = num_cols
    }

    #[test]
    fn col_major_stride() {
        assert_eq!(ColMajor::stride(2, 3), 2); // stride = num_rows
        assert_eq!(ColMajor::contiguous_len(2, 3), 2); // col length = num_rows
    }

    #[test]
    fn strided_iter() {
        let data = [0, 1, 2, 3, 4, 5];
        // Iterate column 1 of a 2x3 row-major matrix: indices 1, 4
        let iter = StridedIter::new(&data, 1, 3, 2);
        let values: Vec<_> = iter.copied().collect();
        assert_eq!(values, vec![1, 4]);
    }

    #[test]
    fn strided_iter_exact_size() {
        let data = [0, 1, 2, 3, 4, 5];
        let iter = StridedIter::new(&data, 0, 2, 3);
        assert_eq!(iter.len(), 3);
    }
}
