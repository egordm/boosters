//! Compressed Sparse Column (CSC) matrix for column-wise access.
//!
//! CSC format is optimal for coordinate descent training where we iterate
//! over features (columns) and need efficient access to all values in a column.

use super::layout::RowMajor;
use super::{DataMatrix, DenseMatrix, RowView};

/// Compressed Sparse Column matrix for efficient column-wise access.
///
/// CSC stores data by columns, making it efficient for:
/// - Iterating over all values in a column (coordinate descent)
/// - Computing column-wise statistics (gradients per feature)
///
/// # Structure
///
/// - `values`: Non-zero values, stored column by column
/// - `row_indices`: Row index for each value
/// - `col_ptrs`: Starting index in values/row_indices for each column
///
/// For column `j`, the values are `values[col_ptrs[j]..col_ptrs[j+1]]`
/// with corresponding rows `row_indices[col_ptrs[j]..col_ptrs[j+1]]`.
///
/// # Example
///
/// ```
/// use booste_rs::data::{CSCMatrix, RowMatrix};
///
/// // Create from dense matrix
/// let dense = RowMatrix::from_vec(vec![
///     1.0, 0.0, 2.0,
///     0.0, 3.0, 0.0,
///     4.0, 0.0, 5.0,
/// ], 3, 3);
///
/// let csc = CSCMatrix::from_dense(&dense);
///
/// // Iterate over column 0: values 1.0, 4.0 at rows 0, 2
/// let col0: Vec<_> = csc.column(0).collect();
/// assert_eq!(col0, vec![(0, 1.0), (2, 4.0)]);
/// ```
#[derive(Debug, Clone)]
pub struct CSCMatrix<T = f32> {
    /// Non-zero values stored column by column.
    values: Box<[T]>,
    /// Row index for each value.
    row_indices: Box<[u32]>,
    /// Column pointers: col_ptrs[j] is the start index for column j.
    /// Length is num_cols + 1, with col_ptrs[num_cols] = nnz.
    col_ptrs: Box<[u32]>,
    /// Number of rows.
    num_rows: usize,
    /// Number of columns (features).
    num_cols: usize,
}

impl<T: Copy + PartialEq + Default> CSCMatrix<T> {
    /// Create a CSC matrix from a dense matrix.
    ///
    /// Zero values (determined by `T::default()`) are not stored.
    /// For f32, this means 0.0 values are omitted.
    ///
    /// # Note
    ///
    /// NaN values ARE stored (they are not equal to default).
    pub fn from_dense<S: AsRef<[T]>>(dense: &DenseMatrix<T, RowMajor, S>) -> Self {
        Self::from_dense_with_predicate(dense, |v| v != T::default())
    }
}

impl<T: Copy> CSCMatrix<T> {
    /// Create a CSC matrix from a dense matrix with a custom inclusion predicate.
    ///
    /// Values for which `include(value)` returns `true` are stored.
    pub fn from_dense_with_predicate<S, F>(dense: &DenseMatrix<T, RowMajor, S>, include: F) -> Self
    where
        S: AsRef<[T]>,
        F: Fn(T) -> bool,
    {
        let num_rows = dense.num_rows();
        let num_cols = dense.num_features();
        let data = dense.as_slice();

        // First pass: count non-zeros per column
        let mut col_counts = vec![0u32; num_cols];
        for row in 0..num_rows {
            for col in 0..num_cols {
                let val = data[row * num_cols + col];
                if include(val) {
                    col_counts[col] += 1;
                }
            }
        }

        // Build column pointers
        let mut col_ptrs = Vec::with_capacity(num_cols + 1);
        col_ptrs.push(0u32);
        let mut cumsum = 0u32;
        for &count in &col_counts {
            cumsum += count;
            col_ptrs.push(cumsum);
        }
        let nnz = cumsum as usize;

        // Allocate storage
        let mut values = vec![unsafe { std::mem::zeroed() }; nnz];
        let mut row_indices = vec![0u32; nnz];

        // Second pass: fill values (column by column)
        let mut col_cursors = vec![0u32; num_cols];
        for col in 0..num_cols {
            col_cursors[col] = col_ptrs[col];
        }

        for row in 0..num_rows {
            for col in 0..num_cols {
                let val = data[row * num_cols + col];
                if include(val) {
                    let idx = col_cursors[col] as usize;
                    values[idx] = val;
                    row_indices[idx] = row as u32;
                    col_cursors[col] += 1;
                }
            }
        }

        Self {
            values: values.into_boxed_slice(),
            row_indices: row_indices.into_boxed_slice(),
            col_ptrs: col_ptrs.into_boxed_slice(),
            num_rows,
            num_cols,
        }
    }

    /// Create a CSC matrix that stores ALL values (including zeros).
    ///
    /// This is useful when you want column-major access to dense data
    /// without sparsity compression.
    pub fn from_dense_full<S: AsRef<[T]>>(dense: &DenseMatrix<T, RowMajor, S>) -> Self {
        Self::from_dense_with_predicate(dense, |_| true)
    }

    /// Create a CSC matrix from any `DataMatrix`.
    ///
    /// This is a generic conversion that works with any matrix type implementing
    /// `DataMatrix`. For better performance with dense matrices, use `from_dense`
    /// or `from_dense_full` directly.
    ///
    /// All values are stored (no sparsity filtering).
    pub fn from_data_matrix<M>(matrix: &M) -> Self
    where
        M: DataMatrix<Element = T>,
    {
        let num_rows = matrix.num_rows();
        let num_cols = matrix.num_features();
        let nnz = num_rows * num_cols;

        // For full storage, column pointers are uniform
        let col_ptrs: Vec<u32> = (0..=num_cols).map(|c| (c * num_rows) as u32).collect();

        // Allocate storage
        let mut values = Vec::with_capacity(nnz);
        let mut row_indices = Vec::with_capacity(nnz);

        // Fill column by column (CSC order)
        for col in 0..num_cols {
            for row in 0..num_rows {
                let val = matrix.row(row).get(col).unwrap_or_else(|| {
                    // Safety: we're iterating within bounds
                    unsafe { std::mem::zeroed() }
                });
                values.push(val);
                row_indices.push(row as u32);
            }
        }

        Self {
            values: values.into_boxed_slice(),
            row_indices: row_indices.into_boxed_slice(),
            col_ptrs: col_ptrs.into_boxed_slice(),
            num_rows,
            num_cols,
        }
    }
}

// =============================================================================
// From implementations
// =============================================================================

/// Convert from any DataMatrix to CSC (stores all values).
impl<T: Copy, M: DataMatrix<Element = T>> From<&M> for CSCMatrix<T> {
    fn from(source: &M) -> Self {
        CSCMatrix::from_data_matrix(source)
    }
}

impl<T: Copy> CSCMatrix<T> {
    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Number of columns (features).
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Number of stored (non-zero) elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Density: ratio of stored elements to total elements.
    pub fn density(&self) -> f64 {
        let total = self.num_rows * self.num_cols;
        if total == 0 {
            return 1.0;
        }
        self.nnz() as f64 / total as f64
    }

    /// Iterate over (row_index, value) pairs in a column.
    #[inline]
    pub fn column(&self, col: usize) -> ColumnIter<'_, T> {
        assert!(col < self.num_cols, "Column {} out of bounds", col);
        let start = self.col_ptrs[col] as usize;
        let end = self.col_ptrs[col + 1] as usize;
        ColumnIter {
            values: &self.values[start..end],
            row_indices: &self.row_indices[start..end],
            pos: 0,
        }
    }

    /// Get the values slice for a column.
    #[inline]
    pub fn column_values(&self, col: usize) -> &[T] {
        assert!(col < self.num_cols, "Column {} out of bounds", col);
        let start = self.col_ptrs[col] as usize;
        let end = self.col_ptrs[col + 1] as usize;
        &self.values[start..end]
    }

    /// Get the row indices slice for a column.
    #[inline]
    pub fn column_row_indices(&self, col: usize) -> &[u32] {
        assert!(col < self.num_cols, "Column {} out of bounds", col);
        let start = self.col_ptrs[col] as usize;
        let end = self.col_ptrs[col + 1] as usize;
        &self.row_indices[start..end]
    }

    /// Number of non-zeros in a specific column.
    #[inline]
    pub fn column_nnz(&self, col: usize) -> usize {
        assert!(col < self.num_cols, "Column {} out of bounds", col);
        (self.col_ptrs[col + 1] - self.col_ptrs[col]) as usize
    }

    /// Get underlying data arrays (values, row_indices, col_ptrs).
    ///
    /// Useful for low-level operations or serialization.
    pub fn into_raw_parts(self) -> (Box<[T]>, Box<[u32]>, Box<[u32]>, usize, usize) {
        (
            self.values,
            self.row_indices,
            self.col_ptrs,
            self.num_rows,
            self.num_cols,
        )
    }

    /// Create from raw parts.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - `col_ptrs.len() == num_cols + 1`
    /// - `col_ptrs[num_cols] == values.len() == row_indices.len()`
    /// - All row indices are `< num_rows`
    /// - Column pointers are monotonically non-decreasing
    pub unsafe fn from_raw_parts(
        values: Box<[T]>,
        row_indices: Box<[u32]>,
        col_ptrs: Box<[u32]>,
        num_rows: usize,
        num_cols: usize,
    ) -> Self {
        Self {
            values,
            row_indices,
            col_ptrs,
            num_rows,
            num_cols,
        }
    }
}

/// Iterator over (row_index, value) pairs in a CSC column.
#[derive(Debug, Clone)]
pub struct ColumnIter<'a, T> {
    values: &'a [T],
    row_indices: &'a [u32],
    pos: usize,
}

impl<'a, T: Copy> Iterator for ColumnIter<'a, T> {
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.values.len() {
            let row = self.row_indices[self.pos] as usize;
            let val = self.values[self.pos];
            self.pos += 1;
            Some((row, val))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.values.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<T: Copy> ExactSizeIterator for ColumnIter<'_, T> {}
impl<T: Copy> std::iter::FusedIterator for ColumnIter<'_, T> {}

// =============================================================================
// ColumnAccess implementation
// =============================================================================

impl<T: Copy> super::ColumnAccess for CSCMatrix<T> {
    type Element = T;
    type ColumnIter<'a> = ColumnIter<'a, T> where T: 'a;

    #[inline]
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    #[inline]
    fn num_columns(&self) -> usize {
        self.num_cols
    }

    #[inline]
    fn column(&self, col: usize) -> Self::ColumnIter<'_> {
        CSCMatrix::column(self, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type alias for convenience in tests
    type RowMatrix<T = f32, S = Box<[T]>> = DenseMatrix<T, RowMajor, S>;

    #[test]
    fn from_dense_basic() {
        // 3x3 matrix with some zeros
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, 2.0, // row 0
                0.0, 3.0, 0.0, // row 1
                4.0, 0.0, 5.0, // row 2
            ],
            3,
            3,
        );

        let csc = CSCMatrix::from_dense(&dense);

        assert_eq!(csc.num_rows(), 3);
        assert_eq!(csc.num_cols(), 3);
        assert_eq!(csc.nnz(), 5); // 5 non-zero values
    }

    #[test]
    fn column_iteration() {
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, 2.0, // row 0
                0.0, 3.0, 0.0, // row 1
                4.0, 0.0, 5.0, // row 2
            ],
            3,
            3,
        );

        let csc = CSCMatrix::from_dense(&dense);

        // Column 0: values 1.0, 4.0 at rows 0, 2
        let col0: Vec<_> = csc.column(0).collect();
        assert_eq!(col0, vec![(0, 1.0), (2, 4.0)]);

        // Column 1: value 3.0 at row 1
        let col1: Vec<_> = csc.column(1).collect();
        assert_eq!(col1, vec![(1, 3.0)]);

        // Column 2: values 2.0, 5.0 at rows 0, 2
        let col2: Vec<_> = csc.column(2).collect();
        assert_eq!(col2, vec![(0, 2.0), (2, 5.0)]);
    }

    #[test]
    fn column_nnz() {
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, 2.0, // row 0
                0.0, 3.0, 0.0, // row 1
                4.0, 0.0, 5.0, // row 2
            ],
            3,
            3,
        );

        let csc = CSCMatrix::from_dense(&dense);

        assert_eq!(csc.column_nnz(0), 2);
        assert_eq!(csc.column_nnz(1), 1);
        assert_eq!(csc.column_nnz(2), 2);
    }

    #[test]
    fn from_dense_full() {
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, // row 0
                0.0, 2.0, // row 1
            ],
            2,
            2,
        );

        let csc = CSCMatrix::from_dense_full(&dense);

        assert_eq!(csc.nnz(), 4); // All values stored

        // Column 0: 1.0, 0.0
        let col0: Vec<_> = csc.column(0).collect();
        assert_eq!(col0, vec![(0, 1.0), (1, 0.0)]);

        // Column 1: 0.0, 2.0
        let col1: Vec<_> = csc.column(1).collect();
        assert_eq!(col1, vec![(0, 0.0), (1, 2.0)]);
    }

    #[test]
    fn density() {
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, 2.0, // row 0
                0.0, 3.0, 0.0, // row 1
            ],
            2,
            3,
        );

        let csc = CSCMatrix::from_dense(&dense);

        // 3 non-zeros out of 6 total
        assert!((csc.density() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn empty_column() {
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, // row 0
                2.0, 0.0, // row 1
            ],
            2,
            2,
        );

        let csc = CSCMatrix::from_dense(&dense);

        // Column 1 is all zeros
        assert_eq!(csc.column_nnz(1), 0);
        let col1: Vec<_> = csc.column(1).collect();
        assert!(col1.is_empty());
    }

    #[test]
    fn handles_nan() {
        let dense = RowMatrix::from_vec(
            vec![
                1.0,
                f32::NAN, // NaN should be stored (it's not 0.0)
            ],
            1,
            2,
        );

        let csc = CSCMatrix::from_dense(&dense);

        assert_eq!(csc.nnz(), 2); // Both values stored

        let col0: Vec<_> = csc.column(0).collect();
        assert_eq!(col0.len(), 1);
        assert_eq!(col0[0].1, 1.0);

        let col1: Vec<_> = csc.column(1).collect();
        assert_eq!(col1.len(), 1);
        assert!(col1[0].1.is_nan());
    }

    #[test]
    fn column_values_and_indices() {
        let dense = RowMatrix::from_vec(
            vec![
                1.0, 0.0, // row 0
                2.0, 3.0, // row 1
            ],
            2,
            2,
        );

        let csc = CSCMatrix::from_dense(&dense);

        // Column 0
        assert_eq!(csc.column_values(0), &[1.0, 2.0]);
        assert_eq!(csc.column_row_indices(0), &[0, 1]);

        // Column 1
        assert_eq!(csc.column_values(1), &[3.0]);
        assert_eq!(csc.column_row_indices(1), &[1]);
    }

    #[test]
    fn large_matrix() {
        // 1000 rows, 100 features, ~10% density
        let mut data = vec![0.0f32; 1000 * 100];
        for i in 0..10000 {
            data[i * 10] = (i as f32) + 1.0;
        }

        let dense = RowMatrix::from_vec(data, 1000, 100);
        let csc = CSCMatrix::from_dense(&dense);

        assert_eq!(csc.num_rows(), 1000);
        assert_eq!(csc.num_cols(), 100);
        assert_eq!(csc.nnz(), 10000);
        assert!((csc.density() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn into_and_from_raw_parts() {
        let dense = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let csc = CSCMatrix::from_dense_full(&dense);

        let (values, row_indices, col_ptrs, num_rows, num_cols) = csc.into_raw_parts();

        let csc2 =
            unsafe { CSCMatrix::from_raw_parts(values, row_indices, col_ptrs, num_rows, num_cols) };

        assert_eq!(csc2.num_rows(), 2);
        assert_eq!(csc2.num_cols(), 2);

        let col0: Vec<_> = csc2.column(0).collect();
        assert_eq!(col0, vec![(0, 1.0), (1, 3.0)]);
    }
}
