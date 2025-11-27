//! Dense row-major matrix implementation.

use super::traits::{DataMatrix, RowView};
use std::iter::FusedIterator;

/// Row-major dense matrix for feature storage.
///
/// Stores all elements contiguously in row-major order. This is the most
/// common format for tabular data and provides optimal cache locality for
/// row-based access patterns.
///
/// # Generic Parameters
///
/// - `T`: Element type (default `f32`)
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
/// use booste_rs::data::{DenseMatrix, DataMatrix};
///
/// // 2 rows, 3 features
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let matrix = DenseMatrix::from_vec(data, 2, 3);
///
/// assert_eq!(matrix.num_rows(), 2);
/// assert_eq!(matrix.num_features(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct DenseMatrix<T = f32, S: AsRef<[T]> = Box<[T]>> {
    data: S,
    num_rows: usize,
    num_features: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> DenseMatrix<T, Box<[T]>> {
    /// Create a dense matrix from a Vec, taking ownership.
    ///
    /// Data should be in row-major order: `[row0_feat0, row0_feat1, ..., row1_feat0, ...]`
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != num_rows * num_features`.
    pub fn from_vec(data: Vec<T>, num_rows: usize, num_features: usize) -> Self {
        assert_eq!(
            data.len(),
            num_rows * num_features,
            "Data length {} does not match dimensions {}x{}",
            data.len(),
            num_rows,
            num_features
        );
        Self {
            data: data.into_boxed_slice(),
            num_rows,
            num_features,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, S: AsRef<[T]>> DenseMatrix<T, S> {
    /// Create a dense matrix from storage.
    ///
    /// # Panics
    ///
    /// Panics if `storage.as_ref().len() != num_rows * num_features`.
    pub fn new(storage: S, num_rows: usize, num_features: usize) -> Self {
        assert_eq!(
            storage.as_ref().len(),
            num_rows * num_features,
            "Storage length {} does not match dimensions {}x{}",
            storage.as_ref().len(),
            num_rows,
            num_features
        );
        Self {
            data: storage,
            num_rows,
            num_features,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T> DenseMatrix<T, &[T]> {
    /// Create a borrowed view of a dense matrix from a slice.
    ///
    /// This allows zero-copy matrix creation from existing data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != num_rows * num_features`.
    pub fn from_slice(data: &[T], num_rows: usize, num_features: usize) -> DenseMatrix<T, &[T]> {
        assert_eq!(
            data.len(),
            num_rows * num_features,
            "Data length {} does not match dimensions {}x{}",
            data.len(),
            num_rows,
            num_features
        );
        DenseMatrix {
            data,
            num_rows,
            num_features,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, S: AsRef<[T]>> DenseMatrix<T, S> {
    /// Get the underlying data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_ref()
    }

    /// Get a row as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `row >= num_rows`.
    #[inline]
    pub fn row_slice(&self, row: usize) -> &[T] {
        assert!(row < self.num_rows, "Row index {} out of bounds", row);
        let start = row * self.num_features;
        let end = start + self.num_features;
        &self.data.as_ref()[start..end]
    }
}

impl<T: Copy, S: AsRef<[T]>> DataMatrix for DenseMatrix<T, S> {
    type Element = T;
    type Row<'a>
        = DenseRowView<'a, T>
    where
        Self: 'a;

    #[inline]
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    #[inline]
    fn num_features(&self) -> usize {
        self.num_features
    }

    #[inline]
    fn row(&self, i: usize) -> Self::Row<'_> {
        DenseRowView {
            data: self.row_slice(i),
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> Option<T> {
        if row >= self.num_rows || col >= self.num_features {
            return None;
        }
        Some(self.data.as_ref()[row * self.num_features + col])
    }

    #[inline]
    fn is_dense(&self) -> bool {
        true
    }

    fn copy_row(&self, i: usize, buf: &mut [T]) {
        assert!(
            buf.len() >= self.num_features,
            "Buffer too small: {} < {}",
            buf.len(),
            self.num_features
        );
        buf[..self.num_features].copy_from_slice(self.row_slice(i));
    }

    #[allow(clippy::eq_op)] // x != x is intentional for NaN detection
    fn has_missing(&self) -> bool
    where
        Self::Element: PartialEq,
    {
        // For f32, NaN != NaN, so we use this property
        self.data.as_ref().iter().any(|&x| x != x)
    }

    #[allow(clippy::eq_op)] // x == x is intentional for non-NaN detection
    fn density(&self) -> f64
    where
        Self::Element: PartialEq,
    {
        if self.num_rows == 0 || self.num_features == 0 {
            return 1.0;
        }
        let total = self.num_rows * self.num_features;
        // Count non-NaN values (using x == x which is false for NaN)
        let non_missing = self.data.as_ref().iter().filter(|&&x| x == x).count();
        non_missing as f64 / total as f64
    }
}

/// View of a single row in a dense matrix.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = DenseMatrix::from_vec(data, 2, 3);

        assert_eq!(matrix.num_rows(), 2);
        assert_eq!(matrix.num_features(), 3);
    }

    #[test]
    fn create_from_slice() {
        let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix: DenseMatrix<f32, &[f32]> = DenseMatrix::from_slice(&data, 2, 3);

        assert_eq!(matrix.num_rows(), 2);
        assert_eq!(matrix.num_features(), 3);
    }

    #[test]
    #[should_panic(expected = "does not match dimensions")]
    fn create_wrong_size_panics() {
        let data = vec![1.0, 2.0, 3.0];
        DenseMatrix::from_vec(data, 2, 3); // 3 != 2*3
    }

    #[test]
    fn get_element() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = DenseMatrix::from_vec(data, 2, 3);

        assert_eq!(matrix.get(0, 0), Some(1.0));
        assert_eq!(matrix.get(0, 2), Some(3.0));
        assert_eq!(matrix.get(1, 0), Some(4.0));
        assert_eq!(matrix.get(1, 2), Some(6.0));

        // Out of bounds
        assert_eq!(matrix.get(2, 0), None);
        assert_eq!(matrix.get(0, 3), None);
    }

    #[test]
    fn row_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = DenseMatrix::from_vec(data, 2, 3);

        assert_eq!(matrix.row_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.row_slice(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn row_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = DenseMatrix::from_vec(data, 2, 3);

        let row0 = matrix.row(0);
        assert_eq!(row0.nnz(), 3);
        assert_eq!(row0.get(0), Some(1.0));
        assert_eq!(row0.get(2), Some(3.0));
        assert_eq!(row0.get(3), None);
    }

    #[test]
    fn row_iteration() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = DenseMatrix::from_vec(data, 1, 3);

        let row = matrix.row(0);
        let pairs: Vec<_> = row.iter().collect();

        assert_eq!(pairs, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
    }

    #[test]
    fn copy_row() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = DenseMatrix::from_vec(data, 2, 3);

        let mut buf = [0.0f32; 5];
        matrix.copy_row(1, &mut buf);

        assert_eq!(&buf[..3], &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn is_dense() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = DenseMatrix::from_vec(data, 1, 3);

        assert!(matrix.is_dense());
    }

    #[test]
    fn has_missing_false() {
        let data = vec![1.0, 2.0, 3.0];
        let matrix = DenseMatrix::from_vec(data, 1, 3);

        assert!(!matrix.has_missing());
    }

    #[test]
    fn has_missing_true() {
        let data = vec![1.0, f32::NAN, 3.0];
        let matrix = DenseMatrix::from_vec(data, 1, 3);

        assert!(matrix.has_missing());
    }

    #[test]
    fn density_no_missing() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = DenseMatrix::from_vec(data, 2, 2);

        assert_eq!(matrix.density(), 1.0);
    }

    #[test]
    fn density_with_missing() {
        let data = vec![1.0, f32::NAN, 3.0, 4.0];
        let matrix = DenseMatrix::from_vec(data, 2, 2);

        assert_eq!(matrix.density(), 0.75); // 3/4 non-missing
    }

    #[test]
    fn density_empty() {
        let data: Vec<f32> = vec![];
        let matrix = DenseMatrix::from_vec(data, 0, 0);

        assert_eq!(matrix.density(), 1.0);
    }
}
