//! Matrix layout traits and types.
//!
//! This module defines [`Layout`] for abstracting over row-major and column-major
//! dense matrix storage. The layout affects how elements are indexed in contiguous
//! memory:
//!
//! - [`RowMajor`]: Rows are contiguous. `index = row * num_cols + col`
//! - [`ColMajor`]: Columns are contiguous. `index = col * num_rows + row`
//!
//! # Zero-Cost Abstraction
//!
//! Layout is a type parameter, so all layout-dependent code is monomorphized.
//! There is no runtime overhead for layout dispatch.
//!
//! # Example
//!
//! ```
//! use booste_rs::data::{DenseMatrix, RowMajor, ColMajor};
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

use std::iter::FusedIterator;

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

#[cfg(test)]
mod tests {
    use super::*;

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
