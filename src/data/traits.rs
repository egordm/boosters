//! Core traits for data matrix access.

use std::iter::FusedIterator;

/// Core trait for feature matrix access.
///
/// Provides a uniform interface for accessing feature values during tree traversal,
/// regardless of underlying storage format (dense, sparse, etc.).
///
/// # Element Type
///
/// The associated `Element` type is typically `f32` for features. Missing values
/// are represented as `f32::NAN`.
///
/// # Row Access
///
/// The primary access pattern is row-based via [`row()`](Self::row), which returns
/// a [`RowView`] that can be iterated or indexed by feature.
///
/// # Integration with Traversal
///
/// This trait is designed to integrate with RFC-0003's visitor/traversal patterns.
/// The [`copy_row()`](Self::copy_row) method supports block-based traversal that
/// needs contiguous feature access.
pub trait DataMatrix {
    /// Element type stored in the matrix (typically f32).
    type Element: Copy;

    /// View type for a single row.
    type Row<'a>: RowView<Element = Self::Element>
    where
        Self: 'a;

    /// Number of rows (samples) in the matrix.
    fn num_rows(&self) -> usize;

    /// Number of features (columns) in the matrix.
    fn num_features(&self) -> usize;

    /// Get a view of row `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_rows()`.
    fn row(&self, i: usize) -> Self::Row<'_>;

    /// Get element at (row, col), or None if out of bounds.
    fn get(&self, row: usize, col: usize) -> Option<Self::Element>;

    /// Whether this is a dense matrix.
    ///
    /// Dense matrices have O(1) random access and store all elements.
    /// Sparse matrices only store non-zero elements.
    fn is_dense(&self) -> bool {
        false
    }

    /// Copy row `i` into a dense buffer.
    ///
    /// For sparse matrices, missing positions are filled with the default
    /// missing value (NaN for f32).
    ///
    /// # Panics
    ///
    /// Panics if `buf.len() < num_features()` or `i >= num_rows()`.
    fn copy_row(&self, i: usize, buf: &mut [Self::Element]);

    /// Whether the matrix contains any missing values.
    ///
    /// For f32 matrices, missing values are represented as NaN.
    fn has_missing(&self) -> bool
    where
        Self::Element: PartialEq;

    /// Density ratio: number of non-missing elements / total elements.
    ///
    /// Returns 1.0 for matrices with no missing values.
    /// Used to choose between row-at-a-time vs block traversal strategies.
    fn density(&self) -> f64
    where
        Self::Element: PartialEq;
}

/// View of a single row in a data matrix.
///
/// Provides iteration over (feature_index, value) pairs and random access
/// by feature index. For dense matrices, iteration yields all features.
/// For sparse matrices, iteration yields only non-zero features.
pub trait RowView {
    /// Element type.
    type Element: Copy;

    /// Iterator over (feature_index, value) pairs.
    type Iter<'a>: Iterator<Item = (usize, Self::Element)> + FusedIterator
    where
        Self: 'a;

    /// Number of stored (non-sparse) elements in this row.
    ///
    /// For dense rows, this equals the number of features.
    /// For sparse rows, this is the number of explicitly stored values.
    fn nnz(&self) -> usize;

    /// Get the value at `feature_idx`, or None if not stored.
    ///
    /// For dense matrices, returns None only if out of bounds.
    /// For sparse matrices, returns None for unstored (zero) entries.
    fn get(&self, feature_idx: usize) -> Option<Self::Element>;

    /// Iterate over (feature_index, value) pairs.
    fn iter(&self) -> Self::Iter<'_>;
}


