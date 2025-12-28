//! Feature column storage types.
//!
//! # Deprecation Notice
//!
//! This module is being replaced by the types in [`super::binned::v2`].
//! Use `BinnedDatasetBuilder::add_features()` instead of building columns manually.
//!
//! This module defines [`Column`] and [`SparseColumn`] for storing feature data.

#![allow(deprecated)] // Allow internal use of deprecated types

use ndarray::Array1;

/// Single feature column storage.
///
/// # Deprecation
///
/// This type is deprecated. Use `BinnedDatasetBuilder::add_features()` instead.
#[deprecated(
    since = "0.2.0",
    note = "Use BinnedDatasetBuilder::add_features() instead of building columns manually"
)]
///
/// All values are `f32`. Categorical features store category IDs as floats
/// (e.g., `0.0, 1.0, 2.0`) and are cast to `i32` during binning.
#[derive(Debug, Clone)]
pub enum Column {
    /// Dense array of values.
    Dense(Array1<f32>),

    /// Sparse CSR-like column.
    Sparse(SparseColumn),
}

impl Column {
    /// Create a dense column from values.
    pub fn dense(values: impl Into<Array1<f32>>) -> Self {
        Self::Dense(values.into())
    }

    /// Create a sparse column.
    pub fn sparse(indices: Vec<u32>, values: Vec<f32>, n_samples: usize, default: f32) -> Self {
        Self::Sparse(SparseColumn::new(indices, values, n_samples, default))
    }

    /// Number of samples in this column.
    pub fn n_samples(&self) -> usize {
        match self {
            Column::Dense(arr) => arr.len(),
            Column::Sparse(s) => s.n_samples,
        }
    }

    /// Get value at index.
    ///
    /// For sparse columns, returns the default value if the index is not present.
    pub fn get(&self, idx: usize) -> f32 {
        match self {
            Column::Dense(arr) => arr[idx],
            Column::Sparse(s) => s.get(idx),
        }
    }

    /// Returns true if this is a sparse column.
    pub fn is_sparse(&self) -> bool {
        matches!(self, Column::Sparse(_))
    }

    /// Returns true if this is a dense column.
    pub fn is_dense(&self) -> bool {
        matches!(self, Column::Dense(_))
    }

    /// Get dense values as a slice (returns None for sparse).
    pub fn as_slice(&self) -> Option<&[f32]> {
        match self {
            Column::Dense(arr) => arr.as_slice(),
            Column::Sparse(_) => None,
        }
    }

    /// Convert to dense array, expanding sparse if necessary.
    pub fn to_dense(&self) -> Array1<f32> {
        match self {
            Column::Dense(arr) => arr.clone(),
            Column::Sparse(s) => s.to_dense(),
        }
    }
}

/// Sparse column storage.
///
/// Stores non-default values at specified indices. Missing indices get the default value.
///
/// # Deprecation
///
/// This type is deprecated. Use `BinnedDatasetBuilder::add_features()` instead.
#[deprecated(
    since = "0.2.0",
    note = "Use BinnedDatasetBuilder::add_features() instead of building columns manually"
)]
#[derive(Debug, Clone)]
pub struct SparseColumn {
    /// Non-default row indices (must be sorted, no duplicates).
    pub indices: Vec<u32>,

    /// Values at those indices.
    pub values: Vec<f32>,

    /// Total number of samples.
    pub n_samples: usize,

    /// Default value for unspecified entries.
    pub default: f32,
}

impl SparseColumn {
    /// Create a new sparse column.
    ///
    /// # Panics
    ///
    /// Debug-asserts that indices and values have the same length.
    pub fn new(indices: Vec<u32>, values: Vec<f32>, n_samples: usize, default: f32) -> Self {
        debug_assert_eq!(
            indices.len(),
            values.len(),
            "indices and values must have same length"
        );
        Self {
            indices,
            values,
            n_samples,
            default,
        }
    }

    /// Get value at index.
    ///
    /// Uses binary search. Returns default if index not found.
    pub fn get(&self, idx: usize) -> f32 {
        let idx = idx as u32;
        match self.indices.binary_search(&idx) {
            Ok(pos) => self.values[pos],
            Err(_) => self.default,
        }
    }

    /// Number of non-default values.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio (1.0 = all default, 0.0 = all specified).
    pub fn sparsity(&self) -> f64 {
        if self.n_samples == 0 {
            return 0.0;
        }
        1.0 - (self.nnz() as f64 / self.n_samples as f64)
    }

    /// Check if indices are sorted and have no duplicates.
    pub fn validate(&self) -> Result<(), (usize, u32)> {
        for i in 1..self.indices.len() {
            if self.indices[i] <= self.indices[i - 1] {
                if self.indices[i] == self.indices[i - 1] {
                    return Err((i, self.indices[i])); // duplicate
                } else {
                    return Err((i, self.indices[i])); // unsorted
                }
            }
        }
        Ok(())
    }

    /// Convert to dense array.
    pub fn to_dense(&self) -> Array1<f32> {
        let mut arr = Array1::from_elem(self.n_samples, self.default);
        for (idx, val) in self.indices.iter().zip(&self.values) {
            arr[*idx as usize] = *val;
        }
        arr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn column_dense_basic() {
        let col = Column::dense(array![1.0, 2.0, 3.0]);
        assert_eq!(col.n_samples(), 3);
        assert!(col.is_dense());
        assert!(!col.is_sparse());
        assert_eq!(col.get(0), 1.0);
        assert_eq!(col.get(2), 3.0);
        assert_eq!(col.as_slice(), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn column_sparse_basic() {
        let col = Column::sparse(vec![1, 3], vec![10.0, 30.0], 5, 0.0);
        assert_eq!(col.n_samples(), 5);
        assert!(col.is_sparse());
        assert!(!col.is_dense());
        assert_eq!(col.get(0), 0.0); // default
        assert_eq!(col.get(1), 10.0);
        assert_eq!(col.get(2), 0.0); // default
        assert_eq!(col.get(3), 30.0);
        assert_eq!(col.get(4), 0.0); // default
        assert!(col.as_slice().is_none());
    }

    #[test]
    fn sparse_column_to_dense() {
        let sparse = SparseColumn::new(vec![1, 3], vec![10.0, 30.0], 5, 0.0);
        let dense = sparse.to_dense();
        assert_eq!(dense.as_slice().unwrap(), &[0.0, 10.0, 0.0, 30.0, 0.0]);
    }

    #[test]
    fn sparse_column_sparsity() {
        let sparse = SparseColumn::new(vec![0, 2], vec![1.0, 2.0], 10, 0.0);
        assert!((sparse.sparsity() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn sparse_column_validate() {
        // Valid
        let sparse = SparseColumn::new(vec![0, 1, 5], vec![1.0, 2.0, 3.0], 10, 0.0);
        assert!(sparse.validate().is_ok());

        // Unsorted
        let sparse = SparseColumn::new(vec![0, 5, 2], vec![1.0, 2.0, 3.0], 10, 0.0);
        assert!(sparse.validate().is_err());

        // Duplicate
        let sparse = SparseColumn::new(vec![0, 1, 1], vec![1.0, 2.0, 3.0], 10, 0.0);
        assert!(sparse.validate().is_err());
    }

    #[test]
    fn column_to_dense() {
        let col = Column::dense(array![1.0, 2.0]);
        assert_eq!(col.to_dense(), array![1.0, 2.0]);

        let col = Column::sparse(vec![0], vec![5.0], 3, 0.0);
        assert_eq!(col.to_dense(), array![5.0, 0.0, 0.0]);
    }

    // Verify Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn types_are_send_sync() {
        assert_send_sync::<Column>();
        assert_send_sync::<SparseColumn>();
    }
}
