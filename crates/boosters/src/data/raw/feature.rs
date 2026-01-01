//! Feature storage types.
//!
//! This module defines [`Feature`] for storing individual feature data.
//! Features can be dense (contiguous array) or sparse (CSR-like storage).

use ndarray::Array1;

/// Single feature storage.
///
/// All values are `f32`. Categorical features store category IDs as floats
/// (e.g., `0.0, 1.0, 2.0`) and are cast to `i32` during binning.
///
/// # Variants
///
/// - `Dense`: Contiguous array of values for all samples
/// - `Sparse`: CSR-like storage with only non-default values stored
#[derive(Debug, Clone)]
pub enum Feature {
    /// Dense array of values, one per sample.
    Dense(Array1<f32>),

    /// Sparse storage with non-default values at specified indices.
    Sparse {
        /// Row indices with non-default values (sorted, no duplicates).
        indices: Vec<u32>,
        /// Values at those indices.
        values: Vec<f32>,
        /// Total number of samples.
        n_samples: usize,
        /// Default value for unspecified entries.
        default: f32,
    },
}

impl Feature {
    /// Create a dense feature from values.
    pub fn dense(values: impl Into<Array1<f32>>) -> Self {
        Self::Dense(values.into())
    }

    /// Create a sparse feature.
    ///
    /// # Arguments
    ///
    /// * `indices` - Row indices with non-default values (must be sorted, no duplicates)
    /// * `values` - Values at those indices
    /// * `n_samples` - Total number of samples
    /// * `default` - Default value for unspecified indices (typically 0.0)
    pub fn sparse(indices: Vec<u32>, values: Vec<f32>, n_samples: usize, default: f32) -> Self {
        debug_assert_eq!(
            indices.len(),
            values.len(),
            "indices and values must have same length"
        );
        Self::Sparse {
            indices,
            values,
            n_samples,
            default,
        }
    }

    /// Number of samples in this feature.
    pub fn n_samples(&self) -> usize {
        match self {
            Feature::Dense(arr) => arr.len(),
            Feature::Sparse { n_samples, .. } => *n_samples,
        }
    }

    /// Get value at index.
    ///
    /// For sparse features, returns the default value if the index is not present.
    pub fn get(&self, idx: usize) -> f32 {
        match self {
            Feature::Dense(arr) => arr[idx],
            Feature::Sparse {
                indices,
                values,
                default,
                ..
            } => {
                let idx = idx as u32;
                match indices.binary_search(&idx) {
                    Ok(pos) => values[pos],
                    Err(_) => *default,
                }
            }
        }
    }

    /// Returns true if this is a sparse feature.
    pub fn is_sparse(&self) -> bool {
        matches!(self, Feature::Sparse { .. })
    }

    /// Returns true if this is a dense feature.
    pub fn is_dense(&self) -> bool {
        matches!(self, Feature::Dense(_))
    }

    /// Get dense values as a slice (returns None for sparse).
    pub fn as_slice(&self) -> Option<&[f32]> {
        match self {
            Feature::Dense(arr) => arr.as_slice(),
            Feature::Sparse { .. } => None,
        }
    }

    /// Number of non-default values (for sparse) or total samples (for dense).
    pub fn nnz(&self) -> usize {
        match self {
            Feature::Dense(arr) => arr.len(),
            Feature::Sparse { values, .. } => values.len(),
        }
    }

    /// Sparsity ratio (0.0 for dense, calculated for sparse).
    pub fn sparsity(&self) -> f64 {
        match self {
            Feature::Dense(_) => 0.0,
            Feature::Sparse {
                values, n_samples, ..
            } => {
                if *n_samples == 0 {
                    0.0
                } else {
                    1.0 - (values.len() as f64 / *n_samples as f64)
                }
            }
        }
    }

    /// Validate sparse indices are sorted and have no duplicates.
    ///
    /// Returns Ok for dense features, or Err with (position, index) for invalid sparse.
    pub fn validate(&self) -> Result<(), (usize, u32)> {
        match self {
            Feature::Dense(_) => Ok(()),
            Feature::Sparse {
                indices, n_samples, ..
            } => {
                for i in 1..indices.len() {
                    if indices[i] <= indices[i - 1] {
                        return Err((i, indices[i]));
                    }
                }
                // Check bounds
                for (i, &idx) in indices.iter().enumerate() {
                    if idx as usize >= *n_samples {
                        return Err((i, idx));
                    }
                }
                Ok(())
            }
        }
    }

    /// Get default value (returns 0.0 for dense features by convention).
    pub fn default_value(&self) -> f32 {
        match self {
            Feature::Dense(_) => 0.0, // Dense features have no default concept
            Feature::Sparse { default, .. } => *default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn feature_dense_basic() {
        let feat = Feature::dense(array![1.0, 2.0, 3.0]);
        assert_eq!(feat.n_samples(), 3);
        assert!(feat.is_dense());
        assert!(!feat.is_sparse());
        assert_eq!(feat.get(0), 1.0);
        assert_eq!(feat.get(2), 3.0);
        assert_eq!(feat.as_slice(), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn feature_sparse_basic() {
        let feat = Feature::sparse(vec![1, 3], vec![10.0, 30.0], 5, 0.0);
        assert_eq!(feat.n_samples(), 5);
        assert!(feat.is_sparse());
        assert!(!feat.is_dense());
        assert_eq!(feat.get(0), 0.0); // default
        assert_eq!(feat.get(1), 10.0);
        assert_eq!(feat.get(2), 0.0); // default
        assert_eq!(feat.get(3), 30.0);
        assert_eq!(feat.get(4), 0.0); // default
        assert!(feat.as_slice().is_none());
    }

    #[test]
    fn feature_sparsity() {
        let sparse = Feature::sparse(vec![0, 2], vec![1.0, 2.0], 10, 0.0);
        assert!((sparse.sparsity() - 0.8).abs() < 1e-6);

        let dense = Feature::dense(array![1.0, 2.0, 3.0]);
        assert_eq!(dense.sparsity(), 0.0);
    }

    #[test]
    fn feature_validate() {
        // Dense is always valid
        let dense = Feature::dense(array![1.0, 2.0]);
        assert!(dense.validate().is_ok());

        // Valid sparse
        let sparse = Feature::sparse(vec![0, 1, 5], vec![1.0, 2.0, 3.0], 10, 0.0);
        assert!(sparse.validate().is_ok());

        // Unsorted
        let sparse = Feature::sparse(vec![0, 5, 2], vec![1.0, 2.0, 3.0], 10, 0.0);
        assert!(sparse.validate().is_err());

        // Duplicate
        let sparse = Feature::sparse(vec![0, 1, 1], vec![1.0, 2.0, 3.0], 10, 0.0);
        assert!(sparse.validate().is_err());

        // Out of bounds
        let sparse = Feature::sparse(vec![0, 15], vec![1.0, 2.0], 10, 0.0);
        assert!(sparse.validate().is_err());
    }

    // Verify Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn types_are_send_sync() {
        assert_send_sync::<Feature>();
    }
}
