//! Dataset construction errors.
//!
//! This module defines error types for dataset validation.

use thiserror::Error;

/// Dataset construction and validation errors.
#[derive(Debug, Clone, Error)]
pub enum DatasetError {
    /// Shape mismatch between components.
    #[error("{field}: expected {expected} samples, got {got}")]
    ShapeMismatch {
        expected: usize,
        got: usize,
        field: &'static str,
    },

    /// Feature count mismatch.
    #[error("schema has {schema} features, but data has {data} features")]
    FeatureCountMismatch { schema: usize, data: usize },

    /// Invalid category value detected.
    #[error("feature {feature_idx}: invalid category value {value} ({reason})")]
    InvalidCategory {
        feature_idx: usize,
        value: f32,
        reason: &'static str,
    },

    /// Sparse column indices not sorted.
    #[error("feature {feature_idx}: sparse indices must be sorted")]
    UnsortedSparseIndices { feature_idx: usize },

    /// Duplicate sparse indices.
    #[error("feature {feature_idx}: duplicate sparse index {index}")]
    DuplicateSparseIndices { feature_idx: usize, index: u32 },

    /// Sparse index out of bounds.
    #[error("feature {feature_idx}: sparse index {index} >= n_samples {n_samples}")]
    SparseIndexOutOfBounds {
        feature_idx: usize,
        index: u32,
        n_samples: usize,
    },

    /// No features provided.
    #[error("dataset has no features")]
    EmptyFeatures,

    /// Categorical features not supported (for GBLinear).
    #[error("categorical features not supported for this model type (feature {feature_idx})")]
    CategoricalNotSupported { feature_idx: usize },
}

impl DatasetError {
    /// Create a shape mismatch error.
    pub fn shape_mismatch(expected: usize, got: usize, field: &'static str) -> Self {
        Self::ShapeMismatch {
            expected,
            got,
            field,
        }
    }

    /// Create a sparse indices not sorted error.
    pub fn unsorted_sparse(feature_idx: usize) -> Self {
        Self::UnsortedSparseIndices { feature_idx }
    }

    /// Create a duplicate sparse indices error.
    pub fn duplicate_sparse(feature_idx: usize, index: u32) -> Self {
        Self::DuplicateSparseIndices { feature_idx, index }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages() {
        let err = DatasetError::shape_mismatch(100, 50, "targets");
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
        assert!(err.to_string().contains("targets"));

        let err = DatasetError::unsorted_sparse(3);
        assert!(err.to_string().contains("3"));
        assert!(err.to_string().contains("sorted"));

        let err = DatasetError::duplicate_sparse(2, 42);
        assert!(err.to_string().contains("2"));
        assert!(err.to_string().contains("42"));
    }

    // Verify Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn error_is_send_sync() {
        assert_send_sync::<DatasetError>();
    }
}
