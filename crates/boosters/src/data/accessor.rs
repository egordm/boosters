//! Sample and data accessor traits for tree traversal.
//!
//! This module provides the core abstractions for accessing feature data:
//!
//! - [`SampleAccessor`]: Access features for a single sample (row)
//! - [`DataAccessor`]: Access samples from a multi-sample dataset (matrix)
//!
//! # Design
//!
//! The two-trait design cleanly separates single-sample and multi-sample access:
//!
//! - `SampleAccessor` is implemented by `&[f32]`, allowing slices to be used directly
//!   for tree traversal without wrapper types
//! - `DataAccessor` is implemented by matrix types (`SamplesView`, `FeaturesView`,
//!   `BinnedDataset`) and provides sample extraction
//!
//! # Example
//!
//! ```ignore
//! use boosters::data::{SampleAccessor, DataAccessor};
//!
//! // Direct slice usage for single-sample traversal
//! let features: &[f32] = &[0.5, 1.2, 3.4];
//! let value = features.feature(0);  // 0.5
//!
//! // Matrix access with sample extraction
//! let matrix: SamplesView = /* ... */;
//! for i in 0..matrix.n_samples() {
//!     let sample = matrix.sample(i);
//!     let value = sample.feature(0);
//! }
//! ```

use crate::dataset::FeatureType;

// ============================================================================
// SampleAccessor Trait
// ============================================================================

/// Access features for a single sample.
///
/// This trait provides read-only access to feature values for one sample (row).
/// It is implemented by `&[f32]` directly, allowing slices to be used for
/// tree traversal without wrapper types.
///
/// # Implementations
///
/// - `&[f32]`: Direct slice access (zero-cost)
/// - Returned by [`DataAccessor::sample`] for matrix types
///
/// # Example
///
/// ```ignore
/// let features: &[f32] = &[0.5, 1.2, 3.4];
/// assert_eq!(features.feature(0), 0.5);
/// assert_eq!(features.n_features(), 3);
/// ```
pub trait SampleAccessor {
    /// Get the feature value at the given index.
    ///
    /// Returns `f32::NAN` for missing values.
    fn feature(&self, index: usize) -> f32;

    /// Number of features in this sample.
    fn n_features(&self) -> usize;
}

// Blanket implementation for slices
impl SampleAccessor for [f32] {
    #[inline]
    fn feature(&self, index: usize) -> f32 {
        self[index]
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.len()
    }
}

// Implementation for fixed-size arrays (enables &[0.5f32, 1.0] syntax)
impl<const N: usize> SampleAccessor for [f32; N] {
    #[inline]
    fn feature(&self, index: usize) -> f32 {
        self[index]
    }

    #[inline]
    fn n_features(&self) -> usize {
        N
    }
}

// Implementation for references to types that can be viewed as slices
impl<T: AsRef<[f32]> + ?Sized> SampleAccessor for &T {
    #[inline]
    fn feature(&self, index: usize) -> f32 {
        self.as_ref()[index]
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.as_ref().len()
    }
}

// Implementation for ndarray ArrayView1 (may be contiguous or strided)
impl SampleAccessor for ndarray::ArrayView1<'_, f32> {
    #[inline]
    fn feature(&self, index: usize) -> f32 {
        self[index]
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.len()
    }
}

// ============================================================================
// DataAccessor Trait
// ============================================================================

/// Access samples from a multi-sample dataset.
///
/// This trait provides read-only access to a collection of samples, where each
/// sample can be accessed via [`sample`](Self::sample). It also provides
/// feature metadata (types) when available.
///
/// # Implementations
///
/// - [`SamplesView`](crate::dataset::SamplesView): Sample-major `[n_samples, n_features]`
/// - [`FeaturesView`](crate::dataset::FeaturesView): Feature-major `[n_features, n_samples]`
/// - [`BinnedDataset`](crate::data::binned::BinnedDataset): Binned data with midpoint conversion
///
/// # Feature Types
///
/// The [`feature_type`](Self::feature_type) method returns the logical type of each feature.
/// This is used during tree traversal to handle categorical splits differently from
/// numeric splits. If no schema is available, all features are assumed numeric.
pub trait DataAccessor {
    /// The type returned when accessing a single sample.
    ///
    /// For contiguous layouts (like `SamplesView`), this is typically `&[f32]`.
    /// For non-contiguous layouts, this may be a wrapper type.
    type Sample<'a>: SampleAccessor
    where
        Self: 'a;

    /// Get a sample (row) by index.
    ///
    /// Returns a view into the sample's features.
    fn sample(&self, index: usize) -> Self::Sample<'_>;

    /// Number of samples in the dataset.
    fn n_samples(&self) -> usize;

    /// Number of features per sample.
    fn n_features(&self) -> usize;

    /// Get the logical type of a feature.
    ///
    /// Returns [`FeatureType::Numeric`] by default if no schema is available.
    #[inline]
    fn feature_type(&self, _feature: usize) -> FeatureType {
        FeatureType::Numeric
    }

    /// Check if any feature is categorical.
    ///
    /// Returns `false` by default if no schema is available.
    #[inline]
    fn has_categorical(&self) -> bool {
        false
    }
}
