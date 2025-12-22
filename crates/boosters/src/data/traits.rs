//! Core traits for data matrix access.

// ============================================================================
// FeatureAccessor Trait
// ============================================================================

/// Uniform access to feature values for tree traversal.
///
/// Abstracts over different data layouts (sample-major, feature-major, binned).
/// Returns `f32` values suitable for comparison with split thresholds.
///
/// # Design
///
/// This trait provides a minimal interface for tree traversal:
/// - Always returns `f32` (suitable for threshold comparison)
/// - Returns `f32::NAN` for missing values
/// - Uses (row, feature) indexing where row = sample index
///
/// # Implementations
///
/// Implemented for semantic wrapper types only (not raw ndarray types):
/// - [`SamplesView`](super::SamplesView): Sample-major `[n_samples, n_features]`
/// - [`FeaturesView`](super::FeaturesView): Feature-major `[n_features, n_samples]`
/// - [`BinnedAccessor`](crate::inference::gbdt::BinnedAccessor): Binned data with midpoint conversion
///
/// Note: We intentionally do not implement this for raw `Array2<f32>` or `ArrayView2<f32>`
/// because the axis semantics (samples vs features) depend on context. Use the wrapper
/// types to make the layout explicit.
pub trait FeatureAccessor {
    /// Get the feature value at (row, feature_index).
    ///
    /// Returns `f32::NAN` for missing values.
    fn get_feature(&self, row: usize, feature: usize) -> f32;

    /// Number of rows (samples) in the dataset.
    fn num_rows(&self) -> usize;

    /// Number of features.
    fn num_features(&self) -> usize;
}
