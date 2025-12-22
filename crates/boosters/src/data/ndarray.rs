//! ndarray integration and semantic array wrappers.
//!
//! This module provides semantic axis constants and newtype wrappers for
//! feature matrices with explicit memory layout guarantees.
//!
//! # Terminology
//!
//! - **Samples**: Training/inference instances (rows in typical ML terminology)
//! - **Features**: Input variables (columns in typical ML terminology)
//! - **Groups**: Output dimensions (1 for regression, K for K-class classification)
//!
//! # Memory Layouts
//!
//! - **Sample-major** (`SampleMajorFeatures`): Samples contiguous - `[s0_f0, s0_f1, ..., s1_f0, ...]`
//! - **Feature-major** (`FeatureMajorFeatures`): Features contiguous - `[f0_s0, f0_s1, ..., f1_s0, ...]`

use ndarray::{Array2, ArrayView1, ArrayView2};
use std::ops::Deref;

/// Semantic axis constants for ML domain.
pub mod axis {
    use ndarray::Axis;

    pub const ROWS: Axis = Axis(0);
    pub const COLS: Axis = Axis(1);
}

// =============================================================================
// Sample-Major Features (Samples Contiguous)
// =============================================================================

/// Feature matrix with sample-major (row-major) layout.
///
/// Shape: `[n_samples, n_features]` - each sample's features are contiguous.
///
/// This is the standard layout for inference where we iterate over samples.
/// Compatible with numpy's default C-order arrays.
///
/// # Example
///
/// ```
/// use boosters::data::SampleMajorFeatures;
/// use ndarray::Array2;
///
/// // Create from raw data: 3 samples, 2 features each
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let features = SampleMajorFeatures::from_vec(data, 3, 2).unwrap();
///
/// assert_eq!(features.n_samples(), 3);
/// assert_eq!(features.n_features(), 2);
/// assert_eq!(features.get(0, 1), 2.0); // sample 0, feature 1
/// ```
#[derive(Debug, Clone)]
pub struct SampleMajorFeatures(Array2<f32>);

impl SampleMajorFeatures {
    /// Create from a flat Vec in sample-major (row-major) order.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat vector of length `n_samples * n_features`
    /// * `n_samples` - Number of samples
    /// * `n_features` - Number of features per sample
    ///
    /// # Returns
    ///
    /// `None` if `data.len() != n_samples * n_features`
    pub fn from_vec(data: Vec<f32>, n_samples: usize, n_features: usize) -> Option<Self> {
        Array2::from_shape_vec((n_samples, n_features), data)
            .ok()
            .map(Self)
    }

    /// Create a view from a slice in sample-major order.
    ///
    /// This is zero-copy.
    pub fn view_from_slice(
        data: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> Option<SampleMajorFeaturesView<'_>> {
        ArrayView2::from_shape((n_samples, n_features), data)
            .ok()
            .map(SampleMajorFeaturesView)
    }

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.0.nrows()
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.0.ncols()
    }

    /// Get feature value at (sample, feature).
    #[inline]
    pub fn get(&self, sample: usize, feature: usize) -> f32 {
        self.0[[sample, feature]]
    }

    /// Get an immutable view.
    pub fn view(&self) -> SampleMajorFeaturesView<'_> {
        SampleMajorFeaturesView(self.0.view())
    }

    /// Get the inner array.
    pub fn into_inner(self) -> Array2<f32> {
        self.0
    }

    /// Get a reference to the inner array.
    pub fn as_array(&self) -> &Array2<f32> {
        &self.0
    }
}

impl Deref for SampleMajorFeatures {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Immutable view into sample-major features.
#[derive(Debug, Clone, Copy)]
pub struct SampleMajorFeaturesView<'a>(ArrayView2<'a, f32>);

impl<'a> SampleMajorFeaturesView<'a> {
    /// Create from an ArrayView2 that is already sample-major.
    ///
    /// # Safety
    ///
    /// Caller must ensure the array is in C-order (sample-major).
    pub fn from_array_view(view: ArrayView2<'a, f32>) -> Self {
        debug_assert!(view.is_standard_layout(), "Array must be in C-order");
        Self(view)
    }

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.0.nrows()
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.0.ncols()
    }

    /// Get feature value at (sample, feature).
    #[inline]
    pub fn get(&self, sample: usize, feature: usize) -> f32 {
        self.0[[sample, feature]]
    }

    /// Get the inner array view.
    pub fn as_array_view(&self) -> ArrayView2<'a, f32> {
        self.0
    }
}

impl<'a> Deref for SampleMajorFeaturesView<'a> {
    type Target = ArrayView2<'a, f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// =============================================================================
// Feature-Major Features (Features Contiguous)
// =============================================================================

/// Feature matrix with feature-major (column-major) layout.
///
/// Shape: `[n_features, n_samples]` - each feature's values across all samples are contiguous.
///
/// This layout is optimal for training where we iterate over features
/// (e.g., histogram building, coordinate descent in GBLinear).
///
/// # Example
///
/// ```
/// use boosters::data::FeatureMajorFeatures;
/// use ndarray::Array2;
///
/// // Create from raw data: 2 features, 3 samples
/// // Data is [f0_s0, f0_s1, f0_s2, f1_s0, f1_s1, f1_s2]
/// let data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
/// let features = FeatureMajorFeatures::from_vec(data, 3, 2).unwrap();
///
/// assert_eq!(features.n_samples(), 3);
/// assert_eq!(features.n_features(), 2);
/// assert_eq!(features.get(0, 1), 10.0); // sample 0, feature 1
/// ```
#[derive(Debug, Clone)]
pub struct FeatureMajorFeatures(Array2<f32>);

impl FeatureMajorFeatures {
    /// Create from a flat Vec in feature-major order.
    ///
    /// Data layout: `[f0_s0, f0_s1, ..., f1_s0, f1_s1, ...]`
    ///
    /// # Arguments
    ///
    /// * `data` - Flat vector of length `n_samples * n_features`
    /// * `n_samples` - Number of samples
    /// * `n_features` - Number of features per sample
    ///
    /// # Returns
    ///
    /// `None` if `data.len() != n_samples * n_features`
    pub fn from_vec(data: Vec<f32>, n_samples: usize, n_features: usize) -> Option<Self> {
        // Shape is [n_features, n_samples] for feature-major
        Array2::from_shape_vec((n_features, n_samples), data)
            .ok()
            .map(Self)
    }

    /// Create a view from a slice in feature-major order.
    ///
    /// This is zero-copy.
    pub fn view_from_slice(
        data: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> Option<FeatureMajorFeaturesView<'_>> {
        // Shape is [n_features, n_samples] for feature-major
        ArrayView2::from_shape((n_features, n_samples), data)
            .ok()
            .map(FeatureMajorFeaturesView)
    }

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.0.ncols()
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.0.nrows()
    }

    /// Get feature value at (sample, feature).
    ///
    /// Note: Internally this accesses `[feature, sample]` due to the layout.
    #[inline]
    pub fn get(&self, sample: usize, feature: usize) -> f32 {
        self.0[[feature, sample]]
    }

    /// Get a contiguous view of all sample values for a feature.
    ///
    /// This is the fast path for feature-major data.
    #[inline]
    pub fn feature_values(&self, feature: usize) -> ArrayView1<'_, f32> {
        self.0.row(feature)
    }

    /// Get an immutable view.
    pub fn view(&self) -> FeatureMajorFeaturesView<'_> {
        FeatureMajorFeaturesView(self.0.view())
    }

    /// Get the inner array.
    pub fn into_inner(self) -> Array2<f32> {
        self.0
    }

    /// Get a reference to the inner array.
    pub fn as_array(&self) -> &Array2<f32> {
        &self.0
    }
}

impl Deref for FeatureMajorFeatures {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Immutable view into feature-major features.
#[derive(Debug, Clone, Copy)]
pub struct FeatureMajorFeaturesView<'a>(ArrayView2<'a, f32>);

impl<'a> FeatureMajorFeaturesView<'a> {
    /// Create from an ArrayView2 that is already feature-major.
    ///
    /// Shape must be `[n_features, n_samples]`.
    ///
    /// # Safety
    ///
    /// Caller must ensure the array is in C-order with shape [n_features, n_samples].
    pub fn from_array_view(view: ArrayView2<'a, f32>) -> Self {
        debug_assert!(view.is_standard_layout(), "Array must be in C-order");
        Self(view)
    }

    /// Number of samples.
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.0.ncols()
    }

    /// Number of features.
    #[inline]
    pub fn n_features(&self) -> usize {
        self.0.nrows()
    }

    /// Get feature value at (sample, feature).
    #[inline]
    pub fn get(&self, sample: usize, feature: usize) -> f32 {
        self.0[[feature, sample]]
    }

    /// Get a contiguous view of all sample values for a feature.
    #[inline]
    pub fn feature_values(&self, feature: usize) -> ArrayView1<'_, f32> {
        self.0.row(feature)
    }

    /// Get the inner array view.
    pub fn as_array_view(&self) -> ArrayView2<'a, f32> {
        self.0
    }
}

impl<'a> Deref for FeatureMajorFeaturesView<'a> {
    type Target = ArrayView2<'a, f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// =============================================================================
// Prediction Helpers
// =============================================================================

/// Initialize a prediction array with base scores.
///
/// Creates an array of shape `[n_groups, n_samples]` where each group's
/// row is filled with the corresponding base score.
///
/// # Arguments
///
/// * `base_scores` - Base score for each group (length = n_groups)
/// * `n_samples` - Number of samples
///
/// # Example
///
/// ```
/// use boosters::data::init_predictions;
///
/// let base_scores = vec![0.5, -0.3, 0.1]; // 3 groups
/// let preds = init_predictions(&base_scores, 100);
///
/// assert_eq!(preds.shape(), &[3, 100]);
/// assert_eq!(preds[[0, 0]], 0.5);  // group 0
/// assert_eq!(preds[[1, 50]], -0.3); // group 1
/// ```
pub fn init_predictions(base_scores: &[f32], n_samples: usize) -> Array2<f32> {
    let n_groups = base_scores.len();
    let mut predictions = Array2::zeros((n_groups, n_samples));
    for (group, &base_score) in base_scores.iter().enumerate() {
        predictions.row_mut(group).fill(base_score);
    }
    predictions
}

/// Initialize predictions from base scores into a flat Vec (column-major layout).
///
/// Layout: `[group0_all_samples, group1_all_samples, ...]`
///
/// This is a convenience for code that still uses Vec<f32> for predictions.
pub fn init_predictions_vec(base_scores: &[f32], n_samples: usize) -> Vec<f32> {
    let n_groups = base_scores.len();
    let mut predictions = vec![0.0f32; n_groups * n_samples];
    for (group, &base_score) in base_scores.iter().enumerate() {
        let start = group * n_samples;
        predictions[start..start + n_samples].fill(base_score);
    }
    predictions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_major_creation() {
        // 2 samples, 3 features: [[1,2,3], [4,5,6]]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let features = SampleMajorFeatures::from_vec(data, 2, 3).unwrap();

        assert_eq!(features.n_samples(), 2);
        assert_eq!(features.n_features(), 3);
        assert_eq!(features.get(0, 0), 1.0);
        assert_eq!(features.get(0, 2), 3.0);
        assert_eq!(features.get(1, 0), 4.0);
        assert_eq!(features.get(1, 2), 6.0);
    }

    #[test]
    fn test_feature_major_creation() {
        // 2 features, 3 samples
        // Feature-major: [f0_s0, f0_s1, f0_s2, f1_s0, f1_s1, f1_s2]
        // = [1, 2, 3, 10, 20, 30]
        let data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let features = FeatureMajorFeatures::from_vec(data, 3, 2).unwrap();

        assert_eq!(features.n_samples(), 3);
        assert_eq!(features.n_features(), 2);
        assert_eq!(features.get(0, 0), 1.0);  // sample 0, feature 0
        assert_eq!(features.get(1, 0), 2.0);  // sample 1, feature 0
        assert_eq!(features.get(0, 1), 10.0); // sample 0, feature 1
        assert_eq!(features.get(2, 1), 30.0); // sample 2, feature 1
    }

    #[test]
    fn test_feature_major_feature_values() {
        let data = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let features = FeatureMajorFeatures::from_vec(data, 3, 2).unwrap();

        // Feature 0 values should be [1, 2, 3]
        assert_eq!(features.feature_values(0).as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        // Feature 1 values should be [10, 20, 30]
        assert_eq!(features.feature_values(1).as_slice().unwrap(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_init_predictions() {
        let base_scores = vec![0.5, -0.3];
        let preds = init_predictions(&base_scores, 3);

        assert_eq!(preds.shape(), &[2, 3]);
        assert_eq!(preds[[0, 0]], 0.5);
        assert_eq!(preds[[0, 2]], 0.5);
        assert_eq!(preds[[1, 0]], -0.3);
        assert_eq!(preds[[1, 1]], -0.3);
    }

    #[test]
    fn test_init_predictions_vec() {
        let base_scores = vec![0.5, -0.3];
        let preds = init_predictions_vec(&base_scores, 3);

        // Layout: [g0_s0, g0_s1, g0_s2, g1_s0, g1_s1, g1_s2]
        assert_eq!(preds.len(), 6);
        assert_eq!(preds[0], 0.5);  // group 0, sample 0
        assert_eq!(preds[2], 0.5);  // group 0, sample 2
        assert_eq!(preds[3], -0.3); // group 1, sample 0
        assert_eq!(preds[5], -0.3); // group 1, sample 2
    }

    #[test]
    fn test_view_from_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        
        let view = SampleMajorFeatures::view_from_slice(&data, 2, 3).unwrap();
        assert_eq!(view.n_samples(), 2);
        assert_eq!(view.get(1, 0), 4.0);

        let view = FeatureMajorFeatures::view_from_slice(&data, 3, 2).unwrap();
        assert_eq!(view.n_features(), 2);
        assert_eq!(view.get(0, 1), 4.0); // sample 0, feature 1 -> data[3]
    }
}
