//! ndarray integration and semantic array wrappers.
//!
//! This module provides semantic newtype wrappers for feature matrices with
//! explicit memory layout guarantees. These wrappers clarify the axis semantics
//! when working with 2D arrays.
//!
//! # Terminology
//!
//! - **Samples**: Training/inference instances (rows in typical ML terminology)
//! - **Features**: Input variables (columns in typical ML terminology)
//! - **Groups**: Output dimensions (1 for regression, K for K-class classification)
//!
//! # Wrappers
//!
//! - [`SamplesView`]: View with shape `[n_samples, n_features]` - samples on rows
//! - [`FeaturesView`]: View with shape `[n_features, n_samples]` - features on rows
//!
//! Both implement [`FeatureAccessor`] for uniform tree traversal access.
//!
//! # Layout Conversion
//!
//! Use [`transpose_to_c_order`] to transpose and ensure C-order (row-major) layout:
//!
//! ```
//! use boosters::data::transpose_to_c_order;
//! use ndarray::Array2;
//!
//! // Feature-major data [n_features, n_samples]
//! let features = Array2::<f32>::zeros((3, 100));
//!
//! // Transpose to sample-major [n_samples, n_features] in C-order
//! let samples = transpose_to_c_order(features.view());
//! ```

use ndarray::{Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};
use std::ops::Deref;

use super::FeatureAccessor;

// =============================================================================
// Layout Utilities
// =============================================================================

/// Transpose a 2D array and return an owned C-order (row-major) Array2.
///
/// This combines transpose with layout normalization. Use this when you need
/// to flip axes (e.g., feature-major to sample-major) and want a contiguous
/// C-order result.
///
/// # Example
///
/// ```
/// use boosters::data::transpose_to_c_order;
/// use ndarray::Array2;
///
/// // Feature-major [n_features, n_samples]
/// let features = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
///
/// // Transpose to sample-major [n_samples, n_features] in C-order
/// let samples = transpose_to_c_order(features.view());
/// assert!(samples.is_standard_layout());
/// assert_eq!(samples.shape(), &[3, 2]);
/// ```
#[inline]
pub fn transpose_to_c_order<S>(arr: ArrayBase<S, Ix2>) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
    arr.t().as_standard_layout().into_owned()
}

/// Semantic axis constants for ML domain.
pub mod axis {
    use ndarray::Axis;

    pub const ROWS: Axis = Axis(0);
    pub const COLS: Axis = Axis(1);
}

// =============================================================================
// SamplesView (Sample-Major Layout)
// =============================================================================

/// View into a feature matrix with sample-major layout.
///
/// Shape: `[n_samples, n_features]` - each sample's features are contiguous in memory.
///
/// This is the standard layout for inference where we iterate over samples.
/// Compatible with numpy's default C-order arrays.
///
/// # Example
///
/// ```
/// use boosters::data::SamplesView;
///
/// // 3 samples, 2 features each
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let view = SamplesView::from_slice(&data, 3, 2).unwrap();
///
/// assert_eq!(view.n_samples(), 3);
/// assert_eq!(view.n_features(), 2);
/// assert_eq!(view.get(0, 1), 2.0); // sample 0, feature 1
/// assert_eq!(view.get(2, 0), 5.0); // sample 2, feature 0
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SamplesView<'a>(ArrayView2<'a, f32>);

impl<'a> SamplesView<'a> {
    /// Create from a slice in sample-major (row-major) order.
    ///
    /// This is zero-copy.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of length `n_samples * n_features`
    /// * `n_samples` - Number of samples (rows)
    /// * `n_features` - Number of features (columns)
    pub fn from_slice(
        data: &'a [f32],
        n_samples: usize,
        n_features: usize,
    ) -> Option<Self> {
        ArrayView2::from_shape((n_samples, n_features), data)
            .ok()
            .map(Self)
    }

    /// Create from an ArrayView2 that is sample-major.
    ///
    /// The array should have shape `[n_samples, n_features]` and be in C-order.
    ///
    /// # Panics (debug only)
    ///
    /// Debug-asserts that the array is in standard (C) layout.
    pub fn from_array(view: ArrayView2<'a, f32>) -> Self {
        debug_assert!(view.is_standard_layout(), "Array must be in C-order");
        Self(view)
    }

    /// Number of samples (rows).
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.0.nrows()
    }

    /// Number of features (columns).
    #[inline]
    pub fn n_features(&self) -> usize {
        self.0.ncols()
    }

    /// Get feature value at (sample, feature).
    #[inline]
    pub fn get(&self, sample: usize, feature: usize) -> f32 {
        self.0[[sample, feature]]
    }

    /// Get sample row as a contiguous slice.
    ///
    /// This is the fast path for sample-major data.
    #[inline]
    pub fn sample(&self, sample: usize) -> ArrayView1<'_, f32> {
        self.0.row(sample)
    }

    /// Get the underlying array view.
    pub fn view(&self) -> ArrayView2<'a, f32> {
        self.0
    }
}

impl<'a> Deref for SamplesView<'a> {
    type Target = ArrayView2<'a, f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FeatureAccessor for SamplesView<'a> {
    #[inline]
    fn get_feature(&self, row: usize, feature: usize) -> f32 {
        self.0[[row, feature]]
    }

    #[inline]
    fn n_rows(&self) -> usize {
        self.0.nrows()
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.0.ncols()
    }
}

// =============================================================================
// FeaturesView (Feature-Major Layout)
// =============================================================================

/// View into a feature matrix with feature-major layout.
///
/// Shape: `[n_features, n_samples]` - each feature's values across all samples are contiguous.
///
/// This layout is optimal for training where we iterate over features
/// (e.g., histogram building, coordinate descent in GBLinear).
///
/// # Example
///
/// ```
/// use boosters::data::FeaturesView;
///
/// // 2 features, 3 samples
/// // Data is [f0_s0, f0_s1, f0_s2, f1_s0, f1_s1, f1_s2]
/// let data = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
/// let view = FeaturesView::from_slice(&data, 3, 2).unwrap();
///
/// assert_eq!(view.n_samples(), 3);
/// assert_eq!(view.n_features(), 2);
/// assert_eq!(view.get(0, 0), 1.0);  // sample 0, feature 0
/// assert_eq!(view.get(0, 1), 10.0); // sample 0, feature 1
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FeaturesView<'a>(ArrayView2<'a, f32>);

impl<'a> FeaturesView<'a> {
    /// Create from an ArrayView2 that is feature-major.
    ///
    /// The array should have shape `[n_features, n_samples]` and be in C-order.
    ///
    /// # Panics (debug only)
    ///
    /// Debug-asserts that the array is in standard (C) layout.
    pub fn new(view: ArrayView2<'a, f32>) -> Self {
        debug_assert!(view.is_standard_layout(), "Array must be in C-order");
        Self(view)
    }

    /// Create from a slice in feature-major order.
    ///
    /// This is zero-copy.
    ///
    /// Data layout: `[f0_s0, f0_s1, ..., f1_s0, f1_s1, ...]`
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of length `n_samples * n_features`
    /// * `n_samples` - Number of samples
    /// * `n_features` - Number of features
    pub fn from_slice(
        data: &'a [f32],
        n_samples: usize,
        n_features: usize,
    ) -> Option<Self> {
        // Shape is [n_features, n_samples] for feature-major
        ArrayView2::from_shape((n_features, n_samples), data)
            .ok()
            .map(Self)
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
    pub fn feature(&self, feature: usize) -> ArrayView1<'_, f32> {
        self.0.row(feature)
    }

    /// Get the underlying array view.
    pub fn as_array(&self) -> ArrayView2<'a, f32> {
        self.0
    }
}

impl<'a> Deref for FeaturesView<'a> {
    type Target = ArrayView2<'a, f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FeatureAccessor for FeaturesView<'a> {
    #[inline]
    fn get_feature(&self, row: usize, feature: usize) -> f32 {
        // Note: transposed access for feature-major layout
        self.0[[feature, row]]
    }

    #[inline]
    fn n_rows(&self) -> usize {
        self.0.ncols() // n_samples
    }

    #[inline]
    fn n_features(&self) -> usize {
        self.0.nrows() // n_features
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
    use ndarray::arr2;

    #[test]
    fn test_samples_view_creation() {
        // 2 samples, 3 features: [[1,2,3], [4,5,6]]
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = SamplesView::from_slice(&data, 2, 3).unwrap();

        assert_eq!(view.n_samples(), 2);
        assert_eq!(view.n_features(), 3);
        assert_eq!(
            view.view(),
            arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        );
    }

    #[test]
    fn test_features_view_creation() {
        // 2 features, 3 samples
        // Feature-major: [f0_s0, f0_s1, f0_s2, f1_s0, f1_s1, f1_s2]
        let data = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let view = FeaturesView::from_slice(&data, 3, 2).unwrap();

        assert_eq!(view.n_samples(), 3);
        assert_eq!(view.n_features(), 2);
        // Access via (sample, feature) coordinates
        assert_eq!(view.get(0, 0), 1.0);  // sample 0, feature 0
        assert_eq!(view.get(1, 0), 2.0);  // sample 1, feature 0
        assert_eq!(view.get(0, 1), 10.0); // sample 0, feature 1
        assert_eq!(view.get(2, 1), 30.0); // sample 2, feature 1
    }

    #[test]
    fn test_features_view_feature_values() {
        let data = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let view = FeaturesView::from_slice(&data, 3, 2).unwrap();

        // Feature 0 values (all samples)
        assert_eq!(view.feature(0).as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        // Feature 1 values (all samples)
        assert_eq!(view.feature(1).as_slice().unwrap(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_samples_view_sample_values() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = SamplesView::from_slice(&data, 2, 3).unwrap();

        // Sample 0 features
        assert_eq!(view.sample(0).as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        // Sample 1 features
        assert_eq!(view.sample(1).as_slice().unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_init_predictions() {
        let base_scores = vec![0.5, -0.3];
        let preds = init_predictions(&base_scores, 3);

        // Direct comparison with expected array
        let expected = arr2(&[
            [0.5, 0.5, 0.5],    // group 0
            [-0.3, -0.3, -0.3], // group 1
        ]);
        assert_eq!(preds, expected);
    }

    #[test]
    fn test_init_predictions_vec() {
        let base_scores = vec![0.5, -0.3];
        let preds = init_predictions_vec(&base_scores, 3);

        // Layout: [g0_s0, g0_s1, g0_s2, g1_s0, g1_s1, g1_s2]
        assert_eq!(preds, vec![0.5, 0.5, 0.5, -0.3, -0.3, -0.3]);
    }

    #[test]
    fn test_from_array_view() {
        let arr = arr2(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let view = SamplesView::from_array(arr.view());

        assert_eq!(view.n_samples(), 3);
        assert_eq!(view.n_features(), 2);
        assert_eq!(view.view(), arr.view());
    }

    #[test]
    fn test_transpose_to_c_order() {
        // Feature-major [2 features, 3 samples]
        let features = arr2(&[
            [1.0, 2.0, 3.0],   // feature 0
            [10.0, 20.0, 30.0], // feature 1
        ]);

        // Transpose to sample-major [3 samples, 2 features]
        let samples = transpose_to_c_order(features.view());

        assert!(samples.is_standard_layout());
        let expected = arr2(&[
            [1.0, 10.0],  // sample 0
            [2.0, 20.0],  // sample 1
            [3.0, 30.0],  // sample 2
        ]);
        assert_eq!(samples, expected);
    }
}
