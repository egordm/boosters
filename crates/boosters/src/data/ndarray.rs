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

use ndarray::{Array2, ArrayBase, Data, Ix2};

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
    init_predictions_into(base_scores, &mut predictions);
    predictions
}

/// Initialize predictions in-place with base scores.
///
/// Fills each row of `predictions` with the corresponding base score.
/// Does not allocate; writes to the provided buffer.
///
/// # Arguments
///
/// * `base_scores` - One score per output/group
/// * `predictions` - Output buffer with shape `[n_groups, n_samples]`
///
/// # Panics
///
/// Panics if `base_scores.len() != predictions.nrows()`.
///
/// # Example
///
/// ```
/// use boosters::data::init_predictions_into;
/// use ndarray::Array2;
///
/// let base_scores = vec![0.5, -0.3];
/// let mut preds = Array2::zeros((2, 100));
/// init_predictions_into(&base_scores, &mut preds);
///
/// assert_eq!(preds[[0, 0]], 0.5);
/// assert_eq!(preds[[1, 0]], -0.3);
/// ```
pub fn init_predictions_into(base_scores: &[f32], predictions: &mut Array2<f32>) {
    debug_assert_eq!(base_scores.len(), predictions.nrows());
    for (group, &base_score) in base_scores.iter().enumerate() {
        predictions.row_mut(group).fill(base_score);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::SamplesView;
    use ndarray::{arr2, array};

    #[test]
    fn test_samples_view_creation() {
        // 2 samples, 3 features (sample-major layout)
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let view = SamplesView::from_array(data.view());

        assert_eq!(view.n_samples(), 2);
        assert_eq!(view.n_features(), 3);
        assert_eq!(view.view(), arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]));
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
            [1.0, 2.0, 3.0],    // feature 0
            [10.0, 20.0, 30.0], // feature 1
        ]);

        // Transpose to sample-major [3 samples, 2 features]
        let samples = transpose_to_c_order(features.view());

        assert!(samples.is_standard_layout());
        let expected = arr2(&[
            [1.0, 10.0], // sample 0
            [2.0, 20.0], // sample 1
            [3.0, 30.0], // sample 2
        ]);
        assert_eq!(samples, expected);
    }
}
