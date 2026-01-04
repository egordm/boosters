//! Output transformation for inference.
//!
//! The [`OutputTransform`] enum defines how raw model outputs (margins)
//! are converted to final predictions. This is persisted with the model
//! so that inference doesn't require the original objective.
//!
//! # Variants
//!
//! - [`Identity`](OutputTransform::Identity): No transformation (regression, raw margins)
//! - [`Sigmoid`](OutputTransform::Sigmoid): Logistic sigmoid for binary classification
//! - [`Softmax`](OutputTransform::Softmax): Softmax for multiclass classification

use ndarray::{ArrayViewMut1, ArrayViewMut2, Axis};

/// Inference-time output transformation.
///
/// Models persist this instead of the full objective so that prediction
/// can work without knowing training configuration.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum OutputTransform {
    /// No transformation; output = margin.
    /// Used for regression and raw margin outputs.
    #[default]
    Identity,

    /// Logistic sigmoid: output = 1 / (1 + exp(-margin)).
    /// Used for binary classification (LogisticLoss).
    Sigmoid,

    /// Softmax: output_i = exp(margin_i) / sum(exp(margin_j)).
    /// Used for multiclass classification (SoftmaxLoss).
    Softmax,
}

impl OutputTransform {
    /// Apply the transformation in-place to an output-major predictions matrix.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Mutable view of predictions, shape `(n_outputs, n_samples)`.
    ///
    /// Layout semantics are **output-major**:
    /// `predictions[[output, sample]]`.
    ///
    /// # Numerical Stability
    ///
    /// - Sigmoid clamps input to [-500, 500] to avoid overflow.
    /// - Softmax subtracts the max per row before exponentiating.
    ///
    /// # Panics
    ///
    /// Panics if `n_outputs == 0`.
    ///
    /// # NaN/Inf Behavior
    ///
    /// NaN and Inf inputs propagate through without panics (garbage-in, garbage-out).
    #[inline]
    pub fn transform_inplace(&self, mut predictions: ArrayViewMut2<'_, f32>) {
        assert!(predictions.nrows() > 0, "n_outputs must be > 0");

        match self {
            OutputTransform::Identity => {
                // No-op
            }
            OutputTransform::Sigmoid => {
                for x in predictions.iter_mut() {
                    *x = sigmoid(*x);
                }
            }
            OutputTransform::Softmax => {
                for sample in predictions.axis_iter_mut(Axis(1)) {
                    softmax_inplace(sample);
                }
            }
        }
    }
}

#[inline]
fn softmax_inplace(mut scores: ArrayViewMut1<'_, f32>) {
    let mut max = f32::NEG_INFINITY;
    for &x in scores.iter() {
        max = max.max(x);
    }

    let mut sum = 0.0f32;
    for x in scores.iter_mut() {
        let e = (*x - max).exp();
        *x = e;
        sum += e;
    }

    if sum > 0.0 {
        for x in scores.iter_mut() {
            *x /= sum;
        }
    }
}

/// Numerically stable sigmoid.
/// Clamps input to [-500, 500] to prevent overflow.
#[inline]
fn sigmoid(x: f32) -> f32 {
    // Clamp to avoid overflow in exp
    let clamped = x.clamp(-500.0, 500.0);
    if clamped >= 0.0 {
        1.0 / (1.0 + (-clamped).exp())
    } else {
        let e = clamped.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    // =========================================================================
    // Identity tests
    // =========================================================================

    #[test]
    fn identity_is_noop() {
        let mut preds = Array2::from_shape_vec((1, 4), vec![1.0, -2.0, 3.5, 0.0]).unwrap();
        let original = preds.clone();
        OutputTransform::Identity.transform_inplace(preds.view_mut());
        assert_eq!(preds, original);
    }

    // =========================================================================
    // Sigmoid tests
    // =========================================================================

    #[test]
    fn sigmoid_zero_is_half() {
        let mut preds = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        OutputTransform::Sigmoid.transform_inplace(preds.view_mut());
        assert_abs_diff_eq!(preds[[0, 0]], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn sigmoid_output_in_zero_one() {
        let mut preds = Array2::from_shape_vec((1, 5), vec![-10.0, -1.0, 0.0, 1.0, 10.0]).unwrap();
        OutputTransform::Sigmoid.transform_inplace(preds.view_mut());
        for &p in preds.as_slice().unwrap() {
            assert!(p > 0.0 && p < 1.0, "sigmoid output {} not in (0,1)", p);
        }
    }

    #[test]
    fn sigmoid_large_values_stable() {
        let mut preds = Array2::from_shape_vec((1, 4), vec![-100.0, 100.0, -500.0, 500.0]).unwrap();
        OutputTransform::Sigmoid.transform_inplace(preds.view_mut());

        // Very negative -> close to 0
        assert!(preds[[0, 0]] < 0.001);
        assert!(preds[[0, 2]] < 0.001);

        // Very positive -> close to 1
        assert!(preds[[0, 1]] > 0.999);
        assert!(preds[[0, 3]] > 0.999);
    }

    #[test]
    fn sigmoid_nan_propagates() {
        let mut preds = Array2::from_shape_vec((1, 1), vec![f32::NAN]).unwrap();
        OutputTransform::Sigmoid.transform_inplace(preds.view_mut());
        assert!(preds[[0, 0]].is_nan());
    }

    #[test]
    fn sigmoid_inf_stable() {
        let mut preds =
            Array2::from_shape_vec((1, 2), vec![f32::INFINITY, f32::NEG_INFINITY]).unwrap();
        OutputTransform::Sigmoid.transform_inplace(preds.view_mut());
        // +inf clamped to 500 -> close to 1
        assert!(preds[[0, 0]] > 0.999);
        // -inf clamped to -500 -> close to 0
        assert!(preds[[0, 1]] < 0.001);
    }

    // =========================================================================
    // Softmax tests
    // =========================================================================

    #[test]
    fn softmax_sums_to_one() {
        // One row with 3 outputs.
        let mut preds = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        OutputTransform::Softmax.transform_inplace(preds.view_mut());

        let sum: f32 = (0..3).map(|o| preds[[o, 0]]).sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn softmax_preserves_order() {
        let mut preds = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        OutputTransform::Softmax.transform_inplace(preds.view_mut());

        assert!(preds[[0, 0]] < preds[[1, 0]]);
        assert!(preds[[1, 0]] < preds[[2, 0]]);
    }

    #[test]
    fn softmax_multiple_rows() {
        // Two rows, 3 outputs each.
        let mut preds = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        OutputTransform::Softmax.transform_inplace(preds.view_mut());

        // Row 0 sums to 1
        let sum0: f32 = (0..3).map(|o| preds[[o, 0]]).sum();
        assert_abs_diff_eq!(sum0, 1.0, epsilon = 1e-6);

        // Row 1 sums to 1 and is uniform
        let sum1: f32 = (0..3).map(|o| preds[[o, 1]]).sum();
        assert_abs_diff_eq!(sum1, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(preds[[0, 1]], preds[[1, 1]], epsilon = 1e-6);
        assert_abs_diff_eq!(preds[[1, 1]], preds[[2, 1]], epsilon = 1e-6);
    }

    #[test]
    fn softmax_large_values_stable() {
        let mut preds = Array2::from_shape_vec((3, 1), vec![100.0, 200.0, 300.0]).unwrap();
        OutputTransform::Softmax.transform_inplace(preds.view_mut());

        let sum: f32 = (0..3).map(|o| preds[[o, 0]]).sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        // Largest input should dominate
        assert!(preds[[2, 0]] > 0.99);
    }

    #[test]
    fn softmax_nan_propagates() {
        let mut preds = Array2::from_shape_vec((3, 1), vec![1.0, f32::NAN, 2.0]).unwrap();
        OutputTransform::Softmax.transform_inplace(preds.view_mut());
        // At least one output should be NaN
        assert!((0..3).any(|o| preds[[o, 0]].is_nan()));
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    #[should_panic(expected = "n_outputs must be > 0")]
    fn panics_on_zero_n_outputs() {
        let mut preds = Array2::<f32>::zeros((0, 0));
        OutputTransform::Identity.transform_inplace(preds.view_mut());
    }

    #[test]
    fn default_is_identity() {
        assert_eq!(OutputTransform::default(), OutputTransform::Identity);
    }
}
