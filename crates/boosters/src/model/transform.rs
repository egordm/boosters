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

/// Inference-time output transformation.
///
/// Models persist this instead of the full objective so that prediction
/// can work without knowing training configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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
    /// Apply the transformation in-place to a row-major predictions buffer.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Mutable slice of predictions, shape `(n_rows, n_outputs)` in row-major order.
    /// * `n_outputs` - Number of output columns (1 for regression/binary, n_classes for multiclass).
    ///
    /// # Numerical Stability
    ///
    /// - Sigmoid clamps input to [-500, 500] to avoid overflow.
    /// - Softmax subtracts the max per row before exponentiating.
    ///
    /// # Panics
    ///
    /// Panics if `predictions.len()` is not divisible by `n_outputs` or if `n_outputs` is 0.
    ///
    /// # NaN/Inf Behavior
    ///
    /// NaN and Inf inputs propagate through without panics (garbage-in, garbage-out).
    #[inline]
    pub fn transform_inplace(&self, predictions: &mut [f32], n_outputs: usize) {
        assert!(n_outputs > 0, "n_outputs must be > 0");
        assert!(
            predictions.len().is_multiple_of(n_outputs),
            "predictions.len() must be divisible by n_outputs"
        );

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
                let n_rows = predictions.len() / n_outputs;
                for row_idx in 0..n_rows {
                    let start = row_idx * n_outputs;
                    let end = start + n_outputs;
                    let row = &mut predictions[start..end];
                    softmax_inplace(row);
                }
            }
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

/// Numerically stable softmax in-place.
/// Subtracts max before exponentiating to avoid overflow.
#[inline]
fn softmax_inplace(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for x in row.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }

    // Normalize
    if sum > 0.0 {
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // =========================================================================
    // Identity tests
    // =========================================================================

    #[test]
    fn identity_is_noop() {
        let mut preds = vec![1.0, -2.0, 3.5, 0.0];
        let original = preds.clone();
        OutputTransform::Identity.transform_inplace(&mut preds, 1);
        assert_eq!(preds, original);
    }

    // =========================================================================
    // Sigmoid tests
    // =========================================================================

    #[test]
    fn sigmoid_zero_is_half() {
        let mut preds = vec![0.0];
        OutputTransform::Sigmoid.transform_inplace(&mut preds, 1);
        assert_abs_diff_eq!(preds[0], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn sigmoid_output_in_zero_one() {
        let mut preds = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        OutputTransform::Sigmoid.transform_inplace(&mut preds, 1);
        for &p in &preds {
            assert!(p > 0.0 && p < 1.0, "sigmoid output {} not in (0,1)", p);
        }
    }

    #[test]
    fn sigmoid_large_values_stable() {
        let mut preds = vec![-100.0, 100.0, -500.0, 500.0];
        OutputTransform::Sigmoid.transform_inplace(&mut preds, 1);

        // Very negative -> close to 0
        assert!(preds[0] < 0.001);
        assert!(preds[2] < 0.001);

        // Very positive -> close to 1
        assert!(preds[1] > 0.999);
        assert!(preds[3] > 0.999);
    }

    #[test]
    fn sigmoid_nan_propagates() {
        let mut preds = vec![f32::NAN];
        OutputTransform::Sigmoid.transform_inplace(&mut preds, 1);
        assert!(preds[0].is_nan());
    }

    #[test]
    fn sigmoid_inf_stable() {
        let mut preds = vec![f32::INFINITY, f32::NEG_INFINITY];
        OutputTransform::Sigmoid.transform_inplace(&mut preds, 1);
        // +inf clamped to 500 -> close to 1
        assert!(preds[0] > 0.999);
        // -inf clamped to -500 -> close to 0
        assert!(preds[1] < 0.001);
    }

    // =========================================================================
    // Softmax tests
    // =========================================================================

    #[test]
    fn softmax_sums_to_one() {
        let mut preds = vec![1.0, 2.0, 3.0];
        OutputTransform::Softmax.transform_inplace(&mut preds, 3);

        let sum: f32 = preds.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn softmax_preserves_order() {
        let mut preds = vec![1.0, 2.0, 3.0];
        OutputTransform::Softmax.transform_inplace(&mut preds, 3);

        assert!(preds[0] < preds[1]);
        assert!(preds[1] < preds[2]);
    }

    #[test]
    fn softmax_multiple_rows() {
        let mut preds = vec![
            1.0, 2.0, 3.0, // row 0
            0.0, 0.0, 0.0, // row 1 (uniform)
        ];
        OutputTransform::Softmax.transform_inplace(&mut preds, 3);

        // Row 0 sums to 1
        let sum0: f32 = preds[0..3].iter().sum();
        assert_abs_diff_eq!(sum0, 1.0, epsilon = 1e-6);

        // Row 1 sums to 1 and is uniform
        let sum1: f32 = preds[3..6].iter().sum();
        assert_abs_diff_eq!(sum1, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(preds[3], preds[4], epsilon = 1e-6);
        assert_abs_diff_eq!(preds[4], preds[5], epsilon = 1e-6);
    }

    #[test]
    fn softmax_large_values_stable() {
        let mut preds = vec![100.0, 200.0, 300.0];
        OutputTransform::Softmax.transform_inplace(&mut preds, 3);

        let sum: f32 = preds.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);

        // Largest input should dominate
        assert!(preds[2] > 0.99);
    }

    #[test]
    fn softmax_nan_propagates() {
        let mut preds = vec![1.0, f32::NAN, 2.0];
        OutputTransform::Softmax.transform_inplace(&mut preds, 3);
        // At least one output should be NaN
        assert!(preds.iter().any(|x| x.is_nan()));
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    #[should_panic(expected = "n_outputs must be > 0")]
    fn panics_on_zero_n_outputs() {
        let mut preds = vec![];
        OutputTransform::Identity.transform_inplace(&mut preds, 0);
    }

    #[test]
    #[should_panic(expected = "predictions.len() must be divisible by n_outputs")]
    fn panics_on_mismatched_length() {
        let mut preds = vec![1.0, 2.0, 3.0];
        OutputTransform::Sigmoid.transform_inplace(&mut preds, 2);
    }

    #[test]
    fn default_is_identity() {
        assert_eq!(OutputTransform::default(), OutputTransform::Identity);
    }
}
