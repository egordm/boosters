//! Regression loss functions.
//!
//! This module provides loss functions for regression tasks:
//!
//! - [`SquaredLoss`]: Standard squared error (L2 loss)
//! - [`PseudoHuberLoss`]: Robust regression, smooth approximation of Huber loss
//! - [`QuantileLoss`]: Quantile regression (pinball loss), supports multi-quantile

// Allow range loops when we need indices to access multiple arrays.
#![allow(clippy::needless_range_loop)]

use super::{GradientBuffer, Loss, MulticlassLoss};

// =============================================================================
// Squared Error Loss
// =============================================================================

/// Squared error loss: L = 0.5 * (pred - label)²
///
/// Derivatives:
/// - grad = pred - label
/// - hess = 1
///
/// Used for regression tasks.
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredLoss;

impl Loss for SquaredLoss {
    /// Compute gradients for squared error loss.
    ///
    /// - grad = pred - label
    /// - hess = 1 (constant)
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), buffer.n_samples());
        debug_assert_eq!(buffer.n_outputs(), 1);

        let (grads, hess) = buffer.as_mut_slices();

        // Vectorizable loop: grad = pred - label
        for i in 0..preds.len() {
            grads[i] = preds[i] - labels[i];
        }

        // Fill hessians with constant (may use memset internally)
        hess.fill(1.0);
    }

    fn name(&self) -> &'static str {
        "squared_error"
    }
}

// =============================================================================
// Quantile Loss
// =============================================================================

/// Quantile (pinball) loss for quantile regression.
///
/// Supports both single-quantile and multi-quantile regression. For multiple
/// quantiles, trains a model with K outputs where each output corresponds to
/// a different quantile level.
///
/// For quantile α ∈ (0, 1):
/// - L = α(y - ŷ) if y ≥ ŷ (under-prediction)
/// - L = (1-α)(ŷ - y) if y < ŷ (over-prediction)
///
/// Derivatives:
/// - grad = (1 - α) if pred ≥ label (over-prediction penalty)
/// - grad = -α if pred < label (under-prediction penalty)
/// - hess = 1 (constant; pinball loss is piecewise linear)
///
/// Common quantiles:
/// - α = 0.5: Median regression (symmetric loss)
/// - α = 0.1: 10th percentile (penalizes over-prediction more)
/// - α = 0.9: 90th percentile (penalizes under-prediction more)
///
/// # Single Quantile
///
/// ```
/// use booste_rs::training::QuantileLoss;
///
/// // Median regression (α = 0.5)
/// let loss = QuantileLoss::new(0.5);
/// assert_eq!(loss.num_quantiles(), 1);
/// ```
///
/// # Multiple Quantiles
///
/// ```
/// use booste_rs::training::QuantileLoss;
///
/// // Predict 10th, 50th, and 90th percentiles simultaneously
/// let loss = QuantileLoss::multi(&[0.1, 0.5, 0.9]);
/// assert_eq!(loss.num_quantiles(), 3);
/// assert_eq!(loss.alphas(), &[0.1, 0.5, 0.9]);
/// ```
///
/// # XGBoost Compatibility
///
/// This matches XGBoost's `reg:quantileerror` with `quantile_alpha=[...]`.
#[derive(Debug, Clone)]
pub struct QuantileLoss {
    /// Quantile levels (each in (0, 1)).
    alphas: Vec<f32>,
}

impl QuantileLoss {
    /// Create a single-quantile loss.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Quantile level in (0, 1). Use 0.5 for median regression.
    ///
    /// # Panics
    ///
    /// Panics if alpha is not in (0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use booste_rs::training::QuantileLoss;
    ///
    /// let loss = QuantileLoss::new(0.5); // Median regression
    /// ```
    pub fn new(alpha: f32) -> Self {
        Self::multi(&[alpha])
    }

    /// Create a multi-quantile loss for predicting multiple quantiles simultaneously.
    ///
    /// # Arguments
    ///
    /// * `alphas` - Quantile levels, each in (0, 1). Common: `[0.1, 0.5, 0.9]`
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `alphas` is empty
    /// - Any alpha is not in (0, 1)
    ///
    /// # Example
    ///
    /// ```
    /// use booste_rs::training::QuantileLoss;
    ///
    /// let loss = QuantileLoss::multi(&[0.1, 0.5, 0.9]);
    /// assert_eq!(loss.num_quantiles(), 3);
    /// ```
    pub fn multi(alphas: &[f32]) -> Self {
        assert!(!alphas.is_empty(), "At least one quantile level required");
        for &alpha in alphas {
            assert!(
                alpha > 0.0 && alpha < 1.0,
                "Quantile alpha must be in (0, 1), got {}",
                alpha
            );
        }
        Self {
            alphas: alphas.to_vec(),
        }
    }

    /// Returns the number of quantiles.
    pub fn num_quantiles(&self) -> usize {
        self.alphas.len()
    }

    /// Returns the quantile levels.
    pub fn alphas(&self) -> &[f32] {
        &self.alphas
    }

    /// Returns the first (or only) quantile level.
    ///
    /// Convenience method for single-quantile losses.
    pub fn alpha(&self) -> f32 {
        self.alphas[0]
    }

    /// Returns true if this is a single-quantile loss.
    pub fn is_single(&self) -> bool {
        self.alphas.len() == 1
    }
}

impl Loss for QuantileLoss {
    /// Compute gradients for single-quantile loss.
    ///
    /// - grad = (1 - α) if pred >= label (over-prediction)
    /// - grad = -α if pred < label (under-prediction)
    /// - hess = 1 (constant for pinball loss)
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), buffer.n_samples());
        debug_assert_eq!(buffer.n_outputs(), 1);

        let alpha = self.alphas[0];
        let grad_over = 1.0 - alpha; // Precompute for over-prediction
        let grad_under = -alpha; // Precompute for under-prediction

        let (grads, hess) = buffer.as_mut_slices();

        // Vectorizable loop with precomputed constants
        for i in 0..preds.len() {
            grads[i] = if preds[i] >= labels[i] {
                grad_over
            } else {
                grad_under
            };
        }

        // Fill hessians with constant
        hess.fill(1.0);
    }

    fn name(&self) -> &'static str {
        "quantile"
    }
}

impl MulticlassLoss for QuantileLoss {
    fn num_classes(&self) -> usize {
        // "classes" here means "outputs" — one per quantile
        self.alphas.len()
    }

    /// Compute gradients for all samples and all quantiles.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions, layout: `preds[sample * num_quantiles + quantile]`
    /// * `labels` - Continuous target values (NOT class indices)
    /// * `buffer` - Output buffer with `n_samples` samples and `n_outputs == num_quantiles`
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        let num_quantiles = self.alphas.len();
        let num_samples = labels.len();
        debug_assert_eq!(preds.len(), num_samples * num_quantiles);
        debug_assert_eq!(buffer.n_samples(), num_samples);
        debug_assert_eq!(buffer.n_outputs(), num_quantiles);

        let (grads, hess) = buffer.as_mut_slices();

        for i in 0..num_samples {
            let label = labels[i];
            for q in 0..num_quantiles {
                let idx = i * num_quantiles + q;
                let pred = preds[idx];
                let alpha = self.alphas[q];

                // Pinball loss gradient:
                // - If pred >= label (over-prediction): grad = 1 - alpha
                // - If pred < label (under-prediction): grad = -alpha
                grads[idx] = if pred >= label { 1.0 - alpha } else { -alpha };

                // Hessian is 1 for pinball loss
                hess[idx] = 1.0;
            }
        }
    }

    fn name(&self) -> &'static str {
        "quantile"
    }
}

// =============================================================================
// Pseudo-Huber Loss
// =============================================================================

/// Pseudo-Huber loss for robust regression.
///
/// A smooth approximation of Huber loss that is differentiable everywhere.
/// Less sensitive to outliers than squared error.
///
/// For residual r = pred - label and slope δ:
/// - L = δ² × (√(1 + (r/δ)²) - 1)
///
/// Derivatives:
/// - grad = r / √(1 + r²/δ²)
/// - hess = δ² / ((δ² + r²) × √(1 + r²/δ²))
///
/// # XGBoost Compatibility
///
/// This matches XGBoost's `reg:pseudohubererror` with `huber_slope` parameter.
#[derive(Debug, Clone, Copy)]
pub struct PseudoHuberLoss {
    /// The slope (delta) parameter controlling the transition point.
    /// Larger values make the loss behave more like squared error.
    /// Smaller values make it more robust to outliers.
    slope: f32,
}

impl PseudoHuberLoss {
    /// Create a new Pseudo-Huber loss with the given slope (delta).
    ///
    /// # Arguments
    ///
    /// * `slope` - The delta parameter. Default in XGBoost is 1.0.
    ///
    /// # Panics
    ///
    /// Panics if slope is zero or negative.
    pub fn new(slope: f32) -> Self {
        assert!(
            slope > 0.0,
            "Pseudo-Huber slope must be positive, got {}",
            slope
        );
        Self { slope }
    }

    /// Returns the slope (delta) parameter.
    pub fn slope(&self) -> f32 {
        self.slope
    }
}

impl Default for PseudoHuberLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Loss for PseudoHuberLoss {
    /// Compute gradients for Pseudo-Huber loss.
    ///
    /// - grad = r / sqrt(1 + r²/δ²)
    /// - hess = δ² / ((δ² + r²) × sqrt(1 + r²/δ²))
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), buffer.n_samples());
        debug_assert_eq!(buffer.n_outputs(), 1);

        let slope = self.slope;
        let slope_sq = slope * slope;

        let (grads, hess) = buffer.as_mut_slices();

        for i in 0..preds.len() {
            let r = preds[i] - labels[i];
            let r_sq = r * r;

            // scale_sqrt = sqrt(1 + r²/δ²)
            let scale_sqrt = (1.0 + r_sq / slope_sq).sqrt();

            grads[i] = r / scale_sqrt;

            // hess = δ² / ((δ² + r²) × scale_sqrt)
            let scale = slope_sq + r_sq;
            hess[i] = slope_sq / (scale * scale_sqrt);
        }
    }

    fn name(&self) -> &'static str {
        "pseudo_huber"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SquaredLoss tests
    // =========================================================================

    #[test]
    fn squared_loss_gradient() {
        let loss = SquaredLoss;
        let preds = vec![1.0, 2.0];
        let labels = vec![0.5, 2.0];
        let mut buffer = GradientBuffer::new(2, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // pred=1.0, label=0.5 → grad = 0.5, hess = 1.0
        assert!((buffer.grad(0, 0) - 0.5).abs() < 1e-6);
        assert!((buffer.hess(0, 0) - 1.0).abs() < 1e-6);

        // pred = label → grad = 0
        assert!(buffer.grad(1, 0).abs() < 1e-6);
    }

    #[test]
    fn squared_loss_batch() {
        let loss = SquaredLoss;
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![0.5, 2.0, 2.5];
        let mut buffer = GradientBuffer::new(3, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // Check gradients match expected values
        assert!((buffer.grad(0, 0) - 0.5).abs() < 1e-6); // 1.0 - 0.5
        assert!((buffer.grad(1, 0) - 0.0).abs() < 1e-6); // 2.0 - 2.0
        assert!((buffer.grad(2, 0) - 0.5).abs() < 1e-6); // 3.0 - 2.5

        // All hessians should be 1.0
        assert!((buffer.hess(0, 0) - 1.0).abs() < 1e-6);
        assert!((buffer.hess(1, 0) - 1.0).abs() < 1e-6);
        assert!((buffer.hess(2, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn squared_loss_name() {
        assert_eq!(Loss::name(&SquaredLoss), "squared_error");
    }

    // =========================================================================
    // QuantileLoss tests (single quantile via Loss trait)
    // =========================================================================

    #[test]
    fn quantile_loss_single() {
        let loss = QuantileLoss::new(0.5);
        let preds = vec![1.0, 3.0, 2.0]; // under, over, exact
        let labels = vec![2.0, 2.0, 2.0];
        let mut buffer = GradientBuffer::new(3, 1);

        Loss::compute_gradients(&loss, &preds, &labels, &mut buffer);

        // Under-prediction: grad = -0.5
        assert!((buffer.grad(0, 0) - (-0.5)).abs() < 1e-6);
        // Over-prediction: grad = 0.5
        assert!((buffer.grad(1, 0) - 0.5).abs() < 1e-6);
        // Exact: grad = 0.5 (convention: pred >= label is over)
        assert!((buffer.grad(2, 0) - 0.5).abs() < 1e-6);

        // All hessians should be 1.0
        for i in 0..3 {
            assert!((buffer.hess(i, 0) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn quantile_loss_asymmetric() {
        // α = 0.1: penalize over-prediction more (want low quantile)
        let loss = QuantileLoss::new(0.1);
        let preds = vec![1.0, 3.0]; // under, over
        let labels = vec![2.0, 2.0];
        let mut buffer = GradientBuffer::new(2, 1);

        Loss::compute_gradients(&loss, &preds, &labels, &mut buffer);

        // Under-prediction: grad = -α = -0.1
        assert!((buffer.grad(0, 0) - (-0.1)).abs() < 1e-6);
        // Over-prediction: grad = 1-α = 0.9
        assert!((buffer.grad(1, 0) - 0.9).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Quantile alpha must be in (0, 1)")]
    fn quantile_loss_invalid_alpha_zero() {
        QuantileLoss::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Quantile alpha must be in (0, 1)")]
    fn quantile_loss_invalid_alpha_one() {
        QuantileLoss::new(1.0);
    }

    #[test]
    fn quantile_loss_name() {
        assert_eq!(Loss::name(&QuantileLoss::new(0.5)), "quantile");
    }

    // =========================================================================
    // Multi-Quantile Loss tests (using QuantileLoss::multi via MulticlassLoss)
    // =========================================================================

    #[test]
    fn multi_quantile_loss_creation() {
        let loss = QuantileLoss::multi(&[0.1, 0.5, 0.9]);
        assert_eq!(loss.num_quantiles(), 3);
        assert_eq!(loss.alphas(), &[0.1, 0.5, 0.9]);
        assert_eq!(loss.num_classes(), 3); // MulticlassLoss trait
        assert_eq!(MulticlassLoss::name(&loss), "quantile");
    }

    #[test]
    #[should_panic(expected = "At least one quantile level required")]
    fn multi_quantile_loss_empty() {
        QuantileLoss::multi(&[]);
    }

    #[test]
    #[should_panic(expected = "Quantile alpha must be in (0, 1)")]
    fn multi_quantile_loss_invalid_alpha() {
        QuantileLoss::multi(&[0.1, 1.0, 0.9]);
    }

    #[test]
    fn multi_quantile_loss_batch() {
        // 3 quantiles: 0.1, 0.5, 0.9
        let loss = QuantileLoss::multi(&[0.1, 0.5, 0.9]);
        let num_quantiles = 3;

        // 2 samples, predictions for each quantile
        // Sample 0: preds = [1.0, 2.0, 3.0], label = 2.0
        //   q=0 (α=0.1): pred=1 < label=2 → grad = -0.1
        //   q=1 (α=0.5): pred=2 >= label=2 → grad = 0.5
        //   q=2 (α=0.9): pred=3 >= label=2 → grad = 0.1
        // Sample 1: preds = [3.0, 2.0, 1.0], label = 2.0
        //   q=0 (α=0.1): pred=3 >= label=2 → grad = 0.9
        //   q=1 (α=0.5): pred=2 >= label=2 → grad = 0.5
        //   q=2 (α=0.9): pred=1 < label=2 → grad = -0.9
        let preds = vec![
            1.0, 2.0, 3.0, // sample 0
            3.0, 2.0, 1.0, // sample 1
        ];
        let labels = vec![2.0, 2.0];
        let mut buffer = GradientBuffer::new(2, num_quantiles);

        MulticlassLoss::compute_gradients(&loss, &preds, &labels, &mut buffer);

        // Sample 0
        assert!((buffer.grad(0, 0) - (-0.1)).abs() < 1e-6); // under, α=0.1
        assert!((buffer.grad(0, 1) - 0.5).abs() < 1e-6); // exact, α=0.5
        assert!((buffer.grad(0, 2) - 0.1).abs() < 1e-6); // over, α=0.9

        // Sample 1
        assert!((buffer.grad(1, 0) - 0.9).abs() < 1e-6); // over, α=0.1
        assert!((buffer.grad(1, 1) - 0.5).abs() < 1e-6); // exact, α=0.5
        assert!((buffer.grad(1, 2) - (-0.9)).abs() < 1e-6); // under, α=0.9

        // All hessians should be 1.0
        for sample in 0..2 {
            for q in 0..num_quantiles {
                assert!((buffer.hess(sample, q) - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn multi_quantile_loss_single_quantile() {
        // With a single quantile, should behave like QuantileLoss::new
        let multi_loss = QuantileLoss::multi(&[0.25]);
        let single_loss = QuantileLoss::new(0.25);

        let preds = vec![1.0, 3.0, 2.0]; // 3 samples
        let labels = vec![2.0, 2.0, 2.0];

        let mut buffer_multi = GradientBuffer::new(3, 1);
        let mut buffer_single = GradientBuffer::new(3, 1);

        MulticlassLoss::compute_gradients(&multi_loss, &preds, &labels, &mut buffer_multi);
        Loss::compute_gradients(&single_loss, &preds, &labels, &mut buffer_single);

        // Results should be identical
        for i in 0..3 {
            assert!(
                (buffer_multi.grad(i, 0) - buffer_single.grad(i, 0)).abs() < 1e-6,
                "Sample {}: multi={}, single={}",
                i,
                buffer_multi.grad(i, 0),
                buffer_single.grad(i, 0)
            );
        }
    }

    #[test]
    fn multi_quantile_loss_extreme_predictions() {
        let loss = QuantileLoss::multi(&[0.1, 0.5, 0.9]);

        // All predictions well above label
        let preds = vec![10.0, 10.0, 10.0]; // label = 0.0
        let labels = vec![0.0];
        let mut buffer = GradientBuffer::new(1, 3);

        MulticlassLoss::compute_gradients(&loss, &preds, &labels, &mut buffer);

        // All over-predictions
        assert!((buffer.grad(0, 0) - 0.9).abs() < 1e-6); // 1 - 0.1
        assert!((buffer.grad(0, 1) - 0.5).abs() < 1e-6); // 1 - 0.5
        assert!((buffer.grad(0, 2) - 0.1).abs() < 1e-6); // 1 - 0.9

        // All predictions well below label
        let preds = vec![-10.0, -10.0, -10.0]; // label = 0.0
        MulticlassLoss::compute_gradients(&loss, &preds, &labels, &mut buffer);

        // All under-predictions
        assert!((buffer.grad(0, 0) - (-0.1)).abs() < 1e-6); // -0.1
        assert!((buffer.grad(0, 1) - (-0.5)).abs() < 1e-6); // -0.5
        assert!((buffer.grad(0, 2) - (-0.9)).abs() < 1e-6); // -0.9
    }

    // =========================================================================
    // PseudoHuberLoss tests
    // =========================================================================

    #[test]
    fn pseudo_huber_loss_gradient() {
        let loss = PseudoHuberLoss::default(); // slope = 1.0
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.0, 1.0, 1.0]; // residuals: 0, 1, 2
        let mut buffer = GradientBuffer::new(3, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // r=0: grad = 0 / sqrt(1 + 0) = 0
        assert!(buffer.grad(0, 0).abs() < 1e-6);

        // r=1, δ=1: grad = 1 / sqrt(1 + 1) = 1/√2 ≈ 0.707
        let expected_grad_1 = 1.0 / 2.0_f32.sqrt();
        assert!((buffer.grad(1, 0) - expected_grad_1).abs() < 1e-5);

        // r=2, δ=1: grad = 2 / sqrt(1 + 4) = 2/√5 ≈ 0.894
        let expected_grad_2 = 2.0 / 5.0_f32.sqrt();
        assert!((buffer.grad(2, 0) - expected_grad_2).abs() < 1e-5);

        // All hessians should be positive
        for i in 0..3 {
            assert!(buffer.hess(i, 0) > 0.0);
        }
    }

    #[test]
    fn pseudo_huber_loss_with_slope() {
        let loss = PseudoHuberLoss::new(2.0); // slope = 2.0
        let preds = vec![3.0]; // r = 3 - 1 = 2
        let labels = vec![1.0];
        let mut buffer = GradientBuffer::new(1, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // r=2, δ=2: grad = 2 / sqrt(1 + 4/4) = 2/√2 ≈ 1.414
        let expected_grad = 2.0 / 2.0_f32.sqrt();
        assert!((buffer.grad(0, 0) - expected_grad).abs() < 1e-5);

        // hess = δ² / ((δ² + r²) × sqrt(1 + r²/δ²))
        // = 4 / ((4 + 4) × √2) = 4 / (8 × 1.414) ≈ 0.354
        let scale_sqrt = 2.0_f32.sqrt();
        let expected_hess = 4.0 / (8.0 * scale_sqrt);
        assert!((buffer.hess(0, 0) - expected_hess).abs() < 1e-5);
    }

    #[test]
    fn pseudo_huber_loss_negative_residual() {
        let loss = PseudoHuberLoss::default();
        let preds = vec![0.0]; // r = 0 - 2 = -2
        let labels = vec![2.0];
        let mut buffer = GradientBuffer::new(1, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // r=-2, δ=1: grad = -2 / sqrt(1 + 4) = -2/√5 ≈ -0.894
        let expected_grad = -2.0 / 5.0_f32.sqrt();
        assert!((buffer.grad(0, 0) - expected_grad).abs() < 1e-5);
    }

    #[test]
    fn pseudo_huber_loss_vs_squared_for_small_residuals() {
        // For small residuals, Pseudo-Huber should behave like squared loss
        let pseudo_huber = PseudoHuberLoss::new(10.0); // large slope → more like squared
        let squared = SquaredLoss;

        let preds = vec![1.0, 1.1, 0.9]; // small residuals
        let labels = vec![1.0, 1.0, 1.0];

        let mut buffer_ph = GradientBuffer::new(3, 1);
        let mut buffer_sq = GradientBuffer::new(3, 1);

        pseudo_huber.compute_gradients(&preds, &labels, &mut buffer_ph);
        squared.compute_gradients(&preds, &labels, &mut buffer_sq);

        // Gradients should be similar for small residuals
        for i in 0..3 {
            let diff = (buffer_ph.grad(i, 0) - buffer_sq.grad(i, 0)).abs();
            assert!(
                diff < 0.01,
                "Sample {}: PH={}, SQ={}",
                i,
                buffer_ph.grad(i, 0),
                buffer_sq.grad(i, 0)
            );
        }
    }

    #[test]
    fn pseudo_huber_loss_robustness_to_outliers() {
        // For large residuals, gradient should be bounded
        let loss = PseudoHuberLoss::default();

        let preds = vec![100.0]; // huge outlier
        let labels = vec![0.0];
        let mut buffer = GradientBuffer::new(1, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // r=100, δ=1: grad = 100 / sqrt(1 + 10000) ≈ 100/100 = 1
        // Should be bounded, not 100 like squared loss would give
        let grad = buffer.grad(0, 0);
        assert!(grad < 2.0, "Gradient should be bounded, got {}", grad);
        assert!(grad > 0.0);
    }

    #[test]
    #[should_panic(expected = "Pseudo-Huber slope must be positive")]
    fn pseudo_huber_loss_invalid_slope_zero() {
        PseudoHuberLoss::new(0.0);
    }

    #[test]
    #[should_panic(expected = "Pseudo-Huber slope must be positive")]
    fn pseudo_huber_loss_invalid_slope_negative() {
        PseudoHuberLoss::new(-1.0);
    }

    #[test]
    fn pseudo_huber_loss_name() {
        assert_eq!(Loss::name(&PseudoHuberLoss::default()), "pseudo_huber");
    }
}
