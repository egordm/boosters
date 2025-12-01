//! Loss functions for computing gradients.
//!
//! Each loss function computes gradient-hessian pairs for optimization.
//! These are used by both GBLinear and GBTree training.
//!
//! # Loss Types
//!
//! - [`Loss`]: Single-output losses (regression, binary classification)
//! - [`MulticlassLoss`]: Multi-output losses requiring all class predictions
//!
//! # Design for GBTree Compatibility
//!
//! The gradient computation is designed to work with both linear and tree models:
//! - Gradients are stored per (sample, group) pair
//! - For regression/binary: 1 group, gradients[i] = gradient for sample i
//! - For multiclass: K groups, gradients[i * K + k] = gradient for sample i, class k

use super::GradientPair;

/// A loss function for single-output models (regression, binary classification).
///
/// For losses where each sample has one prediction and one gradient.
/// Examples: squared error, logistic loss, quantile loss.
///
/// # Implementation Notes
///
/// - `compute_gradient`: Called per sample, returns (grad, hess) pair
/// - `gradient_batch`: Computes gradients for entire batch (can be parallelized)
pub trait Loss: Send + Sync {
    /// Compute gradient and hessian for a single sample.
    ///
    /// # Arguments
    ///
    /// * `pred` - Model prediction (raw score before any transformation)
    /// * `label` - Ground truth label
    ///
    /// # Returns
    ///
    /// Gradient pair (grad, hess) where:
    /// - grad = ∂L/∂pred (first derivative)
    /// - hess = ∂²L/∂pred² (second derivative)
    fn compute_gradient(&self, pred: f32, label: f32) -> GradientPair;

    /// Compute gradients for a batch of samples.
    ///
    /// Default implementation calls `compute_gradient` for each sample.
    /// Can be overridden for vectorized implementations.
    fn gradient_batch(&self, preds: &[f32], labels: &[f32], out: &mut [GradientPair]) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), out.len());

        for ((pred, label), gp) in preds.iter().zip(labels.iter()).zip(out.iter_mut()) {
            *gp = self.compute_gradient(*pred, *label);
        }
    }

    /// Name of the loss function (for logging).
    fn name(&self) -> &'static str;
}

/// A loss function for multiclass classification.
///
/// Unlike [`Loss`], this requires all class predictions simultaneously to compute
/// softmax probabilities. Each sample produces K gradients (one per class).
///
/// # Gradient Layout
///
/// For N samples and K classes, gradients are stored as:
/// `gradients[sample_idx * num_classes + class_idx]`
///
/// This layout matches XGBoost and allows efficient per-group weight updates.
pub trait MulticlassLoss: Send + Sync {
    /// Number of classes.
    fn num_classes(&self) -> usize;

    /// Compute gradients for one sample across all classes.
    ///
    /// # Arguments
    ///
    /// * `preds` - Raw predictions for all classes (length = num_classes)
    /// * `label` - True class index (0-based, must be < num_classes)
    /// * `out` - Output gradient pairs for each class (length = num_classes)
    fn compute_gradient(&self, preds: &[f32], label: usize, out: &mut [GradientPair]);

    /// Compute gradients for a batch of samples.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions, layout: `preds[sample * num_classes + class]`
    /// * `labels` - Class labels (0-based indices)
    /// * `out` - Output gradients, same layout as preds
    fn gradient_batch(&self, preds: &[f32], labels: &[f32], out: &mut [GradientPair]) {
        let num_classes = self.num_classes();
        let num_samples = labels.len();
        debug_assert_eq!(preds.len(), num_samples * num_classes);
        debug_assert_eq!(out.len(), num_samples * num_classes);

        for i in 0..num_samples {
            let start = i * num_classes;
            let end = start + num_classes;
            let label = labels[i] as usize;
            self.compute_gradient(&preds[start..end], label, &mut out[start..end]);
        }
    }

    /// Name of the loss function (for logging).
    fn name(&self) -> &'static str;
}

// =============================================================================
// Squared Error Loss (Regression)
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
    #[inline]
    fn compute_gradient(&self, pred: f32, label: f32) -> GradientPair {
        let grad = pred - label;
        let hess = 1.0;
        GradientPair::new(grad, hess)
    }

    fn name(&self) -> &'static str {
        "squared_error"
    }
}

// =============================================================================
// Logistic Loss (Binary Classification)
// =============================================================================

/// Logistic loss: L = -y*log(p) - (1-y)*log(1-p), where p = sigmoid(pred)
///
/// Derivatives:
/// - grad = p - label (where p = sigmoid(pred))
/// - hess = p * (1 - p)
///
/// Used for binary classification. Expects labels in {0, 1}.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogisticLoss;

impl Loss for LogisticLoss {
    #[inline]
    fn compute_gradient(&self, pred: f32, label: f32) -> GradientPair {
        let p = sigmoid(pred);
        let grad = p - label;
        // Hessian with small epsilon for numerical stability
        let hess = (p * (1.0 - p)).max(1e-16);
        GradientPair::new(grad, hess)
    }

    fn name(&self) -> &'static str {
        "logistic"
    }
}

// =============================================================================
// Quantile Loss (Quantile Regression)
// =============================================================================

/// Quantile (pinball) loss for quantile regression.
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
/// # Example
///
/// ```
/// use booste_rs::training::{Loss, QuantileLoss, GradientPair};
///
/// // Median regression (α = 0.5)
/// let loss = QuantileLoss::new(0.5);
///
/// // Under-prediction: pred=1, label=2 → grad = -0.5
/// let gp = loss.compute_gradient(1.0, 2.0);
/// assert!(gp.grad() < 0.0);
///
/// // Over-prediction: pred=3, label=2 → grad = 0.5
/// let gp = loss.compute_gradient(3.0, 2.0);
/// assert!(gp.grad() > 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QuantileLoss {
    /// Quantile level α ∈ (0, 1).
    alpha: f32,
}

impl QuantileLoss {
    /// Create a new quantile loss with the given quantile level.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Quantile level in (0, 1). Use 0.5 for median regression.
    ///
    /// # Panics
    ///
    /// Panics if alpha is not in (0, 1).
    pub fn new(alpha: f32) -> Self {
        assert!(
            alpha > 0.0 && alpha < 1.0,
            "Quantile alpha must be in (0, 1), got {}",
            alpha
        );
        Self { alpha }
    }

    /// Returns the quantile level α.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

impl Loss for QuantileLoss {
    #[inline]
    fn compute_gradient(&self, pred: f32, label: f32) -> GradientPair {
        // Pinball loss gradient:
        // - If pred >= label (over-prediction): grad = 1 - alpha
        // - If pred < label (under-prediction): grad = -alpha
        let grad = if pred >= label {
            1.0 - self.alpha
        } else {
            -self.alpha
        };

        // Hessian is 1 for pinball loss (piecewise linear)
        // Using 1.0 provides stable updates
        let hess = 1.0;

        GradientPair::new(grad, hess)
    }

    fn name(&self) -> &'static str {
        "quantile"
    }
}

// =============================================================================
// Softmax Loss (Multiclass Classification)
// =============================================================================

/// Softmax cross-entropy loss for multiclass classification.
///
/// For class k with predictions [p₀, p₁, ..., pₖ] and true class y:
/// - L = -log(softmax(pᵧ))
/// - grad_k = softmax_k - 1{k == y}
/// - hess_k = softmax_k * (1 - softmax_k)
///
/// Implements [`MulticlassLoss`] for proper multiclass gradient handling.
/// The trainer must use the multiclass API when num_groups > 1.
#[derive(Debug, Clone, Copy)]
pub struct SoftmaxLoss {
    /// Number of classes.
    num_classes: usize,
}

impl SoftmaxLoss {
    /// Create a new softmax loss with the given number of classes.
    pub fn new(num_classes: usize) -> Self {
        assert!(num_classes >= 2, "Softmax requires at least 2 classes");
        Self { num_classes }
    }
}

impl MulticlassLoss for SoftmaxLoss {
    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn compute_gradient(&self, preds: &[f32], label: usize, out: &mut [GradientPair]) {
        debug_assert_eq!(preds.len(), self.num_classes);
        debug_assert_eq!(out.len(), self.num_classes);
        debug_assert!(
            label < self.num_classes,
            "Label {} >= num_classes {}",
            label,
            self.num_classes
        );

        // Compute softmax probabilities (stack allocated for small num_classes)
        let mut probs = [0.0f32; 64]; // Max 64 classes on stack
        let probs = if self.num_classes <= 64 {
            softmax(preds, &mut probs[..self.num_classes]);
            &probs[..self.num_classes]
        } else {
            // Fall back to heap for large num_classes
            let mut heap_probs = vec![0.0f32; self.num_classes];
            softmax(preds, &mut heap_probs);
            // Can't return reference to local, so inline the computation
            for (k, gp) in out.iter_mut().enumerate() {
                let prob = heap_probs[k];
                let is_true_class = if k == label { 1.0 } else { 0.0 };
                let grad = prob - is_true_class;
                let hess = (prob * (1.0 - prob)).max(1e-16);
                *gp = GradientPair::new(grad, hess);
            }
            return;
        };

        // Compute gradients for each class
        for (k, (prob, gp)) in probs.iter().zip(out.iter_mut()).enumerate() {
            let is_true_class = if k == label { 1.0 } else { 0.0 };
            let grad = *prob - is_true_class;
            // Hessian: p * (1 - p), with epsilon for numerical stability
            let hess = (prob * (1.0 - prob)).max(1e-16);
            *gp = GradientPair::new(grad, hess);
        }
    }

    fn name(&self) -> &'static str {
        "softmax"
    }
}

// =============================================================================
// Helper functions
// =============================================================================

/// Sigmoid function: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax function (numerically stable).
fn softmax(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if input.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) and sum
    let mut sum = 0.0f32;
    for (inp, out) in input.iter().zip(output.iter_mut()) {
        *out = (*inp - max_val).exp();
        sum += *out;
    }

    // Normalize
    if sum > 0.0 {
        for out in output.iter_mut() {
            *out /= sum;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_loss_gradient() {
        let loss = SquaredLoss;

        // pred = 1.0, label = 0.5 → grad = 0.5, hess = 1.0
        let gp = loss.compute_gradient(1.0, 0.5);
        assert!((gp.grad() - 0.5).abs() < 1e-6);
        assert!((gp.hess() - 1.0).abs() < 1e-6);

        // pred = label → grad = 0
        let gp = loss.compute_gradient(2.0, 2.0);
        assert!(gp.grad().abs() < 1e-6);
    }

    #[test]
    fn logistic_loss_gradient() {
        let loss = LogisticLoss;

        // pred = 0.0 → p = 0.5
        // label = 0 → grad = 0.5 - 0 = 0.5
        // hess = 0.5 * 0.5 = 0.25
        let gp = loss.compute_gradient(0.0, 0.0);
        assert!((gp.grad() - 0.5).abs() < 1e-6);
        assert!((gp.hess() - 0.25).abs() < 1e-6);

        // pred = 0.0, label = 1 → grad = 0.5 - 1 = -0.5
        let gp = loss.compute_gradient(0.0, 1.0);
        assert!((gp.grad() - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn logistic_loss_extreme_values() {
        let loss = LogisticLoss;

        // Large positive pred → p ≈ 1.0
        let gp = loss.compute_gradient(100.0, 1.0);
        assert!(gp.grad().abs() < 0.01); // grad ≈ 0 when correct
        assert!(gp.hess() > 0.0); // hess always positive

        // Large negative pred → p ≈ 0.0
        let gp = loss.compute_gradient(-100.0, 0.0);
        assert!(gp.grad().abs() < 0.01);
        assert!(gp.hess() > 0.0);
    }

    #[test]
    fn softmax_loss_gradient() {
        let loss = SoftmaxLoss::new(3);
        let preds = [1.0, 2.0, 3.0];
        let mut grads = [GradientPair::ZERO; 3];

        // True class is 2
        loss.compute_gradient(&preds, 2, &mut grads);

        // All gradients should be prob - indicator
        // Class 2 should have negative gradient (we want to increase it)
        assert!(grads[2].grad() < 0.0);
        // Class 0 and 1 should have positive gradient (we want to decrease them)
        assert!(grads[0].grad() > 0.0);
        assert!(grads[1].grad() > 0.0);

        // Sum of gradients should be approximately 0
        let grad_sum: f32 = grads.iter().map(|gp| gp.grad()).sum();
        assert!(grad_sum.abs() < 1e-6);

        // All hessians should be positive
        for gp in &grads {
            assert!(gp.hess() > 0.0);
        }
    }

    #[test]
    fn softmax_loss_uniform_preds() {
        let loss = SoftmaxLoss::new(3);
        let preds = [0.0, 0.0, 0.0]; // Uniform predictions
        let mut grads = [GradientPair::ZERO; 3];

        loss.compute_gradient(&preds, 1, &mut grads);

        // With uniform preds, p = 1/3 for all
        // True class (1): grad = 1/3 - 1 = -2/3
        assert!((grads[1].grad() - (-2.0 / 3.0)).abs() < 1e-5);
        // Other classes: grad = 1/3 - 0 = 1/3
        assert!((grads[0].grad() - (1.0 / 3.0)).abs() < 1e-5);
        assert!((grads[2].grad() - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn gradient_batch() {
        let loss = SquaredLoss;
        let preds = [1.0, 2.0, 3.0];
        let labels = [0.5, 2.0, 2.5];
        let mut grads = [GradientPair::ZERO; 3];

        loss.gradient_batch(&preds, &labels, &mut grads);

        // Check each gradient
        assert!((grads[0].grad() - 0.5).abs() < 1e-6); // 1.0 - 0.5
        assert!((grads[1].grad() - 0.0).abs() < 1e-6); // 2.0 - 2.0
        assert!((grads[2].grad() - 0.5).abs() < 1e-6); // 3.0 - 2.5
    }

    #[test]
    fn loss_name() {
        assert_eq!(SquaredLoss.name(), "squared_error");
        assert_eq!(LogisticLoss.name(), "logistic");
        assert_eq!(SoftmaxLoss::new(3).name(), "softmax");
        assert_eq!(QuantileLoss::new(0.5).name(), "quantile");
    }

    #[test]
    fn quantile_loss_median() {
        // α = 0.5: symmetric loss (median regression)
        let loss = QuantileLoss::new(0.5);

        // Under-prediction: pred < label → grad = -α = -0.5
        let gp = loss.compute_gradient(1.0, 2.0);
        assert!((gp.grad() - (-0.5)).abs() < 1e-6);
        assert!((gp.hess() - 1.0).abs() < 1e-6);

        // Over-prediction: pred > label → grad = 1-α = 0.5
        let gp = loss.compute_gradient(3.0, 2.0);
        assert!((gp.grad() - 0.5).abs() < 1e-6);
        assert!((gp.hess() - 1.0).abs() < 1e-6);

        // Perfect prediction: pred == label → grad = 1-α = 0.5
        // (convention: pred >= label counts as over-prediction)
        let gp = loss.compute_gradient(2.0, 2.0);
        assert!((gp.grad() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn quantile_loss_low_quantile() {
        // α = 0.1: 10th percentile (heavily penalize over-prediction)
        let loss = QuantileLoss::new(0.1);

        // Under-prediction: grad = -α = -0.1 (small penalty)
        let gp = loss.compute_gradient(1.0, 2.0);
        assert!((gp.grad() - (-0.1)).abs() < 1e-6);

        // Over-prediction: grad = 1-α = 0.9 (large penalty)
        let gp = loss.compute_gradient(3.0, 2.0);
        assert!((gp.grad() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn quantile_loss_high_quantile() {
        // α = 0.9: 90th percentile (heavily penalize under-prediction)
        let loss = QuantileLoss::new(0.9);

        // Under-prediction: grad = -α = -0.9 (large penalty)
        let gp = loss.compute_gradient(1.0, 2.0);
        assert!((gp.grad() - (-0.9)).abs() < 1e-6);

        // Over-prediction: grad = 1-α = 0.1 (small penalty)
        let gp = loss.compute_gradient(3.0, 2.0);
        assert!((gp.grad() - 0.1).abs() < 1e-6);
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
}
