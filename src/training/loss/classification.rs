//! Classification loss functions.
//!
//! This module provides loss functions for classification tasks:
//!
//! - [`LogisticLoss`]: Binary classification (log loss / cross-entropy)
//! - [`HingeLoss`]: SVM-style binary classification
//! - [`SoftmaxLoss`]: Multiclass classification (softmax cross-entropy)

// Allow range loops when we need indices to access multiple arrays.
#![allow(clippy::needless_range_loop)]

use super::{GradientBuffer, Loss, MulticlassLoss};

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
    /// Compute gradients for logistic loss.
    ///
    /// - grad = sigmoid(pred) - label
    /// - hess = p * (1 - p) where p = sigmoid(pred)
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), buffer.n_samples());
        debug_assert_eq!(buffer.n_outputs(), 1);

        let (grads, hess) = buffer.as_mut_slices();

        // Single pass: compute sigmoid, grad, and hess together
        // This keeps p in register for both grad and hess computation
        for i in 0..preds.len() {
            let p = 1.0 / (1.0 + (-preds[i]).exp());
            grads[i] = p - labels[i];
            hess[i] = (p * (1.0 - p)).max(1e-16);
        }
    }

    fn name(&self) -> &'static str {
        "logistic"
    }
}

// =============================================================================
// Hinge Loss (SVM-style Binary Classification)
// =============================================================================

/// Hinge loss for SVM-style binary classification.
///
/// Expects labels in {0, 1}. Internally converts to {-1, 1}.
///
/// For y = label * 2 - 1:
/// - If pred × y < 1 (wrong side of margin): grad = -y, hess = 1
/// - If pred × y >= 1 (correct side): grad = 0, hess = ε (small value)
///
/// # Notes
///
/// The hinge loss is not differentiable at the margin (pred × y = 1).
/// This implementation uses a subgradient with hess = ε when on correct side.
///
/// # XGBoost Compatibility
///
/// This matches XGBoost's `binary:hinge` objective.
#[derive(Debug, Clone, Copy, Default)]
pub struct HingeLoss;

impl HingeLoss {
    /// Create a new hinge loss.
    pub fn new() -> Self {
        Self
    }
}

impl Loss for HingeLoss {
    /// Compute gradients for hinge loss.
    ///
    /// - If pred × y < 1: grad = -y, hess = 1
    /// - If pred × y >= 1: grad = 0, hess = ε
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), buffer.n_samples());
        debug_assert_eq!(buffer.n_outputs(), 1);

        let (grads, hess) = buffer.as_mut_slices();

        for i in 0..preds.len() {
            let p = preds[i];
            // Convert label from {0, 1} to {-1, 1}
            let y = labels[i] * 2.0 - 1.0;

            if p * y < 1.0 {
                // Wrong side of margin (or within margin)
                grads[i] = -y;
                hess[i] = 1.0;
            } else {
                // Correct side of margin
                grads[i] = 0.0;
                hess[i] = f32::MIN_POSITIVE; // Small positive value
            }
        }
    }

    fn name(&self) -> &'static str {
        "hinge"
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

    /// Compute gradients for softmax cross-entropy loss.
    ///
    /// For each sample and class k:
    /// - grad_k = softmax_k - 1{k == y}
    /// - hess_k = softmax_k * (1 - softmax_k)
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        let num_classes = self.num_classes;
        let num_samples = labels.len();
        debug_assert_eq!(preds.len(), num_samples * num_classes);
        debug_assert_eq!(buffer.n_samples(), num_samples);
        debug_assert_eq!(buffer.n_outputs(), num_classes);

        let (grads, hess) = buffer.as_mut_slices();

        for i in 0..num_samples {
            let start = i * num_classes;
            let end = start + num_classes;
            let label = labels[i] as usize;
            let sample_preds = &preds[start..end];
            let sample_grads = &mut grads[start..end];
            let sample_hess = &mut hess[start..end];

            debug_assert!(
                label < num_classes,
                "Label {} >= num_classes {}",
                label,
                num_classes
            );

            // Compute softmax probabilities directly into grads (reuse as temp)
            softmax(sample_preds, sample_grads);

            // Convert softmax probs to gradients
            for k in 0..num_classes {
                let prob = sample_grads[k];
                let is_true_class = if k == label { 1.0 } else { 0.0 };
                sample_grads[k] = prob - is_true_class;
                sample_hess[k] = (prob * (1.0 - prob)).max(1e-16);
            }
        }
    }

    fn name(&self) -> &'static str {
        "softmax"
    }
}

// =============================================================================
// Helper functions
// =============================================================================

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

    // =========================================================================
    // LogisticLoss tests
    // =========================================================================

    #[test]
    fn logistic_loss_gradient() {
        let loss = LogisticLoss;
        let preds = vec![0.0, 0.0]; // p = 0.5 for both
        let labels = vec![0.0, 1.0];
        let mut buffer = GradientBuffer::new(2, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // pred=0.0, label=0: grad = 0.5 - 0 = 0.5, hess = 0.25
        assert!((buffer.grad(0, 0) - 0.5).abs() < 1e-6);
        assert!((buffer.hess(0, 0) - 0.25).abs() < 1e-6);

        // pred=0.0, label=1: grad = 0.5 - 1 = -0.5
        assert!((buffer.grad(1, 0) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn logistic_loss_extreme_values() {
        let loss = LogisticLoss;
        let preds = vec![100.0, -100.0];
        let labels = vec![1.0, 0.0];
        let mut buffer = GradientBuffer::new(2, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // Large positive pred → p ≈ 1.0, grad ≈ 0 when label=1
        assert!(buffer.grad(0, 0).abs() < 0.01);
        assert!(buffer.hess(0, 0) > 0.0);

        // Large negative pred → p ≈ 0.0, grad ≈ 0 when label=0
        assert!(buffer.grad(1, 0).abs() < 0.01);
        assert!(buffer.hess(1, 0) > 0.0);
    }

    #[test]
    fn logistic_loss_batch() {
        let loss = LogisticLoss;
        let preds = vec![0.0, 0.0]; // p = 0.5 for both
        let labels = vec![0.0, 1.0];
        let mut buffer = GradientBuffer::new(2, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // label=0: grad = 0.5 - 0 = 0.5
        assert!((buffer.grad(0, 0) - 0.5).abs() < 1e-6);
        // label=1: grad = 0.5 - 1 = -0.5
        assert!((buffer.grad(1, 0) - (-0.5)).abs() < 1e-6);
        // hess = 0.5 * 0.5 = 0.25
        assert!((buffer.hess(0, 0) - 0.25).abs() < 1e-6);
        assert!((buffer.hess(1, 0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn logistic_loss_name() {
        assert_eq!(Loss::name(&LogisticLoss), "logistic");
    }

    // =========================================================================
    // HingeLoss tests
    // =========================================================================

    #[test]
    fn hinge_loss_gradient_correct_side() {
        let loss = HingeLoss;
        // Label 1 (y=1), prediction > 1 → correct side of margin
        let preds = vec![2.0];
        let labels = vec![1.0];
        let mut buffer = GradientBuffer::new(1, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // p * y = 2 * 1 = 2 >= 1 → grad = 0
        assert!(buffer.grad(0, 0).abs() < 1e-6);
        // hess = ε (very small)
        assert!(buffer.hess(0, 0) > 0.0);
        assert!(buffer.hess(0, 0) < 1e-10);
    }

    #[test]
    fn hinge_loss_gradient_wrong_side() {
        let loss = HingeLoss;
        // Label 1 (y=1), prediction < 1 → wrong side of margin
        let preds = vec![0.5];
        let labels = vec![1.0];
        let mut buffer = GradientBuffer::new(1, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // p * y = 0.5 * 1 = 0.5 < 1 → grad = -y = -1
        assert!((buffer.grad(0, 0) - (-1.0)).abs() < 1e-6);
        // hess = 1
        assert!((buffer.hess(0, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hinge_loss_negative_class() {
        let loss = HingeLoss;
        // Label 0 (y=-1), prediction < -1 → correct side
        let preds = vec![-2.0];
        let labels = vec![0.0];
        let mut buffer = GradientBuffer::new(1, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // y = 0 * 2 - 1 = -1
        // p * y = -2 * -1 = 2 >= 1 → grad = 0
        assert!(buffer.grad(0, 0).abs() < 1e-6);
    }

    #[test]
    fn hinge_loss_batch() {
        let loss = HingeLoss;
        // Various cases:
        // 1. label=1, pred=2 → correct (grad=0)
        // 2. label=1, pred=0 → wrong (grad=-1)
        // 3. label=0, pred=-2 → correct (grad=0)
        // 4. label=0, pred=0 → wrong (grad=1)
        let preds = vec![2.0, 0.0, -2.0, 0.0];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let mut buffer = GradientBuffer::new(4, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // Case 1: correct, grad=0
        assert!(buffer.grad(0, 0).abs() < 1e-6);

        // Case 2: wrong, y=1, grad=-1
        assert!((buffer.grad(1, 0) - (-1.0)).abs() < 1e-6);

        // Case 3: correct, grad=0
        assert!(buffer.grad(2, 0).abs() < 1e-6);

        // Case 4: wrong, y=-1, grad=1
        assert!((buffer.grad(3, 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hinge_loss_at_margin() {
        let loss = HingeLoss;
        // Exactly at margin: p * y = 1
        let preds = vec![1.0];
        let labels = vec![1.0];
        let mut buffer = GradientBuffer::new(1, 1);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // p * y = 1 * 1 = 1, which is NOT < 1, so correct side
        assert!(buffer.grad(0, 0).abs() < 1e-6);
    }

    #[test]
    fn hinge_loss_name() {
        assert_eq!(Loss::name(&HingeLoss), "hinge");
    }

    // =========================================================================
    // SoftmaxLoss tests
    // =========================================================================

    #[test]
    fn softmax_loss_gradient() {
        let loss = SoftmaxLoss::new(3);
        let preds = vec![1.0, 2.0, 3.0]; // 1 sample, 3 classes
        let labels = vec![2.0]; // true class is 2
        let mut buffer = GradientBuffer::new(1, 3);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // Class 2 should have negative gradient (we want to increase it)
        assert!(buffer.grad(0, 2) < 0.0);
        // Class 0 and 1 should have positive gradient (we want to decrease them)
        assert!(buffer.grad(0, 0) > 0.0);
        assert!(buffer.grad(0, 1) > 0.0);

        // Sum of gradients should be approximately 0
        let grad_sum = buffer.grad(0, 0) + buffer.grad(0, 1) + buffer.grad(0, 2);
        assert!(grad_sum.abs() < 1e-6);

        // All hessians should be positive
        for class in 0..3 {
            assert!(buffer.hess(0, class) > 0.0);
        }
    }

    #[test]
    fn softmax_loss_uniform_preds() {
        let loss = SoftmaxLoss::new(3);
        let preds = vec![0.0, 0.0, 0.0]; // Uniform predictions
        let labels = vec![1.0]; // true class is 1
        let mut buffer = GradientBuffer::new(1, 3);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // With uniform preds, p = 1/3 for all
        // True class (1): grad = 1/3 - 1 = -2/3
        assert!((buffer.grad(0, 1) - (-2.0 / 3.0)).abs() < 1e-5);
        // Other classes: grad = 1/3 - 0 = 1/3
        assert!((buffer.grad(0, 0) - (1.0 / 3.0)).abs() < 1e-5);
        assert!((buffer.grad(0, 2) - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn softmax_loss_batch() {
        let loss = SoftmaxLoss::new(3);
        // 2 samples, 3 classes each
        let preds = vec![
            1.0, 2.0, 3.0, // sample 0
            0.0, 0.0, 0.0, // sample 1 (uniform)
        ];
        let labels = vec![2.0, 1.0]; // true classes
        let mut buffer = GradientBuffer::new(2, 3);

        loss.compute_gradients(&preds, &labels, &mut buffer);

        // Sample 0: true class is 2, should have negative gradient
        assert!(buffer.grad(0, 2) < 0.0);
        // Other classes should have positive gradient
        assert!(buffer.grad(0, 0) > 0.0);
        assert!(buffer.grad(0, 1) > 0.0);

        // Sample 1 (uniform): true class is 1
        // grad = 1/3 - 1 = -2/3 for class 1
        assert!((buffer.grad(1, 1) - (-2.0 / 3.0)).abs() < 1e-5);
        // grad = 1/3 - 0 = 1/3 for classes 0, 2
        assert!((buffer.grad(1, 0) - (1.0 / 3.0)).abs() < 1e-5);
        assert!((buffer.grad(1, 2) - (1.0 / 3.0)).abs() < 1e-5);

        // All hessians should be positive
        for sample in 0..2 {
            for class in 0..3 {
                assert!(buffer.hess(sample, class) > 0.0);
            }
        }
    }

    #[test]
    fn softmax_loss_name() {
        assert_eq!(MulticlassLoss::name(&SoftmaxLoss::new(3)), "softmax");
    }
}
