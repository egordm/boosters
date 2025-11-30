//! Loss functions for computing gradients.
//!
//! Each loss function computes gradient-hessian pairs for optimization.
//! These are used by both GBLinear and GBTree training.

use super::GradientPair;

/// A loss function that computes gradients for training.
///
/// Unlike the [`crate::objective::Objective`] enum which handles output transformations
/// during inference, this trait computes the gradients needed for optimization during
/// training.
///
/// # Type Parameters
///
/// Loss functions operate on `f32` predictions and labels.
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
// Softmax Loss (Multiclass Classification)
// =============================================================================

/// Softmax cross-entropy loss for multiclass classification.
///
/// For class k with predictions [p₀, p₁, ..., pₖ] and true class y:
/// - L = -log(softmax(pᵧ))
/// - grad_k = softmax_k - 1{k == y}
/// - hess_k = softmax_k * (1 - softmax_k)
///
/// This struct handles one class at a time. The training loop should
/// handle the multiclass structure by computing gradients for each class.
///
/// Note: For multiclass, labels are typically one-hot encoded or the
/// gradient computation handles class indices directly.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftmaxLoss {
    /// Number of classes.
    pub num_classes: usize,
}

impl SoftmaxLoss {
    /// Create a new softmax loss with the given number of classes.
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }

    /// Compute gradients for all classes at once.
    ///
    /// # Arguments
    ///
    /// * `preds` - Raw predictions for all classes (length = num_classes)
    /// * `label` - True class index (0-based)
    /// * `out` - Output gradient pairs for each class
    pub fn compute_multiclass_gradient(
        &self,
        preds: &[f32],
        label: usize,
        out: &mut [GradientPair],
    ) {
        debug_assert_eq!(preds.len(), self.num_classes);
        debug_assert_eq!(out.len(), self.num_classes);
        debug_assert!(label < self.num_classes);

        // Compute softmax probabilities
        let mut probs = vec![0.0f32; self.num_classes];
        softmax(preds, &mut probs);

        // Compute gradients for each class
        for (k, (prob, gp)) in probs.iter().zip(out.iter_mut()).enumerate() {
            let is_true_class = if k == label { 1.0 } else { 0.0 };
            let grad = *prob - is_true_class;
            // Hessian with small epsilon for numerical stability
            let hess = (prob * (1.0 - prob)).max(1e-16);
            *gp = GradientPair::new(grad, hess);
        }
    }
}

impl Loss for SoftmaxLoss {
    fn compute_gradient(&self, _pred: f32, _label: f32) -> GradientPair {
        // For multiclass, use compute_multiclass_gradient instead
        panic!("SoftmaxLoss requires compute_multiclass_gradient for proper usage");
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
        loss.compute_multiclass_gradient(&preds, 2, &mut grads);

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

        loss.compute_multiclass_gradient(&preds, 1, &mut grads);

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
    }
}
