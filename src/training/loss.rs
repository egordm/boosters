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
//! # Gradient Storage
//!
//! Gradients are stored in [`GradientBuffer`] (Structure-of-Arrays layout):
//! - Separate `grads[]` and `hess[]` arrays for cache efficiency
//! - Shape `[n_samples, n_outputs]` with natural multi-output indexing
//!
//! # Design for GBTree Compatibility
//!
//! The gradient computation is designed to work with both linear and tree models:
//! - Gradients are stored per (sample, output) pair
//! - For regression/binary: 1 output, `buffer.n_outputs() == 1`
//! - For multiclass: K outputs, `buffer.n_outputs() == K`

use super::GradientBuffer;

/// A loss function for single-output models (regression, binary classification).
///
/// For losses where each sample has one prediction and one gradient.
/// Examples: squared error, logistic loss, quantile loss.
///
/// # Implementation Notes
///
/// - `compute_gradient`: Compute grad/hess for a single sample
/// - `gradient_buffer`: Batch method that writes to SoA buffer (primary API)
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
    /// Tuple (grad, hess) where:
    /// - grad = ∂L/∂pred (first derivative)
    /// - hess = ∂²L/∂pred² (second derivative)
    fn compute_gradient(&self, pred: f32, label: f32) -> (f32, f32);

    /// Compute gradients for a batch of samples into SoA buffer.
    ///
    /// This is the primary method for training. Default implementation calls
    /// `compute_gradient` for each sample; can be overridden for vectorized
    /// implementations.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions, length = n_samples (single-output loss)
    /// * `labels` - Labels, length = n_samples
    /// * `buffer` - Output buffer with `n_samples` samples and `n_outputs == 1`
    ///
    /// # Panics
    ///
    /// Panics if buffer dimensions don't match input lengths.
    fn gradient_buffer(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), buffer.n_samples());
        debug_assert_eq!(buffer.n_outputs(), 1, "Loss trait expects n_outputs == 1");

        let (grads, hess) = buffer.as_mut_slices();
        for (i, (pred, label)) in preds.iter().zip(labels.iter()).enumerate() {
            let (g, h) = self.compute_gradient(*pred, *label);
            grads[i] = g;
            hess[i] = h;
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
/// For N samples and K classes, gradients are stored in SoA buffer as:
/// `buffer.grads[sample_idx * num_classes + class_idx]`
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
    /// * `grads` - Output gradient slice for each class (length = num_classes)
    /// * `hess` - Output hessian slice for each class (length = num_classes)
    fn compute_gradient(&self, preds: &[f32], label: usize, grads: &mut [f32], hess: &mut [f32]);

    /// Compute gradients for a batch of samples into SoA buffer.
    ///
    /// This is the primary method for training.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions, layout: `preds[sample * num_classes + class]`
    /// * `labels` - Class labels (0-based indices)
    /// * `buffer` - Output buffer with `n_samples` samples and `n_outputs == num_classes`
    ///
    /// # Panics
    ///
    /// Panics if buffer dimensions don't match.
    fn gradient_buffer(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        let num_classes = self.num_classes();
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

            self.compute_gradient(
                sample_preds,
                label,
                &mut grads[start..end],
                &mut hess[start..end],
            );
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
    fn compute_gradient(&self, pred: f32, label: f32) -> (f32, f32) {
        let grad = pred - label;
        let hess = 1.0;
        (grad, hess)
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
    fn compute_gradient(&self, pred: f32, label: f32) -> (f32, f32) {
        let p = sigmoid(pred);
        let grad = p - label;
        // Hessian with small epsilon for numerical stability
        let hess = (p * (1.0 - p)).max(1e-16);
        (grad, hess)
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
    #[inline]
    fn compute_gradient(&self, pred: f32, label: f32) -> (f32, f32) {
        // For single-quantile, use the first alpha
        let alpha = self.alphas[0];

        // Pinball loss gradient:
        // - If pred >= label (over-prediction): grad = 1 - alpha
        // - If pred < label (under-prediction): grad = -alpha
        let grad = if pred >= label {
            1.0 - alpha
        } else {
            -alpha
        };

        // Hessian is 1 for pinball loss (piecewise linear)
        (grad, 1.0)
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

    fn compute_gradient(&self, preds: &[f32], label: usize, grads: &mut [f32], hess: &mut [f32]) {
        debug_assert_eq!(preds.len(), self.alphas.len());
        debug_assert_eq!(grads.len(), self.alphas.len());
        debug_assert_eq!(hess.len(), self.alphas.len());

        // Note: label is passed as usize but we need the continuous target.
        // This method is not ideal for quantile regression — use gradient_buffer.
        let _ = label;

        // Fill with placeholder values (gradient_buffer is the primary API)
        for i in 0..self.alphas.len() {
            grads[i] = 0.0;
            hess[i] = 1.0;
        }
    }

    /// Compute gradients for all samples and all quantiles.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions, layout: `preds[sample * num_quantiles + quantile]`
    /// * `labels` - Continuous target values (NOT class indices)
    /// * `buffer` - Output buffer with `n_samples` samples and `n_outputs == num_quantiles`
    fn gradient_buffer(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
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
                grads[idx] = if pred >= label {
                    1.0 - alpha
                } else {
                    -alpha
                };

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

    fn compute_gradient(&self, preds: &[f32], label: usize, grads: &mut [f32], hess: &mut [f32]) {
        debug_assert_eq!(preds.len(), self.num_classes);
        debug_assert_eq!(grads.len(), self.num_classes);
        debug_assert_eq!(hess.len(), self.num_classes);
        debug_assert!(
            label < self.num_classes,
            "Label {} >= num_classes {}",
            label,
            self.num_classes
        );

        // Compute softmax probabilities directly into grads (reuse as temp)
        softmax(preds, grads);

        // Now grads contains softmax probabilities, convert to gradients
        for k in 0..self.num_classes {
            let prob = grads[k];
            let is_true_class = if k == label { 1.0 } else { 0.0 };
            grads[k] = prob - is_true_class;
            hess[k] = (prob * (1.0 - prob)).max(1e-16);
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
        let (grad, hess) = loss.compute_gradient(1.0, 0.5);
        assert!((grad - 0.5).abs() < 1e-6);
        assert!((hess - 1.0).abs() < 1e-6);

        // pred = label → grad = 0
        let (grad, _) = loss.compute_gradient(2.0, 2.0);
        assert!(grad.abs() < 1e-6);
    }

    #[test]
    fn logistic_loss_gradient() {
        let loss = LogisticLoss;

        // pred = 0.0 → p = 0.5
        // label = 0 → grad = 0.5 - 0 = 0.5
        // hess = 0.5 * 0.5 = 0.25
        let (grad, hess) = loss.compute_gradient(0.0, 0.0);
        assert!((grad - 0.5).abs() < 1e-6);
        assert!((hess - 0.25).abs() < 1e-6);

        // pred = 0.0, label = 1 → grad = 0.5 - 1 = -0.5
        let (grad, _) = loss.compute_gradient(0.0, 1.0);
        assert!((grad - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn logistic_loss_extreme_values() {
        let loss = LogisticLoss;

        // Large positive pred → p ≈ 1.0
        let (grad, hess) = loss.compute_gradient(100.0, 1.0);
        assert!(grad.abs() < 0.01); // grad ≈ 0 when correct
        assert!(hess > 0.0); // hess always positive

        // Large negative pred → p ≈ 0.0
        let (grad, hess) = loss.compute_gradient(-100.0, 0.0);
        assert!(grad.abs() < 0.01);
        assert!(hess > 0.0);
    }

    #[test]
    fn softmax_loss_gradient() {
        let loss = SoftmaxLoss::new(3);
        let preds = [1.0, 2.0, 3.0];
        let mut grads = [0.0f32; 3];
        let mut hess = [0.0f32; 3];

        // True class is 2
        loss.compute_gradient(&preds, 2, &mut grads, &mut hess);

        // Class 2 should have negative gradient (we want to increase it)
        assert!(grads[2] < 0.0);
        // Class 0 and 1 should have positive gradient (we want to decrease them)
        assert!(grads[0] > 0.0);
        assert!(grads[1] > 0.0);

        // Sum of gradients should be approximately 0
        let grad_sum: f32 = grads.iter().sum();
        assert!(grad_sum.abs() < 1e-6);

        // All hessians should be positive
        for h in &hess {
            assert!(*h > 0.0);
        }
    }

    #[test]
    fn softmax_loss_uniform_preds() {
        let loss = SoftmaxLoss::new(3);
        let preds = [0.0, 0.0, 0.0]; // Uniform predictions
        let mut grads = [0.0f32; 3];
        let mut hess = [0.0f32; 3];

        loss.compute_gradient(&preds, 1, &mut grads, &mut hess);

        // With uniform preds, p = 1/3 for all
        // True class (1): grad = 1/3 - 1 = -2/3
        assert!((grads[1] - (-2.0 / 3.0)).abs() < 1e-5);
        // Other classes: grad = 1/3 - 0 = 1/3
        assert!((grads[0] - (1.0 / 3.0)).abs() < 1e-5);
        assert!((grads[2] - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn loss_name() {
        assert_eq!(Loss::name(&SquaredLoss), "squared_error");
        assert_eq!(Loss::name(&LogisticLoss), "logistic");
        assert_eq!(MulticlassLoss::name(&SoftmaxLoss::new(3)), "softmax");
        assert_eq!(Loss::name(&QuantileLoss::new(0.5)), "quantile");
    }

    // =========================================================================
    // GradientBuffer tests
    // =========================================================================

    #[test]
    fn squared_loss_gradient_buffer() {
        let loss = SquaredLoss;
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![0.5, 2.0, 2.5];
        let mut buffer = GradientBuffer::new(3, 1);

        loss.gradient_buffer(&preds, &labels, &mut buffer);

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
    fn logistic_loss_gradient_buffer() {
        let loss = LogisticLoss;
        let preds = vec![0.0, 0.0]; // p = 0.5 for both
        let labels = vec![0.0, 1.0];
        let mut buffer = GradientBuffer::new(2, 1);

        loss.gradient_buffer(&preds, &labels, &mut buffer);

        // label=0: grad = 0.5 - 0 = 0.5
        assert!((buffer.grad(0, 0) - 0.5).abs() < 1e-6);
        // label=1: grad = 0.5 - 1 = -0.5
        assert!((buffer.grad(1, 0) - (-0.5)).abs() < 1e-6);
        // hess = 0.5 * 0.5 = 0.25
        assert!((buffer.hess(0, 0) - 0.25).abs() < 1e-6);
        assert!((buffer.hess(1, 0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn softmax_loss_gradient_buffer() {
        let loss = SoftmaxLoss::new(3);
        // 2 samples, 3 classes each
        let preds = vec![
            1.0, 2.0, 3.0, // sample 0
            0.0, 0.0, 0.0, // sample 1 (uniform)
        ];
        let labels = vec![2.0, 1.0]; // true classes
        let mut buffer = GradientBuffer::new(2, 3);

        loss.gradient_buffer(&preds, &labels, &mut buffer);

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
    fn quantile_loss_gradient_buffer() {
        let loss = QuantileLoss::new(0.5);
        let preds = vec![1.0, 3.0, 2.0]; // under, over, exact
        let labels = vec![2.0, 2.0, 2.0];
        let mut buffer = GradientBuffer::new(3, 1);

        MulticlassLoss::gradient_buffer(&loss, &preds, &labels, &mut buffer);

        // Under-prediction: grad = -0.5
        assert!((buffer.grad(0, 0) - (-0.5)).abs() < 1e-6);
        // Over-prediction: grad = 0.5
        assert!((buffer.grad(1, 0) - 0.5).abs() < 1e-6);
        // Exact: grad = 0.5 (convention: pred >= label is over)
        assert!((buffer.grad(2, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn quantile_loss_median() {
        // α = 0.5: symmetric loss (median regression)
        let loss = QuantileLoss::new(0.5);

        // Under-prediction: pred < label → grad = -α = -0.5
        let (grad, hess) = Loss::compute_gradient(&loss, 1.0, 2.0);
        assert!((grad - (-0.5)).abs() < 1e-6);
        assert!((hess - 1.0).abs() < 1e-6);

        // Over-prediction: pred > label → grad = 1-α = 0.5
        let (grad, hess) = Loss::compute_gradient(&loss, 3.0, 2.0);
        assert!((grad - 0.5).abs() < 1e-6);
        assert!((hess - 1.0).abs() < 1e-6);

        // Perfect prediction: pred == label → grad = 1-α = 0.5
        // (convention: pred >= label counts as over-prediction)
        let (grad, _) = Loss::compute_gradient(&loss, 2.0, 2.0);
        assert!((grad - 0.5).abs() < 1e-6);
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

    // =========================================================================
    // Multi-Quantile Loss tests (using QuantileLoss::multi)
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
    fn multi_quantile_loss_gradient_buffer() {
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

        MulticlassLoss::gradient_buffer(&loss, &preds, &labels, &mut buffer);

        // Sample 0
        assert!((buffer.grad(0, 0) - (-0.1)).abs() < 1e-6); // under, α=0.1
        assert!((buffer.grad(0, 1) - 0.5).abs() < 1e-6);    // exact, α=0.5
        assert!((buffer.grad(0, 2) - 0.1).abs() < 1e-6);    // over, α=0.9

        // Sample 1
        assert!((buffer.grad(1, 0) - 0.9).abs() < 1e-6);    // over, α=0.1
        assert!((buffer.grad(1, 1) - 0.5).abs() < 1e-6);    // exact, α=0.5
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

        let preds_multi = vec![1.0, 3.0, 2.0]; // 3 samples, 1 quantile each
        let preds_single = vec![1.0, 3.0, 2.0];
        let labels = vec![2.0, 2.0, 2.0];

        let mut buffer_multi = GradientBuffer::new(3, 1);
        let mut buffer_single = GradientBuffer::new(3, 1);

        MulticlassLoss::gradient_buffer(&multi_loss, &preds_multi, &labels, &mut buffer_multi);
        MulticlassLoss::gradient_buffer(&single_loss, &preds_single, &labels, &mut buffer_single);

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

        MulticlassLoss::gradient_buffer(&loss, &preds, &labels, &mut buffer);

        // All over-predictions
        assert!((buffer.grad(0, 0) - 0.9).abs() < 1e-6);  // 1 - 0.1
        assert!((buffer.grad(0, 1) - 0.5).abs() < 1e-6);  // 1 - 0.5
        assert!((buffer.grad(0, 2) - 0.1).abs() < 1e-6);  // 1 - 0.9

        // All predictions well below label
        let preds = vec![-10.0, -10.0, -10.0]; // label = 0.0
        MulticlassLoss::gradient_buffer(&loss, &preds, &labels, &mut buffer);

        // All under-predictions
        assert!((buffer.grad(0, 0) - (-0.1)).abs() < 1e-6);  // -0.1
        assert!((buffer.grad(0, 1) - (-0.5)).abs() < 1e-6);  // -0.5
        assert!((buffer.grad(0, 2) - (-0.9)).abs() < 1e-6);  // -0.9
    }
}
