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
//! # Design Rationale
//!
//! The batch-oriented API (`compute_gradients`) is chosen over per-sample methods for:
//! - **Future GPU support**: Batch operations map naturally to GPU kernels
//! - **Python bindings**: NumPy-based custom losses need batch operations for efficiency
//! - **Vectorization**: Compilers can auto-vectorize batch loops better than callbacks
//!
//! See `docs/benchmarks/2025-11-29-gradient-batch.md` for performance analysis.

use super::GradientBuffer;

/// A loss function for single-output models (regression, binary classification).
///
/// For losses where each sample has one prediction and one gradient.
/// Examples: squared error, logistic loss, quantile loss.
///
/// # Implementing Custom Losses
///
/// Implement `compute_gradients` to write gradients/hessians for all samples:
///
/// ```ignore
/// impl Loss for MyLoss {
///     fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
///         let (grads, hess) = buffer.as_mut_slices();
///         for i in 0..preds.len() {
///             grads[i] = /* your gradient */;
///             hess[i] = /* your hessian */;
///         }
///     }
///     fn name(&self) -> &'static str { "my_loss" }
/// }
/// ```
pub trait Loss: Send + Sync {
    /// Compute gradients and hessians for a batch of samples.
    ///
    /// This is the primary method for training. Implementations should write
    /// gradients and hessians directly to the buffer for best performance.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions, length = n_samples
    /// * `labels` - Labels, length = n_samples
    /// * `buffer` - Output buffer with `n_samples` samples and `n_outputs == 1`
    ///
    /// # Panics
    ///
    /// Panics if buffer dimensions don't match input lengths.
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer);

    /// Name of the loss function (for logging).
    fn name(&self) -> &'static str;
}

/// A loss function for multi-output models (multiclass, multi-quantile).
///
/// Unlike [`Loss`], this handles multiple outputs per sample. Each sample
/// produces K gradients where K = `num_outputs()`.
///
/// # Gradient Layout
///
/// For N samples and K outputs, gradients are stored in SoA buffer as:
/// `buffer.grads[sample_idx * num_outputs + output_idx]`
///
/// This layout matches XGBoost and allows efficient per-group weight updates.
///
/// # Examples
///
/// - **Softmax**: K classes, each with its own gradient
/// - **Multi-quantile**: K quantiles, each predicting a different percentile
pub trait MulticlassLoss: Send + Sync {
    /// Number of outputs per sample.
    fn num_classes(&self) -> usize;

    /// Compute gradients and hessians for a batch of samples.
    ///
    /// This is the primary method for training.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions, layout: `preds[sample * num_outputs + output]`
    /// * `labels` - Labels (interpretation depends on loss type)
    /// * `buffer` - Output buffer with `n_samples` samples and `n_outputs == num_classes()`
    ///
    /// # Panics
    ///
    /// Panics if buffer dimensions don't match.
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer);

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
        let grad_over = 1.0 - alpha;  // Precompute for over-prediction
        let grad_under = -alpha;       // Precompute for under-prediction

        let (grads, hess) = buffer.as_mut_slices();

        // Vectorizable loop with precomputed constants
        for i in 0..preds.len() {
            grads[i] = if preds[i] >= labels[i] { grad_over } else { grad_under };
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
// Pseudo-Huber Loss (Robust Regression)
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
        assert!(slope > 0.0, "Pseudo-Huber slope must be positive, got {}", slope);
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
    fn loss_name() {
        assert_eq!(Loss::name(&SquaredLoss), "squared_error");
        assert_eq!(Loss::name(&LogisticLoss), "logistic");
        assert_eq!(MulticlassLoss::name(&SoftmaxLoss::new(3)), "softmax");
        assert_eq!(Loss::name(&QuantileLoss::new(0.5)), "quantile");
        assert_eq!(Loss::name(&PseudoHuberLoss::default()), "pseudo_huber");
        assert_eq!(Loss::name(&HingeLoss), "hinge");
    }

    // =========================================================================
    // SquaredLoss batch tests
    // =========================================================================

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

    // =========================================================================
    // LogisticLoss batch tests
    // =========================================================================

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

    // =========================================================================
    // SoftmaxLoss batch tests
    // =========================================================================

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
        assert!((buffer.grad(0, 0) - 0.9).abs() < 1e-6);  // 1 - 0.1
        assert!((buffer.grad(0, 1) - 0.5).abs() < 1e-6);  // 1 - 0.5
        assert!((buffer.grad(0, 2) - 0.1).abs() < 1e-6);  // 1 - 0.9

        // All predictions well below label
        let preds = vec![-10.0, -10.0, -10.0]; // label = 0.0
        MulticlassLoss::compute_gradients(&loss, &preds, &labels, &mut buffer);

        // All under-predictions
        assert!((buffer.grad(0, 0) - (-0.1)).abs() < 1e-6);  // -0.1
        assert!((buffer.grad(0, 1) - (-0.5)).abs() < 1e-6);  // -0.5
        assert!((buffer.grad(0, 2) - (-0.9)).abs() < 1e-6);  // -0.9
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
            assert!(diff < 0.01, "Sample {}: PH={}, SQ={}", i, buffer_ph.grad(i, 0), buffer_sq.grad(i, 0));
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
}
