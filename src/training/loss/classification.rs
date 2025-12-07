//! Classification loss functions.
//!
//! This module provides loss functions for classification tasks:
//!
//! - [`LogisticLoss`]: Binary classification (log loss / cross-entropy)
//! - [`HingeLoss`]: SVM-style binary classification
//! - [`SoftmaxLoss`]: Multiclass classification (softmax cross-entropy)

// Allow range loops when we need indices to access multiple arrays.
#![allow(clippy::needless_range_loop)]

use super::{GradientBuffer, Loss};

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
    fn num_outputs(&self) -> usize {
        1
    }

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

    /// Base score = log-odds of positive class.
    ///
    /// Computes `log(p / (1-p))` where p = weighted proportion of positive labels.
    fn init_base_score(&self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
        if labels.is_empty() {
            return vec![0.0];
        }

        let (pos_weight, total_weight) = match weights {
            Some(w) => labels
                .iter()
                .zip(w.iter())
                .fold((0.0f64, 0.0f64), |(pos, total), (&l, &wt)| {
                    let wt = wt as f64;
                    (pos + if l > 0.5 { wt } else { 0.0 }, total + wt)
                }),
            None => {
                let pos = labels.iter().filter(|&&l| l > 0.5).count() as f64;
                (pos, labels.len() as f64)
            }
        };

        if total_weight == 0.0 {
            return vec![0.0];
        }

        // Clamp to avoid log(0) or log(inf)
        let p = (pos_weight / total_weight).clamp(1e-7, 1.0 - 1e-7);
        vec![(p / (1.0 - p)).ln() as f32]
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
    fn num_outputs(&self) -> usize {
        1
    }

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

    fn init_base_score(&self, _labels: &[f32], _weights: Option<&[f32]>) -> Vec<f32> {
        // Hinge loss: start from 0 (no clear optimal base score)
        vec![0.0]
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
/// This is a multi-output loss with `num_outputs() = num_classes`.
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

impl Loss for SoftmaxLoss {
    fn num_outputs(&self) -> usize {
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

        // Two-phase approach for better cache locality:
        // Phase 1: Compute all softmax probabilities (store in temp buffer)
        // Phase 2: Write to gradient buffer in column-major order (class-first)

        // Temporary storage for probabilities - row-major [sample, class]
        let mut all_probs = vec![0.0f32; num_samples * num_classes];

        // Phase 1: Compute softmax for each sample (reading row-major predictions)
        for i in 0..num_samples {
            let pred_start = i * num_classes;
            let sample_preds = &preds[pred_start..pred_start + num_classes];
            let prob_start = i * num_classes;
            let sample_probs = &mut all_probs[prob_start..prob_start + num_classes];

            debug_assert!(
                (labels[i] as usize) < num_classes,
                "Label {} >= num_classes {}",
                labels[i],
                num_classes
            );

            softmax(sample_preds, sample_probs);
        }

        // Phase 2: Write gradients in column-major order (contiguous per class)
        for k in 0..num_classes {
            let grads_k = buffer.output_grads_mut(k);
            for i in 0..num_samples {
                let prob = all_probs[i * num_classes + k];
                let is_true_class = if k == labels[i] as usize { 1.0 } else { 0.0 };
                grads_k[i] = prob - is_true_class;
            }
        }

        // Phase 3: Write hessians in column-major order (separate pass to avoid borrow issues)
        for k in 0..num_classes {
            let hess_k = buffer.output_hess_mut(k);
            for i in 0..num_samples {
                let prob = all_probs[i * num_classes + k];
                hess_k[i] = (prob * (1.0 - prob)).max(1e-16);
            }
        }
    }

    /// Base score = log(class_proportion) for each class.
    ///
    /// Returns log-priors in margin space (before softmax).
    /// For balanced classes, this returns log(1/K) for all classes.
    fn init_base_score(&self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
        let num_classes = self.num_classes;
        if labels.is_empty() {
            return vec![0.0; num_classes];
        }

        // Count weighted class frequencies
        let mut class_weights = vec![0.0f64; num_classes];
        let mut total_weight = 0.0f64;

        match weights {
            Some(w) => {
                for (&label, &wt) in labels.iter().zip(w.iter()) {
                    let class_idx = label.round() as usize;
                    if class_idx < num_classes {
                        class_weights[class_idx] += wt as f64;
                        total_weight += wt as f64;
                    }
                }
            }
            None => {
                for &label in labels {
                    let class_idx = label.round() as usize;
                    if class_idx < num_classes {
                        class_weights[class_idx] += 1.0;
                        total_weight += 1.0;
                    }
                }
            }
        }

        if total_weight == 0.0 {
            return vec![0.0; num_classes];
        }

        // Convert to log-probabilities
        class_weights
            .iter()
            .map(|&count| {
                let prob = (count / total_weight).max(1e-7);
                prob.ln() as f32
            })
            .collect()
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
        assert_eq!(SoftmaxLoss::new(3).name(), "softmax");
    }

    #[test]
    fn softmax_loss_num_outputs() {
        assert_eq!(SoftmaxLoss::new(3).num_outputs(), 3);
        assert_eq!(SoftmaxLoss::new(10).num_outputs(), 10);
    }
}
