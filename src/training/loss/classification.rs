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
    fn compute_gradients(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        buffer: &mut GradientBuffer,
    ) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), buffer.n_samples());
        debug_assert_eq!(buffer.n_outputs(), 1);

        let (grads, hess) = buffer.as_mut_slices();

        match weights {
            Some(w) => {
                debug_assert_eq!(w.len(), preds.len());
                for i in 0..preds.len() {
                    let p = 1.0 / (1.0 + (-preds[i]).exp());
                    grads[i] = w[i] * (p - labels[i]);
                    hess[i] = w[i] * (p * (1.0 - p)).max(1e-16);
                }
            }
            None => {
                // Single pass: compute sigmoid, grad, and hess together
                // This keeps p in register for both grad and hess computation
                for i in 0..preds.len() {
                    let p = 1.0 / (1.0 + (-preds[i]).exp());
                    grads[i] = p - labels[i];
                    hess[i] = (p * (1.0 - p)).max(1e-16);
                }
            }
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
    fn compute_gradients(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        buffer: &mut GradientBuffer,
    ) {
        debug_assert_eq!(preds.len(), labels.len());
        debug_assert_eq!(preds.len(), buffer.n_samples());
        debug_assert_eq!(buffer.n_outputs(), 1);

        let (grads, hess) = buffer.as_mut_slices();

        match weights {
            Some(w) => {
                debug_assert_eq!(w.len(), preds.len());
                for i in 0..preds.len() {
                    let p = preds[i];
                    // Convert label from {0, 1} to {-1, 1}
                    let y = labels[i] * 2.0 - 1.0;

                    if p * y < 1.0 {
                        // Wrong side of margin (or within margin)
                        grads[i] = w[i] * (-y);
                        hess[i] = w[i];
                    } else {
                        // Correct side of margin
                        grads[i] = 0.0;
                        hess[i] = w[i] * f32::MIN_POSITIVE; // Small positive value
                    }
                }
            }
            None => {
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
    ///
    /// Predictions are column-major: index = class * num_samples + sample
    fn compute_gradients(
        &self,
        preds: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        buffer: &mut GradientBuffer,
    ) {
        let num_classes = self.num_classes;
        let num_samples = labels.len();
        debug_assert_eq!(preds.len(), num_samples * num_classes);
        debug_assert_eq!(buffer.n_samples(), num_samples);
        debug_assert_eq!(buffer.n_outputs(), num_classes);

        // Column-major predictions: perfect access pattern for multi-pass softmax
        // Pass 1: Find max per sample (for numerical stability)
        let mut max_per_sample = vec![f32::NEG_INFINITY; num_samples];
        for k in 0..num_classes {
            let preds_k = &preds[k * num_samples..(k + 1) * num_samples];
            for i in 0..num_samples {
                max_per_sample[i] = max_per_sample[i].max(preds_k[i]);
            }
        }

        // Pass 2: Compute exp, store in hess (as temp), accumulate sum
        let mut sum_exp = vec![0.0f32; num_samples];
        for k in 0..num_classes {
            let preds_k = &preds[k * num_samples..(k + 1) * num_samples];
            let hess_k = buffer.output_hess_mut(k);
            for i in 0..num_samples {
                debug_assert!(
                    (labels[i] as usize) < num_classes,
                    "Label {} >= num_classes {}",
                    labels[i],
                    num_classes
                );
                let exp_val = (preds_k[i] - max_per_sample[i]).exp();
                hess_k[i] = exp_val;
                sum_exp[i] += exp_val;
            }
        }

        // Pass 3: Normalize (hess contains exp), compute grad and final hess
        // Use as_mut_slices to avoid borrow checker issues
        let (grads, hess) = buffer.as_mut_slices();

        match weights {
            Some(w) => {
                debug_assert_eq!(w.len(), num_samples);
                for k in 0..num_classes {
                    let start = k * num_samples;
                    for i in 0..num_samples {
                        let prob = hess[start + i] / sum_exp[i];
                        let is_true_class = if k == labels[i] as usize { 1.0 } else { 0.0 };
                        grads[start + i] = w[i] * (prob - is_true_class);
                        hess[start + i] = w[i] * (prob * (1.0 - prob)).max(1e-16);
                    }
                }
            }
            None => {
                for k in 0..num_classes {
                    let start = k * num_samples;
                    for i in 0..num_samples {
                        let prob = hess[start + i] / sum_exp[i];
                        let is_true_class = if k == labels[i] as usize { 1.0 } else { 0.0 };
                        grads[start + i] = prob - is_true_class;
                        hess[start + i] = (prob * (1.0 - prob)).max(1e-16);
                    }
                }
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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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
        // 2 samples, 3 classes each (column-major: predictions grouped by class)
        let preds = vec![
            1.0, 0.0, // class 0: sample 0, sample 1
            2.0, 0.0, // class 1: sample 0, sample 1
            3.0, 0.0, // class 2: sample 0, sample 1 (uniform for sample 1)
        ];
        let labels = vec![2.0, 1.0]; // true classes
        let mut buffer = GradientBuffer::new(2, 3);

        loss.compute_gradients(&preds, &labels, None, &mut buffer);

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

    // =========================================================================
    // Weighted gradient tests
    // =========================================================================

    #[test]
    fn logistic_loss_weighted_gradients() {
        let loss = LogisticLoss;
        let preds = vec![0.0, 0.0]; // sigmoid(0) = 0.5
        let labels = vec![1.0, 0.0];
        let weights = vec![2.0, 0.5];
        let mut buffer = GradientBuffer::new(2, 1);

        loss.compute_gradients(&preds, &labels, Some(&weights), &mut buffer);

        // Unweighted: grad = sigmoid(pred) - label = 0.5 - label
        // Sample 0: label=1, grad = 0.5 - 1 = -0.5
        // Sample 1: label=0, grad = 0.5 - 0 = 0.5
        assert!((buffer.grad(0, 0) - 2.0 * (-0.5)).abs() < 1e-6);
        assert!((buffer.grad(1, 0) - 0.5 * 0.5).abs() < 1e-6);

        // Hessians: hess = sigmoid(pred) * (1 - sigmoid(pred)) = 0.5 * 0.5 = 0.25
        assert!((buffer.hess(0, 0) - 2.0 * 0.25).abs() < 1e-6);
        assert!((buffer.hess(1, 0) - 0.5 * 0.25).abs() < 1e-6);
    }

    #[test]
    fn logistic_loss_uniform_weights_matches_unweighted() {
        let loss = LogisticLoss;
        let preds = vec![1.0, -1.0, 0.0, 2.0];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let uniform_weights = vec![1.0; 4];

        let mut buffer_unweighted = GradientBuffer::new(4, 1);
        let mut buffer_weighted = GradientBuffer::new(4, 1);

        loss.compute_gradients(&preds, &labels, None, &mut buffer_unweighted);
        loss.compute_gradients(&preds, &labels, Some(&uniform_weights), &mut buffer_weighted);

        for i in 0..4 {
            assert!(
                (buffer_weighted.grad(i, 0) - buffer_unweighted.grad(i, 0)).abs() < 1e-6,
                "grad mismatch at {}: weighted={}, unweighted={}",
                i,
                buffer_weighted.grad(i, 0),
                buffer_unweighted.grad(i, 0)
            );
            assert!(
                (buffer_weighted.hess(i, 0) - buffer_unweighted.hess(i, 0)).abs() < 1e-6,
                "hess mismatch at {}: weighted={}, unweighted={}",
                i,
                buffer_weighted.hess(i, 0),
                buffer_unweighted.hess(i, 0)
            );
        }
    }

    #[test]
    fn logistic_loss_zero_weight_produces_zero_gradient() {
        let loss = LogisticLoss;
        let preds = vec![10.0]; // Large value
        let labels = vec![0.0];
        let weights = vec![0.0];
        let mut buffer = GradientBuffer::new(1, 1);

        loss.compute_gradients(&preds, &labels, Some(&weights), &mut buffer);

        assert!(buffer.grad(0, 0).abs() < 1e-10);
        assert!(buffer.hess(0, 0).abs() < 1e-10);
    }

    #[test]
    fn hinge_loss_weighted_gradients() {
        let loss = HingeLoss;
        // Labels are in {0, 1} format, internally converted to {-1, 1}
        // Sample 0: label=1 → y=1, pred=0.5, margin = 1*0.5 = 0.5 < 1, grad = -1
        // Sample 1: label=0 → y=-1, pred=-0.5, margin = -1*-0.5 = 0.5 < 1, grad = 1
        // Sample 2: label=1 → y=1, pred=2.0, margin = 1*2 = 2 >= 1, grad = 0
        let preds = vec![0.5, -0.5, 2.0];
        let labels = vec![1.0, 0.0, 1.0];
        let weights = vec![2.0, 0.5, 1.0];
        let mut buffer = GradientBuffer::new(3, 1);

        loss.compute_gradients(&preds, &labels, Some(&weights), &mut buffer);

        // Sample 0: grad = w * -y = 2.0 * -1 = -2.0
        // Sample 1: grad = w * -y = 0.5 * -(-1) = 0.5
        // Sample 2: grad = 0 (margin >= 1)
        assert!((buffer.grad(0, 0) - (-2.0)).abs() < 1e-6);
        assert!((buffer.grad(1, 0) - 0.5).abs() < 1e-6);
        assert!(buffer.grad(2, 0).abs() < 1e-6);
    }

    #[test]
    fn softmax_loss_weighted_gradients() {
        let loss = SoftmaxLoss::new(3);
        // Use uniform predictions for simplicity
        let preds = vec![0.0, 0.0, 0.0]; // softmax = [1/3, 1/3, 1/3]
        let labels = vec![1.0]; // true class is 1
        let weights = vec![2.0];
        let mut buffer = GradientBuffer::new(1, 3);

        loss.compute_gradients(&preds, &labels, Some(&weights), &mut buffer);

        // Unweighted: grad[c] = softmax[c] - (c == label ? 1 : 0)
        // For class 0: grad = 1/3 - 0 = 1/3
        // For class 1: grad = 1/3 - 1 = -2/3
        // For class 2: grad = 1/3 - 0 = 1/3
        assert!((buffer.grad(0, 0) - 2.0 * (1.0 / 3.0)).abs() < 1e-5);
        assert!((buffer.grad(0, 1) - 2.0 * (-2.0 / 3.0)).abs() < 1e-5);
        assert!((buffer.grad(0, 2) - 2.0 * (1.0 / 3.0)).abs() < 1e-5);

        // Hessians: hess[c] = softmax[c] * (1 - softmax[c])
        // = (1/3) * (2/3) = 2/9
        let expected_hess = 2.0 * (1.0 / 3.0) * (2.0 / 3.0);
        assert!((buffer.hess(0, 0) - expected_hess).abs() < 1e-5);
        assert!((buffer.hess(0, 1) - expected_hess).abs() < 1e-5);
        assert!((buffer.hess(0, 2) - expected_hess).abs() < 1e-5);
    }

    #[test]
    fn softmax_loss_uniform_weights_matches_unweighted() {
        let loss = SoftmaxLoss::new(3);
        let preds = vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.0]; // 2 samples, 3 classes each
        let labels = vec![0.0, 2.0];
        let uniform_weights = vec![1.0, 1.0];

        let mut buffer_unweighted = GradientBuffer::new(2, 3);
        let mut buffer_weighted = GradientBuffer::new(2, 3);

        loss.compute_gradients(&preds, &labels, None, &mut buffer_unweighted);
        loss.compute_gradients(&preds, &labels, Some(&uniform_weights), &mut buffer_weighted);

        for sample in 0..2 {
            for class in 0..3 {
                assert!(
                    (buffer_weighted.grad(sample, class) - buffer_unweighted.grad(sample, class))
                        .abs()
                        < 1e-6,
                    "grad mismatch at ({}, {})",
                    sample,
                    class
                );
                assert!(
                    (buffer_weighted.hess(sample, class) - buffer_unweighted.hess(sample, class))
                        .abs()
                        < 1e-6,
                    "hess mismatch at ({}, {})",
                    sample,
                    class
                );
            }
        }
    }

    // =========================================================================
    // Weighted base score tests (RFC-0024)
    // =========================================================================

    #[test]
    fn logistic_loss_weighted_base_score() {
        let loss = LogisticLoss;
        // Labels: [1, 0, 1, 0], Weights: [2, 1, 1, 1]
        // Weighted positive rate = (2 + 1) / (2 + 1 + 1 + 1) = 3/5 = 0.6
        // Base score = log(0.6 / 0.4) = log(1.5) ≈ 0.405
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let weights = vec![2.0, 1.0, 1.0, 1.0];

        let base_score = loss.init_base_score(&labels, Some(&weights));
        assert_eq!(base_score.len(), 1);
        let expected = (0.6_f32 / 0.4).ln();
        assert!(
            (base_score[0] - expected).abs() < 1e-5,
            "expected {}, got {}",
            expected,
            base_score[0]
        );
    }

    #[test]
    fn logistic_loss_unweighted_base_score() {
        let loss = LogisticLoss;
        // Labels: [1, 0, 1, 1] → 3/4 positive
        // Base score = log(0.75 / 0.25) = log(3) ≈ 1.099
        let labels = vec![1.0, 0.0, 1.0, 1.0];

        let base_score = loss.init_base_score(&labels, None);
        assert_eq!(base_score.len(), 1);
        let expected = (0.75_f32 / 0.25).ln();
        assert!(
            (base_score[0] - expected).abs() < 1e-5,
            "expected {}, got {}",
            expected,
            base_score[0]
        );
    }

    #[test]
    fn softmax_loss_weighted_base_score() {
        let loss = SoftmaxLoss::new(3);
        // Labels: [0, 1, 2, 0], Weights: [2, 1, 1, 1]
        // Weighted class counts: [2+1, 1, 1] = [3, 1, 1], total = 5
        // Class probs: [0.6, 0.2, 0.2]
        let labels = vec![0.0, 1.0, 2.0, 0.0];
        let weights = vec![2.0, 1.0, 1.0, 1.0];

        let base_scores = loss.init_base_score(&labels, Some(&weights));
        assert_eq!(base_scores.len(), 3);

        // Base scores are log-probabilities
        let expected_0 = 0.6_f32.ln();
        let expected_1 = 0.2_f32.ln();
        let expected_2 = 0.2_f32.ln();

        assert!((base_scores[0] - expected_0).abs() < 1e-5);
        assert!((base_scores[1] - expected_1).abs() < 1e-5);
        assert!((base_scores[2] - expected_2).abs() < 1e-5);
    }

    #[test]
    fn softmax_loss_uniform_weights_base_score_matches_unweighted() {
        let loss = SoftmaxLoss::new(4);
        let labels = vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0];
        let uniform_weights = vec![1.0; 6];

        let unweighted = loss.init_base_score(&labels, None);
        let weighted = loss.init_base_score(&labels, Some(&uniform_weights));

        for (u, w) in unweighted.iter().zip(weighted.iter()) {
            assert!((u - w).abs() < 1e-6);
        }
    }
}
