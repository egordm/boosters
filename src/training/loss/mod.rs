//! Loss functions for computing gradients.
//!
//! Each loss function computes gradient-hessian pairs for optimization.
//! These are used by both GBLinear and GBTree training.
//!
//! # Loss Trait
//!
//! All losses implement the unified [`Loss`] trait which provides:
//! - `num_outputs()`: Number of outputs per sample (1 for regression/binary, K for multiclass)
//! - `compute_gradients()`: Batch gradient computation
//! - `name()`: Loss name for logging
//!
//! # Gradient Storage
//!
//! Gradients are stored in [`GradientBuffer`](super::GradientBuffer) (Structure-of-Arrays layout):
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
//!
//! # Available Losses
//!
//! ## Regression (num_outputs = 1)
//! - [`SquaredLoss`]: Standard squared error (L2 loss)
//! - [`PseudoHuberLoss`]: Robust regression, smooth approximation of Huber loss
//! - [`QuantileLoss`]: Quantile regression - supports single or multi-quantile
//!
//! ## Classification
//! - [`LogisticLoss`]: Binary classification (log loss) - num_outputs = 1
//! - [`HingeLoss`]: SVM-style binary classification - num_outputs = 1
//! - [`SoftmaxLoss`]: Multiclass classification - num_outputs = K classes

mod classification;
mod regression;

pub use classification::{HingeLoss, LogisticLoss, SoftmaxLoss};
pub use regression::{PseudoHuberLoss, QuantileLoss, SquaredLoss};

use super::GradientBuffer;

// =============================================================================
// LossFunction Enum (Unified)
// =============================================================================

/// Unified loss function enum for all training scenarios.
///
/// This enum provides a convenient way to specify loss functions without
/// using trait objects. For custom losses, use the [`Loss`] trait directly.
///
/// # Single-Output Losses (num_outputs = 1)
///
/// - `SquaredError`: Regression (L2 loss)
/// - `Logistic`: Binary classification
/// - `Hinge`: SVM-style binary classification
/// - `PseudoHuber`: Robust regression
/// - `Quantile`: Single quantile regression
///
/// # Multi-Output Losses (num_outputs = K)
///
/// - `Softmax`: Multiclass classification (K classes)
/// - `MultiQuantile`: Multi-quantile regression (K quantiles)
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::LossFunction;
///
/// // Regression (default)
/// let loss = LossFunction::SquaredError;
///
/// // Binary classification
/// let loss = LossFunction::Logistic;
///
/// // Multiclass classification (3 classes)
/// let loss = LossFunction::Softmax { num_classes: 3 };
///
/// // Multi-quantile regression
/// let loss = LossFunction::MultiQuantile { alphas: vec![0.1, 0.5, 0.9] };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum LossFunction {
    // ========================================================================
    // Single-output losses (num_outputs = 1)
    // ========================================================================
    /// Squared error loss for regression (L2 loss).
    SquaredError,
    /// Logistic loss for binary classification.
    Logistic,
    /// Hinge loss for SVM-style binary classification.
    Hinge,
    /// Pseudo-Huber loss for robust regression.
    PseudoHuber {
        /// Slope parameter controlling outlier sensitivity (default: 1.0).
        slope: f32,
    },
    /// Quantile loss for single quantile regression.
    Quantile {
        /// Quantile level in (0, 1). Use 0.5 for median regression.
        alpha: f32,
    },

    // ========================================================================
    // Multi-output losses (num_outputs = K)
    // ========================================================================
    /// Softmax cross-entropy for multiclass classification.
    Softmax {
        /// Number of classes (must be >= 2).
        num_classes: usize,
    },
    /// Multi-quantile loss for predicting multiple quantiles simultaneously.
    MultiQuantile {
        /// Quantile levels, each in (0, 1).
        alphas: Vec<f32>,
    },
}

impl Default for LossFunction {
    fn default() -> Self {
        Self::SquaredError
    }
}

impl Loss for LossFunction {
    fn num_outputs(&self) -> usize {
        match self {
            // Single-output
            LossFunction::SquaredError => 1,
            LossFunction::Logistic => 1,
            LossFunction::Hinge => 1,
            LossFunction::PseudoHuber { .. } => 1,
            LossFunction::Quantile { .. } => 1,
            // Multi-output
            LossFunction::Softmax { num_classes } => *num_classes,
            LossFunction::MultiQuantile { alphas } => alphas.len(),
        }
    }

    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
        match self {
            LossFunction::SquaredError => SquaredLoss.compute_gradients(preds, labels, buffer),
            LossFunction::Logistic => LogisticLoss.compute_gradients(preds, labels, buffer),
            LossFunction::Hinge => HingeLoss.compute_gradients(preds, labels, buffer),
            LossFunction::PseudoHuber { slope } => {
                PseudoHuberLoss::new(*slope).compute_gradients(preds, labels, buffer)
            }
            LossFunction::Quantile { alpha } => {
                QuantileLoss::new(*alpha).compute_gradients(preds, labels, buffer)
            }
            LossFunction::Softmax { num_classes } => {
                SoftmaxLoss::new(*num_classes).compute_gradients(preds, labels, buffer)
            }
            LossFunction::MultiQuantile { alphas } => {
                QuantileLoss::multi(alphas).compute_gradients(preds, labels, buffer)
            }
        }
    }

    fn init_base_score(&self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
        match self {
            LossFunction::SquaredError => SquaredLoss.init_base_score(labels, weights),
            LossFunction::Logistic => LogisticLoss.init_base_score(labels, weights),
            LossFunction::Hinge => HingeLoss.init_base_score(labels, weights),
            LossFunction::PseudoHuber { slope } => {
                PseudoHuberLoss::new(*slope).init_base_score(labels, weights)
            }
            LossFunction::Quantile { alpha } => {
                QuantileLoss::new(*alpha).init_base_score(labels, weights)
            }
            LossFunction::Softmax { num_classes } => {
                SoftmaxLoss::new(*num_classes).init_base_score(labels, weights)
            }
            LossFunction::MultiQuantile { alphas } => {
                QuantileLoss::multi(alphas).init_base_score(labels, weights)
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            LossFunction::SquaredError => "squared_error",
            LossFunction::Logistic => "logistic",
            LossFunction::Hinge => "hinge",
            LossFunction::PseudoHuber { .. } => "pseudo_huber",
            LossFunction::Quantile { .. } => "quantile",
            LossFunction::Softmax { .. } => "softmax",
            LossFunction::MultiQuantile { .. } => "multi_quantile",
        }
    }
}

// =============================================================================
// Loss Trait (Unified)
// =============================================================================

/// A loss function for gradient boosting training.
///
/// This unified trait supports both single-output (regression, binary classification)
/// and multi-output (multiclass, multi-quantile) losses through the `num_outputs()` method.
///
/// # Single vs Multi-Output
///
/// - Single-output losses (e.g., `SquaredLoss`, `LogisticLoss`): `num_outputs() = 1`
/// - Multi-output losses (e.g., `SoftmaxLoss`, multi-quantile): `num_outputs() = K`
///
/// # Gradient Layout
///
/// For N samples and K outputs, gradients are stored in the buffer as:
/// `buffer.grads[sample_idx * num_outputs + output_idx]`
///
/// # Implementing Custom Losses
///
/// ```ignore
/// impl Loss for MyLoss {
///     fn num_outputs(&self) -> usize { 1 } // or K for multi-output
///
///     fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer) {
///         let (grads, hess) = buffer.as_mut_slices();
///         for i in 0..preds.len() {
///             grads[i] = /* your gradient */;
///             hess[i] = /* your hessian */;
///         }
///     }
///
///     fn name(&self) -> &'static str { "my_loss" }
/// }
/// ```
pub trait Loss: Send + Sync {
    /// Number of outputs per sample.
    ///
    /// - Returns 1 for single-output losses (regression, binary classification)
    /// - Returns K for multi-output losses (K classes or K quantiles)
    fn num_outputs(&self) -> usize;

    /// Compute gradients and hessians for a batch of samples.
    ///
    /// This is the primary method for training. Implementations should write
    /// gradients and hessians directly to the buffer for best performance.
    ///
    /// # Arguments
    ///
    /// * `preds` - Predictions, length = n_samples × num_outputs
    /// * `labels` - Labels, length = n_samples
    /// * `buffer` - Output buffer with `n_samples` samples and `n_outputs == num_outputs()`
    ///
    /// # Panics
    ///
    /// Panics if buffer dimensions don't match input lengths.
    fn compute_gradients(&self, preds: &[f32], labels: &[f32], buffer: &mut GradientBuffer);

    /// Compute initial base scores for this objective.
    ///
    /// Base scores are the starting predictions before any trees are added.
    /// Proper initialization improves convergence, especially for imbalanced data.
    ///
    /// # Returns
    ///
    /// One score per output. For multi-output models (softmax, multi-quantile),
    /// returns `num_outputs()` values.
    fn init_base_score(&self, labels: &[f32], weights: Option<&[f32]>) -> Vec<f32>;

    /// Name of the loss function (for logging).
    fn name(&self) -> &'static str;
}
