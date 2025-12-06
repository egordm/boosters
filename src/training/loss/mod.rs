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
//! ## Regression
//! - [`SquaredLoss`]: Standard squared error (L2 loss)
//! - [`PseudoHuberLoss`]: Robust regression, smooth approximation of Huber loss
//! - [`QuantileLoss`]: Quantile regression (pinball loss)
//!
//! ## Classification  
//! - [`LogisticLoss`]: Binary classification (log loss)
//! - [`HingeLoss`]: SVM-style binary classification
//! - [`SoftmaxLoss`]: Multiclass classification (cross-entropy)

mod classification;
mod regression;

pub use classification::{HingeLoss, LogisticLoss, SoftmaxLoss};
pub use regression::{PseudoHuberLoss, QuantileLoss, SquaredLoss};

use super::GradientBuffer;

// =============================================================================
// Loss Traits
// =============================================================================

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
