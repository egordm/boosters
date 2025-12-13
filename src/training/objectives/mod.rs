//! Objective (loss) functions for gradient boosting.
//!
//! This module provides loss functions that compute gradients and hessians
//! for training gradient boosted models.
//!
//! # Multi-Output Support
//!
//! All objectives support multi-output by default. Data is stored in
//! **column-major** order: `[output0_row0, output0_row1, ..., output0_rowN, output1_row0, ...]`
//!
//! For example, with 3 rows and 2 outputs:
//! - `predictions[0..3]` = output 0 for all rows
//! - `predictions[3..6]` = output 1 for all rows
//!
//! # Weighted Training
//!
//! All objectives support sample weights via the `weights` parameter.
//! Pass an empty slice `&[]` for unweighted computation.
//!
//! # Available Objectives
//!
//! ## Regression
//! - [`SquaredLoss`]: Standard squared error (L2 loss)
//! - [`AbsoluteLoss`]: Mean absolute error (L1 loss)
//! - [`PinballLoss`]: Pinball loss for quantile regression
//! - [`PseudoHuberLoss`]: Robust regression with configurable delta
//! - [`PoissonLoss`]: Poisson regression for count data
//!
//! ## Classification
//! - [`LogisticLoss`]: Binary classification (log loss / cross-entropy)
//! - [`HingeLoss`]: SVM-style binary classification
//! - [`SoftmaxLoss`]: Multiclass classification (softmax cross-entropy)
//!
//! ## Ranking
//! - [`LambdaRankLoss`]: LambdaMART for learning to rank (NDCG optimization)

mod classification;
mod regression;

pub use classification::{HingeLoss, LambdaRankLoss, LogisticLoss, SoftmaxLoss};
pub use regression::{AbsoluteLoss, PinballLoss, PoissonLoss, PseudoHuberLoss, SquaredLoss};

// =============================================================================
// Helpers
// =============================================================================

/// Returns an iterator over weights, using 1.0 for empty weights.
#[inline]
fn weight_iter(weights: &[f32], n_rows: usize) -> impl Iterator<Item = f32> + '_ {
    let use_weights = !weights.is_empty();
    (0..n_rows).map(move |i| if use_weights { weights[i] } else { 1.0 })
}

/// Validate objective input parameters.
///
/// Panics with a descriptive message if inputs are invalid.
#[inline]
fn validate_objective_inputs(
    n_rows: usize,
    n_outputs: usize,
    predictions_len: usize,
    gradients_len: usize,
    hessians_len: usize,
    weights: &[f32],
) {
    assert!(
        n_rows > 0 && n_outputs > 0,
        "n_rows ({}) and n_outputs ({}) must be positive",
        n_rows,
        n_outputs
    );
    let required = n_rows * n_outputs;
    assert!(
        predictions_len >= required,
        "predictions.len() ({}) < n_rows * n_outputs ({})",
        predictions_len,
        required
    );
    assert!(
        gradients_len >= required,
        "gradients.len() ({}) < n_rows * n_outputs ({})",
        gradients_len,
        required
    );
    assert!(
        hessians_len >= required,
        "hessians.len() ({}) < n_rows * n_outputs ({})",
        hessians_len,
        required
    );
    assert!(
        weights.is_empty() || weights.len() >= n_rows,
        "weights.len() ({}) < n_rows ({})",
        weights.len(),
        n_rows
    );
}

// =============================================================================
// Objective Trait
// =============================================================================

/// An objective (loss) function for training gradient boosted models.
///
/// Objectives compute gradients and hessians for optimization.
///
/// # Multi-Output Layout
///
/// All data is stored in **column-major** order:
/// - `predictions`: `[n_outputs * n_rows]` - each output's predictions are contiguous
/// - `targets`: `[n_targets * n_rows]` - each target's values are contiguous
/// - `gradients/hessians`: `[n_outputs * n_rows]` - matching predictions layout
///
/// The relationship between `n_outputs` and `n_targets` depends on the objective:
/// - SquaredLoss: `n_outputs == n_targets` (1:1 mapping)
/// - PinballLoss: `n_outputs` quantiles, `n_targets` can be 1 (shared) or `n_outputs`
/// - SoftmaxLoss: `n_outputs` = num_classes, `n_targets` = 1 (class indices)
///
/// # Weighted Training
///
/// The `weights` slice can be empty for unweighted training.
/// When non-empty, it must have length `n_rows`.
pub trait Objective: Send + Sync {
    /// Number of outputs (predictions per sample).
    ///
    /// For most objectives this is 1 (single-output).
    /// For multiclass (SoftmaxLoss) this is num_classes.
    /// For multi-quantile this is the number of quantiles.
    fn n_outputs(&self) -> usize {
        1
    }

    /// Compute gradients and hessians for the given predictions.
    ///
    /// # Arguments
    ///
    /// * `n_rows` - Number of samples
    /// * `n_outputs` - Number of outputs (predictions per sample)
    /// * `predictions` - Model predictions, column-major `[n_outputs * n_rows]`
    /// * `targets` - Ground truth labels, column-major `[n_targets * n_rows]`
    /// * `weights` - Sample weights (empty slice for unweighted)
    /// * `gradients` - Output gradients, column-major `[n_outputs * n_rows]`
    /// * `hessians` - Output hessians, column-major `[n_outputs * n_rows]`
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    );

    /// Compute the initial base score (bias) from targets.
    ///
    /// This is the optimal constant prediction before any trees are added.
    ///
    /// # Arguments
    ///
    /// * `n_rows` - Number of samples
    /// * `n_outputs` - Number of outputs
    /// * `targets` - Ground truth labels, column-major
    /// * `weights` - Sample weights (empty slice for unweighted)
    /// * `outputs` - Output base scores, length `n_outputs`
    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],
    );

    /// Name of the objective (for logging).
    fn name(&self) -> &'static str;

    // =========================================================================
    // Convenience methods for trainer integration
    // =========================================================================

    /// Convenience alias for n_outputs().
    #[inline]
    fn num_outputs(&self) -> usize {
        self.n_outputs()
    }

    /// Convenience method: compute initial base scores and return as Vec.
    ///
    /// This is a wrapper around `compute_base_score` that allocates the output.
    fn init_base_score(&self, targets: &[f32], weights: Option<&[f32]>) -> Vec<f32> {
        let n_outputs = self.n_outputs();
        let n_rows = targets.len();
        let mut base_scores = vec![0.0f32; n_outputs];
        let w = weights.unwrap_or(&[]);
        self.compute_base_score(n_rows, n_outputs, targets, w, &mut base_scores);
        base_scores
    }
}

use crate::training::Gradients;

/// Extension trait for objective functions that work with Gradients.
///
/// This provides a convenient interface for trainers that use Gradients
/// for gradient storage.
pub trait ObjectiveExt: Objective {
    /// Compute gradients into a Gradients.
    ///
    /// This is a convenience wrapper that extracts the mutable slices from
    /// the buffer and calls the underlying compute_gradients.
    fn compute_gradients_buffer(
        &self,
        predictions: &[f32],
        targets: &[f32],
        weights: Option<&[f32]>,
        buffer: &mut Gradients,
    ) {
        let n_rows = buffer.n_samples();
        let n_outputs = buffer.n_outputs();
        let (grads, hess) = buffer.as_mut_slices();
        let w = weights.unwrap_or(&[]);
        self.compute_gradients(n_rows, n_outputs, predictions, targets, w, grads, hess);
    }
}

// Implement ObjectiveExt for all Objective types
impl<T: Objective + ?Sized> ObjectiveExt for T {}

// =============================================================================
// ObjectiveFunction Enum (Convenience wrapper)
// =============================================================================

/// Objective function enum for easy configuration.
///
/// This enum wraps all available objective types and provides a unified
/// interface for trainers. It implements `Objective` by delegating to the
/// underlying concrete type.
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::{GBTreeTrainer, ObjectiveFunction};
///
/// let trainer = GBTreeTrainer::builder()
///     .objective(ObjectiveFunction::Logistic)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveFunction {
    /// Squared error loss (L2) for regression.
    SquaredError,
    /// Absolute error loss (L1) for robust regression.
    AbsoluteError,
    /// Logistic loss for binary classification.
    Logistic,
    /// Hinge loss for SVM-style classification.
    Hinge,
    /// Softmax loss for multiclass classification.
    Softmax { num_classes: usize },
    /// Pinball loss for quantile regression.
    Quantile { alpha: f32 },
    /// Multi-quantile regression.
    MultiQuantile { alphas: Vec<f32> },
    /// Pseudo-Huber loss for robust regression.
    PseudoHuber { delta: f32 },
    /// Poisson regression for count data.
    Poisson,
}

impl Default for ObjectiveFunction {
    fn default() -> Self {
        Self::SquaredError
    }
}

impl Objective for ObjectiveFunction {
    fn n_outputs(&self) -> usize {
        match self {
            Self::SquaredError => 1,
            Self::AbsoluteError => 1,
            Self::Logistic => 1,
            Self::Hinge => 1,
            Self::Softmax { num_classes } => *num_classes,
            Self::Quantile { .. } => 1,
            Self::MultiQuantile { alphas } => alphas.len(),
            Self::PseudoHuber { .. } => 1,
            Self::Poisson => 1,
        }
    }

    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        match self {
            Self::SquaredError => {
                SquaredLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
            Self::AbsoluteError => {
                AbsoluteLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
            Self::Logistic => {
                LogisticLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
            Self::Hinge => {
                HingeLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
            Self::Softmax { num_classes } => {
                SoftmaxLoss::new(*num_classes).compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
            Self::Quantile { alpha } => {
                PinballLoss::new(*alpha).compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
            Self::MultiQuantile { alphas } => {
                PinballLoss::with_quantiles(alphas.clone()).compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
            Self::PseudoHuber { delta } => {
                PseudoHuberLoss::new(*delta).compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
            Self::Poisson => {
                PoissonLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, gradients, hessians)
            }
        }
    }

    fn compute_base_score(
        &self,
        n_rows: usize,
        n_outputs: usize,
        targets: &[f32],
        weights: &[f32],
        outputs: &mut [f32],
    ) {
        match self {
            Self::SquaredError => {
                SquaredLoss.compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
            Self::AbsoluteError => {
                AbsoluteLoss.compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
            Self::Logistic => {
                LogisticLoss.compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
            Self::Hinge => {
                HingeLoss.compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
            Self::Softmax { num_classes } => {
                SoftmaxLoss::new(*num_classes).compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
            Self::Quantile { alpha } => {
                PinballLoss::new(*alpha).compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
            Self::MultiQuantile { alphas } => {
                PinballLoss::with_quantiles(alphas.clone()).compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
            Self::PseudoHuber { delta } => {
                PseudoHuberLoss::new(*delta).compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
            Self::Poisson => {
                PoissonLoss.compute_base_score(n_rows, n_outputs, targets, weights, outputs)
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::SquaredError => "squared",
            Self::AbsoluteError => "absolute",
            Self::Logistic => "logistic",
            Self::Hinge => "hinge",
            Self::Softmax { .. } => "softmax",
            Self::Quantile { .. } => "quantile",
            Self::MultiQuantile { .. } => "multi_quantile",
            Self::PseudoHuber { .. } => "pseudo_huber",
            Self::Poisson => "poisson",
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
    fn squared_loss_gradients() {
        let obj = SquaredLoss;
        let preds = [1.0f32, 2.0, 3.0];
        let targets = [0.5f32, 2.5, 2.5];
        let mut grads = [0.0f32; 3];
        let mut hess = [0.0f32; 3];

        obj.compute_gradients(3, 1, &preds, &targets, &[], &mut grads, &mut hess);

        // grad = pred - target
        assert!((grads[0] - 0.5).abs() < 1e-6);
        assert!((grads[1] - -0.5).abs() < 1e-6);
        assert!((grads[2] - 0.5).abs() < 1e-6);

        // hess = 1.0
        assert!((hess[0] - 1.0).abs() < 1e-6);
        assert!((hess[1] - 1.0).abs() < 1e-6);
        assert!((hess[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_squared_loss() {
        let obj = SquaredLoss;
        let preds = [1.0f32, 2.0];
        let targets = [0.5f32, 2.5];
        let weights = [2.0f32, 0.5];
        let mut grads = [0.0f32; 2];
        let mut hess = [0.0f32; 2];

        obj.compute_gradients(2, 1, &preds, &targets, &weights, &mut grads, &mut hess);

        // grad = weight * (pred - target)
        assert!((grads[0] - 1.0).abs() < 1e-6); // 2.0 * 0.5
        assert!((grads[1] - -0.25).abs() < 1e-6); // 0.5 * -0.5

        // hess = weight
        assert!((hess[0] - 2.0).abs() < 1e-6);
        assert!((hess[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pinball_loss_median() {
        let obj = PinballLoss::new(0.5);
        let preds = [1.0f32, 2.0, 3.0];
        let targets = [0.5f32, 2.5, 2.5];
        let mut grads = [0.0f32; 3];
        let mut hess = [0.0f32; 3];

        obj.compute_gradients(3, 1, &preds, &targets, &[], &mut grads, &mut hess);

        // For alpha=0.5: grad = 0.5 if pred > target, -0.5 if pred < target
        assert!((grads[0] - 0.5).abs() < 1e-6); // pred > target
        assert!((grads[1] - -0.5).abs() < 1e-6); // pred < target
        assert!((grads[2] - 0.5).abs() < 1e-6); // pred > target
    }
}
