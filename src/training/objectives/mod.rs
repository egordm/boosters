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

use crate::inference::common::{PredictionKind, PredictionOutput, Predictions};
use crate::training::metrics::MetricKind;
use crate::training::GradsTuple;

// =============================================================================
// RFC 0028: Task + Target Semantics
// =============================================================================

/// High-level task kind implied by an objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Regression,
    BinaryClassification,
    MulticlassClassification,
    Ranking,
}

/// Target encoding/schema expected by an objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSchema {
    /// Continuous real-valued targets.
    Continuous,
    /// Binary targets encoded as {0, 1}.
    Binary01,
    /// Binary targets encoded as {-1, +1} (or convertible from {0, 1}).
    BinarySigned,
    /// Multiclass targets encoded as a single class index in [0, K).
    MulticlassIndex,
    /// Non-negative counts (Poisson-style).
    CountNonNegative,
}

// =============================================================================
// Helpers
// =============================================================================

// Re-export weight_iter from utils for internal use
pub(super) use crate::utils::weight_iter;

/// Validate objective input parameters.
///
/// Panics with a descriptive message if inputs are invalid.
#[inline]
fn validate_objective_inputs(
    n_rows: usize,
    n_outputs: usize,
    predictions_len: usize,
    grad_hess_len: usize,
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
        grad_hess_len >= required,
        "grad_hess.len() ({}) < n_rows * n_outputs ({})",
        grad_hess_len,
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
    /// * `grad_hess` - Interleaved (grad, hess) pairs, column-major `[n_outputs * n_rows]`
    fn compute_gradients(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        grad_hess: &mut [GradsTuple],
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

    // =========================================================================
    // RFC 0028: Semantics + Prediction Transforms
    // =========================================================================

    /// High-level task kind implied by this objective.
    fn task_kind(&self) -> TaskKind {
        TaskKind::Regression
    }

    /// Target encoding/schema expected by this objective.
    fn target_schema(&self) -> TargetSchema {
        TargetSchema::Continuous
    }

    /// Default metric suggested by this objective.
    fn default_metric(&self) -> MetricKind {
        MetricKind::Rmse
    }

    /// Transform raw predictions in-place (column-major layout).
    ///
    /// Converts margins/logits to semantic predictions (probabilities, values, etc.).
    /// Returns the semantic kind of the resulting values.
    ///
    /// Layout: `predictions[output * n_rows + row]`
    ///
    /// Most regression objectives are no-ops. Classification objectives apply
    /// sigmoid (binary) or softmax (multiclass).
    fn transform_predictions(&self, _predictions: &mut [f32], _n_rows: usize, _n_outputs: usize) -> PredictionKind {
        PredictionKind::Value
    }

    /// Transform a [`PredictionOutput`] in-place.
    ///
    /// Convenience wrapper around [`transform_predictions`](Self::transform_predictions).
    fn transform_prediction_inplace(&self, raw: &mut PredictionOutput) -> PredictionKind {
        let n_rows = raw.num_rows();
        let n_groups = raw.num_groups();
        self.transform_predictions(raw.as_mut_slice(), n_rows, n_groups)
    }

    /// Transform raw model outputs (margins/logits) into semantic predictions.
    ///
    /// This consumes `raw` and returns an explicitly-labeled prediction output.
    fn transform_prediction(&self, mut raw: PredictionOutput) -> Predictions {
        let kind = self.transform_prediction_inplace(&mut raw);
        Predictions { kind, output: raw }
    }

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

    /// Compute initial base scores (optimal constant prediction).
    ///
    /// Convenience wrapper around `compute_base_score` that allocates the output.
    ///
    /// # Arguments
    ///
    /// * `n_rows` - Number of samples
    /// * `targets` - Ground truth labels
    /// * `weights` - Sample weights (empty slice for unweighted)
    fn base_scores(&self, n_rows: usize, targets: &[f32], weights: &[f32]) -> Vec<f32> {
        let n_outputs = self.n_outputs();
        let mut output = vec![0.0f32; n_outputs];
        self.compute_base_score(n_rows, n_outputs, targets, weights, &mut output);
        output
    }

    /// Fill a column-major prediction buffer with computed base scores.
    ///
    /// Useful for initializing prediction buffers before tree accumulation.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Column-major buffer to fill `[n_outputs * n_rows]`
    /// * `n_rows` - Number of samples
    /// * `targets` - Ground truth labels
    /// * `weights` - Sample weights (empty slice for unweighted)
    fn fill_base_scores(
        &self,
        predictions: &mut [f32],
        n_rows: usize,
        targets: &[f32],
        weights: &[f32],
    ) {
        let n_outputs = self.n_outputs();
        let mut base_scores = vec![0.0f32; n_outputs];
        self.compute_base_score(n_rows, n_outputs, targets, weights, &mut base_scores);

        // Fill column-major: each output column gets its base score
        for (out_idx, &score) in base_scores.iter().enumerate() {
            let start = out_idx * n_rows;
            predictions[start..start + n_rows].fill(score);
        }
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
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions
    /// * `targets` - Ground truth labels
    /// * `weights` - Sample weights (empty slice for unweighted)
    /// * `buffer` - Gradient buffer to fill
    fn compute_gradients_buffer(
        &self,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        buffer: &mut Gradients,
    ) {
        let n_rows = buffer.n_samples();
        let n_outputs = buffer.n_outputs();
        let grad_hess = buffer.pairs_mut();
        self.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess);
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
/// use boosters::training::{GBTreeTrainer, ObjectiveFunction};
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
        grad_hess: &mut [GradsTuple],
    ) {
        match self {
            Self::SquaredError => {
                SquaredLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
            }
            Self::AbsoluteError => {
                AbsoluteLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
            }
            Self::Logistic => {
                LogisticLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
            }
            Self::Hinge => {
                HingeLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
            }
            Self::Softmax { num_classes } => {
                SoftmaxLoss::new(*num_classes).compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
            }
            Self::Quantile { alpha } => {
                PinballLoss::new(*alpha).compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
            }
            Self::MultiQuantile { alphas } => {
                PinballLoss::with_quantiles(alphas.clone()).compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
            }
            Self::PseudoHuber { delta } => {
                PseudoHuberLoss::new(*delta).compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
            }
            Self::Poisson => {
                PoissonLoss.compute_gradients(n_rows, n_outputs, predictions, targets, weights, grad_hess)
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

    fn task_kind(&self) -> TaskKind {
        match self {
            Self::SquaredError => SquaredLoss.task_kind(),
            Self::AbsoluteError => AbsoluteLoss.task_kind(),
            Self::Logistic => LogisticLoss.task_kind(),
            Self::Hinge => HingeLoss.task_kind(),
            Self::Softmax { num_classes } => SoftmaxLoss::new(*num_classes).task_kind(),
            Self::Quantile { alpha } => PinballLoss::new(*alpha).task_kind(),
            Self::MultiQuantile { alphas } => PinballLoss::with_quantiles(alphas.clone()).task_kind(),
            Self::PseudoHuber { delta } => PseudoHuberLoss::new(*delta).task_kind(),
            Self::Poisson => PoissonLoss.task_kind(),
        }
    }

    fn target_schema(&self) -> TargetSchema {
        match self {
            Self::SquaredError => SquaredLoss.target_schema(),
            Self::AbsoluteError => AbsoluteLoss.target_schema(),
            Self::Logistic => LogisticLoss.target_schema(),
            Self::Hinge => HingeLoss.target_schema(),
            Self::Softmax { num_classes } => SoftmaxLoss::new(*num_classes).target_schema(),
            Self::Quantile { alpha } => PinballLoss::new(*alpha).target_schema(),
            Self::MultiQuantile { alphas } => PinballLoss::with_quantiles(alphas.clone()).target_schema(),
            Self::PseudoHuber { delta } => PseudoHuberLoss::new(*delta).target_schema(),
            Self::Poisson => PoissonLoss.target_schema(),
        }
    }

    fn default_metric(&self) -> MetricKind {
        match self {
            Self::SquaredError => SquaredLoss.default_metric(),
            Self::AbsoluteError => AbsoluteLoss.default_metric(),
            Self::Logistic => LogisticLoss.default_metric(),
            Self::Hinge => HingeLoss.default_metric(),
            Self::Softmax { num_classes } => SoftmaxLoss::new(*num_classes).default_metric(),
            Self::Quantile { alpha } => PinballLoss::new(*alpha).default_metric(),
            Self::MultiQuantile { alphas } => PinballLoss::with_quantiles(alphas.clone()).default_metric(),
            Self::PseudoHuber { delta } => PseudoHuberLoss::new(*delta).default_metric(),
            Self::Poisson => PoissonLoss.default_metric(),
        }
    }

    fn transform_predictions(&self, predictions: &mut [f32], n_rows: usize, n_outputs: usize) -> PredictionKind {
        match self {
            Self::SquaredError => SquaredLoss.transform_predictions(predictions, n_rows, n_outputs),
            Self::AbsoluteError => AbsoluteLoss.transform_predictions(predictions, n_rows, n_outputs),
            Self::Logistic => LogisticLoss.transform_predictions(predictions, n_rows, n_outputs),
            Self::Hinge => HingeLoss.transform_predictions(predictions, n_rows, n_outputs),
            Self::Softmax { num_classes } => SoftmaxLoss::new(*num_classes).transform_predictions(predictions, n_rows, n_outputs),
            Self::Quantile { alpha } => PinballLoss::new(*alpha).transform_predictions(predictions, n_rows, n_outputs),
            Self::MultiQuantile { alphas } => PinballLoss::with_quantiles(alphas.clone()).transform_predictions(predictions, n_rows, n_outputs),
            Self::PseudoHuber { delta } => PseudoHuberLoss::new(*delta).transform_predictions(predictions, n_rows, n_outputs),
            Self::Poisson => PoissonLoss.transform_predictions(predictions, n_rows, n_outputs),
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
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 3];

        obj.compute_gradients(3, 1, &preds, &targets, &[], &mut grad_hess);

        // grad = pred - target
        assert!((grad_hess[0].grad - 0.5).abs() < 1e-6);
        assert!((grad_hess[1].grad - -0.5).abs() < 1e-6);
        assert!((grad_hess[2].grad - 0.5).abs() < 1e-6);

        // hess = 1.0
        assert!((grad_hess[0].hess - 1.0).abs() < 1e-6);
        assert!((grad_hess[1].hess - 1.0).abs() < 1e-6);
        assert!((grad_hess[2].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_squared_loss() {
        let obj = SquaredLoss;
        let preds = [1.0f32, 2.0];
        let targets = [0.5f32, 2.5];
        let weights = [2.0f32, 0.5];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 2];

        obj.compute_gradients(2, 1, &preds, &targets, &weights, &mut grad_hess);

        // grad = weight * (pred - target)
        assert!((grad_hess[0].grad - 1.0).abs() < 1e-6); // 2.0 * 0.5
        assert!((grad_hess[1].grad - -0.25).abs() < 1e-6); // 0.5 * -0.5

        // hess = weight
        assert!((grad_hess[0].hess - 2.0).abs() < 1e-6);
        assert!((grad_hess[1].hess - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pinball_loss_median() {
        let obj = PinballLoss::new(0.5);
        let preds = [1.0f32, 2.0, 3.0];
        let targets = [0.5f32, 2.5, 2.5];
        let mut grad_hess = [GradsTuple { grad: 0.0, hess: 0.0 }; 3];

        obj.compute_gradients(3, 1, &preds, &targets, &[], &mut grad_hess);

        // For alpha=0.5: grad = 0.5 if pred > target, -0.5 if pred < target
        assert!((grad_hess[0].grad - 0.5).abs() < 1e-6); // pred > target
        assert!((grad_hess[1].grad - -0.5).abs() < 1e-6); // pred < target
        assert!((grad_hess[2].grad - 0.5).abs() < 1e-6); // pred > target
    }
}
