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

use crate::data::{TargetsView, WeightsView};
use crate::inference::PredictionKind;
use crate::training::GradsTuple;
use ndarray::{ArrayView2, ArrayViewMut2};

// Re-export TaskKind from model module for unified usage
pub use crate::model::TaskKind;

// =============================================================================
// RFC 0028: Target Semantics
// =============================================================================

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
/// - SoftmaxLoss: `n_outputs` = n_classes, `n_targets` = 1 (class indices)
///
/// # Weighted Training
///
/// Pass `None` for `weights` for unweighted training.
/// When `Some`, the array view must have length `n_rows`.
pub trait ObjectiveFn: Send + Sync {
    /// Number of outputs (predictions per sample).
    ///
    /// For most objectives this is 1 (single-output).
    /// For multiclass (SoftmaxLoss) this is n_classes.
    /// For multi-quantile this is the number of quantiles.
    fn n_outputs(&self) -> usize;

    /// Compute gradients and hessians for the given predictions.
    ///
    /// # Arguments
    ///
    /// * `n_rows` - Number of samples
    /// * `n_outputs` - Number of outputs (predictions per sample)
    /// * `predictions` - Model predictions, column-major `[n_outputs * n_rows]`
    /// * `targets` - Ground truth labels, column-major `[n_targets * n_rows]`
    /// * `weights` - Sample weights (use `WeightsView::uniform(n)` for unweighted)
    /// * `grad_hess` - Interleaved (grad, hess) pairs, column-major `[n_outputs * n_rows]`
    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        grad_hess: ArrayViewMut2<GradsTuple>,
    );

    /// Compute the initial base score (bias) from targets.
    ///
    /// This is the optimal constant prediction before any trees are added.
    /// Returns a vector of length `n_outputs()`.
    ///
    /// # Arguments
    ///
    /// * `targets` - Ground truth labels
    /// * `weights` - Sample weights (use `WeightsView::uniform(n)` for unweighted)
    fn compute_base_score(
        &self,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> Vec<f32>;

    // =========================================================================
    // RFC 0028: Semantics + Prediction Transforms
    // =========================================================================

    /// High-level task kind implied by this objective.
    fn task_kind(&self) -> TaskKind;

    /// Target encoding/schema expected by this objective.
    fn target_schema(&self) -> TargetSchema;

    /// Transform raw predictions in-place (column-major layout).
    ///
    /// Converts margins/logits to semantic predictions (probabilities, values, etc.).
    /// Returns the semantic kind of the resulting values.
    ///
    /// Layout: `predictions[output * n_rows + row]`
    ///
    /// Most regression objectives are no-ops. Classification objectives apply
    /// sigmoid (binary) or softmax (multiclass).
    fn transform_predictions_inplace(&self, predictions: ArrayViewMut2<f32>) -> PredictionKind;

    /// Name of the objective (for logging).
    fn name(&self) -> &'static str;
}

// =============================================================================
// Objective Enum (Convenience wrapper)
// =============================================================================

/// Objective function enum for easy configuration.
///
/// This enum wraps all available objective types and provides a unified
/// interface for trainers. It implements `ObjectiveFn` by delegating to the
/// underlying concrete type.
///
/// Each variant stores a pre-constructed instance of the underlying loss type,
/// avoiding allocation on each method call (newtype pattern).
///
/// # Example
///
/// ```ignore
/// use boosters::training::{GBDTTrainer, Objective};
///
/// let trainer = GBDTTrainer::builder()
///     .objective(Objective::logistic())
///     .build()
///     .unwrap();
/// ```
#[derive(Clone)]
pub enum Objective {
    /// Squared error loss (L2) for regression.
    SquaredLoss(SquaredLoss),
    /// Absolute error loss (L1) for robust regression.
    AbsoluteLoss(AbsoluteLoss),
    /// Logistic loss for binary classification.
    LogisticLoss(LogisticLoss),
    /// Hinge loss for SVM-style classification.
    HingeLoss(HingeLoss),
    /// Softmax loss for multiclass classification.
    SoftmaxLoss(SoftmaxLoss),
    /// Pinball loss for quantile regression (single or multi-quantile).
    PinballLoss(PinballLoss),
    /// Pseudo-Huber loss for robust regression.
    PseudoHuberLoss(PseudoHuberLoss),
    /// Poisson loss for count data regression.
    PoissonLoss(PoissonLoss),
    /// Custom objective (user-provided implementation).
    Custom(std::sync::Arc<dyn ObjectiveFn>),
}

impl std::fmt::Debug for Objective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SquaredLoss(inner) => f.debug_tuple("SquaredLoss").field(inner).finish(),
            Self::AbsoluteLoss(inner) => f.debug_tuple("AbsoluteLoss").field(inner).finish(),
            Self::LogisticLoss(inner) => f.debug_tuple("LogisticLoss").field(inner).finish(),
            Self::HingeLoss(inner) => f.debug_tuple("HingeLoss").field(inner).finish(),
            Self::SoftmaxLoss(inner) => f.debug_tuple("SoftmaxLoss").field(inner).finish(),
            Self::PinballLoss(inner) => f.debug_tuple("PinballLoss").field(inner).finish(),
            Self::PseudoHuberLoss(inner) => f.debug_tuple("PseudoHuberLoss").field(inner).finish(),
            Self::PoissonLoss(inner) => f.debug_tuple("PoissonLoss").field(inner).finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<dyn ObjectiveFn>").finish(),
        }
    }
}

/// Convenience constructors for common objectives.
impl Objective {
    /// Squared error (L2) loss for regression.
    pub fn squared() -> Self {
        Self::SquaredLoss(SquaredLoss)
    }
    
    /// Absolute error (L1) loss for robust regression.
    pub fn absolute() -> Self {
        Self::AbsoluteLoss(AbsoluteLoss)
    }
    
    /// Binary logistic loss for classification.
    pub fn logistic() -> Self {
        Self::LogisticLoss(LogisticLoss)
    }
    
    /// Hinge loss for SVM-style classification.
    pub fn hinge() -> Self {
        Self::HingeLoss(HingeLoss)
    }
    
    /// Softmax loss for multiclass classification.
    pub fn softmax(n_classes: usize) -> Self {
        Self::SoftmaxLoss(SoftmaxLoss::new(n_classes))
    }
    
    /// Pinball loss for single quantile regression.
    pub fn quantile(alpha: f32) -> Self {
        Self::PinballLoss(PinballLoss::new(alpha))
    }
    
    /// Pinball loss for multiple quantile regression.
    pub fn multi_quantile(alphas: Vec<f32>) -> Self {
        Self::PinballLoss(PinballLoss::with_quantiles(alphas))
    }
    
    /// Pseudo-Huber loss for robust regression.
    pub fn pseudo_huber(delta: f32) -> Self {
        Self::PseudoHuberLoss(PseudoHuberLoss::new(delta))
    }
    
    /// Poisson loss for count data regression.
    pub fn poisson() -> Self {
        Self::PoissonLoss(PoissonLoss)
    }

    /// Custom objective with user-provided implementation.
    pub fn custom<O: ObjectiveFn + 'static>(objective: O) -> Self {
        Self::Custom(std::sync::Arc::new(objective))
    }
}

impl Default for Objective {
    fn default() -> Self {
        Self::SquaredLoss(SquaredLoss)
    }
}

impl ObjectiveFn for Objective {
    fn n_outputs(&self) -> usize {
        match self {
            Self::SquaredLoss(inner) => inner.n_outputs(),
            Self::AbsoluteLoss(inner) => inner.n_outputs(),
            Self::LogisticLoss(inner) => inner.n_outputs(),
            Self::HingeLoss(inner) => inner.n_outputs(),
            Self::SoftmaxLoss(inner) => inner.n_outputs(),
            Self::PinballLoss(inner) => inner.n_outputs(),
            Self::PseudoHuberLoss(inner) => inner.n_outputs(),
            Self::PoissonLoss(inner) => inner.n_outputs(),
            Self::Custom(inner) => inner.n_outputs(),
        }
    }

    fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        match self {
            Self::SquaredLoss(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
            Self::AbsoluteLoss(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
            Self::LogisticLoss(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
            Self::HingeLoss(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
            Self::SoftmaxLoss(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
            Self::PinballLoss(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
            Self::PseudoHuberLoss(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
            Self::PoissonLoss(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
            Self::Custom(inner) => inner.compute_gradients_into(predictions, targets, weights, grad_hess),
        }
    }

    fn compute_base_score(
        &self,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> Vec<f32> {
        match self {
            Self::SquaredLoss(inner) => inner.compute_base_score(targets, weights),
            Self::AbsoluteLoss(inner) => inner.compute_base_score(targets, weights),
            Self::LogisticLoss(inner) => inner.compute_base_score(targets, weights),
            Self::HingeLoss(inner) => inner.compute_base_score(targets, weights),
            Self::SoftmaxLoss(inner) => inner.compute_base_score(targets, weights),
            Self::PinballLoss(inner) => inner.compute_base_score(targets, weights),
            Self::PseudoHuberLoss(inner) => inner.compute_base_score(targets, weights),
            Self::PoissonLoss(inner) => inner.compute_base_score(targets, weights),
            Self::Custom(inner) => inner.compute_base_score(targets, weights),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::SquaredLoss(inner) => inner.name(),
            Self::AbsoluteLoss(inner) => inner.name(),
            Self::LogisticLoss(inner) => inner.name(),
            Self::HingeLoss(inner) => inner.name(),
            Self::SoftmaxLoss(inner) => inner.name(),
            Self::PinballLoss(inner) => inner.name(),
            Self::PseudoHuberLoss(inner) => inner.name(),
            Self::PoissonLoss(inner) => inner.name(),
            Self::Custom(inner) => inner.name(),
        }
    }

    fn task_kind(&self) -> TaskKind {
        match self {
            Self::SquaredLoss(inner) => inner.task_kind(),
            Self::AbsoluteLoss(inner) => inner.task_kind(),
            Self::LogisticLoss(inner) => inner.task_kind(),
            Self::HingeLoss(inner) => inner.task_kind(),
            Self::SoftmaxLoss(inner) => inner.task_kind(),
            Self::PinballLoss(inner) => inner.task_kind(),
            Self::PseudoHuberLoss(inner) => inner.task_kind(),
            Self::PoissonLoss(inner) => inner.task_kind(),
            Self::Custom(inner) => inner.task_kind(),
        }
    }

    fn target_schema(&self) -> TargetSchema {
        match self {
            Self::SquaredLoss(inner) => inner.target_schema(),
            Self::AbsoluteLoss(inner) => inner.target_schema(),
            Self::LogisticLoss(inner) => inner.target_schema(),
            Self::HingeLoss(inner) => inner.target_schema(),
            Self::SoftmaxLoss(inner) => inner.target_schema(),
            Self::PinballLoss(inner) => inner.target_schema(),
            Self::PseudoHuberLoss(inner) => inner.target_schema(),
            Self::PoissonLoss(inner) => inner.target_schema(),
            Self::Custom(inner) => inner.target_schema(),
        }
    }

    fn transform_predictions_inplace(&self, predictions: ArrayViewMut2<f32>) -> PredictionKind {
        match self {
            Self::SquaredLoss(inner) => inner.transform_predictions_inplace(predictions),
            Self::AbsoluteLoss(inner) => inner.transform_predictions_inplace(predictions),
            Self::LogisticLoss(inner) => inner.transform_predictions_inplace(predictions),
            Self::HingeLoss(inner) => inner.transform_predictions_inplace(predictions),
            Self::SoftmaxLoss(inner) => inner.transform_predictions_inplace(predictions),
            Self::PinballLoss(inner) => inner.transform_predictions_inplace(predictions),
            Self::PseudoHuberLoss(inner) => inner.transform_predictions_inplace(predictions),
            Self::PoissonLoss(inner) => inner.transform_predictions_inplace(predictions),
            Self::Custom(inner) => inner.transform_predictions_inplace(predictions),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{TargetsView, WeightsView};
    use ndarray::Array2;

    fn make_grad_hess_array(n_outputs: usize, n_samples: usize) -> Array2<GradsTuple> {
        Array2::from_elem((n_outputs, n_samples), GradsTuple { grad: 0.0, hess: 0.0 })
    }

    fn make_targets(data: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((1, data.len()), data.to_vec()).unwrap()
    }

    #[test]
    fn squared_loss_gradients() {
        let obj = SquaredLoss;
        let preds = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let targets = make_targets(&[0.5, 2.5, 2.5]);
        let mut grad_hess = make_grad_hess_array(1, 3);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::None, grad_hess.view_mut());

        // grad = pred - target
        assert!((grad_hess[[0, 0]].grad - 0.5).abs() < 1e-6);
        assert!((grad_hess[[0, 1]].grad - -0.5).abs() < 1e-6);
        assert!((grad_hess[[0, 2]].grad - 0.5).abs() < 1e-6);

        // hess = 1.0
        assert!((grad_hess[[0, 0]].hess - 1.0).abs() < 1e-6);
        assert!((grad_hess[[0, 1]].hess - 1.0).abs() < 1e-6);
        assert!((grad_hess[[0, 2]].hess - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_squared_loss() {
        let obj = SquaredLoss;
        let preds = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let targets = make_targets(&[0.5, 2.5]);
        let weights = ndarray::array![2.0f32, 0.5];
        let mut grad_hess = make_grad_hess_array(1, 2);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::from_array(weights.view()), grad_hess.view_mut());

        // grad = weight * (pred - target)
        assert!((grad_hess[[0, 0]].grad - 1.0).abs() < 1e-6); // 2.0 * 0.5
        assert!((grad_hess[[0, 1]].grad - -0.25).abs() < 1e-6); // 0.5 * -0.5

        // hess = weight
        assert!((grad_hess[[0, 0]].hess - 2.0).abs() < 1e-6);
        assert!((grad_hess[[0, 1]].hess - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pinball_loss_median() {
        let obj = PinballLoss::new(0.5);
        let preds = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let targets = make_targets(&[0.5, 2.5, 2.5]);
        let mut grad_hess = make_grad_hess_array(1, 3);

        obj.compute_gradients_into(preds.view(), TargetsView::new(targets.view()), WeightsView::None, grad_hess.view_mut());

        // For alpha=0.5: grad = 0.5 if pred > target, -0.5 if pred < target
        assert!((grad_hess[[0, 0]].grad - 0.5).abs() < 1e-6); // pred > target
        assert!((grad_hess[[0, 1]].grad - -0.5).abs() < 1e-6); // pred < target
        assert!((grad_hess[[0, 2]].grad - 0.5).abs() < 1e-6); // pred > target
    }
}
