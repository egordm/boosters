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
//! Configure objectives via the [`Objective`] enum:
//! - Regression: [`Objective::SquaredLoss`], [`Objective::AbsoluteLoss`],
//!   [`Objective::PinballLoss`], [`Objective::PseudoHuberLoss`], [`Objective::PoissonLoss`]
//! - Classification: [`Objective::LogisticLoss`], [`Objective::HingeLoss`],
//!   [`Objective::SoftmaxLoss`]

mod classification;
mod regression;

use classification::{
    compute_hinge_base_score, compute_hinge_gradients_into, compute_logistic_base_score,
    compute_logistic_gradients_into, compute_softmax_base_score, compute_softmax_gradients_into,
};
use regression::{
    compute_absolute_base_score, compute_absolute_gradients_into, compute_pinball_base_score,
    compute_pinball_gradients_into, compute_poisson_base_score, compute_poisson_gradients_into,
    compute_pseudo_huber_base_score, compute_pseudo_huber_gradients_into,
    compute_squared_base_score, compute_squared_gradients_into,
};

use crate::data::{TargetsView, WeightsView};
use crate::training::GradsTuple;
use ndarray::{ArrayView2, ArrayViewMut2};
use std::sync::Arc;

// =============================================================================
// Custom Objective (boxed closures)
// =============================================================================

/// Type alias for custom gradient computation function.
///
/// Arguments: (predictions, targets, weights, grad_hess_out)
pub type GradientFn = Arc<
    dyn Fn(ArrayView2<f32>, TargetsView<'_>, WeightsView<'_>, ArrayViewMut2<GradsTuple>)
        + Send
        + Sync
        + 'static,
>;

/// Type alias for custom base score computation function.
///
/// Arguments: (targets, weights) -> base_scores
pub type BaseScoreFn =
    Arc<dyn Fn(TargetsView<'_>, WeightsView<'_>) -> Vec<f32> + Send + Sync + 'static>;

/// A user-defined objective function using boxed closures.
///
/// This allows users to define custom objectives without implementing
/// a trait. All behavior is specified via closures.
///
/// # Example
///
/// ```ignore
/// use boosters::training::{CustomObjective, Objective};
/// use boosters::model::OutputTransform;
///
/// let custom = CustomObjective {
///     name: "my_loss".into(),
///     compute_gradients_into: std::sync::Arc::new(|preds, targets, weights, mut grad_hess| {
///         // Custom gradient computation
///     }),
///     compute_base_score: std::sync::Arc::new(|targets, weights| vec![0.0]),
///     output_transform: OutputTransform::Identity,
///     n_outputs: 1,
/// };
///
/// let objective = Objective::Custom(custom);
/// ```
#[derive(Clone)]
pub struct CustomObjective {
    /// Name of the objective (for logging).
    pub name: String,

    /// Function to compute gradients and hessians.
    pub compute_gradients_into: GradientFn,

    /// Function to compute initial base score from targets.
    pub compute_base_score: BaseScoreFn,

    /// Output transform for inference.
    pub output_transform: crate::model::OutputTransform,

    /// Number of outputs per sample.
    pub n_outputs: usize,
}

impl std::fmt::Debug for CustomObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomObjective")
            .field("name", &self.name)
            .field("output_transform", &self.output_transform)
            .field("n_outputs", &self.n_outputs)
            .finish_non_exhaustive()
    }
}

impl CustomObjective {
    pub fn new(
        name: impl Into<String>,
        compute_gradients_into: impl Fn(
            ArrayView2<f32>,
            TargetsView<'_>,
            WeightsView<'_>,
            ArrayViewMut2<GradsTuple>,
        ) + Send
        + Sync
        + 'static,
        compute_base_score: impl Fn(TargetsView<'_>, WeightsView<'_>) -> Vec<f32>
        + Send
        + Sync
        + 'static,
        output_transform: crate::model::OutputTransform,
        n_outputs: usize,
    ) -> Self {
        Self {
            name: name.into(),
            compute_gradients_into: Arc::new(compute_gradients_into),
            compute_base_score: Arc::new(compute_base_score),
            output_transform,
            n_outputs,
        }
    }

    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    pub fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        (self.compute_gradients_into)(predictions, targets, weights, grad_hess);
    }

    pub fn compute_base_score(
        &self,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> Vec<f32> {
        (self.compute_base_score)(targets, weights)
    }

    pub fn output_transform(&self) -> crate::model::OutputTransform {
        self.output_transform
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

// =============================================================================
// Objective Enum (Convenience wrapper)
// =============================================================================

/// Objective function enum for easy configuration.
///
/// This enum wraps all available objective types and provides a unified
/// interface for trainers.
///
/// # Example
///
/// ```ignore
/// use boosters::training::{GBDTTrainer, Objective};
///
/// let trainer = GBDTTrainer::builder()
///     .objective(Objective::LogisticLoss)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Default, Clone)]
pub enum Objective {
    /// Squared error loss (L2) for regression.
    #[default]
    SquaredLoss,
    /// Absolute error loss (L1) for robust regression.
    AbsoluteLoss,
    /// Logistic loss for binary classification.
    LogisticLoss,
    /// Hinge loss for SVM-style classification.
    HingeLoss,
    /// Softmax loss for multiclass classification.
    SoftmaxLoss { n_classes: usize },
    /// Pinball loss for quantile regression (single or multi-quantile).
    PinballLoss {
        /// Quantile levels for each output, each in (0, 1).
        alphas: Vec<f32>,
    },
    /// Pseudo-Huber loss for robust regression.
    PseudoHuberLoss { delta: f32 },
    /// Poisson loss for count data regression.
    PoissonLoss,
    /// Custom objective (user-provided implementation).
    Custom(CustomObjective),
}

impl Objective {
    pub fn n_outputs(&self) -> usize {
        match self {
            Self::SquaredLoss => 1,
            Self::AbsoluteLoss => 1,
            Self::LogisticLoss => 1,
            Self::HingeLoss => 1,
            Self::SoftmaxLoss { n_classes } => *n_classes,
            Self::PinballLoss { alphas } => alphas.len(),
            Self::PseudoHuberLoss { .. } => 1,
            Self::PoissonLoss => 1,
            Self::Custom(inner) => inner.n_outputs(),
        }
    }

    pub fn compute_gradients_into(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
        grad_hess: ArrayViewMut2<GradsTuple>,
    ) {
        match self {
            Self::SquaredLoss => {
                compute_squared_gradients_into(predictions, targets, weights, grad_hess)
            }
            Self::AbsoluteLoss => {
                compute_absolute_gradients_into(predictions, targets, weights, grad_hess)
            }
            Self::LogisticLoss => {
                compute_logistic_gradients_into(predictions, targets, weights, grad_hess)
            }
            Self::HingeLoss => {
                compute_hinge_gradients_into(predictions, targets, weights, grad_hess)
            }
            Self::SoftmaxLoss { n_classes } => {
                compute_softmax_gradients_into(predictions, targets, weights, *n_classes, grad_hess)
            }
            Self::PinballLoss { alphas } => {
                compute_pinball_gradients_into(predictions, targets, weights, alphas, grad_hess)
            }
            Self::PseudoHuberLoss { delta } => compute_pseudo_huber_gradients_into(
                predictions,
                targets,
                weights,
                *delta,
                grad_hess,
            ),
            Self::PoissonLoss => {
                compute_poisson_gradients_into(predictions, targets, weights, grad_hess)
            }
            Self::Custom(inner) => {
                inner.compute_gradients_into(predictions, targets, weights, grad_hess)
            }
        }
    }

    pub fn compute_base_score(
        &self,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> Vec<f32> {
        match self {
            Self::SquaredLoss => compute_squared_base_score(targets, weights),
            Self::AbsoluteLoss => compute_absolute_base_score(targets, weights),
            Self::LogisticLoss => compute_logistic_base_score(targets, weights),
            Self::HingeLoss => compute_hinge_base_score(targets, weights),
            Self::SoftmaxLoss { n_classes } => {
                compute_softmax_base_score(targets, weights, *n_classes)
            }
            Self::PinballLoss { alphas } => compute_pinball_base_score(targets, weights, alphas),
            Self::PseudoHuberLoss { delta } => {
                compute_pseudo_huber_base_score(targets, weights, *delta)
            }
            Self::PoissonLoss => compute_poisson_base_score(targets, weights),
            Self::Custom(inner) => inner.compute_base_score(targets, weights),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::SquaredLoss => "squared",
            Self::AbsoluteLoss => "absolute",
            Self::LogisticLoss => "logistic",
            Self::HingeLoss => "hinge",
            Self::SoftmaxLoss { .. } => "softmax",
            Self::PinballLoss { .. } => "quantile",
            Self::PseudoHuberLoss { .. } => "pseudo_huber",
            Self::PoissonLoss => "poisson",
            Self::Custom(inner) => inner.name(),
        }
    }

    /// Returns the output transform for this objective.
    ///
    /// This is used by models to transform raw predictions (margins)
    /// to final predictions (probabilities, values, etc.) at inference time.
    pub fn output_transform(&self) -> crate::model::OutputTransform {
        use crate::model::OutputTransform;
        match self {
            // Regression: identity (raw values)
            Self::SquaredLoss
            | Self::AbsoluteLoss
            | Self::PinballLoss { .. }
            | Self::PseudoHuberLoss { .. }
            | Self::PoissonLoss => OutputTransform::Identity,

            // Binary classification: sigmoid
            Self::LogisticLoss => OutputTransform::Sigmoid,

            // Hinge: identity (margins)
            Self::HingeLoss => OutputTransform::Identity,

            // Multiclass: softmax
            Self::SoftmaxLoss { .. } => OutputTransform::Softmax,

            // Custom: delegate to the objective implementation
            Self::Custom(inner) => inner.output_transform(),
        }
    }

    /// Returns whether this objective requires leaf value renewing after tree growth.
    ///
    /// Some objectives (e.g., quantile regression, L1 loss) have optimal leaf values
    /// that differ from the Newton-step formula used during tree construction.
    /// For these objectives, leaf values should be recomputed after the tree structure
    /// is finalized.
    ///
    /// This follows the XGBoost/LightGBM pattern where quantile objectives use
    /// `UpdateTreeLeaf()` / `RenewTreeOutput()` to compute the α-quantile of
    /// residuals within each leaf instead of the gradient-based Newton step.
    #[inline]
    pub fn needs_leaf_renew(&self) -> bool {
        matches!(self, Self::PinballLoss { .. } | Self::AbsoluteLoss)
    }

    /// Compute renewed leaf value for a single leaf.
    ///
    /// For objectives like quantile regression and L1 loss, the optimal leaf value
    /// is the α-quantile of residuals rather than the Newton-step formula.
    ///
    /// # Arguments
    /// * `output` - Which output index this is (for multi-output objectives)
    /// * `indices` - Row indices in this leaf
    /// * `targets` - Target values
    /// * `predictions` - Current predictions
    /// * `weights` - Sample weights
    /// * `scratch` - Scratch space for quantile computation
    ///
    /// # Returns
    /// The renewed leaf value, or NAN if not applicable.
    #[inline]
    pub fn renew_single_leaf_value(
        &self,
        output: usize,
        indices: &[u32],
        targets: TargetsView<'_>,
        predictions: ndarray::ArrayView2<f32>,
        weights: WeightsView<'_>,
        scratch: &mut Vec<u32>,
    ) -> f32 {
        if indices.is_empty() {
            return f32::NAN;
        }

        let alpha = match self {
            Self::PinballLoss { alphas } => alphas.get(output).copied().unwrap_or(0.5),
            Self::AbsoluteLoss => 0.5,
            // Other objectives don't need renewing
            _ => return f32::NAN,
        };

        let targets_row = targets.output(0);
        let preds_row = predictions.row(output);

        // Zero-allocation residual computation via closure
        let residual = |idx: u32| {
            let i = idx as usize;
            targets_row[i] - preds_row[i]
        };

        // Use optimized O(n) algorithm for unweighted case, O(n log n) for weighted
        if weights.is_none() {
            crate::utils::unweighted_quantile_indexed(indices, residual, alpha, scratch)
        } else {
            let weight = |idx: u32| weights.get(idx as usize);
            crate::utils::weighted_quantile_indexed(indices, residual, weight, alpha, scratch)
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
        Array2::from_elem(
            (n_outputs, n_samples),
            GradsTuple {
                grad: 0.0,
                hess: 0.0,
            },
        )
    }

    fn make_targets(data: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((1, data.len()), data.to_vec()).unwrap()
    }

    #[test]
    fn squared_loss_gradients() {
        let obj = Objective::SquaredLoss;
        let preds = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let targets = make_targets(&[0.5, 2.5, 2.5]);
        let mut grad_hess = make_grad_hess_array(1, 3);

        obj.compute_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::None,
            grad_hess.view_mut(),
        );

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
        let obj = Objective::SquaredLoss;
        let preds = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let targets = make_targets(&[0.5, 2.5]);
        let weights = ndarray::array![2.0f32, 0.5];
        let mut grad_hess = make_grad_hess_array(1, 2);

        obj.compute_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::from_array(weights.view()),
            grad_hess.view_mut(),
        );

        // grad = weight * (pred - target)
        assert!((grad_hess[[0, 0]].grad - 1.0).abs() < 1e-6); // 2.0 * 0.5
        assert!((grad_hess[[0, 1]].grad - -0.25).abs() < 1e-6); // 0.5 * -0.5

        // hess = weight
        assert!((grad_hess[[0, 0]].hess - 2.0).abs() < 1e-6);
        assert!((grad_hess[[0, 1]].hess - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pinball_loss_median() {
        let obj = Objective::PinballLoss { alphas: vec![0.5] };
        let preds = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let targets = make_targets(&[0.5, 2.5, 2.5]);
        let mut grad_hess = make_grad_hess_array(1, 3);

        obj.compute_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::None,
            grad_hess.view_mut(),
        );

        // For alpha=0.5: grad = 0.5 if pred > target, -0.5 if pred < target
        assert!((grad_hess[[0, 0]].grad - 0.5).abs() < 1e-6); // pred > target
        assert!((grad_hess[[0, 1]].grad - -0.5).abs() < 1e-6); // pred < target
        assert!((grad_hess[[0, 2]].grad - 0.5).abs() < 1e-6); // pred > target
    }

    #[test]
    fn output_transform_mapping() {
        use crate::model::OutputTransform;

        // Regression objectives -> Identity
        assert_eq!(
            Objective::SquaredLoss.output_transform(),
            OutputTransform::Identity
        );
        assert_eq!(
            Objective::AbsoluteLoss.output_transform(),
            OutputTransform::Identity
        );
        assert_eq!(
            Objective::PinballLoss { alphas: vec![0.5] }.output_transform(),
            OutputTransform::Identity
        );
        assert_eq!(
            Objective::PseudoHuberLoss { delta: 1.0 }.output_transform(),
            OutputTransform::Identity
        );
        assert_eq!(
            Objective::PoissonLoss.output_transform(),
            OutputTransform::Identity
        );

        // Binary classification
        assert_eq!(
            Objective::LogisticLoss.output_transform(),
            OutputTransform::Sigmoid
        );
        assert_eq!(
            Objective::HingeLoss.output_transform(),
            OutputTransform::Identity
        );

        // Multiclass
        assert_eq!(
            Objective::SoftmaxLoss { n_classes: 3 }.output_transform(),
            OutputTransform::Softmax
        );
    }

    #[test]
    fn custom_objective_works() {
        use crate::model::OutputTransform;

        // Create a simple custom objective that mimics squared loss
        let custom = CustomObjective {
            name: "test_squared".into(),
            compute_gradients_into: std::sync::Arc::new(
                |preds, targets, _weights, mut grad_hess| {
                    let targets_view = targets.view();
                    for col in 0..preds.ncols() {
                        let pred = preds[[0, col]];
                        let target = targets_view[[0, col]];
                        grad_hess[[0, col]] = GradsTuple {
                            grad: pred - target,
                            hess: 1.0,
                        };
                    }
                },
            ),
            compute_base_score: std::sync::Arc::new(|_targets, _weights| vec![0.0]),
            output_transform: OutputTransform::Identity,
            n_outputs: 1,
        };

        // Wrap in Objective enum
        let obj = Objective::Custom(custom);

        // Test that it works
        let preds = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let targets = make_targets(&[0.5, 2.5, 2.5]);
        let mut grad_hess = make_grad_hess_array(1, 3);

        obj.compute_gradients_into(
            preds.view(),
            TargetsView::new(targets.view()),
            WeightsView::None,
            grad_hess.view_mut(),
        );

        // Should match squared loss behavior
        assert!((grad_hess[[0, 0]].grad - 0.5).abs() < 1e-6);
        assert!((grad_hess[[0, 1]].grad - -0.5).abs() < 1e-6);
        assert!((grad_hess[[0, 2]].grad - 0.5).abs() < 1e-6);
    }
}
