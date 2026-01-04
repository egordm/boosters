//! Evaluation metrics for model quality.
//!
//! Metrics are separate from loss functions — a model might be trained with
//! one loss but evaluated with different metrics.
//!
//! # Multi-Output Support
//!
//! Metrics support multi-output models (multiclass, multi-quantile) via the
//! `n_outputs` parameter. The predictions buffer has shape `[n_outputs, n_rows]`
//! in **column-major** order (output-first), matching the training prediction layout.
//!
//! # Weighted Evaluation
//!
//! All metrics support optional sample weights via the `weights` parameter.
//! When `None`, unweighted computation is used. When `Some(&weights)`, weighted
//! formulas are applied (e.g., `sum(w * error) / sum(w)`).
//!
//! # Validation Set
//!
//! Pass a validation `Dataset` to training methods for early stopping and monitoring.
//! The validation set is implicitly named "valid" in training logs.
//!
//! # Available Metrics
//!
//! ## Regression
//! - `Metric::Rmse`: Root Mean Squared Error
//! - `Metric::Mae`: Mean Absolute Error
//! - `Metric::Mape`: Mean Absolute Percentage Error
//! - `Metric::Quantile { alphas }`: Pinball loss for quantile regression
//!
//! ## Classification
//! - `Metric::LogLoss`: Binary cross-entropy
//! - `Metric::Accuracy { threshold }`: Binary classification accuracy
//! - `Metric::Auc`: Area Under ROC Curve
//! - `Metric::MulticlassLogLoss`: Multiclass cross-entropy
//! - `Metric::MulticlassAccuracy`: Multiclass accuracy

mod classification;
mod regression;
use ndarray::ArrayView2;
use std::sync::Arc;

use super::objectives::Objective;

use crate::data::{TargetsView, WeightsView};
use crate::inference::PredictionKind;

// =============================================================================
// Custom Metric
// =============================================================================

/// Type alias for the custom metric compute function.
///
/// Takes predictions (shape `[n_outputs, n_samples]`), targets, and weights.
/// Returns a scalar metric value.
pub type CustomMetricFn =
    Arc<dyn Fn(ArrayView2<f32>, TargetsView<'_>, WeightsView<'_>) -> f64 + Send + Sync + 'static>;

/// A user-provided custom metric.
///
/// Allows defining metrics via closures rather than implementing a trait.
/// This is the preferred way to create custom metrics in the vNext API.
///
/// # Example
///
/// ```ignore
/// use boosters::training::{CustomMetric, Metric};
/// use boosters::inference::PredictionKind;
///
/// // Custom MAE metric
/// let custom_mae = CustomMetric::new(
///     "custom_mae",
///     |preds, targets, weights| {
///         let targets = targets.output(0);
///         let preds = preds.row(0);
///         let n = preds.len() as f64;
///         preds.iter()
///             .zip(targets.iter())
///             .map(|(p, t)| (*p as f64 - *t as f64).abs())
///             .sum::<f64>() / n
///     },
///     PredictionKind::Value,
///     false, // lower is better
/// );
///
/// let metric = Metric::Custom(custom_mae);
/// ```
#[derive(Clone)]
pub struct CustomMetric {
    /// Name of the metric (for logging).
    pub name: String,
    /// The compute function.
    compute_fn: CustomMetricFn,
    /// What prediction space this metric expects.
    pub expected_prediction_kind: PredictionKind,
    /// Whether higher values indicate better performance.
    pub higher_is_better: bool,
}

impl CustomMetric {
    /// Create a new custom metric.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for logging (e.g., "custom_mae").
    /// * `compute_fn` - Function that computes the metric value
    /// * `expected_prediction_kind` - What prediction space this metric expects
    /// * `higher_is_better` - Whether higher values indicate better performance
    pub fn new(
        name: impl Into<String>,
        compute_fn: impl Fn(ArrayView2<f32>, TargetsView<'_>, WeightsView<'_>) -> f64
        + Send
        + Sync
        + 'static,
        expected_prediction_kind: PredictionKind,
        higher_is_better: bool,
    ) -> Self {
        Self {
            name: name.into(),
            compute_fn: Arc::new(compute_fn),
            expected_prediction_kind,
            higher_is_better,
        }
    }

    /// Compute the metric value.
    pub fn compute(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> f64 {
        (self.compute_fn)(predictions, targets, weights)
    }

    /// Get the metric name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Debug for CustomMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomMetric")
            .field("name", &self.name)
            .field("expected_prediction_kind", &self.expected_prediction_kind)
            .field("higher_is_better", &self.higher_is_better)
            .finish()
    }
}

// =============================================================================
// Metric Enum (Convenience wrapper)
// =============================================================================

/// A dynamically-dispatched metric function.
///
/// This enum provides runtime metric selection without generics.
/// Most variants are simple unit variants; a few carry configuration
/// (e.g. `Accuracy { threshold }`, `Quantile { alphas }`).
///
/// # Example
///
/// ```ignore
/// use boosters::training::Metric;
///
/// // Use enum variants
/// let rmse = Metric::Rmse;
/// let logloss = Metric::LogLoss;
/// let accuracy = Metric::Accuracy { threshold: 0.7 };
/// let none = Metric::None;
/// ```
#[derive(Debug, Default, Clone)]
pub enum Metric {
    /// No metric - skips evaluation entirely.
    ///
    /// When used, the trainer skips metric computation and prediction
    /// transformation, avoiding wasted compute.
    #[default]
    None,
    /// Root Mean Squared Error (regression).
    Rmse,
    /// Mean Absolute Error (regression).
    Mae,
    /// Mean Absolute Percentage Error (regression).
    Mape,
    /// Log Loss / Binary Cross-Entropy (binary classification).
    LogLoss,
    /// Accuracy (binary classification) with configurable threshold.
    Accuracy { threshold: f32 },
    /// Margin-based accuracy (threshold=0.0, for hinge loss).
    MarginAccuracy,
    /// Area Under ROC Curve (binary classification).
    Auc,
    /// Multiclass Log Loss (multiclass classification).
    MulticlassLogLoss,
    /// Multiclass Accuracy (multiclass classification).
    MulticlassAccuracy,
    /// Quantile metric (pinball loss).
    Quantile { alphas: Vec<f32> },
    /// Huber metric for robust regression.
    Huber { delta: f64 },
    /// Poisson Deviance for count data.
    PoissonDeviance,
    /// Custom user-provided metric.
    Custom(CustomMetric),
}

/// Returns the default evaluation metric for a given training objective.
///
/// This is the single, centralized mapping used when the user does not
/// explicitly configure a metric.
pub fn default_metric_for_objective(objective: &Objective) -> Metric {
    match objective {
        // Regression
        Objective::SquaredLoss
        | Objective::AbsoluteLoss
        | Objective::PseudoHuberLoss { .. }
        | Objective::PoissonLoss => Metric::Rmse,
        Objective::PinballLoss { alphas } => Metric::Quantile {
            alphas: alphas.clone(),
        },

        // Classification
        Objective::LogisticLoss => Metric::LogLoss,
        Objective::HingeLoss => Metric::MarginAccuracy,
        Objective::SoftmaxLoss { .. } => Metric::MulticlassLogLoss,

        // Custom
        Objective::Custom(_) => Metric::None,
    }
}

impl Metric {
    pub fn compute(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> f64 {
        match self {
            Self::None => f64::NAN,
            Self::Rmse => regression::compute_rmse(predictions, targets, weights),
            Self::Mae => regression::compute_mae(predictions, targets, weights),
            Self::Mape => regression::compute_mape(predictions, targets, weights),
            Self::LogLoss => classification::compute_logloss(predictions, targets, weights),
            Self::Accuracy { threshold } => {
                classification::compute_accuracy(predictions, targets, weights, *threshold)
            }
            Self::MarginAccuracy => {
                classification::compute_margin_accuracy(predictions, targets, weights, 0.0)
            }
            Self::Auc => classification::compute_auc(predictions, targets, weights),
            Self::MulticlassLogLoss => {
                classification::compute_multiclass_logloss(predictions, targets, weights)
            }
            Self::MulticlassAccuracy => {
                classification::compute_multiclass_accuracy(predictions, targets, weights)
            }
            Self::Quantile { alphas } => {
                regression::compute_quantile_pinball(predictions, targets, weights, alphas)
            }
            Self::Huber { delta } => {
                regression::compute_huber(predictions, targets, weights, *delta)
            }
            Self::PoissonDeviance => {
                regression::compute_poisson_deviance(predictions, targets, weights)
            }
            Self::Custom(inner) => inner.compute(predictions, targets, weights),
        }
    }

    pub fn expected_prediction_kind(&self) -> PredictionKind {
        match self {
            Self::None => PredictionKind::Margin,
            Self::Rmse | Self::Mae | Self::Mape | Self::Quantile { .. } | Self::Huber { .. } => {
                PredictionKind::Value
            }
            Self::LogLoss | Self::Accuracy { .. } => PredictionKind::Probability,
            Self::MarginAccuracy => PredictionKind::Margin,
            Self::Auc => PredictionKind::Probability,
            Self::MulticlassLogLoss | Self::MulticlassAccuracy => PredictionKind::Probability,
            Self::PoissonDeviance => PredictionKind::Value,
            Self::Custom(inner) => inner.expected_prediction_kind,
        }
    }

    pub fn higher_is_better(&self) -> bool {
        match self {
            Self::None => false,
            Self::Rmse
            | Self::Mae
            | Self::Mape
            | Self::LogLoss
            | Self::MulticlassLogLoss
            | Self::Quantile { .. }
            | Self::Huber { .. }
            | Self::PoissonDeviance => false,
            Self::Accuracy { .. } | Self::MarginAccuracy | Self::Auc | Self::MulticlassAccuracy => {
                true
            }
            Self::Custom(inner) => inner.higher_is_better,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::None => "<none>",
            Self::Rmse => "rmse",
            Self::Mae => "mae",
            Self::Mape => "mape",
            Self::LogLoss => "logloss",
            Self::Accuracy { .. } => "accuracy",
            Self::MarginAccuracy => "margin_accuracy",
            Self::Auc => "auc",
            Self::MulticlassLogLoss => "mlogloss",
            Self::MulticlassAccuracy => "merror",
            Self::Quantile { .. } => "quantile",
            Self::Huber { .. } => "huber",
            Self::PoissonDeviance => "poisson",
            Self::Custom(inner) => inner.name.as_ref(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::WeightsView;
    use ndarray::Array2;

    fn make_preds(n_outputs: usize, n_samples: usize, data: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((n_outputs, n_samples), data.to_vec()).unwrap()
    }

    fn make_targets(data: &[f32]) -> Array2<f32> {
        Array2::from_shape_vec((1, data.len()), data.to_vec()).unwrap()
    }

    #[test]
    fn custom_metric_works() {
        // Create a simple custom MAE metric
        let custom_mae = CustomMetric::new(
            "custom_mae",
            |preds, targets, _weights| {
                let targets = targets.output(0);
                let preds = preds.row(0);
                let n = preds.len() as f64;
                preds
                    .iter()
                    .zip(targets.iter())
                    .map(|(p, t)| (*p as f64 - *t as f64).abs())
                    .sum::<f64>()
                    / n
            },
            PredictionKind::Value,
            false, // lower is better
        );

        let metric = Metric::Custom(custom_mae);

        // Test compute
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let labels = make_targets(&[0.0, 0.0]);
        let value = metric.compute(
            preds.view(),
            TargetsView::new(labels.view()),
            WeightsView::None,
        );
        approx::assert_abs_diff_eq!(value, 1.5, epsilon = 1e-10);

        // Test properties
        assert_eq!(metric.name(), "custom_mae");
        assert!(!metric.higher_is_better());
        assert_eq!(metric.expected_prediction_kind(), PredictionKind::Value);
    }

    #[test]
    fn metric_none_disabled() {
        let metric = Metric::None;
        assert_eq!(metric.name(), "<none>");
    }
}
