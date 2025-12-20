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
//! # Evaluation Sets
//!
//! Use [`EvalSet`] to define named datasets for evaluation during training:
//!
//! ```ignore
//! let eval_sets = vec![
//!     EvalSet::new("train", &train_data, &train_labels),
//!     EvalSet::new("val", &val_data, &val_labels),
//! ];
//! ```
//!
//! # Available Metrics
//!
//! ## Regression
//! - [`Rmse`]: Root Mean Squared Error
//! - [`Mae`]: Mean Absolute Error
//! - [`Mape`]: Mean Absolute Percentage Error
//! - [`QuantileMetric`]: Pinball loss for quantile regression
//!
//! ## Classification
//! - [`LogLoss`]: Binary cross-entropy
//! - [`Accuracy`]: Binary classification accuracy
//! - [`Auc`]: Area Under ROC Curve
//! - [`MulticlassLogLoss`]: Multiclass cross-entropy
//! - [`MulticlassAccuracy`]: Multiclass accuracy

mod classification;
mod regression;

pub use classification::{Accuracy, Auc, LogLoss, MarginAccuracy, MulticlassAccuracy, MulticlassLogLoss};
pub use regression::{HuberMetric, Mae, Mape, PoissonDeviance, QuantileMetric, Rmse};

use crate::inference::common::PredictionKind;

// =============================================================================
// Helpers
// =============================================================================

// Re-export weight_iter from utils for internal use
pub(super) use crate::utils::weight_iter;

// =============================================================================
// MetricKind (for defaults / configuration)
// =============================================================================

/// Metric identifier used for configuration and objective defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum MetricKind {
    // Regression
    Rmse,
    Mae,
    Mape,
    Huber,
    PoissonDeviance,
    Quantile,

    // Classification
    LogLoss,
    Accuracy,
    MarginAccuracy,
    Auc,
    MulticlassLogLoss,
    MulticlassAccuracy,
}

// =============================================================================
// Metric Enum (Convenience wrapper)
// =============================================================================

/// A dynamically-dispatched metric function.
///
/// This enum wraps all available metrics using the newtype pattern,
/// allowing metric selection at runtime without generics.
///
/// Each variant wraps its corresponding struct type directly, enabling
/// clean delegation and avoiding per-call struct instantiation.
///
/// # Example
///
/// ```ignore
/// use boosters::training::Metric;
///
/// // Use convenience constructors
/// let rmse = Metric::rmse();
/// let logloss = Metric::logloss();
/// let accuracy = Metric::accuracy_with_threshold(0.7);
///
/// // No metric (skips evaluation)
/// let none = Metric::none();
/// ```
#[derive(Clone)]
pub enum Metric {
    /// No metric - skips evaluation entirely.
    ///
    /// When used, the trainer skips metric computation and prediction
    /// transformation, avoiding wasted compute.
    None,
    /// Root Mean Squared Error (regression).
    Rmse(Rmse),
    /// Mean Absolute Error (regression).
    Mae(Mae),
    /// Mean Absolute Percentage Error (regression).
    Mape(Mape),
    /// Log Loss / Binary Cross-Entropy (binary classification).
    LogLoss(LogLoss),
    /// Accuracy (binary classification) with configurable threshold.
    Accuracy(Accuracy),
    /// Margin-based accuracy (threshold=0.0, for hinge loss).
    MarginAccuracy(MarginAccuracy),
    /// Area Under ROC Curve (binary classification).
    Auc(Auc),
    /// Multiclass Log Loss (multiclass classification).
    MulticlassLogLoss(MulticlassLogLoss),
    /// Multiclass Accuracy (multiclass classification).
    MulticlassAccuracy(MulticlassAccuracy),
    /// Quantile metric (pinball loss).
    Quantile(QuantileMetric),
    /// Huber metric for robust regression.
    Huber(HuberMetric),
    /// Poisson Deviance for count data.
    PoissonDeviance(PoissonDeviance),
    /// Custom user-provided metric.
    Custom(std::sync::Arc<dyn MetricFn>),
}

impl std::fmt::Debug for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::Rmse(inner) => f.debug_tuple("Rmse").field(inner).finish(),
            Self::Mae(inner) => f.debug_tuple("Mae").field(inner).finish(),
            Self::Mape(inner) => f.debug_tuple("Mape").field(inner).finish(),
            Self::LogLoss(inner) => f.debug_tuple("LogLoss").field(inner).finish(),
            Self::Accuracy(inner) => f.debug_tuple("Accuracy").field(inner).finish(),
            Self::MarginAccuracy(inner) => f.debug_tuple("MarginAccuracy").field(inner).finish(),
            Self::Auc(inner) => f.debug_tuple("Auc").field(inner).finish(),
            Self::MulticlassLogLoss(inner) => f.debug_tuple("MulticlassLogLoss").field(inner).finish(),
            Self::MulticlassAccuracy(inner) => f.debug_tuple("MulticlassAccuracy").field(inner).finish(),
            Self::Quantile(inner) => f.debug_tuple("Quantile").field(inner).finish(),
            Self::Huber(inner) => f.debug_tuple("Huber").field(inner).finish(),
            Self::PoissonDeviance(inner) => f.debug_tuple("PoissonDeviance").field(inner).finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"<dyn MetricFn>").finish(),
        }
    }
}

impl Default for Metric {
    fn default() -> Self {
        Self::Rmse(Rmse)
    }
}

impl Metric {
    // =========================================================================
    // Convenience Constructors
    // =========================================================================

    /// No metric - skips evaluation entirely.
    ///
    /// When used, the trainer skips metric computation and prediction
    /// transformation, avoiding wasted compute.
    pub fn none() -> Self {
        Self::None
    }

    /// Root Mean Squared Error for regression.
    pub fn rmse() -> Self {
        Self::Rmse(Rmse)
    }

    /// Mean Absolute Error for regression.
    pub fn mae() -> Self {
        Self::Mae(Mae)
    }

    /// Mean Absolute Percentage Error for regression.
    pub fn mape() -> Self {
        Self::Mape(Mape)
    }

    /// Log Loss for binary classification.
    pub fn logloss() -> Self {
        Self::LogLoss(LogLoss)
    }

    /// Accuracy for binary classification with threshold 0.5.
    pub fn accuracy() -> Self {
        Self::Accuracy(Accuracy::default())
    }

    /// Accuracy for binary classification with custom threshold.
    pub fn accuracy_with_threshold(threshold: f32) -> Self {
        Self::Accuracy(Accuracy::with_threshold(threshold))
    }

    /// Margin-based accuracy (threshold=0.0) for hinge loss.
    pub fn margin_accuracy() -> Self {
        Self::MarginAccuracy(MarginAccuracy::default())
    }

    /// Area Under ROC Curve for binary classification.
    pub fn auc() -> Self {
        Self::Auc(Auc)
    }

    /// Multiclass Log Loss for multiclass classification.
    pub fn multiclass_logloss() -> Self {
        Self::MulticlassLogLoss(MulticlassLogLoss)
    }

    /// Multiclass Accuracy.
    pub fn multiclass_accuracy() -> Self {
        Self::MulticlassAccuracy(MulticlassAccuracy)
    }

    /// Quantile metric (pinball loss) with given alpha.
    pub fn quantile(alpha: f32) -> Self {
        Self::Quantile(QuantileMetric::new(vec![alpha]))
    }

    /// Multi-quantile metric with given alphas.
    pub fn multi_quantile(alphas: Vec<f32>) -> Self {
        Self::Quantile(QuantileMetric::new(alphas))
    }

    /// Huber metric with given delta.
    pub fn huber(delta: f32) -> Self {
        Self::Huber(HuberMetric::new(delta.into()))
    }

    /// Poisson Deviance for count data.
    pub fn poisson_deviance() -> Self {
        Self::PoissonDeviance(PoissonDeviance)
    }

    /// Custom user-provided metric.
    pub fn custom(metric: impl MetricFn + 'static) -> Self {
        Self::Custom(std::sync::Arc::new(metric))
    }

    // =========================================================================
    // Factory Methods
    // =========================================================================

    /// Get the default metric for a MetricKind.
    pub fn from_kind(kind: MetricKind) -> Self {
        match kind {
            MetricKind::Rmse => Self::rmse(),
            MetricKind::Mae => Self::mae(),
            MetricKind::Mape => Self::mape(),
            MetricKind::Huber => Self::huber(1.0),
            MetricKind::PoissonDeviance => Self::poisson_deviance(),
            MetricKind::Quantile => Self::quantile(0.5),
            MetricKind::LogLoss => Self::logloss(),
            MetricKind::Accuracy => Self::accuracy(),
            MetricKind::MarginAccuracy => Self::margin_accuracy(),
            MetricKind::Auc => Self::auc(),
            MetricKind::MulticlassLogLoss => Self::multiclass_logloss(),
            MetricKind::MulticlassAccuracy => Self::multiclass_accuracy(),
        }
    }
}

impl MetricFn for Metric {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        match self {
            Self::None => f64::NAN,
            Self::Rmse(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Mae(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Mape(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::LogLoss(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Accuracy(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::MarginAccuracy(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Auc(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::MulticlassLogLoss(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::MulticlassAccuracy(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Quantile(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Huber(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::PoissonDeviance(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Custom(inner) => inner.compute(n_rows, n_outputs, predictions, targets, weights),
        }
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        match self {
            Self::None => PredictionKind::Margin,
            Self::Rmse(inner) => inner.expected_prediction_kind(),
            Self::Mae(inner) => inner.expected_prediction_kind(),
            Self::Mape(inner) => inner.expected_prediction_kind(),
            Self::LogLoss(inner) => inner.expected_prediction_kind(),
            Self::Accuracy(inner) => inner.expected_prediction_kind(),
            Self::MarginAccuracy(inner) => inner.expected_prediction_kind(),
            Self::Auc(inner) => inner.expected_prediction_kind(),
            Self::MulticlassLogLoss(inner) => inner.expected_prediction_kind(),
            Self::MulticlassAccuracy(inner) => inner.expected_prediction_kind(),
            Self::Quantile(inner) => inner.expected_prediction_kind(),
            Self::Huber(inner) => inner.expected_prediction_kind(),
            Self::PoissonDeviance(inner) => inner.expected_prediction_kind(),
            Self::Custom(inner) => inner.expected_prediction_kind(),
        }
    }

    fn higher_is_better(&self) -> bool {
        match self {
            Self::None => false,
            Self::Rmse(inner) => inner.higher_is_better(),
            Self::Mae(inner) => inner.higher_is_better(),
            Self::Mape(inner) => inner.higher_is_better(),
            Self::LogLoss(inner) => inner.higher_is_better(),
            Self::Accuracy(inner) => inner.higher_is_better(),
            Self::MarginAccuracy(inner) => inner.higher_is_better(),
            Self::Auc(inner) => inner.higher_is_better(),
            Self::MulticlassLogLoss(inner) => inner.higher_is_better(),
            Self::MulticlassAccuracy(inner) => inner.higher_is_better(),
            Self::Quantile(inner) => inner.higher_is_better(),
            Self::Huber(inner) => inner.higher_is_better(),
            Self::PoissonDeviance(inner) => inner.higher_is_better(),
            Self::Custom(inner) => inner.higher_is_better(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::None => "<none>",
            Self::Rmse(inner) => inner.name(),
            Self::Mae(inner) => inner.name(),
            Self::Mape(inner) => inner.name(),
            Self::LogLoss(inner) => inner.name(),
            Self::Accuracy(inner) => inner.name(),
            Self::MarginAccuracy(inner) => inner.name(),
            Self::Auc(inner) => inner.name(),
            Self::MulticlassLogLoss(inner) => inner.name(),
            Self::MulticlassAccuracy(inner) => inner.name(),
            Self::Quantile(inner) => inner.name(),
            Self::Huber(inner) => inner.name(),
            Self::PoissonDeviance(inner) => inner.name(),
            Self::Custom(inner) => inner.name(),
        }
    }

    fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}

// =============================================================================
// Metric Trait
// =============================================================================

/// A metric for evaluating model quality.
///
/// Unlike objectives (which compute gradients for optimization),
/// metrics compute scalar values for model evaluation and monitoring.
///
/// # Multi-Output Support
///
/// For multi-output models (multiclass, multi-quantile), pass `n_outputs > 1`.
/// The predictions buffer uses **column-major** layout: `predictions[output * n_rows + row]`.
/// This matches the training prediction layout for zero-copy evaluation.
///
/// # Weighted Evaluation
///
/// Pass `weights: Some(&weights)` for sample-weighted metrics. The default
/// implementation treats `None` as uniform weights.
///
/// # Implementation Notes
///
/// - `evaluate`: Called with predictions, labels, optional weights, and n_outputs
/// - Higher is better for some metrics (accuracy, AUC), lower for others (RMSE, logloss)
/// - Use `higher_is_better()` to determine the direction
pub trait MetricFn: Send + Sync {
    /// Compute metric value.
    ///
    /// Predictions are expected in **column-major** layout: `predictions[output * n_rows + row]`.
    ///
    /// Pass an empty `weights` slice for unweighted computation.
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64;

    /// What prediction space does this metric expect?
    fn expected_prediction_kind(&self) -> PredictionKind;

    /// Whether higher values indicate better performance.
    ///
    /// - `true`: Higher is better (accuracy, AUC)
    /// - `false`: Lower is better (RMSE, MAE, logloss)
    fn higher_is_better(&self) -> bool;

    /// Name of the metric (for logging).
    fn name(&self) -> &'static str;

    /// Whether this metric is enabled.
    ///
    /// When `false`, the trainer should skip metric computation entirely,
    /// avoiding wasted compute on prediction transformation and evaluation.
    ///
    /// Default implementation returns `true`. Only `NoMetric` returns `false`.
    fn is_enabled(&self) -> bool {
        true
    }
}

// hello? Read the comment above ^^