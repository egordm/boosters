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
// MetricFunction Enum (Convenience wrapper)
// =============================================================================

/// A dynamically-dispatched metric function.
///
/// This enum wraps all available metrics and implements the `Metric` trait,
/// allowing metric selection at runtime without generics.
///
/// # Example
///
/// ```ignore
/// use boosters::training::MetricFunction;
///
/// let metric = MetricFunction::from_objective_str("binary:logistic");
/// // metric is MetricFunction::LogLoss
/// ```
#[derive(Debug, Clone)]
pub enum MetricFunction {
    /// Root Mean Squared Error (regression).
    Rmse,
    /// Mean Absolute Error (regression).
    Mae,
    /// Mean Absolute Percentage Error (regression).
    Mape,
    /// Log Loss / Binary Cross-Entropy (binary classification).
    LogLoss,
    /// Accuracy (binary classification).
    Accuracy { threshold: f32 },
    /// Area Under ROC Curve (binary classification).
    Auc,
    /// Multiclass Log Loss (multiclass classification).
    MulticlassLogLoss,
    /// Multiclass Accuracy (multiclass classification).
    MulticlassAccuracy,
    /// Quantile metric (quantile regression).
    Quantile { alpha: f32 },
    /// Multi-quantile metric.
    MultiQuantile { alphas: Vec<f32> },
    /// Huber metric.
    Huber { delta: f32 },
    /// Poisson Deviance.
    PoissonDeviance,
}

impl Default for MetricFunction {
    fn default() -> Self {
        Self::Rmse
    }
}

impl MetricFunction {
    /// Get the default metric for an objective string.
    pub fn from_objective_str(objective: &str) -> Self {
        match objective {
            "squared_error" | "reg:squared_error" | "reg:squarederror" => Self::Rmse,
            "absolute_error" | "reg:absoluteerror" | "mae" => Self::Mae,
            "binary:logistic" | "binary_logistic" | "logistic" => Self::LogLoss,
            "binary:hinge" => Self::Accuracy { threshold: 0.0 },
            "multi:softmax" | "multiclass" | "softmax" | "multi:softprob" => Self::MulticlassLogLoss,
            "reg:quantile" | "quantile" => Self::Quantile { alpha: 0.5 },
            "reg:pseudohuber" | "huber" => Self::Huber { delta: 1.0 },
            "poisson" | "count:poisson" => Self::PoissonDeviance,
            _ => Self::Rmse,
        }
    }

    /// Get the default metric for a MetricKind.
    pub fn from_kind(kind: MetricKind) -> Self {
        match kind {
            MetricKind::Rmse => Self::Rmse,
            MetricKind::Mae => Self::Mae,
            MetricKind::Mape => Self::Mape,
            MetricKind::Huber => Self::Huber { delta: 1.0 },
            MetricKind::PoissonDeviance => Self::PoissonDeviance,
            MetricKind::Quantile => Self::Quantile { alpha: 0.5 },
            MetricKind::LogLoss => Self::LogLoss,
            MetricKind::Accuracy => Self::Accuracy { threshold: 0.5 },
            MetricKind::MarginAccuracy => Self::Accuracy { threshold: 0.0 },
            MetricKind::Auc => Self::Auc,
            MetricKind::MulticlassLogLoss => Self::MulticlassLogLoss,
            MetricKind::MulticlassAccuracy => Self::MulticlassAccuracy,
        }
    }
}

impl Metric for MetricFunction {
    fn compute(
        &self,
        n_rows: usize,
        n_outputs: usize,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
    ) -> f64 {
        match self {
            Self::Rmse => Rmse.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Mae => Mae.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Mape => Mape.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::LogLoss => LogLoss.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::Accuracy { threshold } => {
                Accuracy { threshold: *threshold }.compute(n_rows, n_outputs, predictions, targets, weights)
            }
            Self::Auc => Auc.compute(n_rows, n_outputs, predictions, targets, weights),
            Self::MulticlassLogLoss => {
                MulticlassLogLoss.compute(n_rows, n_outputs, predictions, targets, weights)
            }
            Self::MulticlassAccuracy => {
                MulticlassAccuracy.compute(n_rows, n_outputs, predictions, targets, weights)
            }
            Self::Quantile { alpha } => {
                QuantileMetric::compute_single(*alpha, n_rows, predictions, targets, weights)
            }
            Self::MultiQuantile { alphas } => {
                QuantileMetric::new(alphas.clone()).compute(n_rows, n_outputs, predictions, targets, weights)
            }
            Self::Huber { delta } => {
                HuberMetric::new((*delta).into()).compute(n_rows, n_outputs, predictions, targets, weights)
            }
            Self::PoissonDeviance => {
                PoissonDeviance.compute(n_rows, n_outputs, predictions, targets, weights)
            }
        }
    }

    fn expected_prediction_kind(&self) -> PredictionKind {
        match self {
            Self::Rmse | Self::Mae | Self::Mape | Self::Huber { .. } | Self::PoissonDeviance => {
                PredictionKind::Value
            }
            Self::Quantile { .. } | Self::MultiQuantile { .. } => PredictionKind::Value,
            Self::LogLoss | Self::Auc | Self::Accuracy { .. } => PredictionKind::Probability,
            Self::MulticlassLogLoss | Self::MulticlassAccuracy => PredictionKind::Probability,
        }
    }

    fn higher_is_better(&self) -> bool {
        match self {
            Self::Rmse | Self::Mae | Self::Mape | Self::LogLoss | Self::MulticlassLogLoss => false,
            Self::Quantile { .. } | Self::MultiQuantile { .. } | Self::Huber { .. } => false,
            Self::PoissonDeviance => false,
            Self::Accuracy { .. } | Self::Auc | Self::MulticlassAccuracy => true,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Rmse => "rmse",
            Self::Mae => "mae",
            Self::Mape => "mape",
            Self::LogLoss => "logloss",
            Self::Accuracy { .. } => "accuracy",
            Self::Auc => "auc",
            Self::MulticlassLogLoss => "mlogloss",
            Self::MulticlassAccuracy => "accuracy",
            Self::Quantile { .. } => "quantile",
            Self::MultiQuantile { .. } => "multi_quantile",
            Self::Huber { .. } => "huber",
            Self::PoissonDeviance => "poisson_deviance",
        }
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
pub trait Metric: Send + Sync {
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
}
