//! Evaluation metrics for model quality.
//!
//! Metrics are separate from loss functions — a model might be trained with
//! one loss but evaluated with different metrics.
//!
//! # Multi-Output Support
//!
//! Metrics support multi-output models (multiclass, multi-quantile) via the
//! `n_outputs` parameter. The predictions buffer has shape `[n_samples, n_outputs]`
//! in row-major order.
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
// Metric Trait
// =============================================================================

/// A metric for evaluating model quality.
///
/// Unlike [`super::Loss`] which computes gradients for optimization,
/// metrics compute scalar values for model evaluation and monitoring.
///
/// # Multi-Output Support
///
/// For multi-output models (multiclass, multi-quantile), pass `n_outputs > 1`.
/// The predictions buffer has shape `[n_samples, n_outputs]` in row-major order.
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
    /// Predictions are expected in **row-major** layout with shape `(n_rows, n_outputs)`.
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
