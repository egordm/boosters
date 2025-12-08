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

pub use classification::{Accuracy, Auc, LogLoss, MulticlassAccuracy, MulticlassLogLoss};
pub use regression::{Mae, Mape, QuantileMetric, Rmse};

use crate::data::ColumnAccess;

// =============================================================================
// Evaluation Set
// =============================================================================

/// Named dataset for evaluation during training.
///
/// Evaluation sets are used to track model performance on multiple datasets
/// (train, validation, test) during training. Each set has a name that appears
/// as a prefix in training logs.
///
/// # Example
///
/// ```ignore
/// let eval_sets = vec![
///     EvalSet::new("train", &train_data, &train_labels),
///     EvalSet::new("val", &val_data, &val_labels),
/// ];
/// // Logs: [0] train-rmse:15.23  val-rmse:16.12
///
/// // With sample weights:
/// let weighted_eval = EvalSet::with_weights("train", &data, &labels, &weights);
/// ```
pub struct EvalSet<'a, D> {
    /// Dataset name (appears in logs as prefix, e.g., "train", "val", "test").
    pub name: &'a str,
    /// Feature matrix (must implement `ColumnAccess` for training).
    pub data: &'a D,
    /// Labels (length = n_samples for single-output, or n_samples for multi-output
    /// where labels are class indices or target values).
    pub labels: &'a [f32],
    /// Optional sample weights for weighted metric computation.
    pub weights: Option<&'a [f32]>,
}

impl<'a, D: ColumnAccess> EvalSet<'a, D> {
    /// Create a new evaluation set without sample weights.
    pub fn new(name: &'a str, data: &'a D, labels: &'a [f32]) -> Self {
        Self {
            name,
            data,
            labels,
            weights: None,
        }
    }

    /// Create a new evaluation set with sample weights.
    pub fn with_weights(
        name: &'a str,
        data: &'a D,
        labels: &'a [f32],
        weights: &'a [f32],
    ) -> Self {
        Self {
            name,
            data,
            labels,
            weights: Some(weights),
        }
    }
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
    /// Evaluate the metric with optional sample weights.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions, shape `[n_samples, n_outputs]` flattened
    /// * `labels` - Ground truth labels, shape `[n_samples]`
    /// * `weights` - Optional sample weights, shape `[n_samples]`. `None` = uniform weights.
    /// * `n_outputs` - Number of outputs per sample (1 for regression/binary, K for multiclass)
    ///
    /// # Returns
    ///
    /// Scalar metric value.
    fn evaluate(
        &self,
        predictions: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        n_outputs: usize,
    ) -> f64;

    /// Whether higher values indicate better performance.
    ///
    /// - `true`: Higher is better (accuracy, AUC)
    /// - `false`: Lower is better (RMSE, MAE, logloss)
    fn higher_is_better(&self) -> bool;

    /// Name of the metric (for logging).
    fn name(&self) -> &str;
}

// =============================================================================
// EvalMetric Enum
// =============================================================================

/// Evaluation metric for training and early stopping.
///
/// This enum provides a convenient way to specify which metric to use
/// for evaluation during training. It implements the same interface as
/// the [`Metric`] trait but without requiring boxing.
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::EvalMetric;
///
/// let metric = EvalMetric::Rmse;
/// let value = metric.evaluate(&predictions, &labels, None, 1);
/// println!("{}: {:.4}", metric.name(), value);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub enum EvalMetric {
    /// Root Mean Squared Error (default for regression).
    #[default]
    Rmse,
    /// Mean Absolute Error.
    Mae,
    /// Binary cross-entropy (log loss).
    LogLoss,
    /// Binary classification accuracy with configurable threshold.
    Accuracy {
        /// Threshold for converting probabilities to classes (default: 0.5).
        threshold: f32,
    },
    /// Area Under the ROC Curve.
    Auc,
    /// Mean Absolute Percentage Error.
    Mape,
    /// Multiclass cross-entropy (log loss).
    MulticlassLogLoss,
    /// Multiclass accuracy (expects class indices).
    MulticlassAccuracy,
    /// Quantile loss (pinball loss) for quantile regression.
    Quantile {
        /// Quantile levels (e.g., `vec![0.1, 0.5, 0.9]`).
        alphas: Vec<f32>,
    },
}

impl Metric for EvalMetric {
    fn evaluate(
        &self,
        predictions: &[f32],
        labels: &[f32],
        weights: Option<&[f32]>,
        n_outputs: usize,
    ) -> f64 {
        match self {
            EvalMetric::Rmse => Rmse.evaluate(predictions, labels, weights, n_outputs),
            EvalMetric::Mae => Mae.evaluate(predictions, labels, weights, n_outputs),
            EvalMetric::LogLoss => LogLoss.evaluate(predictions, labels, weights, n_outputs),
            EvalMetric::Accuracy { threshold } => {
                Accuracy::with_threshold(*threshold).evaluate(predictions, labels, weights, n_outputs)
            }
            EvalMetric::Auc => Auc.evaluate(predictions, labels, weights, n_outputs),
            EvalMetric::Mape => Mape.evaluate(predictions, labels, weights, n_outputs),
            EvalMetric::MulticlassLogLoss => {
                MulticlassLogLoss.evaluate(predictions, labels, weights, n_outputs)
            }
            EvalMetric::MulticlassAccuracy => {
                MulticlassAccuracy.evaluate(predictions, labels, weights, n_outputs)
            }
            EvalMetric::Quantile { alphas } => {
                QuantileMetric::new(alphas.clone()).evaluate(predictions, labels, weights, n_outputs)
            }
        }
    }

    fn higher_is_better(&self) -> bool {
        match self {
            EvalMetric::Rmse => false,
            EvalMetric::Mae => false,
            EvalMetric::LogLoss => false,
            EvalMetric::Accuracy { .. } => true,
            EvalMetric::Auc => true,
            EvalMetric::Mape => false,
            EvalMetric::MulticlassLogLoss => false,
            EvalMetric::MulticlassAccuracy => true,
            EvalMetric::Quantile { .. } => false,
        }
    }

    fn name(&self) -> &str {
        match self {
            EvalMetric::Rmse => "rmse",
            EvalMetric::Mae => "mae",
            EvalMetric::LogLoss => "logloss",
            EvalMetric::Accuracy { .. } => "accuracy",
            EvalMetric::Auc => "auc",
            EvalMetric::Mape => "mape",
            EvalMetric::MulticlassLogLoss => "mlogloss",
            EvalMetric::MulticlassAccuracy => "accuracy",
            EvalMetric::Quantile { .. } => "quantile",
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
    fn eval_metric_higher_is_better() {
        assert!(!EvalMetric::Rmse.higher_is_better());
        assert!(!EvalMetric::Mae.higher_is_better());
        assert!(!EvalMetric::LogLoss.higher_is_better());
        assert!(EvalMetric::Accuracy { threshold: 0.5 }.higher_is_better());
        assert!(EvalMetric::Auc.higher_is_better());
        assert!(!EvalMetric::Mape.higher_is_better());
        assert!(!EvalMetric::MulticlassLogLoss.higher_is_better());
        assert!(EvalMetric::MulticlassAccuracy.higher_is_better());
        assert!(!EvalMetric::Quantile { alphas: vec![0.5] }.higher_is_better());
    }

    #[test]
    fn eval_metric_names() {
        assert_eq!(EvalMetric::Rmse.name(), "rmse");
        assert_eq!(EvalMetric::Mae.name(), "mae");
        assert_eq!(EvalMetric::LogLoss.name(), "logloss");
        assert_eq!(EvalMetric::Accuracy { threshold: 0.5 }.name(), "accuracy");
        assert_eq!(EvalMetric::Auc.name(), "auc");
        assert_eq!(EvalMetric::Mape.name(), "mape");
        assert_eq!(EvalMetric::MulticlassLogLoss.name(), "mlogloss");
        assert_eq!(EvalMetric::MulticlassAccuracy.name(), "accuracy");
        assert_eq!(EvalMetric::Quantile { alphas: vec![0.5] }.name(), "quantile");
    }

    #[test]
    fn eval_metric_delegates_to_implementations() {
        let preds = vec![1.0, 2.0, 3.0];
        let labels = vec![1.5, 2.5, 3.5];

        // Test that EvalMetric delegates correctly
        let enum_result = EvalMetric::Rmse.evaluate(&preds, &labels, None, 1);
        let direct_result = Rmse.evaluate(&preds, &labels, None, 1);
        assert!((enum_result - direct_result).abs() < 1e-10);
    }
}
