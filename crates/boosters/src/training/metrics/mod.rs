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

pub use classification::{
    Accuracy, Auc, LogLoss, MarginAccuracy, MulticlassAccuracy, MulticlassLogLoss,
};
use ndarray::ArrayView2;
pub use regression::{HuberMetric, Mae, Mape, PoissonDeviance, QuantileMetric, Rmse};

use crate::data::{TargetsView, WeightsView};
use crate::inference::PredictionKind;

// =============================================================================
// Custom Metric
// =============================================================================

/// Type alias for the custom metric compute function.
///
/// Takes predictions (shape `[n_outputs, n_samples]`), targets, and weights.
/// Returns a scalar metric value.
pub type CustomMetricFn = Box<
    dyn Fn(ArrayView2<f32>, TargetsView<'_>, WeightsView<'_>) -> f64 + Send + Sync + 'static,
>;

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
pub struct CustomMetric {
    /// Name of the metric (for logging).
    pub name: &'static str,
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
    /// * `name` - Name for logging (e.g., "custom_mae"). Must be a static string.
    /// * `compute_fn` - Function that computes the metric value
    /// * `expected_prediction_kind` - What prediction space this metric expects
    /// * `higher_is_better` - Whether higher values indicate better performance
    pub fn new(
        name: &'static str,
        compute_fn: impl Fn(ArrayView2<f32>, TargetsView<'_>, WeightsView<'_>) -> f64
            + Send
            + Sync
            + 'static,
        expected_prediction_kind: PredictionKind,
        higher_is_better: bool,
    ) -> Self {
        Self {
            name,
            compute_fn: Box::new(compute_fn),
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
    pub fn name(&self) -> &'static str {
        self.name
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

impl Clone for CustomMetric {
    fn clone(&self) -> Self {
        panic!("CustomMetric cannot be cloned because it contains a boxed closure. Use Arc<CustomMetric> if sharing is needed.");
    }
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
    Custom(std::sync::Arc<CustomMetric>),
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
            Self::MulticlassLogLoss(inner) => {
                f.debug_tuple("MulticlassLogLoss").field(inner).finish()
            }
            Self::MulticlassAccuracy(inner) => {
                f.debug_tuple("MulticlassAccuracy").field(inner).finish()
            }
            Self::Quantile(inner) => f.debug_tuple("Quantile").field(inner).finish(),
            Self::Huber(inner) => f.debug_tuple("Huber").field(inner).finish(),
            Self::PoissonDeviance(inner) => f.debug_tuple("PoissonDeviance").field(inner).finish(),
            Self::Custom(inner) => f.debug_tuple("Custom").field(inner).finish(),
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// use boosters::training::{CustomMetric, Metric};
    /// use boosters::inference::PredictionKind;
    ///
    /// let metric = Metric::custom(CustomMetric::new(
    ///     "custom_mae",
    ///     |preds, targets, _weights| {
    ///         // Compute custom metric
    ///         0.0
    ///     },
    ///     PredictionKind::Value,
    ///     false, // lower is better
    /// ));
    /// ```
    pub fn custom(metric: CustomMetric) -> Self {
        Self::Custom(std::sync::Arc::new(metric))
    }
}

impl MetricFn for Metric {
    fn compute(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> f64 {
        match self {
            Self::None => f64::NAN,
            Self::Rmse(inner) => inner.compute(predictions, targets, weights),
            Self::Mae(inner) => inner.compute(predictions, targets, weights),
            Self::Mape(inner) => inner.compute(predictions, targets, weights),
            Self::LogLoss(inner) => inner.compute(predictions, targets, weights),
            Self::Accuracy(inner) => inner.compute(predictions, targets, weights),
            Self::MarginAccuracy(inner) => inner.compute(predictions, targets, weights),
            Self::Auc(inner) => inner.compute(predictions, targets, weights),
            Self::MulticlassLogLoss(inner) => inner.compute(predictions, targets, weights),
            Self::MulticlassAccuracy(inner) => inner.compute(predictions, targets, weights),
            Self::Quantile(inner) => inner.compute(predictions, targets, weights),
            Self::Huber(inner) => inner.compute(predictions, targets, weights),
            Self::PoissonDeviance(inner) => inner.compute(predictions, targets, weights),
            Self::Custom(inner) => inner.compute(predictions, targets, weights),
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
            Self::Custom(inner) => inner.expected_prediction_kind,
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
            Self::Custom(inner) => inner.higher_is_better,
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
            Self::Custom(inner) => inner.name,
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
    /// Use `WeightsView::none()` for unweighted computation.
    fn compute(
        &self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
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

        let metric = Metric::custom(custom_mae);

        // Test compute
        let preds = make_preds(1, 2, &[1.0, 2.0]);
        let labels = make_targets(&[0.0, 0.0]);
        let value = metric.compute(preds.view(), TargetsView::new(labels.view()), WeightsView::None);
        approx::assert_abs_diff_eq!(value, 1.5, epsilon = 1e-10);

        // Test properties
        assert_eq!(metric.name(), "custom_mae");
        assert!(!metric.higher_is_better());
        assert_eq!(metric.expected_prediction_kind(), PredictionKind::Value);
        assert!(metric.is_enabled());
    }

    #[test]
    fn metric_none_disabled() {
        let metric = Metric::none();
        assert!(!metric.is_enabled());
        assert_eq!(metric.name(), "<none>");
    }
}
