//! Evaluation utilities for training.
//!
//! Provides the [`Evaluator`] component for computing metrics during training,
//! and [`MetricValue`] for wrapping computed metrics with metadata.

use ndarray::ArrayView2;

use super::Metric;
use crate::data::{Dataset, TargetsView, WeightsView};
use crate::inference::PredictionKind;
use crate::model::OutputTransform;

// =============================================================================
// MetricValue
// =============================================================================

/// A computed metric value with metadata.
///
/// Wraps a metric value with its name and direction information.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricValue {
    /// Name of the metric (e.g., "train-rmse", "valid-logloss").
    pub name: String,
    /// The computed value.
    pub value: f64,
    /// Whether higher values are better (true for accuracy, false for RMSE).
    pub higher_is_better: bool,
}

impl MetricValue {
    /// Create a new metric value.
    pub fn new(name: impl Into<String>, value: f64, higher_is_better: bool) -> Self {
        Self {
            name: name.into(),
            value,
            higher_is_better,
        }
    }

    /// Returns true if this value is better than another.
    pub fn is_better_than(&self, other: &Self) -> bool {
        if self.higher_is_better {
            self.value > other.value
        } else {
            self.value < other.value
        }
    }

    /// Returns true if this value is better than a raw value.
    pub fn is_better_than_value(&self, other_value: f64) -> bool {
        if self.higher_is_better {
            self.value > other_value
        } else {
            self.value < other_value
        }
    }
}

impl std::fmt::Display for MetricValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {:.6}", self.name, self.value)
    }
}

// =============================================================================
// Evaluator
// =============================================================================

/// Evaluation state for computing metrics during training.
///
/// The Evaluator manages buffers and provides a clean interface for computing
/// metrics on training and validation datasets.
///
/// # Example
///
/// ```ignore
/// use boosters::model::OutputTransform;
/// use boosters::training::{Evaluator, Metric, Objective};
///
/// let objective = Objective::SquaredLoss;
/// let metric = Metric::Rmse;
/// let n_outputs = objective.n_outputs();
/// let mut evaluator = Evaluator::new(&metric, objective.output_transform(), n_outputs);
///
/// // During training loop:
/// let metrics = evaluator.evaluate_round(
///     train_predictions, train_targets, train_weights,
///     val_set.as_ref(), val_predictions.as_ref().map(|p| p.view()),
/// );
/// let early_stop_value = Evaluator::early_stop_value(&metrics, val_set.is_some());
/// ```
pub struct Evaluator<'a> {
    metric: &'a Metric,
    output_transform: OutputTransform,
    n_outputs: usize,
    transform_buffer: Vec<f32>,
}

impl<'a> Evaluator<'a> {
    /// Create a new evaluator.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric to compute
    /// * `output_transform` - Output transform to apply when metric expects non-margin predictions
    /// * `n_outputs` - Number of outputs per sample
    pub fn new(metric: &'a Metric, output_transform: OutputTransform, n_outputs: usize) -> Self {
        Self {
            metric,
            output_transform,
            n_outputs,
            transform_buffer: Vec::new(),
        }
    }

    /// Reset the evaluator for a new dataset size.
    ///
    /// Call this before training if you know the maximum dataset size.
    pub fn reset(&mut self, max_rows: usize) {
        let required = max_rows * self.n_outputs;
        if self.transform_buffer.len() < required {
            self.transform_buffer.resize(required, 0.0);
        }
    }

    /// Whether higher metric values are better.
    pub fn higher_is_better(&self) -> bool {
        self.metric.higher_is_better()
    }

    /// The metric name.
    pub fn metric_name(&self) -> &str {
        self.metric.name()
    }

    /// Compute a single metric value.
    ///
    /// Handles transformation if the metric requires it.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Prediction array, shape `[n_outputs, n_samples]`
    /// * `targets` - Target values
    /// * `weights` - Sample weights, `None` for uniform
    pub fn compute(
        &mut self,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> f64 {
        let needs_transform = self.metric.expected_prediction_kind() != PredictionKind::Margin;

        let n_samples = targets.n_samples();

        if needs_transform {
            // Ensure buffer is large enough
            let required = n_samples * self.n_outputs;
            if self.transform_buffer.len() < required {
                self.transform_buffer.resize(required, 0.0);
            }

            // Copy predictions to buffer for in-place transformation
            let pred_slice = predictions
                .as_slice()
                .expect("predictions must be contiguous");
            self.transform_buffer[..required].copy_from_slice(&pred_slice[..required]);

            let view = ndarray::ArrayViewMut2::from_shape(
                (self.n_outputs, n_samples),
                &mut self.transform_buffer[..required],
            )
            .expect("transform buffer shape mismatch");
            self.output_transform.transform_inplace(view);

            let preds_view = ArrayView2::from_shape(
                (self.n_outputs, n_samples),
                &self.transform_buffer[..required],
            )
            .expect("predictions shape mismatch");

            self.metric.compute(preds_view, targets, weights)
        } else {
            self.metric.compute(predictions, targets, weights)
        }
    }

    /// Compute metric and wrap in MetricValue.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the metric (e.g., "train-rmse")
    /// * `predictions` - Prediction array, shape `[n_outputs, n_samples]`
    /// * `targets` - Target values
    /// * `weights` - Sample weights, `None` for uniform
    pub fn compute_metric(
        &mut self,
        name: impl Into<String>,
        predictions: ArrayView2<f32>,
        targets: TargetsView<'_>,
        weights: WeightsView<'_>,
    ) -> MetricValue {
        let value = self.compute(predictions, targets, weights);
        MetricValue::new(name, value, self.higher_is_better())
    }

    /// Evaluate predictions on training and validation sets for one round.
    ///
    /// Returns a vector of metric values for all datasets.
    /// If the metric is not enabled (e.g., `Metric::None`), returns an empty vector.
    ///
    /// # Arguments
    ///
    /// * `train_predictions` - Training predictions, shape `[n_outputs, n_train_samples]`
    /// * `train_targets` - Training targets
    /// * `train_weights` - Training weights, `None` for uniform
    /// * `val_set` - Optional validation dataset (with targets)
    /// * `val_predictions` - Predictions for validation set (required if val_set is Some)
    pub fn evaluate_round(
        &mut self,
        train_predictions: ArrayView2<f32>,
        train_targets: TargetsView<'_>,
        train_weights: WeightsView<'_>,
        val_set: Option<&Dataset>,
        val_predictions: Option<ArrayView2<f32>>,
    ) -> Vec<MetricValue> {
        // Skip evaluation entirely if metric is Metric::None
        if matches!(self.metric, Metric::None) {
            return Vec::new();
        }

        let mut metrics = Vec::with_capacity(if val_set.is_some() { 2 } else { 1 });

        // Compute training metric
        let train_metric = self.compute_metric(
            format!("train-{}", self.metric_name()),
            train_predictions,
            train_targets,
            train_weights,
        );
        metrics.push(train_metric);

        // Compute validation metric if provided
        if let (Some(val), Some(preds)) = (val_set, val_predictions) {
            let targets = val.targets().expect("validation set must have targets");
            let weights = val.weights();

            let metric = self.compute_metric(
                format!("valid-{}", self.metric_name()),
                preds,
                targets,
                weights,
            );
            metrics.push(metric);
        }

        metrics
    }

    /// Get the early stopping value from metrics.
    ///
    /// Returns the validation metric if available, otherwise the training metric.
    pub fn early_stop_value(metrics: &[MetricValue], has_val_set: bool) -> f64 {
        // Index 0 is training, index 1 is validation (if present)
        let idx = if has_val_set && metrics.len() > 1 {
            1
        } else {
            0
        };
        metrics.get(idx).map(|m| m.value).unwrap_or(f64::NAN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_value_comparison() {
        // Lower is better (RMSE)
        let rmse1 = MetricValue::new("rmse", 0.5, false);
        let rmse2 = MetricValue::new("rmse", 0.7, false);
        assert!(rmse1.is_better_than(&rmse2));
        assert!(!rmse2.is_better_than(&rmse1));

        // Higher is better (accuracy)
        let acc1 = MetricValue::new("acc", 0.9, true);
        let acc2 = MetricValue::new("acc", 0.8, true);
        assert!(acc1.is_better_than(&acc2));
        assert!(!acc2.is_better_than(&acc1));
    }

    #[test]
    fn metric_value_display() {
        let m = MetricValue::new("train-rmse", 0.123456, false);
        assert_eq!(format!("{}", m), "train-rmse: 0.123456");
    }
}
