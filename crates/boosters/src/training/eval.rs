//! Evaluation utilities for training.
//!
//! Provides the [`Evaluator`] component for computing metrics during training,
//! and [`MetricValue`] for wrapping computed metrics with metadata.

use crate::data::Dataset;
use crate::inference::common::PredictionKind;

use super::metrics::MetricFn;
use super::objectives::ObjectiveFn;

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
// EvalSet
// =============================================================================

/// Named evaluation dataset.
#[derive(Debug, Clone, Copy)]
pub struct EvalSet<'a> {
    pub name: &'a str,
    pub dataset: &'a Dataset,
}

impl<'a> EvalSet<'a> {
    pub fn new(name: &'a str, dataset: &'a Dataset) -> Self {
        Self { name, dataset }
    }
}

// =============================================================================
// Evaluator
// =============================================================================

/// Evaluation state for computing metrics during training.
///
/// The Evaluator manages buffers and provides a clean interface for computing
/// metrics on training and evaluation datasets.
///
/// # Example
///
/// ```ignore
/// use boosters::training::{Evaluator, SquaredLoss, Rmse, EvalSet};
///
/// let objective = SquaredLoss;
/// let metric = Rmse;
/// let mut evaluator = Evaluator::new(&objective, &metric, n_outputs);
///
/// // During training loop:
/// let metrics = evaluator.evaluate_round(
///     train_predictions, train_targets, train_weights, train_n_rows,
///     &eval_sets, &eval_predictions,
/// );
/// let early_stop_value = evaluator.early_stop_value(&metrics, eval_set_idx);
/// ```
pub struct Evaluator<'a, O: ObjectiveFn, M: MetricFn> {
    objective: &'a O,
    metric: &'a M,
    n_outputs: usize,
    transform_buffer: Vec<f32>,
}

impl<'a, O: ObjectiveFn, M: MetricFn> Evaluator<'a, O, M> {
    /// Create a new evaluator.
    ///
    /// # Arguments
    ///
    /// * `objective` - The objective function (for prediction transforms)
    /// * `metric` - The metric to compute
    /// * `n_outputs` - Number of outputs per sample
    pub fn new(objective: &'a O, metric: &'a M, n_outputs: usize) -> Self {
        Self {
            objective,
            metric,
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
    pub fn metric_name(&self) -> &'static str {
        self.metric.name()
    }

    /// Whether the metric is enabled.
    ///
    /// When `false`, evaluation should be skipped entirely.
    pub fn is_enabled(&self) -> bool {
        self.metric.is_enabled()
    }

    /// Compute a single metric value.
    ///
    /// Handles transformation if the metric requires it.
    pub fn compute(
        &mut self,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        n_rows: usize,
    ) -> f64 {
        let needs_transform =
            self.metric.expected_prediction_kind() != PredictionKind::Margin;

        if needs_transform {
            // Ensure buffer is large enough
            let required = n_rows * self.n_outputs;
            if self.transform_buffer.len() < required {
                self.transform_buffer.resize(required, 0.0);
            }

            self.transform_buffer[..required]
                .copy_from_slice(&predictions[..required]);
            self.objective.transform_predictions(
                &mut self.transform_buffer[..required],
                n_rows,
                self.n_outputs,
            );
            self.metric.compute(
                n_rows,
                self.n_outputs,
                &self.transform_buffer[..required],
                targets,
                weights,
            )
        } else {
            self.metric
                .compute(n_rows, self.n_outputs, predictions, targets, weights)
        }
    }

    /// Compute metric and wrap in MetricValue.
    pub fn compute_metric(
        &mut self,
        name: impl Into<String>,
        predictions: &[f32],
        targets: &[f32],
        weights: &[f32],
        n_rows: usize,
    ) -> MetricValue {
        let value = self.compute(predictions, targets, weights, n_rows);
        MetricValue::new(name, value, self.higher_is_better())
    }

    /// Evaluate predictions on training and eval sets for one round.
    ///
    /// Returns a vector of metric values for all datasets.
    /// If the metric is not enabled (e.g., `Metric::None`), returns an empty vector.
    pub fn evaluate_round(
        &mut self,
        train_predictions: &[f32],
        train_targets: &[f32],
        train_weights: &[f32],
        train_n_rows: usize,
        eval_sets: &[EvalSet<'_>],
        eval_predictions: &[Vec<f32>],
    ) -> Vec<MetricValue> {
        // Skip evaluation entirely if metric is not enabled
        if !self.metric.is_enabled() {
            return Vec::new();
        }
        
        let mut metrics = Vec::with_capacity(1 + eval_sets.len());

        // Compute training metric
        let train_metric = self.compute_metric(
            format!("train-{}", self.metric_name()),
            train_predictions,
            train_targets,
            train_weights,
            train_n_rows,
        );
        metrics.push(train_metric);

        // Compute eval set metrics
        for (set_idx, eval_set) in eval_sets.iter().enumerate() {
            let raw_preds = &eval_predictions[set_idx];
            let eval_rows = raw_preds.len() / self.n_outputs;
            let targets = eval_set.dataset.targets();
            let weights = eval_set.dataset.weights().unwrap_or(&[]);

            let metric = self.compute_metric(
                format!("{}-{}", eval_set.name, self.metric_name()),
                raw_preds,
                targets,
                weights,
                eval_rows,
            );
            metrics.push(metric);
        }

        metrics
    }

    /// Get the early stopping value from metrics.
    ///
    /// If `eval_set_idx` is valid, returns the corresponding eval set's metric.
    /// Otherwise, returns the training metric.
    pub fn early_stop_value(metrics: &[MetricValue], eval_set_idx: usize) -> f64 {
        // Index 0 is training, eval sets are 1, 2, ...
        let idx = if eval_set_idx + 1 < metrics.len() {
            eval_set_idx + 1
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
