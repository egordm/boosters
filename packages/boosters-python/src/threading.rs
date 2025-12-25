//! GIL management and threading utilities for Python bindings.
//!
//! This module provides utilities for releasing the Python GIL during expensive
//! Rust operations and capturing training results for Python consumption.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use boosters::training::{EarlyStopAction, EarlyStopping, MetricValue};

/// Evaluation log entry for a single training round.
#[derive(Debug, Clone)]
pub struct RoundLog {
    /// Round number (0-indexed).
    pub round: usize,
    /// Metrics computed for this round.
    pub metrics: Vec<MetricValue>,
}

/// Collects evaluation metrics during training.
///
/// This struct accumulates metrics from each training round. After training,
/// the collected data is converted to a Python dict for the `eval_results` attribute.
///
/// # Structure of eval_results
///
/// ```python
/// {
///     "train-rmse": [0.5, 0.4, 0.3, ...],  # metric per round
///     "valid-rmse": [0.6, 0.5, 0.4, ...],
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct EvalLogger {
    /// Accumulated logs per round.
    rounds: Vec<RoundLog>,
}

impl EvalLogger {
    /// Create a new evaluation logger.
    pub fn new() -> Self {
        Self { rounds: Vec::new() }
    }

    /// Log metrics for a training round.
    pub fn log_round(&mut self, round: usize, metrics: Vec<MetricValue>) {
        self.rounds.push(RoundLog { round, metrics });
    }

    /// Get the number of logged rounds.
    pub fn n_rounds(&self) -> usize {
        self.rounds.len()
    }

    /// Convert logged metrics to Python dict.
    ///
    /// Returns a dict mapping metric names to lists of values.
    pub fn to_python_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        // Collect all metric names
        let mut metric_names: Vec<String> = Vec::new();
        for round in &self.rounds {
            for m in &round.metrics {
                if !metric_names.contains(&m.name) {
                    metric_names.push(m.name.clone());
                }
            }
        }

        // Build value lists per metric
        let mut metric_values: HashMap<String, Vec<f64>> = HashMap::new();
        for name in &metric_names {
            metric_values.insert(name.clone(), Vec::with_capacity(self.rounds.len()));
        }

        for round in &self.rounds {
            // Create lookup for this round's metrics
            let round_metrics: HashMap<&str, f64> =
                round.metrics.iter().map(|m| (m.name.as_str(), m.value)).collect();

            for name in &metric_names {
                let value = round_metrics.get(name.as_str()).copied().unwrap_or(f64::NAN);
                metric_values.get_mut(name).unwrap().push(value);
            }
        }

        // Convert to Python dict
        let dict = PyDict::new_bound(py);
        for (name, values) in metric_values {
            dict.set_item(name, values)?;
        }

        Ok(dict)
    }
}

/// Training result container.
///
/// Captures all results from training that need to be returned to Python:
/// - The trained model (via forest)
/// - Best iteration from early stopping
/// - Best score from early stopping
/// - Evaluation results log
#[derive(Debug)]
pub struct TrainingResult<T> {
    /// The trained model/forest.
    pub model: T,
    /// Best iteration (from early stopping, if enabled).
    pub best_iteration: Option<usize>,
    /// Best score (from early stopping, if enabled).
    pub best_score: Option<f64>,
    /// Evaluation log for all rounds.
    pub eval_log: EvalLogger,
}

impl<T> TrainingResult<T> {
    /// Create a new training result.
    pub fn new(model: T) -> Self {
        Self {
            model,
            best_iteration: None,
            best_score: None,
            eval_log: EvalLogger::new(),
        }
    }

    /// Set best iteration and score from early stopping.
    pub fn with_early_stopping(mut self, early_stopping: &EarlyStopping) -> Self {
        if early_stopping.is_enabled() {
            if let Some(best_value) = early_stopping.best_value() {
                self.best_iteration = Some(early_stopping.best_round());
                self.best_score = Some(best_value);
            }
        }
        self
    }

    /// Set evaluation log.
    pub fn with_eval_log(mut self, eval_log: EvalLogger) -> Self {
        self.eval_log = eval_log;
        self
    }
}

/// Trait extension for early stopping callback tracking during training.
///
/// Wraps `EarlyStopping` to track best iteration for Python results.
pub struct EarlyStoppingTracker {
    inner: EarlyStopping,
    best_n_trees: usize,
}

impl EarlyStoppingTracker {
    /// Create a new tracker.
    pub fn new(patience: usize, higher_is_better: bool) -> Self {
        Self {
            inner: EarlyStopping::new(patience, higher_is_better),
            best_n_trees: 0,
        }
    }

    /// Create a disabled tracker.
    pub fn disabled() -> Self {
        Self {
            inner: EarlyStopping::disabled(),
            best_n_trees: 0,
        }
    }

    /// Check if tracking is enabled.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }

    /// Update with a metric value.
    ///
    /// Returns the action to take and whether this was an improvement.
    pub fn update(&mut self, value: f64, current_n_trees: usize) -> EarlyStopAction {
        let action = self.inner.update(value);
        if action == EarlyStopAction::Improved {
            self.best_n_trees = current_n_trees;
        }
        action
    }

    /// Get the best iteration (number of trees at best score).
    pub fn best_iteration(&self) -> Option<usize> {
        if self.inner.is_enabled() && self.inner.best_value().is_some() {
            Some(self.inner.best_round())
        } else {
            None
        }
    }

    /// Get the best score.
    pub fn best_score(&self) -> Option<f64> {
        self.inner.best_value()
    }

    /// Get the number of trees at best iteration.
    pub fn best_n_trees(&self) -> usize {
        self.best_n_trees
    }

    /// Get the inner EarlyStopping for access to additional methods.
    pub fn inner(&self) -> &EarlyStopping {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_logger_collects_metrics() {
        let mut logger = EvalLogger::new();

        logger.log_round(
            0,
            vec![
                MetricValue::new("train-rmse", 0.5, false),
                MetricValue::new("valid-rmse", 0.6, false),
            ],
        );
        logger.log_round(
            1,
            vec![
                MetricValue::new("train-rmse", 0.4, false),
                MetricValue::new("valid-rmse", 0.5, false),
            ],
        );

        assert_eq!(logger.n_rounds(), 2);
    }

    #[test]
    fn early_stopping_tracker_tracks_best() {
        let mut tracker = EarlyStoppingTracker::new(3, false); // lower is better

        assert_eq!(tracker.update(0.5, 1), EarlyStopAction::Improved);
        assert_eq!(tracker.best_iteration(), Some(0));
        assert_eq!(tracker.best_n_trees(), 1);

        assert_eq!(tracker.update(0.4, 2), EarlyStopAction::Improved);
        assert_eq!(tracker.best_iteration(), Some(1));
        assert_eq!(tracker.best_n_trees(), 2);

        assert_eq!(tracker.update(0.5, 3), EarlyStopAction::Continue);
        assert_eq!(tracker.best_iteration(), Some(1)); // Still 1
        assert_eq!(tracker.best_n_trees(), 2); // Still 2
    }

    #[test]
    fn early_stopping_tracker_disabled() {
        let mut tracker = EarlyStoppingTracker::disabled();
        assert!(!tracker.is_enabled());

        tracker.update(0.5, 1);
        assert_eq!(tracker.best_iteration(), None);
        assert_eq!(tracker.best_score(), None);
    }
}
