//! Early stopping callback for training.
//!
//! Monitors a validation metric and stops training when no improvement is seen
//! for a specified number of rounds.

use super::Metric;

/// Early stopping configuration and state.
///
/// Monitors a validation metric during training and signals when to stop
/// based on lack of improvement over a patience window.
///
/// # Example
///
/// ```
/// use booste_rs::training::{EarlyStopping, Rmse};
///
/// let mut early_stop = EarlyStopping::new(Box::new(Rmse), 5);
///
/// // Simulated training loop
/// for round in 0..100 {
///     let val_preds = vec![0.1, 0.2, 0.3];
///     let val_labels = vec![0.0, 0.1, 0.2];
///
///     if early_stop.should_stop(&val_preds, &val_labels) {
///         println!("Early stopping at round {}", round);
///         break;
///     }
/// }
/// ```
pub struct EarlyStopping {
    /// Metric to monitor.
    metric: Box<dyn Metric>,
    /// Number of rounds without improvement before stopping.
    patience: usize,
    /// Best metric value seen so far.
    best_value: Option<f64>,
    /// Round at which best value was observed.
    best_round: usize,
    /// Current round.
    current_round: usize,
    /// Whether higher metric values are better.
    higher_is_better: bool,
}

impl EarlyStopping {
    /// Create a new early stopping callback.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric to monitor
    /// * `patience` - Number of rounds without improvement before stopping
    pub fn new(metric: Box<dyn Metric>, patience: usize) -> Self {
        let higher_is_better = metric.higher_is_better();
        Self {
            metric,
            patience,
            best_value: None,
            best_round: 0,
            current_round: 0,
            higher_is_better,
        }
    }

    /// Update with new predictions and check if training should stop.
    ///
    /// Returns `true` if training should stop (no improvement for `patience` rounds).
    pub fn should_stop(&mut self, preds: &[f32], labels: &[f32]) -> bool {
        let value = self.metric.compute(preds, labels);
        self.update_with_value(value)
    }

    /// Update with a pre-computed metric value.
    ///
    /// Useful when the metric has already been computed elsewhere.
    pub fn update_with_value(&mut self, value: f64) -> bool {
        let is_improvement = match self.best_value {
            None => true,
            Some(best) => {
                if self.higher_is_better {
                    value > best
                } else {
                    value < best
                }
            }
        };

        if is_improvement {
            self.best_value = Some(value);
            self.best_round = self.current_round;
        }

        self.current_round += 1;

        // Check if we should stop
        self.current_round - self.best_round > self.patience
    }

    /// Get the best metric value observed.
    pub fn best_value(&self) -> Option<f64> {
        self.best_value
    }

    /// Get the round at which the best value was observed.
    pub fn best_round(&self) -> usize {
        self.best_round
    }

    /// Get the current round number.
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get the metric name.
    pub fn metric_name(&self) -> &'static str {
        self.metric.name()
    }

    /// Reset the early stopping state.
    pub fn reset(&mut self) {
        self.best_value = None;
        self.best_round = 0;
        self.current_round = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::Rmse;

    #[test]
    fn early_stopping_no_stop_while_improving() {
        let mut early_stop = EarlyStopping::new(Box::new(Rmse), 3);

        // Simulated improving metrics (RMSE decreasing)
        assert!(!early_stop.update_with_value(1.0));
        assert!(!early_stop.update_with_value(0.9));
        assert!(!early_stop.update_with_value(0.8));
        assert!(!early_stop.update_with_value(0.7));
        assert!(!early_stop.update_with_value(0.6));

        assert_eq!(early_stop.best_round(), 4);
        assert!((early_stop.best_value().unwrap() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_stops_after_patience() {
        let mut early_stop = EarlyStopping::new(Box::new(Rmse), 3);

        // Best at round 0, then no improvement
        // After update: current_round increments, then check current_round - best_round > patience
        assert!(!early_stop.update_with_value(0.5)); // current=1, best=0, 1-0=1 > 3? NO
        assert!(!early_stop.update_with_value(0.6)); // current=2, best=0, 2-0=2 > 3? NO
        assert!(!early_stop.update_with_value(0.7)); // current=3, best=0, 3-0=3 > 3? NO
        assert!(early_stop.update_with_value(0.8)); // current=4, best=0, 4-0=4 > 3? YES!

        assert_eq!(early_stop.best_round(), 0);
        assert!((early_stop.best_value().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_resets_on_improvement() {
        let mut early_stop = EarlyStopping::new(Box::new(Rmse), 3);

        // Initial improvement
        assert!(!early_stop.update_with_value(1.0)); // current=1, best=0
        assert!(!early_stop.update_with_value(1.1)); // current=2, best=0, worse
        assert!(!early_stop.update_with_value(1.2)); // current=3, best=0, worse

        // New improvement resets counter
        assert!(!early_stop.update_with_value(0.9)); // current=4, best=3 (new best!)
        assert!(!early_stop.update_with_value(1.0)); // current=5, best=3, 5-3=2 > 3? NO
        assert!(!early_stop.update_with_value(1.1)); // current=6, best=3, 6-3=3 > 3? NO
        assert!(early_stop.update_with_value(1.2)); // current=7, best=3, 7-3=4 > 3? YES!

        assert_eq!(early_stop.best_round(), 3);
    }

    #[test]
    fn early_stopping_higher_is_better() {
        use crate::training::Accuracy;

        let mut early_stop = EarlyStopping::new(Box::new(Accuracy::default()), 2);

        // Accuracy is higher-is-better
        assert!(!early_stop.update_with_value(0.8)); // current=1, best=0
        assert!(!early_stop.update_with_value(0.9)); // current=2, best=1 (new best!)
        assert!(!early_stop.update_with_value(0.85)); // current=3, best=1, 3-1=2 > 2? NO
        assert!(early_stop.update_with_value(0.85)); // current=4, best=1, 4-1=3 > 2? YES!

        assert_eq!(early_stop.best_round(), 1);
        assert!((early_stop.best_value().unwrap() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_reset() {
        let mut early_stop = EarlyStopping::new(Box::new(Rmse), 3);

        early_stop.update_with_value(0.5);
        early_stop.update_with_value(0.6);
        early_stop.update_with_value(0.7);

        assert_eq!(early_stop.current_round(), 3);
        assert_eq!(early_stop.best_round(), 0);

        early_stop.reset();

        assert_eq!(early_stop.current_round(), 0);
        assert_eq!(early_stop.best_round(), 0);
        assert!(early_stop.best_value().is_none());
    }

    #[test]
    fn early_stopping_should_stop_with_data() {
        let mut early_stop = EarlyStopping::new(Box::new(Rmse), 2);

        let labels = vec![0.0, 1.0, 2.0];

        // Good predictions (RMSE ≈ 0)
        assert!(!early_stop.should_stop(&[0.0, 1.0, 2.0], &labels)); // current=1, best=0

        // Bad predictions (RMSE > 0)
        assert!(!early_stop.should_stop(&[1.0, 2.0, 3.0], &labels)); // current=2, best=0, 2-0=2 > 2? NO

        // Should stop now (3-0=3 > 2)
        assert!(early_stop.should_stop(&[2.0, 3.0, 4.0], &labels)); // current=3, best=0, 3-0=3 > 2? YES!
    }

    #[test]
    fn early_stopping_metric_name() {
        let early_stop = EarlyStopping::new(Box::new(Rmse), 3);
        assert_eq!(early_stop.metric_name(), "rmse");
    }
}
