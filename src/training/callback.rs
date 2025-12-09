//! Early stopping callback for training.
//!
//! Monitors a validation metric and stops training when no improvement is seen
//! for a specified number of rounds.

/// Early stopping configuration and state.
///
/// Monitors a validation metric during training and signals when to stop
/// based on lack of improvement over a patience window.
///
/// # Example
///
/// ```
/// use booste_rs::training::EarlyStopping;
///
/// // Monitor a metric where lower is better (e.g., RMSE)
/// let mut early_stop = EarlyStopping::new(5, false);
///
/// // Simulated training loop
/// for round in 0..100 {
///     let val_metric = compute_validation_rmse(); // Your metric computation
///
///     if early_stop.should_stop(val_metric) {
///         println!("Early stopping at round {}", round);
///         break;
///     }
/// }
/// # fn compute_validation_rmse() -> f64 { 0.0 }
/// ```
pub struct EarlyStopping {
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
    /// * `patience` - Number of rounds without improvement before stopping
    /// * `higher_is_better` - Whether higher metric values indicate improvement
    pub fn new(patience: usize, higher_is_better: bool) -> Self {
        Self {
            patience,
            best_value: None,
            best_round: 0,
            current_round: 0,
            higher_is_better,
        }
    }

    /// Update with a metric value and check if training should stop.
    ///
    /// # Arguments
    ///
    /// * `value` - The metric value for the current round
    ///
    /// Returns `true` if training should stop (no improvement for `patience` rounds).
    pub fn should_stop(&mut self, value: f64) -> bool {
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

    #[test]
    fn early_stopping_no_stop_while_improving() {
        let mut early_stop = EarlyStopping::new(3, false); // lower is better

        // Simulated improving metrics (decreasing)
        assert!(!early_stop.should_stop(1.0));
        assert!(!early_stop.should_stop(0.9));
        assert!(!early_stop.should_stop(0.8));
        assert!(!early_stop.should_stop(0.7));
        assert!(!early_stop.should_stop(0.6));

        assert_eq!(early_stop.best_round(), 4);
        assert!((early_stop.best_value().unwrap() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_stops_after_patience() {
        let mut early_stop = EarlyStopping::new(3, false); // lower is better

        // Best at round 0, then no improvement
        assert!(!early_stop.should_stop(0.5)); // current=1, best=0, 1-0=1 > 3? NO
        assert!(!early_stop.should_stop(0.6)); // current=2, best=0, 2-0=2 > 3? NO
        assert!(!early_stop.should_stop(0.7)); // current=3, best=0, 3-0=3 > 3? NO
        assert!(early_stop.should_stop(0.8)); // current=4, best=0, 4-0=4 > 3? YES!

        assert_eq!(early_stop.best_round(), 0);
        assert!((early_stop.best_value().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_resets_on_improvement() {
        let mut early_stop = EarlyStopping::new(3, false); // lower is better

        // Initial improvement
        assert!(!early_stop.should_stop(1.0)); // current=1, best=0
        assert!(!early_stop.should_stop(1.1)); // current=2, best=0, worse
        assert!(!early_stop.should_stop(1.2)); // current=3, best=0, worse

        // New improvement resets counter
        assert!(!early_stop.should_stop(0.9)); // current=4, best=3 (new best!)
        assert!(!early_stop.should_stop(1.0)); // current=5, best=3, 5-3=2 > 3? NO
        assert!(!early_stop.should_stop(1.1)); // current=6, best=3, 6-3=3 > 3? NO
        assert!(early_stop.should_stop(1.2)); // current=7, best=3, 7-3=4 > 3? YES!

        assert_eq!(early_stop.best_round(), 3);
    }

    #[test]
    fn early_stopping_higher_is_better() {
        let mut early_stop = EarlyStopping::new(2, true); // higher is better

        assert!(!early_stop.should_stop(0.8)); // current=1, best=0
        assert!(!early_stop.should_stop(0.9)); // current=2, best=1 (new best!)
        assert!(!early_stop.should_stop(0.85)); // current=3, best=1, 3-1=2 > 2? NO
        assert!(early_stop.should_stop(0.85)); // current=4, best=1, 4-1=3 > 2? YES!

        assert_eq!(early_stop.best_round(), 1);
        assert!((early_stop.best_value().unwrap() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_reset() {
        let mut early_stop = EarlyStopping::new(3, false);

        early_stop.should_stop(0.5);
        early_stop.should_stop(0.6);
        early_stop.should_stop(0.7);

        assert_eq!(early_stop.current_round(), 3);
        assert_eq!(early_stop.best_round(), 0);

        early_stop.reset();

        assert_eq!(early_stop.current_round(), 0);
        assert_eq!(early_stop.best_round(), 0);
        assert!(early_stop.best_value().is_none());
    }
}
