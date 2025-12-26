//! Early stopping callback for training.
//!
//! Monitors a validation metric and stops training when no improvement is seen
//! for a specified number of rounds.

/// Early stopping configuration and state.
///
/// Monitors a validation metric during training and signals when to stop
/// based on lack of improvement over a patience window.
///
/// When disabled (patience = 0), `update()` always returns `Continue` and never
/// signals improvement, so callers don't need special `if enabled` checks.
///
/// # Example
///
/// ```
/// use boosters::training::{EarlyStopping, EarlyStopAction};
///
/// // Monitor a metric where lower is better (e.g., RMSE)
/// let mut early_stop = EarlyStopping::new(5, false);
///
/// // Simulated training loop
/// for round in 0..100 {
///     let val_metric = compute_validation_rmse(); // Your metric computation
///
///     match early_stop.update(val_metric) {
///         EarlyStopAction::Stop => break,
///         EarlyStopAction::Improved | EarlyStopAction::Continue => {}
///     }
/// }
/// # fn compute_validation_rmse() -> f64 { 0.0 }
/// ```
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Number of rounds without improvement before stopping.
    /// 0 = disabled.
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

/// Action to take after an early stopping update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStopAction {
    /// Training should continue (no improvement, but patience not exhausted).
    Continue,
    /// New best value found - caller may want to checkpoint the model.
    Improved,
    /// Training should stop (patience exhausted).
    Stop,
}

impl EarlyStopping {
    /// Create a new early stopping callback.
    ///
    /// # Arguments
    ///
    /// * `patience` - Number of rounds without improvement before stopping.
    ///   Use 0 to disable early stopping.
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

    /// Create a disabled early stopping callback.
    ///
    /// Always returns `Continue` and never tracks best values.
    pub fn disabled() -> Self {
        Self::new(0, false)
    }

    /// Check if early stopping is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.patience > 0
    }

    /// Update with a metric value and get the action to take.
    ///
    /// # Arguments
    ///
    /// * `value` - The metric value for the current round
    ///
    /// # Returns
    ///
    /// - `Stop`: Training should stop (patience exhausted)
    /// - `Improved`: New best value found (caller may checkpoint)
    /// - `Continue`: Keep training (no improvement but within patience)
    ///
    /// When disabled (patience = 0), always returns `Continue`.
    pub fn update(&mut self, value: f64) -> EarlyStopAction {
        // Disabled: always continue, don't track anything
        if self.patience == 0 {
            self.current_round += 1;
            return EarlyStopAction::Continue;
        }

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
        if self.current_round - self.best_round > self.patience {
            EarlyStopAction::Stop
        } else if is_improvement {
            EarlyStopAction::Improved
        } else {
            EarlyStopAction::Continue
        }
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
        assert_eq!(early_stop.update(1.0), EarlyStopAction::Improved);
        assert_eq!(early_stop.update(0.9), EarlyStopAction::Improved);
        assert_eq!(early_stop.update(0.8), EarlyStopAction::Improved);
        assert_eq!(early_stop.update(0.7), EarlyStopAction::Improved);
        assert_eq!(early_stop.update(0.6), EarlyStopAction::Improved);

        assert_eq!(early_stop.best_round(), 4);
        assert!((early_stop.best_value().unwrap() - 0.6).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_stops_after_patience() {
        let mut early_stop = EarlyStopping::new(3, false); // lower is better

        // Best at round 0, then no improvement
        // After update: current_round is incremented, then we check gap > patience
        assert_eq!(early_stop.update(0.5), EarlyStopAction::Improved); // current=1, best=0, gap=1
        assert_eq!(early_stop.update(0.6), EarlyStopAction::Continue); // current=2, best=0, gap=2
        assert_eq!(early_stop.update(0.7), EarlyStopAction::Continue); // current=3, best=0, gap=3
        assert_eq!(early_stop.update(0.8), EarlyStopAction::Stop); // current=4, best=0, gap=4 > 3

        assert_eq!(early_stop.best_round(), 0);
        assert!((early_stop.best_value().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_resets_on_improvement() {
        let mut early_stop = EarlyStopping::new(3, false); // lower is better

        // Initial improvement
        assert_eq!(early_stop.update(1.0), EarlyStopAction::Improved); // current=1, best=0
        assert_eq!(early_stop.update(1.1), EarlyStopAction::Continue); // current=2, gap=2
        assert_eq!(early_stop.update(1.2), EarlyStopAction::Continue); // current=3, gap=3

        // New improvement resets counter
        assert_eq!(early_stop.update(0.9), EarlyStopAction::Improved); // current=4, best=3
        assert_eq!(early_stop.update(1.0), EarlyStopAction::Continue); // current=5, gap=2
        assert_eq!(early_stop.update(1.1), EarlyStopAction::Continue); // current=6, gap=3
        assert_eq!(early_stop.update(1.2), EarlyStopAction::Stop); // current=7, gap=4 > 3

        assert_eq!(early_stop.best_round(), 3);
    }

    #[test]
    fn early_stopping_higher_is_better() {
        let mut early_stop = EarlyStopping::new(2, true); // higher is better

        assert_eq!(early_stop.update(0.8), EarlyStopAction::Improved); // current=1, best=0
        assert_eq!(early_stop.update(0.9), EarlyStopAction::Improved); // current=2, best=1
        assert_eq!(early_stop.update(0.85), EarlyStopAction::Continue); // current=3, gap=2
        assert_eq!(early_stop.update(0.85), EarlyStopAction::Stop); // current=4, gap=3 > 2

        assert_eq!(early_stop.best_round(), 1);
        assert!((early_stop.best_value().unwrap() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn early_stopping_reset() {
        let mut early_stop = EarlyStopping::new(3, false);

        early_stop.update(0.5);
        early_stop.update(0.6);
        early_stop.update(0.7);

        assert_eq!(early_stop.current_round(), 3);
        assert_eq!(early_stop.best_round(), 0);

        early_stop.reset();

        assert_eq!(early_stop.current_round(), 0);
        assert_eq!(early_stop.best_round(), 0);
        assert!(early_stop.best_value().is_none());
    }

    #[test]
    fn early_stopping_disabled() {
        let mut early_stop = EarlyStopping::disabled();
        assert!(!early_stop.is_enabled());

        // Always returns Continue
        assert_eq!(early_stop.update(1.0), EarlyStopAction::Continue);
        assert_eq!(early_stop.update(0.5), EarlyStopAction::Continue);
        assert_eq!(early_stop.update(10.0), EarlyStopAction::Continue);

        // Still tracks rounds
        assert_eq!(early_stop.current_round(), 3);
        // But not best values
        assert!(early_stop.best_value().is_none());
    }
}
