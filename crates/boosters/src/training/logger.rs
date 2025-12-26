//! Training logger with verbosity levels.
//!
//! Provides structured logging during training with configurable verbosity.

use std::io::{self, Write};
use std::time::Instant;

use super::eval::MetricValue;

/// Verbosity level for training output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Verbosity {
    /// No output.
    #[default]
    Silent,
    /// Errors and warnings only.
    Warning,
    /// Progress and important information.
    Info,
    /// Detailed debugging information.
    Debug,
}

/// Training logger for structured output.
///
/// Logs training progress, metrics, and timing information based on
/// the configured verbosity level.
///
/// # Example
///
/// ```
/// use boosters::training::{TrainingLogger, Verbosity};
///
/// let mut logger = TrainingLogger::new(Verbosity::Info);
///
/// logger.start_training(100);
/// for round in 0..100 {
///     // ... training code ...
///     let metrics = vec![("rmse".to_string(), 0.5)];
///     logger.log_round(round, &metrics);
/// }
/// logger.finish_training();
/// ```
pub struct TrainingLogger<W: Write + Send = io::Stdout> {
    /// Verbosity level.
    verbosity: Verbosity,
    /// Training start time.
    start_time: Option<Instant>,
    /// Total number of rounds (if known).
    total_rounds: Option<usize>,
    /// Writer for output (default: stdout).
    writer: W,
}

impl Default for TrainingLogger<io::Stdout> {
    fn default() -> Self {
        Self::new(Verbosity::Info)
    }
}

impl TrainingLogger<io::Stdout> {
    /// Create a new logger with the specified verbosity.
    pub fn new(verbosity: Verbosity) -> Self {
        Self {
            verbosity,
            start_time: None,
            total_rounds: None,
            writer: io::stdout(),
        }
    }

    /// Create a silent logger (no output).
    pub fn silent() -> Self {
        Self::new(Verbosity::Silent)
    }

    /// Set a custom writer (e.g., for testing or file output).
    pub fn with_writer<W2: Write + Send>(self, writer: W2) -> TrainingLogger<W2> {
        TrainingLogger {
            verbosity: self.verbosity,
            start_time: self.start_time,
            total_rounds: self.total_rounds,
            writer,
        }
    }
}

impl<W: Write + Send> TrainingLogger<W> {
    /// Get the current verbosity level.
    pub fn verbosity(&self) -> Verbosity {
        self.verbosity
    }

    /// Log the start of training.
    pub fn start_training(&mut self, total_rounds: usize) {
        self.start_time = Some(Instant::now());
        self.total_rounds = Some(total_rounds);

        if self.verbosity >= Verbosity::Info {
            let _ = writeln!(self.writer, "[Training] Starting {} rounds", total_rounds);
        }
    }

    /// Log a training round with metrics.
    ///
    /// # Arguments
    ///
    /// * `round` - Current round number (0-indexed)
    /// * `metrics` - List of (name, value) pairs
    pub fn log_round(&mut self, round: usize, metrics: &[(String, f64)]) {
        if self.verbosity < Verbosity::Info {
            return;
        }

        let round_str = match self.total_rounds {
            Some(total) => format!("[{}/{}]", round + 1, total),
            None => format!("[{}]", round + 1),
        };

        let metrics_str: Vec<String> = metrics
            .iter()
            .map(|(name, value)| format!("{}: {:.6}", name, value))
            .collect();

        let _ = writeln!(
            self.writer,
            "[Training] {} {}",
            round_str,
            metrics_str.join(", ")
        );
    }

    /// Log a training round with MetricValue metrics.
    pub fn log_metrics(&mut self, round: usize, metrics: &[MetricValue]) {
        if self.verbosity < Verbosity::Info {
            return;
        }

        let round_str = match self.total_rounds {
            Some(total) => format!("[{}/{}]", round + 1, total),
            None => format!("[{}]", round + 1),
        };

        let metrics_str: Vec<String> = metrics
            .iter()
            .map(|m| format!("{}: {:.6}", m.name, m.value))
            .collect();

        let _ = writeln!(
            self.writer,
            "[Training] {} {}",
            round_str,
            metrics_str.join(", ")
        );
    }

    /// Log a round with timing information (debug level).
    pub fn log_round_debug(&mut self, round: usize, metrics: &[(String, f64)], elapsed_ms: f64) {
        if self.verbosity < Verbosity::Debug {
            return;
        }

        let round_str = match self.total_rounds {
            Some(total) => format!("[{}/{}]", round + 1, total),
            None => format!("[{}]", round + 1),
        };

        let metrics_str: Vec<String> = metrics
            .iter()
            .map(|(name, value)| format!("{}: {:.6}", name, value))
            .collect();

        let _ = writeln!(
            self.writer,
            "[Training] {} {} ({:.2}ms)",
            round_str,
            metrics_str.join(", "),
            elapsed_ms
        );
    }

    /// Log early stopping.
    pub fn log_early_stopping(&mut self, round: usize, best_round: usize, metric_name: &str) {
        if self.verbosity >= Verbosity::Info {
            let _ = writeln!(
                self.writer,
                "[Training] Early stopping at round {}. Best {} at round {}.",
                round + 1,
                metric_name,
                best_round + 1
            );
        }
    }

    /// Log the end of training.
    pub fn finish_training(&mut self) {
        if self.verbosity >= Verbosity::Info {
            if let Some(start) = self.start_time {
                let elapsed = start.elapsed();
                let _ = writeln!(
                    self.writer,
                    "[Training] Finished in {:.2}s",
                    elapsed.as_secs_f64()
                );
            } else {
                let _ = writeln!(self.writer, "[Training] Finished");
            }
        }
    }

    /// Log a warning message.
    pub fn warn(&mut self, msg: &str) {
        if self.verbosity >= Verbosity::Warning {
            let _ = writeln!(self.writer, "[Warning] {}", msg);
        }
    }

    /// Log a debug message.
    pub fn debug(&mut self, msg: &str) {
        if self.verbosity >= Verbosity::Debug {
            let _ = writeln!(self.writer, "[Debug] {}", msg);
        }
    }

    /// Log an info message.
    pub fn info(&mut self, msg: &str) {
        if self.verbosity >= Verbosity::Info {
            let _ = writeln!(self.writer, "[Info] {}", msg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A thread-safe buffer for capturing log output.
    #[derive(Clone, Default)]
    struct TestBuffer {
        inner: Arc<Mutex<Vec<u8>>>,
    }

    impl TestBuffer {
        fn new() -> Self {
            Self::default()
        }

        fn into_string(self) -> String {
            let guard = self.inner.lock().unwrap();
            String::from_utf8(guard.clone()).unwrap()
        }
    }

    impl Write for TestBuffer {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            let mut guard = self.inner.lock().unwrap();
            guard.write(buf)
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn create_logger(verbosity: Verbosity) -> (TrainingLogger<TestBuffer>, TestBuffer) {
        let buffer = TestBuffer::new();
        let logger = TrainingLogger::new(verbosity).with_writer(buffer.clone());
        (logger, buffer)
    }

    #[test]
    fn silent_produces_no_output() {
        let (mut logger, buffer) = create_logger(Verbosity::Silent);
        logger.start_training(10);
        logger.log_round(0, &[("rmse".to_string(), 0.5)]);
        logger.finish_training();
        assert!(buffer.into_string().is_empty());
    }

    #[test]
    fn warning_only_logs_warnings() {
        let (mut logger, buffer) = create_logger(Verbosity::Warning);
        logger.start_training(10);
        logger.log_round(0, &[("rmse".to_string(), 0.5)]);
        logger.warn("This is a warning");
        logger.finish_training();
        let output = buffer.into_string();
        assert!(output.contains("Warning"));
        assert!(!output.contains("Training"));
    }

    #[test]
    fn info_logs_training_progress() {
        let (mut logger, buffer) = create_logger(Verbosity::Info);
        logger.start_training(10);
        logger.log_round(0, &[("rmse".to_string(), 0.5)]);
        let output = buffer.into_string();
        assert!(output.contains("[Training]"));
        assert!(output.contains("[1/10]"));
        assert!(output.contains("rmse: 0.500000"));
    }

    #[test]
    fn debug_logs_timing() {
        let (mut logger, buffer) = create_logger(Verbosity::Debug);
        logger.log_round_debug(0, &[("rmse".to_string(), 0.5)], 12.34);
        let output = buffer.into_string();
        assert!(output.contains("12.34ms"));
    }

    #[test]
    fn early_stopping_logged() {
        let (mut logger, buffer) = create_logger(Verbosity::Info);
        logger.log_early_stopping(10, 5, "rmse");
        let output = buffer.into_string();
        assert!(output.contains("Early stopping"));
        assert!(output.contains("round 11"));
        assert!(output.contains("round 6"));
        assert!(output.contains("rmse"));
    }

    #[test]
    fn multiple_metrics() {
        let (mut logger, buffer) = create_logger(Verbosity::Info);
        logger.log_round(
            0,
            &[
                ("train_rmse".to_string(), 0.5),
                ("val_rmse".to_string(), 0.6),
            ],
        );
        let output = buffer.into_string();
        assert!(output.contains("train_rmse: 0.500000"));
        assert!(output.contains("val_rmse: 0.600000"));
    }

    #[test]
    fn verbosity_ordering() {
        assert!(Verbosity::Silent < Verbosity::Warning);
        assert!(Verbosity::Warning < Verbosity::Info);
        assert!(Verbosity::Info < Verbosity::Debug);
    }

    #[test]
    fn default_verbosity_is_silent() {
        // Default is Silent: libraries should be quiet by default
        assert_eq!(Verbosity::default(), Verbosity::Silent);
    }
}
