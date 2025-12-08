//! Gradient boosting trainer for linear models.
//!
//! This module implements coordinate descent training for GBLinear models.
//! It supports both single-output (regression, binary classification) and
//! multi-output (multiclass) training through the unified [`LossFunction`] enum.
//!
//! # Example
//!
//! The simplest way to train:
//!
//! ```ignore
//! use booste_rs::training::GBLinearTrainer;
//!
//! let trainer = GBLinearTrainer::default();
//! let model = trainer.train(&data, &labels, &[]);
//! ```
//!
//! For more control, use the builder:
//!
//! ```ignore
//! use booste_rs::training::{GBLinearTrainer, LossFunction};
//!
//! let trainer = GBLinearTrainer::builder()
//!     .loss(LossFunction::SquaredError)
//!     .num_rounds(100)
//!     .learning_rate(0.5)
//!     .lambda(1.0)
//!     .build()
//!     .unwrap();
//!
//! let model = trainer.train(&data, &labels, &[]);
//! ```
//!
//! For multiclass training:
//!
//! ```ignore
//! use booste_rs::training::{GBLinearTrainer, LossFunction};
//!
//! let trainer = GBLinearTrainer::builder()
//!     .loss(LossFunction::Softmax { num_classes: 3 })
//!     .num_rounds(100)
//!     .build()
//!     .unwrap();
//!
//! let model = trainer.train(&data, &labels, &[]);
//! ```

use derive_builder::Builder;

use crate::data::ColumnAccess;
use crate::linear::LinearModel;
use crate::training::{
    EvalMetric, EvalSet, GradientBuffer, Loss, LossFunction, Metric, TrainingLogger, Verbosity,
};

use super::selector::FeatureSelectorKind;
use super::updater::{update_bias, UpdateConfig, UpdaterKind};

// ============================================================================
// GBLinearTrainer
// ============================================================================

/// Gradient boosted linear model trainer with all parameters inlined.
///
/// Use [`GBLinearTrainer::builder()`] for a fluent configuration API,
/// or [`GBLinearTrainer::default()`] for sensible defaults.
///
/// # Example
///
/// ```ignore
/// use booste_rs::training::{GBLinearTrainer, LossFunction};
///
/// // Simple usage with defaults
/// let trainer = GBLinearTrainer::default();
/// let model = trainer.train(&data, &labels, &[]);
///
/// // Configured via builder
/// let trainer = GBLinearTrainer::builder()
///     .loss(LossFunction::Logistic)
///     .num_rounds(100)
///     .learning_rate(0.3)
///     .lambda(1.0)
///     .build()
///     .unwrap();
///
/// // Multiclass
/// let trainer = GBLinearTrainer::builder()
///     .loss(LossFunction::Softmax { num_classes: 3 })
///     .num_rounds(100)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(setter(into), default)]
pub struct GBLinearTrainer {
    // ========================================================================
    // Loss function
    // ========================================================================
    /// Loss function for training (supports both single and multi-output).
    #[builder(default)]
    pub loss: LossFunction,

    // ========================================================================
    // Training parameters
    // ========================================================================
    /// Number of boosting rounds.
    #[builder(default = "100")]
    pub num_rounds: usize,

    /// Learning rate (eta). Controls step size for weight updates.
    #[builder(default = "0.5")]
    pub learning_rate: f32,

    /// L1 regularization (alpha). Encourages sparse weights.
    #[builder(default = "0.0")]
    pub alpha: f32,

    /// L2 regularization (lambda). Prevents large weights.
    #[builder(default = "1.0")]
    pub lambda: f32,

    // ========================================================================
    // Update strategy
    // ========================================================================
    /// Use parallel (shotgun) coordinate descent updates.
    /// When true, features are updated in parallel within each round.
    #[builder(default = "true")]
    pub parallel: bool,

    /// Feature selection strategy for coordinate descent.
    #[builder(default)]
    pub feature_selector: FeatureSelectorKind,

    /// Random seed for feature shuffling.
    #[builder(default = "42")]
    pub seed: u64,

    // ========================================================================
    // Evaluation and early stopping
    // ========================================================================
    /// Evaluation metric for logging and early stopping.
    #[builder(default)]
    pub eval_metric: EvalMetric,

    /// Early stopping rounds. Training stops if no improvement for this many rounds.
    /// Set to 0 to disable.
    #[builder(default = "0")]
    pub early_stopping_rounds: usize,

    /// Index of eval set to use for early stopping (default: last eval set).
    #[builder(default)]
    pub early_stopping_eval_set: Option<usize>,

    // ========================================================================
    // Logging
    // ========================================================================
    /// Verbosity level for training output.
    #[builder(default)]
    pub verbosity: Verbosity,
}

impl Default for GBLinearTrainer {
    fn default() -> Self {
        Self {
            loss: LossFunction::default(),
            num_rounds: 100,
            learning_rate: 0.5,
            alpha: 0.0,
            lambda: 1.0,
            parallel: true,
            feature_selector: FeatureSelectorKind::default(),
            seed: 42,
            eval_metric: EvalMetric::default(),
            early_stopping_rounds: 0,
            early_stopping_eval_set: None,
            verbosity: Verbosity::default(),
        }
    }
}

impl GBLinearTrainer {
    /// Create a builder for configuring the trainer.
    pub fn builder() -> GBLinearTrainerBuilder {
        GBLinearTrainerBuilder::default()
    }

    /// Train a linear model on column-accessible data.
    ///
    /// Uses the configured `loss` function for gradient computation.
    /// The loss function determines the number of outputs:
    /// - Single-output: `SquaredError`, `Logistic`, `Hinge`, etc.
    /// - Multi-output: `Softmax { num_classes }`, `MultiQuantile { alphas }`
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training features (must implement `ColumnAccess`)
    /// * `train_labels` - Training labels
    /// * `eval_sets` - Evaluation sets for monitoring (pass `&[]` if not needed)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use booste_rs::data::ColMatrix;
    /// use booste_rs::training::{GBLinearTrainer, LossFunction};
    ///
    /// // Regression
    /// let trainer = GBLinearTrainer::default();
    /// let model = trainer.train(&data, &labels, &[]);
    ///
    /// // Multiclass
    /// let trainer = GBLinearTrainer::builder()
    ///     .loss(LossFunction::Softmax { num_classes: 3 })
    ///     .build()
    ///     .unwrap();
    /// let model = trainer.train(&data, &labels, &[]);
    /// ```
    pub fn train<C>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        eval_sets: &[EvalSet<'_, C>],
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
    {
        self.train_internal(train_data, train_labels, eval_sets, &self.loss)
    }

    // ========================================================================
    // Internal training implementation (unified for single and multi-output)
    // ========================================================================

    fn train_internal<C, L>(
        &self,
        train_data: &C,
        train_labels: &[f32],
        eval_sets: &[EvalSet<'_, C>],
        loss: &L,
    ) -> LinearModel
    where
        C: ColumnAccess<Element = f32> + Sync,
        L: Loss + ?Sized,
    {
        let num_features = train_data.num_columns();
        let num_samples = train_data.num_rows();
        let num_outputs = loss.num_outputs();

        assert!(
            num_outputs >= 1,
            "Loss must have at least 1 output, got {}",
            num_outputs
        );
        assert_eq!(
            train_labels.len(),
            num_samples,
            "Labels length must match number of samples"
        );

        // Initialize model
        let mut model = LinearModel::zeros(num_features, num_outputs);

        // Create updater and selector
        let updater = if self.parallel {
            UpdaterKind::Parallel
        } else {
            UpdaterKind::Sequential
        };
        let mut selector = self.feature_selector.create_state(self.seed);

        let update_config = UpdateConfig {
            alpha: self.alpha,
            lambda: self.lambda,
            learning_rate: self.learning_rate,
        };

        // Gradient and prediction buffers
        let mut gradients = GradientBuffer::new(num_samples, num_outputs);
        let mut predictions = vec![0.0f32; num_samples * num_outputs];
        let mut eval_predictions: Vec<Vec<f32>> = eval_sets
            .iter()
            .map(|es| vec![0.0f32; es.data.num_rows() * num_outputs])
            .collect();

        // Early stopping state
        let early_stopping_eval_idx = self
            .early_stopping_eval_set
            .unwrap_or(eval_sets.len().saturating_sub(1));
        let mut best_metric_value: Option<f64> = None;
        let mut best_round = 0;
        let mut rounds_without_improvement = 0;
        let higher_is_better = self.eval_metric.higher_is_better();

        // Logger
        let mut logger = TrainingLogger::new(self.verbosity);
        logger.start_training(self.num_rounds);

        // Training loop
        for round in 0..self.num_rounds {
            // Compute predictions
            Self::compute_predictions(&model, train_data, &mut predictions);

            // Compute gradients
            loss.compute_gradients(&predictions, train_labels, &mut gradients);

            // Update each output
            for output in 0..num_outputs {
                update_bias(&mut model, &gradients, output, self.learning_rate);

                selector.setup_round(
                    &model,
                    train_data,
                    &gradients,
                    output,
                    self.alpha,
                    self.lambda,
                );

                updater.update_round(
                    &mut model,
                    train_data,
                    &gradients,
                    &mut selector,
                    output,
                    &update_config,
                );
            }

            // Evaluation
            let (round_metrics, early_stop_value) = self.evaluate_round(
                &model,
                eval_sets,
                &mut eval_predictions,
                num_outputs,
                early_stopping_eval_idx,
            );

            if self.verbosity >= Verbosity::Info {
                logger.log_round(round, &round_metrics);
            }

            // Early stopping check
            if self.check_early_stopping(
                early_stop_value,
                higher_is_better,
                &mut best_metric_value,
                &mut best_round,
                &mut rounds_without_improvement,
                round,
                &mut logger,
            ) {
                break;
            }
        }

        logger.finish_training();
        model
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    /// Compute predictions for all samples.
    ///
    /// Handles both single-output and multi-output models.
    /// Output is column-major: index = group * num_rows + row_idx
    fn compute_predictions<C: ColumnAccess<Element = f32>>(
        model: &LinearModel,
        data: &C,
        output: &mut [f32],
    ) {
        let num_rows = data.num_rows();
        let num_groups = model.num_groups();
        let num_features = model.num_features();

        // Initialize with bias (column-major: group-first)
        for group in 0..num_groups {
            let group_start = group * num_rows;
            for row_idx in 0..num_rows {
                output[group_start + row_idx] = model.bias(group);
            }
        }

        // Add weighted features (iterate by group for column-major writes)
        for feat_idx in 0..num_features {
            for (row_idx, value) in data.column(feat_idx) {
                for group in 0..num_groups {
                    output[group * num_rows + row_idx] += value * model.weight(feat_idx, group);
                }
            }
        }
    }

    fn evaluate_round<C: ColumnAccess<Element = f32>>(
        &self,
        model: &LinearModel,
        eval_sets: &[EvalSet<'_, C>],
        eval_predictions: &mut [Vec<f32>],
        num_outputs: usize,
        early_stopping_eval_idx: usize,
    ) -> (Vec<(String, f64)>, Option<f64>) {
        let mut round_metrics = Vec::new();
        let mut early_stop_value = None;

        for (set_idx, eval_set) in eval_sets.iter().enumerate() {
            Self::compute_predictions(model, eval_set.data, &mut eval_predictions[set_idx]);

            let value = self.eval_metric.evaluate(
                &eval_predictions[set_idx],
                eval_set.labels,
                num_outputs,
            );

            round_metrics.push((format!("{}-{}", eval_set.name, self.eval_metric.name()), value));

            if set_idx == early_stopping_eval_idx {
                early_stop_value = Some(value);
            }
        }

        (round_metrics, early_stop_value)
    }

    #[allow(clippy::too_many_arguments)]
    fn check_early_stopping(
        &self,
        early_stop_value: Option<f64>,
        higher_is_better: bool,
        best_metric_value: &mut Option<f64>,
        best_round: &mut usize,
        rounds_without_improvement: &mut usize,
        round: usize,
        logger: &mut TrainingLogger,
    ) -> bool {
        if self.early_stopping_rounds == 0 {
            return false;
        }

        if let Some(current_value) = early_stop_value {
            let is_improvement = match *best_metric_value {
                None => true,
                Some(best) => {
                    if higher_is_better {
                        current_value > best
                    } else {
                        current_value < best
                    }
                }
            };

            if is_improvement {
                *best_metric_value = Some(current_value);
                *best_round = round;
                *rounds_without_improvement = 0;
            } else {
                *rounds_without_improvement += 1;
            }

            if *rounds_without_improvement > self.early_stopping_rounds {
                if self.verbosity >= Verbosity::Info {
                    logger.log_early_stopping(round, *best_round, self.eval_metric.name());
                }
                return true;
            }
        }

        false
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColMatrix, RowMatrix};

    #[test]
    fn test_default_trainer() {
        let trainer = GBLinearTrainer::default();
        assert_eq!(trainer.num_rounds, 100);
        assert_eq!(trainer.learning_rate, 0.5);
        assert_eq!(trainer.lambda, 1.0);
    }

    #[test]
    fn test_builder() {
        let trainer = GBLinearTrainer::builder()
            .num_rounds(50usize)
            .learning_rate(0.3)
            .lambda(2.0)
            .alpha(0.1)
            .build()
            .unwrap();

        assert_eq!(trainer.num_rounds, 50);
        assert_eq!(trainer.learning_rate, 0.3);
        assert_eq!(trainer.lambda, 2.0);
        assert_eq!(trainer.alpha, 0.1);
    }

    #[test]
    fn train_simple_regression() {
        // y = 2*x + 1
        let row_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];

        let trainer = GBLinearTrainer::builder()
            .num_rounds(100usize)
            .learning_rate(0.5)
            .alpha(0.0)
            .lambda(0.0)
            .parallel(false)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let model = trainer.train(&train_data, &train_labels, &[]);

        // Check predictions
        let pred1 = model.predict_row(&[1.0], &[0.0])[0];
        let pred2 = model.predict_row(&[2.0], &[0.0])[0];

        assert!((pred1 - 3.0).abs() < 0.5);
        assert!((pred2 - 5.0).abs() < 0.5);
    }

    #[test]
    fn train_with_regularization() {
        let row_data = RowMatrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 4, 1);
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![3.0, 5.0, 7.0, 9.0];

        // Train without regularization
        let trainer_no_reg = GBLinearTrainer::builder()
            .num_rounds(50usize)
            .lambda(0.0)
            .parallel(false)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();
        let model_no_reg = trainer_no_reg.train(&train_data, &train_labels, &[]);

        // Train with L2 regularization
        let trainer_l2 = GBLinearTrainer::builder()
            .num_rounds(50usize)
            .lambda(10.0)
            .parallel(false)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();
        let model_l2 = trainer_l2.train(&train_data, &train_labels, &[]);

        // L2 should produce smaller weights
        let w_no_reg = model_no_reg.weight(0, 0).abs();
        let w_l2 = model_l2.weight(0, 0).abs();
        assert!(w_l2 < w_no_reg);
    }

    #[test]
    fn train_multifeature() {
        // y = x0 + 2*x1
        let row_data = RowMatrix::from_vec(
            vec![
                1.0, 1.0, // y=3
                2.0, 1.0, // y=4
                1.0, 2.0, // y=5
                2.0, 2.0, // y=6
            ],
            4,
            2,
        );
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![3.0, 4.0, 5.0, 6.0];

        let trainer = GBLinearTrainer::builder()
            .num_rounds(200usize)
            .learning_rate(0.3)
            .lambda(0.0)
            .parallel(false)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let model = trainer.train(&train_data, &train_labels, &[]);

        let w0 = model.weight(0, 0);
        let w1 = model.weight(1, 0);

        // w0 should be ~1, w1 should be ~2
        assert!((w0 - 1.0).abs() < 0.3);
        assert!((w1 - 2.0).abs() < 0.3);
    }

    #[test]
    fn train_multiclass() {
        // Simple 3-class classification
        let row_data = RowMatrix::from_vec(
            vec![
                2.0, 1.0, // Class 0
                0.0, 1.0, // Class 1
                3.0, 1.0, // Class 0
                1.0, 3.0, // Class 2
                0.5, 0.5, // Class 1
                2.0, 2.0, // Class 2
            ],
            6,
            2,
        );
        let train_data: ColMatrix = (&row_data).into();
        let train_labels = vec![0.0, 1.0, 0.0, 2.0, 1.0, 2.0];

        let trainer = GBLinearTrainer::builder()
            .loss(LossFunction::Softmax { num_classes: 3 })
            .num_rounds(200usize)
            .learning_rate(0.3)
            .lambda(0.1)
            .parallel(false)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        let model = trainer.train(&train_data, &train_labels, &[]);

        // Model should have 3 output groups
        assert_eq!(model.num_groups(), 3);

        // Verify model produces different outputs for different classes
        let preds0 = model.predict_row(&[2.0, 1.0], &[0.0, 0.0, 0.0]);
        let preds1 = model.predict_row(&[0.0, 1.0], &[0.0, 0.0, 0.0]);

        let diff: f32 = preds0
            .iter()
            .zip(preds1.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1);
    }

    #[test]
    fn test_loss_function_enum() {
        let trainer = GBLinearTrainer::builder()
            .loss(LossFunction::Logistic)
            .num_rounds(5usize)
            .verbosity(Verbosity::Silent)
            .build()
            .unwrap();

        assert_eq!(trainer.loss, LossFunction::Logistic);
    }
}
